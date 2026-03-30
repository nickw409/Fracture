//! Distributed scheduler loop for the coordinator binary.
//!
//! This is the distributed equivalent of `fracture_server::scheduler_loop`.
//! Instead of calling `batched_forward()` on a local engine, it sends
//! `BatchedForward` messages through the distributed pipeline. Cache is
//! managed on workers via `alloc_cache`/`free_cache`.
//!
//! Heartbeats are sent inline between batch iterations. On worker death,
//! all active sequences are aborted and the pipeline is flagged as degraded.

use fracture_coordinator::heartbeat::{self, HeartbeatTracker};
use fracture_coordinator::pipeline::DistributedPipeline;
use fracture_coordinator::registry::PeerRegistry;
use fracture_engine::{GenerationEvent, PendingRequest};
use fracture_generate::{Sampler, SamplingParams, StopReason};
use fracture_protocol::connection::FramedConnection;
use fracture_protocol::frame::MessageType;
use fracture_protocol::messages::{HeartbeatAckPayload, HeartbeatPayload, SequenceMetadataWire};
use fracture_protocol::FramedReader;
use fracture_server::SchedulerHandle;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, watch, Mutex};

/// An active distributed sequence.
struct DistributedSequence {
    seq_id: u64,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: Option<u64>,
    stop_tokens: Vec<u32>,
    current_pos: usize,
    generated_tokens: Vec<u32>,
    event_tx: mpsc::UnboundedSender<GenerationEvent>,
    /// Remaining prompt tokens for chunked prefill.
    remaining_prefill: Vec<u32>,
    /// True if the initial prefill has finished (decode phase).
    prefill_done: bool,
}

/// Configuration for the distributed scheduler loop.
pub struct DistributedLoopConfig {
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
    pub max_prefill_tokens: usize,
    /// Minimum free blocks on the bottleneck worker before admitting new sequences.
    pub min_free_blocks_reserve: u32,
    /// Heartbeat interval.
    pub heartbeat_interval: Duration,
    /// Max missed heartbeats before marking a worker dead.
    pub heartbeat_max_missed: usize,
    /// Model config (needed for rebalancing after worker death).
    pub model_config: Option<fracture_core::ModelConfig>,
    /// Scheduling mode (needed for rebalancing).
    pub scheduling_mode: fracture_coordinator::scheduler::SchedulingMode,
    /// Max sequence length (needed for rebalancing).
    pub max_seq_len: usize,
}

impl Default for DistributedLoopConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_batch_tokens: 4096,
            max_prefill_tokens: 512,
            min_free_blocks_reserve: 4,
            heartbeat_interval: heartbeat::DEFAULT_INTERVAL,
            heartbeat_max_missed: heartbeat::DEFAULT_MAX_MISSED,
            model_config: None,
            scheduling_mode: fracture_coordinator::scheduler::SchedulingMode::Auto,
            max_seq_len: 4096,
        }
    }
}

/// Start the distributed scheduler loop as a background tokio task.
///
/// Returns a SchedulerHandle compatible with `BatchedAppState`.
///
/// The `pipeline_rx` channel receives pipeline updates when the pipeline
/// is reconfigured after worker death/reconnection.
pub fn start_distributed_loop(
    pipeline: Arc<DistributedPipeline>,
    registry: Arc<Mutex<PeerRegistry>>,
    config: DistributedLoopConfig,
    pipeline_rx: watch::Receiver<Arc<DistributedPipeline>>,
) -> SchedulerHandle {
    let (tx, rx) = mpsc::unbounded_channel();
    let handle = SchedulerHandle::from_sender(tx);

    tokio::spawn(distributed_loop_task(pipeline, registry, rx, config, pipeline_rx));

    handle
}

async fn distributed_loop_task(
    mut pipeline: Arc<DistributedPipeline>,
    registry: Arc<Mutex<PeerRegistry>>,
    mut request_rx: mpsc::UnboundedReceiver<PendingRequest>,
    config: DistributedLoopConfig,
    mut pipeline_rx: watch::Receiver<Arc<DistributedPipeline>>,
) {
    let mut pending: Vec<PendingRequest> = Vec::new();
    let mut active: HashMap<u64, DistributedSequence> = HashMap::new();
    let mut last_heartbeat = Instant::now();
    let mut tracker = HeartbeatTracker::new();
    let mut pipeline_degraded = false;

    loop {
        // Check for pipeline reconfiguration.
        if pipeline_rx.has_changed().unwrap_or(false) {
            pipeline = pipeline_rx.borrow_and_update().clone();
            pipeline_degraded = false;
            tracing::info!("distributed loop received reconfigured pipeline");
        }

        // Drain all pending requests from the channel.
        loop {
            match request_rx.try_recv() {
                Ok(req) => pending.push(req),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    if pending.is_empty() && active.is_empty() {
                        return;
                    }
                    break;
                }
            }
        }

        // Heartbeat: poll acks for the current nonce, then evaluate misses,
        // then send a fresh heartbeat. This order ensures workers have the
        // full interval to respond before being judged.
        if last_heartbeat.elapsed() >= config.heartbeat_interval {
            // 1. Poll acks for the nonce we sent last round.
            //    Workers had the full interval to respond; collect now.
            if tracker.pending_nonce().is_some() {
                let reader_handles = {
                    let reg = registry.lock().await;
                    reg.reader_handles()
                };
                let acks = poll_worker_acks_concurrent(&reader_handles).await;

                // Process acks through the tracker (validates nonce, resets missed count).
                // Also detect any LeaveIntent messages.
                let mut leaving_workers: Vec<String> = Vec::new();
                {
                    let mut reg = registry.lock().await;
                    for (node_id, poll) in &acks {
                        match poll {
                            PollResult::Ack(ack) => {
                                tracker.process_ack(&mut reg, node_id, ack);
                            }
                            PollResult::LeaveIntent => {
                                tracing::info!("worker '{}' sent LeaveIntent — marking as draining", node_id);
                                reg.mark_draining(node_id);
                                leaving_workers.push(node_id.clone());
                            }
                            PollResult::Nothing => {}
                        }
                    }
                }

                // 2. Increment missed for workers that didn't ack this round.
                let ready_node_ids = {
                    let reg = registry.lock().await;
                    reg.pipeline_order()
                };
                let dead_ids = tracker.increment_missed(
                    &ready_node_ids,
                    config.heartbeat_max_missed,
                );
                if !dead_ids.is_empty() {
                    tracing::error!("dead workers detected: {dead_ids:?}");

                    {
                        let mut reg = registry.lock().await;
                        heartbeat::mark_dead_workers(&mut reg, &dead_ids);
                    }

                    // Abort active sequences that used dead workers' layers.
                    let aborted = {
                        let mut reg = registry.lock().await;
                        pipeline.abort_all_sequences(&mut reg).await
                    };
                    for seq_id in &aborted {
                        if let Some(seq) = active.remove(seq_id) {
                            let _ = seq.event_tx.send(GenerationEvent::Error(
                                format!("worker died: {dead_ids:?}"),
                            ));
                        }
                    }

                    // Try to rebalance with remaining workers (FT-5).
                    if let Some(ref model_cfg) = config.model_config {
                        tracing::info!("attempting crash recovery rebalance without dead workers");
                        match fracture_coordinator::rebalance::forced_rebalance(
                            &registry,
                            &pipeline,
                            model_cfg,
                            &config.scheduling_mode,
                            config.max_seq_len,
                            &dead_ids,
                        )
                        .await
                        {
                            Ok(result) => {
                                pipeline = result.pipeline;
                                tracing::info!(
                                    "crash recovery rebalance succeeded: {} stages",
                                    pipeline.pipeline_order().len()
                                );
                                // Pipeline recovered — not degraded.
                                // Pending requests can proceed.
                            }
                            Err(e) => {
                                tracing::error!("crash recovery rebalance failed: {e}");
                                tracing::error!("pipeline degraded — waiting for manual intervention or worker reconnection");
                                pipeline_degraded = true;
                                for req in pending.drain(..) {
                                    let _ = req.event_tx.send(GenerationEvent::Error(
                                        format!("pipeline degraded: worker(s) {dead_ids:?} died"),
                                    ));
                                }
                            }
                        }
                    } else {
                        // No model config — can't rebalance, fall back to old behavior.
                        pipeline_degraded = true;
                        for req in pending.drain(..) {
                            let _ = req.event_tx.send(GenerationEvent::Error(
                                format!("pipeline degraded: worker(s) {dead_ids:?} died"),
                            ));
                        }
                    }
                }
            }

            // 3. Send new heartbeat with fresh nonce for the next round.
            let nonce = send_heartbeats(&registry).await;
            tracker.set_pending_nonce(nonce);

            last_heartbeat = Instant::now();
        }

        // Handle draining/pending workers when all active sequences complete.
        if active.is_empty() && !pipeline_degraded {
            let draining_workers: Vec<String> = {
                let reg = registry.lock().await;
                reg.iter()
                    .filter(|(_, e)| e.status == fracture_coordinator::registry::WorkerStatus::Draining)
                    .map(|(id, _)| id.clone())
                    .collect()
            };
            let pending_workers: Vec<String> = {
                let reg = registry.lock().await;
                reg.pending_workers()
            };
            let needs_rebalance = !draining_workers.is_empty() || !pending_workers.is_empty();

            if needs_rebalance {
                if !draining_workers.is_empty() {
                    tracing::info!("draining workers ready to leave: {draining_workers:?}");
                }
                if !pending_workers.is_empty() {
                    tracing::info!("pending workers ready to join: {pending_workers:?}");
                }

                // Send Shutdown to draining workers and mark them dead.
                {
                    let mut reg = registry.lock().await;
                    for node_id in &draining_workers {
                        if let Some(entry) = reg.get_mut(node_id) {
                            let _ = entry.writer.send_empty(MessageType::Shutdown, 0).await;
                        }
                        reg.mark_dead(node_id);
                    }
                }

                // Rebalance: exclude draining (now dead) workers, include pending workers.
                if let Some(ref model_cfg) = config.model_config {
                    match fracture_coordinator::rebalance::forced_rebalance(
                        &registry,
                        &pipeline,
                        model_cfg,
                        &config.scheduling_mode,
                        config.max_seq_len,
                        &draining_workers,
                    )
                    .await
                    {
                        Ok(result) => {
                            pipeline = result.pipeline;
                            tracing::info!(
                                "rebalanced (leave: {}, join: {}): {} stages",
                                draining_workers.len(),
                                pending_workers.len(),
                                pipeline.pipeline_order().len()
                            );
                        }
                        Err(e) => {
                            tracing::error!("rebalance failed: {e}");
                            pipeline_degraded = true;
                        }
                    }
                }
            }
        }

        // If pipeline is degraded, reject new requests until reconfigured.
        if pipeline_degraded {
            for req in pending.drain(..) {
                let _ = req.event_tx.send(GenerationEvent::Error(
                    "pipeline degraded: waiting for reconfiguration".to_string(),
                ));
            }
            if active.is_empty() {
                tokio::select! {
                    _ = pipeline_rx.changed() => {
                        pipeline = pipeline_rx.borrow_and_update().clone();
                        pipeline_degraded = false;
                        tracing::info!("distributed loop received reconfigured pipeline");
                    }
                    req = request_rx.recv() => {
                        match req {
                            Some(r) => pending.push(r),
                            None => return,
                        }
                    }
                }
                continue;
            }
            tokio::task::yield_now().await;
            continue;
        }

        // If no work, wait for the next request (with heartbeat timeout).
        if pending.is_empty() && active.is_empty() {
            let heartbeat_deadline = last_heartbeat + config.heartbeat_interval;
            tokio::select! {
                req = request_rx.recv() => {
                    match req {
                        Some(r) => pending.push(r),
                        None => return,
                    }
                }
                _ = tokio::time::sleep_until(heartbeat_deadline.into()) => {
                    // Heartbeat timer fired while idle — loop back to send + poll.
                    continue;
                }
                _ = pipeline_rx.changed() => {
                    pipeline = pipeline_rx.borrow_and_update().clone();
                    pipeline_degraded = false;
                    tracing::info!("distributed loop received reconfigured pipeline");
                    continue;
                }
            }
        }

        // Admission: check distributed free blocks before admitting new sequences.
        // min_free_blocks() returns 0 when no heartbeat data is available yet,
        // so we always admit at least one request (the alloc will fail with OOM
        // if the worker truly has no blocks).
        let free_blocks = {
            let reg = registry.lock().await;
            reg.min_free_blocks()
        };
        let can_admit = free_blocks > config.min_free_blocks_reserve || free_blocks == 0;

        while !pending.is_empty() && can_admit {
            let req = pending.remove(0);
            let seq_id = req.seq_id;

            // Allocate cache on all workers.
            let alloc_result = {
                let mut reg = registry.lock().await;
                pipeline.alloc_cache(&mut reg, seq_id, 0).await
            };

            match alloc_result {
                Ok(()) => {
                    let prompt_len = req.prompt_tokens.len();
                    active.insert(
                        seq_id,
                        DistributedSequence {
                            seq_id,
                            max_tokens: req.max_tokens,
                            temperature: req.temperature,
                            top_k: req.top_k,
                            top_p: req.top_p,
                            seed: req.seed,
                            stop_tokens: req.stop_tokens,
                            current_pos: prompt_len,
                            generated_tokens: Vec::new(),
                            event_tx: req.event_tx,
                            remaining_prefill: req.prompt_tokens,
                            prefill_done: false,
                        },
                    );
                }
                Err(e) => {
                    let _ = req.event_tx.send(GenerationEvent::Error(
                        format!("cache allocation failed: {e}"),
                    ));
                }
            }
            // Only admit one per iteration to re-check free blocks.
            break;
        }

        if active.is_empty() {
            tokio::task::yield_now().await;
            continue;
        }

        // Build the batch: prefills first, then decodes.
        let mut batch_seq_ids: Vec<u64> = Vec::new();
        let mut batch_is_prefill: Vec<bool> = Vec::new();
        let mut all_token_ids: Vec<u32> = Vec::new();
        let mut seq_metas: Vec<SequenceMetadataWire> = Vec::new();
        let mut total_tokens = 0usize;

        // Prefills (chunked).
        for seq in active.values() {
            if seq.prefill_done || seq.remaining_prefill.is_empty() {
                continue;
            }
            if total_tokens >= config.max_batch_tokens || batch_seq_ids.len() >= config.max_batch_size
            {
                break;
            }

            let chunk_size = seq
                .remaining_prefill
                .len()
                .min(config.max_prefill_tokens)
                .min(config.max_batch_tokens - total_tokens);
            let start_pos = seq.current_pos - seq.remaining_prefill.len();
            let chunk_tokens: Vec<u32> = seq.remaining_prefill[..chunk_size].to_vec();
            let chunk_positions: Vec<u32> =
                (start_pos as u32..(start_pos + chunk_size) as u32).collect();

            seq_metas.push(SequenceMetadataWire {
                seq_id: seq.seq_id,
                num_tokens: chunk_size,
                positions: chunk_positions.clone(),
                block_table: Vec::new(), // Workers manage their own block tables.
                cache_seq_len: 0,        // Workers track this locally.
                last_block_tokens: 0,
            });
            all_token_ids.extend_from_slice(&chunk_tokens);
            batch_seq_ids.push(seq.seq_id);
            batch_is_prefill.push(true);
            total_tokens += chunk_size;
        }

        // Decodes.
        for seq in active.values() {
            if !seq.prefill_done {
                continue;
            }
            if total_tokens >= config.max_batch_tokens || batch_seq_ids.len() >= config.max_batch_size
            {
                break;
            }

            let last_token = seq
                .generated_tokens
                .last()
                .copied()
                .unwrap_or(0);

            seq_metas.push(SequenceMetadataWire {
                seq_id: seq.seq_id,
                num_tokens: 1,
                positions: vec![seq.current_pos as u32],
                block_table: Vec::new(),
                cache_seq_len: 0,
                last_block_tokens: 0,
            });
            all_token_ids.push(last_token);
            batch_seq_ids.push(seq.seq_id);
            batch_is_prefill.push(false);
            total_tokens += 1;
        }

        if batch_seq_ids.is_empty() {
            tokio::task::yield_now().await;
            continue;
        }

        // Run distributed batched forward.
        let is_prefill = batch_is_prefill.iter().any(|&p| p);
        let forward_result = {
            let mut reg = registry.lock().await;
            pipeline
                .batched_forward(&mut reg, &seq_metas, &all_token_ids, is_prefill)
                .await
        };

        let per_seq_logits = match forward_result {
            Ok(logits) => logits,
            Err(e) => {
                // Forward failed — could be a dead worker. Abort all sequences in batch.
                let mut reg = registry.lock().await;
                for seq_id in &batch_seq_ids {
                    if let Some(seq) = active.remove(seq_id) {
                        let _ = seq.event_tx.send(GenerationEvent::Error(
                            format!("forward pass failed: {e}"),
                        ));
                        pipeline.free_cache_best_effort(&mut reg, seq.seq_id).await;
                    }
                }
                continue;
            }
        };

        // Sample tokens and stream to clients.
        let mut completed_seq_ids: Vec<u64> = Vec::new();

        for (i, seq_id) in batch_seq_ids.iter().enumerate() {
            let is_prefill = batch_is_prefill[i];
            let Some(seq) = active.get_mut(seq_id) else {
                continue;
            };

            if is_prefill {
                // Advance chunked prefill.
                let chunk_size = seq_metas[i].num_tokens;
                seq.remaining_prefill.drain(..chunk_size);
                if !seq.remaining_prefill.is_empty() {
                    // More chunks to go — don't sample yet.
                    continue;
                }
                seq.prefill_done = true;
            }

            let logits = &per_seq_logits[i];
            let params = SamplingParams {
                temperature: seq.temperature,
                top_k: seq.top_k,
                top_p: seq.top_p,
                seed: seq.seed,
            };

            match Sampler::sample(logits, &params) {
                Ok(token) => {
                    seq.generated_tokens.push(token);

                    // Increment position only for decode steps. After prefill,
                    // current_pos is already prompt_len (the correct position for
                    // the first decode). Each decode then advances by 1.
                    if !is_prefill {
                        seq.current_pos += 1;
                    }

                    if seq.stop_tokens.contains(&token) {
                        completed_seq_ids.push(*seq_id);
                        continue;
                    }

                    let _ = seq.event_tx.send(GenerationEvent::Token(token));

                    if seq.generated_tokens.len() >= seq.max_tokens {
                        completed_seq_ids.push(*seq_id);
                    }
                }
                Err(e) => {
                    let _ = seq.event_tx.send(GenerationEvent::Error(
                        format!("sampling failed: {e}"),
                    ));
                    completed_seq_ids.push(*seq_id);
                }
            }
        }

        // Cleanup completed sequences.
        for seq_id in &completed_seq_ids {
            if let Some(seq) = active.remove(seq_id) {
                let stop_reason = if seq.stop_tokens.contains(
                    seq.generated_tokens.last().unwrap_or(&0),
                ) {
                    StopReason::Stop
                } else {
                    StopReason::Length
                };

                let _ = seq.event_tx.send(GenerationEvent::Finished {
                    stop_reason,
                    completion_tokens: seq.generated_tokens.len(),
                });

                let mut reg = registry.lock().await;
                let _ = pipeline.free_cache(&mut reg, seq.seq_id).await;
            }
        }
    }
}

/// Send heartbeat pings to all ready workers. Returns the nonce used.
async fn send_heartbeats(registry: &Mutex<PeerRegistry>) -> u64 {
    let mut reg = registry.lock().await;
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let nonce: u64 = rand::random();
    let payload = HeartbeatPayload {
        timestamp_ns: now_ns,
        nonce,
    };
    let node_ids = reg.pipeline_order();
    for node_id in &node_ids {
        if let Some(entry) = reg.get_mut(node_id) {
            let _ = entry.writer.send(MessageType::Heartbeat, 0, &payload).await;
        }
    }
    nonce
}

/// Result of polling a single worker during heartbeat phase.
enum PollResult {
    Ack(HeartbeatAckPayload),
    LeaveIntent,
    Nothing,
}

/// Poll all worker readers concurrently for heartbeat acks and leave intents.
///
/// Each reader is polled independently with a short timeout.
async fn poll_worker_acks_concurrent(
    reader_handles: &[(String, Arc<tokio::sync::Mutex<FramedReader>>)],
) -> Vec<(String, PollResult)> {
    let mut set = tokio::task::JoinSet::new();

    for (node_id, reader) in reader_handles {
        let node_id = node_id.clone();
        let reader = reader.clone();
        set.spawn(async move {
            let result = tokio::time::timeout(Duration::from_millis(100), async {
                let mut r = reader.lock().await;
                r.recv().await
            })
            .await;

            let poll = match result {
                Ok(Ok((header, payload)))
                    if header.msg_type == MessageType::HeartbeatAck =>
                {
                    match FramedConnection::deserialize_payload::<HeartbeatAckPayload>(&payload) {
                        Ok(ack) => PollResult::Ack(ack),
                        Err(_) => PollResult::Nothing,
                    }
                }
                Ok(Ok((header, _)))
                    if header.msg_type == MessageType::LeaveIntent =>
                {
                    PollResult::LeaveIntent
                }
                _ => PollResult::Nothing,
            };
            (node_id, poll)
        });
    }

    let mut results = Vec::with_capacity(reader_handles.len());
    while let Some(Ok(result)) = set.join_next().await {
        results.push(result);
    }
    results
}
