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
    let mut heartbeat_tracker = HeartbeatTracker::new();
    let mut last_heartbeat = Instant::now();
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

        // Send heartbeats periodically (inline, between batch iterations).
        if last_heartbeat.elapsed() >= config.heartbeat_interval {
            let dead_workers = send_heartbeat_round(
                &registry,
                &mut heartbeat_tracker,
                config.heartbeat_max_missed,
            ).await;

            if !dead_workers.is_empty() {
                tracing::error!("dead workers detected: {dead_workers:?}");
                pipeline_degraded = true;

                // Abort all active sequences — their KV cache on the dead worker is lost.
                let mut reg = registry.lock().await;
                let aborted = pipeline.abort_all_sequences(&mut reg).await;
                drop(reg);

                for seq_id in &aborted {
                    if let Some(seq) = active.remove(seq_id) {
                        let _ = seq.event_tx.send(GenerationEvent::Error(
                            format!("worker died: {dead_workers:?}"),
                        ));
                    }
                }

                // Also fail all pending requests.
                for req in pending.drain(..) {
                    let _ = req.event_tx.send(GenerationEvent::Error(
                        format!("pipeline degraded: worker(s) {dead_workers:?} died"),
                    ));
                }
            }

            last_heartbeat = Instant::now();
        }

        // If pipeline is degraded, reject new requests until reconfigured.
        if pipeline_degraded {
            for req in pending.drain(..) {
                let _ = req.event_tx.send(GenerationEvent::Error(
                    "pipeline degraded: waiting for reconfiguration".to_string(),
                ));
            }
            if active.is_empty() {
                // Wait for pipeline reconfiguration or new requests.
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
                    // Heartbeat timer fired while idle — loop back to send heartbeat.
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
        let mut all_positions: Vec<u32> = Vec::new();
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
            all_positions.extend_from_slice(&chunk_positions);
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
            all_positions.push(seq.current_pos as u32);
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
                .batched_forward(&mut reg, &seq_metas, &all_token_ids, &all_positions, is_prefill)
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

/// Send heartbeats to all workers, wait for acks, and check for dead ones.
/// Returns the list of newly-dead worker node IDs.
async fn send_heartbeat_round(
    registry: &Mutex<PeerRegistry>,
    tracker: &mut HeartbeatTracker,
    max_missed: usize,
) -> Vec<String> {
    let mut reg = registry.lock().await;

    let nonce = rand::random::<u64>();
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let payload = HeartbeatPayload {
        timestamp_ns: now_ns,
        nonce,
    };
    tracker.set_pending_nonce(nonce);

    let node_ids = reg.pipeline_order();

    // Send heartbeats (best-effort).
    for node_id in &node_ids {
        if let Some(entry) = reg.get_mut(node_id) {
            if let Err(e) = entry.connection.send(MessageType::Heartbeat, 0, &payload).await {
                tracing::warn!("heartbeat send to '{node_id}' failed: {e}");
            }
        }
    }

    // Wait for acks with a timeout, then process them.
    let ack_deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    let mut acks_received = 0usize;

    while acks_received < node_ids.len() {
        // Try to recv an ack from each worker that hasn't responded yet.
        let mut got_one = false;
        for node_id in &node_ids {
            if let Some(entry) = reg.get_mut(node_id) {
                match tokio::time::timeout(
                    Duration::from_millis(100),
                    entry.connection.recv(),
                ).await {
                    Ok(Ok((header, payload_bytes))) => {
                        if header.msg_type == MessageType::HeartbeatAck {
                            if let Ok(ack) = FramedConnection::deserialize_payload::<HeartbeatAckPayload>(&payload_bytes) {
                                tracker.process_ack(&mut reg, node_id, &ack);
                                acks_received += 1;
                                got_one = true;
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        tracing::warn!("heartbeat ack recv from '{node_id}' failed: {e}");
                    }
                    Err(_) => {} // timeout, try next worker
                }
            }
        }
        if !got_one || tokio::time::Instant::now() >= ack_deadline {
            break;
        }
    }

    // Increment missed counters for workers that didn't respond and detect dead ones.
    let timed_out = tracker.increment_missed(&node_ids, max_missed);
    heartbeat::mark_dead_workers(&mut reg, &timed_out);

    timed_out
}
