//! Distributed scheduler loop for the coordinator binary.
//!
//! This is the distributed equivalent of `fracture_server::scheduler_loop`.
//! Instead of calling `batched_forward()` on a local engine, it sends
//! `BatchedForward` messages through the distributed pipeline. Cache is
//! managed on workers via `alloc_cache`/`free_cache`.

use fracture_coordinator::pipeline::DistributedPipeline;
use fracture_coordinator::registry::PeerRegistry;
use fracture_engine::{GenerationEvent, PendingRequest};
use fracture_generate::{Sampler, SamplingParams, StopReason};
use fracture_protocol::messages::SequenceMetadataWire;
use fracture_server::SchedulerHandle;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

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
}

impl Default for DistributedLoopConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_batch_tokens: 4096,
            max_prefill_tokens: 512,
            min_free_blocks_reserve: 4,
        }
    }
}

/// Start the distributed scheduler loop as a background tokio task.
///
/// Returns a SchedulerHandle compatible with `BatchedAppState`.
pub fn start_distributed_loop(
    pipeline: Arc<DistributedPipeline>,
    registry: Arc<Mutex<PeerRegistry>>,
    config: DistributedLoopConfig,
) -> SchedulerHandle {
    let (tx, rx) = mpsc::unbounded_channel();
    let handle = SchedulerHandle::from_sender(tx);

    tokio::spawn(distributed_loop_task(pipeline, registry, rx, config));

    handle
}

async fn distributed_loop_task(
    pipeline: Arc<DistributedPipeline>,
    registry: Arc<Mutex<PeerRegistry>>,
    mut request_rx: mpsc::UnboundedReceiver<PendingRequest>,
    config: DistributedLoopConfig,
) {
    let mut pending: Vec<PendingRequest> = Vec::new();
    let mut active: HashMap<u64, DistributedSequence> = HashMap::new();

    loop {
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

        // If no work, wait for the next request.
        if pending.is_empty() && active.is_empty() {
            match request_rx.recv().await {
                Some(req) => pending.push(req),
                None => return,
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
                // Abort all sequences in this batch.
                let mut reg = registry.lock().await;
                for seq_id in &batch_seq_ids {
                    if let Some(seq) = active.remove(seq_id) {
                        let _ = seq.event_tx.send(GenerationEvent::Error(
                            format!("forward pass failed: {e}"),
                        ));
                        let _ = pipeline.free_cache(&mut reg, seq.seq_id).await;
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
