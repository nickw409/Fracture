use crate::kv_cache::CacheHandle;
use crate::paged_kv_cache::PagedKvCacheManager;
use fracture_core::StopReason;
use std::collections::{HashMap, VecDeque};
use tokio::sync::{mpsc, oneshot};

/// Events sent from the scheduler to a client's response stream.
#[derive(Debug, Clone)]
pub enum GenerationEvent {
    /// A new token was generated.
    Token(u32),
    /// Generation finished.
    Finished {
        stop_reason: StopReason,
        completion_tokens: usize,
    },
    /// Generation failed mid-stream.
    Error(String),
}

/// A request waiting in the prefill queue.
pub struct PendingRequest {
    pub seq_id: u64,
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: Option<u64>,
    pub stop_tokens: Vec<u32>,
    /// Channel to send events to the client.
    pub event_tx: mpsc::UnboundedSender<GenerationEvent>,
}

/// A sequence actively generating tokens.
pub struct ActiveSequence {
    pub seq_id: u64,
    pub handle: CacheHandle,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: Option<u64>,
    pub stop_tokens: Vec<u32>,
    pub current_pos: usize,
    pub generated_tokens: Vec<u32>,
    pub event_tx: mpsc::UnboundedSender<GenerationEvent>,
    /// Remaining prompt tokens for chunked prefill. Empty = fully prefilled.
    pub remaining_prefill: Vec<u32>,
}

/// A job to prefill (part of) a sequence.
pub struct PrefillJob {
    pub seq_id: u64,
    pub token_ids: Vec<u32>,
    pub positions: Vec<u32>,
    pub handle: CacheHandle,
}

/// A job to decode one token for a sequence.
pub struct DecodeJob {
    pub seq_id: u64,
    pub token_id: u32,
    pub position: u32,
    pub handle: CacheHandle,
}

/// The scheduler's decision for one iteration.
pub struct SchedulerDecision {
    pub prefills: Vec<PrefillJob>,
    pub decodes: Vec<DecodeJob>,
    /// Total tokens in this batch.
    pub total_tokens: usize,
}

/// Iteration-level batch scheduler.
///
/// Decides which sequences to include in each forward pass using
/// a decode-priority policy with configurable prefill chunking.
pub struct BatchScheduler {
    /// Requests waiting for their first prefill.
    pub prefill_queue: VecDeque<PendingRequest>,
    /// Sequences actively generating.
    pub active: HashMap<u64, ActiveSequence>,
    /// Maximum sequences in a single batch.
    pub max_batch_size: usize,
    /// Maximum total tokens per iteration.
    pub max_batch_tokens: usize,
    /// Maximum prefill tokens per iteration.
    pub max_prefill_tokens: usize,
    /// Fraction of block pool to reserve for active sequence growth (0.0-1.0).
    pub block_pool_reserve: f32,
    /// Next sequence ID.
    next_seq_id: u64,
}

impl BatchScheduler {
    pub fn new(
        max_batch_size: usize,
        max_batch_tokens: usize,
        max_prefill_tokens: usize,
        block_pool_reserve: f32,
    ) -> Self {
        Self {
            prefill_queue: VecDeque::new(),
            active: HashMap::new(),
            max_batch_size,
            max_batch_tokens,
            max_prefill_tokens,
            block_pool_reserve,
            next_seq_id: 0,
        }
    }

    /// Allocate a new sequence ID.
    pub fn next_seq_id(&mut self) -> u64 {
        let id = self.next_seq_id;
        self.next_seq_id += 1;
        id
    }

    /// Enqueue a new request for prefill.
    pub fn enqueue(&mut self, request: PendingRequest) {
        self.prefill_queue.push_back(request);
    }

    /// Build the batch for this iteration.
    ///
    /// Policy: decode-priority with prefill slots.
    /// 1. Include all active decodes (cheap, latency-sensitive).
    /// 2. Continue chunked prefills for partially-prefilled sequences.
    /// 3. Admit new requests if capacity and memory allow.
    pub fn schedule(&mut self, cache: &PagedKvCacheManager) -> SchedulerDecision {
        let mut decision = SchedulerDecision {
            prefills: Vec::new(),
            decodes: Vec::new(),
            total_tokens: 0,
        };

        let pool_capacity = cache.pool().capacity();
        let reserved_blocks =
            (pool_capacity as f32 * self.block_pool_reserve).ceil() as usize;
        let free_blocks = cache.num_free_blocks();

        // 1. All active decodes first (skip sequences still doing chunked prefill).
        let decode_seq_ids: Vec<u64> = self
            .active
            .values()
            .filter(|s| s.remaining_prefill.is_empty())
            .map(|s| s.seq_id)
            .collect();

        for seq_id in decode_seq_ids {
            if decision.decodes.len() + decision.prefills.len() >= self.max_batch_size {
                break;
            }
            if decision.total_tokens >= self.max_batch_tokens {
                break;
            }
            let Some(seq) = self.active.get(&seq_id) else { continue };
            // Check if client disconnected.
            if seq.event_tx.is_closed() {
                continue; // will be cleaned up after iteration
            }
            let last_token = seq
                .generated_tokens
                .last()
                .copied()
                .unwrap_or(0);
            decision.decodes.push(DecodeJob {
                seq_id,
                token_id: last_token,
                position: seq.current_pos as u32,
                handle: seq.handle,
            });
            decision.total_tokens += 1;
        }

        // 2. Continue chunked prefills for partially-prefilled sequences.
        let chunked_seq_ids: Vec<u64> = self
            .active
            .values()
            .filter(|s| !s.remaining_prefill.is_empty())
            .map(|s| s.seq_id)
            .collect();

        let mut prefill_tokens_this_iter = 0usize;

        for seq_id in chunked_seq_ids {
            if decision.decodes.len() + decision.prefills.len() >= self.max_batch_size {
                break;
            }
            let remaining_batch_cap = self.max_batch_tokens.saturating_sub(decision.total_tokens);
            let remaining_prefill_cap = self
                .max_prefill_tokens
                .saturating_sub(prefill_tokens_this_iter);
            if remaining_batch_cap == 0 || remaining_prefill_cap == 0 {
                break;
            }

            let Some(seq) = self.active.get_mut(&seq_id) else { continue };
            let chunk_size = seq
                .remaining_prefill
                .len()
                .min(remaining_batch_cap)
                .min(remaining_prefill_cap);

            let chunk: Vec<u32> = seq.remaining_prefill.drain(..chunk_size).collect();
            let start_pos = seq.current_pos;
            let positions: Vec<u32> = (start_pos..start_pos + chunk.len())
                .map(|p| p as u32)
                .collect();
            seq.current_pos += chunk.len();

            decision.prefills.push(PrefillJob {
                seq_id,
                token_ids: chunk,
                positions,
                handle: seq.handle,
            });
            decision.total_tokens += chunk_size;
            prefill_tokens_this_iter += chunk_size;
        }

        // 3. Admit new requests from the prefill queue.
        while let Some(req) = self.prefill_queue.front() {
            if decision.decodes.len() + decision.prefills.len() >= self.max_batch_size {
                break;
            }
            let remaining_batch_cap = self.max_batch_tokens.saturating_sub(decision.total_tokens);
            let remaining_prefill_cap = self
                .max_prefill_tokens
                .saturating_sub(prefill_tokens_this_iter);
            if remaining_batch_cap == 0 || remaining_prefill_cap == 0 {
                break;
            }

            // Memory check: estimate blocks needed for this prompt.
            let prompt_len = req.prompt_tokens.len();
            let blocks_needed = (prompt_len + 15) / 16; // ceil(prompt_len / BLOCK_SIZE)
            let available = free_blocks.saturating_sub(reserved_blocks);
            if blocks_needed > available {
                break; // not enough memory
            }

            let Some(req) = self.prefill_queue.pop_front() else { break };
            let seq_id = req.seq_id;

            let chunk_size = prompt_len
                .min(remaining_batch_cap)
                .min(remaining_prefill_cap);

            let (chunk, remaining) = if chunk_size < prompt_len {
                (
                    req.prompt_tokens[..chunk_size].to_vec(),
                    req.prompt_tokens[chunk_size..].to_vec(),
                )
            } else {
                (req.prompt_tokens.clone(), Vec::new())
            };

            let positions: Vec<u32> = (0..chunk.len()).map(|p| p as u32).collect();

            // We need a CacheHandle — the caller must alloc before adding to active.
            // Use CacheHandle(seq_id) as a placeholder; the loop will alloc.
            let handle = CacheHandle(seq_id);

            decision.prefills.push(PrefillJob {
                seq_id,
                token_ids: chunk,
                positions,
                handle,
            });

            self.active.insert(
                seq_id,
                ActiveSequence {
                    seq_id,
                    handle,
                    max_tokens: req.max_tokens,
                    temperature: req.temperature,
                    top_k: req.top_k,
                    top_p: req.top_p,
                    seed: req.seed,
                    stop_tokens: req.stop_tokens,
                    current_pos: chunk_size,
                    generated_tokens: Vec::new(),
                    event_tx: req.event_tx,
                    remaining_prefill: remaining,
                },
            );

            decision.total_tokens += chunk_size;
            prefill_tokens_this_iter += chunk_size;
        }

        decision
    }

    /// Remove completed or disconnected sequences.
    /// Returns the list of cache handles to free.
    pub fn cleanup_completed(&mut self) -> Vec<(u64, CacheHandle)> {
        let mut to_remove = Vec::new();

        for (seq_id, seq) in &self.active {
            // Check stop conditions.
            let finished = if seq.generated_tokens.len() >= seq.max_tokens {
                Some(StopReason::Length)
            } else if let Some(last) = seq.generated_tokens.last() {
                if seq.stop_tokens.contains(last) {
                    Some(StopReason::Stop)
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(reason) = finished {
                let _ = seq.event_tx.send(GenerationEvent::Finished {
                    stop_reason: reason,
                    completion_tokens: seq.generated_tokens.len(),
                });
                to_remove.push((*seq_id, seq.handle));
                continue;
            }

            // Check if client disconnected.
            if seq.event_tx.is_closed() {
                to_remove.push((*seq_id, seq.handle));
                continue;
            }

            // Check if still doing chunked prefill — don't clean up yet.
        }

        for (seq_id, _) in &to_remove {
            self.active.remove(seq_id);
        }

        to_remove
    }

    /// Whether there's any work to do (pending requests or active sequences).
    pub fn has_work(&self) -> bool {
        !self.prefill_queue.is_empty() || !self.active.is_empty()
    }

    /// Number of active sequences.
    pub fn num_active(&self) -> usize {
        self.active.len()
    }

    /// Number of pending requests.
    pub fn num_pending(&self) -> usize {
        self.prefill_queue.len()
    }
}

#[cfg(test)]
mod scheduler_tests;
