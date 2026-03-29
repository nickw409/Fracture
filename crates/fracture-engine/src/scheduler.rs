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
mod tests {
    use super::*;
    use fracture_core::{Backend, DType, DeviceTensor, DeviceTimer, TensorId};
    use std::sync::atomic::{AtomicU64, Ordering};

    struct MockBackend {
        next_id: AtomicU64,
    }
    impl MockBackend {
        fn new() -> Self { Self { next_id: AtomicU64::new(1) } }
    }
    impl Backend for MockBackend {
        fn alloc(&self, shape: &[usize], dtype: DType) -> fracture_core::Result<DeviceTensor> {
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
        }
        fn free(&self, _: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
        fn copy_to_device(&self, _: &DeviceTensor, _: &[u8]) -> fracture_core::Result<()> { Ok(()) }
        fn copy_to_host(&self, _: &DeviceTensor, _: &mut [u8]) -> fracture_core::Result<()> { Ok(()) }
        fn matmul(&self, _: &DeviceTensor, _: &DeviceTensor, _: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
        fn rmsnorm(&self, _: &DeviceTensor, _: &DeviceTensor, _: f64, _: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
        fn rope(&self, _: &DeviceTensor, _: &DeviceTensor, _: &[u32], _: f64, _: usize) -> fracture_core::Result<()> { Ok(()) }
        fn attention(&self, _: &DeviceTensor, _: &DeviceTensor, _: &DeviceTensor, _: usize, _: usize, _: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
        fn silu_mul(&self, _: &DeviceTensor, _: &DeviceTensor, _: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
        fn embedding(&self, _: &[u32], _: &DeviceTensor, _: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
        fn add(&self, _: &DeviceTensor, _: &DeviceTensor, _: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
        fn copy_rows(&self, _: &DeviceTensor, _: &DeviceTensor, _: usize, _: usize, _: usize) -> fracture_core::Result<()> { Ok(()) }
        fn device_name(&self) -> &str { "mock" }
        fn total_memory(&self) -> usize { 1 << 30 }
        fn available_memory(&self) -> usize { 1 << 30 }
        fn synchronize(&self) -> fracture_core::Result<()> { Ok(()) }
        fn create_timer(&self) -> fracture_core::Result<DeviceTimer> { Ok(DeviceTimer(0)) }
        fn start_timer(&self, _: &DeviceTimer) -> fracture_core::Result<()> { Ok(()) }
        fn stop_timer(&self, _: &DeviceTimer) -> fracture_core::Result<f32> { Ok(0.0) }
        fn destroy_timer(&self, _: &DeviceTimer) -> fracture_core::Result<()> { Ok(()) }
    }

    fn make_cache(backend: &MockBackend) -> PagedKvCacheManager {
        PagedKvCacheManager::new(100, 2, 2, 16, backend).unwrap()
    }

    fn make_request(scheduler: &mut BatchScheduler, prompt_len: usize) -> mpsc::UnboundedReceiver<GenerationEvent> {
        let (tx, rx) = mpsc::unbounded_channel();
        let seq_id = scheduler.next_seq_id();
        scheduler.enqueue(PendingRequest {
            seq_id,
            prompt_tokens: (0..prompt_len as u32).collect(),
            max_tokens: 10,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
            stop_tokens: vec![999],
            event_tx: tx,
        });
        rx
    }

    #[test]
    fn test_scheduler_empty() {
        let backend = MockBackend::new();
        let cache = make_cache(&backend);
        let mut sched = BatchScheduler::new(64, 4096, 512, 0.1);
        let decision = sched.schedule(&cache);
        assert_eq!(decision.total_tokens, 0);
        assert!(decision.prefills.is_empty());
        assert!(decision.decodes.is_empty());
        assert!(!sched.has_work());
    }

    #[test]
    fn test_scheduler_admits_new_request() {
        let backend = MockBackend::new();
        let cache = make_cache(&backend);
        let mut sched = BatchScheduler::new(64, 4096, 512, 0.1);

        let _rx = make_request(&mut sched, 5);
        assert!(sched.has_work());

        let decision = sched.schedule(&cache);
        assert_eq!(decision.prefills.len(), 1);
        assert_eq!(decision.prefills[0].token_ids.len(), 5);
        assert_eq!(decision.total_tokens, 5);
        assert_eq!(sched.num_active(), 1);
        assert_eq!(sched.num_pending(), 0);
    }

    #[test]
    fn test_scheduler_prefill_chunking() {
        let backend = MockBackend::new();
        let cache = make_cache(&backend);
        // max_prefill_tokens = 10, prompt = 25 tokens → 3 chunks (10, 10, 5)
        let mut sched = BatchScheduler::new(64, 4096, 10, 0.1);

        let _rx = make_request(&mut sched, 25);

        // Iteration 1: first chunk of 10
        let d1 = sched.schedule(&cache);
        assert_eq!(d1.prefills.len(), 1);
        assert_eq!(d1.prefills[0].token_ids.len(), 10);
        assert_eq!(d1.total_tokens, 10);

        // Iteration 2: second chunk of 10
        let d2 = sched.schedule(&cache);
        assert_eq!(d2.prefills.len(), 1);
        assert_eq!(d2.prefills[0].token_ids.len(), 10);

        // Iteration 3: final chunk of 5
        let d3 = sched.schedule(&cache);
        assert_eq!(d3.prefills.len(), 1);
        assert_eq!(d3.prefills[0].token_ids.len(), 5);

        // Iteration 4: sequence should now be in decode mode (no remaining prefill)
        // but it has no generated tokens yet, so no decode job either.
        // The sequence needs at least one token to decode.
        let seq = sched.active.get(&0).unwrap();
        assert!(seq.remaining_prefill.is_empty());
        assert_eq!(seq.current_pos, 25);
    }

    #[test]
    fn test_scheduler_decode_priority() {
        let backend = MockBackend::new();
        let cache = make_cache(&backend);
        let mut sched = BatchScheduler::new(64, 4096, 512, 0.1);

        // Add and "prefill" a sequence, then give it a generated token.
        let _rx1 = make_request(&mut sched, 3);
        let _d = sched.schedule(&cache); // prefills seq 0

        // Manually mark it as having a generated token (simulating sampling).
        let seq = sched.active.get_mut(&0).unwrap();
        seq.generated_tokens.push(42);

        // Add a new request.
        let _rx2 = make_request(&mut sched, 5);

        // Schedule: decode for seq 0 should come first, then prefill for seq 1.
        let decision = sched.schedule(&cache);
        assert_eq!(decision.decodes.len(), 1);
        assert_eq!(decision.decodes[0].seq_id, 0);
        assert_eq!(decision.prefills.len(), 1);
        assert_eq!(decision.prefills[0].seq_id, 1);
        // Total: 1 decode + 5 prefill = 6
        assert_eq!(decision.total_tokens, 6);
    }

    #[test]
    fn test_scheduler_max_batch_tokens_limit() {
        let backend = MockBackend::new();
        let cache = make_cache(&backend);
        // max_batch_tokens = 8
        let mut sched = BatchScheduler::new(64, 8, 512, 0.1);

        // Two requests: 5 tokens + 5 tokens = 10 > 8
        // First admitted fully (5 tokens), second chunked to 3 (filling to 8).
        let _rx1 = make_request(&mut sched, 5);
        let _rx2 = make_request(&mut sched, 5);

        let decision = sched.schedule(&cache);
        assert_eq!(decision.prefills.len(), 2);
        assert_eq!(decision.prefills[0].token_ids.len(), 5); // full first request
        assert_eq!(decision.prefills[1].token_ids.len(), 3); // chunked second request
        assert_eq!(decision.total_tokens, 8);
        assert_eq!(sched.num_pending(), 0); // both admitted (second partially)

        // Second request has 2 remaining tokens for next iteration
        let seq1 = sched.active.get(&1).unwrap();
        assert_eq!(seq1.remaining_prefill.len(), 2);
    }

    #[test]
    fn test_scheduler_cleanup_on_max_tokens() {
        let backend = MockBackend::new();
        let cache = make_cache(&backend);
        let mut sched = BatchScheduler::new(64, 4096, 512, 0.1);

        let (tx, _rx) = mpsc::unbounded_channel();
        let seq_id = sched.next_seq_id();
        sched.active.insert(seq_id, ActiveSequence {
            seq_id,
            handle: CacheHandle(seq_id),
            max_tokens: 3,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
            stop_tokens: vec![],
            current_pos: 5,
            generated_tokens: vec![1, 2, 3], // hit max_tokens
            event_tx: tx,
            remaining_prefill: Vec::new(),
        });

        let removed = sched.cleanup_completed();
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].0, seq_id);
        assert_eq!(sched.num_active(), 0);
    }

    #[test]
    fn test_scheduler_cleanup_on_stop_token() {
        let backend = MockBackend::new();
        let cache = make_cache(&backend);
        let mut sched = BatchScheduler::new(64, 4096, 512, 0.1);

        let (tx, mut rx) = mpsc::unbounded_channel();
        let seq_id = sched.next_seq_id();
        sched.active.insert(seq_id, ActiveSequence {
            seq_id,
            handle: CacheHandle(seq_id),
            max_tokens: 100,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
            stop_tokens: vec![999],
            current_pos: 5,
            generated_tokens: vec![1, 2, 999], // stop token
            event_tx: tx,
            remaining_prefill: Vec::new(),
        });

        let removed = sched.cleanup_completed();
        assert_eq!(removed.len(), 1);

        // Should have sent Finished event.
        let event = rx.try_recv().unwrap();
        assert!(matches!(event, GenerationEvent::Finished { stop_reason: StopReason::Stop, .. }));
    }

    #[test]
    fn test_scheduler_cleanup_on_disconnect() {
        let mut sched = BatchScheduler::new(64, 4096, 512, 0.1);

        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx); // simulate client disconnect

        let seq_id = sched.next_seq_id();
        sched.active.insert(seq_id, ActiveSequence {
            seq_id,
            handle: CacheHandle(seq_id),
            max_tokens: 100,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
            stop_tokens: vec![],
            current_pos: 5,
            generated_tokens: vec![1],
            event_tx: tx,
            remaining_prefill: Vec::new(),
        });

        let removed = sched.cleanup_completed();
        assert_eq!(removed.len(), 1);
    }
}
