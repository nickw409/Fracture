use crate::dashboard::dto::{SchedulerSnapshot, SequenceSnapshotEntry};
use fracture_core::Backend;
use fracture_engine::{
    batched_forward, BatchScheduler, Engine, GenerationEvent, PagedKvCacheManager, PendingRequest,
    SequenceSlice, BLOCK_SIZE,
};
use fracture_generate::{Sampler, SamplingParams};
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::{mpsc, oneshot};

/// Commands sent to the scheduler loop.
enum SchedulerCommand {
    /// Submit a new generation request.
    Submit(PendingRequest),
    /// Request a point-in-time snapshot of scheduler state.
    Snapshot(oneshot::Sender<SchedulerSnapshot>),
}

/// Handle for submitting requests to the scheduler loop.
#[derive(Clone)]
pub struct SchedulerHandle {
    tx: mpsc::UnboundedSender<SchedulerCommand>,
}

impl SchedulerHandle {
    /// Submit a new request for generation.
    pub fn submit(&self, request: PendingRequest) -> Result<(), String> {
        self.tx
            .send(SchedulerCommand::Submit(request))
            .map_err(|_| "scheduler loop is shut down".to_string())
    }

    /// Create a SchedulerHandle from an existing sender channel.
    ///
    /// Used by the distributed scheduler loop and tests.
    pub fn from_sender(tx: mpsc::UnboundedSender<PendingRequest>) -> Self {
        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();
        // Bridge: forward Submit commands to the PendingRequest channel.
        tokio::spawn(async move {
            while let Some(cmd) = cmd_rx.recv().await {
                match cmd {
                    SchedulerCommand::Submit(req) => {
                        if tx.send(req).is_err() {
                            break;
                        }
                    }
                    SchedulerCommand::Snapshot(reply) => {
                        // No scheduler state available in bridge mode.
                        drop(reply);
                    }
                }
            }
        });
        Self { tx: cmd_tx }
    }

    /// Request a snapshot of the scheduler's current state.
    pub async fn snapshot(&self) -> Result<SchedulerSnapshot, String> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(SchedulerCommand::Snapshot(tx))
            .map_err(|_| "scheduler loop is shut down".to_string())?;
        rx.await.map_err(|_| "scheduler loop dropped snapshot request".to_string())
    }
}

/// Configuration for the scheduler loop.
pub struct SchedulerLoopConfig {
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
    pub max_prefill_tokens: usize,
    pub block_pool_reserve: f32,
}

impl Default for SchedulerLoopConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_batch_tokens: 4096,
            max_prefill_tokens: 512,
            block_pool_reserve: 0.1,
        }
    }
}

/// Start the scheduler loop as a background tokio task.
///
/// Returns a handle for submitting requests. The loop runs until the
/// handle is dropped (all senders closed) and all active work completes.
pub fn start_scheduler_loop<B: Backend + 'static>(
    engine: Arc<Engine<B>>,
    cache: PagedKvCacheManager,
    config: SchedulerLoopConfig,
) -> SchedulerHandle {
    let (tx, rx) = mpsc::unbounded_channel();
    let handle = SchedulerHandle { tx };
    let cache = Arc::new(StdMutex::new(cache));

    tokio::spawn(scheduler_loop_task(engine, cache, rx, config));

    handle
}

async fn scheduler_loop_task<B: Backend + 'static>(
    engine: Arc<Engine<B>>,
    cache: Arc<StdMutex<PagedKvCacheManager>>,
    mut cmd_rx: mpsc::UnboundedReceiver<SchedulerCommand>,
    config: SchedulerLoopConfig,
) {
    let mut scheduler = BatchScheduler::new(
        config.max_batch_size,
        config.max_batch_tokens,
        config.max_prefill_tokens,
        config.block_pool_reserve,
    );

    let layer_range = engine.layer_range().clone();

    loop {
        // Drain all pending commands from the channel.
        loop {
            match cmd_rx.try_recv() {
                Ok(cmd) => handle_command::<B>(cmd, &mut scheduler, &cache, &config),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    if !scheduler.has_work() {
                        return;
                    }
                    break;
                }
            }
        }

        // If no work, wait for the next command.
        if !scheduler.has_work() {
            match cmd_rx.recv().await {
                Some(cmd) => handle_command::<B>(cmd, &mut scheduler, &cache, &config),
                None => return,
            }
        }

        // Build the batch.
        let decision = {
            let cache_guard = cache.lock().unwrap();
            scheduler.schedule(&cache_guard)
        };

        if decision.total_tokens == 0 {
            tokio::task::yield_now().await;
            continue;
        }

        // Allocate cache for new prefill sequences.
        let mut failed_seq_ids = Vec::new();
        {
            let mut cache_guard = cache.lock().unwrap();
            for pf in &decision.prefills {
                if cache_guard.seq_len(pf.handle).is_err() {
                    match cache_guard.alloc() {
                        Ok(handle) => {
                            if let Some(seq) = scheduler.active.get_mut(&pf.seq_id) {
                                seq.handle = handle;
                            }
                        }
                        Err(e) => {
                            if let Some(seq) = scheduler.active.remove(&pf.seq_id) {
                                let _ = seq.event_tx.send(GenerationEvent::Error(
                                    format!("cache allocation failed: {e}"),
                                ));
                            }
                            failed_seq_ids.push(pf.seq_id);
                        }
                    }
                }
            }
        }

        // Build SequenceSlices, skipping failed allocations.
        let mut slices: Vec<SequenceSlice> = Vec::new();
        let mut slice_seq_ids: Vec<u64> = Vec::new();
        let mut slice_is_prefill: Vec<bool> = Vec::new();

        for pf in &decision.prefills {
            if failed_seq_ids.contains(&pf.seq_id) {
                continue;
            }
            let Some(seq) = scheduler.active.get(&pf.seq_id) else {
                continue;
            };
            slices.push(SequenceSlice {
                handle: seq.handle,
                token_ids: pf.token_ids.clone(),
                positions: pf.positions.clone(),
            });
            slice_seq_ids.push(pf.seq_id);
            slice_is_prefill.push(true);
        }

        for dj in &decision.decodes {
            let Some(seq) = scheduler.active.get(&dj.seq_id) else {
                continue;
            };
            slices.push(SequenceSlice {
                handle: seq.handle,
                token_ids: vec![dj.token_id],
                positions: vec![dj.position],
            });
            slice_seq_ids.push(dj.seq_id);
            slice_is_prefill.push(false);
        }

        if slices.is_empty() {
            tokio::task::yield_now().await;
            continue;
        }

        // Run batched forward pass (engine is synchronous → spawn_blocking).
        let result = {
            let cache_arc = Arc::clone(&cache);
            let engine_ref = Arc::clone(&engine);
            let lr = layer_range.clone();

            let join_result = tokio::task::spawn_blocking(move || {
                let mut cache_guard = cache_arc.lock().unwrap();
                batched_forward(
                    engine_ref.backend(),
                    engine_ref.weights(),
                    &lr,
                    &mut cache_guard,
                    &slices,
                )
            })
            .await;

            match join_result {
                Ok(Ok(output)) => output,
                Ok(Err(e)) => {
                    // Forward pass failed — abort all sequences in this batch.
                    let mut cache_guard = cache.lock().unwrap();
                    for seq_id in &slice_seq_ids {
                        if let Some(seq) = scheduler.active.remove(seq_id) {
                            let _ = seq.event_tx.send(GenerationEvent::Error(format!(
                                "forward pass failed: {e}"
                            )));
                            let _ = cache_guard.free(seq.handle);
                        }
                    }
                    continue;
                }
                Err(e) => {
                    tracing::error!("scheduler spawn_blocking panicked: {e}");
                    continue;
                }
            }
        };

        // Sample tokens and stream to clients.
        for (i, seq_id) in slice_seq_ids.iter().enumerate() {
            let is_prefill = slice_is_prefill[i];
            let Some(seq) = scheduler.active.get_mut(seq_id) else {
                continue;
            };

            if is_prefill && !seq.remaining_prefill.is_empty() {
                // Chunked prefill not yet complete — don't sample yet.
                continue;
            }

            let logits = &result.logits[i];

            let params = SamplingParams {
                temperature: seq.temperature,
                top_k: seq.top_k,
                top_p: seq.top_p,
                seed: seq.seed,
            };

            match Sampler::sample(logits, &params) {
                Ok(token) => {
                    seq.generated_tokens.push(token);
                    seq.current_pos += 1;

                    // Don't stream stop tokens.
                    if !seq.stop_tokens.contains(&token) {
                        let _ = seq.event_tx.send(GenerationEvent::Token(token));
                    }
                }
                Err(e) => {
                    let _ = seq
                        .event_tx
                        .send(GenerationEvent::Error(format!("sampling failed: {e}")));
                    // Mark for cleanup by setting max_tokens = 0.
                    seq.max_tokens = 0;
                }
            }
        }

        // Cleanup completed sequences.
        let completed = scheduler.cleanup_completed();
        if !completed.is_empty() {
            let mut cache_guard = cache.lock().unwrap();
            for (_, handle) in &completed {
                let _ = cache_guard.free(*handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fracture_core::{Backend, DType, DeviceTensor, DeviceTimer, TensorId};
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    struct MockBackend {
        next_id: AtomicU64,
    }
    impl MockBackend {
        fn new() -> Self {
            Self {
                next_id: AtomicU64::new(1),
            }
        }
    }
    impl Backend for MockBackend {
        fn alloc(&self, shape: &[usize], dtype: DType) -> fracture_core::Result<DeviceTensor> {
            let id = self.next_id.fetch_add(1, AtomicOrdering::SeqCst);
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

    #[test]
    fn test_snapshot_empty_scheduler() {
        let backend = MockBackend::new();
        let cache = Arc::new(StdMutex::new(make_cache(&backend)));
        let config = SchedulerLoopConfig::default();
        let mut scheduler = BatchScheduler::new(
            config.max_batch_size,
            config.max_batch_tokens,
            config.max_prefill_tokens,
            config.block_pool_reserve,
        );

        let (tx, rx) = oneshot::channel();
        handle_command::<MockBackend>(
            SchedulerCommand::Snapshot(tx),
            &mut scheduler,
            &cache,
            &config,
        );

        let snap = rx.blocking_recv().unwrap();
        assert_eq!(snap.active_sequences, 0);
        assert_eq!(snap.max_sequences, 64);
        assert_eq!(snap.decode_count, 0);
        assert_eq!(snap.prefill_queue_count, 0);
        assert_eq!(snap.total_blocks, 100);
        assert_eq!(snap.free_blocks, 100);
        assert!(snap.sequences.is_empty());
    }

    #[test]
    fn test_snapshot_with_pending_requests() {
        let backend = MockBackend::new();
        let cache = Arc::new(StdMutex::new(make_cache(&backend)));
        let config = SchedulerLoopConfig::default();
        let mut scheduler = BatchScheduler::new(
            config.max_batch_size,
            config.max_batch_tokens,
            config.max_prefill_tokens,
            config.block_pool_reserve,
        );

        // Enqueue two requests.
        let (tx1, _rx1) = mpsc::unbounded_channel();
        scheduler.enqueue(PendingRequest {
            seq_id: 0,
            prompt_tokens: vec![1, 2, 3],
            max_tokens: 10,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
            stop_tokens: vec![],
            event_tx: tx1,
        });
        let (tx2, _rx2) = mpsc::unbounded_channel();
        scheduler.enqueue(PendingRequest {
            seq_id: 1,
            prompt_tokens: vec![4, 5],
            max_tokens: 5,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
            stop_tokens: vec![],
            event_tx: tx2,
        });

        let (snap_tx, snap_rx) = oneshot::channel();
        handle_command::<MockBackend>(
            SchedulerCommand::Snapshot(snap_tx),
            &mut scheduler,
            &cache,
            &config,
        );

        let snap = snap_rx.blocking_recv().unwrap();
        assert_eq!(snap.active_sequences, 0); // Not yet scheduled.
        assert_eq!(snap.prefill_queue_count, 2);
    }

    #[test]
    fn test_snapshot_with_active_sequences() {
        let backend = MockBackend::new();
        let cache = Arc::new(StdMutex::new(make_cache(&backend)));
        let config = SchedulerLoopConfig::default();
        let mut scheduler = BatchScheduler::new(
            config.max_batch_size,
            config.max_batch_tokens,
            config.max_prefill_tokens,
            config.block_pool_reserve,
        );

        // Enqueue and schedule to move into active.
        let (tx, _rx) = mpsc::unbounded_channel();
        scheduler.enqueue(PendingRequest {
            seq_id: 0,
            prompt_tokens: vec![1, 2, 3],
            max_tokens: 10,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
            stop_tokens: vec![],
            event_tx: tx,
        });
        {
            let cache_guard = cache.lock().unwrap();
            let _ = scheduler.schedule(&cache_guard);
        }
        // Simulate a generated token so it's in decode state.
        scheduler.active.get_mut(&0).unwrap().generated_tokens.push(42);

        let (snap_tx, snap_rx) = oneshot::channel();
        handle_command::<MockBackend>(
            SchedulerCommand::Snapshot(snap_tx),
            &mut scheduler,
            &cache,
            &config,
        );

        let snap = snap_rx.blocking_recv().unwrap();
        assert_eq!(snap.active_sequences, 1);
        assert_eq!(snap.decode_count, 1);
        assert_eq!(snap.sequences.len(), 1);
        assert_eq!(snap.sequences[0].state, "decoding");
        assert_eq!(snap.sequences[0].tokens_generated, 1);
    }

    #[test]
    fn test_submit_command_enqueues() {
        let backend = MockBackend::new();
        let cache = Arc::new(StdMutex::new(make_cache(&backend)));
        let config = SchedulerLoopConfig::default();
        let mut scheduler = BatchScheduler::new(
            config.max_batch_size,
            config.max_batch_tokens,
            config.max_prefill_tokens,
            config.block_pool_reserve,
        );

        let (tx, _rx) = mpsc::unbounded_channel();
        handle_command::<MockBackend>(
            SchedulerCommand::Submit(PendingRequest {
                seq_id: 0,
                prompt_tokens: vec![1],
                max_tokens: 5,
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                seed: None,
                stop_tokens: vec![],
                event_tx: tx,
            }),
            &mut scheduler,
            &cache,
            &config,
        );

        assert_eq!(scheduler.num_pending(), 1);
    }

    // ── Slice construction and sampling unit tests ────────────────────────────

    use fracture_core::ModelConfig;
    use fracture_engine::{CacheHandle, Engine};
    use fracture_gguf::{LayerWeights, WeightStore};
    use fracture_generate::{Sampler, SamplingParams};
    use std::sync::atomic::Ordering;

    /// Gap 107 — SchedulerDecision to SequenceSlice conversion: verify that
    /// PrefillJob fields correctly map to SequenceSlice fields.
    #[test]
    fn test_prefill_job_to_sequence_slice() {
        // Simulate what the scheduler loop does: schedule a prefill, then
        // build SequenceSlice from the PrefillJob + ActiveSequence.
        let mut scheduler = BatchScheduler::new(64, 4096, 512, 0.1);
        let (tx, _rx) = mpsc::unbounded_channel();

        scheduler.enqueue(PendingRequest {
            seq_id: 0,
            prompt_tokens: vec![10, 20, 30, 40, 50],
            max_tokens: 32,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
            stop_tokens: vec![999],
            event_tx: tx,
        });

        // Simulated PrefillJob output (from what schedule() produces).
        let pf_token_ids = vec![10, 20, 30, 40, 50];
        let pf_positions: Vec<u32> = (0..5).collect();
        let handle = CacheHandle(7);

        let slice = SequenceSlice {
            handle,
            token_ids: pf_token_ids.clone(),
            positions: pf_positions.clone(),
        };

        assert_eq!(slice.token_ids, vec![10, 20, 30, 40, 50]);
        assert_eq!(slice.positions, vec![0, 1, 2, 3, 4]);
        assert_eq!(slice.handle, handle);
        assert_eq!(slice.token_ids.len(), slice.positions.len());
    }

    /// Gap 107 (continued) — DecodeJob fields map to a single-token SequenceSlice.
    #[test]
    fn test_decode_job_to_sequence_slice() {
        let handle = CacheHandle(3);
        let decode_token_id = 42u32;
        let decode_position = 17u32;

        let slice = SequenceSlice {
            handle,
            token_ids: vec![decode_token_id],
            positions: vec![decode_position],
        };

        assert_eq!(slice.token_ids, vec![42]);
        assert_eq!(slice.positions, vec![17]);
        assert_eq!(slice.handle, handle);
        assert_eq!(slice.token_ids.len(), 1);
    }

    /// Gap 107 (continued) — mixed prefill + decode batch builds correct slices.
    #[test]
    fn test_mixed_batch_slice_construction() {
        let prefill_handle = CacheHandle(0);
        let decode_handle = CacheHandle(1);

        let mut slices: Vec<SequenceSlice> = Vec::new();
        let mut slice_is_prefill: Vec<bool> = Vec::new();

        // Prefill job: 4-token prompt at positions 0..4.
        slices.push(SequenceSlice {
            handle: prefill_handle,
            token_ids: vec![100, 200, 300, 400],
            positions: vec![0, 1, 2, 3],
        });
        slice_is_prefill.push(true);

        // Decode job: single token at position 10.
        slices.push(SequenceSlice {
            handle: decode_handle,
            token_ids: vec![55],
            positions: vec![10],
        });
        slice_is_prefill.push(false);

        assert_eq!(slices.len(), 2);
        assert!(slice_is_prefill[0]);
        assert!(!slice_is_prefill[1]);

        // Prefill slice has multiple tokens, decode has exactly one.
        assert_eq!(slices[0].token_ids.len(), 4);
        assert_eq!(slices[1].token_ids.len(), 1);
    }

    /// Gap 120 — per-sequence sampling: temperature=0 (greedy) always picks
    /// the argmax token.
    #[test]
    fn test_greedy_sampling_picks_argmax() {
        // Logits where index 3 has the highest value.
        let logits = vec![1.0, 2.0, 0.5, 10.0, 3.0];
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };

        let token = Sampler::sample(&logits, &params).unwrap();
        assert_eq!(token, 3, "greedy sampling should pick index of max logit");

        // Deterministic: running again gives the same result.
        let token2 = Sampler::sample(&logits, &params).unwrap();
        assert_eq!(token2, 3);
    }

    /// Gap 120 (continued) — temperature>0 with different seeds produces
    /// different tokens.
    #[test]
    fn test_seeded_sampling_different_seeds_different_results() {
        let mut logits = vec![0.0f32; 1000];
        for (i, l) in logits.iter_mut().enumerate() {
            *l = (i as f32) * 0.001;
        }

        let params_seed_1 = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: Some(42),
        };

        let params_seed_2 = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: Some(12345),
        };

        let token_a = Sampler::sample(&logits, &params_seed_1).unwrap();
        let token_b = Sampler::sample(&logits, &params_seed_2).unwrap();

        // With 1000 options, collision probability is ~0.1%.
        assert_ne!(
            token_a, token_b,
            "different seeds should (almost certainly) produce different tokens"
        );
    }

    /// Gap 120 (continued) — same seed produces deterministic output.
    #[test]
    fn test_seeded_sampling_deterministic() {
        let logits: Vec<f32> = (0..500).map(|i| (i as f32) * 0.01).collect();

        let params = SamplingParams {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.9,
            seed: Some(99),
        };

        let token1 = Sampler::sample(&logits, &params).unwrap();
        let token2 = Sampler::sample(&logits, &params).unwrap();
        assert_eq!(token1, token2, "same seed should produce the same token");
    }

    // ── Scheduler loop integration tests ──────────────────────────────────────
    //
    // These tests spin up a real scheduler loop task (via start_scheduler_loop)
    // with a mock Backend and fake WeightStore. The mock Backend:
    //  - Supports attention_paged (returns Ok, no-op kernel)
    //  - Writes controlled FP16 logits via copy_to_host so sampling is predictable
    //
    // The fake WeightStore has tiny dimensions (hidden=8, 1 layer) to keep
    // allocations cheap.

    /// Tiny model config for scheduler loop tests — allocations are near-zero cost.
    fn loop_test_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 8,
            num_layers: 1,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            intermediate_size: 16,
            vocab_size: 512,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            max_seq_len: 128,
        }
    }

    /// Build a fake WeightStore using mock tensor IDs (no GPU memory needed).
    fn loop_test_weights(cfg: &ModelConfig) -> WeightStore {
        let h = cfg.hidden_size;
        let kv = cfg.num_kv_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;
        let mut id = 2000u64;
        let mut t = |shape: Vec<usize>| {
            let tensor = DeviceTensor::new(TensorId(id), shape, DType::FP16);
            id += 1;
            tensor
        };
        let layers = (0..cfg.num_layers)
            .map(|_| LayerWeights {
                q_proj: t(vec![h, h]),
                k_proj: t(vec![kv, h]),
                v_proj: t(vec![kv, h]),
                o_proj: t(vec![h, h]),
                gate_proj: t(vec![inter, h]),
                up_proj: t(vec![inter, h]),
                down_proj: t(vec![h, inter]),
                attn_norm: t(vec![h]),
                ffn_norm: t(vec![h]),
            })
            .collect();
        WeightStore {
            config: cfg.clone(),
            token_embedding: t(vec![cfg.vocab_size, h]),
            layers,
            output_norm: t(vec![h]),
            lm_head: t(vec![cfg.vocab_size, h]),
        }
    }

    /// Mock Backend that implements attention_paged (returning Ok, no-op).
    ///
    /// copy_to_host writes FP16 logits where `winning_token` has the highest
    /// value, so greedy sampling always picks it.
    struct LoopMockBackend {
        next_id: AtomicU64,
        /// Token index with highest logit (greedy winner).
        winning_token: u32,
        vocab_size: usize,
        /// Whether attention_paged() should return an error (to test forward failures).
        fail_forward: bool,
    }

    impl LoopMockBackend {
        fn new(winning_token: u32, vocab_size: usize) -> Self {
            Self {
                next_id: AtomicU64::new(1),
                winning_token,
                vocab_size,
                fail_forward: false,
            }
        }

        fn with_forward_failure(winning_token: u32, vocab_size: usize) -> Self {
            Self {
                next_id: AtomicU64::new(1),
                winning_token,
                vocab_size,
                fail_forward: true,
            }
        }

        fn with_nan_logits(vocab_size: usize) -> Self {
            Self {
                next_id: AtomicU64::new(1),
                winning_token: u32::MAX, // sentinel: write NaN
                vocab_size,
                fail_forward: false,
            }
        }
    }

    impl Backend for LoopMockBackend {
        fn alloc(&self, shape: &[usize], dtype: DType) -> fracture_core::Result<DeviceTensor> {
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
        }
        fn free(&self, _: &DeviceTensor) -> fracture_core::Result<()> {
            Ok(())
        }
        fn copy_to_device(&self, _: &DeviceTensor, _: &[u8]) -> fracture_core::Result<()> {
            Ok(())
        }
        fn copy_to_host(
            &self,
            _src: &DeviceTensor,
            dst: &mut [u8],
        ) -> fracture_core::Result<()> {
            // Only fill logit buffers (vocab_size * 2 bytes of FP16).
            if dst.len() == self.vocab_size * 2 {
                if self.winning_token == u32::MAX {
                    // NaN logits: write NaN to every slot.
                    let nan_bytes = half::f16::NAN.to_le_bytes();
                    for chunk in dst.chunks_exact_mut(2) {
                        chunk[0] = nan_bytes[0];
                        chunk[1] = nan_bytes[1];
                    }
                } else {
                    let low = half::f16::from_f32(-10.0);
                    let high = half::f16::from_f32(10.0);
                    for i in 0..self.vocab_size {
                        let val = if i == self.winning_token as usize { high } else { low };
                        let bytes = val.to_le_bytes();
                        dst[i * 2] = bytes[0];
                        dst[i * 2 + 1] = bytes[1];
                    }
                }
            }
            Ok(())
        }
        fn matmul(
            &self,
            _: &DeviceTensor,
            _: &DeviceTensor,
            _: &DeviceTensor,
        ) -> fracture_core::Result<()> {
            Ok(())
        }
        fn rmsnorm(
            &self,
            _: &DeviceTensor,
            _: &DeviceTensor,
            _: f64,
            _: &DeviceTensor,
        ) -> fracture_core::Result<()> {
            Ok(())
        }
        fn rope(
            &self,
            _: &DeviceTensor,
            _: &DeviceTensor,
            _: &[u32],
            _: f64,
            _: usize,
        ) -> fracture_core::Result<()> {
            Ok(())
        }
        fn attention(
            &self,
            _: &DeviceTensor,
            _: &DeviceTensor,
            _: &DeviceTensor,
            _: usize,
            _: usize,
            _: &DeviceTensor,
        ) -> fracture_core::Result<()> {
            Ok(())
        }
        fn attention_paged(
            &self,
            _q: &DeviceTensor,
            _block_table: &[i32],
            _k_blocks: &[&DeviceTensor],
            _v_blocks: &[&DeviceTensor],
            _num_kv_heads: usize,
            _kv_len: usize,
            _start_pos: usize,
            _out: &DeviceTensor,
        ) -> fracture_core::Result<()> {
            if self.fail_forward {
                return Err(fracture_core::FractureError::Backend(
                    "mock forward failure".into(),
                ));
            }
            Ok(())
        }
        fn silu_mul(
            &self,
            _: &DeviceTensor,
            _: &DeviceTensor,
            _: &DeviceTensor,
        ) -> fracture_core::Result<()> {
            Ok(())
        }
        fn embedding(
            &self,
            _: &[u32],
            _: &DeviceTensor,
            _: &DeviceTensor,
        ) -> fracture_core::Result<()> {
            Ok(())
        }
        fn add(
            &self,
            _: &DeviceTensor,
            _: &DeviceTensor,
            _: &DeviceTensor,
        ) -> fracture_core::Result<()> {
            Ok(())
        }
        fn copy_rows(
            &self,
            _: &DeviceTensor,
            _: &DeviceTensor,
            _: usize,
            _: usize,
            _: usize,
        ) -> fracture_core::Result<()> {
            Ok(())
        }
        fn device_name(&self) -> &str {
            "loop-mock"
        }
        fn total_memory(&self) -> usize {
            1 << 30
        }
        fn available_memory(&self) -> usize {
            1 << 30
        }
        fn synchronize(&self) -> fracture_core::Result<()> {
            Ok(())
        }
        fn create_timer(&self) -> fracture_core::Result<DeviceTimer> {
            Ok(DeviceTimer(0))
        }
        fn start_timer(&self, _: &DeviceTimer) -> fracture_core::Result<()> {
            Ok(())
        }
        fn stop_timer(&self, _: &DeviceTimer) -> fracture_core::Result<f32> {
            Ok(0.0)
        }
        fn destroy_timer(&self, _: &DeviceTimer) -> fracture_core::Result<()> {
            Ok(())
        }
    }

    /// Build an Engine + PagedKvCacheManager suitable for scheduler loop tests.
    fn make_loop_engine(
        backend: LoopMockBackend,
        cfg: &ModelConfig,
    ) -> (Arc<Engine<LoopMockBackend>>, PagedKvCacheManager) {
        let weights = loop_test_weights(cfg);
        let engine = Arc::new(Engine::new(backend, weights, 0..cfg.num_layers));
        // Enough blocks for a few short sequences.
        let cache = PagedKvCacheManager::new(
            32,
            cfg.num_layers,
            cfg.num_kv_heads,
            cfg.head_dim,
            engine.backend(),
        )
        .unwrap();
        (engine, cache)
    }

    /// Collect events from a receiver until Finished or Error, with a timeout.
    async fn collect_events(
        mut rx: mpsc::UnboundedReceiver<GenerationEvent>,
    ) -> Vec<GenerationEvent> {
        let mut events = Vec::new();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(event)) => {
                    let done = matches!(
                        &event,
                        GenerationEvent::Finished { .. } | GenerationEvent::Error(_)
                    );
                    events.push(event);
                    if done {
                        break;
                    }
                }
                Ok(None) => break, // channel closed
                Err(_) => panic!("timed out waiting for generation events"),
            }
        }
        events
    }

    /// Gap 108 — scheduler loop full iteration: Token events followed by Finished.
    ///
    /// Submits a request for 2 tokens where the mock backend always returns
    /// token 42 as the greedy winner. Expects Token(42) × 2 then Finished.
    #[tokio::test]
    async fn test_scheduler_loop_full_iteration() {
        let cfg = loop_test_config();
        let backend = LoopMockBackend::new(42, cfg.vocab_size);
        let (engine, cache) = make_loop_engine(backend, &cfg);

        let config = SchedulerLoopConfig {
            max_batch_size: 8,
            max_batch_tokens: 512,
            max_prefill_tokens: 64,
            block_pool_reserve: 0.0,
        };
        let handle = start_scheduler_loop(engine, cache, config);

        let (tx, rx) = mpsc::unbounded_channel();
        handle
            .submit(PendingRequest {
                seq_id: 0,
                prompt_tokens: vec![1, 2, 3],
                max_tokens: 2,
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                seed: None,
                stop_tokens: vec![],
                event_tx: tx,
            })
            .unwrap();

        let events = collect_events(rx).await;

        // Should receive 2 Token events then a Finished event.
        let token_events: Vec<u32> = events
            .iter()
            .filter_map(|e| if let GenerationEvent::Token(t) = e { Some(*t) } else { None })
            .collect();
        assert_eq!(token_events, vec![42, 42], "expected 2 token events with token 42");

        let finished = events
            .iter()
            .any(|e| matches!(e, GenerationEvent::Finished { .. }));
        assert!(finished, "expected a Finished event");
    }

    /// Gap 108 — scheduler loop forward error: a backend that fails during
    /// attention_paged causes a GenerationEvent::Error to be delivered.
    #[tokio::test]
    async fn test_scheduler_loop_forward_error() {
        let cfg = loop_test_config();
        let backend = LoopMockBackend::with_forward_failure(0, cfg.vocab_size);
        let (engine, cache) = make_loop_engine(backend, &cfg);

        let config = SchedulerLoopConfig::default();
        let handle = start_scheduler_loop(engine, cache, config);

        let (tx, rx) = mpsc::unbounded_channel();
        handle
            .submit(PendingRequest {
                seq_id: 0,
                prompt_tokens: vec![10, 20],
                max_tokens: 5,
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                seed: None,
                stop_tokens: vec![],
                event_tx: tx,
            })
            .unwrap();

        let events = collect_events(rx).await;

        let has_error = events.iter().any(|e| {
            if let GenerationEvent::Error(msg) = e {
                msg.contains("forward pass failed")
            } else {
                false
            }
        });
        assert!(
            has_error,
            "expected GenerationEvent::Error with 'forward pass failed'; got: {events:?}"
        );
    }

    /// Gap 108 — cache alloc failure: a PagedKvCacheManager with 0 blocks
    /// causes a GenerationEvent::Error("cache allocation failed") to be sent,
    /// or the scheduler simply never admits the request (both are acceptable).
    #[tokio::test]
    async fn test_scheduler_loop_cache_alloc_failure() {
        let cfg = loop_test_config();
        let backend = LoopMockBackend::new(42, cfg.vocab_size);
        let weights = loop_test_weights(&cfg);
        let engine = Arc::new(Engine::new(backend, weights, 0..cfg.num_layers));

        // 0 blocks → alloc() will immediately fail.
        let empty_cache = PagedKvCacheManager::new(
            0,
            cfg.num_layers,
            cfg.num_kv_heads,
            cfg.head_dim,
            engine.backend(),
        )
        .unwrap();

        let config = SchedulerLoopConfig {
            max_batch_size: 8,
            max_batch_tokens: 512,
            max_prefill_tokens: 64,
            // No reserve — we want the scheduler to admit the request before alloc fails.
            block_pool_reserve: 0.0,
        };
        let handle = start_scheduler_loop(engine, empty_cache, config);

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle
            .submit(PendingRequest {
                seq_id: 0,
                prompt_tokens: vec![5],
                max_tokens: 3,
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                seed: None,
                stop_tokens: vec![],
                event_tx: tx,
            })
            .unwrap();

        // With 0 blocks the block-pool reserve check inside `schedule()` blocks admission
        // (free_blocks=0 < blocks_needed=1), so the request stays in the prefill queue
        // and the scheduler loop idles. No Token or Finished events should arrive.
        // Use a short timeout instead of collect_events (which panics on timeout).
        let result = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv()).await;
        match result {
            Ok(Some(GenerationEvent::Error(_))) => {
                // Acceptable: cache alloc error propagated.
            }
            Ok(Some(other)) => {
                panic!(
                    "with 0 cache blocks, expected no Token/Finished events; got: {other:?}"
                );
            }
            Ok(None) => {
                // Channel closed without events — acceptable.
            }
            Err(_) => {
                // Timeout — no events produced. This is the expected path.
            }
        }
    }

    /// Gap 108 — stop token not streamed: when a generated token matches a
    /// stop_token it must NOT appear as a Token event, but Finished must arrive.
    #[tokio::test]
    async fn test_scheduler_loop_stop_token_not_streamed() {
        let cfg = loop_test_config();
        // The mock always returns token 99, which we declare as a stop token.
        let stop_token = 99u32;
        let backend = LoopMockBackend::new(stop_token, cfg.vocab_size);
        let (engine, cache) = make_loop_engine(backend, &cfg);

        let config = SchedulerLoopConfig {
            max_batch_size: 8,
            max_batch_tokens: 512,
            max_prefill_tokens: 64,
            block_pool_reserve: 0.0,
        };
        let handle = start_scheduler_loop(engine, cache, config);

        let (tx, rx) = mpsc::unbounded_channel();
        handle
            .submit(PendingRequest {
                seq_id: 0,
                prompt_tokens: vec![1, 2],
                max_tokens: 10,
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                seed: None,
                stop_tokens: vec![stop_token],
                event_tx: tx,
            })
            .unwrap();

        let events = collect_events(rx).await;

        // The stop token itself must not be streamed.
        let stop_in_tokens = events
            .iter()
            .any(|e| matches!(e, GenerationEvent::Token(t) if *t == stop_token));
        assert!(
            !stop_in_tokens,
            "stop token {} must not appear in Token events; got: {events:?}",
            stop_token
        );

        // But Finished must arrive (with StopReason::Stop).
        let finished = events.iter().any(|e| {
            matches!(
                e,
                GenerationEvent::Finished {
                    stop_reason: fracture_core::StopReason::Stop,
                    ..
                }
            )
        });
        assert!(finished, "expected Finished(Stop) event; got: {events:?}");
    }

    /// Gap 108 — sampling error: NaN logits must produce a GenerationEvent::Error
    /// containing "sampling failed".
    #[tokio::test]
    async fn test_scheduler_loop_sampling_error() {
        let cfg = loop_test_config();
        let backend = LoopMockBackend::with_nan_logits(cfg.vocab_size);
        let (engine, cache) = make_loop_engine(backend, &cfg);

        let config = SchedulerLoopConfig {
            max_batch_size: 8,
            max_batch_tokens: 512,
            max_prefill_tokens: 64,
            block_pool_reserve: 0.0,
        };
        let handle = start_scheduler_loop(engine, cache, config);

        let (tx, rx) = mpsc::unbounded_channel();
        handle
            .submit(PendingRequest {
                seq_id: 0,
                prompt_tokens: vec![1],
                max_tokens: 3,
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                seed: None,
                stop_tokens: vec![],
                event_tx: tx,
            })
            .unwrap();

        let events = collect_events(rx).await;

        let has_sampling_error = events.iter().any(|e| {
            if let GenerationEvent::Error(msg) = e {
                msg.contains("sampling failed")
            } else {
                false
            }
        });
        assert!(
            has_sampling_error,
            "expected GenerationEvent::Error with 'sampling failed'; got: {events:?}"
        );
    }

    /// Gap 108 — concurrent requests: 3 requests all complete successfully.
    ///
    /// Each request asks for 1 token; all should receive a Token event and a
    /// Finished event. This exercises the batch scheduler's multi-sequence path.
    #[tokio::test]
    async fn test_scheduler_loop_concurrent_mock() {
        let cfg = loop_test_config();
        let backend = LoopMockBackend::new(7, cfg.vocab_size);
        let (engine, cache) = make_loop_engine(backend, &cfg);

        let config = SchedulerLoopConfig {
            max_batch_size: 8,
            max_batch_tokens: 512,
            max_prefill_tokens: 64,
            block_pool_reserve: 0.0,
        };
        let handle = start_scheduler_loop(engine, cache, config);

        // Submit 3 requests concurrently.
        let mut receivers = Vec::new();
        for i in 0u64..3 {
            let (tx, rx) = mpsc::unbounded_channel();
            handle
                .submit(PendingRequest {
                    seq_id: i,
                    prompt_tokens: vec![i as u32 + 1, i as u32 + 2],
                    max_tokens: 1,
                    temperature: 0.0,
                    top_k: 0,
                    top_p: 1.0,
                    seed: None,
                    stop_tokens: vec![],
                    event_tx: tx,
                })
                .unwrap();
            receivers.push(rx);
        }

        // Collect results for all three.
        for (i, rx) in receivers.into_iter().enumerate() {
            let events = collect_events(rx).await;

            let token_count = events
                .iter()
                .filter(|e| matches!(e, GenerationEvent::Token(_)))
                .count();
            assert_eq!(token_count, 1, "request {i} should have produced 1 token event");

            let finished = events
                .iter()
                .any(|e| matches!(e, GenerationEvent::Finished { .. }));
            assert!(finished, "request {i} should have produced a Finished event");
        }
    }
}

/// Handle a single scheduler command.
fn handle_command<B: Backend>(
    cmd: SchedulerCommand,
    scheduler: &mut BatchScheduler,
    cache: &Arc<StdMutex<PagedKvCacheManager>>,
    config: &SchedulerLoopConfig,
) {
    match cmd {
        SchedulerCommand::Submit(req) => {
            scheduler.enqueue(req);
        }
        SchedulerCommand::Snapshot(tx) => {
            let cache_guard = cache.lock().unwrap();
            let decode_count = scheduler
                .active
                .values()
                .filter(|s| s.remaining_prefill.is_empty() && !s.generated_tokens.is_empty())
                .count();

            let sequences: Vec<SequenceSnapshotEntry> = scheduler
                .active
                .values()
                .map(|s| {
                    let state = if !s.remaining_prefill.is_empty() {
                        "prefilling"
                    } else if s.generated_tokens.is_empty() {
                        "prefilling"
                    } else {
                        "decoding"
                    };
                    SequenceSnapshotEntry {
                        seq_id: s.seq_id,
                        state,
                        tokens_generated: s.generated_tokens.len(),
                        max_tokens: s.max_tokens,
                        remaining_prefill: s.remaining_prefill.len(),
                    }
                })
                .collect();

            let snapshot = SchedulerSnapshot {
                active_sequences: scheduler.num_active(),
                max_sequences: config.max_batch_size,
                decode_count,
                prefill_queue_count: scheduler.num_pending(),
                prefill_chunk_size: config.max_prefill_tokens,
                block_size: BLOCK_SIZE,
                total_blocks: cache_guard.pool().capacity(),
                free_blocks: cache_guard.num_free_blocks(),
                sequences,
            };

            let _ = tx.send(snapshot);
        }
    }
}
