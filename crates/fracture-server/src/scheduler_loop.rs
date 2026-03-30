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
