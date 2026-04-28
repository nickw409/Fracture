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
    let mut cache = make_cache(&backend);
    let mut sched = BatchScheduler::new(64, 4096, 512, 0.1);
    let decision = sched.schedule(&mut cache);
    assert_eq!(decision.total_tokens, 0);
    assert!(decision.prefills.is_empty());
    assert!(decision.decodes.is_empty());
    assert!(!sched.has_work());
}

#[test]
fn test_scheduler_admits_new_request() {
    let backend = MockBackend::new();
    let mut cache = make_cache(&backend);
    let mut sched = BatchScheduler::new(64, 4096, 512, 0.1);

    let _rx = make_request(&mut sched, 5);
    assert!(sched.has_work());

    let decision = sched.schedule(&mut cache);
    assert_eq!(decision.prefills.len(), 1);
    assert_eq!(decision.prefills[0].token_ids.len(), 5);
    assert_eq!(decision.total_tokens, 5);
    assert_eq!(sched.num_active(), 1);
    assert_eq!(sched.num_pending(), 0);
}

#[test]
fn test_scheduler_prefill_chunking() {
    let backend = MockBackend::new();
    let mut cache = make_cache(&backend);
    // max_prefill_tokens = 10, prompt = 25 tokens → 3 chunks (10, 10, 5)
    let mut sched = BatchScheduler::new(64, 4096, 10, 0.1);

    let _rx = make_request(&mut sched, 25);

    // Iteration 1: first chunk of 10
    let d1 = sched.schedule(&mut cache);
    assert_eq!(d1.prefills.len(), 1);
    assert_eq!(d1.prefills[0].token_ids.len(), 10);
    assert_eq!(d1.total_tokens, 10);

    // Iteration 2: second chunk of 10
    let d2 = sched.schedule(&mut cache);
    assert_eq!(d2.prefills.len(), 1);
    assert_eq!(d2.prefills[0].token_ids.len(), 10);

    // Iteration 3: final chunk of 5
    let d3 = sched.schedule(&mut cache);
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
    let mut cache = make_cache(&backend);
    let mut sched = BatchScheduler::new(64, 4096, 512, 0.1);

    // Add and "prefill" a sequence, then give it a generated token.
    let _rx1 = make_request(&mut sched, 3);
    let _d = sched.schedule(&mut cache); // prefills seq 0

    // Manually mark it as having a generated token (simulating sampling).
    let seq = sched.active.get_mut(&0).unwrap();
    seq.generated_tokens.push(42);

    // Add a new request.
    let _rx2 = make_request(&mut sched, 5);

    // Schedule: decode for seq 0 should come first, then prefill for seq 1.
    let decision = sched.schedule(&mut cache);
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
    let mut cache = make_cache(&backend);
    // max_batch_tokens = 8
    let mut sched = BatchScheduler::new(64, 8, 512, 0.1);

    // Two requests: 5 tokens + 5 tokens = 10 > 8
    // First admitted fully (5 tokens), second chunked to 3 (filling to 8).
    let _rx1 = make_request(&mut sched, 5);
    let _rx2 = make_request(&mut sched, 5);

    let decision = sched.schedule(&mut cache);
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
    let _cache = make_cache(&backend);
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

/// Verify that cleanup_completed() sends a GenerationEvent::Finished with
/// StopReason::Length when a sequence hits its max_tokens limit.
///
/// The previous test dropped `_rx` immediately, so the send was never
/// verified. This test retains the receiver and asserts the event is delivered.
#[test]
fn test_scheduler_cleanup_sends_length_event() {
    let mut sched = BatchScheduler::new(64, 4096, 512, 0.1);

    let (tx, mut rx) = mpsc::unbounded_channel();
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
        generated_tokens: vec![10, 20, 30], // exactly at max_tokens
        event_tx: tx,
        remaining_prefill: Vec::new(),
    });

    let removed = sched.cleanup_completed();
    assert_eq!(removed.len(), 1, "sequence should be removed");
    assert_eq!(sched.num_active(), 0);

    // The Finished event must have been sent to the client channel.
    let event = rx.try_recv().expect("expected GenerationEvent::Finished on rx");
    match event {
        GenerationEvent::Finished { stop_reason, completion_tokens } => {
            assert_eq!(stop_reason, StopReason::Length, "stop reason should be Length");
            assert_eq!(completion_tokens, 3, "completion_tokens should equal generated token count");
        }
        other => panic!("expected Finished event, got: {other:?}"),
    }
}

#[test]
fn test_scheduler_cleanup_on_stop_token() {
    let backend = MockBackend::new();
    let _cache = make_cache(&backend);
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

#[test]
fn test_admission_rejected_by_block_pool() {
    let backend = MockBackend::new();
    // 3 blocks × BLOCK_SIZE(16) = 48 tokens max capacity.
    let mut cache = PagedKvCacheManager::new(3, 2, 2, 16, &backend).unwrap();
    // block_pool_reserve = 0.5 → reserved = ceil(3 * 0.5) = 2 blocks.
    // Available after reserve = 3 - 2 = 1 block = 16 tokens.
    let mut sched = BatchScheduler::new(64, 4096, 512, 0.5);

    // Prompt of 20 tokens → needs ceil(20/16) = 2 blocks > 1 available.
    let _rx = make_request(&mut sched, 20);

    let decision = sched.schedule(&mut cache);
    assert!(decision.prefills.is_empty(), "should not admit when blocks insufficient after reserve");
    assert_eq!(decision.total_tokens, 0);
    // Request stays in the prefill queue.
    assert_eq!(sched.num_pending(), 1);
}

#[test]
fn test_block_pool_reserve_prevents_starvation() {
    let backend = MockBackend::new();
    // 6 blocks total; reserve = 0.5 → reserved = 3 blocks.
    let mut cache = PagedKvCacheManager::new(6, 2, 2, 16, &backend).unwrap();
    let mut sched = BatchScheduler::new(64, 4096, 512, 0.5);

    // Insert an active decode sequence (already prefilled, has a generated token).
    let (tx1, _rx1) = mpsc::unbounded_channel();
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
        current_pos: 10,
        generated_tokens: vec![42],
        event_tx: tx1,
        remaining_prefill: Vec::new(),
    });

    // Enqueue a new prefill request needing ceil(48/16)=3 blocks.
    // Available after reserve: 6 free - 3 reserved = 3.
    // But 3 blocks needed == 3 available, so it should be admitted at this point.
    // Enqueue a bigger one: 64 tokens → needs 4 blocks > 3 available.
    let _rx2 = make_request(&mut sched, 64);

    let decision = sched.schedule(&mut cache);
    // Active decode should still be scheduled.
    assert_eq!(decision.decodes.len(), 1, "active decode should still run");
    assert_eq!(decision.decodes[0].seq_id, seq_id);
    // New prefill should be blocked (needs 4 blocks, only 3 available after reserve).
    assert!(decision.prefills.is_empty(), "prefill should be blocked by reserve");
    assert_eq!(sched.num_pending(), 1);
}

#[test]
fn test_max_batch_size_limit() {
    let backend = MockBackend::new();
    let mut cache = make_cache(&backend);
    let mut sched = BatchScheduler::new(2, 4096, 512, 0.1);

    // Admit 3 sequences via prefill, then move them to active decode state.
    let mut rxs = Vec::new();
    for _ in 0..3 {
        rxs.push(make_request(&mut sched, 3));
    }
    // First schedule admits up to max_batch_size=2 prefills.
    let d1 = sched.schedule(&mut cache);
    assert_eq!(d1.prefills.len(), 2);
    assert_eq!(sched.num_pending(), 1);

    // Give the 2 admitted sequences generated tokens so they become decodes.
    for (_, seq) in sched.active.iter_mut() {
        seq.generated_tokens.push(42);
    }

    // Schedule again: 2 active decodes fill max_batch_size, third still pending.
    let d2 = sched.schedule(&mut cache);
    assert!(d2.decodes.len() <= 2, "decodes should be capped at max_batch_size");
    assert_eq!(
        d2.decodes.len() + d2.prefills.len(),
        2,
        "total scheduled should not exceed max_batch_size"
    );
}

#[test]
fn test_cleanup_completed_frees_cache_blocks() {
    let backend = MockBackend::new();
    let mut cache = PagedKvCacheManager::new(10, 2, 2, 16, &backend).unwrap();
    let mut sched = BatchScheduler::new(64, 4096, 512, 0.1);

    // Allocate a cache handle (takes 1 block from pool).
    let handle = cache.alloc().unwrap();
    let free_before = cache.num_free_blocks();

    // Insert an active sequence that has hit its stop token.
    let (tx, _rx) = mpsc::unbounded_channel();
    let seq_id = sched.next_seq_id();
    sched.active.insert(seq_id, ActiveSequence {
        seq_id,
        handle,
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
    assert_eq!(removed[0].0, seq_id);

    // Free the returned handle and verify blocks returned to pool.
    cache.free(removed[0].1).unwrap();
    let free_after = cache.num_free_blocks();
    assert!(free_after > free_before, "freeing handle should return blocks to pool");
}
