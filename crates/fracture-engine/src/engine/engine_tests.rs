// prefill-decode-consistency is tested in bins/fracture-server-cuda/tests/gpu_integration.rs
// via test_gpu_prefill_decode_consistency, which runs on the actual CudaBackend with a
// tiny model built directly on the GPU.

use super::*;
use fracture_core::{DType, DeviceTimer, FractureError, ModelConfig, TensorId};
use fracture_gguf::{LayerWeights, WeightStore};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

struct MockBackend {
    next_id: AtomicU64,
    fail_on_matmul: bool,
}

impl MockBackend {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1000),
            fail_on_matmul: false,
        }
    }

    fn failing_matmul() -> Self {
        Self {
            next_id: AtomicU64::new(1000),
            fail_on_matmul: true,
        }
    }
}

impl Backend for MockBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }
    fn free(&self, _tensor: &DeviceTensor) -> Result<()> { Ok(()) }
    fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> { Ok(()) }
    fn copy_to_host(&self, _src: &DeviceTensor, dst: &mut [u8]) -> Result<()> {
        // Zero-fill so logits decode to valid f16 zeros
        dst.fill(0);
        Ok(())
    }
    fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> {
        if self.fail_on_matmul {
            return Err(FractureError::Backend("mock matmul failure".into()));
        }
        Ok(())
    }
    fn rmsnorm(&self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _positions: &[u32], _theta: f64, _head_dim: usize) -> Result<()> { Ok(()) }
    fn attention(&self, _q: &DeviceTensor, _k_cache: &DeviceTensor, _v_cache: &DeviceTensor, _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn attention_paged(&self, _q: &DeviceTensor, _bt: &[i32], _kb: &[&DeviceTensor], _vb: &[&DeviceTensor], _nkv: usize, _kvl: usize, _sp: usize, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn silu_mul(&self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn embedding(&self, _token_ids: &[u32], _embedding_table: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn copy_rows(&self, _src: &DeviceTensor, _dst: &DeviceTensor, _src_offset: usize, _dst_offset: usize, _count: usize) -> Result<()> { Ok(()) }
    fn device_name(&self) -> &str { "mock" }
    fn total_memory(&self) -> usize { 1_000_000_000 }
    fn available_memory(&self) -> usize { 1_000_000_000 }
    fn synchronize(&self) -> Result<()> { Ok(()) }
    fn create_timer(&self) -> Result<DeviceTimer> { Ok(DeviceTimer(0)) }
    fn start_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }
    fn stop_timer(&self, _timer: &DeviceTimer) -> Result<f32> { Ok(0.0) }
    fn destroy_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }
}

fn test_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 8,
        num_layers: 1,
        num_q_heads: 2,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 16,
        vocab_size: 32,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        max_seq_len: 512,
    }
}

fn mock_tensor(id: u64, shape: Vec<usize>) -> DeviceTensor {
    DeviceTensor::new(TensorId(id), shape, DType::FP16)
}

fn mock_weights(cfg: &ModelConfig) -> WeightStore {
    let h = cfg.hidden_size;
    let kv = cfg.num_kv_heads * cfg.head_dim;
    let inter = cfg.intermediate_size;
    let mut id = 1u64;
    let mut t = |shape: Vec<usize>| -> DeviceTensor {
        let t = mock_tensor(id, shape);
        id += 1;
        t
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

/// Verify that forward() with empty token_ids returns an error, not a panic.
#[test]
fn test_forward_empty_token_ids_returns_error() {
    let cfg = test_config();
    let backend = MockBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..cfg.num_layers);
    let mut cache = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(engine.backend()).unwrap();

    // Empty token_ids should fail (seq_len=0 causes zero-size allocs or index OOB)
    let result = engine.forward(&[], &[], &mut cache, handle, None);
    assert!(result.is_err(), "forward with empty tokens should return Err");
}

/// Verify that a backend error during forward() propagates as FractureError::Backend.
#[test]
fn test_forward_backend_error_propagation() {
    let cfg = test_config();
    let backend = MockBackend::failing_matmul();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..cfg.num_layers);
    let mut cache = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(engine.backend()).unwrap();

    let result = engine.forward(&[1], &[0], &mut cache, handle, None);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, FractureError::Backend(_)),
        "expected Backend error, got: {err:?}"
    );
    assert!(err.to_string().contains("mock matmul failure"));
}

// ── RecordingMockBackend ──────────────────────────────────────

/// A mock backend that records profiling-related calls (timers and markers)
/// while providing the same no-op behavior as MockBackend for compute ops.
struct RecordingMockBackend {
    next_id: AtomicU64,
    timer_create_count: AtomicU64,
    marker_names: Mutex<Vec<String>>,
    marker_pop_count: AtomicU64,
    timer_stop_value: f32,
}

impl RecordingMockBackend {
    fn new(timer_stop_value: f32) -> Self {
        Self {
            next_id: AtomicU64::new(1000),
            timer_create_count: AtomicU64::new(0),
            marker_names: Mutex::new(Vec::new()),
            marker_pop_count: AtomicU64::new(0),
            timer_stop_value,
        }
    }
}

impl Backend for RecordingMockBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }
    fn free(&self, _tensor: &DeviceTensor) -> Result<()> { Ok(()) }
    fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> { Ok(()) }
    fn copy_to_host(&self, _src: &DeviceTensor, dst: &mut [u8]) -> Result<()> {
        dst.fill(0);
        Ok(())
    }
    fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn rmsnorm(&self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _positions: &[u32], _theta: f64, _head_dim: usize) -> Result<()> { Ok(()) }
    fn attention(&self, _q: &DeviceTensor, _k_cache: &DeviceTensor, _v_cache: &DeviceTensor, _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn silu_mul(&self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn embedding(&self, _token_ids: &[u32], _embedding_table: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn copy_rows(&self, _src: &DeviceTensor, _dst: &DeviceTensor, _src_offset: usize, _dst_offset: usize, _count: usize) -> Result<()> { Ok(()) }
    fn device_name(&self) -> &str { "recording-mock" }
    fn total_memory(&self) -> usize { 1_000_000_000 }
    fn available_memory(&self) -> usize { 1_000_000_000 }
    fn synchronize(&self) -> Result<()> { Ok(()) }

    fn create_timer(&self) -> Result<DeviceTimer> {
        self.timer_create_count.fetch_add(1, Ordering::Relaxed);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceTimer(id))
    }
    fn start_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }
    fn stop_timer(&self, _timer: &DeviceTimer) -> Result<f32> {
        Ok(self.timer_stop_value)
    }
    fn destroy_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }

    fn marker_push(&self, name: &str) {
        self.marker_names.lock().unwrap().push(name.to_string());
    }
    fn marker_pop(&self) {
        self.marker_pop_count.fetch_add(1, Ordering::Relaxed);
    }
}

// ── Profiling dispatch tests ──────────────────────────────────

/// Verify that forward() with profiling enabled populates ForwardProfile
/// with the correct number of LayerProfile entries and timing data.
#[test]
fn test_forward_with_profiling_collects_layer_profiles() {
    let cfg = test_config(); // 1 layer
    let backend = RecordingMockBackend::new(1.0); // stop_timer returns 1.0 ms
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..cfg.num_layers);
    let mut cache = KvCacheManager::new(
        cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len,
    );
    let handle = cache.alloc(engine.backend()).unwrap();

    let mut profile = ForwardProfile {
        total_ms: 0.0,
        prefill: false,
        seq_len: 0,
        layer_profiles: Vec::new(),
    };

    let result = engine.forward(&[1], &[0], &mut cache, handle, Some(&mut profile));
    assert!(result.is_ok(), "forward should succeed: {:?}", result.err());

    assert_eq!(
        profile.layer_profiles.len(),
        1,
        "should have exactly 1 layer profile for 1-layer config"
    );
    assert_eq!(profile.layer_profiles[0].layer_idx, 0);
    assert!(
        profile.total_ms > 0.0,
        "total_ms should be positive when stop_timer returns 1.0, got {}",
        profile.total_ms
    );
}

/// Verify that forward() emits NVTX markers (marker_push/marker_pop)
/// for each layer, regardless of whether profiling is enabled.
#[test]
fn test_forward_emits_nvtx_markers() {
    let cfg = test_config(); // 1 layer
    let backend = RecordingMockBackend::new(0.0);
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..cfg.num_layers);
    let mut cache = KvCacheManager::new(
        cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len,
    );
    let handle = cache.alloc(engine.backend()).unwrap();

    // Run with profile=None — markers should still fire
    let result = engine.forward(&[1], &[0], &mut cache, handle, None);
    assert!(result.is_ok(), "forward should succeed: {:?}", result.err());

    let marker_names = engine.backend().marker_names.lock().unwrap();
    assert!(
        marker_names.contains(&"layer_0".to_string()),
        "marker_push should be called with 'layer_0', got: {:?}",
        *marker_names
    );

    let pop_count = engine.backend().marker_pop_count.load(Ordering::Relaxed);
    assert_eq!(
        pop_count,
        marker_names.len() as u64,
        "marker_pop count ({}) should equal marker_push count ({})",
        pop_count,
        marker_names.len()
    );
}

/// Verify that forward() with profile=None does NOT create any GPU timers,
/// ensuring zero overhead when profiling is disabled.
#[test]
fn test_forward_no_profiling_skips_timers() {
    let cfg = test_config(); // 1 layer
    let backend = RecordingMockBackend::new(0.0);
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..cfg.num_layers);
    let mut cache = KvCacheManager::new(
        cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len,
    );
    let handle = cache.alloc(engine.backend()).unwrap();

    let result = engine.forward(&[1], &[0], &mut cache, handle, None);
    assert!(result.is_ok(), "forward should succeed: {:?}", result.err());

    let timer_count = engine.backend().timer_create_count.load(Ordering::Relaxed);
    assert_eq!(
        timer_count, 0,
        "create_timer should not be called when profiling is disabled, but was called {} times",
        timer_count
    );
}

// ── CopyRowsRecordingBackend ─────────────────────────────────

/// Records copy_rows calls to verify KV cache append behavior.
struct CopyRowsRecordingBackend {
    next_id: AtomicU64,
    /// (src_id, dst_id, src_offset, dst_offset, count)
    copy_rows_calls: Mutex<Vec<(u64, u64, usize, usize, usize)>>,
}

impl CopyRowsRecordingBackend {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1000),
            copy_rows_calls: Mutex::new(Vec::new()),
        }
    }
}

impl Backend for CopyRowsRecordingBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }
    fn free(&self, _tensor: &DeviceTensor) -> Result<()> { Ok(()) }
    fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> { Ok(()) }
    fn copy_to_host(&self, _src: &DeviceTensor, dst: &mut [u8]) -> Result<()> {
        dst.fill(0);
        Ok(())
    }
    fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn rmsnorm(&self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _positions: &[u32], _theta: f64, _head_dim: usize) -> Result<()> { Ok(()) }
    fn attention(&self, _q: &DeviceTensor, _k_cache: &DeviceTensor, _v_cache: &DeviceTensor, _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn silu_mul(&self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn embedding(&self, _token_ids: &[u32], _embedding_table: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn copy_rows(&self, src: &DeviceTensor, dst: &DeviceTensor, src_offset: usize, dst_offset: usize, count: usize) -> Result<()> {
        self.copy_rows_calls.lock().unwrap().push((src.id.0, dst.id.0, src_offset, dst_offset, count));
        Ok(())
    }
    fn device_name(&self) -> &str { "copy-rows-mock" }
    fn total_memory(&self) -> usize { 1_000_000_000 }
    fn available_memory(&self) -> usize { 1_000_000_000 }
    fn synchronize(&self) -> Result<()> { Ok(()) }
    fn create_timer(&self) -> Result<DeviceTimer> { Ok(DeviceTimer(0)) }
    fn start_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }
    fn stop_timer(&self, _timer: &DeviceTimer) -> Result<f32> { Ok(0.0) }
    fn destroy_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }
}

// ── Cache append tests ───────────────────────────────────────

/// Verify that prefill with multiple tokens calls copy_rows with
/// src_offset=0, dst_offset=0 (start_pos), count=seq_len for K and V.
#[test]
fn test_cache_append_prefill() {
    let cfg = test_config();
    let backend = CopyRowsRecordingBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..cfg.num_layers);
    let mut cache = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(engine.backend()).unwrap();

    // Prefill with 3 tokens
    let result = engine.forward(&[1, 2, 3], &[0, 1, 2], &mut cache, handle, None);
    assert!(result.is_ok(), "forward should succeed: {:?}", result.err());

    let calls = engine.backend().copy_rows_calls.lock().unwrap();
    // For 1 layer: 2 copy_rows (K and V), each with count=3, dst_offset=0
    let prefill_copies: Vec<_> = calls.iter().filter(|c| c.4 == 3).collect(); // count==3
    assert!(
        prefill_copies.len() >= 2,
        "expected at least 2 copy_rows with count=3 (K+V), got {}: {:?}",
        prefill_copies.len(), *calls
    );
    // All prefill copies should have dst_offset=0 (start_pos was 0)
    for c in &prefill_copies {
        assert_eq!(c.3, 0, "prefill dst_offset should be 0, got {}", c.3);
    }
}

/// Verify that a single-token decode step calls copy_rows with
/// count=1 and dst_offset=start_pos (the position after prefill).
#[test]
fn test_cache_append_decode() {
    let cfg = test_config();
    let backend = CopyRowsRecordingBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..cfg.num_layers);
    let mut cache = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(engine.backend()).unwrap();

    // Prefill with 3 tokens first
    engine.forward(&[1, 2, 3], &[0, 1, 2], &mut cache, handle, None).unwrap();

    // Clear recorded calls
    engine.backend().copy_rows_calls.lock().unwrap().clear();

    // Decode: single token at position 3
    let result = engine.forward(&[4], &[3], &mut cache, handle, None);
    assert!(result.is_ok(), "decode forward should succeed: {:?}", result.err());

    let calls = engine.backend().copy_rows_calls.lock().unwrap();
    // Should have copy_rows with count=1, dst_offset=3 (start_pos after prefill set seq_len=3)
    let decode_copies: Vec<_> = calls.iter().filter(|c| c.4 == 1).collect();
    assert!(
        decode_copies.len() >= 2,
        "expected at least 2 copy_rows with count=1 (K+V), got {}: {:?}",
        decode_copies.len(), *calls
    );
    for c in &decode_copies {
        assert_eq!(c.3, 3, "decode dst_offset should be 3 (start_pos), got {}", c.3);
    }
}

/// Verify that prefill copy_rows calls target the correct per-layer K and V
/// cache tensors in a 2-layer engine. For each layer, the dst tensor ID in
/// the copy_rows call must match the tensor ID returned by k_cache/v_cache
/// for that layer, not a tensor from a different layer.
#[test]
fn test_cache_append_prefill_multi_layer() {
    let mut cfg = test_config();
    cfg.num_layers = 2;
    let backend = CopyRowsRecordingBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..cfg.num_layers);

    // Allocate a 2-layer cache. Record the per-layer tensor IDs before running forward.
    let mut cache = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(engine.backend()).unwrap();

    // Capture k/v cache tensor IDs for each layer before forward.
    let k_ids: Vec<u64> = (0..cfg.num_layers)
        .map(|l| cache.k_cache(handle, l).unwrap().id.0)
        .collect();
    let v_ids: Vec<u64> = (0..cfg.num_layers)
        .map(|l| cache.v_cache(handle, l).unwrap().id.0)
        .collect();

    // Layer 0 and layer 1 must have distinct K and V tensor IDs.
    assert_ne!(k_ids[0], k_ids[1], "layer 0 and layer 1 K caches must be distinct tensors");
    assert_ne!(v_ids[0], v_ids[1], "layer 0 and layer 1 V caches must be distinct tensors");

    // Prefill with 3 tokens
    let result = engine.forward(&[1, 2, 3], &[0, 1, 2], &mut cache, handle, None);
    assert!(result.is_ok(), "2-layer forward should succeed: {:?}", result.err());

    let calls = engine.backend().copy_rows_calls.lock().unwrap();

    // Expect exactly 4 copy_rows calls: K+V for layer 0, K+V for layer 1.
    // Filter to count=3 (prefill copies).
    let prefill_copies: Vec<_> = calls.iter().filter(|c| c.4 == 3).collect();
    assert_eq!(
        prefill_copies.len(), 4,
        "expected 4 prefill copy_rows (K+V × 2 layers), got {}: {:?}",
        prefill_copies.len(), *calls
    );

    // The dst IDs in copy_rows calls must be exactly the set of per-layer K and V tensor IDs.
    let dst_ids: std::collections::HashSet<u64> = prefill_copies.iter().map(|c| c.1).collect();

    for layer in 0..cfg.num_layers {
        assert!(
            dst_ids.contains(&k_ids[layer]),
            "layer {layer} k_cache tensor (id={}) not found in copy_rows dst IDs: {:?}",
            k_ids[layer], dst_ids
        );
        assert!(
            dst_ids.contains(&v_ids[layer]),
            "layer {layer} v_cache tensor (id={}) not found in copy_rows dst IDs: {:?}",
            v_ids[layer], dst_ids
        );
    }

    // Each layer's K and V tensor IDs must appear in separate copy_rows calls
    // (no layer should have K and V written to the same tensor).
    for layer in 0..cfg.num_layers {
        assert_ne!(
            k_ids[layer], v_ids[layer],
            "layer {layer} K and V cache must be different tensors"
        );
    }
}

/// Verify partial layer range (head node): Engine with layer_range 0..1 out of 2 layers
/// returns Activations via forward_node() since it's not a tail node.
#[test]
fn test_partial_layer_range_head_node() {
    let mut cfg = test_config();
    cfg.num_layers = 2;
    let backend = MockBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..1);
    // Cache for 1 layer (this node's range)
    let mut cache = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(engine.backend()).unwrap();

    let node_config = NodeConfig::new(0..1, 2).unwrap();
    assert!(node_config.is_head());
    assert!(!node_config.is_tail());

    let input = NodeInput::TokenIds {
        ids: vec![1],
        positions: vec![0],
    };
    let result = engine.forward_node(input, &node_config, &mut cache, handle, None);
    assert!(result.is_ok(), "head node forward should succeed: {:?}", result.err());

    match result.unwrap() {
        NodeOutput::Activations(tensor) => {
            assert_eq!(tensor.shape, vec![1, cfg.hidden_size]);
        }
        NodeOutput::Logits(_) => panic!("head node should return Activations, not Logits"),
    }
}

/// Verify non-zero-starting layer range (tail node): Engine with layer_range 1..2 out of 2 layers
/// uses local indexing for weights and KV cache and returns Logits.
#[test]
fn test_nonzero_layer_range_tail_node() {
    let mut cfg = test_config();
    cfg.num_layers = 2;
    let backend = MockBackend::new();

    // Build weights with only 1 layer (representing model layer 1)
    let h = cfg.hidden_size;
    let kv = cfg.num_kv_heads * cfg.head_dim;
    let inter = cfg.intermediate_size;
    let mut id = 1u64;
    let mut t = |shape: Vec<usize>| -> DeviceTensor {
        let t = mock_tensor(id, shape);
        id += 1;
        t
    };
    let layer = LayerWeights {
        q_proj: t(vec![h, h]),
        k_proj: t(vec![kv, h]),
        v_proj: t(vec![kv, h]),
        o_proj: t(vec![h, h]),
        gate_proj: t(vec![inter, h]),
        up_proj: t(vec![inter, h]),
        down_proj: t(vec![h, inter]),
        attn_norm: t(vec![h]),
        ffn_norm: t(vec![h]),
    };
    let weights = WeightStore {
        config: cfg.clone(),
        token_embedding: t(vec![cfg.vocab_size, h]),
        layers: vec![layer],
        output_norm: t(vec![h]),
        lm_head: t(vec![cfg.vocab_size, h]),
    };

    let engine = Engine::new(backend, weights, 1..2);
    let mut cache = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(engine.backend()).unwrap();

    let node_config = NodeConfig::new(1..2, 2).unwrap();
    assert!(!node_config.is_head());
    assert!(node_config.is_tail());

    // Tail node receives activations from the head
    let fake_hidden = engine.backend().alloc(&[1, cfg.hidden_size], DType::FP16).unwrap();
    let input = NodeInput::Activations {
        hidden_states: fake_hidden,
        positions: vec![0],
    };
    let result = engine.forward_node(input, &node_config, &mut cache, handle, None);
    assert!(result.is_ok(), "tail node forward should succeed: {:?}", result.err());

    match result.unwrap() {
        NodeOutput::Logits(logits) => {
            assert_eq!(logits.len(), cfg.vocab_size);
        }
        NodeOutput::Activations(_) => panic!("tail node should return Logits, not Activations"),
    }
}

/// Sending TokenIds to a non-head node (tail) should return a Pipeline error.
#[test]
fn test_token_ids_to_non_head_returns_error() {
    let mut cfg = test_config();
    cfg.num_layers = 2;
    let backend = MockBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..2);
    let mut cache = KvCacheManager::new(2, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(engine.backend()).unwrap();

    // Create a tail NodeConfig (not head)
    let node_config = NodeConfig::new(1..2, 2).unwrap();
    assert!(!node_config.is_head());

    let input = NodeInput::TokenIds {
        ids: vec![1],
        positions: vec![0],
    };
    let result = engine.forward_node(input, &node_config, &mut cache, handle, None);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected error for TokenIds to non-head"),
    };
    assert!(err.to_string().contains("non-head"));
}

/// Sending Activations to a head node should return a Pipeline error.
#[test]
fn test_activations_to_head_returns_error() {
    let mut cfg = test_config();
    cfg.num_layers = 2;
    let backend = MockBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..2);
    let mut cache = KvCacheManager::new(2, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(engine.backend()).unwrap();

    let node_config = NodeConfig::new(0..1, 2).unwrap();
    assert!(node_config.is_head());

    let fake_hidden = engine.backend().alloc(&[1, cfg.hidden_size], DType::FP16).unwrap();
    let input = NodeInput::Activations {
        hidden_states: fake_hidden,
        positions: vec![0],
    };
    let result = engine.forward_node(input, &node_config, &mut cache, handle, None);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected error for Activations to head"),
    };
    assert!(err.to_string().contains("head"));
}

/// Full node (is_head + is_tail) returns Logits from TokenIds.
#[test]
fn test_full_node_forward_returns_logits() {
    let cfg = test_config();
    let backend = MockBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..cfg.num_layers);
    let mut cache = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(engine.backend()).unwrap();

    let node_config = NodeConfig::new(0..cfg.num_layers, cfg.num_layers).unwrap();
    assert!(node_config.is_full());

    let input = NodeInput::TokenIds {
        ids: vec![1],
        positions: vec![0],
    };
    let result = engine.forward_node(input, &node_config, &mut cache, handle, None);
    assert!(result.is_ok());
    match result.unwrap() {
        NodeOutput::Logits(logits) => {
            assert_eq!(logits.len(), cfg.vocab_size);
        }
        NodeOutput::Activations(_) => panic!("full node should return Logits"),
    }
}

// ── AllocCountingMockBackend ─────────────────────────────────

/// A mock backend that counts alloc() calls to verify scratch tensor reuse.
struct AllocCountingMockBackend {
    next_id: AtomicU64,
    alloc_count: AtomicU64,
}

impl AllocCountingMockBackend {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1000),
            alloc_count: AtomicU64::new(0),
        }
    }

    fn alloc_count(&self) -> u64 {
        self.alloc_count.load(Ordering::Relaxed)
    }
}

impl Backend for AllocCountingMockBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }
    fn free(&self, _tensor: &DeviceTensor) -> Result<()> { Ok(()) }
    fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> { Ok(()) }
    fn copy_to_host(&self, _src: &DeviceTensor, dst: &mut [u8]) -> Result<()> {
        dst.fill(0);
        Ok(())
    }
    fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn rmsnorm(&self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _positions: &[u32], _theta: f64, _head_dim: usize) -> Result<()> { Ok(()) }
    fn attention(&self, _q: &DeviceTensor, _k_cache: &DeviceTensor, _v_cache: &DeviceTensor, _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn silu_mul(&self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn embedding(&self, _token_ids: &[u32], _embedding_table: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn copy_rows(&self, _src: &DeviceTensor, _dst: &DeviceTensor, _src_offset: usize, _dst_offset: usize, _count: usize) -> Result<()> { Ok(()) }
    fn device_name(&self) -> &str { "alloc-counting-mock" }
    fn total_memory(&self) -> usize { 1_000_000_000 }
    fn available_memory(&self) -> usize { 1_000_000_000 }
    fn synchronize(&self) -> Result<()> { Ok(()) }
    fn create_timer(&self) -> Result<DeviceTimer> { Ok(DeviceTimer(0)) }
    fn start_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }
    fn stop_timer(&self, _timer: &DeviceTimer) -> Result<f32> { Ok(0.0) }
    fn destroy_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }
}

// ── Scratch tensor reuse test ────────────────────────────────

/// Verify that scratch tensors are allocated once before the layer loop and reused
/// across all layers. The alloc count should NOT scale with num_layers — only
/// weight and KV cache tensors scale with layer count.
///
/// Strategy: run forward passes with 2-layer and 4-layer configs. The difference
/// in alloc counts should be exactly the KV cache allocs that scale with layers
/// (2 per layer for K and V caches), NOT scratch tensors.
#[test]
fn test_scratch_tensor_reuse_across_layers() {
    // Helper: run a forward pass with N layers and return the alloc count.
    fn run_forward_and_count_allocs(num_layers: usize) -> u64 {
        let mut cfg = test_config();
        cfg.num_layers = num_layers;
        let backend = AllocCountingMockBackend::new();
        let weights = mock_weights(&cfg);
        let engine = Engine::new(backend, weights, 0..num_layers);
        let mut cache = KvCacheManager::new(
            num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len,
        );
        let handle = cache.alloc(engine.backend()).unwrap();

        engine.forward(&[1], &[0], &mut cache, handle, None).unwrap();
        engine.backend().alloc_count()
    }

    let allocs_2_layers = run_forward_and_count_allocs(2);
    let allocs_4_layers = run_forward_and_count_allocs(4);

    // The difference in allocs between 4-layer and 2-layer should come only from
    // KV cache allocation (2 allocs per extra layer: K cache + V cache).
    // KV cache allocates 2 tensors per layer, so delta for 2 extra layers = 4.
    let delta = allocs_4_layers - allocs_2_layers;
    let expected_cache_delta = 2 * 2; // 2 extra layers * 2 (K + V) per layer

    assert_eq!(
        delta, expected_cache_delta,
        "alloc count should only scale with KV cache tensors (2 per layer), \
         not scratch tensors. 2-layer allocs: {}, 4-layer allocs: {}, delta: {} \
         (expected {} for cache-only scaling)",
        allocs_2_layers, allocs_4_layers, delta, expected_cache_delta
    );

    // Sanity check: scratch tensors should be a fixed count.
    // For a full (head+tail) forward pass with seq_len=1:
    // - 1 embedding (hidden_state)
    // - 10 scratch tensors (normed, q_flat, k_flat, v_flat, attn_out_mh,
    //   projected, gate, up, ffn_mid, ffn_out)
    // - 1 logits_tensor
    // Total fixed allocs = 12
    // Per-layer allocs = 2 (K cache + V cache) from KvCacheManager::alloc
    // So 2-layer total = 12 + 2*2 = 16
    let expected_2_layer = 12 + 2 * 2;
    assert_eq!(
        allocs_2_layers, expected_2_layer,
        "2-layer alloc count mismatch: expected {} (12 fixed + 4 cache), got {}",
        expected_2_layer, allocs_2_layers
    );
}

// ── Paged forward tests ─────────────────────────────────────

/// Gap 90: forward_paged() with multiple tokens returns logits with correct vocab_size.
#[test]
fn test_forward_paged_prefill_returns_correct_vocab_size() {
    let mut cfg = test_config();
    cfg.num_layers = 2;
    let backend = MockBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..2);
    let mut cache = PagedKvCacheManager::new(8, 2, cfg.num_kv_heads, cfg.head_dim, engine.backend()).unwrap();
    let handle = cache.alloc().unwrap();

    let token_ids = vec![1, 2, 3, 4];
    let positions = vec![0, 1, 2, 3];
    let logits = engine.forward_paged(&token_ids, &positions, &mut cache, handle).unwrap();
    assert_eq!(
        logits.len(),
        cfg.vocab_size,
        "forward_paged prefill should return vocab_size logits, got {}",
        logits.len()
    );
}

/// Gap 91: forward_paged() decode after prefill works and cache.seq_len advances.
#[test]
fn test_forward_paged_decode_after_prefill_advances_seq_len() {
    let mut cfg = test_config();
    cfg.num_layers = 2;
    let backend = MockBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..2);
    let mut cache = PagedKvCacheManager::new(8, 2, cfg.num_kv_heads, cfg.head_dim, engine.backend()).unwrap();
    let handle = cache.alloc().unwrap();

    // Prefill with 3 tokens
    let prefill_ids = vec![10, 20, 30];
    let prefill_pos = vec![0, 1, 2];
    engine.forward_paged(&prefill_ids, &prefill_pos, &mut cache, handle).unwrap();
    assert_eq!(cache.seq_len(handle).unwrap(), 3, "seq_len should be 3 after prefill");

    // Decode with 1 token
    let decode_ids = vec![40];
    let decode_pos = vec![3];
    let logits = engine.forward_paged(&decode_ids, &decode_pos, &mut cache, handle).unwrap();
    assert_eq!(logits.len(), cfg.vocab_size, "decode should return vocab_size logits");
    assert_eq!(cache.seq_len(handle).unwrap(), 4, "seq_len should be 4 after decode");
}

/// Gap 95: forward_node_paged() head node returns Activations, tail node returns Logits.
#[test]
fn test_forward_node_paged_head_tail_outputs() {
    let mut cfg = test_config();
    cfg.num_layers = 2;
    let backend = MockBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..2);
    let mut cache = PagedKvCacheManager::new(8, 2, cfg.num_kv_heads, cfg.head_dim, engine.backend()).unwrap();

    // Head node (layer 0..1 of 2): should return Activations
    let head_config = NodeConfig::new(0..1, 2).unwrap();
    assert!(head_config.is_head());
    assert!(!head_config.is_tail());

    let handle_head = cache.alloc().unwrap();
    let input_head = NodeInput::TokenIds {
        ids: vec![1],
        positions: vec![0],
    };
    let result = engine.forward_node_paged(input_head, &head_config, &mut cache, handle_head);
    assert!(result.is_ok(), "head node forward_node_paged should succeed: {:?}", result.err());
    match result.unwrap() {
        NodeOutput::Activations(tensor) => {
            assert_eq!(tensor.shape, vec![1, cfg.hidden_size]);
        }
        NodeOutput::Logits(_) => panic!("head node should return Activations, not Logits"),
    }

    // Tail node (layer 1..2 of 2): should return Logits
    let tail_config = NodeConfig::new(1..2, 2).unwrap();
    assert!(!tail_config.is_head());
    assert!(tail_config.is_tail());

    let handle_tail = cache.alloc().unwrap();
    let fake_hidden = engine.backend().alloc(&[1, cfg.hidden_size], DType::FP16).unwrap();
    let input_tail = NodeInput::Activations {
        hidden_states: fake_hidden,
        positions: vec![0],
    };
    let result = engine.forward_node_paged(input_tail, &tail_config, &mut cache, handle_tail);
    assert!(result.is_ok(), "tail node forward_node_paged should succeed: {:?}", result.err());
    match result.unwrap() {
        NodeOutput::Logits(logits) => {
            assert_eq!(logits.len(), cfg.vocab_size);
        }
        NodeOutput::Activations(_) => panic!("tail node should return Logits, not Activations"),
    }
}

/// Gap 96: forward_node_paged() error paths — wrong input type for node role.
#[test]
fn test_forward_node_paged_error_paths() {
    let mut cfg = test_config();
    cfg.num_layers = 2;
    let backend = MockBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..2);
    let mut cache = PagedKvCacheManager::new(8, 2, cfg.num_kv_heads, cfg.head_dim, engine.backend()).unwrap();

    // Non-head (tail) receiving TokenIds should error
    let tail_config = NodeConfig::new(1..2, 2).unwrap();
    assert!(!tail_config.is_head());

    let handle1 = cache.alloc().unwrap();
    let input_tokens = NodeInput::TokenIds {
        ids: vec![1],
        positions: vec![0],
    };
    let result = engine.forward_node_paged(input_tokens, &tail_config, &mut cache, handle1);
    assert!(result.is_err(), "non-head receiving TokenIds should error");
    assert!(
        result.err().unwrap().to_string().contains("non-head"),
        "error should mention non-head"
    );

    // Head receiving Activations should error
    let head_config = NodeConfig::new(0..1, 2).unwrap();
    assert!(head_config.is_head());

    let handle2 = cache.alloc().unwrap();
    let fake_hidden = engine.backend().alloc(&[1, cfg.hidden_size], DType::FP16).unwrap();
    let input_act = NodeInput::Activations {
        hidden_states: fake_hidden,
        positions: vec![0],
    };
    let result = engine.forward_node_paged(input_act, &head_config, &mut cache, handle2);
    assert!(result.is_err(), "head receiving Activations should error");
    assert!(
        result.err().unwrap().to_string().contains("head"),
        "error should mention head"
    );
}

// ── OpSequenceBackend ────────────────────────────────────────

/// Records every backend compute operation in call order, tagging by name.
struct OpSequenceBackend {
    next_id: AtomicU64,
    ops: Mutex<Vec<&'static str>>,
    /// Track freed tensor IDs
    freed: Mutex<Vec<u64>>,
}

impl OpSequenceBackend {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1000),
            ops: Mutex::new(Vec::new()),
            freed: Mutex::new(Vec::new()),
        }
    }

    fn record(&self, name: &'static str) {
        self.ops.lock().unwrap().push(name);
    }
}

impl Backend for OpSequenceBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }
    fn free(&self, tensor: &DeviceTensor) -> Result<()> {
        self.freed.lock().unwrap().push(tensor.id.0);
        Ok(())
    }
    fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> { Ok(()) }
    fn copy_to_host(&self, _src: &DeviceTensor, dst: &mut [u8]) -> Result<()> {
        dst.fill(0);
        Ok(())
    }
    fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> {
        self.record("matmul");
        Ok(())
    }
    fn rmsnorm(&self, _i: &DeviceTensor, _w: &DeviceTensor, _e: f64, _o: &DeviceTensor) -> Result<()> {
        self.record("rmsnorm");
        Ok(())
    }
    fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _p: &[u32], _t: f64, _h: usize) -> Result<()> {
        self.record("rope");
        Ok(())
    }
    fn attention(&self, _q: &DeviceTensor, _k: &DeviceTensor, _v: &DeviceTensor, _n: usize, _s: usize, _o: &DeviceTensor) -> Result<()> {
        self.record("attention");
        Ok(())
    }
    fn attention_paged(&self, _q: &DeviceTensor, _bt: &[i32], _kb: &[&DeviceTensor], _vb: &[&DeviceTensor], _nkv: usize, _kvl: usize, _sp: usize, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn silu_mul(&self, _g: &DeviceTensor, _u: &DeviceTensor, _o: &DeviceTensor) -> Result<()> {
        self.record("silu_mul");
        Ok(())
    }
    fn embedding(&self, _t: &[u32], _e: &DeviceTensor, _o: &DeviceTensor) -> Result<()> {
        self.record("embedding");
        Ok(())
    }
    fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _o: &DeviceTensor) -> Result<()> {
        self.record("add");
        Ok(())
    }
    fn copy_rows(&self, _s: &DeviceTensor, _d: &DeviceTensor, _so: usize, _do_: usize, _c: usize) -> Result<()> {
        self.record("copy_rows");
        Ok(())
    }
    fn device_name(&self) -> &str { "op-seq-mock" }
    fn total_memory(&self) -> usize { 1_000_000_000 }
    fn available_memory(&self) -> usize { 1_000_000_000 }
    fn synchronize(&self) -> Result<()> { Ok(()) }
    fn create_timer(&self) -> Result<DeviceTimer> { Ok(DeviceTimer(0)) }
    fn start_timer(&self, _t: &DeviceTimer) -> Result<()> { Ok(()) }
    fn stop_timer(&self, _t: &DeviceTimer) -> Result<f32> { Ok(0.0) }
    fn destroy_timer(&self, _t: &DeviceTimer) -> Result<()> { Ok(()) }
}

/// Verify that a single-layer forward pass calls backend operations in the
/// correct order: rmsnorm, matmul(Q), matmul(K), matmul(V), rope,
/// copy_rows(K), copy_rows(V), attention, matmul(out_proj), add(residual),
/// rmsnorm, matmul(gate), matmul(up), silu_mul, matmul(down), add(residual).
///
/// Uses a full (head+tail) node with 1 layer to isolate exactly one layer's
/// worth of operations. The embedding and final norm+lm_head ops are also
/// verified to appear in the correct relative positions.
#[test]
fn test_single_layer_operation_sequence() {
    let cfg = test_config(); // num_layers=1
    let backend = OpSequenceBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..cfg.num_layers);
    let mut cache = KvCacheManager::new(
        cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len,
    );
    let handle = cache.alloc(engine.backend()).unwrap();

    let node_config = NodeConfig::new(0..cfg.num_layers, cfg.num_layers).unwrap();
    assert!(node_config.is_full());

    let input = NodeInput::TokenIds {
        ids: vec![1],
        positions: vec![0],
    };
    let result = engine.forward_node(input, &node_config, &mut cache, handle, None);
    assert!(result.is_ok(), "forward_node should succeed: {:?}", result.err());

    let ops = engine.backend().ops.lock().unwrap().clone();

    // The sequence for a full (head+tail) 1-layer forward pass:
    //   embedding
    //   [layer 0]:
    //     rmsnorm (attn norm)
    //     matmul (Q proj)
    //     matmul (K proj)
    //     matmul (V proj)
    //     rope
    //     copy_rows (K to cache)
    //     copy_rows (V to cache)
    //     attention
    //     matmul (out proj)
    //     add (attn residual)
    //     rmsnorm (ffn norm)
    //     matmul (gate proj)
    //     matmul (up proj)
    //     silu_mul
    //     matmul (down proj)
    //     add (ffn residual)
    //   [tail]:
    //     rmsnorm (output norm)
    //     copy_rows (extract last position) — only for seq_len > 1; skipped for seq_len=1
    //     matmul (lm_head)

    // Verify the per-layer subsequence appears in order.
    let layer_ops: Vec<&str> = vec![
        "rmsnorm",    // attn norm
        "matmul",     // Q proj
        "matmul",     // K proj
        "matmul",     // V proj
        "rope",
        "copy_rows",  // K to cache
        "copy_rows",  // V to cache
        "attention",
        "matmul",     // out proj
        "add",        // attn residual
        "rmsnorm",    // ffn norm
        "matmul",     // gate proj
        "matmul",     // up proj
        "silu_mul",
        "matmul",     // down proj
        "add",        // ffn residual
    ];

    // Find embedding first, then verify the layer ops follow in order.
    let embed_pos = ops.iter().position(|&op| op == "embedding")
        .expect("embedding should be called");

    // Find the start of layer ops after embedding.
    let mut search_from = embed_pos + 1;
    for (i, &expected_op) in layer_ops.iter().enumerate() {
        let found = ops[search_from..].iter().position(|&op| op == expected_op)
            .unwrap_or_else(|| panic!(
                "layer op #{i} ('{expected_op}') not found at or after position {search_from} in ops: {ops:?}"
            ));
        search_from = search_from + found + 1;
    }

    // Verify final norm and lm_head matmul appear after the layer ops.
    let tail_rmsnorm = ops[search_from.saturating_sub(1)..].iter().position(|&op| op == "rmsnorm");
    assert!(
        tail_rmsnorm.is_some() || ops[search_from..].contains(&"rmsnorm"),
        "output rmsnorm should appear after layer ops. ops so far: {ops:?}"
    );

    // Verify counts match expectations:
    // rmsnorm: 2 per layer (attn + ffn) + 1 output norm = 3 total
    let rmsnorm_count = ops.iter().filter(|&&op| op == "rmsnorm").count();
    assert_eq!(
        rmsnorm_count, 3,
        "expected 3 rmsnorm calls (attn_norm + ffn_norm + output_norm), got {rmsnorm_count}. ops: {ops:?}"
    );

    // matmul: Q+K+V+out_proj+gate+up+down+lm_head = 8 per layer (seq_len=1 so no copy_rows for last_hidden)
    let matmul_count = ops.iter().filter(|&&op| op == "matmul").count();
    assert_eq!(
        matmul_count, 8,
        "expected 8 matmul calls (Q,K,V,out_proj,gate,up,down,lm_head), got {matmul_count}. ops: {ops:?}"
    );

    // copy_rows: 2 per layer (K + V to cache); no extra copy_rows for last_hidden when seq_len=1
    let copy_rows_count = ops.iter().filter(|&&op| op == "copy_rows").count();
    assert_eq!(
        copy_rows_count, 2,
        "expected 2 copy_rows calls (K cache + V cache) for seq_len=1, got {copy_rows_count}. ops: {ops:?}"
    );

    // add: 2 per layer (attn residual + ffn residual)
    let add_count = ops.iter().filter(|&&op| op == "add").count();
    assert_eq!(
        add_count, 2,
        "expected 2 add calls per layer, got {add_count}. ops: {ops:?}"
    );

    // rope: 1 per layer
    let rope_count = ops.iter().filter(|&&op| op == "rope").count();
    assert_eq!(rope_count, 1, "expected 1 rope call, got {rope_count}. ops: {ops:?}");

    // attention: 1 per layer
    let attention_count = ops.iter().filter(|&&op| op == "attention").count();
    assert_eq!(attention_count, 1, "expected 1 attention call, got {attention_count}. ops: {ops:?}");

    // silu_mul: 1 per layer
    let silu_count = ops.iter().filter(|&&op| op == "silu_mul").count();
    assert_eq!(silu_count, 1, "expected 1 silu_mul call, got {silu_count}. ops: {ops:?}");
}

// ── FreeTrackingBackend ──────────────────────────────────────

/// Records alloc and free calls to verify ownership semantics.
struct FreeTrackingBackend {
    next_id: AtomicU64,
    /// IDs allocated during forward_node (excluding those pre-allocated before the call)
    allocated: Mutex<Vec<u64>>,
    freed: Mutex<Vec<u64>>,
}

impl FreeTrackingBackend {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1000),
            allocated: Mutex::new(Vec::new()),
            freed: Mutex::new(Vec::new()),
        }
    }
}

impl Backend for FreeTrackingBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.allocated.lock().unwrap().push(id);
        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }
    fn free(&self, tensor: &DeviceTensor) -> Result<()> {
        self.freed.lock().unwrap().push(tensor.id.0);
        Ok(())
    }
    fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> { Ok(()) }
    fn copy_to_host(&self, _src: &DeviceTensor, dst: &mut [u8]) -> Result<()> {
        dst.fill(0);
        Ok(())
    }
    fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _o: &DeviceTensor) -> Result<()> { Ok(()) }
    fn rmsnorm(&self, _i: &DeviceTensor, _w: &DeviceTensor, _e: f64, _o: &DeviceTensor) -> Result<()> { Ok(()) }
    fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _p: &[u32], _t: f64, _h: usize) -> Result<()> { Ok(()) }
    fn attention(&self, _q: &DeviceTensor, _k: &DeviceTensor, _v: &DeviceTensor, _n: usize, _s: usize, _o: &DeviceTensor) -> Result<()> { Ok(()) }
    fn attention_paged(&self, _q: &DeviceTensor, _bt: &[i32], _kb: &[&DeviceTensor], _vb: &[&DeviceTensor], _nkv: usize, _kvl: usize, _sp: usize, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn silu_mul(&self, _g: &DeviceTensor, _u: &DeviceTensor, _o: &DeviceTensor) -> Result<()> { Ok(()) }
    fn embedding(&self, _t: &[u32], _e: &DeviceTensor, _o: &DeviceTensor) -> Result<()> { Ok(()) }
    fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _o: &DeviceTensor) -> Result<()> { Ok(()) }
    fn copy_rows(&self, _s: &DeviceTensor, _d: &DeviceTensor, _so: usize, _do_: usize, _c: usize) -> Result<()> { Ok(()) }
    fn device_name(&self) -> &str { "free-tracking-mock" }
    fn total_memory(&self) -> usize { 1_000_000_000 }
    fn available_memory(&self) -> usize { 1_000_000_000 }
    fn synchronize(&self) -> Result<()> { Ok(()) }
    fn create_timer(&self) -> Result<DeviceTimer> { Ok(DeviceTimer(0)) }
    fn start_timer(&self, _t: &DeviceTimer) -> Result<()> { Ok(()) }
    fn stop_timer(&self, _t: &DeviceTimer) -> Result<f32> { Ok(0.0) }
    fn destroy_timer(&self, _t: &DeviceTimer) -> Result<()> { Ok(()) }
}

/// Verify that when a head node (non-tail) returns Activations, the returned
/// tensor is NOT passed to free(), but all scratch tensors ARE freed.
///
/// The spec says: "Scratch tensors are freed; the output hidden state is NOT
/// freed (ownership transfers to the caller)."
#[test]
fn test_activation_handoff_no_free() {
    let mut cfg = test_config();
    cfg.num_layers = 2; // 2-layer model so head node is not tail
    let backend = FreeTrackingBackend::new();
    let weights = mock_weights(&cfg);

    // Engine holds layers 0..2; node config is 0..1 (head, not tail)
    let engine = Engine::new(backend, weights, 0..2);

    // Cache sized for 1 layer (the head node's layer range)
    let mut cache = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(engine.backend()).unwrap();

    let node_config = NodeConfig::new(0..1, 2).unwrap();
    assert!(node_config.is_head(), "node_config should be head");
    assert!(!node_config.is_tail(), "node_config should not be tail");

    // Clear alloc list so we only track what forward_node allocates.
    engine.backend().allocated.lock().unwrap().clear();

    let input = NodeInput::TokenIds {
        ids: vec![1],
        positions: vec![0],
    };
    let result = engine.forward_node(input, &node_config, &mut cache, handle, None);
    assert!(result.is_ok(), "head forward_node should succeed: {:?}", result.err());

    let activation_tensor = match result.unwrap() {
        NodeOutput::Activations(t) => t,
        NodeOutput::Logits(_) => panic!("head node should return Activations, not Logits"),
    };

    let activation_id = activation_tensor.id.0;
    let freed = engine.backend().freed.lock().unwrap().clone();
    let allocated = engine.backend().allocated.lock().unwrap().clone();

    // The activation tensor must NOT have been freed.
    assert!(
        !freed.contains(&activation_id),
        "activation tensor (id={activation_id}) should NOT be freed, but it was. freed: {freed:?}"
    );

    // The activation tensor must have been allocated during this call.
    assert!(
        allocated.contains(&activation_id),
        "activation tensor (id={activation_id}) should have been allocated by forward_node. allocated: {allocated:?}"
    );

    // Every other tensor allocated during the call (scratch tensors) must have been freed.
    // Scratch tensors are: normed, q_flat, k_flat, v_flat, attn_out_mh, projected,
    // gate, up, ffn_mid, ffn_out — all 10 of them.
    let scratch_ids: Vec<u64> = allocated.iter()
        .copied()
        .filter(|&id| id != activation_id)
        .collect();

    for &scratch_id in &scratch_ids {
        assert!(
            freed.contains(&scratch_id),
            "scratch tensor (id={scratch_id}) should have been freed, but it was not. freed: {freed:?}, scratch_ids: {scratch_ids:?}"
        );
    }

    // There should be exactly 10 scratch tensors for a head-only (non-tail) forward:
    // normed, q_flat, k_flat, v_flat, attn_out_mh, projected, gate, up, ffn_mid, ffn_out
    assert_eq!(
        scratch_ids.len(), 10,
        "expected exactly 10 scratch tensors (head node, non-tail), got {}. allocated: {allocated:?}",
        scratch_ids.len()
    );
}

// ── Position bounds validation ────────────────────────────────

/// Verify that forward() with a position value >= max_seq_len returns
/// FractureError::InvalidShape containing "exceeds max_seq_len".
#[test]
fn test_forward_position_beyond_max_seq_len() {
    let mut cfg = test_config();
    cfg.max_seq_len = 128;
    let backend = MockBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..cfg.num_layers);
    let mut cache = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(engine.backend()).unwrap();

    // Position 128 equals max_seq_len — must be rejected (valid range is 0..max_seq_len).
    let result = engine.forward(&[1], &[128], &mut cache, handle, None);
    let err = result.expect_err("forward with out-of-bounds position should return Err");
    assert!(
        matches!(err, FractureError::InvalidShape(_)),
        "expected InvalidShape, got: {err:?}"
    );
    assert!(
        err.to_string().contains("exceeds max_seq_len"),
        "error should mention 'exceeds max_seq_len', got: {err}"
    );
}

// ── Middle node forward ───────────────────────────────────────

/// Verify that a middle (non-head, non-tail) node accepts Activations input
/// and returns Activations output with the correct hidden-state shape.
#[test]
fn test_middle_node_forward() {
    let mut cfg = test_config();
    cfg.num_layers = 4;
    let backend = MockBackend::new();

    // Build weights for layers 1..3 only (middle node owns those layers).
    let h = cfg.hidden_size;
    let kv = cfg.num_kv_heads * cfg.head_dim;
    let inter = cfg.intermediate_size;
    let mut id = 100u64;
    let mut t = |shape: Vec<usize>| -> DeviceTensor {
        let tensor = mock_tensor(id, shape);
        id += 1;
        tensor
    };
    let middle_layers: Vec<LayerWeights> = (0..2)
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
    // token_embedding, output_norm, lm_head are unused by a middle node but required
    // by WeightStore. They can be dummy tensors.
    let weights = WeightStore {
        config: cfg.clone(),
        token_embedding: t(vec![cfg.vocab_size, h]),
        layers: middle_layers,
        output_norm: t(vec![h]),
        lm_head: t(vec![cfg.vocab_size, h]),
    };

    // Engine owns layer range 1..3 (middle), total model has 4 layers.
    let engine = Engine::new(backend, weights, 1..3);

    // Middle node config: layers 1..3 of a 4-layer model.
    let node_config = NodeConfig::new(1..3, 4).unwrap();
    assert!(!node_config.is_head(), "1..3 of 4 should not be head");
    assert!(!node_config.is_tail(), "1..3 of 4 should not be tail");

    // KV cache sized for 2 layers (the node's layer range).
    let mut cache = KvCacheManager::new(2, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(engine.backend()).unwrap();

    // Provide fake hidden state activations from the head node.
    let fake_hidden = engine.backend().alloc(&[1, cfg.hidden_size], DType::FP16).unwrap();
    let input = NodeInput::Activations {
        hidden_states: fake_hidden,
        positions: vec![0],
    };

    let result = engine.forward_node(input, &node_config, &mut cache, handle, None);
    assert!(result.is_ok(), "middle node forward should succeed: {:?}", result.err());

    match result.unwrap() {
        NodeOutput::Activations(tensor) => {
            assert_eq!(
                tensor.shape,
                vec![1, cfg.hidden_size],
                "middle node output should be [seq_len, hidden_size]"
            );
        }
        NodeOutput::Logits(_) => panic!("middle node should return Activations, not Logits"),
    }
}

// ── forward() delegates to forward_node() ────────────────────

/// Verify that forward() and forward_node() with a full-range NodeConfig produce
/// identical-length logit vectors for the same input.
#[test]
fn test_forward_delegates_to_forward_node() {
    let cfg = test_config();
    let backend = MockBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..cfg.num_layers);

    // Two independent caches and handles — one for each call.
    let mut cache_a = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle_a = cache_a.alloc(engine.backend()).unwrap();
    let mut cache_b = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle_b = cache_b.alloc(engine.backend()).unwrap();

    // Call forward() (the wrapper).
    let logits_a = engine.forward(&[5], &[0], &mut cache_a, handle_a, None)
        .expect("forward() should succeed");

    // Call forward_node() with a full-range config.
    let node_config = NodeConfig::new(0..cfg.num_layers, cfg.num_layers).unwrap();
    assert!(node_config.is_full());
    let input = NodeInput::TokenIds { ids: vec![5], positions: vec![0] };
    let logits_b = match engine.forward_node(input, &node_config, &mut cache_b, handle_b, None)
        .expect("forward_node() should succeed")
    {
        NodeOutput::Logits(l) => l,
        NodeOutput::Activations(_) => panic!("full node should return Logits"),
    };

    // Both should produce exactly vocab_size logits.
    assert_eq!(
        logits_a.len(), cfg.vocab_size,
        "forward() should return vocab_size logits"
    );
    assert_eq!(
        logits_b.len(), cfg.vocab_size,
        "forward_node() should return vocab_size logits"
    );
    assert_eq!(
        logits_a.len(), logits_b.len(),
        "forward() and forward_node() must produce the same number of logits"
    );
}

// ── weight_idx vs cache_idx divergence ───────────────────────

/// Verify that an engine with layer_range 2..4 correctly maps weight indices
/// (relative to the engine's owned layers) and cache indices (relative to the
/// node config's exec range) without panicking or index-out-of-bounds errors.
///
/// The engine stores weights for layers 2..4 at positions [0, 1] in
/// self.weights.layers.  The node config 2..4 maps weight_idx = layer_idx - 2
/// and cache_idx = layer_idx - 2 (both start at 2), so both indices are 0 and 1.
#[test]
fn test_weight_idx_vs_cache_idx_diverge() {
    let mut cfg = test_config();
    cfg.num_layers = 4;
    let backend = MockBackend::new();

    // Build 2-layer weights representing model layers 2 and 3.
    let h = cfg.hidden_size;
    let kv = cfg.num_kv_heads * cfg.head_dim;
    let inter = cfg.intermediate_size;
    let mut id = 200u64;
    let mut t = |shape: Vec<usize>| -> DeviceTensor {
        let tensor = mock_tensor(id, shape);
        id += 1;
        tensor
    };
    let tail_layers: Vec<LayerWeights> = (0..2)
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
    let weights = WeightStore {
        config: cfg.clone(),
        token_embedding: t(vec![cfg.vocab_size, h]),
        layers: tail_layers,
        output_norm: t(vec![h]),
        lm_head: t(vec![cfg.vocab_size, h]),
    };

    // Engine owns layer range 2..4 (tail of a 4-layer model).
    let engine = Engine::new(backend, weights, 2..4);

    // Tail node config: layers 2..4 of a 4-layer model.
    let node_config = NodeConfig::new(2..4, 4).unwrap();
    assert!(!node_config.is_head(), "2..4 of 4 should not be head");
    assert!(node_config.is_tail(), "2..4 of 4 should be tail");

    // KV cache for 2 layers.
    let mut cache = KvCacheManager::new(2, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let handle = cache.alloc(engine.backend()).unwrap();

    // Provide activations (tail node receives hidden states from head/middle).
    let fake_hidden = engine.backend().alloc(&[1, cfg.hidden_size], DType::FP16).unwrap();
    let input = NodeInput::Activations {
        hidden_states: fake_hidden,
        positions: vec![0],
    };

    let result = engine.forward_node(input, &node_config, &mut cache, handle, None);
    assert!(
        result.is_ok(),
        "tail node with non-zero layer_range should succeed — weight_idx and cache_idx are \
         both offset by the same amount (2): {:?}",
        result.err()
    );
    match result.unwrap() {
        NodeOutput::Logits(logits) => {
            assert_eq!(logits.len(), cfg.vocab_size, "tail node should produce vocab_size logits");
        }
        NodeOutput::Activations(_) => panic!("tail node should return Logits"),
    }
}

// ── Paged forward: backend error propagation ──────────────────

/// Verify that a backend error during forward_paged() propagates as FractureError::Backend.
#[test]
fn test_forward_paged_backend_error_propagation() {
    let mut cfg = test_config();
    cfg.num_layers = 1;
    let backend = MockBackend::failing_matmul();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..cfg.num_layers);
    let mut cache = PagedKvCacheManager::new(
        8, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, engine.backend(),
    ).unwrap();
    let handle = cache.alloc().unwrap();

    let result = engine.forward_paged(&[1], &[0], &mut cache, handle);
    let err = result.expect_err("forward_paged with failing matmul should return Err");
    assert!(
        matches!(err, FractureError::Backend(_)),
        "expected FractureError::Backend, got: {err:?}"
    );
    assert!(
        err.to_string().contains("mock matmul failure"),
        "error should contain 'mock matmul failure', got: {err}"
    );
}

// ── Paged forward: attention_paged call count ─────────────────

/// A recording backend that counts attention_paged calls.
struct AttentionPagedCountingBackend {
    next_id: AtomicU64,
    attention_paged_count: AtomicU64,
}

impl AttentionPagedCountingBackend {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1000),
            attention_paged_count: AtomicU64::new(0),
        }
    }
}

impl Backend for AttentionPagedCountingBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }
    fn free(&self, _tensor: &DeviceTensor) -> Result<()> { Ok(()) }
    fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> { Ok(()) }
    fn copy_to_host(&self, _src: &DeviceTensor, dst: &mut [u8]) -> Result<()> {
        dst.fill(0);
        Ok(())
    }
    fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
    fn rmsnorm(&self, _i: &DeviceTensor, _w: &DeviceTensor, _e: f64, _o: &DeviceTensor) -> Result<()> { Ok(()) }
    fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _p: &[u32], _t: f64, _h: usize) -> Result<()> { Ok(()) }
    fn attention(&self, _q: &DeviceTensor, _k: &DeviceTensor, _v: &DeviceTensor, _n: usize, _s: usize, _o: &DeviceTensor) -> Result<()> { Ok(()) }
    fn attention_paged(
        &self,
        _q: &DeviceTensor,
        _bt: &[i32],
        _kb: &[&DeviceTensor],
        _vb: &[&DeviceTensor],
        _nkv: usize,
        _kvl: usize,
        _sp: usize,
        _out: &DeviceTensor,
    ) -> Result<()> {
        self.attention_paged_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn silu_mul(&self, _g: &DeviceTensor, _u: &DeviceTensor, _o: &DeviceTensor) -> Result<()> { Ok(()) }
    fn embedding(&self, _t: &[u32], _e: &DeviceTensor, _o: &DeviceTensor) -> Result<()> { Ok(()) }
    fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _o: &DeviceTensor) -> Result<()> { Ok(()) }
    fn copy_rows(&self, _s: &DeviceTensor, _d: &DeviceTensor, _so: usize, _do_: usize, _c: usize) -> Result<()> { Ok(()) }
    fn device_name(&self) -> &str { "attn-paged-counting-mock" }
    fn total_memory(&self) -> usize { 1_000_000_000 }
    fn available_memory(&self) -> usize { 1_000_000_000 }
    fn synchronize(&self) -> Result<()> { Ok(()) }
    fn create_timer(&self) -> Result<DeviceTimer> { Ok(DeviceTimer(0)) }
    fn start_timer(&self, _t: &DeviceTimer) -> Result<()> { Ok(()) }
    fn stop_timer(&self, _t: &DeviceTimer) -> Result<f32> { Ok(0.0) }
    fn destroy_timer(&self, _t: &DeviceTimer) -> Result<()> { Ok(()) }
}

/// Verify that forward_paged() calls attention_paged exactly once per layer
/// during a prefill pass.
#[test]
fn test_forward_paged_prefill_calls_attention_paged() {
    let mut cfg = test_config();
    cfg.num_layers = 2; // Use 2 layers to verify the per-layer count.
    let backend = AttentionPagedCountingBackend::new();
    let weights = mock_weights(&cfg);
    let engine = Engine::new(backend, weights, 0..cfg.num_layers);
    let mut cache = PagedKvCacheManager::new(
        8, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, engine.backend(),
    ).unwrap();
    let handle = cache.alloc().unwrap();

    // Single-token prefill; attention_paged should be called once per layer.
    let result = engine.forward_paged(&[1], &[0], &mut cache, handle);
    assert!(result.is_ok(), "forward_paged should succeed: {:?}", result.err());

    let paged_count = engine.backend().attention_paged_count.load(Ordering::Relaxed);
    assert_eq!(
        paged_count,
        cfg.num_layers as u64,
        "attention_paged should be called exactly once per layer ({} layers), got {paged_count}",
        cfg.num_layers
    );
}

// ── KvCacheBackend enum tests ─────────────────────────────────

/// Construct a Contiguous KvCacheBackend, verify alloc/seq_len/free work
/// and is_paged() returns false.
#[test]
fn test_kv_cache_backend_contiguous() {
    let cfg = test_config();
    let backend = MockBackend::new();

    let mut kv_backend = KvCacheBackend::Contiguous(KvCacheManager::new(
        cfg.num_layers,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.max_seq_len,
    ));

    assert!(!kv_backend.is_paged(), "Contiguous variant should return false for is_paged()");

    // alloc_contiguous should succeed.
    let handle = kv_backend.alloc_contiguous(&backend)
        .expect("alloc_contiguous should succeed on Contiguous variant");

    // seq_len should be 0 after alloc.
    let len = kv_backend.seq_len(handle)
        .expect("seq_len should succeed on Contiguous variant");
    assert_eq!(len, 0, "seq_len should be 0 after alloc");

    // free should succeed.
    kv_backend.free(handle, &backend)
        .expect("free should succeed on Contiguous variant");
}

/// Construct a Paged KvCacheBackend, verify alloc/seq_len/free work
/// and is_paged() returns true.
#[test]
fn test_kv_cache_backend_paged() {
    let cfg = test_config();
    let backend = MockBackend::new();

    let mut kv_backend = KvCacheBackend::Paged(
        PagedKvCacheManager::new(
            10,
            cfg.num_layers,
            cfg.num_kv_heads,
            cfg.head_dim,
            &backend,
        ).expect("PagedKvCacheManager::new should succeed"),
    );

    assert!(kv_backend.is_paged(), "Paged variant should return true for is_paged()");

    // alloc_paged should succeed.
    let handle = kv_backend.alloc_paged()
        .expect("alloc_paged should succeed on Paged variant");

    // seq_len should be 0 after alloc.
    let len = kv_backend.seq_len(handle)
        .expect("seq_len should succeed on Paged variant");
    assert_eq!(len, 0, "seq_len should be 0 after alloc");

    // free should succeed.
    kv_backend.free(handle, &backend)
        .expect("free should succeed on Paged variant");
}

/// Cross-variant error: alloc_contiguous on Paged returns error,
/// alloc_paged on Contiguous returns error.
#[test]
fn test_kv_cache_backend_cross_variant_errors() {
    let cfg = test_config();
    let backend = MockBackend::new();

    // Paged variant rejects alloc_contiguous.
    let mut paged = KvCacheBackend::Paged(
        PagedKvCacheManager::new(
            10,
            cfg.num_layers,
            cfg.num_kv_heads,
            cfg.head_dim,
            &backend,
        ).unwrap(),
    );
    let err = paged.alloc_contiguous(&backend).unwrap_err();
    assert!(
        matches!(err, FractureError::KvCache(_)),
        "alloc_contiguous on Paged should return KvCache error, got: {err:?}"
    );
    assert!(
        err.to_string().contains("alloc_contiguous"),
        "error should mention 'alloc_contiguous', got: {err}"
    );

    // Contiguous variant rejects alloc_paged.
    let mut contiguous = KvCacheBackend::Contiguous(KvCacheManager::new(
        cfg.num_layers,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.max_seq_len,
    ));
    let err = contiguous.alloc_paged().unwrap_err();
    assert!(
        matches!(err, FractureError::KvCache(_)),
        "alloc_paged on Contiguous should return KvCache error, got: {err:?}"
    );
    assert!(
        err.to_string().contains("alloc_paged"),
        "error should mention 'alloc_paged', got: {err}"
    );
}

/// Construct a QuantizedPaged KvCacheBackend, verify alloc/seq_len/free work
/// and is_paged()/is_quantized() return correct values.
#[test]
fn test_kv_cache_backend_quantized_paged() {
    let cfg = test_config();
    let backend = MockBackend::new();
    let tq_config = fracture_core::TurboQuantConfig::default();

    let mut kv_backend = KvCacheBackend::QuantizedPaged(
        crate::quantized_paged_kv_cache::QuantizedKvCacheManager::new(
            10,
            cfg.num_layers,
            cfg.num_kv_heads,
            cfg.head_dim,
            16,
            tq_config,
            &backend,
        )
        .expect("QuantizedKvCacheManager::new should succeed"),
    );

    assert!(
        kv_backend.is_paged(),
        "QuantizedPaged should return true for is_paged()"
    );
    assert!(
        kv_backend.is_quantized(),
        "QuantizedPaged should return true for is_quantized()"
    );

    // alloc_paged should succeed.
    let handle = kv_backend
        .alloc_paged()
        .expect("alloc_paged should succeed on QuantizedPaged variant");

    // Generic alloc should also succeed.
    let handle2 = kv_backend
        .alloc(&backend)
        .expect("alloc should succeed on QuantizedPaged variant");

    // seq_len should be 0 after alloc.
    let len = kv_backend
        .seq_len(handle)
        .expect("seq_len should succeed");
    assert_eq!(len, 0);

    // free should succeed.
    kv_backend
        .free(handle, &backend)
        .expect("free should succeed");
    kv_backend
        .free(handle2, &backend)
        .expect("free should succeed");
}

/// QuantizedPaged rejects alloc_contiguous.
#[test]
fn test_kv_cache_backend_quantized_paged_rejects_contiguous() {
    let cfg = test_config();
    let backend = MockBackend::new();
    let tq_config = fracture_core::TurboQuantConfig::default();

    let mut kv_backend = KvCacheBackend::QuantizedPaged(
        crate::quantized_paged_kv_cache::QuantizedKvCacheManager::new(
            4,
            cfg.num_layers,
            cfg.num_kv_heads,
            cfg.head_dim,
            16,
            tq_config,
            &backend,
        )
        .unwrap(),
    );

    let err = kv_backend.alloc_contiguous(&backend).unwrap_err();
    assert!(
        matches!(err, FractureError::KvCache(_)),
        "alloc_contiguous on QuantizedPaged should return KvCache error, got: {err:?}"
    );
}

/// Contiguous and Paged variants return false for is_quantized().
#[test]
fn test_kv_cache_backend_is_quantized() {
    let cfg = test_config();
    let backend = MockBackend::new();

    let contiguous = KvCacheBackend::Contiguous(KvCacheManager::new(
        cfg.num_layers,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.max_seq_len,
    ));
    assert!(!contiguous.is_quantized());

    let paged = KvCacheBackend::Paged(
        PagedKvCacheManager::new(4, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, &backend)
            .unwrap(),
    );
    assert!(!paged.is_quantized());
}
