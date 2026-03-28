use crate::kv_cache::{CacheHandle, KvCacheManager};
use fracture_core::{Backend, DType, DeviceTensor, ForwardProfile, LayerProfile, Result};
use fracture_gguf::WeightStore;
use std::ops::Range;

/// Time a single operation if profiling is active.
///
/// When `profiling` is false, simply executes the closure with zero overhead.
/// When true, brackets the closure with GPU timer start/stop and returns elapsed ms.
fn timed_op<B: Backend>(
    backend: &B,
    profiling: bool,
    f: impl FnOnce() -> Result<()>,
) -> Result<f32> {
    if profiling {
        let timer = backend.create_timer()?;
        backend.start_timer(&timer)?;
        f()?;
        let elapsed = backend.stop_timer(&timer)?;
        backend.destroy_timer(&timer)?;
        Ok(elapsed)
    } else {
        f()?;
        Ok(0.0)
    }
}

/// The backend-agnostic transformer forward pass engine.
///
/// Generic over `B: Backend` — contains no CUDA or Metal imports.
/// Dispatches all GPU operations through Backend trait methods.
///
/// # Layer Range
///
/// The `layer_range` field controls which transformer layers this engine instance
/// processes. In Phase 1, this is always `0..num_layers` and `forward()` accepts
/// token IDs and returns logits for the full model. In Phase 2, partial layer ranges
/// will accept/return intermediate activation tensors for pipeline-parallel inference
/// across multiple nodes.
pub struct Engine<B: Backend> {
    backend: B,
    weights: WeightStore,
    layer_range: Range<usize>,
}

impl<B: Backend> Engine<B> {
    pub fn new(backend: B, weights: WeightStore, layer_range: Range<usize>) -> Self {
        Self {
            backend,
            weights,
            layer_range,
        }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn config(&self) -> &fracture_core::ModelConfig {
        &self.weights.config
    }

    pub fn weights(&self) -> &WeightStore {
        &self.weights
    }

    /// Run the forward pass: token_ids → logits.
    ///
    /// In Phase 1, layer_range is always [0, num_layers) and this accepts token IDs
    /// and returns logits [vocab_size] (last position only).
    /// In Phase 2, a partial layer range accepts/returns activation tensors.
    ///
    /// When `profile` is `Some`, per-layer GPU timing is recorded into the provided
    /// `ForwardProfile`. When `None`, no timers are created and there is zero overhead.
    /// NVTX markers (marker_push/marker_pop) are always emitted regardless of profiling
    /// state — they are no-ops unless the backend overrides them.
    ///
    /// # Error Propagation
    ///
    /// All Backend trait calls use `?` for error propagation. Errors carry context
    /// from the Backend implementation (e.g., CUDA error codes, allocation failures).
    /// No panics — every fallible path returns `Result<T, FractureError>`.
    pub fn forward(
        &self,
        token_ids: &[u32],
        positions: &[u32],
        cache: &mut KvCacheManager,
        cache_handle: CacheHandle,
        profile: Option<&mut ForwardProfile>,
    ) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(fracture_core::FractureError::InvalidShape(
                "token_ids must not be empty".into(),
            ));
        }

        let cfg = &self.weights.config;
        let seq_len = token_ids.len();
        let hidden = cfg.hidden_size;
        let num_q_heads = cfg.num_q_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let intermediate = cfg.intermediate_size;

        let profiling = profile.is_some();

        // 1. Embedding lookup
        let hidden_state = self.backend.alloc(&[seq_len, hidden], DType::FP16)?;
        self.backend
            .embedding(token_ids, &self.weights.token_embedding, &hidden_state)?;

        // Pre-allocate reusable scratch tensors for the forward pass.
        // These are reused across layers to avoid repeated alloc/free.
        let normed = self.backend.alloc(&[seq_len, hidden], DType::FP16)?;
        let q_flat = self.backend.alloc(&[seq_len, hidden], DType::FP16)?;
        let k_flat = self
            .backend
            .alloc(&[seq_len, num_kv_heads * head_dim], DType::FP16)?;
        let v_flat = self
            .backend
            .alloc(&[seq_len, num_kv_heads * head_dim], DType::FP16)?;
        let attn_out_mh = self.backend.alloc(&[seq_len, hidden], DType::FP16)?;
        let projected = self.backend.alloc(&[seq_len, hidden], DType::FP16)?;
        let gate = self
            .backend
            .alloc(&[seq_len, intermediate], DType::FP16)?;
        let up = self
            .backend
            .alloc(&[seq_len, intermediate], DType::FP16)?;
        let ffn_mid = self
            .backend
            .alloc(&[seq_len, intermediate], DType::FP16)?;
        let ffn_out = self.backend.alloc(&[seq_len, hidden], DType::FP16)?;

        let start_pos = cache.seq_len(cache_handle)?;

        let mut layer_profiles: Vec<LayerProfile> = Vec::new();

        // 2. Transformer layers
        for layer_idx in self.layer_range.clone() {
            self.backend.marker_push(&format!("layer_{}", layer_idx));

            let w = &self.weights.layers[layer_idx];

            // 2a. Pre-attention RMSNorm
            let rmsnorm_attn_ms = timed_op(&self.backend, profiling, || {
                self.backend
                    .rmsnorm(&hidden_state, &w.attn_norm, cfg.rms_norm_eps, &normed)
            })?;

            // 2b. QKV projections
            let qkv_proj_ms = timed_op(&self.backend, profiling, || {
                self.backend.matmul(&normed, &w.q_proj, &q_flat)?;
                self.backend.matmul(&normed, &w.k_proj, &k_flat)?;
                self.backend.matmul(&normed, &w.v_proj, &v_flat)
            })?;

            // 2c. Reshape for multi-head (metadata-only, same underlying memory)
            let q_mh = DeviceTensor::new(
                q_flat.id,
                vec![seq_len, num_q_heads, head_dim],
                DType::FP16,
            );
            let k_mh = DeviceTensor::new(
                k_flat.id,
                vec![seq_len, num_kv_heads, head_dim],
                DType::FP16,
            );
            let v_mh = DeviceTensor::new(
                v_flat.id,
                vec![seq_len, num_kv_heads, head_dim],
                DType::FP16,
            );

            // 2d. Apply RoPE to Q and K
            let rope_ms = timed_op(&self.backend, profiling, || {
                self.backend
                    .rope(&q_mh, &k_mh, positions, cfg.rope_theta, head_dim)
            })?;

            // 2e-2f. KV cache update + grouped-query attention
            // Get cache refs before the timed closure to avoid borrow conflicts.
            let k_cache = cache.k_cache(cache_handle, layer_idx)?;
            let v_cache = cache.v_cache(cache_handle, layer_idx)?;

            let new_seq_len = start_pos + seq_len;

            let attn_out = DeviceTensor::new(
                attn_out_mh.id,
                vec![seq_len, num_q_heads, head_dim],
                DType::FP16,
            );

            let attention_ms = timed_op(&self.backend, profiling, || {
                self.backend
                    .copy_rows(&k_mh, k_cache, 0, start_pos, seq_len)?;
                self.backend
                    .copy_rows(&v_mh, v_cache, 0, start_pos, seq_len)?;
                self.backend.attention(
                    &q_mh,
                    k_cache,
                    v_cache,
                    num_kv_heads,
                    start_pos,
                    &attn_out,
                )
            })?;

            // 2g. Output projection (reshape attn_out back to [seq_len, hidden])
            let attn_out_flat = DeviceTensor::new(
                attn_out_mh.id,
                vec![seq_len, hidden],
                DType::FP16,
            );

            let output_proj_ms = timed_op(&self.backend, profiling, || {
                self.backend
                    .matmul(&attn_out_flat, &w.o_proj, &projected)
            })?;

            // 2h. Residual connection
            self.backend
                .add(&hidden_state, &projected, &hidden_state)?;

            // 2i. Pre-FFN RMSNorm
            let rmsnorm_ffn_ms = timed_op(&self.backend, profiling, || {
                self.backend
                    .rmsnorm(&hidden_state, &w.ffn_norm, cfg.rms_norm_eps, &normed)
            })?;

            // 2j. SwiGLU FFN
            let gate_up_proj_ms = timed_op(&self.backend, profiling, || {
                self.backend.matmul(&normed, &w.gate_proj, &gate)?;
                self.backend.matmul(&normed, &w.up_proj, &up)
            })?;

            let silu_mul_ms = timed_op(&self.backend, profiling, || {
                self.backend.silu_mul(&gate, &up, &ffn_mid)
            })?;

            let down_proj_ms = timed_op(&self.backend, profiling, || {
                self.backend.matmul(&ffn_mid, &w.down_proj, &ffn_out)
            })?;

            // 2k. Residual connection
            self.backend
                .add(&hidden_state, &ffn_out, &hidden_state)?;

            // Collect layer profile if profiling is active.
            if profiling {
                let total_ms = rmsnorm_attn_ms
                    + qkv_proj_ms
                    + rope_ms
                    + attention_ms
                    + output_proj_ms
                    + rmsnorm_ffn_ms
                    + gate_up_proj_ms
                    + silu_mul_ms
                    + down_proj_ms;

                layer_profiles.push(LayerProfile {
                    layer_idx,
                    total_ms,
                    rmsnorm_attn_ms,
                    qkv_proj_ms,
                    rope_ms,
                    attention_ms,
                    output_proj_ms,
                    rmsnorm_ffn_ms,
                    gate_up_proj_ms,
                    silu_mul_ms,
                    down_proj_ms,
                });
            }

            self.backend.marker_pop();

            // Update cached seq_len after first layer processes it
            // (only update once, not per layer)
            if layer_idx == self.layer_range.start {
                cache.set_seq_len(cache_handle, new_seq_len)?;
            }
        }

        // 3. Final RMSNorm
        self.backend
            .rmsnorm(&hidden_state, &self.weights.output_norm, cfg.rms_norm_eps, &normed)?;

        // 4. LM head: [seq_len, hidden] @ [vocab_size, hidden]^T → [seq_len, vocab_size]
        // We only need logits for the last token position.
        // For efficiency, extract the last row and matmul just that.
        let last_hidden = if seq_len > 1 {
            let last = self.backend.alloc(&[1, hidden], DType::FP16)?;
            self.backend
                .copy_rows(&normed, &last, seq_len - 1, 0, 1)?;
            last
        } else {
            // seq_len == 1 (decode), normed is already [1, hidden]
            DeviceTensor::new(normed.id, vec![1, hidden], DType::FP16)
        };

        let logits_tensor = self
            .backend
            .alloc(&[1, cfg.vocab_size], DType::FP16)?;
        self.backend
            .matmul(&last_hidden, &self.weights.lm_head, &logits_tensor)?;

        // Copy logits to host as FP32
        let mut logits_fp16 = vec![0u8; cfg.vocab_size * 2]; // FP16 = 2 bytes each
        self.backend.copy_to_host(&logits_tensor, &mut logits_fp16)?;
        self.backend.synchronize()?;

        // Finalize profiling data.
        if let Some(profile) = profile {
            let total_ms = layer_profiles.iter().map(|lp| lp.total_ms).sum();
            profile.total_ms = total_ms;
            profile.prefill = seq_len > 1;
            profile.seq_len = seq_len;
            profile.layer_profiles = layer_profiles;
        }

        // Convert FP16 → FP32 on CPU
        let logits: Vec<f32> = logits_fp16
            .chunks_exact(2)
            .map(|bytes| {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect();

        // Free scratch tensors (but not hidden_state — it aliases normed via reshape)
        // We need to free only tensors we actually allocated, not reshapes
        self.backend.free(&hidden_state)?;
        self.backend.free(&normed)?;
        self.backend.free(&q_flat)?;
        self.backend.free(&k_flat)?;
        self.backend.free(&v_flat)?;
        self.backend.free(&attn_out_mh)?;
        self.backend.free(&projected)?;
        self.backend.free(&gate)?;
        self.backend.free(&up)?;
        self.backend.free(&ffn_mid)?;
        self.backend.free(&ffn_out)?;
        self.backend.free(&logits_tensor)?;
        if seq_len > 1 {
            // last_hidden was a separate allocation
            self.backend.free(&last_hidden)?;
        }

        Ok(logits)
    }
}

// Profiling dispatch is tested implicitly: the timed_op function's zero-overhead path
// (profile=None) is exercised by all generation tests, which call forward() without a
// ForwardProfile. The profiling-active path (profile=Some) requires a real GPU backend
// to return meaningful timer values.

#[cfg(test)]
mod tests {
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

    /// Verify partial layer range: Engine with layer_range 0..1 out of 2 layers
    /// should still produce logits (since we have all the global tensors).
    #[test]
    fn test_partial_layer_range() {
        let mut cfg = test_config();
        cfg.num_layers = 2; // model has 2 layers
        let backend = MockBackend::new();
        let weights = mock_weights(&cfg);
        // Only process layer 0
        let engine = Engine::new(backend, weights, 0..1);
        let mut cache = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
        let handle = cache.alloc(engine.backend()).unwrap();

        let result = engine.forward(&[1], &[0], &mut cache, handle, None);
        assert!(result.is_ok(), "partial layer range should succeed: {:?}", result.err());

        let logits = result.unwrap();
        assert_eq!(logits.len(), cfg.vocab_size, "should still produce vocab_size logits");
    }
}
