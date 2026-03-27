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
}
