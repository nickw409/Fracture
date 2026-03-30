use crate::kv_cache::{CacheHandle, KvCacheManager};
use crate::node::{NodeConfig, NodeInput, NodeOutput};
use crate::paged_kv_cache::PagedKvCacheManager;
use crate::quantized_paged_kv_cache::QuantizedKvCacheManager;
use fracture_core::{Backend, DType, DeviceTensor, ForwardProfile, FractureError, LayerProfile, Result};
use fracture_gguf::WeightStore;
use std::ops::Range;

/// Runtime-selected KV cache implementation.
#[allow(clippy::large_enum_variant)]
pub enum KvCacheBackend {
    Contiguous(KvCacheManager),
    Paged(PagedKvCacheManager),
    /// TurboQuant-compressed paged KV cache (opt-in via `--kv-quant turboquant`).
    QuantizedPaged(QuantizedKvCacheManager),
}

impl KvCacheBackend {
    pub fn alloc_contiguous<B: Backend>(&mut self, backend: &B) -> Result<CacheHandle> {
        match self {
            Self::Contiguous(c) => c.alloc(backend),
            _ => Err(FractureError::KvCache(
                "alloc_contiguous called on non-contiguous cache".into(),
            )),
        }
    }

    pub fn alloc_paged(&mut self) -> Result<CacheHandle> {
        match self {
            Self::Paged(p) => p.alloc(),
            Self::QuantizedPaged(q) => q.alloc(),
            Self::Contiguous(_) => Err(FractureError::KvCache(
                "alloc_paged called on contiguous cache".into(),
            )),
        }
    }

    pub fn alloc<B: Backend>(&mut self, backend: &B) -> Result<CacheHandle> {
        match self {
            Self::Contiguous(c) => c.alloc(backend),
            Self::Paged(p) => p.alloc(),
            Self::QuantizedPaged(q) => q.alloc(),
        }
    }

    pub fn seq_len(&self, handle: CacheHandle) -> Result<usize> {
        match self {
            Self::Contiguous(c) => c.seq_len(handle),
            Self::Paged(p) => p.seq_len(handle),
            Self::QuantizedPaged(q) => q.seq_len(handle),
        }
    }

    pub fn free<B: Backend>(&mut self, handle: CacheHandle, backend: &B) -> Result<()> {
        match self {
            Self::Contiguous(c) => c.free(handle, backend),
            Self::Paged(p) => {
                p.free(handle)?;
                Ok(())
            }
            Self::QuantizedPaged(q) => {
                q.free(handle)?;
                Ok(())
            }
        }
    }

    pub fn is_paged(&self) -> bool {
        matches!(self, Self::Paged(_) | Self::QuantizedPaged(_))
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::QuantizedPaged(_))
    }
}

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

    /// Consume the engine and return the backend for reuse (e.g., during
    /// worker reconfiguration where the GPU context survives but weights change).
    pub fn into_backend(self) -> B {
        self.backend
    }

    pub fn config(&self) -> &fracture_core::ModelConfig {
        &self.weights.config
    }

    pub fn layer_range(&self) -> &Range<usize> {
        &self.layer_range
    }

    pub fn weights(&self) -> &WeightStore {
        &self.weights
    }

    /// Run the forward pass: token_ids → logits (Phase 1 backward-compatible API).
    ///
    /// This is a thin wrapper around `forward_node()` with a full-model NodeConfig.
    /// The layer_range, profiling, and error propagation semantics are unchanged.
    pub fn forward(
        &self,
        token_ids: &[u32],
        positions: &[u32],
        cache: &mut KvCacheManager,
        cache_handle: CacheHandle,
        profile: Option<&mut ForwardProfile>,
    ) -> Result<Vec<f32>> {
        let node_config = NodeConfig::new(
            self.layer_range.clone(),
            self.weights.config.num_layers,
        )?;
        let input = NodeInput::TokenIds {
            ids: token_ids.to_vec(),
            positions: positions.to_vec(),
        };
        match self.forward_node(input, &node_config, cache, cache_handle, profile)? {
            NodeOutput::Logits(logits) => Ok(logits),
            NodeOutput::Activations(_) => Err(FractureError::Pipeline(
                "full forward expected Logits but got Activations".into(),
            )),
        }
    }

    /// Phase 2 forward pass: accepts NodeInput, returns NodeOutput based on NodeConfig.
    ///
    /// - Head node (is_head): TokenIds → embedding → layers → Activations
    /// - Middle node: Activations → layers → Activations
    /// - Tail node (is_tail): input → layers → rmsnorm → lm_head → Logits
    /// - Full node (is_head + is_tail): TokenIds → everything → Logits
    ///
    /// When `profile` is `Some`, per-layer GPU timing is recorded. When `None`,
    /// no timers are created (zero overhead). NVTX markers are always emitted.
    pub fn forward_node(
        &self,
        input: NodeInput,
        node_config: &NodeConfig,
        cache: &mut KvCacheManager,
        cache_handle: CacheHandle,
        profile: Option<&mut ForwardProfile>,
    ) -> Result<NodeOutput> {
        let cfg = &self.weights.config;
        let hidden = cfg.hidden_size;
        let num_q_heads = cfg.num_q_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let intermediate = cfg.intermediate_size;

        let profiling = profile.is_some();

        // 1. Resolve input: embedding lookup (head) or use provided activations
        let (hidden_state, positions, seq_len, owns_hidden) = match input {
            NodeInput::TokenIds { ids, positions } => {
                if !node_config.is_head() {
                    return Err(FractureError::Pipeline(
                        "non-head node received TokenIds input".into(),
                    ));
                }
                if ids.is_empty() {
                    return Err(FractureError::InvalidShape(
                        "token_ids must not be empty".into(),
                    ));
                }
                let seq_len = ids.len();
                let hidden_state = self.backend.alloc(&[seq_len, hidden], DType::FP16)?;
                self.backend
                    .embedding(&ids, &self.weights.token_embedding, &hidden_state)?;
                (hidden_state, positions, seq_len, true)
            }
            NodeInput::Activations {
                hidden_states,
                positions,
            } => {
                if node_config.is_head() {
                    return Err(FractureError::Pipeline(
                        "head node received Activations input".into(),
                    ));
                }
                let seq_len = positions.len();
                // hidden_states is owned by the caller (previous node output);
                // we do NOT free it — we work on it in-place via residual connections.
                (hidden_states, positions, seq_len, false)
            }
        };

        // Validate positions are within max_seq_len bounds (RoPE table size).
        if let Some(&max_pos) = positions.iter().max()
            && max_pos as usize >= cfg.max_seq_len {
                return Err(FractureError::InvalidShape(format!(
                    "position {} exceeds max_seq_len {}",
                    max_pos, cfg.max_seq_len,
                )));
            }

        // Pre-allocate reusable scratch tensors for the forward pass.
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

        // 2. Transformer layers — iterate the node config's range, not the engine's.
        // The engine's layer_range defines which weights are loaded (weight indexing base),
        // but the node config controls which layers to actually execute.
        // weight_idx: index into self.weights.layers (relative to engine's layer_range)
        // cache_idx: index into the KV cache (relative to the node config's layer_range)
        let exec_range = &node_config.layer_range;
        for layer_idx in exec_range.clone() {
            let weight_idx = layer_idx - self.layer_range.start;
            let cache_idx = layer_idx - exec_range.start;
            self.backend.marker_push(&format!("layer_{}", layer_idx));

            let w = &self.weights.layers[weight_idx];

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
                    .rope(&q_mh, &k_mh, &positions, cfg.rope_theta, head_dim)
            })?;

            // 2e-2f. KV cache update + grouped-query attention
            // cache_idx: cache is allocated for exec_range.len() layers,
            // so exec_range.start maps to cache slot 0.
            let k_cache = cache.k_cache(cache_handle, cache_idx)?;
            let v_cache = cache.v_cache(cache_handle, cache_idx)?;

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

            // 2g. Output projection
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
            if layer_idx == exec_range.start {
                cache.set_seq_len(cache_handle, new_seq_len)?;
            }
        }

        // Finalize profiling data.
        if let Some(profile) = profile {
            let total_ms = layer_profiles.iter().map(|lp| lp.total_ms).sum();
            profile.total_ms = total_ms;
            profile.prefill = seq_len > 1;
            profile.seq_len = seq_len;
            profile.layer_profiles = layer_profiles;
        }

        // 3. Output phase: tail produces logits, non-tail returns activations
        if node_config.is_tail() {
            // Final RMSNorm
            self.backend
                .rmsnorm(&hidden_state, &self.weights.output_norm, cfg.rms_norm_eps, &normed)?;

            // LM head: extract last position, matmul to vocab
            let last_hidden = if seq_len > 1 {
                let last = self.backend.alloc(&[1, hidden], DType::FP16)?;
                self.backend
                    .copy_rows(&normed, &last, seq_len - 1, 0, 1)?;
                last
            } else {
                DeviceTensor::new(normed.id, vec![1, hidden], DType::FP16)
            };

            let logits_tensor = self
                .backend
                .alloc(&[1, cfg.vocab_size], DType::FP16)?;
            self.backend
                .matmul(&last_hidden, &self.weights.lm_head, &logits_tensor)?;

            // Copy logits to host as FP32
            let mut logits_fp16 = vec![0u8; cfg.vocab_size * 2];
            self.backend.copy_to_host(&logits_tensor, &mut logits_fp16)?;
            self.backend.synchronize()?;

            let logits: Vec<f32> = logits_fp16
                .chunks_exact(2)
                .map(|bytes| {
                    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect();

            // Free all scratch tensors + hidden_state (we're done with it)
            if owns_hidden {
                self.backend.free(&hidden_state)?;
            }
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
                self.backend.free(&last_hidden)?;
            }

            Ok(NodeOutput::Logits(logits))
        } else {
            // Non-tail: return hidden_state as activations for the next node.
            // Do NOT free hidden_state — it is the output.
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

            Ok(NodeOutput::Activations(hidden_state))
        }
    }

    /// Paged KV cache forward pass: token_ids → logits.
    ///
    /// Same as `forward()` but uses paged attention with block tables.
    pub fn forward_paged(
        &self,
        token_ids: &[u32],
        positions: &[u32],
        cache: &mut PagedKvCacheManager,
        cache_handle: CacheHandle,
    ) -> Result<Vec<f32>> {
        let node_config = NodeConfig::new(
            self.layer_range.clone(),
            self.weights.config.num_layers,
        )?;
        let input = NodeInput::TokenIds {
            ids: token_ids.to_vec(),
            positions: positions.to_vec(),
        };
        match self.forward_node_paged(input, &node_config, cache, cache_handle)? {
            NodeOutput::Logits(logits) => Ok(logits),
            NodeOutput::Activations(_) => Err(FractureError::Pipeline(
                "full forward expected Logits but got Activations".into(),
            )),
        }
    }

    /// Paged KV cache variant of forward_node.
    ///
    /// Identical to forward_node except:
    /// - KV cache write uses paged append_kv instead of copy_rows into contiguous tensors
    /// - Attention uses attention_paged with block tables instead of contiguous k/v cache
    pub fn forward_node_paged(
        &self,
        input: NodeInput,
        node_config: &NodeConfig,
        cache: &mut PagedKvCacheManager,
        cache_handle: CacheHandle,
    ) -> Result<NodeOutput> {
        let cfg = &self.weights.config;
        let hidden = cfg.hidden_size;
        let num_q_heads = cfg.num_q_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let intermediate = cfg.intermediate_size;

        // 1. Resolve input (identical to contiguous path)
        let (hidden_state, positions, seq_len, owns_hidden) = match input {
            NodeInput::TokenIds { ids, positions } => {
                if !node_config.is_head() {
                    return Err(FractureError::Pipeline(
                        "non-head node received TokenIds input".into(),
                    ));
                }
                if ids.is_empty() {
                    return Err(FractureError::InvalidShape(
                        "token_ids must not be empty".into(),
                    ));
                }
                let seq_len = ids.len();
                let hidden_state = self.backend.alloc(&[seq_len, hidden], DType::FP16)?;
                self.backend
                    .embedding(&ids, &self.weights.token_embedding, &hidden_state)?;
                (hidden_state, positions, seq_len, true)
            }
            NodeInput::Activations {
                hidden_states,
                positions,
            } => {
                if node_config.is_head() {
                    return Err(FractureError::Pipeline(
                        "head node received Activations input".into(),
                    ));
                }
                let seq_len = positions.len();
                (hidden_states, positions, seq_len, false)
            }
        };

        // Position bounds check
        if let Some(&max_pos) = positions.iter().max()
            && max_pos as usize >= cfg.max_seq_len {
                return Err(FractureError::InvalidShape(format!(
                    "position {} exceeds max_seq_len {}",
                    max_pos, cfg.max_seq_len,
                )));
            }

        // Pre-allocate scratch tensors (identical to contiguous path)
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

        // 2. Transformer layers
        let exec_range = &node_config.layer_range;
        for layer_idx in exec_range.clone() {
            let weight_idx = layer_idx - self.layer_range.start;
            let cache_idx = layer_idx - exec_range.start;
            self.backend.marker_push(&format!("layer_{}", layer_idx));

            let w = &self.weights.layers[weight_idx];

            // 2a. Pre-attention RMSNorm
            self.backend
                .rmsnorm(&hidden_state, &w.attn_norm, cfg.rms_norm_eps, &normed)?;

            // 2b. QKV projections
            self.backend.matmul(&normed, &w.q_proj, &q_flat)?;
            self.backend.matmul(&normed, &w.k_proj, &k_flat)?;
            self.backend.matmul(&normed, &w.v_proj, &v_flat)?;

            // 2c. Reshape for multi-head
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

            // 2d. Apply RoPE
            self.backend
                .rope(&q_mh, &k_mh, &positions, cfg.rope_theta, head_dim)?;

            // 2e. KV cache update — PAGED: append into block pool
            cache.append_kv(cache_handle, cache_idx, &k_mh, &v_mh, &self.backend)?;

            let new_seq_len = start_pos + seq_len;

            // 2f. Paged attention — reads from block table
            let block_table = cache.block_table(cache_handle)?;
            let block_table_i32: Vec<i32> = block_table.iter().map(|&b| b as i32).collect();

            // Collect block K/V DeviceTensors for this layer
            let pool = cache.pool();
            let k_blocks: Vec<&DeviceTensor> = (0..pool.capacity())
                .map(|bid| pool.k_tensor(bid, cache_idx))
                .collect();
            let v_blocks: Vec<&DeviceTensor> = (0..pool.capacity())
                .map(|bid| pool.v_tensor(bid, cache_idx))
                .collect();

            let attn_out = DeviceTensor::new(
                attn_out_mh.id,
                vec![seq_len, num_q_heads, head_dim],
                DType::FP16,
            );

            self.backend.attention_paged(
                &q_mh,
                &block_table_i32,
                &k_blocks,
                &v_blocks,
                num_kv_heads,
                new_seq_len,
                start_pos,
                &attn_out,
            )?;

            // 2g. Output projection (identical to contiguous)
            let attn_out_flat = DeviceTensor::new(
                attn_out_mh.id,
                vec![seq_len, hidden],
                DType::FP16,
            );
            self.backend
                .matmul(&attn_out_flat, &w.o_proj, &projected)?;

            // 2h. Residual
            self.backend
                .add(&hidden_state, &projected, &hidden_state)?;

            // 2i-2k. FFN (identical to contiguous)
            self.backend
                .rmsnorm(&hidden_state, &w.ffn_norm, cfg.rms_norm_eps, &normed)?;
            self.backend.matmul(&normed, &w.gate_proj, &gate)?;
            self.backend.matmul(&normed, &w.up_proj, &up)?;
            self.backend.silu_mul(&gate, &up, &ffn_mid)?;
            self.backend.matmul(&ffn_mid, &w.down_proj, &ffn_out)?;
            self.backend
                .add(&hidden_state, &ffn_out, &hidden_state)?;

            self.backend.marker_pop();
        }

        // 3. Output phase (identical to contiguous)
        if node_config.is_tail() {
            self.backend
                .rmsnorm(&hidden_state, &self.weights.output_norm, cfg.rms_norm_eps, &normed)?;

            let last_hidden = if seq_len > 1 {
                let last = self.backend.alloc(&[1, hidden], DType::FP16)?;
                self.backend
                    .copy_rows(&normed, &last, seq_len - 1, 0, 1)?;
                last
            } else {
                DeviceTensor::new(normed.id, vec![1, hidden], DType::FP16)
            };

            let logits_tensor = self
                .backend
                .alloc(&[1, cfg.vocab_size], DType::FP16)?;
            self.backend
                .matmul(&last_hidden, &self.weights.lm_head, &logits_tensor)?;

            let mut logits_fp16 = vec![0u8; cfg.vocab_size * 2];
            self.backend.copy_to_host(&logits_tensor, &mut logits_fp16)?;
            self.backend.synchronize()?;

            let logits: Vec<f32> = logits_fp16
                .chunks_exact(2)
                .map(|bytes| {
                    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect();

            if owns_hidden {
                self.backend.free(&hidden_state)?;
            }
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
                self.backend.free(&last_hidden)?;
            }

            Ok(NodeOutput::Logits(logits))
        } else {
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

            Ok(NodeOutput::Activations(hidden_state))
        }
    }
}

// Profiling dispatch is tested implicitly: the timed_op function's zero-overhead path
// (profile=None) is exercised by all generation tests, which call forward() without a
// ForwardProfile. The profiling-active path (profile=Some) requires a real GPU backend
// to return meaningful timer values.

#[cfg(test)]
mod engine_tests;
