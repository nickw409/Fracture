use crate::kv_cache::{CacheHandle, KvCacheManager};
use crate::node::{NodeConfig, NodeInput, NodeOutput};
use crate::paged_kv_cache::PagedKvCacheManager;
use fracture_core::{Backend, DType, DeviceTensor, ForwardProfile, FractureError, LayerProfile, Result};
use fracture_gguf::WeightStore;
use std::ops::Range;

/// Runtime-selected KV cache implementation.
pub enum KvCacheBackend {
    Contiguous(KvCacheManager),
    Paged(PagedKvCacheManager),
}

impl KvCacheBackend {
    pub fn alloc_contiguous<B: Backend>(&mut self, backend: &B) -> Result<CacheHandle> {
        match self {
            Self::Contiguous(c) => c.alloc(backend),
            Self::Paged(_) => Err(FractureError::KvCache(
                "alloc_contiguous called on paged cache".into(),
            )),
        }
    }

    pub fn alloc_paged(&mut self) -> Result<CacheHandle> {
        match self {
            Self::Paged(p) => p.alloc(),
            Self::Contiguous(_) => Err(FractureError::KvCache(
                "alloc_paged called on contiguous cache".into(),
            )),
        }
    }

    pub fn alloc<B: Backend>(&mut self, backend: &B) -> Result<CacheHandle> {
        match self {
            Self::Contiguous(c) => c.alloc(backend),
            Self::Paged(p) => p.alloc(),
        }
    }

    pub fn seq_len(&self, handle: CacheHandle) -> Result<usize> {
        match self {
            Self::Contiguous(c) => c.seq_len(handle),
            Self::Paged(p) => p.seq_len(handle),
        }
    }

    pub fn free<B: Backend>(&mut self, handle: CacheHandle, backend: &B) -> Result<()> {
        match self {
            Self::Contiguous(c) => c.free(handle, backend),
            Self::Paged(p) => {
                p.free(handle)?;
                Ok(())
            }
        }
    }

    pub fn is_paged(&self) -> bool {
        matches!(self, Self::Paged(_))
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
        if let Some(&max_pos) = positions.iter().max() {
            if max_pos as usize >= cfg.max_seq_len {
                return Err(FractureError::InvalidShape(format!(
                    "position {} exceeds max_seq_len {}",
                    max_pos, cfg.max_seq_len,
                )));
            }
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
        if let Some(&max_pos) = positions.iter().max() {
            if max_pos as usize >= cfg.max_seq_len {
                return Err(FractureError::InvalidShape(format!(
                    "position {} exceeds max_seq_len {}",
                    max_pos, cfg.max_seq_len,
                )));
            }
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
}
