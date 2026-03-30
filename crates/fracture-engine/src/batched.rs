use crate::kv_cache::CacheHandle;
use crate::node::{NodeConfig, NodeOutput};
use crate::paged_kv_cache::PagedKvCacheManager;
use fracture_core::{Backend, DType, DeviceTensor, FractureError, Result};
use fracture_gguf::WeightStore;
use std::ops::Range;

/// A single sequence's contribution to a batched forward pass.
pub struct SequenceSlice {
    /// Cache handle for this sequence.
    pub handle: CacheHandle,
    /// Token IDs for this iteration.
    /// Prefill: full prompt (or chunk). Decode: single token.
    pub token_ids: Vec<u32>,
    /// Absolute positions for RoPE.
    pub positions: Vec<u32>,
}

/// Output from a batched forward pass.
/// Contains per-sequence logits (last token only per sequence).
pub struct BatchedOutput {
    /// (seq_index, logits) — one entry per sequence in the batch.
    pub logits: Vec<Vec<f32>>,
}

/// Run a batched forward pass through the full model.
///
/// All sequences share the same weights but have independent KV caches
/// (paged block tables). Operations that are per-token (matmul, RMSNorm,
/// RoPE, FFN) are concatenated into a single large tensor for GPU
/// efficiency. Attention is dispatched per-sequence since each has
/// its own block table.
pub fn batched_forward<B: Backend>(
    backend: &B,
    weights: &WeightStore,
    layer_range: &Range<usize>,
    cache: &mut PagedKvCacheManager,
    sequences: &[SequenceSlice],
) -> Result<BatchedOutput> {
    if sequences.is_empty() {
        return Ok(BatchedOutput { logits: Vec::new() });
    }

    let cfg = &weights.config;
    let hidden = cfg.hidden_size;
    let num_q_heads = cfg.num_q_heads;
    let num_kv_heads = cfg.num_kv_heads;
    let head_dim = cfg.head_dim;
    let intermediate = cfg.intermediate_size;

    // Build concatenated batch: all tokens from all sequences.
    let mut all_token_ids: Vec<u32> = Vec::new();
    let mut all_positions: Vec<u32> = Vec::new();
    let mut seq_boundaries: Vec<(usize, usize)> = Vec::new(); // (start, end) into the concatenated batch

    let mut offset = 0;
    for seq in sequences {
        let n = seq.token_ids.len();
        all_token_ids.extend_from_slice(&seq.token_ids);
        all_positions.extend_from_slice(&seq.positions);
        seq_boundaries.push((offset, offset + n));
        offset += n;
    }

    let total_tokens = all_token_ids.len();

    // Pre-compute start_pos for each sequence (before this batch's tokens).
    let start_positions: Vec<usize> = sequences
        .iter()
        .map(|s| cache.seq_len(s.handle))
        .collect::<Result<Vec<_>>>()?;

    // 1. Embedding — batched
    let hidden_state = backend.alloc(&[total_tokens, hidden], DType::FP16)?;
    backend.embedding(&all_token_ids, &weights.token_embedding, &hidden_state)?;

    // Scratch tensors — allocated once, reused across layers
    let normed = backend.alloc(&[total_tokens, hidden], DType::FP16)?;
    let q_flat = backend.alloc(&[total_tokens, hidden], DType::FP16)?;
    let k_flat = backend.alloc(&[total_tokens, num_kv_heads * head_dim], DType::FP16)?;
    let v_flat = backend.alloc(&[total_tokens, num_kv_heads * head_dim], DType::FP16)?;
    let attn_out_buf = backend.alloc(&[total_tokens, hidden], DType::FP16)?;
    let projected = backend.alloc(&[total_tokens, hidden], DType::FP16)?;
    let gate = backend.alloc(&[total_tokens, intermediate], DType::FP16)?;
    let up = backend.alloc(&[total_tokens, intermediate], DType::FP16)?;
    let ffn_mid = backend.alloc(&[total_tokens, intermediate], DType::FP16)?;
    let ffn_out = backend.alloc(&[total_tokens, hidden], DType::FP16)?;

    // Temporary tensors for per-sequence attention slicing
    // (we need per-sequence Q and attention output views)

    // 2. Layer loop
    for layer_idx in layer_range.clone() {
        let weight_idx = layer_idx - layer_range.start;
        let cache_idx = weight_idx; // cache layers indexed from 0
        let w = &weights.layers[weight_idx];

        // 2a. RMSNorm — batched
        backend.rmsnorm(&hidden_state, &w.attn_norm, cfg.rms_norm_eps, &normed)?;

        // 2b. QKV projections — batched
        backend.matmul(&normed, &w.q_proj, &q_flat)?;
        backend.matmul(&normed, &w.k_proj, &k_flat)?;
        backend.matmul(&normed, &w.v_proj, &v_flat)?;

        // 2c-d. Reshape + RoPE — batched
        let q_mh = DeviceTensor::new(
            q_flat.id,
            vec![total_tokens, num_q_heads, head_dim],
            DType::FP16,
        );
        let k_mh = DeviceTensor::new(
            k_flat.id,
            vec![total_tokens, num_kv_heads, head_dim],
            DType::FP16,
        );
        let v_mh = DeviceTensor::new(
            v_flat.id,
            vec![total_tokens, num_kv_heads, head_dim],
            DType::FP16,
        );

        backend.rope(&q_mh, &k_mh, &all_positions, cfg.rope_theta, head_dim)?;

        // 2e. KV cache append — PER-SEQUENCE
        // Each sequence's K/V slice is written to its own blocks.
        for (i, seq) in sequences.iter().enumerate() {
            let (start, end) = seq_boundaries[i];
            let n = end - start;

            // Create views into the concatenated K/V tensors for this sequence
            let seq_k = DeviceTensor::new(
                k_mh.id,
                vec![total_tokens, num_kv_heads, head_dim],
                DType::FP16,
            );
            let seq_v = DeviceTensor::new(
                v_mh.id,
                vec![total_tokens, num_kv_heads, head_dim],
                DType::FP16,
            );

            // Allocate temp tensors for the slice and copy
            let k_slice = backend.alloc(&[n, num_kv_heads, head_dim], DType::FP16)?;
            let v_slice = backend.alloc(&[n, num_kv_heads, head_dim], DType::FP16)?;
            backend.copy_rows(&seq_k, &k_slice, start, 0, n)?;
            backend.copy_rows(&seq_v, &v_slice, start, 0, n)?;

            cache.append_kv(seq.handle, cache_idx, &k_slice, &v_slice, backend)?;

            backend.free(&k_slice)?;
            backend.free(&v_slice)?;
        }

        // 2f. Attention — PER-SEQUENCE
        // Each sequence gets its own paged attention call. Results are
        // written into the correct slice of attn_out_buf.
        for (i, seq) in sequences.iter().enumerate() {
            let (start, end) = seq_boundaries[i];
            let n = end - start;
            let start_pos = start_positions[i];
            let new_seq_len = start_pos + n;

            // Slice Q for this sequence
            let q_slice = backend.alloc(&[n, num_q_heads, head_dim], DType::FP16)?;
            backend.copy_rows(&q_mh, &q_slice, start, 0, n)?;

            let attn_slice = backend.alloc(&[n, num_q_heads, head_dim], DType::FP16)?;

            let block_table = cache.block_table(seq.handle)?;
            let block_table_i32: Vec<i32> = block_table.iter().map(|&b| b as i32).collect();

            let pool = cache.pool();
            let k_blocks: Vec<&DeviceTensor> = (0..pool.capacity())
                .map(|bid| pool.k_tensor(bid, cache_idx))
                .collect();
            let v_blocks: Vec<&DeviceTensor> = (0..pool.capacity())
                .map(|bid| pool.v_tensor(bid, cache_idx))
                .collect();

            backend.attention_paged(
                &q_slice,
                &block_table_i32,
                &k_blocks,
                &v_blocks,
                num_kv_heads,
                new_seq_len,
                start_pos,
                &attn_slice,
            )?;

            // Copy attention output back into the batched buffer
            let attn_out_mh = DeviceTensor::new(
                attn_out_buf.id,
                vec![total_tokens, num_q_heads, head_dim],
                DType::FP16,
            );
            backend.copy_rows(&attn_slice, &attn_out_mh, 0, start, n)?;

            backend.free(&q_slice)?;
            backend.free(&attn_slice)?;
        }

        // 2g. Output projection — batched
        let attn_out_flat = DeviceTensor::new(
            attn_out_buf.id,
            vec![total_tokens, hidden],
            DType::FP16,
        );
        backend.matmul(&attn_out_flat, &w.o_proj, &projected)?;

        // 2h. Residual — batched
        backend.add(&hidden_state, &projected, &hidden_state)?;

        // 2i-k. FFN — batched
        backend.rmsnorm(&hidden_state, &w.ffn_norm, cfg.rms_norm_eps, &normed)?;
        backend.matmul(&normed, &w.gate_proj, &gate)?;
        backend.matmul(&normed, &w.up_proj, &up)?;
        backend.silu_mul(&gate, &up, &ffn_mid)?;
        backend.matmul(&ffn_mid, &w.down_proj, &ffn_out)?;
        backend.add(&hidden_state, &ffn_out, &hidden_state)?;
    }

    // 3. Final RMSNorm + LM Head — extract last token per sequence
    backend.rmsnorm(&hidden_state, &weights.output_norm, cfg.rms_norm_eps, &normed)?;

    let mut per_seq_logits = Vec::with_capacity(sequences.len());

    for (i, _seq) in sequences.iter().enumerate() {
        let (_start, end) = seq_boundaries[i];
        let last_idx = end - 1; // last token of this sequence

        let last_hidden = backend.alloc(&[1, hidden], DType::FP16)?;
        backend.copy_rows(&normed, &last_hidden, last_idx, 0, 1)?;

        let logits_tensor = backend.alloc(&[1, cfg.vocab_size], DType::FP16)?;
        backend.matmul(&last_hidden, &weights.lm_head, &logits_tensor)?;

        let mut logits_fp16 = vec![0u8; cfg.vocab_size * 2];
        backend.copy_to_host(&logits_tensor, &mut logits_fp16)?;

        let logits: Vec<f32> = logits_fp16
            .chunks_exact(2)
            .map(|bytes| {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect();

        per_seq_logits.push(logits);

        backend.free(&last_hidden)?;
        backend.free(&logits_tensor)?;
    }

    backend.synchronize()?;

    // Free scratch tensors
    backend.free(&hidden_state)?;
    backend.free(&normed)?;
    backend.free(&q_flat)?;
    backend.free(&k_flat)?;
    backend.free(&v_flat)?;
    backend.free(&attn_out_buf)?;
    backend.free(&projected)?;
    backend.free(&gate)?;
    backend.free(&up)?;
    backend.free(&ffn_mid)?;
    backend.free(&ffn_out)?;

    Ok(BatchedOutput { logits: per_seq_logits })
}

/// Run a batched forward pass through a pipeline node's layer range.
///
/// Like `batched_forward`, but supports distributed pipeline execution:
/// - Head nodes: receive token IDs, produce activations
/// - Middle nodes: receive activations, produce activations
/// - Tail nodes: receive activations, produce per-sequence logits
///
/// The `sequences` slice provides per-sequence cache handles and positions.
/// For head nodes, token IDs come from `SequenceSlice::token_ids`.
/// For middle/tail nodes, activations are provided via `input_hidden_states`.
pub fn batched_forward_node<B: Backend>(
    backend: &B,
    weights: &WeightStore,
    node_config: &NodeConfig,
    cache: &mut PagedKvCacheManager,
    sequences: &[SequenceSlice],
    input_hidden_states: Option<DeviceTensor>,
) -> Result<NodeOutput> {
    if sequences.is_empty() {
        return Err(FractureError::Pipeline(
            "batched_forward_node: no sequences".into(),
        ));
    }

    let cfg = &weights.config;
    let hidden = cfg.hidden_size;
    let num_q_heads = cfg.num_q_heads;
    let num_kv_heads = cfg.num_kv_heads;
    let head_dim = cfg.head_dim;
    let intermediate = cfg.intermediate_size;

    // Build concatenated positions and sequence boundaries.
    let mut all_positions: Vec<u32> = Vec::new();
    let mut seq_boundaries: Vec<(usize, usize)> = Vec::new();
    let mut offset = 0;
    for seq in sequences {
        let n = seq.positions.len();
        all_positions.extend_from_slice(&seq.positions);
        seq_boundaries.push((offset, offset + n));
        offset += n;
    }
    let total_tokens = all_positions.len();

    // 1. Resolve input: embedding (head) or provided activations (middle/tail).
    let (hidden_state, owns_hidden) = if node_config.is_head() {
        if input_hidden_states.is_some() {
            return Err(FractureError::Pipeline(
                "head node received activations in batched_forward_node".into(),
            ));
        }
        let mut all_token_ids: Vec<u32> = Vec::new();
        for seq in sequences {
            all_token_ids.extend_from_slice(&seq.token_ids);
        }
        if all_token_ids.len() != total_tokens {
            return Err(FractureError::InvalidShape(format!(
                "token count mismatch: {} token_ids vs {} positions",
                all_token_ids.len(),
                total_tokens
            )));
        }
        let hs = backend.alloc(&[total_tokens, hidden], DType::FP16)?;
        backend.embedding(&all_token_ids, &weights.token_embedding, &hs)?;
        (hs, true)
    } else {
        match input_hidden_states {
            Some(hs) => {
                if hs.shape[0] != total_tokens || hs.shape[1] != hidden {
                    return Err(FractureError::InvalidShape(format!(
                        "activation shape [{}, {}] doesn't match expected [{}, {}]",
                        hs.shape[0], hs.shape[1], total_tokens, hidden
                    )));
                }
                (hs, false)
            }
            None => {
                return Err(FractureError::Pipeline(
                    "non-head node requires activations in batched_forward_node".into(),
                ));
            }
        }
    };

    // Pre-compute start_pos for each sequence (tokens already in cache).
    let start_positions: Vec<usize> = sequences
        .iter()
        .map(|s| cache.seq_len(s.handle))
        .collect::<Result<Vec<_>>>()?;

    // Scratch tensors — allocated once, reused across layers.
    let normed = backend.alloc(&[total_tokens, hidden], DType::FP16)?;
    let q_flat = backend.alloc(&[total_tokens, hidden], DType::FP16)?;
    let k_flat = backend.alloc(&[total_tokens, num_kv_heads * head_dim], DType::FP16)?;
    let v_flat = backend.alloc(&[total_tokens, num_kv_heads * head_dim], DType::FP16)?;
    let attn_out_buf = backend.alloc(&[total_tokens, hidden], DType::FP16)?;
    let projected = backend.alloc(&[total_tokens, hidden], DType::FP16)?;
    let gate = backend.alloc(&[total_tokens, intermediate], DType::FP16)?;
    let up = backend.alloc(&[total_tokens, intermediate], DType::FP16)?;
    let ffn_mid = backend.alloc(&[total_tokens, intermediate], DType::FP16)?;
    let ffn_out = backend.alloc(&[total_tokens, hidden], DType::FP16)?;

    // 2. Layer loop (uses node_config's range, weight indexing from engine's range).
    let exec_range = &node_config.layer_range;
    for layer_idx in exec_range.clone() {
        let weight_idx = layer_idx - exec_range.start;
        let cache_idx = weight_idx;
        let w = &weights.layers[weight_idx];

        // 2a. RMSNorm — batched
        backend.rmsnorm(&hidden_state, &w.attn_norm, cfg.rms_norm_eps, &normed)?;

        // 2b. QKV projections — batched
        backend.matmul(&normed, &w.q_proj, &q_flat)?;
        backend.matmul(&normed, &w.k_proj, &k_flat)?;
        backend.matmul(&normed, &w.v_proj, &v_flat)?;

        // 2c-d. Reshape + RoPE — batched
        let q_mh = DeviceTensor::new(
            q_flat.id,
            vec![total_tokens, num_q_heads, head_dim],
            DType::FP16,
        );
        let k_mh = DeviceTensor::new(
            k_flat.id,
            vec![total_tokens, num_kv_heads, head_dim],
            DType::FP16,
        );
        let v_mh = DeviceTensor::new(
            v_flat.id,
            vec![total_tokens, num_kv_heads, head_dim],
            DType::FP16,
        );

        backend.rope(&q_mh, &k_mh, &all_positions, cfg.rope_theta, head_dim)?;

        // 2e. KV cache append — PER-SEQUENCE
        for (i, seq) in sequences.iter().enumerate() {
            let (start, end) = seq_boundaries[i];
            let n = end - start;

            let seq_k = DeviceTensor::new(
                k_mh.id,
                vec![total_tokens, num_kv_heads, head_dim],
                DType::FP16,
            );
            let seq_v = DeviceTensor::new(
                v_mh.id,
                vec![total_tokens, num_kv_heads, head_dim],
                DType::FP16,
            );

            let k_slice = backend.alloc(&[n, num_kv_heads, head_dim], DType::FP16)?;
            let v_slice = backend.alloc(&[n, num_kv_heads, head_dim], DType::FP16)?;
            backend.copy_rows(&seq_k, &k_slice, start, 0, n)?;
            backend.copy_rows(&seq_v, &v_slice, start, 0, n)?;

            cache.append_kv(seq.handle, cache_idx, &k_slice, &v_slice, backend)?;

            backend.free(&k_slice)?;
            backend.free(&v_slice)?;
        }

        // 2f. Attention — PER-SEQUENCE
        for (i, seq) in sequences.iter().enumerate() {
            let (start, end) = seq_boundaries[i];
            let n = end - start;
            let start_pos = start_positions[i];
            let new_seq_len = start_pos + n;

            let q_slice = backend.alloc(&[n, num_q_heads, head_dim], DType::FP16)?;
            backend.copy_rows(&q_mh, &q_slice, start, 0, n)?;

            let attn_slice = backend.alloc(&[n, num_q_heads, head_dim], DType::FP16)?;

            let block_table = cache.block_table(seq.handle)?;
            let block_table_i32: Vec<i32> = block_table.iter().map(|&b| b as i32).collect();

            let pool = cache.pool();
            let k_blocks: Vec<&DeviceTensor> = (0..pool.capacity())
                .map(|bid| pool.k_tensor(bid, cache_idx))
                .collect();
            let v_blocks: Vec<&DeviceTensor> = (0..pool.capacity())
                .map(|bid| pool.v_tensor(bid, cache_idx))
                .collect();

            backend.attention_paged(
                &q_slice,
                &block_table_i32,
                &k_blocks,
                &v_blocks,
                num_kv_heads,
                new_seq_len,
                start_pos,
                &attn_slice,
            )?;

            let attn_out_mh = DeviceTensor::new(
                attn_out_buf.id,
                vec![total_tokens, num_q_heads, head_dim],
                DType::FP16,
            );
            backend.copy_rows(&attn_slice, &attn_out_mh, 0, start, n)?;

            backend.free(&q_slice)?;
            backend.free(&attn_slice)?;
        }

        // 2g. Output projection — batched
        let attn_out_flat = DeviceTensor::new(
            attn_out_buf.id,
            vec![total_tokens, hidden],
            DType::FP16,
        );
        backend.matmul(&attn_out_flat, &w.o_proj, &projected)?;

        // 2h. Residual — batched
        backend.add(&hidden_state, &projected, &hidden_state)?;

        // 2i-k. FFN — batched
        backend.rmsnorm(&hidden_state, &w.ffn_norm, cfg.rms_norm_eps, &normed)?;
        backend.matmul(&normed, &w.gate_proj, &gate)?;
        backend.matmul(&normed, &w.up_proj, &up)?;
        backend.silu_mul(&gate, &up, &ffn_mid)?;
        backend.matmul(&ffn_mid, &w.down_proj, &ffn_out)?;
        backend.add(&hidden_state, &ffn_out, &hidden_state)?;
    }

    // 3. Output phase: tail produces logits, head/middle produce activations.
    let free_scratch = |backend: &B| -> Result<()> {
        backend.free(&normed)?;
        backend.free(&q_flat)?;
        backend.free(&k_flat)?;
        backend.free(&v_flat)?;
        backend.free(&attn_out_buf)?;
        backend.free(&projected)?;
        backend.free(&gate)?;
        backend.free(&up)?;
        backend.free(&ffn_mid)?;
        backend.free(&ffn_out)?;
        Ok(())
    };

    if node_config.is_tail() {
        // Final RMSNorm + LM head — extract last token per sequence
        backend.rmsnorm(&hidden_state, &weights.output_norm, cfg.rms_norm_eps, &normed)?;

        let mut per_seq_logits = Vec::with_capacity(sequences.len());

        for (i, _seq) in sequences.iter().enumerate() {
            let (_start, end) = seq_boundaries[i];
            let last_idx = end - 1;

            let last_hidden = backend.alloc(&[1, hidden], DType::FP16)?;
            backend.copy_rows(&normed, &last_hidden, last_idx, 0, 1)?;

            let logits_tensor = backend.alloc(&[1, cfg.vocab_size], DType::FP16)?;
            backend.matmul(&last_hidden, &weights.lm_head, &logits_tensor)?;

            let mut logits_fp16 = vec![0u8; cfg.vocab_size * 2];
            backend.copy_to_host(&logits_tensor, &mut logits_fp16)?;

            let logits: Vec<f32> = logits_fp16
                .chunks_exact(2)
                .map(|bytes| {
                    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect();

            per_seq_logits.push(logits);

            backend.free(&last_hidden)?;
            backend.free(&logits_tensor)?;
        }

        backend.synchronize()?;

        if owns_hidden {
            backend.free(&hidden_state)?;
        }
        free_scratch(backend)?;

        // Flatten per-sequence logits into a single Vec<f32> for NodeOutput::Logits
        let flat_logits: Vec<f32> = per_seq_logits.into_iter().flatten().collect();
        Ok(NodeOutput::Logits(flat_logits))
    } else {
        // Head/middle: return hidden state as activations
        backend.synchronize()?;
        free_scratch(backend)?;
        Ok(NodeOutput::Activations(hidden_state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fracture_core::{DeviceTimer, TensorId};
    use std::sync::atomic::{AtomicU64, Ordering};

    struct MockBackend {
        next_id: AtomicU64,
    }

    impl MockBackend {
        fn new() -> Self {
            Self { next_id: AtomicU64::new(1) }
        }
    }

    impl Backend for MockBackend {
        fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
        }
        fn free(&self, _t: &DeviceTensor) -> Result<()> { Ok(()) }
        fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> { Ok(()) }
        fn copy_to_host(&self, _src: &DeviceTensor, dst: &mut [u8]) -> Result<()> {
            // Zero-fill — greedy sampling will pick token 0
            dst.fill(0);
            Ok(())
        }
        fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn rmsnorm(&self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _positions: &[u32], _theta: f64, _head_dim: usize) -> Result<()> { Ok(()) }
        fn attention(&self, _q: &DeviceTensor, _k_cache: &DeviceTensor, _v_cache: &DeviceTensor, _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn attention_paged(&self, _q: &DeviceTensor, _bt: &[i32], _kb: &[&DeviceTensor], _vb: &[&DeviceTensor], _nkv: usize, _kvl: usize, _sp: usize, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn silu_mul(&self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn embedding(&self, _token_ids: &[u32], _embedding_table: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn copy_rows(&self, _src: &DeviceTensor, _dst: &DeviceTensor, _src_offset: usize, _dst_offset: usize, _count: usize) -> Result<()> { Ok(()) }
        fn device_name(&self) -> &str { "mock" }
        fn total_memory(&self) -> usize { 1 << 30 }
        fn available_memory(&self) -> usize { 1 << 30 }
        fn synchronize(&self) -> Result<()> { Ok(()) }
        fn create_timer(&self) -> Result<DeviceTimer> { Ok(DeviceTimer(0)) }
        fn start_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }
        fn stop_timer(&self, _timer: &DeviceTimer) -> Result<f32> { Ok(0.0) }
        fn destroy_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }
    }

    fn tiny_config() -> fracture_core::ModelConfig {
        fracture_core::ModelConfig {
            hidden_size: 8,
            num_layers: 2,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            intermediate_size: 16,
            vocab_size: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            max_seq_len: 64,
        }
    }

    fn fake_weights(backend: &MockBackend) -> WeightStore {
        let cfg = tiny_config();
        let h = cfg.hidden_size;
        let kv = cfg.num_kv_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;
        let v = cfg.vocab_size;

        let mut layers = Vec::new();
        for _ in 0..cfg.num_layers {
            layers.push(fracture_gguf::LayerWeights {
                q_proj: backend.alloc(&[h, h], DType::FP16).unwrap(),
                k_proj: backend.alloc(&[kv, h], DType::FP16).unwrap(),
                v_proj: backend.alloc(&[kv, h], DType::FP16).unwrap(),
                o_proj: backend.alloc(&[h, h], DType::FP16).unwrap(),
                gate_proj: backend.alloc(&[inter, h], DType::FP16).unwrap(),
                up_proj: backend.alloc(&[inter, h], DType::FP16).unwrap(),
                down_proj: backend.alloc(&[h, inter], DType::FP16).unwrap(),
                attn_norm: backend.alloc(&[h], DType::FP16).unwrap(),
                ffn_norm: backend.alloc(&[h], DType::FP16).unwrap(),
            });
        }

        WeightStore {
            config: cfg,
            token_embedding: backend.alloc(&[v, h], DType::FP16).unwrap(),
            layers,
            output_norm: backend.alloc(&[h], DType::FP16).unwrap(),
            lm_head: backend.alloc(&[v, h], DType::FP16).unwrap(),
        }
    }

    #[test]
    fn test_batched_forward_empty() {
        let backend = MockBackend::new();
        let weights = fake_weights(&backend);
        let mut cache = PagedKvCacheManager::new(
            10, weights.config.num_layers, weights.config.num_kv_heads,
            weights.config.head_dim, &backend,
        ).unwrap();

        let result = batched_forward(
            &backend, &weights, &(0..weights.config.num_layers), &mut cache, &[],
        ).unwrap();
        assert!(result.logits.is_empty());
    }

    #[test]
    fn test_batched_forward_single_sequence() {
        let backend = MockBackend::new();
        let weights = fake_weights(&backend);
        let cfg = &weights.config;
        let mut cache = PagedKvCacheManager::new(
            20, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, &backend,
        ).unwrap();

        let h = cache.alloc().unwrap();
        let seqs = vec![SequenceSlice {
            handle: h,
            token_ids: vec![1, 2, 3],
            positions: vec![0, 1, 2],
        }];

        let result = batched_forward(
            &backend, &weights, &(0..cfg.num_layers), &mut cache, &seqs,
        ).unwrap();
        assert_eq!(result.logits.len(), 1);
        assert_eq!(result.logits[0].len(), cfg.vocab_size);

        cache.free(h).unwrap();
    }

    #[test]
    fn test_batched_forward_multiple_sequences() {
        let backend = MockBackend::new();
        let weights = fake_weights(&backend);
        let cfg = &weights.config;
        let mut cache = PagedKvCacheManager::new(
            50, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, &backend,
        ).unwrap();

        let h1 = cache.alloc().unwrap();
        let h2 = cache.alloc().unwrap();
        let h3 = cache.alloc().unwrap();

        let seqs = vec![
            SequenceSlice { handle: h1, token_ids: vec![1, 2, 3], positions: vec![0, 1, 2] },
            SequenceSlice { handle: h2, token_ids: vec![10], positions: vec![5] }, // decode step
            SequenceSlice { handle: h3, token_ids: vec![20, 21], positions: vec![0, 1] },
        ];

        let result = batched_forward(
            &backend, &weights, &(0..cfg.num_layers), &mut cache, &seqs,
        ).unwrap();
        assert_eq!(result.logits.len(), 3);
        for logits in &result.logits {
            assert_eq!(logits.len(), cfg.vocab_size);
        }

        cache.free(h1).unwrap();
        cache.free(h2).unwrap();
        cache.free(h3).unwrap();
    }

    // ── batched_forward_node tests ──────────────────────────────────────

    #[test]
    fn test_batched_forward_node_head_returns_activations() {
        let backend = MockBackend::new();
        let weights = fake_weights(&backend);
        let cfg = &weights.config;
        let node_config = NodeConfig::new(0..1, cfg.num_layers).unwrap();
        let mut cache = PagedKvCacheManager::new(
            20, 1, cfg.num_kv_heads, cfg.head_dim, &backend,
        ).unwrap();

        let h = cache.alloc().unwrap();
        let seqs = vec![SequenceSlice {
            handle: h,
            token_ids: vec![1, 2, 3],
            positions: vec![0, 1, 2],
        }];

        let result = batched_forward_node(
            &backend, &weights, &node_config, &mut cache, &seqs, None,
        ).unwrap();
        match result {
            NodeOutput::Activations(t) => {
                assert_eq!(t.shape, vec![3, cfg.hidden_size]);
            }
            NodeOutput::Logits(_) => panic!("head node should return activations"),
        }

        cache.free(h).unwrap();
    }

    #[test]
    fn test_batched_forward_node_tail_returns_logits() {
        let backend = MockBackend::new();
        let weights = fake_weights(&backend);
        let cfg = &weights.config;
        // Tail node: last layer only
        let node_config = NodeConfig::new(1..2, cfg.num_layers).unwrap();
        let mut cache = PagedKvCacheManager::new(
            20, 1, cfg.num_kv_heads, cfg.head_dim, &backend,
        ).unwrap();

        let h = cache.alloc().unwrap();
        let seqs = vec![SequenceSlice {
            handle: h,
            token_ids: vec![], // tail doesn't use token_ids
            positions: vec![0, 1],
        }];

        // Provide activations as if from a previous node
        let hidden = backend.alloc(&[2, cfg.hidden_size], DType::FP16).unwrap();
        let result = batched_forward_node(
            &backend, &weights, &node_config, &mut cache, &seqs, Some(hidden),
        ).unwrap();
        match result {
            NodeOutput::Logits(logits) => {
                assert_eq!(logits.len(), cfg.vocab_size);
            }
            NodeOutput::Activations(_) => panic!("tail node should return logits"),
        }

        cache.free(h).unwrap();
    }

    #[test]
    fn test_batched_forward_node_full_model() {
        let backend = MockBackend::new();
        let weights = fake_weights(&backend);
        let cfg = &weights.config;
        // Full model: all layers
        let node_config = NodeConfig::new(0..cfg.num_layers, cfg.num_layers).unwrap();
        let mut cache = PagedKvCacheManager::new(
            50, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, &backend,
        ).unwrap();

        let h1 = cache.alloc().unwrap();
        let h2 = cache.alloc().unwrap();
        let seqs = vec![
            SequenceSlice { handle: h1, token_ids: vec![1, 2, 3], positions: vec![0, 1, 2] },
            SequenceSlice { handle: h2, token_ids: vec![10], positions: vec![0] },
        ];

        let result = batched_forward_node(
            &backend, &weights, &node_config, &mut cache, &seqs, None,
        ).unwrap();
        match result {
            NodeOutput::Logits(logits) => {
                // Two sequences, each producing vocab_size logits, flattened
                assert_eq!(logits.len(), 2 * cfg.vocab_size);
            }
            NodeOutput::Activations(_) => panic!("full model should return logits"),
        }

        cache.free(h1).unwrap();
        cache.free(h2).unwrap();
    }

    #[test]
    fn test_batched_forward_node_head_rejects_activations() {
        let backend = MockBackend::new();
        let weights = fake_weights(&backend);
        let cfg = &weights.config;
        let node_config = NodeConfig::new(0..1, cfg.num_layers).unwrap();
        let mut cache = PagedKvCacheManager::new(
            10, 1, cfg.num_kv_heads, cfg.head_dim, &backend,
        ).unwrap();

        let h = cache.alloc().unwrap();
        let seqs = vec![SequenceSlice {
            handle: h,
            token_ids: vec![1],
            positions: vec![0],
        }];

        let hidden = backend.alloc(&[1, cfg.hidden_size], DType::FP16).unwrap();
        let result = batched_forward_node(
            &backend, &weights, &node_config, &mut cache, &seqs, Some(hidden),
        );
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("head node received activations"));

        cache.free(h).unwrap();
    }

    #[test]
    fn test_batched_forward_node_nonhead_requires_activations() {
        let backend = MockBackend::new();
        let weights = fake_weights(&backend);
        let cfg = &weights.config;
        let node_config = NodeConfig::new(1..2, cfg.num_layers).unwrap();
        let mut cache = PagedKvCacheManager::new(
            10, 1, cfg.num_kv_heads, cfg.head_dim, &backend,
        ).unwrap();

        let h = cache.alloc().unwrap();
        let seqs = vec![SequenceSlice {
            handle: h,
            token_ids: vec![],
            positions: vec![0],
        }];

        let result = batched_forward_node(
            &backend, &weights, &node_config, &mut cache, &seqs, None,
        );
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("non-head node requires activations"));

        cache.free(h).unwrap();
    }

    #[test]
    fn test_batched_forward_node_empty_sequences_rejected() {
        let backend = MockBackend::new();
        let weights = fake_weights(&backend);
        let cfg = &weights.config;
        let node_config = NodeConfig::new(0..cfg.num_layers, cfg.num_layers).unwrap();
        let mut cache = PagedKvCacheManager::new(
            10, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, &backend,
        ).unwrap();

        let result = batched_forward_node(
            &backend, &weights, &node_config, &mut cache, &[], None,
        );
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("no sequences"));
    }

    // ── FailingMockBackend — fails on matmul ───────────────────────

    struct FailingMockBackend {
        next_id: AtomicU64,
    }

    impl FailingMockBackend {
        fn new() -> Self {
            Self { next_id: AtomicU64::new(1) }
        }
    }

    impl Backend for FailingMockBackend {
        fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
        }
        fn free(&self, _t: &DeviceTensor) -> Result<()> { Ok(()) }
        fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> { Ok(()) }
        fn copy_to_host(&self, _src: &DeviceTensor, dst: &mut [u8]) -> Result<()> {
            dst.fill(0);
            Ok(())
        }
        fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> {
            Err(fracture_core::FractureError::Backend("induced matmul failure".into()))
        }
        fn rmsnorm(&self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _positions: &[u32], _theta: f64, _head_dim: usize) -> Result<()> { Ok(()) }
        fn attention(&self, _q: &DeviceTensor, _k_cache: &DeviceTensor, _v_cache: &DeviceTensor, _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn attention_paged(&self, _q: &DeviceTensor, _bt: &[i32], _kb: &[&DeviceTensor], _vb: &[&DeviceTensor], _nkv: usize, _kvl: usize, _sp: usize, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn silu_mul(&self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn embedding(&self, _token_ids: &[u32], _embedding_table: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn copy_rows(&self, _src: &DeviceTensor, _dst: &DeviceTensor, _src_offset: usize, _dst_offset: usize, _count: usize) -> Result<()> { Ok(()) }
        fn device_name(&self) -> &str { "failing-mock" }
        fn total_memory(&self) -> usize { 1 << 30 }
        fn available_memory(&self) -> usize { 1 << 30 }
        fn synchronize(&self) -> Result<()> { Ok(()) }
        fn create_timer(&self) -> Result<DeviceTimer> { Ok(DeviceTimer(0)) }
        fn start_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }
        fn stop_timer(&self, _timer: &DeviceTimer) -> Result<f32> { Ok(0.0) }
        fn destroy_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }
    }

    /// batched_forward propagates a backend matmul error as FractureError::Backend.
    #[test]
    fn test_batched_forward_backend_error_propagation() {
        let failing = FailingMockBackend::new();
        // Use the non-failing backend to build weights (need valid alloc),
        // then run with the failing backend.
        let good = MockBackend::new();
        let weights = fake_weights(&good);
        let cfg = &weights.config;

        // Build cache with the failing backend (alloc succeeds, matmul fails later).
        let mut cache = PagedKvCacheManager::new(
            20, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, &failing,
        ).unwrap();

        let h = cache.alloc().unwrap();
        let seqs = vec![SequenceSlice {
            handle: h,
            token_ids: vec![1, 2],
            positions: vec![0, 1],
        }];

        let result = batched_forward(
            &failing, &weights, &(0..cfg.num_layers), &mut cache, &seqs,
        );
        assert!(result.is_err(), "expected error from failing matmul");
        let err = result.err().unwrap();
        assert!(
            matches!(err, fracture_core::FractureError::Backend(_)),
            "expected Backend error, got: {err:?}"
        );
        assert!(err.to_string().contains("induced matmul failure"));
    }

    /// batched_forward_node propagates a backend matmul error as FractureError::Backend.
    #[test]
    fn test_batched_forward_node_backend_error_propagation() {
        let failing = FailingMockBackend::new();
        let good = MockBackend::new();
        let weights = fake_weights(&good);
        let cfg = &weights.config;

        // Full model node config — head node embeds then runs layers.
        let node_config = NodeConfig::new(0..cfg.num_layers, cfg.num_layers).unwrap();

        let mut cache = PagedKvCacheManager::new(
            20, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, &failing,
        ).unwrap();

        let h = cache.alloc().unwrap();
        let seqs = vec![SequenceSlice {
            handle: h,
            token_ids: vec![1],
            positions: vec![0],
        }];

        let result = batched_forward_node(
            &failing, &weights, &node_config, &mut cache, &seqs, None,
        );
        assert!(result.is_err(), "expected error from failing matmul");
        let err = result.err().unwrap();
        assert!(
            matches!(err, fracture_core::FractureError::Backend(_)),
            "expected Backend error, got: {err:?}"
        );
        assert!(err.to_string().contains("induced matmul failure"));
    }

    /// Providing a DeviceTensor with wrong shape to a non-head node returns an
    /// error containing "activation shape".
    #[test]
    fn test_batched_forward_node_activation_shape_mismatch() {
        let backend = MockBackend::new();
        let weights = fake_weights(&backend);
        let cfg = &weights.config;

        // Tail node: last layer only, expects activations.
        let node_config = NodeConfig::new(1..2, cfg.num_layers).unwrap();
        let mut cache = PagedKvCacheManager::new(
            20, 1, cfg.num_kv_heads, cfg.head_dim, &backend,
        ).unwrap();

        let h = cache.alloc().unwrap();
        let seqs = vec![SequenceSlice {
            handle: h,
            token_ids: vec![],
            positions: vec![0, 1], // 2 tokens
        }];

        // Wrong shape: expect [2, hidden_size] but provide [3, hidden_size].
        let wrong_shape = backend.alloc(&[3, cfg.hidden_size], DType::FP16).unwrap();

        let result = batched_forward_node(
            &backend, &weights, &node_config, &mut cache, &seqs, Some(wrong_shape),
        );
        assert!(result.is_err(), "expected shape mismatch error");
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("activation shape"),
            "error should mention 'activation shape', got: {msg}"
        );

        cache.free(h).unwrap();
    }

    /// Providing SequenceSlice where token_ids.len() != positions.len() for a head
    /// node returns an error containing "token count mismatch".
    #[test]
    fn test_batched_forward_node_token_position_count_mismatch() {
        let backend = MockBackend::new();
        let weights = fake_weights(&backend);
        let cfg = &weights.config;

        // Head node (full model) — token_ids are used.
        let node_config = NodeConfig::new(0..cfg.num_layers, cfg.num_layers).unwrap();
        let mut cache = PagedKvCacheManager::new(
            20, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, &backend,
        ).unwrap();

        let h = cache.alloc().unwrap();
        let seqs = vec![SequenceSlice {
            handle: h,
            token_ids: vec![1, 2, 3], // 3 tokens
            positions: vec![0, 1],    // 2 positions — mismatch
        }];

        let result = batched_forward_node(
            &backend, &weights, &node_config, &mut cache, &seqs, None,
        );
        assert!(result.is_err(), "expected token count mismatch error");
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("token count mismatch"),
            "error should mention 'token count mismatch', got: {msg}"
        );

        cache.free(h).unwrap();
    }

    /// batched_forward returns FractureError::OutOfMemory when the paged block pool
    /// is exhausted mid-batch.
    #[test]
    fn test_batched_forward_cache_oom() {
        let backend = MockBackend::new();
        let weights = fake_weights(&backend);
        let cfg = &weights.config;

        // Only 3 blocks: each alloc() takes 1 block, so 3 sequences use all 3.
        // Appending tokens to any of them requires a new block — but pool is empty.
        let mut cache = PagedKvCacheManager::new(
            3, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, &backend,
        ).unwrap();

        let h1 = cache.alloc().unwrap(); // 1 block
        let h2 = cache.alloc().unwrap(); // 1 block
        let h3 = cache.alloc().unwrap(); // 1 block — pool now empty
        assert_eq!(cache.num_free_blocks(), 0);

        // Each sequence provides BLOCK_SIZE + 1 tokens, which fills the initial block
        // completely and then needs one more block — but the pool is empty → OOM.
        use crate::paged_kv_cache::BLOCK_SIZE;
        let tokens: Vec<u32> = (0..BLOCK_SIZE as u32 + 1).collect();
        let positions: Vec<u32> = (0..BLOCK_SIZE as u32 + 1).collect();

        let seqs = vec![
            SequenceSlice { handle: h1, token_ids: tokens.clone(), positions: positions.clone() },
            SequenceSlice { handle: h2, token_ids: tokens.clone(), positions: positions.clone() },
            SequenceSlice { handle: h3, token_ids: tokens.clone(), positions: positions.clone() },
        ];

        let result = batched_forward(
            &backend, &weights, &(0..cfg.num_layers), &mut cache, &seqs,
        );
        assert!(result.is_err(), "expected OOM error");
        assert!(
            matches!(result.err().unwrap(), fracture_core::FractureError::OutOfMemory { .. }),
            "expected OutOfMemory error"
        );
    }
}
