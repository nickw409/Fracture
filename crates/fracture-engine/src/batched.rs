use crate::kv_cache::CacheHandle;
use crate::paged_kv_cache::PagedKvCacheManager;
use fracture_core::{Backend, DType, DeviceTensor, Result};
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
}
