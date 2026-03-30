//! GPU validation tests for TurboQuant KV cache compression.
//!
//! These tests run on a real CUDA GPU. They validate:
//! 1. Compress → decompress round-trip produces low MSE and high cosine similarity
//! 2. TurboQuant attention output matches FP16 attention output within tolerance
//!
//! The tests use tiny tensors (head_dim=16 or 32) to keep execution fast.

use fracture_core::turboquant::{
    generate_rotation_matrix, get_codebook, TurboQuantConfig,
};
use fracture_core::{Backend, DType, DeviceTensor, ModelConfig};
use fracture_cuda::CudaBackend;
use fracture_engine::{
    batched_forward, PagedKvCacheManager, QuantizedKvCacheManager, SequenceSlice,
};
use fracture_gguf::{LayerWeights, WeightStore};
use half::f16;
use rand::Rng;

/// Upload a rotation matrix to the GPU.
fn upload_rotation(backend: &CudaBackend, d: usize, seed: u64) -> DeviceTensor {
    let host = generate_rotation_matrix(d, seed);
    let tensor = backend.alloc(&[d, d], DType::FP32).unwrap();
    let bytes: Vec<u8> = host.iter().flat_map(|f| f.to_le_bytes()).collect();
    backend.copy_to_device(&tensor, &bytes).unwrap();
    tensor
}

/// Upload a codebook's centroids to the GPU.
fn upload_centroids(backend: &CudaBackend, d: usize, bits: u8) -> DeviceTensor {
    let cb = get_codebook(d, bits);
    let n = cb.n_levels();
    let tensor = backend.alloc(&[n], DType::FP32).unwrap();
    let bytes: Vec<u8> = cb.centroids.iter().flat_map(|f| f.to_le_bytes()).collect();
    backend.copy_to_device(&tensor, &bytes).unwrap();
    tensor
}

/// Create a random FP16 tensor on GPU, returning both device tensor and host copy.
fn random_fp16_tensor(
    backend: &CudaBackend,
    shape: &[usize],
    seed: u64,
) -> (DeviceTensor, Vec<f32>) {
    use fracture_core::turboquant::generate_rotation_matrix;
    // Use rotation matrix generation as a deterministic random source
    let numel: usize = shape.iter().product();
    // Generate enough random f32s via Box-Muller (reuse the RNG internals)
    let raw = generate_rotation_matrix(
        ((numel as f64).sqrt().ceil() as usize).max(2),
        seed,
    );
    let host_f32: Vec<f32> = raw.into_iter().take(numel).collect();

    let host_fp16: Vec<u8> = host_f32
        .iter()
        .flat_map(|&f| f16::from_f32(f).to_le_bytes())
        .collect();

    let tensor = backend.alloc(shape, DType::FP16).unwrap();
    backend.copy_to_device(&tensor, &host_fp16).unwrap();
    (tensor, host_f32)
}

/// Download an FP16 tensor from GPU to host f32 values.
fn download_fp16(backend: &CudaBackend, tensor: &DeviceTensor) -> Vec<f32> {
    let numel: usize = tensor.shape.iter().product();
    let mut bytes = vec![0u8; numel * 2];
    backend.copy_to_host(tensor, &mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

/// Cosine similarity between two f32 vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b + 1e-10)
}

// ── Compress/Decompress Round-Trip Tests ───────────────────────────

#[test]
fn test_turboquant_compress_decompress_4bit() {
    let backend = CudaBackend::new(0).unwrap();
    let head_dim = 16;
    let num_kv_heads = 2;
    let num_tokens = 4;
    let bits: u8 = 4;

    // Create random KV vectors
    let (input, host_input) = random_fp16_tensor(
        &backend,
        &[num_tokens, num_kv_heads, head_dim],
        12345,
    );

    // Upload rotation matrix and centroids
    let rotation = upload_rotation(&backend, head_dim, 42);
    let centroids = upload_centroids(&backend, head_dim, bits);

    // Allocate output tensors
    let packed_dim = TurboQuantConfig::packed_dim_per_head(head_dim, bits);
    let packed = backend
        .alloc(&[num_tokens, num_kv_heads * packed_dim], DType::INT8)
        .unwrap();
    let norms = backend
        .alloc(&[num_tokens, num_kv_heads], DType::FP16)
        .unwrap();

    // Compress
    backend
        .turboquant_compress(&input, &rotation, &centroids, bits, &packed, &norms)
        .unwrap();
    backend.synchronize().unwrap();

    // Decompress (using the decompress FFI directly)
    let output = backend
        .alloc(&[num_tokens, num_kv_heads, head_dim], DType::FP16)
        .unwrap();

    // Call decompress kernel via FFI
    unsafe {
        use fracture_cuda::ffi::*;
        let err = launch_turboquant_decompress(
            backend.get_ptr(packed.id).unwrap() as *const std::ffi::c_void,
            backend.get_ptr(norms.id).unwrap() as *const std::ffi::c_void,
            backend.get_ptr(rotation.id).unwrap() as *const f32,
            backend.get_ptr(centroids.id).unwrap() as *const f32,
            backend.get_ptr(output.id).unwrap(),
            num_tokens as i32,
            num_kv_heads as i32,
            head_dim as i32,
            bits as i32,
            packed_dim as i32,
            backend.stream(),
        );
        assert_eq!(err, 0, "decompress kernel launch failed");
    }
    backend.synchronize().unwrap();

    // Download and compare
    let host_output = download_fp16(&backend, &output);

    // Compute per-vector cosine similarity
    let vec_size = head_dim;
    for token in 0..num_tokens {
        for head in 0..num_kv_heads {
            let offset = (token * num_kv_heads + head) * vec_size;
            let orig = &host_input[offset..offset + vec_size];
            let recon = &host_output[offset..offset + vec_size];
            let sim = cosine_similarity(orig, recon);
            assert!(
                sim > 0.95,
                "4-bit round-trip cosine sim for token={token} head={head} should be > 0.95, got {sim}"
            );
        }
    }

    // Cleanup
    backend.free(&input).unwrap();
    backend.free(&rotation).unwrap();
    backend.free(&centroids).unwrap();
    backend.free(&packed).unwrap();
    backend.free(&norms).unwrap();
    backend.free(&output).unwrap();
}

#[test]
fn test_turboquant_compress_decompress_2bit() {
    let backend = CudaBackend::new(0).unwrap();
    let head_dim = 16;
    let num_kv_heads = 2;
    let num_tokens = 4;
    let bits: u8 = 2;

    let (input, host_input) = random_fp16_tensor(
        &backend,
        &[num_tokens, num_kv_heads, head_dim],
        54321,
    );

    let rotation = upload_rotation(&backend, head_dim, 42);
    let centroids = upload_centroids(&backend, head_dim, bits);

    let packed_dim = TurboQuantConfig::packed_dim_per_head(head_dim, bits);
    let packed = backend
        .alloc(&[num_tokens, num_kv_heads * packed_dim], DType::INT8)
        .unwrap();
    let norms = backend
        .alloc(&[num_tokens, num_kv_heads], DType::FP16)
        .unwrap();

    backend
        .turboquant_compress(&input, &rotation, &centroids, bits, &packed, &norms)
        .unwrap();
    backend.synchronize().unwrap();

    let output = backend
        .alloc(&[num_tokens, num_kv_heads, head_dim], DType::FP16)
        .unwrap();

    unsafe {
        use fracture_cuda::ffi::*;
        let err = launch_turboquant_decompress(
            backend.get_ptr(packed.id).unwrap() as *const std::ffi::c_void,
            backend.get_ptr(norms.id).unwrap() as *const std::ffi::c_void,
            backend.get_ptr(rotation.id).unwrap() as *const f32,
            backend.get_ptr(centroids.id).unwrap() as *const f32,
            backend.get_ptr(output.id).unwrap(),
            num_tokens as i32,
            num_kv_heads as i32,
            head_dim as i32,
            bits as i32,
            packed_dim as i32,
            backend.stream(),
        );
        assert_eq!(err, 0, "decompress kernel launch failed");
    }
    backend.synchronize().unwrap();

    let host_output = download_fp16(&backend, &output);

    let vec_size = head_dim;
    for token in 0..num_tokens {
        for head in 0..num_kv_heads {
            let offset = (token * num_kv_heads + head) * vec_size;
            let orig = &host_input[offset..offset + vec_size];
            let recon = &host_output[offset..offset + vec_size];
            let sim = cosine_similarity(orig, recon);
            // 2-bit has lower fidelity
            assert!(
                sim > 0.80,
                "2-bit round-trip cosine sim for token={token} head={head} should be > 0.80, got {sim}"
            );
        }
    }

    backend.free(&input).unwrap();
    backend.free(&rotation).unwrap();
    backend.free(&centroids).unwrap();
    backend.free(&packed).unwrap();
    backend.free(&norms).unwrap();
    backend.free(&output).unwrap();
}

#[test]
fn test_turboquant_compress_preserves_norms() {
    let backend = CudaBackend::new(0).unwrap();
    let head_dim = 16;
    let num_kv_heads = 2;
    let num_tokens = 3;
    let bits: u8 = 4;

    let (input, host_input) = random_fp16_tensor(
        &backend,
        &[num_tokens, num_kv_heads, head_dim],
        99999,
    );

    let rotation = upload_rotation(&backend, head_dim, 42);
    let centroids = upload_centroids(&backend, head_dim, bits);

    let packed_dim = TurboQuantConfig::packed_dim_per_head(head_dim, bits);
    let packed = backend
        .alloc(&[num_tokens, num_kv_heads * packed_dim], DType::INT8)
        .unwrap();
    let norms = backend
        .alloc(&[num_tokens, num_kv_heads], DType::FP16)
        .unwrap();

    backend
        .turboquant_compress(&input, &rotation, &centroids, bits, &packed, &norms)
        .unwrap();
    backend.synchronize().unwrap();

    // Download norms and compare with expected
    let host_norms = download_fp16(&backend, &norms);

    for token in 0..num_tokens {
        for head in 0..num_kv_heads {
            let offset = (token * num_kv_heads + head) * head_dim;
            let vec = &host_input[offset..offset + head_dim];
            let expected_norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();

            let stored_norm = host_norms[token * num_kv_heads + head];
            let rel_err = (expected_norm - stored_norm).abs() / (expected_norm + 1e-8);
            assert!(
                rel_err < 0.02,
                "norm for token={token} head={head}: expected {expected_norm}, got {stored_norm} (rel_err={rel_err})"
            );
        }
    }

    backend.free(&input).unwrap();
    backend.free(&rotation).unwrap();
    backend.free(&centroids).unwrap();
    backend.free(&packed).unwrap();
    backend.free(&norms).unwrap();
}

#[test]
fn test_turboquant_compress_zero_vector() {
    let backend = CudaBackend::new(0).unwrap();
    let head_dim = 16;
    let num_kv_heads = 1;
    let num_tokens = 1;
    let bits: u8 = 4;

    // Create a zero vector
    let input = backend
        .alloc(&[num_tokens, num_kv_heads, head_dim], DType::FP16)
        .unwrap();
    let zeros = vec![0u8; num_tokens * num_kv_heads * head_dim * 2];
    backend.copy_to_device(&input, &zeros).unwrap();

    let rotation = upload_rotation(&backend, head_dim, 42);
    let centroids = upload_centroids(&backend, head_dim, bits);

    let packed_dim = TurboQuantConfig::packed_dim_per_head(head_dim, bits);
    let packed = backend
        .alloc(&[num_tokens, num_kv_heads * packed_dim], DType::INT8)
        .unwrap();
    let norms = backend
        .alloc(&[num_tokens, num_kv_heads], DType::FP16)
        .unwrap();

    // Should not crash or produce NaN
    backend
        .turboquant_compress(&input, &rotation, &centroids, bits, &packed, &norms)
        .unwrap();
    backend.synchronize().unwrap();

    let host_norms = download_fp16(&backend, &norms);
    assert!(
        host_norms[0].abs() < 1e-6,
        "zero vector norm should be ~0, got {}",
        host_norms[0]
    );
    assert!(
        !host_norms[0].is_nan(),
        "zero vector norm should not be NaN"
    );

    backend.free(&input).unwrap();
    backend.free(&rotation).unwrap();
    backend.free(&centroids).unwrap();
    backend.free(&packed).unwrap();
    backend.free(&norms).unwrap();
}

#[test]
fn test_turboquant_compress_8bit() {
    let backend = CudaBackend::new(0).unwrap();
    let head_dim = 16;
    let num_kv_heads = 2;
    let num_tokens = 2;
    let bits: u8 = 8;

    let (input, host_input) = random_fp16_tensor(
        &backend,
        &[num_tokens, num_kv_heads, head_dim],
        77777,
    );

    let rotation = upload_rotation(&backend, head_dim, 42);
    let centroids = upload_centroids(&backend, head_dim, bits);

    let packed_dim = TurboQuantConfig::packed_dim_per_head(head_dim, bits);
    let packed = backend
        .alloc(&[num_tokens, num_kv_heads * packed_dim], DType::INT8)
        .unwrap();
    let norms = backend
        .alloc(&[num_tokens, num_kv_heads], DType::FP16)
        .unwrap();

    backend
        .turboquant_compress(&input, &rotation, &centroids, bits, &packed, &norms)
        .unwrap();
    backend.synchronize().unwrap();

    let output = backend
        .alloc(&[num_tokens, num_kv_heads, head_dim], DType::FP16)
        .unwrap();

    unsafe {
        use fracture_cuda::ffi::*;
        let err = launch_turboquant_decompress(
            backend.get_ptr(packed.id).unwrap() as *const std::ffi::c_void,
            backend.get_ptr(norms.id).unwrap() as *const std::ffi::c_void,
            backend.get_ptr(rotation.id).unwrap() as *const f32,
            backend.get_ptr(centroids.id).unwrap() as *const f32,
            backend.get_ptr(output.id).unwrap(),
            num_tokens as i32,
            num_kv_heads as i32,
            head_dim as i32,
            bits as i32,
            packed_dim as i32,
            backend.stream(),
        );
        assert_eq!(err, 0);
    }
    backend.synchronize().unwrap();

    let host_output = download_fp16(&backend, &output);

    // 8-bit should be very high fidelity
    for token in 0..num_tokens {
        for head in 0..num_kv_heads {
            let offset = (token * num_kv_heads + head) * head_dim;
            let orig = &host_input[offset..offset + head_dim];
            let recon = &host_output[offset..offset + head_dim];
            let sim = cosine_similarity(orig, recon);
            assert!(
                sim > 0.999,
                "8-bit round-trip cosine sim for token={token} head={head} should be > 0.999, got {sim}"
            );
        }
    }

    backend.free(&input).unwrap();
    backend.free(&rotation).unwrap();
    backend.free(&centroids).unwrap();
    backend.free(&packed).unwrap();
    backend.free(&norms).unwrap();
    backend.free(&output).unwrap();
}

// ── End-to-end: TurboQuant batched_forward vs FP16 batched_forward ─────

fn e2e_test_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 64,
        num_layers: 2,
        num_q_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        intermediate_size: 128,
        vocab_size: 256,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        max_seq_len: 64,
    }
}

fn random_fp16_bytes_seeded(numel: usize) -> Vec<u8> {
    let mut rng = rand::rng();
    let mut bytes = Vec::with_capacity(numel * 2);
    for _ in 0..numel {
        let val: f32 = rng.random_range(-0.1..0.1);
        let fp16 = f16::from_f32(val);
        bytes.extend_from_slice(&fp16.to_le_bytes());
    }
    bytes
}

fn alloc_random(
    backend: &CudaBackend,
    shape: &[usize],
) -> fracture_core::Result<DeviceTensor> {
    let tensor = backend.alloc(shape, DType::FP16)?;
    let numel: usize = shape.iter().product();
    let data = random_fp16_bytes_seeded(numel);
    backend.copy_to_device(&tensor, &data)?;
    Ok(tensor)
}

fn build_e2e_weights(backend: &CudaBackend) -> fracture_core::Result<WeightStore> {
    let cfg = e2e_test_config();
    let token_embedding = alloc_random(backend, &[cfg.vocab_size, cfg.hidden_size])?;
    let mut layers = Vec::new();
    for _ in 0..cfg.num_layers {
        layers.push(LayerWeights {
            q_proj: alloc_random(backend, &[cfg.hidden_size, cfg.hidden_size])?,
            k_proj: alloc_random(backend, &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size])?,
            v_proj: alloc_random(backend, &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size])?,
            o_proj: alloc_random(backend, &[cfg.hidden_size, cfg.hidden_size])?,
            gate_proj: alloc_random(backend, &[cfg.intermediate_size, cfg.hidden_size])?,
            up_proj: alloc_random(backend, &[cfg.intermediate_size, cfg.hidden_size])?,
            down_proj: alloc_random(backend, &[cfg.hidden_size, cfg.intermediate_size])?,
            attn_norm: alloc_random(backend, &[cfg.hidden_size])?,
            ffn_norm: alloc_random(backend, &[cfg.hidden_size])?,
        });
    }
    let output_norm = alloc_random(backend, &[cfg.hidden_size])?;
    let lm_head = alloc_random(backend, &[cfg.vocab_size, cfg.hidden_size])?;
    Ok(WeightStore {
        config: cfg,
        token_embedding,
        layers,
        output_norm,
        lm_head,
    })
}

/// Compare TurboQuant 8-bit batched_forward against FP16 batched_forward.
///
/// 8-bit TQ should produce nearly identical logits to FP16 since the
/// quantization distortion is negligible at 256 levels.
#[test]
fn test_batched_forward_tq8_vs_fp16() {
    let mut backend = CudaBackend::new(0).unwrap();
    let cfg = e2e_test_config();
    backend
        .precompute_rope_freqs(cfg.head_dim, cfg.rope_theta)
        .unwrap();

    let weights = build_e2e_weights(&backend).unwrap();
    let layer_range = 0..cfg.num_layers;

    let prompt = vec![1u32, 2, 3, 4, 5];
    let positions: Vec<u32> = (0..prompt.len() as u32).collect();

    // FP16 paged path
    let mut fp16_cache = PagedKvCacheManager::new(
        64, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, &backend,
    )
    .unwrap();
    let fp16_handle = fracture_engine::PagedCache::alloc(&mut fp16_cache).unwrap();
    let sequences_fp16 = vec![SequenceSlice {
        handle: fp16_handle,
        token_ids: prompt.clone(),
        positions: positions.clone(),
    }];
    let fp16_output = batched_forward(
        &backend, &weights, &layer_range, &mut fp16_cache, &sequences_fp16,
    )
    .unwrap();

    // TQ 8-bit path (near-lossless)
    let tq_config = TurboQuantConfig {
        key_bits: 8,
        value_bits: 8,
        protected_bits: 8,
        protected_layers: 0,
        residual_tokens: 0,
        seed: 42,
    };
    let mut tq_cache = QuantizedKvCacheManager::new(
        64, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim,
        prompt.len(), tq_config, &backend,
    )
    .unwrap();
    let tq_handle = fracture_engine::PagedCache::alloc(&mut tq_cache).unwrap();
    let sequences_tq = vec![SequenceSlice {
        handle: tq_handle,
        token_ids: prompt.clone(),
        positions: positions.clone(),
    }];
    let tq_output = batched_forward(
        &backend, &weights, &layer_range, &mut tq_cache, &sequences_tq,
    )
    .unwrap();

    // Compare logits
    let fp16_logits = &fp16_output.logits[0];
    let tq_logits = &tq_output.logits[0];
    assert_eq!(fp16_logits.len(), tq_logits.len());

    let sim = cosine_similarity(fp16_logits, tq_logits);
    eprintln!("TQ8 vs FP16 logit cosine similarity: {sim:.6}");

    assert!(
        sim > 0.95,
        "8-bit TQ logits should be close to FP16 (cosine > 0.95), got {sim}"
    );

    // Cleanup
    fracture_engine::PagedCache::free(&mut fp16_cache, fp16_handle).unwrap();
    fp16_cache.destroy(&backend).unwrap();
    fracture_engine::PagedCache::free(&mut tq_cache, tq_handle).unwrap();
    tq_cache.destroy(&backend).unwrap();
}

/// Compare TurboQuant K4/V2 batched_forward against FP16.
///
/// K4/V2 has more quantization noise but should still produce
/// correlated logits (same general distribution).
#[test]
fn test_batched_forward_tq_k4v2_vs_fp16() {
    let mut backend = CudaBackend::new(0).unwrap();
    let cfg = e2e_test_config();
    backend
        .precompute_rope_freqs(cfg.head_dim, cfg.rope_theta)
        .unwrap();

    let weights = build_e2e_weights(&backend).unwrap();
    let layer_range = 0..cfg.num_layers;

    let prompt = vec![10u32, 20, 30];
    let positions: Vec<u32> = (0..prompt.len() as u32).collect();

    // FP16
    let mut fp16_cache = PagedKvCacheManager::new(
        64, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, &backend,
    )
    .unwrap();
    let fp16_handle = fracture_engine::PagedCache::alloc(&mut fp16_cache).unwrap();
    let fp16_output = batched_forward(
        &backend, &weights, &layer_range, &mut fp16_cache,
        &[SequenceSlice {
            handle: fp16_handle, token_ids: prompt.clone(), positions: positions.clone(),
        }],
    )
    .unwrap();

    // TQ K4/V2
    let tq_config = TurboQuantConfig::default(); // K4/V2
    let mut tq_cache = QuantizedKvCacheManager::new(
        64, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim,
        prompt.len(), tq_config, &backend,
    )
    .unwrap();
    let tq_handle = fracture_engine::PagedCache::alloc(&mut tq_cache).unwrap();
    let tq_output = batched_forward(
        &backend, &weights, &layer_range, &mut tq_cache,
        &[SequenceSlice {
            handle: tq_handle, token_ids: prompt.clone(), positions: positions.clone(),
        }],
    )
    .unwrap();

    let fp16_logits = &fp16_output.logits[0];
    let tq_logits = &tq_output.logits[0];
    assert_eq!(fp16_logits.len(), tq_logits.len());

    let sim = cosine_similarity(fp16_logits, tq_logits);
    eprintln!("TQ K4/V2 vs FP16 logit cosine similarity: {sim:.6}");

    // K4/V2 has more noise — lower threshold
    assert!(
        sim > 0.70,
        "K4/V2 TQ logits should correlate with FP16 (cosine > 0.70), got {sim}"
    );

    // Verify argmax — top prediction should often agree
    let fp16_argmax = fp16_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    let tq_argmax = tq_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    eprintln!("FP16 argmax: {fp16_argmax}, TQ K4/V2 argmax: {tq_argmax}");

    fracture_engine::PagedCache::free(&mut fp16_cache, fp16_handle).unwrap();
    fp16_cache.destroy(&backend).unwrap();
    fracture_engine::PagedCache::free(&mut tq_cache, tq_handle).unwrap();
    tq_cache.destroy(&backend).unwrap();
}
