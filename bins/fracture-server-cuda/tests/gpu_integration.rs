//! GPU integration tests for the Fracture generation pipeline.
//!
//! These tests run on a real CUDA GPU (no mocks). They construct a tiny model with
//! random weights directly on the GPU and exercise the full forward pass and
//! generation loop through real CUDA kernels. The outputs are not meaningful text,
//! but they validate the complete pipeline end-to-end.

use fracture_core::{Backend, DType, ModelConfig};
use fracture_cuda::CudaBackend;
use fracture_engine::{
    batched_forward, CacheHandle, Engine, GenerationEvent, KvCacheManager, NodeConfig,
    PagedKvCacheManager, PendingRequest, SequenceSlice,
};
use fracture_generate::{GenerationConfig, GenerationLoop};
use fracture_gguf::{LayerWeights, WeightStore};
use fracture_server::{start_scheduler_loop, SchedulerLoopConfig};
use half::f16;
use rand::Rng;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Tiny model configuration for integration tests.
fn test_config() -> ModelConfig {
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

/// Generate random FP16 data in [-0.1, 0.1] as bytes.
fn random_fp16_bytes(numel: usize) -> Vec<u8> {
    let mut rng = rand::rng();
    let mut bytes = Vec::with_capacity(numel * 2);
    for _ in 0..numel {
        let val: f32 = rng.random_range(-0.1..0.1);
        let fp16 = f16::from_f32(val);
        bytes.extend_from_slice(&fp16.to_le_bytes());
    }
    bytes
}

/// Allocate a tensor on the GPU and fill it with small random FP16 values.
fn alloc_random_tensor(
    backend: &CudaBackend,
    shape: &[usize],
) -> fracture_core::Result<fracture_core::DeviceTensor> {
    let tensor = backend.alloc(shape, DType::FP16)?;
    let numel: usize = shape.iter().product();
    let data = random_fp16_bytes(numel);
    backend.copy_to_device(&tensor, &data)?;
    Ok(tensor)
}

/// Build a complete WeightStore with random weights on the GPU.
fn build_test_weights(backend: &CudaBackend) -> fracture_core::Result<WeightStore> {
    let cfg = test_config();

    let token_embedding = alloc_random_tensor(backend, &[cfg.vocab_size, cfg.hidden_size])?;

    let mut layers = Vec::new();
    for _ in 0..cfg.num_layers {
        let q_proj = alloc_random_tensor(backend, &[cfg.hidden_size, cfg.hidden_size])?;
        let k_proj =
            alloc_random_tensor(backend, &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size])?;
        let v_proj =
            alloc_random_tensor(backend, &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size])?;
        let o_proj = alloc_random_tensor(backend, &[cfg.hidden_size, cfg.hidden_size])?;
        let gate_proj =
            alloc_random_tensor(backend, &[cfg.intermediate_size, cfg.hidden_size])?;
        let up_proj =
            alloc_random_tensor(backend, &[cfg.intermediate_size, cfg.hidden_size])?;
        let down_proj =
            alloc_random_tensor(backend, &[cfg.hidden_size, cfg.intermediate_size])?;
        let attn_norm = alloc_random_tensor(backend, &[cfg.hidden_size])?;
        let ffn_norm = alloc_random_tensor(backend, &[cfg.hidden_size])?;

        layers.push(LayerWeights {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
            attn_norm,
            ffn_norm,
        });
    }

    let output_norm = alloc_random_tensor(backend, &[cfg.hidden_size])?;
    let lm_head = alloc_random_tensor(backend, &[cfg.vocab_size, cfg.hidden_size])?;

    Ok(WeightStore {
        config: cfg,
        token_embedding,
        layers,
        output_norm,
        lm_head,
    })
}

/// Create a fully initialized engine with random weights on GPU.
fn setup_engine() -> fracture_core::Result<(Engine<CudaBackend>, ModelConfig)> {
    let mut backend = CudaBackend::new(0)?;
    let cfg = test_config();
    backend.precompute_rope_freqs(cfg.head_dim, cfg.rope_theta)?;
    let weights = build_test_weights(&backend)?;
    let num_layers = cfg.num_layers;
    let engine = Engine::new(backend, weights, 0..num_layers);
    Ok((engine, cfg))
}

/// Create engine + KV cache manager ready for inference.
fn setup_engine_and_cache() -> fracture_core::Result<(Engine<CudaBackend>, KvCacheManager, ModelConfig)> {
    let (engine, cfg) = setup_engine()?;
    let cache = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    Ok((engine, cache, cfg))
}

// ---------------------------------------------------------------------------
// Test 1: Basic generation produces tokens
// ---------------------------------------------------------------------------
#[test]
fn test_gpu_generation_basic() {
    let (engine, mut cache, _cfg) = setup_engine_and_cache().expect("setup failed");
    let (tx, mut rx) = mpsc::unbounded_channel();

    let gen_config = GenerationConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        stop_tokens: vec![], // no stop tokens — we want all 5
        seed: None,
    };

    let result = GenerationLoop::generate(&engine, &[1, 2, 3], &gen_config, &mut cache, &tx);
    let gen_result = result.expect("generation failed");

    assert!(!gen_result.tokens.is_empty(), "generation should produce at least one token");
    assert!(gen_result.tokens.len() <= 5, "should produce at most max_tokens");

    // Verify tokens were sent through the channel
    drop(tx);
    let mut streamed = Vec::new();
    while let Ok(t) = rx.try_recv() {
        streamed.push(t);
    }
    assert_eq!(streamed, gen_result.tokens, "streamed tokens should match returned tokens");
}

// ---------------------------------------------------------------------------
// Test 2: Generation respects max_tokens
// ---------------------------------------------------------------------------
#[test]
fn test_gpu_generation_stop_on_max_tokens() {
    let (engine, mut cache, _cfg) = setup_engine_and_cache().expect("setup failed");
    let (tx, _rx) = mpsc::unbounded_channel();

    let gen_config = GenerationConfig {
        max_tokens: 3,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        stop_tokens: vec![], // no stop tokens
        seed: None,
    };

    let gen_result = GenerationLoop::generate(&engine, &[1, 2, 3], &gen_config, &mut cache, &tx)
        .expect("generation failed");

    assert!(
        gen_result.tokens.len() <= 3,
        "should produce at most 3 tokens, got {}",
        gen_result.tokens.len()
    );
}

// ---------------------------------------------------------------------------
// Test 3: Tokens stream through the channel as generated
// ---------------------------------------------------------------------------
#[test]
fn test_gpu_generation_streaming() {
    let (engine, mut cache, _cfg) = setup_engine_and_cache().expect("setup failed");
    let (tx, mut rx) = mpsc::unbounded_channel();

    let gen_config = GenerationConfig {
        max_tokens: 4,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        stop_tokens: vec![],
        seed: None,
    };

    let gen_result = GenerationLoop::generate(&engine, &[1, 2, 3], &gen_config, &mut cache, &tx)
        .expect("generation failed");

    // Drop the sender so the channel closes
    drop(tx);

    // Collect all streamed tokens
    let mut streamed = Vec::new();
    while let Ok(t) = rx.try_recv() {
        streamed.push(t);
    }

    assert_eq!(
        streamed.len(),
        gen_result.tokens.len(),
        "channel should have exactly as many tokens as returned"
    );
    assert_eq!(streamed, gen_result.tokens, "streamed order should match return order");
}

// ---------------------------------------------------------------------------
// Test 4: Forward pass returns logits of correct size
// ---------------------------------------------------------------------------
#[test]
fn test_gpu_forward_pass() {
    let (engine, mut cache, cfg) = setup_engine_and_cache().expect("setup failed");

    let handle = cache.alloc(engine.backend()).expect("cache alloc failed");
    let logits = engine
        .forward(&[1, 2, 3], &[0, 1, 2], &mut cache, handle, None)
        .expect("forward failed");

    assert_eq!(
        logits.len(),
        cfg.vocab_size,
        "logits length should equal vocab_size"
    );

    // Logits should be finite (not NaN or inf) — random weights produce finite values
    for (i, &val) in logits.iter().enumerate() {
        assert!(
            val.is_finite(),
            "logit[{}] = {} is not finite",
            i,
            val
        );
    }

    // Clean up
    cache.free(handle, engine.backend()).expect("cache free failed");
}

// ---------------------------------------------------------------------------
// Test 5: Prefill then decode step
// ---------------------------------------------------------------------------
#[test]
fn test_gpu_forward_decode() {
    let (engine, mut cache, cfg) = setup_engine_and_cache().expect("setup failed");

    let handle = cache.alloc(engine.backend()).expect("cache alloc failed");

    // Prefill with [1, 2, 3]
    let prefill_logits = engine
        .forward(&[1, 2, 3], &[0, 1, 2], &mut cache, handle, None)
        .expect("prefill failed");
    assert_eq!(prefill_logits.len(), cfg.vocab_size);

    // Decode with [4] at position 3
    let decode_logits = engine
        .forward(&[4], &[3], &mut cache, handle, None)
        .expect("decode failed");
    assert_eq!(decode_logits.len(), cfg.vocab_size);

    // Decode logits should also be finite
    for (i, &val) in decode_logits.iter().enumerate() {
        assert!(
            val.is_finite(),
            "decode logit[{}] = {} is not finite",
            i,
            val
        );
    }

    cache.free(handle, engine.backend()).expect("cache free failed");
}

// ---------------------------------------------------------------------------
// Test 6: Prefill/decode consistency — the spec test
// ---------------------------------------------------------------------------
#[test]
fn test_gpu_prefill_decode_consistency() {
    let (engine, mut cache, cfg) = setup_engine_and_cache().expect("setup failed");

    // Run A: prefill [1, 2, 3], get logits, sample next token
    let handle_a = cache.alloc(engine.backend()).expect("cache alloc failed");
    let logits_a = engine
        .forward(&[1, 2, 3], &[0, 1, 2], &mut cache, handle_a, None)
        .expect("prefill A failed");

    // Pick the argmax token from logits_a
    let next_token = logits_a
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap();

    // Decode step with that token
    let decode_logits = engine
        .forward(&[next_token], &[3], &mut cache, handle_a, None)
        .expect("decode A failed");

    cache.free(handle_a, engine.backend()).expect("free A failed");

    // Run B: prefill [1, 2, 3, next_token] in one shot, take last-position logits
    let handle_b = cache.alloc(engine.backend()).expect("cache alloc B failed");
    let full_seq = [1u32, 2, 3, next_token];
    let full_logits = engine
        .forward(&full_seq, &[0, 1, 2, 3], &mut cache, handle_b, None)
        .expect("prefill B failed");

    cache.free(handle_b, engine.backend()).expect("free B failed");

    // Compare decode_logits vs full_logits — should match within FP16 tolerance
    assert_eq!(decode_logits.len(), full_logits.len());
    let mut max_diff: f32 = 0.0;
    for i in 0..cfg.vocab_size {
        let diff = (decode_logits[i] - full_logits[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    // FP16 has ~3 decimal digits of precision. With accumulated error across
    // 2 layers of matmuls, norms, and attention, tolerance needs to account
    // for numerical differences between processing sequences in one shot vs
    // incrementally. Use a generous tolerance.
    let tolerance = 0.05;
    assert!(
        max_diff < tolerance,
        "prefill/decode logits differ by {max_diff} (tolerance {tolerance})"
    );
}

// ---------------------------------------------------------------------------
// Test 7: GPU memory is reclaimed after generation
// ---------------------------------------------------------------------------
#[test]
fn test_gpu_cache_freed_after_generation() {
    let (engine, mut cache, _cfg) = setup_engine_and_cache().expect("setup failed");

    // Warm up: run a short generation so PTX JIT compilation (if using forward-
    // compatible PTX instead of native SASS) allocates its kernel cache before
    // we take the memory snapshot.
    {
        let (tx, _rx) = mpsc::unbounded_channel();
        let warmup_config = GenerationConfig {
            max_tokens: 1,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            stop_tokens: vec![],
            seed: None,
        };
        let _ = GenerationLoop::generate(&engine, &[1], &warmup_config, &mut cache, &tx)
            .expect("warmup generation failed");
    }

    // Snapshot memory before generation
    engine.backend().synchronize().expect("sync failed");
    let mem_before = engine.backend().available_memory();

    {
        let (tx, _rx) = mpsc::unbounded_channel();
        let gen_config = GenerationConfig {
            max_tokens: 4,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            stop_tokens: vec![],
            seed: None,
        };
        let _ = GenerationLoop::generate(&engine, &[1, 2, 3], &gen_config, &mut cache, &tx)
            .expect("generation failed");
    }

    // After generation completes, KV cache should be freed and scratch tensors
    // deallocated. Memory should return to approximately the same level.
    engine.backend().synchronize().expect("sync failed");
    let mem_after = engine.backend().available_memory();

    // Allow 8 MB tolerance for CUDA allocator fragmentation, suballocator
    // rounding, and internal cuBLAS workspace state
    let tolerance = 8 * 1024 * 1024;
    let leaked = if mem_before > mem_after {
        mem_before - mem_after
    } else {
        0
    };
    assert!(
        leaked < tolerance,
        "potential GPU memory leak: {leaked} bytes not reclaimed (before={mem_before}, after={mem_after})"
    );
}

// ---------------------------------------------------------------------------
// Phase 2: Split equivalence tests
//
// These use forward_node() on a single engine (single backend, single tensor
// registry) to manually chain head→tail, then compare against monolithic.
// ---------------------------------------------------------------------------

/// Helper: run a 2-node split forward on a single engine by calling forward_node()
/// twice (head config, then tail config) and chaining the activation tensor.
fn split_forward(
    engine: &Engine<CudaBackend>,
    token_ids: &[u32],
    positions: &[u32],
    head_config: &NodeConfig,
    tail_config: &NodeConfig,
    cache_head: &mut KvCacheManager,
    handle_head: CacheHandle,
    cache_tail: &mut KvCacheManager,
    handle_tail: CacheHandle,
) -> Vec<f32> {
    use fracture_engine::{NodeInput, NodeOutput};

    // Head: tokens → activations
    let head_input = NodeInput::TokenIds {
        ids: token_ids.to_vec(),
        positions: positions.to_vec(),
    };
    let head_output = engine
        .forward_node(head_input, head_config, cache_head, handle_head, None)
        .expect("head forward failed");

    let activation = match head_output {
        NodeOutput::Activations(t) => t,
        NodeOutput::Logits(_) => panic!("head should return Activations"),
    };

    // Tail: activations → logits
    let tail_input = NodeInput::Activations {
        hidden_states: activation,
        positions: positions.to_vec(),
    };
    let tail_output = engine
        .forward_node(tail_input, tail_config, cache_tail, handle_tail, None)
        .expect("tail forward failed");

    match tail_output {
        NodeOutput::Logits(logits) => logits,
        NodeOutput::Activations(_) => panic!("tail should return Logits"),
    }
}

fn assert_logits_match(full: &[f32], split: &[f32], label: &str) {
    assert_eq!(full.len(), split.len(), "{label}: length mismatch");
    for (i, (a, b)) in full.iter().zip(split.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "{label}: logit mismatch at index {i}: full={a}, split={b} (diff={})",
            (a - b).abs()
        );
    }
}

/// The critical Phase 2 test: 2-node split produces identical logits to monolithic.
#[test]
fn test_gpu_split_equivalence_2node() {
    let (engine, cfg) = setup_engine().expect("setup failed");
    let head_config = NodeConfig::new(0..1, cfg.num_layers).unwrap();
    let tail_config = NodeConfig::new(1..2, cfg.num_layers).unwrap();

    // Monolithic
    let mut cache_full = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let hf = cache_full.alloc(engine.backend()).expect("alloc");
    let logits_full = engine
        .forward(&[1, 2, 3], &[0, 1, 2], &mut cache_full, hf, None)
        .expect("monolithic forward");

    // Split
    let mut cache_head = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let mut cache_tail = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let hh = cache_head.alloc(engine.backend()).expect("alloc");
    let ht = cache_tail.alloc(engine.backend()).expect("alloc");

    let logits_split = split_forward(
        &engine, &[1, 2, 3], &[0, 1, 2],
        &head_config, &tail_config,
        &mut cache_head, hh, &mut cache_tail, ht,
    );

    assert_logits_match(&logits_full, &logits_split, "prefill");
}

/// Split equivalence over prefill + 5 decode steps.
#[test]
fn test_gpu_split_equivalence_decode() {
    let (engine, cfg) = setup_engine().expect("setup failed");
    let head_config = NodeConfig::new(0..1, cfg.num_layers).unwrap();
    let tail_config = NodeConfig::new(1..2, cfg.num_layers).unwrap();

    // Monolithic: prefill + 5 decode
    let mut cache_full = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let hf = cache_full.alloc(engine.backend()).expect("alloc");
    let logits_prefill_full = engine
        .forward(&[1, 2, 3], &[0, 1, 2], &mut cache_full, hf, None)
        .expect("prefill");
    let mut full_decode = Vec::new();
    for step in 0..5u32 {
        full_decode.push(
            engine.forward(&[42], &[3 + step], &mut cache_full, hf, None).expect("decode"),
        );
    }

    // Split: prefill + 5 decode
    let mut ch = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let mut ct = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let hh = ch.alloc(engine.backend()).expect("alloc");
    let ht = ct.alloc(engine.backend()).expect("alloc");

    let logits_prefill_split = split_forward(
        &engine, &[1, 2, 3], &[0, 1, 2],
        &head_config, &tail_config, &mut ch, hh, &mut ct, ht,
    );
    assert_logits_match(&logits_prefill_full, &logits_prefill_split, "prefill");

    for step in 0..5u32 {
        let split = split_forward(
            &engine, &[42], &[3 + step],
            &head_config, &tail_config, &mut ch, hh, &mut ct, ht,
        );
        assert_logits_match(&full_decode[step as usize], &split, &format!("decode step {step}"));
    }
}

/// Asymmetric split on a 4-layer model: [0,1) + [1,4).
#[test]
fn test_gpu_split_equivalence_asymmetric() {
    let cfg = ModelConfig {
        hidden_size: 64,
        num_layers: 4,
        num_q_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        intermediate_size: 128,
        vocab_size: 256,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        max_seq_len: 64,
    };

    let mut backend = CudaBackend::new(0).expect("backend");
    backend.precompute_rope_freqs(cfg.head_dim, cfg.rope_theta).expect("rope");

    let token_embedding = alloc_random_tensor(&backend, &[cfg.vocab_size, cfg.hidden_size]).unwrap();
    let mut layers = Vec::new();
    for _ in 0..cfg.num_layers {
        layers.push(LayerWeights {
            q_proj: alloc_random_tensor(&backend, &[cfg.hidden_size, cfg.hidden_size]).unwrap(),
            k_proj: alloc_random_tensor(&backend, &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size]).unwrap(),
            v_proj: alloc_random_tensor(&backend, &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size]).unwrap(),
            o_proj: alloc_random_tensor(&backend, &[cfg.hidden_size, cfg.hidden_size]).unwrap(),
            gate_proj: alloc_random_tensor(&backend, &[cfg.intermediate_size, cfg.hidden_size]).unwrap(),
            up_proj: alloc_random_tensor(&backend, &[cfg.intermediate_size, cfg.hidden_size]).unwrap(),
            down_proj: alloc_random_tensor(&backend, &[cfg.hidden_size, cfg.intermediate_size]).unwrap(),
            attn_norm: alloc_random_tensor(&backend, &[cfg.hidden_size]).unwrap(),
            ffn_norm: alloc_random_tensor(&backend, &[cfg.hidden_size]).unwrap(),
        });
    }
    let output_norm = alloc_random_tensor(&backend, &[cfg.hidden_size]).unwrap();
    let lm_head = alloc_random_tensor(&backend, &[cfg.vocab_size, cfg.hidden_size]).unwrap();

    let weights = WeightStore { config: cfg.clone(), token_embedding, layers, output_norm, lm_head };
    let engine = Engine::new(backend, weights, 0..cfg.num_layers);

    let head_config = NodeConfig::new(0..1, cfg.num_layers).unwrap();
    let tail_config = NodeConfig::new(1..4, cfg.num_layers).unwrap();

    // Monolithic
    let mut cache_full = KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let hf = cache_full.alloc(engine.backend()).expect("alloc");
    let logits_full = engine.forward(&[10, 20], &[0, 1], &mut cache_full, hf, None).expect("monolithic");

    // Split: [0,1) + [1,4)
    let mut ch = KvCacheManager::new(1, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let mut ct = KvCacheManager::new(3, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len);
    let hh = ch.alloc(engine.backend()).expect("alloc");
    let ht = ct.alloc(engine.backend()).expect("alloc");

    let logits_split = split_forward(
        &engine, &[10, 20], &[0, 1],
        &head_config, &tail_config, &mut ch, hh, &mut ct, ht,
    );

    assert_logits_match(&logits_full, &logits_split, "asymmetric split");
}

// ---------------------------------------------------------------------------
// Test: Paged KV cache produces identical output to contiguous KV cache
// ---------------------------------------------------------------------------
//
// This is the Phase 4 Step 1f correctness validation. The paged attention
// kernel must produce byte-identical (within FP16 tolerance) results to
// the contiguous attention kernel on the same model and input.

#[test]
fn test_paged_vs_contiguous_prefill() {
    let (engine, cfg) = setup_engine().expect("setup failed");

    let prompt = vec![1u32, 2, 3, 4, 5];
    let positions: Vec<u32> = (0..prompt.len() as u32).collect();

    // Contiguous path
    let mut cont_cache = KvCacheManager::new(
        cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len,
    );
    let cont_handle = cont_cache.alloc(engine.backend()).expect("contiguous alloc");
    let logits_cont = engine
        .forward(&prompt, &positions, &mut cont_cache, cont_handle, None)
        .expect("contiguous forward");
    cont_cache.free(cont_handle, engine.backend()).expect("free");

    // Paged path
    // Allocate enough blocks: ceil(5/16) = 1 block per layer, but pool needs
    // at least 1 block. Use a generous pool.
    let mut paged_cache = PagedKvCacheManager::new(
        64, // 64 blocks — plenty for a 5-token prompt
        cfg.num_layers,
        cfg.num_kv_heads,
        cfg.head_dim,
        engine.backend(),
    )
    .expect("paged cache creation");
    let paged_handle = paged_cache.alloc().expect("paged alloc");
    let logits_paged = engine
        .forward_paged(&prompt, &positions, &mut paged_cache, paged_handle)
        .expect("paged forward");
    paged_cache.free(paged_handle).expect("free");
    paged_cache.destroy(engine.backend()).expect("pool destroy");

    // Compare logits — should be identical within FP16 tolerance
    assert_eq!(
        logits_cont.len(),
        logits_paged.len(),
        "logits length mismatch: contiguous={} paged={}",
        logits_cont.len(),
        logits_paged.len()
    );

    let mut max_diff: f32 = 0.0;
    let mut diff_count = 0usize;
    for (i, (c, p)) in logits_cont.iter().zip(logits_paged.iter()).enumerate() {
        let diff = (c - p).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0.01 {
            diff_count += 1;
            if diff_count <= 5 {
                eprintln!(
                    "  logit[{i}]: contiguous={c:.6} paged={p:.6} diff={diff:.6}"
                );
            }
        }
    }

    eprintln!(
        "Paged vs contiguous prefill: max_diff={max_diff:.6}, diffs>0.01: {diff_count}/{}",
        logits_cont.len()
    );
    assert!(
        max_diff < 0.05,
        "paged attention diverges from contiguous: max_diff={max_diff:.6} (threshold 0.05)"
    );
}

#[test]
fn test_paged_vs_contiguous_decode_steps() {
    let (engine, cfg) = setup_engine().expect("setup failed");

    let prompt = vec![1u32, 2, 3];
    let prompt_positions: Vec<u32> = (0..prompt.len() as u32).collect();
    let num_decode_steps = 5;

    // --- Contiguous path ---
    let mut cont_cache = KvCacheManager::new(
        cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len,
    );
    let cont_handle = cont_cache.alloc(engine.backend()).expect("cont alloc");

    // Prefill
    let mut cont_logits = engine
        .forward(&prompt, &prompt_positions, &mut cont_cache, cont_handle, None)
        .expect("cont prefill");
    let mut cont_tokens = Vec::new();

    // Decode steps
    for step in 0..num_decode_steps {
        let token = cont_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();
        cont_tokens.push(token);

        let pos = (prompt.len() + step) as u32;
        cont_logits = engine
            .forward(&[token], &[pos], &mut cont_cache, cont_handle, None)
            .expect(&format!("cont decode step {step}"));
    }
    cont_cache.free(cont_handle, engine.backend()).expect("free");

    // --- Paged path ---
    let mut paged_cache = PagedKvCacheManager::new(
        64,
        cfg.num_layers,
        cfg.num_kv_heads,
        cfg.head_dim,
        engine.backend(),
    )
    .expect("paged cache creation");
    let paged_handle = paged_cache.alloc().expect("paged alloc");

    // Prefill
    let mut paged_logits = engine
        .forward_paged(&prompt, &prompt_positions, &mut paged_cache, paged_handle)
        .expect("paged prefill");
    let mut paged_tokens = Vec::new();

    // Decode steps
    for step in 0..num_decode_steps {
        let token = paged_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();
        paged_tokens.push(token);

        let pos = (prompt.len() + step) as u32;
        paged_logits = engine
            .forward_paged(&[token], &[pos], &mut paged_cache, paged_handle)
            .expect(&format!("paged decode step {step}"));
    }
    paged_cache.free(paged_handle).expect("free");
    paged_cache.destroy(engine.backend()).expect("pool destroy");

    // Compare: greedy token sequences must be identical
    assert_eq!(
        cont_tokens, paged_tokens,
        "paged decode produced different tokens than contiguous:\n  contiguous: {cont_tokens:?}\n  paged:      {paged_tokens:?}"
    );

    // Compare final logits
    let mut max_diff: f32 = 0.0;
    for (c, p) in cont_logits.iter().zip(paged_logits.iter()) {
        let diff = (c - p).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    eprintln!(
        "Paged vs contiguous after {num_decode_steps} decode steps: max_diff={max_diff:.6}, tokens match: {}",
        cont_tokens == paged_tokens
    );
    assert!(
        max_diff < 0.05,
        "final logits diverge: max_diff={max_diff:.6}"
    );
}

// ---------------------------------------------------------------------------
// Test: Paged attention across multiple blocks (>16 tokens)
// ---------------------------------------------------------------------------
//
// Validates that the paged attention kernel correctly handles sequences
// spanning multiple physical blocks. A 20-token prefill spans 2 blocks
// (block 0: tokens 0-15, block 1: tokens 16-19). Subsequent decode steps
// continue filling block 1 and eventually spill into block 2.

#[test]
fn test_paged_vs_contiguous_multi_block() {
    let (engine, cfg) = setup_engine().expect("setup failed");

    // 20-token prompt → 2 blocks (16 + 4)
    let prompt: Vec<u32> = (1..=20).collect();
    let prompt_positions: Vec<u32> = (0..prompt.len() as u32).collect();
    let num_decode_steps = 15; // total will be 35 tokens → 3 blocks (16+16+3)

    // --- Contiguous path ---
    let mut cont_cache = KvCacheManager::new(
        cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len,
    );
    let cont_handle = cont_cache.alloc(engine.backend()).expect("cont alloc");

    let mut cont_logits = engine
        .forward(&prompt, &prompt_positions, &mut cont_cache, cont_handle, None)
        .expect("cont prefill");
    let mut cont_tokens = Vec::new();

    for step in 0..num_decode_steps {
        let token = cont_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();
        cont_tokens.push(token);
        let pos = (prompt.len() + step) as u32;
        cont_logits = engine
            .forward(&[token], &[pos], &mut cont_cache, cont_handle, None)
            .expect(&format!("cont decode step {step}"));
    }
    cont_cache.free(cont_handle, engine.backend()).expect("free");

    // --- Paged path ---
    let mut paged_cache = PagedKvCacheManager::new(
        64, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, engine.backend(),
    )
    .expect("paged cache creation");
    let paged_handle = paged_cache.alloc().expect("paged alloc");

    let mut paged_logits = engine
        .forward_paged(&prompt, &prompt_positions, &mut paged_cache, paged_handle)
        .expect("paged prefill");
    let mut paged_tokens = Vec::new();

    for step in 0..num_decode_steps {
        let token = paged_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();
        paged_tokens.push(token);
        let pos = (prompt.len() + step) as u32;
        paged_logits = engine
            .forward_paged(&[token], &[pos], &mut paged_cache, paged_handle)
            .expect(&format!("paged decode step {step}"));
    }

    // Verify block table spans 3 blocks: ceil(35/16) = 3
    let block_table = paged_cache.block_table(paged_handle).expect("block table");
    assert_eq!(
        block_table.len(), 3,
        "35 tokens should span 3 blocks, got {}",
        block_table.len()
    );

    paged_cache.free(paged_handle).expect("free");
    paged_cache.destroy(engine.backend()).expect("pool destroy");

    // Compare: greedy token sequences must match
    assert_eq!(
        cont_tokens, paged_tokens,
        "multi-block paged decode diverges:\n  contiguous: {cont_tokens:?}\n  paged:      {paged_tokens:?}"
    );

    // Compare final logits
    let mut max_diff: f32 = 0.0;
    for (c, p) in cont_logits.iter().zip(paged_logits.iter()) {
        max_diff = max_diff.max((c - p).abs());
    }
    eprintln!(
        "Multi-block paged vs contiguous (20 prefill + {num_decode_steps} decode = {} tokens, 3 blocks): \
         max_diff={max_diff:.6}, tokens match: {}",
        prompt.len() + num_decode_steps,
        cont_tokens == paged_tokens
    );
    assert!(
        max_diff < 0.05,
        "multi-block logits diverge: max_diff={max_diff:.6}"
    );
}

// ---------------------------------------------------------------------------
// Test: Batched forward produces identical per-sequence output to sequential
// ---------------------------------------------------------------------------
//
// Phase 4 Step 2 validation. Run two sequences individually through
// forward_paged, then run them together through batched_forward. Each
// sequence's logits must be identical.

#[test]
fn test_batched_vs_sequential_prefill() {
    let (engine, cfg) = setup_engine().expect("setup failed");

    let prompt_a: Vec<u32> = vec![1, 2, 3, 4, 5];
    let prompt_b: Vec<u32> = vec![10, 20, 30];
    let pos_a: Vec<u32> = (0..prompt_a.len() as u32).collect();
    let pos_b: Vec<u32> = (0..prompt_b.len() as u32).collect();

    // --- Sequential: run each sequence individually ---
    let mut cache_a = PagedKvCacheManager::new(
        64, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, engine.backend(),
    ).expect("cache_a");
    let ha = cache_a.alloc().expect("alloc_a");
    let logits_a_seq = engine
        .forward_paged(&prompt_a, &pos_a, &mut cache_a, ha)
        .expect("forward_a");
    cache_a.free(ha).expect("free_a");
    cache_a.destroy(engine.backend()).expect("destroy_a");

    let mut cache_b = PagedKvCacheManager::new(
        64, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, engine.backend(),
    ).expect("cache_b");
    let hb = cache_b.alloc().expect("alloc_b");
    let logits_b_seq = engine
        .forward_paged(&prompt_b, &pos_b, &mut cache_b, hb)
        .expect("forward_b");
    cache_b.free(hb).expect("free_b");
    cache_b.destroy(engine.backend()).expect("destroy_b");

    // --- Batched: run both sequences together ---
    let mut cache_batch = PagedKvCacheManager::new(
        128, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, engine.backend(),
    ).expect("cache_batch");
    let h1 = cache_batch.alloc().expect("alloc_1");
    let h2 = cache_batch.alloc().expect("alloc_2");

    let seqs = vec![
        SequenceSlice { handle: h1, token_ids: prompt_a.clone(), positions: pos_a.clone() },
        SequenceSlice { handle: h2, token_ids: prompt_b.clone(), positions: pos_b.clone() },
    ];

    let layer_range = 0..cfg.num_layers;
    let batch_result = batched_forward(
        engine.backend(), engine.weights(), &layer_range, &mut cache_batch, &seqs,
    ).expect("batched forward");

    cache_batch.free(h1).expect("free_1");
    cache_batch.free(h2).expect("free_2");
    cache_batch.destroy(engine.backend()).expect("destroy_batch");

    // --- Compare ---
    assert_eq!(batch_result.logits.len(), 2);

    let logits_a_batch = &batch_result.logits[0];
    let logits_b_batch = &batch_result.logits[1];

    assert_eq!(logits_a_seq.len(), logits_a_batch.len());
    assert_eq!(logits_b_seq.len(), logits_b_batch.len());

    let mut max_diff_a: f32 = 0.0;
    for (s, b) in logits_a_seq.iter().zip(logits_a_batch.iter()) {
        max_diff_a = max_diff_a.max((s - b).abs());
    }

    let mut max_diff_b: f32 = 0.0;
    for (s, b) in logits_b_seq.iter().zip(logits_b_batch.iter()) {
        max_diff_b = max_diff_b.max((s - b).abs());
    }

    // Greedy token from each
    let greedy_a_seq = logits_a_seq.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let greedy_a_bat = logits_a_batch.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let greedy_b_seq = logits_b_seq.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let greedy_b_bat = logits_b_batch.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;

    eprintln!(
        "Batched vs sequential prefill:\n  Seq A: max_diff={max_diff_a:.6}, greedy seq={greedy_a_seq} batch={greedy_a_bat}\n  Seq B: max_diff={max_diff_b:.6}, greedy seq={greedy_b_seq} batch={greedy_b_bat}"
    );

    assert!(max_diff_a < 0.05, "seq A logits diverge: max_diff={max_diff_a:.6}");
    assert!(max_diff_b < 0.05, "seq B logits diverge: max_diff={max_diff_b:.6}");
    assert_eq!(greedy_a_seq, greedy_a_bat, "seq A greedy token mismatch");
    assert_eq!(greedy_b_seq, greedy_b_bat, "seq B greedy token mismatch");
}

// ---------------------------------------------------------------------------
// Test: Concurrent requests through the batched scheduler produce correct results
// ---------------------------------------------------------------------------
//
// Phase 4 Step 3e validation. Starts the scheduler loop, submits 3 requests
// concurrently, collects their outputs, and verifies each sequence's greedy
// tokens match what it would produce when run alone.

/// Run greedy generation for a prompt through forward_paged (sequential, reference).
fn greedy_sequential(
    engine: &Engine<CudaBackend>,
    cfg: &ModelConfig,
    prompt: &[u32],
    max_tokens: usize,
) -> Vec<u32> {
    let mut cache = PagedKvCacheManager::new(
        128, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, engine.backend(),
    )
    .expect("cache");
    let handle = cache.alloc().expect("alloc");

    let positions: Vec<u32> = (0..prompt.len() as u32).collect();
    let mut logits = engine
        .forward_paged(prompt, &positions, &mut cache, handle)
        .expect("prefill");

    let mut tokens = Vec::new();
    for step in 0..max_tokens {
        let token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();
        tokens.push(token);
        let pos = (prompt.len() + step) as u32;
        logits = engine
            .forward_paged(&[token], &[pos], &mut cache, handle)
            .expect("decode");
    }

    cache.free(handle).expect("free");
    cache.destroy(engine.backend()).expect("destroy");
    tokens
}

#[tokio::test]
async fn test_scheduler_concurrent_requests() {
    let (engine, cfg) = setup_engine().expect("setup failed");
    let engine = Arc::new(engine);
    let max_tokens = 5;

    // --- Reference: run each prompt individually ---
    let prompts: Vec<Vec<u32>> = vec![
        vec![1, 2, 3, 4, 5],
        vec![10, 20, 30],
        vec![100, 101, 102, 103],
    ];

    let mut reference_tokens = Vec::new();
    for prompt in &prompts {
        reference_tokens.push(greedy_sequential(&engine, &cfg, prompt, max_tokens));
    }

    // --- Batched: submit all through the scheduler loop ---
    let cache = PagedKvCacheManager::new(
        256, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, engine.backend(),
    )
    .expect("paged cache");

    let scheduler_config = SchedulerLoopConfig {
        max_batch_size: 64,
        max_batch_tokens: 4096,
        max_prefill_tokens: 512,
        block_pool_reserve: 0.1,
    };

    let handle = start_scheduler_loop(Arc::clone(&engine), cache, scheduler_config);

    // Submit all requests and collect receivers.
    let mut receivers = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let (tx, rx) = mpsc::unbounded_channel();
        let req = PendingRequest {
            seq_id: i as u64,
            prompt_tokens: prompt.clone(),
            max_tokens,
            temperature: 0.0, // greedy
            top_k: 0,
            top_p: 1.0,
            seed: None,
            stop_tokens: vec![], // no stop tokens — generate exactly max_tokens
            event_tx: tx,
        };
        handle.submit(req).expect("submit");
        receivers.push(rx);
    }

    // Collect tokens from each receiver.
    let mut batched_tokens: Vec<Vec<u32>> = Vec::new();
    for mut rx in receivers {
        let mut tokens = Vec::new();
        while let Some(event) = rx.recv().await {
            match event {
                GenerationEvent::Token(t) => tokens.push(t),
                GenerationEvent::Finished { .. } => break,
                GenerationEvent::Error(e) => panic!("generation error: {e}"),
            }
        }
        batched_tokens.push(tokens);
    }

    // --- Compare ---
    assert_eq!(batched_tokens.len(), prompts.len());

    for (i, (ref_tokens, bat_tokens)) in reference_tokens
        .iter()
        .zip(batched_tokens.iter())
        .enumerate()
    {
        eprintln!(
            "Seq {i}: ref={ref_tokens:?} batched={bat_tokens:?} match={}",
            ref_tokens == bat_tokens
        );
        assert_eq!(
            ref_tokens, bat_tokens,
            "seq {i} tokens diverge:\n  reference: {ref_tokens:?}\n  batched:   {bat_tokens:?}"
        );
    }

    eprintln!(
        "Scheduler concurrent test: {} sequences, {} tokens each, all match reference",
        prompts.len(),
        max_tokens
    );
}
