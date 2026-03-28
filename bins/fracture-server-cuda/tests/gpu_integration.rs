//! GPU integration tests for the Fracture generation pipeline.
//!
//! These tests run on a real CUDA GPU (no mocks). They construct a tiny model with
//! random weights directly on the GPU and exercise the full forward pass and
//! generation loop through real CUDA kernels. The outputs are not meaningful text,
//! but they validate the complete pipeline end-to-end.

use fracture_core::{Backend, DType, ModelConfig};
use fracture_cuda::CudaBackend;
use fracture_engine::{CacheHandle, Engine, KvCacheManager, NodeConfig};
use fracture_generate::{GenerationConfig, GenerationLoop};
use fracture_gguf::{LayerWeights, WeightStore};
use half::f16;
use rand::Rng;
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
    };

    let result = GenerationLoop::generate(&engine, &[1, 2, 3], &gen_config, &mut cache, &tx);
    let tokens = result.expect("generation failed");

    assert!(!tokens.is_empty(), "generation should produce at least one token");
    assert!(tokens.len() <= 5, "should produce at most max_tokens");

    // Verify tokens were sent through the channel
    drop(tx);
    let mut streamed = Vec::new();
    while let Ok(t) = rx.try_recv() {
        streamed.push(t);
    }
    assert_eq!(streamed, tokens, "streamed tokens should match returned tokens");
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
    };

    let tokens = GenerationLoop::generate(&engine, &[1, 2, 3], &gen_config, &mut cache, &tx)
        .expect("generation failed");

    assert!(
        tokens.len() <= 3,
        "should produce at most 3 tokens, got {}",
        tokens.len()
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
    };

    let tokens = GenerationLoop::generate(&engine, &[1, 2, 3], &gen_config, &mut cache, &tx)
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
        tokens.len(),
        "channel should have exactly as many tokens as returned"
    );
    assert_eq!(streamed, tokens, "streamed order should match return order");
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
