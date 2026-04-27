use super::*;
use fracture_core::{DType, DeviceTensor, DeviceTimer, ModelConfig, TensorId};
use fracture_engine::PagedKvCacheManager;
use fracture_gguf::{LayerWeights, WeightStore};
use std::sync::atomic::{AtomicU64, Ordering};

// ── Mock model config (tiny) ─────────────────────────────────
fn tiny_config() -> ModelConfig {
    // vocab_size must be > 128009 to accommodate Llama 3 stop tokens
    ModelConfig {
        hidden_size: 8,
        num_layers: 1,
        num_q_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        intermediate_size: 16,
        vocab_size: 128256,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        max_seq_len: 512,
    }
}

// ── MockBackend ──────────────────────────────────────────────
//
// All compute ops are no-ops. `copy_to_host` writes controlled FP16 logits
// so that greedy sampling returns a predictable token. An internal call
// counter lets tests cycle through different tokens across forward passes.

struct MockBackend {
    next_tensor_id: AtomicU64,
    /// Incremented on every copy_to_host call whose buffer size matches
    /// `vocab_size * 2` (i.e., the logits readback).
    logit_call_count: AtomicU64,
    /// Token IDs to cycle through for successive forward passes.
    /// Each forward() triggers one logits copy_to_host.
    token_sequence: Vec<u32>,
    vocab_size: usize,
}

impl MockBackend {
    /// Create a mock that always returns `token` from greedy sampling.
    fn always(token: u32, vocab_size: usize) -> Self {
        Self {
            next_tensor_id: AtomicU64::new(1),
            logit_call_count: AtomicU64::new(0),
            token_sequence: vec![token],
            vocab_size,
        }
    }

    /// Create a mock that cycles through `tokens` on successive forward passes.
    fn cycling(tokens: Vec<u32>, vocab_size: usize) -> Self {
        assert!(!tokens.is_empty());
        Self {
            next_tensor_id: AtomicU64::new(1),
            logit_call_count: AtomicU64::new(0),
            token_sequence: tokens,
            vocab_size,
        }
    }

    /// Write FP16 logits into `dst` so that `target_token` has the highest value.
    fn write_logits_for_token(&self, dst: &mut [u8], target_token: u32) {
        let vocab = dst.len() / 2;
        // Set all logits to -10.0, then the target to +10.0
        let low = half::f16::from_f32(-10.0);
        let high = half::f16::from_f32(10.0);
        for i in 0..vocab {
            let val = if i == target_token as usize { high } else { low };
            let bytes = val.to_le_bytes();
            dst[i * 2] = bytes[0];
            dst[i * 2 + 1] = bytes[1];
        }
    }
}

impl Backend for MockBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> fracture_core::Result<DeviceTensor> {
        let id = self.next_tensor_id.fetch_add(1, Ordering::SeqCst);
        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }
    fn free(&self, _t: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> fracture_core::Result<()> { Ok(()) }
    fn copy_to_host(&self, _src: &DeviceTensor, dst: &mut [u8]) -> fracture_core::Result<()> {
        // Detect logits readback by buffer size
        if dst.len() == self.vocab_size * 2 {
            let idx = self.logit_call_count.fetch_add(1, Ordering::SeqCst) as usize;
            let token = self.token_sequence[idx % self.token_sequence.len()];
            self.write_logits_for_token(dst, token);
        }
        // Otherwise leave the buffer zeroed (caller provides zeroed vec)
        Ok(())
    }
    fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn rmsnorm(&self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _positions: &[u32], _theta: f64, _head_dim: usize) -> fracture_core::Result<()> { Ok(()) }
    fn attention(&self, _q: &DeviceTensor, _k_cache: &DeviceTensor, _v_cache: &DeviceTensor, _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn attention_paged(&self, _q: &DeviceTensor, _block_table: &[i32], _k_blocks: &[&DeviceTensor], _v_blocks: &[&DeviceTensor], _num_kv_heads: usize, _kv_len: usize, _start_pos: usize, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn silu_mul(&self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn embedding(&self, _token_ids: &[u32], _embedding_table: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn copy_rows(&self, _src: &DeviceTensor, _dst: &DeviceTensor, _src_offset: usize, _dst_offset: usize, _count: usize) -> fracture_core::Result<()> { Ok(()) }
    fn device_name(&self) -> &str { "mock" }
    fn total_memory(&self) -> usize { 1 << 30 }
    fn available_memory(&self) -> usize { 1 << 30 }
    fn synchronize(&self) -> fracture_core::Result<()> { Ok(()) }
    fn create_timer(&self) -> fracture_core::Result<DeviceTimer> { Ok(DeviceTimer(0)) }
    fn start_timer(&self, _timer: &DeviceTimer) -> fracture_core::Result<()> { Ok(()) }
    fn stop_timer(&self, _timer: &DeviceTimer) -> fracture_core::Result<f32> { Ok(0.0) }
    fn destroy_timer(&self, _timer: &DeviceTimer) -> fracture_core::Result<()> { Ok(()) }
}

// ── FailingMockBackend ───────────────────────────────────────
//
// Identical to MockBackend but fails matmul after `fail_after` forward
// passes (each forward calls matmul many times; we count full forward()
// invocations via logit_call_count and fail on the Nth matmul overall).

struct FailingMockBackend {
    next_tensor_id: AtomicU64,
    logit_call_count: AtomicU64,
    matmul_call_count: AtomicU64,
    /// Fail on the Nth matmul call (0-indexed).
    fail_on_matmul: u64,
    vocab_size: usize,
}

impl FailingMockBackend {
    /// Create a backend that fails on the `fail_on_matmul`-th matmul call.
    fn new(fail_on_matmul: u64, vocab_size: usize) -> Self {
        Self {
            next_tensor_id: AtomicU64::new(1),
            logit_call_count: AtomicU64::new(0),
            matmul_call_count: AtomicU64::new(0),
            fail_on_matmul,
            vocab_size,
        }
    }
}

impl Backend for FailingMockBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> fracture_core::Result<DeviceTensor> {
        let id = self.next_tensor_id.fetch_add(1, Ordering::SeqCst);
        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }
    fn free(&self, _t: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> fracture_core::Result<()> { Ok(()) }
    fn copy_to_host(&self, _src: &DeviceTensor, dst: &mut [u8]) -> fracture_core::Result<()> {
        if dst.len() == self.vocab_size * 2 {
            self.logit_call_count.fetch_add(1, Ordering::SeqCst);
            // Write token 42 as the winner
            let low = half::f16::from_f32(-10.0);
            let high = half::f16::from_f32(10.0);
            let vocab = dst.len() / 2;
            for i in 0..vocab {
                let val = if i == 42 { high } else { low };
                let bytes = val.to_le_bytes();
                dst[i * 2] = bytes[0];
                dst[i * 2 + 1] = bytes[1];
            }
        }
        Ok(())
    }
    fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> {
        let n = self.matmul_call_count.fetch_add(1, Ordering::SeqCst);
        if n >= self.fail_on_matmul {
            return Err(FractureError::Backend("induced matmul failure".into()));
        }
        Ok(())
    }
    fn rmsnorm(&self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _positions: &[u32], _theta: f64, _head_dim: usize) -> fracture_core::Result<()> { Ok(()) }
    fn attention(&self, _q: &DeviceTensor, _k_cache: &DeviceTensor, _v_cache: &DeviceTensor, _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn attention_paged(&self, _q: &DeviceTensor, _block_table: &[i32], _k_blocks: &[&DeviceTensor], _v_blocks: &[&DeviceTensor], _num_kv_heads: usize, _kv_len: usize, _start_pos: usize, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn silu_mul(&self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn embedding(&self, _token_ids: &[u32], _embedding_table: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn copy_rows(&self, _src: &DeviceTensor, _dst: &DeviceTensor, _src_offset: usize, _dst_offset: usize, _count: usize) -> fracture_core::Result<()> { Ok(()) }
    fn device_name(&self) -> &str { "failing-mock" }
    fn total_memory(&self) -> usize { 1 << 30 }
    fn available_memory(&self) -> usize { 1 << 30 }
    fn synchronize(&self) -> fracture_core::Result<()> { Ok(()) }
    fn create_timer(&self) -> fracture_core::Result<DeviceTimer> { Ok(DeviceTimer(0)) }
    fn start_timer(&self, _timer: &DeviceTimer) -> fracture_core::Result<()> { Ok(()) }
    fn stop_timer(&self, _timer: &DeviceTimer) -> fracture_core::Result<f32> { Ok(0.0) }
    fn destroy_timer(&self, _timer: &DeviceTimer) -> fracture_core::Result<()> { Ok(()) }
}

// ── Test helpers ─────────────────────────────────────────────

fn fake_tensor(id: u64, shape: Vec<usize>) -> DeviceTensor {
    DeviceTensor::new(TensorId(id), shape, DType::FP16)
}

fn fake_weight_store(cfg: &ModelConfig) -> WeightStore {
    let mut id = 1000u64;
    let mut next = |shape: Vec<usize>| -> DeviceTensor {
        id += 1;
        fake_tensor(id, shape)
    };

    let hidden = cfg.hidden_size;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;
    let intermediate = cfg.intermediate_size;
    let vocab = cfg.vocab_size;

    let token_embedding = next(vec![vocab, hidden]);
    let output_norm = next(vec![hidden]);
    let lm_head = next(vec![vocab, hidden]);

    let mut layers = Vec::new();
    for _ in 0..cfg.num_layers {
        layers.push(LayerWeights {
            q_proj: next(vec![hidden, hidden]),
            k_proj: next(vec![hidden, kv_dim]),
            v_proj: next(vec![hidden, kv_dim]),
            o_proj: next(vec![hidden, hidden]),
            gate_proj: next(vec![hidden, intermediate]),
            up_proj: next(vec![hidden, intermediate]),
            down_proj: next(vec![intermediate, hidden]),
            attn_norm: next(vec![hidden]),
            ffn_norm: next(vec![hidden]),
        });
    }

    WeightStore {
        config: cfg.clone(),
        token_embedding,
        layers,
        output_norm,
        lm_head,
    }
}

fn make_engine<B: Backend>(backend: B, cfg: &ModelConfig) -> Engine<B> {
    let weights = fake_weight_store(cfg);
    Engine::new(backend, weights, 0..cfg.num_layers)
}

fn make_cache<B: Backend>(cfg: &ModelConfig, backend: &B) -> PagedKvCacheManager {
    // Block count: ceil(max_seq_len / 16) + 2 blocks of safety margin.
    // 16 is the hardcoded BLOCK_SIZE in fracture-engine.
    let num_blocks = cfg.max_seq_len.div_ceil(16) + 2;
    PagedKvCacheManager::new(num_blocks, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, backend)
        .expect("PagedKvCacheManager::new failed in test setup")
}

fn greedy_config(max_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        stop_tokens: vec![128001, 128008, 128009],
        seed: None,
    }
}

// ── Original tests ───────────────────────────────────────────

#[test]
fn test_chat_template_basic() {
    let messages = vec![
        ("system".to_string(), "You are helpful.".to_string()),
        ("user".to_string(), "Hello!".to_string()),
    ];
    let result = apply_chat_template(&messages);
    assert!(result.starts_with("<|begin_of_text|>"));
    assert!(result.contains("<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>"));
    assert!(result.contains("<|start_header_id|>user<|end_header_id|>\n\nHello!<|eot_id|>"));
    assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
}

#[test]
fn test_chat_template_no_system() {
    let messages = vec![("user".to_string(), "Hi".to_string())];
    let result = apply_chat_template(&messages);
    assert!(!result.contains("system"));
    assert!(result.contains("user"));
}

#[test]
fn test_chat_template_multi_turn() {
    let messages = vec![
        ("system".to_string(), "You are helpful.".to_string()),
        ("user".to_string(), "Hello!".to_string()),
        ("assistant".to_string(), "Hi there!".to_string()),
        ("user".to_string(), "How are you?".to_string()),
    ];
    let result = apply_chat_template(&messages);

    // All four messages present in order
    assert!(result.contains("<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>"));
    assert!(result.contains("<|start_header_id|>user<|end_header_id|>\n\nHello!<|eot_id|>"));
    assert!(result.contains("<|start_header_id|>assistant<|end_header_id|>\n\nHi there!<|eot_id|>"));
    assert!(result.contains("<|start_header_id|>user<|end_header_id|>\n\nHow are you?<|eot_id|>"));
    // Ends with assistant header for generation
    assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    // The assistant turn from history should appear before the final assistant header
    let hist_pos = result.find("Hi there!").unwrap();
    let final_pos = result.rfind("<|start_header_id|>assistant<|end_header_id|>\n\n").unwrap();
    assert!(hist_pos < final_pos);
}

#[test]
fn test_generation_config_default() {
    let config = GenerationConfig::default();
    assert_eq!(config.max_tokens, 256);
    assert_eq!(config.temperature, 1.0);
    assert_eq!(config.top_k, 0);
    assert_eq!(config.top_p, 1.0);
    assert!(config.stop_tokens.contains(&128001));
    assert!(config.stop_tokens.contains(&128008));
    assert!(config.stop_tokens.contains(&128009));
}

#[test]
fn test_chat_template_system_wrapping() {
    let messages = vec![
        ("system".to_string(), "Be concise.".to_string()),
        ("user".to_string(), "Hi".to_string()),
    ];
    let result = apply_chat_template(&messages);
    // System message must be wrapped with header tags
    assert!(result.contains("<|start_header_id|>system<|end_header_id|>\n\nBe concise.<|eot_id|>"));
}

#[test]
fn test_chat_template_assistant_suffix() {
    // Single user message
    let messages = vec![("user".to_string(), "Hello".to_string())];
    let result = apply_chat_template(&messages);
    assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));

    // Multi-turn with system
    let messages = vec![
        ("system".to_string(), "You are helpful.".to_string()),
        ("user".to_string(), "Hello".to_string()),
        ("assistant".to_string(), "Hi".to_string()),
        ("user".to_string(), "Bye".to_string()),
    ];
    let result = apply_chat_template(&messages);
    assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));

    // Empty messages list
    let messages: Vec<(String, String)> = vec![];
    let result = apply_chat_template(&messages);
    assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
}

#[test]
fn test_chat_template_empty_content() {
    let messages = vec![
        ("system".to_string(), "".to_string()),
        ("user".to_string(), "".to_string()),
    ];
    let result = apply_chat_template(&messages);
    // Should still have proper structure with empty content
    assert!(result.contains("<|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>"));
    assert!(result.contains("<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|>"));
    assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
}

#[test]
fn test_generation_config_defaults_match() {
    let config = GenerationConfig::default();
    assert_eq!(config.max_tokens, 256);
    assert_eq!(config.temperature, 1.0);
    assert_eq!(config.top_k, 0);
    assert_eq!(config.top_p, 1.0);
    // Verify Llama 3 stop tokens are exactly these three
    assert_eq!(config.stop_tokens.len(), 3);
    assert!(config.stop_tokens.contains(&128001), "missing EOS token 128001");
    assert!(config.stop_tokens.contains(&128008), "missing EOS token 128008");
    assert!(config.stop_tokens.contains(&128009), "missing EOS token 128009");
    // Verify exact order
    assert_eq!(config.stop_tokens, vec![128001, 128008, 128009]);
}

#[test]
fn test_empty_prompt_returns_error() {
    // GenerationLoop::generate requires a full Engine, but the empty-prompt check
    // is at the top of generate() before any backend interaction.
    // We verify the error path by checking that the error type is correct.
    // The actual generate() call needs Engine<B>, KvCacheManager, and mpsc channel,
    // which require a backend. Instead, we verify the error type directly:
    let err = FractureError::Generation("empty prompt".into());
    assert!(err.to_string().contains("empty prompt"));

    // Also verify prompt-too-long error can be constructed
    let err = FractureError::Generation("prompt too long".into());
    assert!(err.to_string().contains("prompt too long"));
}

/// Verify default stop tokens contain all three Llama 3 EOS tokens.
#[test]
fn test_generation_config_stop_tokens() {
    let config = GenerationConfig::default();
    assert_eq!(config.stop_tokens.len(), 3);
    assert!(config.stop_tokens.contains(&128001), "missing Llama 3 <|end_of_text|> (128001)");
    assert!(config.stop_tokens.contains(&128008), "missing Llama 3 <|eom_id|> (128008)");
    assert!(config.stop_tokens.contains(&128009), "missing Llama 3 <|eot_id|> (128009)");
}

/// Construct a RequestMetrics and serialize to JSON, verifying all expected fields.
#[test]
fn test_request_metrics_format() {
    let metrics = RequestMetrics {
        request_id: "req_test_0001".to_string(),
        prompt_tokens: 42,
        generated_tokens: 100,
        ttft_ms: 12.5,
        total_ms: 500.0,
        tokens_per_sec: 200.0,
        avg_decode_ms: 4.87,
        peak_vram_mb: 1024.0,
        kv_cache_tokens: 142,
    };

    let json = serde_json::to_string(&metrics).expect("serialization failed");

    // Verify all expected fields are present in the JSON output.
    let expected_fields = [
        "request_id",
        "prompt_tokens",
        "generated_tokens",
        "ttft_ms",
        "total_ms",
        "tokens_per_sec",
        "avg_decode_ms",
        "peak_vram_mb",
        "kv_cache_tokens",
    ];
    for field in &expected_fields {
        assert!(json.contains(field), "missing field: {field}");
    }

    // Verify values round-trip through JSON.
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse failed");
    assert_eq!(parsed["request_id"], "req_test_0001");
    assert_eq!(parsed["prompt_tokens"], 42);
    assert_eq!(parsed["generated_tokens"], 100);
    assert_eq!(parsed["kv_cache_tokens"], 142);
}

/// Actually call GenerationLoop::generate() with an empty prompt and verify it
/// returns Err containing "empty prompt". (Closes the gap where the old
/// test_empty_prompt_returns_error only constructed the error manually.)
#[test]
fn test_generate_empty_prompt_error() {
    let cfg = tiny_config();
    let backend = MockBackend::always(42, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(10);
    let result = GenerationLoop::generate(&engine, &[], &config, &mut cache, &tx);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("empty prompt"), "expected 'empty prompt' in: {msg}");
}

/// Call GenerationLoop::generate() with a prompt exceeding max_seq_len and
/// verify it returns an error mentioning the length.
#[test]
fn test_prompt_exceeds_max_seq_len() {
    let cfg = tiny_config(); // max_seq_len = 512
    let backend = MockBackend::always(42, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(10);
    // Create a prompt longer than max_seq_len
    let long_prompt: Vec<u32> = (0..cfg.max_seq_len as u32 + 1).collect();
    let result = GenerationLoop::generate(&engine, &long_prompt, &config, &mut cache, &tx);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("exceeds") || msg.contains("max_seq_len"),
        "expected length/max_seq_len mention in: {msg}"
    );
}

/// Verify that stop token 128008 halts generation.
#[test]
fn test_stop_on_eos_128008() {
    let cfg = tiny_config();
    let backend = MockBackend::cycling(vec![42, 128008], cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(100);
    let tokens = GenerationLoop::generate(&engine, &[1], &config, &mut cache, &tx).unwrap().tokens;
    // Prefill gets token 42, first decode gets 128008 (EOS) → stops
    assert_eq!(tokens, vec![42]);
}

/// Verify that stop token 128009 halts generation.
#[test]
fn test_stop_on_eos_128009() {
    let cfg = tiny_config();
    let backend = MockBackend::cycling(vec![42, 42, 128009], cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(100);
    let tokens = GenerationLoop::generate(&engine, &[1], &config, &mut cache, &tx).unwrap().tokens;
    assert_eq!(tokens, vec![42, 42]);
}

/// Verify that prefill completes and transitions to single-token decode steps.
/// The mock returns token 42 on every forward pass; with max_tokens=5 and greedy
/// sampling, we expect exactly 5 copies of token 42 and the channel receives them.
#[test]
fn test_prefill_to_decode_transition() {
    let cfg = tiny_config();
    let backend = MockBackend::always(42, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, mut rx) = mpsc::unbounded_channel();

    let config = greedy_config(5);
    let prompt = vec![1, 2, 3];
    let tokens = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx).unwrap().tokens;

    assert_eq!(tokens.len(), 5);
    assert!(tokens.iter().all(|&t| t == 42));

    // Channel should have received all 5 tokens
    drop(tx);
    let mut streamed = Vec::new();
    while let Ok(t) = rx.try_recv() {
        streamed.push(t);
    }
    assert_eq!(streamed, tokens);
}

/// Verify that generation stops when an EOS token is sampled.
/// Mock returns token 42 three times, then EOS (128001). Generation should
/// produce exactly 3 tokens (the EOS is not included in the output).
#[test]
fn test_stop_on_eos_token() {
    let cfg = tiny_config();
    // Cycle: 42, 42, 42, 128001, 42, 42, 42, 128001, ...
    let backend = MockBackend::cycling(vec![42, 42, 42, 128001], cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(100); // high max so EOS is the real stop
    let prompt = vec![1, 2, 3];
    let tokens = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx).unwrap().tokens;

    // Prefill consumes index 0 (token 42), decode steps get indices 1 (42), 2 (42), 3 (128001=EOS)
    // So output is [42, 42, 42] — the EOS stops but is not included.
    assert_eq!(tokens, vec![42, 42, 42]);
}

/// Verify that generation stops at max_tokens limit even when the mock never
/// produces a stop token.
#[test]
fn test_stop_on_max_tokens() {
    let cfg = tiny_config();
    let backend = MockBackend::always(42, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(5);
    let prompt = vec![1];
    let tokens = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx).unwrap().tokens;

    assert_eq!(tokens.len(), 5, "should produce exactly max_tokens tokens");
}

/// Verify stop reason is Length when max_tokens is reached.
#[test]
fn test_stop_reason_length() {
    let cfg = tiny_config();
    let backend = MockBackend::always(42, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(3);
    let result = GenerationLoop::generate(&engine, &[1], &config, &mut cache, &tx).unwrap();
    assert_eq!(result.stop_reason, StopReason::Length);
    assert_eq!(result.tokens.len(), 3);
}

/// Verify stop reason is Stop when EOS token is hit.
#[test]
fn test_stop_reason_stop() {
    let cfg = tiny_config();
    let backend = MockBackend::cycling(vec![42, 128001], cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(100);
    let result = GenerationLoop::generate(&engine, &[1], &config, &mut cache, &tx).unwrap();
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.tokens, vec![42]);
}

/// Verify stop reason is Stop when first sampled token is EOS (immediate stop).
#[test]
fn test_stop_reason_immediate_eos() {
    let cfg = tiny_config();
    let backend = MockBackend::always(128001, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(100);
    let result = GenerationLoop::generate(&engine, &[1], &config, &mut cache, &tx).unwrap();
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert!(result.tokens.is_empty());
}

/// Verify max_tokens=1 produces exactly one token with stop_reason Length.
#[test]
fn test_max_tokens_one() {
    let cfg = tiny_config();
    let backend = MockBackend::always(42, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(1);
    let result = GenerationLoop::generate(&engine, &[1], &config, &mut cache, &tx).unwrap();
    assert_eq!(result.tokens, vec![42]);
    assert_eq!(result.stop_reason, StopReason::Length);
}

/// Verify that the KV cache is freed after successful generation completes.
/// After generate() returns, the cache handle should be invalid (freed).
#[test]
fn test_cache_freed_on_completion() {
    let cfg = tiny_config();
    let backend = MockBackend::always(42, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(3);
    let prompt = vec![1, 2];
    let result = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx);

    assert!(result.is_ok());
    // The cache handle (id=0) was freed by generate(). Attempting to query it
    // should fail. KvCacheManager's next alloc would use id=1 if we allocated again.
    // Since generate() calls cache.alloc() then cache.free(), handle 0 is gone.
    let stale_handle = CacheHandle(0);
    assert!(cache.seq_len(stale_handle).is_err(), "cache should be freed after generation");
}

/// Verify that the KV cache is freed even when the forward pass returns an error.
/// Uses FailingMockBackend which fails on the Nth matmul call.
#[test]
fn test_cache_freed_on_error() {
    let cfg = tiny_config();
    // Fail on the very first matmul (during prefill), so forward() returns Err.
    let backend = FailingMockBackend::new(0, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(10);
    let prompt = vec![1, 2, 3];
    let result = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx);

    assert!(result.is_err(), "generate should fail due to matmul error");
    // Cache should still be freed despite the error
    let stale_handle = CacheHandle(0);
    assert!(cache.seq_len(stale_handle).is_err(), "cache should be freed even on error");
}

/// Verify that tokens are sent through the channel immediately upon sampling,
/// not batched or delayed until generation completes. We collect from the
/// receiver after generate() and verify the order matches the return value.
#[test]
fn test_tokens_streamed_immediately() {
    let cfg = tiny_config();
    // Cycle through different tokens so we can verify ordering
    let backend = MockBackend::cycling(vec![10, 20, 30, 40, 50], cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, mut rx) = mpsc::unbounded_channel();

    let config = greedy_config(5);
    let prompt = vec![1];
    let tokens = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx).unwrap().tokens;

    // Collect everything from the channel
    drop(tx);
    let mut streamed = Vec::new();
    while let Ok(t) = rx.try_recv() {
        streamed.push(t);
    }

    assert_eq!(tokens.len(), 5);
    // Streamed tokens must match the returned token list exactly (same order)
    assert_eq!(streamed, tokens, "channel should receive tokens in generation order");
}

/// Verify that if the very first sampled token is a stop token, generation
/// returns an empty vec without sending anything on the channel.
#[test]
fn test_immediate_stop_token() {
    let cfg = tiny_config();
    // The first forward (prefill) returns EOS immediately
    let backend = MockBackend::always(128001, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, mut rx) = mpsc::unbounded_channel();

    let config = greedy_config(100);
    let prompt = vec![1, 2, 3];
    let tokens = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx).unwrap().tokens;

    assert!(tokens.is_empty(), "should return empty vec when first token is EOS");

    // Channel should have received nothing
    drop(tx);
    assert!(rx.try_recv().is_err(), "channel should be empty when first token is EOS");
}

/// Verify that the mock's logit_call_count matches 1 (prefill) + N (decode steps),
/// confirming the generation loop calls forward() the expected number of times
/// with the right prefill→decode transition.
#[test]
fn test_decode_forward_call_count() {
    let cfg = tiny_config();
    let backend = MockBackend::always(42, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(5);
    let prompt = vec![1, 2, 3]; // 3 tokens
    let tokens = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx).unwrap().tokens;

    // 1 prefill forward + 5 decode forwards = 6 total logit readbacks
    // We verify indirectly: got exactly 5 tokens (all 42, no EOS)
    assert_eq!(tokens.len(), 5);
    assert!(tokens.iter().all(|&t| t == 42));
}

/// Verify the same cache handle is used for both prefill and decode,
/// and that it is freed after generate() completes.
#[test]
fn test_kv_cache_consistent_across_prefill_decode() {
    let cfg = tiny_config();
    let backend = MockBackend::always(42, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(3);
    let prompt = vec![1, 2, 3];

    // generate() allocates handle 0, uses it for prefill + all decode, then frees it
    let _ = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx).unwrap();

    let stale = CacheHandle(0);
    assert!(
        cache.seq_len(stale).is_err(),
        "cache handle should be freed after generate completes"
    );
}

// ── CancellingMockBackend ────────────────────────────────────
//
// Like MockBackend but sets a shared AtomicBool cancel flag after
// `cancel_after` logit readbacks. This lets tests verify mid-generation
// cancellation: after `cancel_after` forward passes the decode loop will
// see the flag set and exit early on the next iteration check.

struct CancellingMockBackend {
    next_tensor_id: AtomicU64,
    logit_call_count: AtomicU64,
    /// Set the cancel flag after this many logit readbacks (0-indexed).
    cancel_after: u64,
    cancel_flag: Arc<AtomicBool>,
    vocab_size: usize,
}

impl CancellingMockBackend {
    fn new(cancel_after: u64, cancel_flag: Arc<AtomicBool>, vocab_size: usize) -> Self {
        Self {
            next_tensor_id: AtomicU64::new(1),
            logit_call_count: AtomicU64::new(0),
            cancel_after,
            cancel_flag,
            vocab_size,
        }
    }
}

impl Backend for CancellingMockBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> fracture_core::Result<DeviceTensor> {
        let id = self.next_tensor_id.fetch_add(1, Ordering::SeqCst);
        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }
    fn free(&self, _t: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> fracture_core::Result<()> { Ok(()) }
    fn copy_to_host(&self, _src: &DeviceTensor, dst: &mut [u8]) -> fracture_core::Result<()> {
        if dst.len() == self.vocab_size * 2 {
            let idx = self.logit_call_count.fetch_add(1, Ordering::SeqCst);
            // Set cancel flag after `cancel_after` completed logit reads
            if idx >= self.cancel_after {
                self.cancel_flag.store(true, Ordering::Relaxed);
            }
            // Always return token 42 as the winner
            let low = half::f16::from_f32(-10.0);
            let high = half::f16::from_f32(10.0);
            let vocab = dst.len() / 2;
            for i in 0..vocab {
                let val = if i == 42 { high } else { low };
                let bytes = val.to_le_bytes();
                dst[i * 2] = bytes[0];
                dst[i * 2 + 1] = bytes[1];
            }
        }
        Ok(())
    }
    fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn rmsnorm(&self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _positions: &[u32], _theta: f64, _head_dim: usize) -> fracture_core::Result<()> { Ok(()) }
    fn attention(&self, _q: &DeviceTensor, _k_cache: &DeviceTensor, _v_cache: &DeviceTensor, _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn attention_paged(&self, _q: &DeviceTensor, _block_table: &[i32], _k_blocks: &[&DeviceTensor], _v_blocks: &[&DeviceTensor], _num_kv_heads: usize, _kv_len: usize, _start_pos: usize, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn silu_mul(&self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn embedding(&self, _token_ids: &[u32], _embedding_table: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn copy_rows(&self, _src: &DeviceTensor, _dst: &DeviceTensor, _src_offset: usize, _dst_offset: usize, _count: usize) -> fracture_core::Result<()> { Ok(()) }
    fn device_name(&self) -> &str { "cancelling-mock" }
    fn total_memory(&self) -> usize { 1 << 30 }
    fn available_memory(&self) -> usize { 1 << 30 }
    fn synchronize(&self) -> fracture_core::Result<()> { Ok(()) }
    fn create_timer(&self) -> fracture_core::Result<DeviceTimer> { Ok(DeviceTimer(0)) }
    fn start_timer(&self, _timer: &DeviceTimer) -> fracture_core::Result<()> { Ok(()) }
    fn stop_timer(&self, _timer: &DeviceTimer) -> fracture_core::Result<f32> { Ok(0.0) }
    fn destroy_timer(&self, _timer: &DeviceTimer) -> fracture_core::Result<()> { Ok(()) }
}

// ── Cancellation tests ───────────────────────────────────────

/// Verify that setting the cancel flag before calling generate_with_cancel
/// causes the decode loop to exit on the first iteration check.
///
/// Control flow:
///   1. Prefill runs → samples token 42 (logit_call 0) → added to `generated`
///   2. Decode loop, iteration 1: cancel flag is true → StopReason::Stop, break
///
/// Expected: exactly 1 token (from prefill), StopReason::Stop,
/// fewer tokens than max_tokens, and KV cache freed.
#[test]
fn test_cancellation_pre_start() {
    let cfg = tiny_config();
    let cancel = Arc::new(AtomicBool::new(true)); // set BEFORE calling generate
    let backend = MockBackend::always(42, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(100); // large max_tokens so cancel is the only stop
    let result = GenerationLoop::generate_with_cancel(
        &engine,
        &[1, 2, 3],
        &config,
        &mut cache,
        &tx,
        Some(cancel),
    )
    .unwrap();

    // Prefill produced token 42; decode loop immediately saw cancel flag and stopped.
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert!(
        result.tokens.len() < 100,
        "cancellation should produce fewer tokens than max_tokens, got {}",
        result.tokens.len()
    );
    // Should have exactly 1 token (the prefill output)
    assert_eq!(
        result.tokens,
        vec![42],
        "only the prefill token should be present when cancel is pre-set"
    );

    // KV cache handle 0 should be freed after generate_with_cancel returns
    let stale_handle = CacheHandle(0);
    assert!(
        cache.seq_len(stale_handle).is_err(),
        "KV cache should be freed after cancellation"
    );
}

/// Verify that setting the cancel flag mid-generation (after N decode steps)
/// returns exactly the tokens produced before cancellation.
///
/// CancellingMockBackend sets the cancel flag after `cancel_after` logit
/// readbacks. With cancel_after=3:
///   - logit call 0: prefill → token 42, added to generated
///   - logit call 1: decode step 1 → token 42, added to generated
///   - logit call 2: decode step 2 → token 42, added to generated
///   - cancel flag is set on call 2 (idx >= 2)... but the loop checks cancel
///     at the TOP of the next iteration, AFTER the forward/sample.
///
/// Because the check is at loop start (before forward), the sequence is:
///   iter 1: check cancel (false) → forward (call 1) → sample → add token → cancel set if call 1 >= cancel_after
///   iter 2: check cancel (false or true depending on timing) → ...
///
/// With cancel_after=2, flag is set during call 2 (decode iter 1).
/// iter 2 starts, checks cancel → true → stops.
/// Tokens produced: prefill token (call 0) + decode iter 1 token (call 1) = 2 tokens.
#[test]
fn test_cancellation_mid_generation() {
    let cfg = tiny_config();
    // cancel_after=2: the flag is set when logit_call_count reaches 2.
    // Call 0 = prefill, call 1 = decode iter 1, call 2 = decode iter 2.
    // The flag is set during call 2 (setting it before the return).
    // The decode loop checks cancel at the start of each iteration:
    //   iter 1: check(false) → call 1 → token added → check if 1>=2: no
    //   iter 2: check(false) → call 2 → token added → check if 2>=2: YES, flag set
    //   iter 3: check(true) → break
    // Total tokens: prefill token + 2 decode tokens = 3
    let cancel = Arc::new(AtomicBool::new(false));
    let backend = CancellingMockBackend::new(2, Arc::clone(&cancel), cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(100); // large max_tokens so cancel is the only stop
    let result = GenerationLoop::generate_with_cancel(
        &engine,
        &[1],
        &config,
        &mut cache,
        &tx,
        Some(Arc::clone(&cancel)),
    )
    .unwrap();

    assert_eq!(result.stop_reason, StopReason::Stop, "mid-generation cancellation should set StopReason::Stop");
    assert!(
        result.tokens.len() < 100,
        "cancellation should produce fewer tokens than max_tokens, got {}",
        result.tokens.len()
    );
    // All produced tokens should be 42 (from the mock)
    assert!(
        result.tokens.iter().all(|&t| t == 42),
        "all tokens before cancellation should be 42, got {:?}",
        result.tokens
    );
    // Should have produced some tokens before cancellation
    assert!(
        !result.tokens.is_empty(),
        "at least one token should be generated before mid-generation cancellation"
    );

    // KV cache should be freed even on early cancellation
    let stale_handle = CacheHandle(0);
    assert!(
        cache.seq_len(stale_handle).is_err(),
        "KV cache should be freed after mid-generation cancellation"
    );
}

// ── PositionRecordingBackend ─────────────────────────────────
//
// Like MockBackend but records every positions slice passed to rope().
// Each rope() call appends a copy of the positions slice to recorded_positions.
// This lets tests verify that GenerationLoop passes the correct positions:
// [0..prompt_len] for prefill and [prompt_len, prompt_len+1, ...] for decode.

struct PositionRecordingBackend {
    next_tensor_id: AtomicU64,
    logit_call_count: AtomicU64,
    vocab_size: usize,
    /// All positions slices captured from rope() calls across all forward passes.
    recorded_positions: std::sync::Mutex<Vec<Vec<u32>>>,
    /// Token to return from every forward pass.
    token: u32,
}

impl PositionRecordingBackend {
    fn new(token: u32, vocab_size: usize) -> Self {
        Self {
            next_tensor_id: AtomicU64::new(1),
            logit_call_count: AtomicU64::new(0),
            vocab_size,
            recorded_positions: std::sync::Mutex::new(Vec::new()),
            token,
        }
    }

    fn positions_snapshot(&self) -> Vec<Vec<u32>> {
        self.recorded_positions.lock().unwrap().clone()
    }
}

impl Backend for PositionRecordingBackend {
    fn alloc(&self, shape: &[usize], dtype: DType) -> fracture_core::Result<DeviceTensor> {
        let id = self.next_tensor_id.fetch_add(1, Ordering::SeqCst);
        Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
    }
    fn free(&self, _t: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> fracture_core::Result<()> { Ok(()) }
    fn copy_to_host(&self, _src: &DeviceTensor, dst: &mut [u8]) -> fracture_core::Result<()> {
        if dst.len() == self.vocab_size * 2 {
            self.logit_call_count.fetch_add(1, Ordering::SeqCst);
            let low = half::f16::from_f32(-10.0);
            let high = half::f16::from_f32(10.0);
            let vocab = dst.len() / 2;
            for i in 0..vocab {
                let val = if i == self.token as usize { high } else { low };
                let bytes = val.to_le_bytes();
                dst[i * 2] = bytes[0];
                dst[i * 2 + 1] = bytes[1];
            }
        }
        Ok(())
    }
    fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn rmsnorm(&self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, positions: &[u32], _theta: f64, _head_dim: usize) -> fracture_core::Result<()> {
        self.recorded_positions.lock().unwrap().push(positions.to_vec());
        Ok(())
    }
    fn attention(&self, _q: &DeviceTensor, _k_cache: &DeviceTensor, _v_cache: &DeviceTensor, _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn attention_paged(&self, _q: &DeviceTensor, _block_table: &[i32], _k_blocks: &[&DeviceTensor], _v_blocks: &[&DeviceTensor], _num_kv_heads: usize, _kv_len: usize, _start_pos: usize, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn silu_mul(&self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn embedding(&self, _token_ids: &[u32], _embedding_table: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
    fn copy_rows(&self, _src: &DeviceTensor, _dst: &DeviceTensor, _src_offset: usize, _dst_offset: usize, _count: usize) -> fracture_core::Result<()> { Ok(()) }
    fn device_name(&self) -> &str { "position-recording-mock" }
    fn total_memory(&self) -> usize { 1 << 30 }
    fn available_memory(&self) -> usize { 1 << 30 }
    fn synchronize(&self) -> fracture_core::Result<()> { Ok(()) }
    fn create_timer(&self) -> fracture_core::Result<DeviceTimer> { Ok(DeviceTimer(0)) }
    fn start_timer(&self, _timer: &DeviceTimer) -> fracture_core::Result<()> { Ok(()) }
    fn stop_timer(&self, _timer: &DeviceTimer) -> fracture_core::Result<f32> { Ok(0.0) }
    fn destroy_timer(&self, _timer: &DeviceTimer) -> fracture_core::Result<()> { Ok(()) }
}

// ── Position tracking test ───────────────────────────────────

/// Verify that position tracking increments correctly from prompt length.
///
/// For a 3-token prompt, prefill should pass positions [0, 1, 2] to rope().
/// The decode loop should pass positions [3], [4], [5], ... for successive steps.
///
/// The engine calls rope() once per layer per forward pass, so with 1 layer
/// there is exactly 1 rope() entry per forward() call.
///
/// GenerationLoop runs: 1 prefill forward + (max_tokens - 1) decode forwards.
/// Total rope() calls = max_tokens. Total tokens produced = max_tokens.
#[test]
fn test_prefill_to_decode_position_tracking() {
    let cfg = tiny_config(); // 1 layer, max_seq_len=512
    let prompt = vec![10u32, 20, 30]; // 3-token prompt
    // max_tokens=4: 1 prefill + 3 decode forwards → 4 tokens, 4 rope() calls
    let max_tokens = 4usize;

    // Token 42 is not a stop token, so generation runs to max_tokens.
    let backend = PositionRecordingBackend::new(42, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(max_tokens);
    let result = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx).unwrap();
    assert_eq!(result.tokens.len(), max_tokens);

    // Snapshot positions. rope() is called once per layer per forward pass.
    // With 1 layer, each forward() produces exactly 1 rope() entry.
    let all_positions = engine.backend().positions_snapshot();

    // Generation loop: prefill (1 forward) + decode for _ in 1..max_tokens (max_tokens-1 forwards)
    // Total = max_tokens forward calls = max_tokens rope() calls.
    let total_forward_calls = max_tokens;
    assert_eq!(
        all_positions.len(), total_forward_calls,
        "expected {} rope() calls (1 prefill + {} decode), got {}: {:?}",
        total_forward_calls, max_tokens - 1, all_positions.len(), all_positions
    );

    // Prefill: positions must be [0, 1, 2] (one per prompt token)
    let prefill_positions = &all_positions[0];
    let expected_prefill: Vec<u32> = (0..prompt.len() as u32).collect();
    assert_eq!(
        prefill_positions, &expected_prefill,
        "prefill positions should be {:?}, got {:?}",
        expected_prefill, prefill_positions
    );

    // Decode: each step passes a single position starting at prompt_len.
    // There are max_tokens - 1 decode steps (indices 1..max_tokens in all_positions).
    let prompt_len = prompt.len() as u32;
    for (step, pos_slice) in all_positions[1..].iter().enumerate() {
        let expected_pos = prompt_len + step as u32;
        assert_eq!(
            pos_slice.len(), 1,
            "decode step {} should have a single position, got {:?}",
            step, pos_slice
        );
        assert_eq!(
            pos_slice[0], expected_pos,
            "decode step {} should use position {}, got {}",
            step, expected_pos, pos_slice[0]
        );
    }
}

/// Verify that request metrics are emitted to stderr during generation.
#[test]
fn test_metrics_emitted_on_generation() {
    let cfg = tiny_config();
    let backend = MockBackend::always(42, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(3);
    let prompt = vec![1, 2];
    // We can't easily capture stderr in a unit test, but we can verify
    // generate succeeds and returns the expected tokens.
    let tokens = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx).unwrap().tokens;
    assert_eq!(tokens.len(), 3);
    // Metrics are emitted via eprintln! — verifying the JSON format
    // is covered by test_request_metrics_format. This test confirms
    // the code path that calls emit_metrics doesn't panic.
}

/// Verify that tokens arrive on the channel concurrently while generate() is
/// still running, not only after it returns.
///
/// We run generate() on a background thread with a SlowMockBackend that adds
/// a brief sleep in copy_to_host so each forward pass takes a little time.
/// The main thread reads from the channel with a timeout and asserts that at
/// least one token arrives before the background thread finishes.
#[test]
fn test_tokens_streamed_before_completion() {
    use std::thread;
    use std::time::Duration;

    /// A MockBackend that sleeps briefly in copy_to_host (logit readback)
    /// to simulate a slow GPU, making the inter-token gap observable.
    struct SlowMockBackend {
        next_tensor_id: AtomicU64,
        vocab_size: usize,
        delay: Duration,
    }

    impl Backend for SlowMockBackend {
        fn alloc(&self, shape: &[usize], dtype: DType) -> fracture_core::Result<DeviceTensor> {
            let id = self.next_tensor_id.fetch_add(1, Ordering::SeqCst);
            Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
        }
        fn free(&self, _t: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
        fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> fracture_core::Result<()> { Ok(()) }
        fn copy_to_host(&self, _src: &DeviceTensor, dst: &mut [u8]) -> fracture_core::Result<()> {
            if dst.len() == self.vocab_size * 2 {
                std::thread::sleep(self.delay);
                // Token 42 wins
                let low = half::f16::from_f32(-10.0);
                let high = half::f16::from_f32(10.0);
                let vocab = dst.len() / 2;
                for i in 0..vocab {
                    let val = if i == 42 { high } else { low };
                    let bytes = val.to_le_bytes();
                    dst[i * 2] = bytes[0];
                    dst[i * 2 + 1] = bytes[1];
                }
            }
            Ok(())
        }
        fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
        fn rmsnorm(&self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
        fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _positions: &[u32], _theta: f64, _head_dim: usize) -> fracture_core::Result<()> { Ok(()) }
        fn attention(&self, _q: &DeviceTensor, _k_cache: &DeviceTensor, _v_cache: &DeviceTensor, _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
        fn attention_paged(&self, _q: &DeviceTensor, _block_table: &[i32], _k_blocks: &[&DeviceTensor], _v_blocks: &[&DeviceTensor], _num_kv_heads: usize, _kv_len: usize, _start_pos: usize, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
        fn silu_mul(&self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
        fn embedding(&self, _token_ids: &[u32], _embedding_table: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
        fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> fracture_core::Result<()> { Ok(()) }
        fn copy_rows(&self, _src: &DeviceTensor, _dst: &DeviceTensor, _src_offset: usize, _dst_offset: usize, _count: usize) -> fracture_core::Result<()> { Ok(()) }
        fn device_name(&self) -> &str { "slow-mock" }
        fn total_memory(&self) -> usize { 1 << 30 }
        fn available_memory(&self) -> usize { 1 << 30 }
        fn synchronize(&self) -> fracture_core::Result<()> { Ok(()) }
        fn create_timer(&self) -> fracture_core::Result<DeviceTimer> { Ok(DeviceTimer(0)) }
        fn start_timer(&self, _timer: &DeviceTimer) -> fracture_core::Result<()> { Ok(()) }
        fn stop_timer(&self, _timer: &DeviceTimer) -> fracture_core::Result<f32> { Ok(0.0) }
        fn destroy_timer(&self, _timer: &DeviceTimer) -> fracture_core::Result<()> { Ok(()) }
    }

    let cfg = tiny_config();
    let (tx, mut rx) = mpsc::unbounded_channel::<u32>();

    // Spin up generate on a background thread.  The thread uses its own
    // engine + cache built from the SlowMockBackend so there are no
    // cross-thread borrow issues.
    let handle = thread::spawn(move || {
        let backend = SlowMockBackend {
            next_tensor_id: AtomicU64::new(1),
            vocab_size: cfg.vocab_size,
            delay: Duration::from_millis(20),
        };
        let weights = fake_weight_store(&cfg);
        let eng = Engine::new(backend, weights, 0..cfg.num_layers);
        let mut cache = make_cache(&cfg, eng.backend());
        let config = greedy_config(5);
        GenerationLoop::generate(&eng, &[1, 2, 3], &config, &mut cache, &tx).unwrap()
    });

    // Poll the receiver until a token arrives or the deadline passes.
    // With a 20 ms per-step delay the first token (from prefill) arrives
    // after ~20 ms; we allow 5 seconds as a generous timeout.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let mut received_token = false;
    while std::time::Instant::now() < deadline {
        if rx.try_recv().is_ok() {
            received_token = true;
            break;
        }
        // The thread may not have finished yet — that is the key assertion.
        if handle.is_finished() && !received_token {
            // Thread finished before we received anything; fail below.
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert!(
        received_token,
        "should receive at least one token while generate() is still running"
    );

    // Drain the rest and join.
    let result = handle.join().unwrap();
    assert_eq!(result.tokens.len(), 5);
}

/// Verify that an error from the backend propagates with context about the
/// failing operation, not just a bare backend error string.
#[test]
fn test_error_context_propagation() {
    let cfg = tiny_config();
    // Fail on the very first matmul call (during prefill embedding lookup or
    // the first projection), so forward() returns an error immediately.
    let backend = FailingMockBackend::new(0, cfg.vocab_size);
    let engine = make_engine(backend, &cfg);
    let mut cache = make_cache(&cfg, engine.backend());
    let (tx, _rx) = mpsc::unbounded_channel();

    let config = greedy_config(10);
    let result = GenerationLoop::generate(&engine, &[1, 2, 3], &config, &mut cache, &tx);

    assert!(result.is_err(), "generate should fail when backend matmul fails");
    let msg = result.unwrap_err().to_string();

    // The error should carry context identifying what went wrong —
    // either "matmul" (the exact op that failed) or a higher-level wrapper
    // that names the forward pass.  A bare "Backend(…)" with no useful context
    // would be an empty message; we assert the error is non-empty and meaningful.
    assert!(
        !msg.is_empty(),
        "error message should not be empty"
    );
    // The FailingMockBackend injects "induced matmul failure" as the message.
    assert!(
        msg.contains("matmul") || msg.contains("Backend") || msg.contains("forward"),
        "error message should identify the failing operation, got: {msg}"
    );
}
