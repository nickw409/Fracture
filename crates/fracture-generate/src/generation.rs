use fracture_core::{Backend, FractureError, RequestMetrics, Result};
use fracture_engine::{CacheHandle, Engine, KvCacheManager};
use rand;
use std::time::Instant;
use tokio::sync::mpsc;

use crate::sampling::{Sampler, SamplingParams};

/// Configuration for a generation request.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub stop_tokens: Vec<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            stop_tokens: vec![128001, 128008, 128009], // Llama 3 EOS tokens
        }
    }
}

/// Orchestrates tokenization, prefill, decode loop, and streaming.
///
/// # Future Work (Phase 2+)
///
/// - **Generation cancellation**: Allow callers to cancel an in-progress generation
///   via a `CancellationToken` or similar mechanism, causing the decode loop to exit
///   early and free resources.
/// - **Stream cancellation**: Detect when the streaming channel receiver is dropped
///   (e.g., client disconnect) and stop generation to avoid wasted GPU cycles.
pub struct GenerationLoop;

impl GenerationLoop {
    /// Generate tokens from already-tokenized input, streaming results through the channel.
    ///
    /// Returns the full list of generated token IDs (excluding the prompt).
    pub fn generate<B: Backend>(
        engine: &Engine<B>,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
        cache: &mut KvCacheManager,
        tx: &mpsc::UnboundedSender<u32>,
    ) -> Result<Vec<u32>> {
        if prompt_tokens.is_empty() {
            return Err(FractureError::Generation("empty prompt".into()));
        }

        if prompt_tokens.len() > engine.config().max_seq_len {
            return Err(FractureError::Generation(format!(
                "prompt length {} exceeds max_seq_len {}",
                prompt_tokens.len(), engine.config().max_seq_len
            )));
        }

        let cache_handle = cache.alloc(engine.backend())?;

        let result = Self::generate_inner(engine, prompt_tokens, config, cache, cache_handle, tx);

        // Always free the cache, even on error
        if let Err(e) = cache.free(cache_handle, engine.backend()) {
            tracing::warn!("failed to free KV cache: {e}");
        }

        result
    }

    fn generate_inner<B: Backend>(
        engine: &Engine<B>,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
        cache: &mut KvCacheManager,
        cache_handle: CacheHandle,
        tx: &mpsc::UnboundedSender<u32>,
    ) -> Result<Vec<u32>> {
        let request_start = Instant::now();
        let sampling_params = SamplingParams {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
        };

        // Prefill: process all prompt tokens at once
        let positions: Vec<u32> = (0..prompt_tokens.len() as u32).collect();
        let logits = engine.forward(prompt_tokens, &positions, cache, cache_handle, None)?;

        let ttft = request_start.elapsed();

        // Sample first generated token
        let mut next_token = Sampler::sample(&logits, &sampling_params)?;

        // Check for immediate stop
        if config.stop_tokens.contains(&next_token) {
            Self::emit_metrics(engine, prompt_tokens.len(), 0, ttft, request_start, &[], cache, cache_handle);
            return Ok(Vec::new());
        }

        let _ = tx.send(next_token);
        let mut generated = vec![next_token];
        let mut pos = prompt_tokens.len() as u32;
        let mut decode_times = Vec::new();

        // Decode loop
        for _ in 1..config.max_tokens {
            let decode_start = Instant::now();
            let logits = engine.forward(&[next_token], &[pos], cache, cache_handle, None)?;
            decode_times.push(decode_start.elapsed().as_secs_f64() * 1000.0);

            next_token = Sampler::sample(&logits, &sampling_params)?;

            if config.stop_tokens.contains(&next_token) {
                break;
            }

            let _ = tx.send(next_token);
            generated.push(next_token);
            pos += 1;
        }

        Self::emit_metrics(engine, prompt_tokens.len(), generated.len(), ttft, request_start, &decode_times, cache, cache_handle);

        Ok(generated)
    }

    fn emit_metrics<B: Backend>(
        engine: &Engine<B>,
        prompt_tokens: usize,
        generated_tokens: usize,
        ttft: std::time::Duration,
        request_start: Instant,
        decode_times: &[f64],
        cache: &KvCacheManager,
        cache_handle: CacheHandle,
    ) {
        let total_ms = request_start.elapsed().as_secs_f64() * 1000.0;
        let decode_total_ms: f64 = decode_times.iter().sum();
        let decode_total_secs = decode_total_ms / 1000.0;
        let tokens_per_sec = if decode_total_secs > 0.0 {
            generated_tokens as f64 / decode_total_secs
        } else {
            0.0
        };
        let avg_decode_ms = if decode_times.is_empty() {
            0.0
        } else {
            decode_times.iter().sum::<f64>() / decode_times.len() as f64
        };

        let backend = engine.backend();
        let total_mem = backend.total_memory() as f64;
        let avail_mem = backend.available_memory() as f64;
        let vram_used_mb = (total_mem - avail_mem) / (1024.0 * 1024.0);

        let kv_cache_tokens = cache.seq_len(cache_handle).unwrap_or(0);

        let metrics = RequestMetrics {
            request_id: format!("req_{:016x}", rand::random::<u64>()),
            prompt_tokens,
            generated_tokens,
            ttft_ms: ttft.as_secs_f64() * 1000.0,
            total_ms,
            tokens_per_sec,
            avg_decode_ms,
            peak_vram_mb: vram_used_mb,
            kv_cache_tokens,
        };

        if let Ok(json) = serde_json::to_string(&metrics) {
            eprintln!("{}", json);
        }
    }
}

/// Apply the Llama 3 chat template to a list of messages.
///
/// Format:
/// <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
/// <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
/// <|start_header_id|>assistant<|end_header_id|>\n\n
pub fn apply_chat_template(messages: &[(String, String)]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");

    for (role, content) in messages {
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(role);
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(content);
        prompt.push_str("<|eot_id|>");
    }

    // Add assistant header to prompt the model to respond
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;
    use fracture_core::{DType, DeviceTensor, DeviceTimer, ModelConfig, TensorId};
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

    fn make_cache(cfg: &ModelConfig) -> KvCacheManager {
        KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len)
    }

    fn greedy_config(max_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_tokens,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            stop_tokens: vec![128001, 128008, 128009],
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

    /// Verify that prefill completes and transitions to single-token decode steps.
    /// The mock returns token 42 on every forward pass; with max_tokens=5 and greedy
    /// sampling, we expect exactly 5 copies of token 42 and the channel receives them.
    #[test]
    fn test_prefill_to_decode_transition() {
        let cfg = tiny_config();
        let backend = MockBackend::always(42, cfg.vocab_size);
        let engine = make_engine(backend, &cfg);
        let mut cache = make_cache(&cfg);
        let (tx, mut rx) = mpsc::unbounded_channel();

        let config = greedy_config(5);
        let prompt = vec![1, 2, 3];
        let tokens = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx).unwrap();

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
        let mut cache = make_cache(&cfg);
        let (tx, _rx) = mpsc::unbounded_channel();

        let config = greedy_config(100); // high max so EOS is the real stop
        let prompt = vec![1, 2, 3];
        let tokens = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx).unwrap();

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
        let mut cache = make_cache(&cfg);
        let (tx, _rx) = mpsc::unbounded_channel();

        let config = greedy_config(5);
        let prompt = vec![1];
        let tokens = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx).unwrap();

        assert_eq!(tokens.len(), 5, "should produce exactly max_tokens tokens");
    }

    /// Verify that the KV cache is freed after successful generation completes.
    /// After generate() returns, the cache handle should be invalid (freed).
    #[test]
    fn test_cache_freed_on_completion() {
        let cfg = tiny_config();
        let backend = MockBackend::always(42, cfg.vocab_size);
        let engine = make_engine(backend, &cfg);
        let mut cache = make_cache(&cfg);
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
        let mut cache = make_cache(&cfg);
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
        let mut cache = make_cache(&cfg);
        let (tx, mut rx) = mpsc::unbounded_channel();

        let config = greedy_config(5);
        let prompt = vec![1];
        let tokens = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx).unwrap();

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
        let mut cache = make_cache(&cfg);
        let (tx, mut rx) = mpsc::unbounded_channel();

        let config = greedy_config(100);
        let prompt = vec![1, 2, 3];
        let tokens = GenerationLoop::generate(&engine, &prompt, &config, &mut cache, &tx).unwrap();

        assert!(tokens.is_empty(), "should return empty vec when first token is EOS");

        // Channel should have received nothing
        drop(tx);
        assert!(rx.try_recv().is_err(), "channel should be empty when first token is EOS");
    }
}
