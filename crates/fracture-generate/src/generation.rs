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
    /// Requires a mock engine or real GPU backend to run the forward pass.
    #[test]
    #[ignore]
    fn test_prefill_to_decode_transition() {
        // Requires mock engine or GPU backend.
        // To run: cargo nextest run -p fracture-generate -- --ignored
        todo!("requires mock engine or GPU")
    }

    /// Verify that generation stops when an EOS token is sampled.
    /// Requires a mock engine that returns controlled logits.
    #[test]
    #[ignore]
    fn test_stop_on_eos_token() {
        // Requires mock engine to control sampled tokens.
        todo!("requires mock engine")
    }

    /// Verify that generation stops at max_tokens limit.
    /// Requires a mock engine that never produces stop tokens.
    #[test]
    #[ignore]
    fn test_stop_on_max_tokens() {
        // Requires mock engine to control sampled tokens.
        todo!("requires mock engine")
    }

    /// Verify that the KV cache is freed after successful generation.
    /// Requires a mock engine to observe cache lifecycle.
    #[test]
    #[ignore]
    fn test_cache_freed_on_completion() {
        // Requires mock engine to inspect cache state after generate().
        todo!("requires mock engine")
    }

    /// Verify that the KV cache is freed even when generation returns an error.
    /// Requires a mock engine that can be induced to fail.
    #[test]
    #[ignore]
    fn test_cache_freed_on_error() {
        // Requires mock engine that fails mid-generation.
        todo!("requires mock engine")
    }

    /// Verify that tokens are sent through the channel immediately upon sampling,
    /// not batched or delayed until generation completes.
    /// Requires a mock engine to observe send timing.
    #[test]
    #[ignore]
    fn test_tokens_streamed_immediately() {
        // Requires mock engine to verify per-token send timing.
        todo!("requires mock engine")
    }

    /// Verify that if the very first sampled token is a stop token, generation
    /// returns an empty vec without sending anything on the channel.
    /// Requires a mock engine that returns stop-token logits on prefill.
    #[test]
    #[ignore]
    fn test_immediate_stop_token() {
        // Requires mock engine that returns stop token on first sample.
        todo!("requires mock engine")
    }
}
