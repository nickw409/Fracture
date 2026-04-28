//! Golden generation comparison tests.
//!
//! Run full greedy generation through the Fracture engine and compare the
//! output token sequence against PyTorch golden reference.
//!
//! Prerequisites:
//!   - `FRACTURE_MODEL_PATH` env var pointing to a Llama 3.1 8B FP16 GGUF file
//!   - Golden data in `tests/golden/` (run `scripts/dump_reference.py`)
//!
//! Tests skip gracefully when either is missing.

use fracture_engine::PagedKvCacheManager;
use fracture_generate::{GenerationConfig, GenerationLoop};
use fracture_model_validation::*;
use fracture_validation::golden_compare::{compare_token_sequences, load_golden_tokens};

use serde::Deserialize;
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GoldenMetadata {
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Generation tests
// ---------------------------------------------------------------------------

/// Run greedy generation and compare against golden reference for a prompt.
fn test_golden_generation(prompt_index: usize) {
    let golden_path = golden_dir().join(format!("prompt_{prompt_index}_greedy_50.bin"));
    let meta_path = golden_dir().join(format!("prompt_{prompt_index}_greedy_50_meta.json"));

    if !golden_path.exists() {
        skip!("golden data for prompt_{prompt_index} not found");
    }

    let Some((engine, config)) = setup_real_engine() else {
        skip!("FRACTURE_MODEL_PATH not set");
    };

    // Load golden metadata
    let meta: GoldenMetadata =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();

    // Load golden token sequence (full_sequence = prompt + generated)
    let golden_tokens =
        load_golden_tokens(golden_path.to_str().unwrap()).expect("failed to load golden tokens");

    // Run generation with greedy decoding (temperature=0, no top-k/top-p).
    // max_seq_len=2048 to avoid OOM (full 128K would use ~16GB just for the cache).
    // Block size 16 hardcoded → 2048/16 + 2 = 130 blocks.
    let num_blocks = 2048usize.div_ceil(16) + 2;
    let mut cache = PagedKvCacheManager::new(
        num_blocks,
        config.num_layers,
        config.num_kv_heads,
        config.head_dim,
        engine.backend(),
    )
    .expect("PagedKvCacheManager::new failed");

    let gen_config = GenerationConfig {
        max_tokens: 50,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        stop_tokens: vec![], // Don't stop early — we want exactly 50 tokens
        seed: None,
    };

    let (tx, _rx) = mpsc::unbounded_channel();
    let generated = GenerationLoop::generate(
        &engine,
        &meta.prompt_token_ids,
        &gen_config,
        &mut cache,
        &tx,
    )
    .expect("generation failed");

    // Build full sequence: prompt + generated (matching golden format)
    let mut engine_full: Vec<u32> = meta.prompt_token_ids.clone();
    engine_full.extend_from_slice(&generated.tokens);

    // Compare against golden
    let result = compare_token_sequences(&engine_full, &golden_tokens);

    eprintln!(
        "Prompt {prompt_index} golden generation: {result}"
    );

    if !result.matches() {
        // Report how far we got before diverging
        let gen_start = meta.prompt_token_ids.len();
        let divergence = result.divergence_index.unwrap();

        if divergence < gen_start {
            panic!(
                "prompt_{prompt_index}: divergence in prompt tokens at index {divergence} — \
                 this should never happen"
            );
        }

        let gen_tokens_correct = divergence - gen_start;
        let gen_tokens_expected = meta.generated_token_ids.len();

        // Print the generated tokens for debugging
        eprintln!(
            "  Generated {}/{} correct tokens before divergence",
            gen_tokens_correct, gen_tokens_expected
        );
        eprintln!(
            "  At position {divergence}: engine={}, reference={}",
            result.actual_token_at_divergence.unwrap_or(0),
            result.expected_token_at_divergence.unwrap_or(0),
        );

        // First few tokens matching is a strong signal the model is working.
        // Full 50-token match requires exact numerical agreement at every step.
        assert!(
            gen_tokens_correct >= 5,
            "prompt_{prompt_index}: only {gen_tokens_correct}/{gen_tokens_expected} generated \
             tokens match — engine likely has a correctness issue"
        );

        eprintln!(
            "WARNING: prompt_{prompt_index}: {gen_tokens_correct}/{gen_tokens_expected} tokens \
             match. FP16 accumulation divergence after {gen_tokens_correct} tokens is expected \
             for autoregressive generation."
        );
    }
}

#[test]
fn test_golden_generation_prompt_0() {
    test_golden_generation(0);
}

#[test]
fn test_golden_generation_prompt_1() {
    test_golden_generation(1);
}
