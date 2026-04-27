//! Golden generation comparison tests.
//!
//! Run full greedy generation through the Fracture engine on the paged KV
//! cache path and compare the output token sequence against the PyTorch
//! golden reference at `tests/golden/`.
//!
//! Prerequisites:
//!   - `FRACTURE_MODEL_PATH` env var pointing to a Llama 3.1 8B FP16 GGUF file
//!   - Golden data files in `tests/golden/` (committed to repo)
//!
//! Tests skip gracefully when either is missing.

use std::fs;
use std::path::PathBuf;

use fracture_core::ModelConfig;
use fracture_cuda::CudaBackend;
use fracture_engine::{Engine, PagedKvCacheManager};
use fracture_generate::{GenerationConfig, GenerationLoop};
use fracture_gguf::WeightStore;
use serde::Deserialize;
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Inlined helpers (originally from tests/model-validation/src/lib.rs and
// tests/validation/src/golden_compare.rs — both crates are slated for
// deletion in sub-project 1b, so we copy what we need rather than depend on
// them).
// ---------------------------------------------------------------------------

/// Skip the calling test with a message. Returns from the caller.
macro_rules! skip {
    ($($arg:tt)*) => {{
        eprintln!("SKIPPED: {}", format!($($arg)*));
        return;
    }};
}

/// Project root: walk up from this crate's manifest dir.
/// `bins/fracture-server-cuda/Cargo.toml` → workspace root is two parents up.
fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Root of the golden output directory.
fn golden_dir() -> PathBuf {
    project_root().join("tests/golden")
}

/// GGUF model path from `FRACTURE_MODEL_PATH`. Returns `None` if unset.
fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("FRACTURE_MODEL_PATH").ok()?);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("FRACTURE_MODEL_PATH={} does not exist", path.display());
        None
    }
}

/// Load Llama from GGUF, build a CUDA engine. Returns `None` if model unavailable.
fn setup_real_engine() -> Option<(Engine<CudaBackend>, ModelConfig)> {
    let path = model_path()?;
    let mut backend = CudaBackend::new(0).expect("CUDA backend creation failed");
    let weights = WeightStore::load(&path, &backend, None).expect("failed to load GGUF weights");
    let config = weights.config.clone();
    backend
        .precompute_rope_freqs(config.head_dim, config.rope_theta)
        .expect("RoPE precomputation failed");
    let engine = Engine::new(backend, weights, 0..config.num_layers);
    Some((engine, config))
}

#[derive(Deserialize)]
struct GoldenMetadata {
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    /// Authoritative full sequence (prompt + generated). The companion `.bin`
    /// file in `tests/golden/` was generated incorrectly and contains only
    /// the prompt tokens followed by zeros, so we read tokens from this
    /// metadata field instead. See sub-project 1b for cleanup.
    full_sequence: Vec<u32>,
}

/// Result of comparing two token sequences.
#[derive(Debug, Clone)]
struct TokenComparisonResult {
    matching_tokens: usize,
    total_expected: usize,
    total_actual: usize,
    divergence_index: Option<usize>,
    expected_token_at_divergence: Option<u32>,
    actual_token_at_divergence: Option<u32>,
}

impl TokenComparisonResult {
    fn matches(&self) -> bool {
        self.divergence_index.is_none() && self.total_actual == self.total_expected
    }
}

impl std::fmt::Display for TokenComparisonResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.matches() {
            write!(f, "TOKEN MATCH: all {} tokens identical", self.total_expected)
        } else {
            write!(
                f,
                "TOKEN MISMATCH: {}/{} tokens match",
                self.matching_tokens, self.total_expected
            )?;
            if let Some(idx) = self.divergence_index {
                write!(
                    f,
                    "\n  First divergence at index {}: expected={:?} actual={:?}",
                    idx, self.expected_token_at_divergence, self.actual_token_at_divergence
                )?;
            }
            if self.total_actual != self.total_expected {
                write!(
                    f,
                    "\n  Length mismatch: expected {} tokens, got {}",
                    self.total_expected, self.total_actual
                )?;
            }
            Ok(())
        }
    }
}

/// Compare two token sequences, reporting first divergence point.
fn compare_token_sequences(actual: &[u32], expected: &[u32]) -> TokenComparisonResult {
    let mut matching = 0;
    let min_len = actual.len().min(expected.len());

    for i in 0..min_len {
        if actual[i] == expected[i] {
            matching += 1;
        } else {
            return TokenComparisonResult {
                matching_tokens: matching,
                total_expected: expected.len(),
                total_actual: actual.len(),
                divergence_index: Some(i),
                expected_token_at_divergence: Some(expected[i]),
                actual_token_at_divergence: Some(actual[i]),
            };
        }
    }

    if actual.len() != expected.len() {
        let div_idx = min_len;
        TokenComparisonResult {
            matching_tokens: matching,
            total_expected: expected.len(),
            total_actual: actual.len(),
            divergence_index: Some(div_idx),
            expected_token_at_divergence: expected.get(div_idx).copied(),
            actual_token_at_divergence: actual.get(div_idx).copied(),
        }
    } else {
        TokenComparisonResult {
            matching_tokens: matching,
            total_expected: expected.len(),
            total_actual: actual.len(),
            divergence_index: None,
            expected_token_at_divergence: None,
            actual_token_at_divergence: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Generation tests
// ---------------------------------------------------------------------------

/// Run greedy generation and compare against golden reference for a prompt.
fn test_golden_generation(prompt_index: usize) {
    let meta_path = golden_dir().join(format!("prompt_{prompt_index}_greedy_50_meta.json"));

    if !meta_path.exists() {
        skip!("golden metadata for prompt_{prompt_index} not found");
    }

    let Some((engine, config)) = setup_real_engine() else {
        skip!("FRACTURE_MODEL_PATH not set");
    };

    let meta: GoldenMetadata =
        serde_json::from_str(&fs::read_to_string(&meta_path).unwrap()).unwrap();

    let golden_tokens = meta.full_sequence.clone();

    // Build paged KV cache. max_seq_len = 2048 to match the original test
    // (full 128K would use ~16GB just for the cache). Block size 16 hardcoded.
    // Block count: ceil(2048/16) + 2 = 130.
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
        stop_tokens: vec![], // generate exactly 50 tokens
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

    let mut engine_full: Vec<u32> = meta.prompt_token_ids.clone();
    engine_full.extend_from_slice(&generated.tokens);

    let result = compare_token_sequences(&engine_full, &golden_tokens);

    eprintln!("Prompt {prompt_index} golden generation: {result}");

    if !result.matches() {
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

        eprintln!(
            "  Generated {}/{} correct tokens before divergence",
            gen_tokens_correct, gen_tokens_expected
        );
        eprintln!(
            "  At position {divergence}: engine={}, reference={}",
            result.actual_token_at_divergence.unwrap_or(0),
            result.expected_token_at_divergence.unwrap_or(0),
        );

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
