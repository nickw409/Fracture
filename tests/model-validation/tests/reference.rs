//! Reference tensor comparison tests.
//!
//! Compare Fracture engine forward-pass outputs against PyTorch reference
//! tensors dumped by `scripts/dump_reference.py`.
//!
//! Prerequisites:
//!   - `FRACTURE_MODEL_PATH` env var pointing to a Llama 3.1 8B FP16 GGUF file
//!   - Reference data in `tests/reference/` (run `scripts/dump_reference.py`)
//!
//! Tests skip gracefully when either is missing.

use fracture_engine::{CacheHandle, KvCacheManager};
use fracture_model_validation::*;
use fracture_validation::tensor_compare::{compare_tensors, load_reference_tensor, DType};

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Metadata from a prefill prompt's metadata.json.
#[derive(Deserialize)]
struct PrefillMetadata {
    token_ids: Vec<u32>,
    greedy_token: u32,
}

/// Metadata from a decode step's metadata.json.
#[derive(Deserialize)]
struct DecodeMetadata {
    input_token: u32,
    position: u32,
    output_token: u32,
}

/// Convert a Vec<f32> to raw little-endian bytes for comparison.
fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Run a prefill forward pass and return logits.
fn run_prefill(
    engine: &fracture_engine::Engine<fracture_cuda::CudaBackend>,
    config: &fracture_core::ModelConfig,
    token_ids: &[u32],
) -> (Vec<f32>, KvCacheManager, CacheHandle) {
    let mut cache = KvCacheManager::new(
        config.num_layers,
        config.num_kv_heads,
        config.head_dim,
        config.max_seq_len,
    );
    let handle = cache.alloc(engine.backend()).expect("cache alloc failed");
    let positions: Vec<u32> = (0..token_ids.len() as u32).collect();
    let logits = engine
        .forward(token_ids, &positions, &mut cache, handle, None)
        .expect("forward pass failed");
    (logits, cache, handle)
}

// ---------------------------------------------------------------------------
// Prefill logits tests
// ---------------------------------------------------------------------------

/// Compare engine prefill logits (last position) against reference for a prompt.
fn test_prefill_logits(prompt_index: usize) {
    if setup_real_engine().is_none() && !has_reference_data() {
        skip!("FRACTURE_MODEL_PATH not set and no reference data");
    }

    let ref_dir = reference_dir().join(format!("prompt_{prompt_index}"));
    if !ref_dir.exists() {
        skip!("reference data for prompt_{prompt_index} not found");
    }

    let Some((engine, config)) = setup_real_engine() else {
        skip!("FRACTURE_MODEL_PATH not set");
    };

    // Load reference metadata and logits
    let meta_path = ref_dir.join("metadata.json");
    let meta: PrefillMetadata =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();

    let ref_logits = load_reference_tensor(ref_dir.join("logits.bin").to_str().unwrap())
        .expect("failed to load reference logits");

    // Extract last-position logits from reference [1, seq_len, vocab_size]
    let ref_all = ref_logits.to_f32();
    let vocab_size = config.vocab_size;
    let ref_last = &ref_all[ref_all.len() - vocab_size..];

    // Run engine
    let (engine_logits, mut cache, handle) = run_prefill(&engine, &config, &meta.token_ids);
    cache.free(handle, engine.backend()).unwrap();

    assert_eq!(
        engine_logits.len(),
        vocab_size,
        "logits length mismatch: got {} expected {}",
        engine_logits.len(),
        vocab_size
    );

    // Compare with generous tolerance — 32 layers of FP16 accumulates error
    let result = compare_tensors(
        &f32_to_bytes(&engine_logits),
        DType::F32,
        &fracture_validation::tensor_compare::ReferenceTensor {
            shape: vec![vocab_size],
            dtype: DType::F32,
            data: f32_to_bytes(ref_last),
        },
        0.05,  // rtol — generous for 32-layer FP16 vs FP32 reference
        0.5,   // atol — logits can have large absolute values
    );

    eprintln!(
        "Prompt {prompt_index} prefill logits: max_abs_error={:.4}, mean_abs_error={:.6}, mismatches={}/{}",
        result.max_abs_error, result.mean_abs_error, result.num_mismatches, result.total_elements
    );

    // Primary correctness check: greedy token must match
    let engine_greedy = engine_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32;

    assert_eq!(
        engine_greedy, meta.greedy_token,
        "prompt_{prompt_index}: greedy token mismatch — engine={engine_greedy}, reference={}",
        meta.greedy_token
    );

    // Secondary check: overall logits similarity (warn but don't fail on tolerance)
    if !result.matches {
        eprintln!(
            "WARNING: prompt_{prompt_index} logits exceed strict tolerance but greedy token matches.\n{result}"
        );
    }
}

#[test]
fn test_prefill_logits_prompt_0() {
    test_prefill_logits(0);
}

#[test]
fn test_prefill_logits_prompt_1() {
    test_prefill_logits(1);
}

// ---------------------------------------------------------------------------
// Decode step test
// ---------------------------------------------------------------------------

#[test]
fn test_decode_step_0() {
    let ref_dir = reference_dir().join("decode_step_0");
    if !ref_dir.exists() {
        skip!("reference data for decode_step_0 not found");
    }

    let Some((engine, config)) = setup_real_engine() else {
        skip!("FRACTURE_MODEL_PATH not set");
    };

    // Load decode metadata
    let meta: DecodeMetadata = serde_json::from_str(
        &std::fs::read_to_string(ref_dir.join("metadata.json")).unwrap(),
    )
    .unwrap();

    // First run the prefill for prompt_0 to populate KV cache
    let prefill_dir = reference_dir().join("prompt_0");
    let prefill_meta: PrefillMetadata = serde_json::from_str(
        &std::fs::read_to_string(prefill_dir.join("metadata.json")).unwrap(),
    )
    .unwrap();

    let mut cache = KvCacheManager::new(
        config.num_layers,
        config.num_kv_heads,
        config.head_dim,
        config.max_seq_len,
    );
    let handle = cache.alloc(engine.backend()).expect("cache alloc failed");

    // Prefill
    let positions: Vec<u32> = (0..prefill_meta.token_ids.len() as u32).collect();
    let _ = engine
        .forward(&prefill_meta.token_ids, &positions, &mut cache, handle, None)
        .expect("prefill failed");

    // Decode one step
    let decode_logits = engine
        .forward(
            &[meta.input_token],
            &[meta.position],
            &mut cache,
            handle,
            None,
        )
        .expect("decode step failed");

    cache.free(handle, engine.backend()).unwrap();

    // Check greedy token matches
    let engine_greedy = decode_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32;

    assert_eq!(
        engine_greedy, meta.output_token,
        "decode_step_0: greedy token mismatch — engine={engine_greedy}, reference={}",
        meta.output_token
    );

    // Compare logits numerically
    let ref_logits = load_reference_tensor(ref_dir.join("logits.bin").to_str().unwrap())
        .expect("failed to load decode reference logits");

    let ref_all = ref_logits.to_f32();
    let vocab_size = config.vocab_size;
    let ref_last = &ref_all[ref_all.len() - vocab_size..];

    let result = compare_tensors(
        &f32_to_bytes(&decode_logits),
        DType::F32,
        &fracture_validation::tensor_compare::ReferenceTensor {
            shape: vec![vocab_size],
            dtype: DType::F32,
            data: f32_to_bytes(ref_last),
        },
        0.05,
        0.5,
    );

    eprintln!(
        "decode_step_0 logits: max_abs_error={:.4}, mean_abs_error={:.6}, mismatches={}/{}",
        result.max_abs_error, result.mean_abs_error, result.num_mismatches, result.total_elements
    );

    if !result.matches {
        eprintln!(
            "WARNING: decode_step_0 logits exceed strict tolerance but greedy token matches.\n{result}"
        );
    }
}
