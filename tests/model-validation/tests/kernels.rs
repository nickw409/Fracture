//! Tier 2: Per-kernel validation against PyTorch reference tensors.
//!
//! Each test loads a specific kernel's input from the reference dump,
//! runs the kernel through the Backend trait with real model weights,
//! and compares the output against the PyTorch reference.
//!
//! Prerequisites:
//!   - `FRACTURE_MODEL_PATH` env var pointing to a Llama 3.1 8B FP16 GGUF file
//!   - Reference data in `tests/reference/` (run `scripts/dump_reference.py`)

use fracture_core::{Backend, DType, ModelConfig};
use fracture_cuda::CudaBackend;
use fracture_gguf::WeightStore;
use fracture_model_validation::*;
use fracture_validation::tensor_compare::{
    compare_tensors, load_reference_tensor, DType as RefDType, ReferenceTensor,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load model weights and backend. Returns None if prerequisites missing.
fn setup_backend_and_weights() -> Option<(CudaBackend, WeightStore, ModelConfig)> {
    let path = model_path()?;
    if !has_reference_data() {
        return None;
    }
    let mut backend = CudaBackend::new(0).expect("CUDA backend creation failed");
    let weights = WeightStore::load(&path, &backend, None).expect("failed to load GGUF weights");
    let config = weights.config.clone();
    backend
        .precompute_rope_freqs(config.head_dim, config.rope_theta)
        .expect("RoPE precomputation failed");
    Some((backend, weights, config))
}

/// Reference tensor path for prompt_0 prefill.
fn prefill_ref(relative: &str) -> String {
    reference_dir()
        .join("prompt_0")
        .join(relative)
        .to_str()
        .unwrap()
        .to_string()
}

/// Reference tensor path for decode_step_0.
fn decode_ref(relative: &str) -> String {
    reference_dir()
        .join("decode_step_0")
        .join(relative)
        .to_str()
        .unwrap()
        .to_string()
}

/// Load a reference tensor, stripping a leading batch dimension of 1.
fn load_ref(path: &str) -> ReferenceTensor {
    let t = load_reference_tensor(path).unwrap_or_else(|e| panic!("load {path}: {e}"));
    // Strip batch dim [1, ...] -> [...]
    if t.shape.len() > 1 && t.shape[0] == 1 {
        ReferenceTensor {
            shape: t.shape[1..].to_vec(),
            ..t
        }
    } else {
        t
    }
}

/// Convert f32 reference tensor to FP16 bytes for GPU upload.
fn ref_to_fp16_bytes(t: &ReferenceTensor) -> Vec<u8> {
    t.to_f32()
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect()
}

/// Upload a reference tensor to GPU as FP16, using the given shape.
fn upload_ref(backend: &CudaBackend, t: &ReferenceTensor, shape: &[usize]) -> fracture_core::DeviceTensor {
    let tensor = backend.alloc(shape, DType::FP16).unwrap();
    let bytes = ref_to_fp16_bytes(t);
    backend.copy_to_device(&tensor, &bytes).unwrap();
    tensor
}

/// Download a GPU tensor as raw FP16 bytes.
fn download_fp16(backend: &CudaBackend, t: &fracture_core::DeviceTensor) -> Vec<u8> {
    let mut buf = vec![0u8; t.size_bytes()];
    backend.copy_to_host(t, &mut buf).unwrap();
    buf
}

/// Compare GPU output (FP16) against reference (F32) and assert closeness.
/// Returns the ComparisonResult for additional inspection.
fn assert_kernel_close(
    backend: &CudaBackend,
    actual: &fracture_core::DeviceTensor,
    expected: &ReferenceTensor,
    rtol: f32,
    atol: f32,
    label: &str,
) {
    let actual_bytes = download_fp16(backend, actual);
    let result = compare_tensors(&actual_bytes, RefDType::F16, expected, rtol, atol);
    eprintln!(
        "{label}: max_err={:.6}, mean_err={:.6}, mismatches={}/{}",
        result.max_abs_error, result.mean_abs_error, result.num_mismatches, result.total_elements
    );
    assert!(
        result.matches,
        "{label}: kernel output exceeds tolerance\n{result}"
    );
}

// Standard FP16 tolerances — single kernel, no error accumulation.
const RTOL: f32 = 1e-3;
const ATOL: f32 = 1e-3;

// Slightly looser for numerically sensitive ops (norms, softmax).
const LOOSE_RTOL: f32 = 5e-3;
const LOOSE_ATOL: f32 = 5e-3;

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

#[test]
fn test_kernel_embedding() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    let expected = load_ref(&prefill_ref("embeddings.bin"));
    let seq_len = expected.shape[0]; // 6
    let hidden = config.hidden_size;

    // Token IDs from reference
    let token_ids_ref = load_ref(&prefill_ref("token_ids.bin"));
    let token_ids: Vec<u32> = token_ids_ref.to_f32().iter().map(|&v| v as u32).collect();

    let output = backend.alloc(&[seq_len, hidden], DType::FP16).unwrap();
    backend
        .embedding(&token_ids, &weights.token_embedding, &output)
        .unwrap();

    assert_kernel_close(&backend, &output, &expected, RTOL, ATOL, "embedding");
    backend.free(&output).unwrap();
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

#[test]
fn test_kernel_rmsnorm_attn_layer0() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    let input = load_ref(&prefill_ref("layer_00/input_hidden.bin"));
    let expected = load_ref(&prefill_ref("layer_00/post_attn_norm.bin"));
    let seq_len = input.shape[0];
    let hidden = config.hidden_size;

    let dev_input = upload_ref(&backend, &input, &[seq_len, hidden]);
    let dev_output = backend.alloc(&[seq_len, hidden], DType::FP16).unwrap();

    backend
        .rmsnorm(
            &dev_input,
            &weights.layers[0].attn_norm,
            config.rms_norm_eps,
            &dev_output,
        )
        .unwrap();

    assert_kernel_close(&backend, &dev_output, &expected, LOOSE_RTOL, LOOSE_ATOL, "rmsnorm_attn_layer0");
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_rmsnorm_ffn_layer0() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    let input = load_ref(&prefill_ref("layer_00/post_attn_residual.bin"));
    let expected = load_ref(&prefill_ref("layer_00/post_ffn_norm.bin"));
    let seq_len = input.shape[0];
    let hidden = config.hidden_size;

    let dev_input = upload_ref(&backend, &input, &[seq_len, hidden]);
    let dev_output = backend.alloc(&[seq_len, hidden], DType::FP16).unwrap();

    backend
        .rmsnorm(
            &dev_input,
            &weights.layers[0].ffn_norm,
            config.rms_norm_eps,
            &dev_output,
        )
        .unwrap();

    assert_kernel_close(&backend, &dev_output, &expected, LOOSE_RTOL, LOOSE_ATOL, "rmsnorm_ffn_layer0");
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_rmsnorm_attn_last_layer() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    let last = config.num_layers - 1;
    let input = load_ref(&prefill_ref(&format!("layer_{last:02}/input_hidden.bin")));
    let expected = load_ref(&prefill_ref(&format!("layer_{last:02}/post_attn_norm.bin")));
    let seq_len = input.shape[0];
    let hidden = config.hidden_size;

    let dev_input = upload_ref(&backend, &input, &[seq_len, hidden]);
    let dev_output = backend.alloc(&[seq_len, hidden], DType::FP16).unwrap();

    backend
        .rmsnorm(
            &dev_input,
            &weights.layers[last].attn_norm,
            config.rms_norm_eps,
            &dev_output,
        )
        .unwrap();

    assert_kernel_close(
        &backend,
        &dev_output,
        &expected,
        LOOSE_RTOL,
        LOOSE_ATOL,
        &format!("rmsnorm_attn_layer{last}"),
    );
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

// ---------------------------------------------------------------------------
// MatMul — QKV projections (layer 0)
// ---------------------------------------------------------------------------

#[test]
fn test_kernel_matmul_q_proj_layer0() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    let input = load_ref(&prefill_ref("layer_00/post_attn_norm.bin"));
    let expected = load_ref(&prefill_ref("layer_00/q.bin"));
    let seq_len = input.shape[0];

    let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
    let dev_output = backend
        .alloc(&[seq_len, config.hidden_size], DType::FP16)
        .unwrap();

    backend
        .matmul(&dev_input, &weights.layers[0].q_proj, &dev_output)
        .unwrap();

    assert_kernel_close(&backend, &dev_output, &expected, RTOL, ATOL, "matmul_q_proj_layer0");
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_matmul_k_proj_layer0() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    let input = load_ref(&prefill_ref("layer_00/post_attn_norm.bin"));
    let expected = load_ref(&prefill_ref("layer_00/k.bin"));
    let seq_len = input.shape[0];
    let kv_dim = config.num_kv_heads * config.head_dim;

    let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
    let dev_output = backend.alloc(&[seq_len, kv_dim], DType::FP16).unwrap();

    backend
        .matmul(&dev_input, &weights.layers[0].k_proj, &dev_output)
        .unwrap();

    assert_kernel_close(&backend, &dev_output, &expected, RTOL, ATOL, "matmul_k_proj_layer0");
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_matmul_v_proj_layer0() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    let input = load_ref(&prefill_ref("layer_00/post_attn_norm.bin"));
    let expected = load_ref(&prefill_ref("layer_00/v.bin"));
    let seq_len = input.shape[0];
    let kv_dim = config.num_kv_heads * config.head_dim;

    let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
    let dev_output = backend.alloc(&[seq_len, kv_dim], DType::FP16).unwrap();

    backend
        .matmul(&dev_input, &weights.layers[0].v_proj, &dev_output)
        .unwrap();

    assert_kernel_close(&backend, &dev_output, &expected, RTOL, ATOL, "matmul_v_proj_layer0");
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

// ---------------------------------------------------------------------------
// MatMul — FFN projections (layer 0)
// ---------------------------------------------------------------------------

#[test]
fn test_kernel_matmul_gate_proj_layer0() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    let input = load_ref(&prefill_ref("layer_00/post_ffn_norm.bin"));
    let expected = load_ref(&prefill_ref("layer_00/gate.bin"));
    let seq_len = input.shape[0];

    let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
    let dev_output = backend
        .alloc(&[seq_len, config.intermediate_size], DType::FP16)
        .unwrap();

    backend
        .matmul(&dev_input, &weights.layers[0].gate_proj, &dev_output)
        .unwrap();

    assert_kernel_close(&backend, &dev_output, &expected, RTOL, ATOL, "matmul_gate_proj_layer0");
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_matmul_up_proj_layer0() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    let input = load_ref(&prefill_ref("layer_00/post_ffn_norm.bin"));
    let expected = load_ref(&prefill_ref("layer_00/up.bin"));
    let seq_len = input.shape[0];

    let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
    let dev_output = backend
        .alloc(&[seq_len, config.intermediate_size], DType::FP16)
        .unwrap();

    backend
        .matmul(&dev_input, &weights.layers[0].up_proj, &dev_output)
        .unwrap();

    assert_kernel_close(&backend, &dev_output, &expected, RTOL, ATOL, "matmul_up_proj_layer0");
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_matmul_down_proj_layer0() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    let input = load_ref(&prefill_ref("layer_00/silu_mul.bin"));
    let expected = load_ref(&prefill_ref("layer_00/ffn_output.bin"));
    let seq_len = input.shape[0];

    let dev_input = upload_ref(&backend, &input, &[seq_len, config.intermediate_size]);
    let dev_output = backend
        .alloc(&[seq_len, config.hidden_size], DType::FP16)
        .unwrap();

    backend
        .matmul(&dev_input, &weights.layers[0].down_proj, &dev_output)
        .unwrap();

    assert_kernel_close(&backend, &dev_output, &expected, RTOL, ATOL, "matmul_down_proj_layer0");
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

// ---------------------------------------------------------------------------
// SiLU × Mul
// ---------------------------------------------------------------------------

#[test]
fn test_kernel_silu_mul_layer0() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };
    let _ = weights; // only needed for model loading side-effect

    let gate = load_ref(&prefill_ref("layer_00/gate.bin"));
    let up = load_ref(&prefill_ref("layer_00/up.bin"));
    let expected = load_ref(&prefill_ref("layer_00/silu_mul.bin"));
    let seq_len = gate.shape[0];

    let dev_gate = upload_ref(&backend, &gate, &[seq_len, config.intermediate_size]);
    let dev_up = upload_ref(&backend, &up, &[seq_len, config.intermediate_size]);
    let dev_output = backend
        .alloc(&[seq_len, config.intermediate_size], DType::FP16)
        .unwrap();

    backend.silu_mul(&dev_gate, &dev_up, &dev_output).unwrap();

    assert_kernel_close(&backend, &dev_output, &expected, RTOL, ATOL, "silu_mul_layer0");
    backend.free(&dev_gate).unwrap();
    backend.free(&dev_up).unwrap();
    backend.free(&dev_output).unwrap();
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

#[test]
fn test_kernel_rope_layer0() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };
    let _ = weights;

    // Load pre-RoPE Q and K (flat projections), reshape to multi-head format
    let q_flat = load_ref(&prefill_ref("layer_00/q.bin"));
    let k_flat = load_ref(&prefill_ref("layer_00/k.bin"));
    let seq_len = q_flat.shape[0];

    // Upload as multi-head shape: [seq_len, num_heads, head_dim]
    let dev_q = upload_ref(
        &backend,
        &q_flat,
        &[seq_len, config.num_q_heads, config.head_dim],
    );
    let dev_k = upload_ref(
        &backend,
        &k_flat,
        &[seq_len, config.num_kv_heads, config.head_dim],
    );

    let positions: Vec<u32> = (0..seq_len as u32).collect();
    backend
        .rope(&dev_q, &dev_k, &positions, config.rope_theta, config.head_dim)
        .unwrap();

    // Load expected post-RoPE tensors [seq_len, num_heads, head_dim]
    let expected_q = load_ref(&prefill_ref("layer_00/q_rope.bin"));
    let expected_k = load_ref(&prefill_ref("layer_00/k_rope.bin"));

    // RoPE uses trig functions on FP16 values — slightly higher tolerance needed
    let rope_rtol = 0.01;
    let rope_atol = 0.025;
    assert_kernel_close(&backend, &dev_q, &expected_q, rope_rtol, rope_atol, "rope_q_layer0");
    assert_kernel_close(&backend, &dev_k, &expected_k, rope_rtol, rope_atol, "rope_k_layer0");
    backend.free(&dev_q).unwrap();
    backend.free(&dev_k).unwrap();
}

// ---------------------------------------------------------------------------
// Add (residual connections)
// ---------------------------------------------------------------------------

#[test]
fn test_kernel_add_attn_residual_layer0() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };
    let _ = weights;

    let a = load_ref(&prefill_ref("layer_00/input_hidden.bin"));
    let b = load_ref(&prefill_ref("layer_00/attn_output.bin"));
    let expected = load_ref(&prefill_ref("layer_00/post_attn_residual.bin"));
    let seq_len = a.shape[0];
    let hidden = config.hidden_size;

    let dev_a = upload_ref(&backend, &a, &[seq_len, hidden]);
    let dev_b = upload_ref(&backend, &b, &[seq_len, hidden]);
    let dev_output = backend.alloc(&[seq_len, hidden], DType::FP16).unwrap();

    backend.add(&dev_a, &dev_b, &dev_output).unwrap();

    assert_kernel_close(&backend, &dev_output, &expected, RTOL, ATOL, "add_attn_residual_layer0");
    backend.free(&dev_a).unwrap();
    backend.free(&dev_b).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_add_ffn_residual_layer0() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };
    let _ = weights;

    let a = load_ref(&prefill_ref("layer_00/post_attn_residual.bin"));
    let b = load_ref(&prefill_ref("layer_00/ffn_output.bin"));
    let expected = load_ref(&prefill_ref("layer_00/output_hidden.bin"));
    let seq_len = a.shape[0];
    let hidden = config.hidden_size;

    let dev_a = upload_ref(&backend, &a, &[seq_len, hidden]);
    let dev_b = upload_ref(&backend, &b, &[seq_len, hidden]);
    let dev_output = backend.alloc(&[seq_len, hidden], DType::FP16).unwrap();

    backend.add(&dev_a, &dev_b, &dev_output).unwrap();

    assert_kernel_close(&backend, &dev_output, &expected, RTOL, ATOL, "add_ffn_residual_layer0");
    backend.free(&dev_a).unwrap();
    backend.free(&dev_b).unwrap();
    backend.free(&dev_output).unwrap();
}

// ---------------------------------------------------------------------------
// Decode-path kernels (seq_len=1)
// ---------------------------------------------------------------------------

#[test]
fn test_kernel_rmsnorm_attn_decode() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    let input = load_ref(&decode_ref("layer_00/input_hidden.bin"));
    let expected = load_ref(&decode_ref("layer_00/post_attn_norm.bin"));
    let hidden = config.hidden_size;

    let dev_input = upload_ref(&backend, &input, &[1, hidden]);
    let dev_output = backend.alloc(&[1, hidden], DType::FP16).unwrap();

    backend
        .rmsnorm(
            &dev_input,
            &weights.layers[0].attn_norm,
            config.rms_norm_eps,
            &dev_output,
        )
        .unwrap();

    assert_kernel_close(&backend, &dev_output, &expected, LOOSE_RTOL, LOOSE_ATOL, "rmsnorm_attn_decode");
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_matmul_q_proj_decode() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    let input = load_ref(&decode_ref("layer_00/post_attn_norm.bin"));
    let expected = load_ref(&decode_ref("layer_00/q.bin"));

    let dev_input = upload_ref(&backend, &input, &[1, config.hidden_size]);
    let dev_output = backend
        .alloc(&[1, config.hidden_size], DType::FP16)
        .unwrap();

    backend
        .matmul(&dev_input, &weights.layers[0].q_proj, &dev_output)
        .unwrap();

    assert_kernel_close(&backend, &dev_output, &expected, RTOL, ATOL, "matmul_q_proj_decode");
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_silu_mul_decode() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };
    let _ = weights;

    let gate = load_ref(&decode_ref("layer_00/gate.bin"));
    let up = load_ref(&decode_ref("layer_00/up.bin"));
    let expected = load_ref(&decode_ref("layer_00/silu_mul.bin"));

    let dev_gate = upload_ref(&backend, &gate, &[1, config.intermediate_size]);
    let dev_up = upload_ref(&backend, &up, &[1, config.intermediate_size]);
    let dev_output = backend
        .alloc(&[1, config.intermediate_size], DType::FP16)
        .unwrap();

    backend.silu_mul(&dev_gate, &dev_up, &dev_output).unwrap();

    assert_kernel_close(&backend, &dev_output, &expected, RTOL, ATOL, "silu_mul_decode");
    backend.free(&dev_gate).unwrap();
    backend.free(&dev_up).unwrap();
    backend.free(&dev_output).unwrap();
}

// ---------------------------------------------------------------------------
// Final RMSNorm (output norm, applied to full hidden state)
// ---------------------------------------------------------------------------

#[test]
fn test_kernel_rmsnorm_final() {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    // The input to final norm is layer_31/output_hidden (last layer output).
    let last = config.num_layers - 1;
    let input = load_ref(&prefill_ref(&format!("layer_{last:02}/output_hidden.bin")));
    let expected = load_ref(&prefill_ref("final_norm.bin"));
    let seq_len = input.shape[0];
    let hidden = config.hidden_size;

    let dev_input = upload_ref(&backend, &input, &[seq_len, hidden]);
    let dev_output = backend.alloc(&[seq_len, hidden], DType::FP16).unwrap();

    backend
        .rmsnorm(&dev_input, &weights.output_norm, config.rms_norm_eps, &dev_output)
        .unwrap();

    assert_kernel_close(&backend, &dev_output, &expected, LOOSE_RTOL, LOOSE_ATOL, "rmsnorm_final");
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}
