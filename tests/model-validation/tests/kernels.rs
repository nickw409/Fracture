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

fn run_real_rmsnorm_attn(layer: usize) {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    let dir = format!("layer_{layer:02}");
    let input = load_ref(&prefill_ref(&format!("{dir}/input_hidden.bin")));
    let expected = load_ref(&prefill_ref(&format!("{dir}/post_attn_norm.bin")));
    let seq_len = input.shape[0];
    let hidden = config.hidden_size;

    let dev_input = upload_ref(&backend, &input, &[seq_len, hidden]);
    let dev_output = backend.alloc(&[seq_len, hidden], DType::FP16).unwrap();

    backend
        .rmsnorm(
            &dev_input,
            &weights.layers[layer].attn_norm,
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
        &format!("rmsnorm_attn_layer{layer}"),
    );
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_rmsnorm_attn_layer0() {
    run_real_rmsnorm_attn(0);
}
#[test]
fn test_kernel_rmsnorm_attn_layer8() {
    run_real_rmsnorm_attn(8);
}
#[test]
fn test_kernel_rmsnorm_attn_layer16() {
    run_real_rmsnorm_attn(16);
}
#[test]
fn test_kernel_rmsnorm_attn_layer24() {
    run_real_rmsnorm_attn(24);
}
#[test]
fn test_kernel_rmsnorm_attn_layer31() {
    run_real_rmsnorm_attn(31);
}

// Preserved name from the original "last layer" test for CI compatibility.
#[test]
fn test_kernel_rmsnorm_attn_last_layer() {
    let last = {
        let Some((_, _, config)) = setup_backend_and_weights() else {
            skip!("model or reference data not available");
        };
        config.num_layers - 1
    };
    run_real_rmsnorm_attn(last);
}

fn run_real_rmsnorm_ffn(layer: usize) {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    let dir = format!("layer_{layer:02}");
    let input = load_ref(&prefill_ref(&format!("{dir}/post_attn_residual.bin")));
    let expected = load_ref(&prefill_ref(&format!("{dir}/post_ffn_norm.bin")));
    let seq_len = input.shape[0];
    let hidden = config.hidden_size;

    let dev_input = upload_ref(&backend, &input, &[seq_len, hidden]);
    let dev_output = backend.alloc(&[seq_len, hidden], DType::FP16).unwrap();

    backend
        .rmsnorm(
            &dev_input,
            &weights.layers[layer].ffn_norm,
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
        &format!("rmsnorm_ffn_layer{layer}"),
    );
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_rmsnorm_ffn_layer0() {
    run_real_rmsnorm_ffn(0);
}
#[test]
fn test_kernel_rmsnorm_ffn_layer8() {
    run_real_rmsnorm_ffn(8);
}
#[test]
fn test_kernel_rmsnorm_ffn_layer16() {
    run_real_rmsnorm_ffn(16);
}
#[test]
fn test_kernel_rmsnorm_ffn_layer24() {
    run_real_rmsnorm_ffn(24);
}
#[test]
fn test_kernel_rmsnorm_ffn_layer31() {
    run_real_rmsnorm_ffn(31);
}

// ---------------------------------------------------------------------------
// MatMul — QKV projections (layer 0)
// ---------------------------------------------------------------------------

fn run_real_matmul_q_proj(layer: usize) {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };
    let dir = format!("layer_{layer:02}");
    let input = load_ref(&prefill_ref(&format!("{dir}/post_attn_norm.bin")));
    let expected = load_ref(&prefill_ref(&format!("{dir}/q.bin")));
    let seq_len = input.shape[0];

    let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
    let dev_output = backend
        .alloc(&[seq_len, config.hidden_size], DType::FP16)
        .unwrap();

    backend
        .matmul(&dev_input, &weights.layers[layer].q_proj, &dev_output)
        .unwrap();

    assert_kernel_close(
        &backend,
        &dev_output,
        &expected,
        RTOL,
        ATOL,
        &format!("matmul_q_proj_layer{layer}"),
    );
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_matmul_q_proj_layer0() {
    run_real_matmul_q_proj(0);
}
#[test]
fn test_kernel_matmul_q_proj_layer8() {
    run_real_matmul_q_proj(8);
}
#[test]
fn test_kernel_matmul_q_proj_layer16() {
    run_real_matmul_q_proj(16);
}
#[test]
fn test_kernel_matmul_q_proj_layer24() {
    run_real_matmul_q_proj(24);
}
#[test]
fn test_kernel_matmul_q_proj_layer31() {
    run_real_matmul_q_proj(31);
}

fn run_real_matmul_k_proj(layer: usize) {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };
    let dir = format!("layer_{layer:02}");
    let input = load_ref(&prefill_ref(&format!("{dir}/post_attn_norm.bin")));
    let expected = load_ref(&prefill_ref(&format!("{dir}/k.bin")));
    let seq_len = input.shape[0];
    let kv_dim = config.num_kv_heads * config.head_dim;

    let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
    let dev_output = backend.alloc(&[seq_len, kv_dim], DType::FP16).unwrap();

    backend
        .matmul(&dev_input, &weights.layers[layer].k_proj, &dev_output)
        .unwrap();

    assert_kernel_close(
        &backend,
        &dev_output,
        &expected,
        RTOL,
        ATOL,
        &format!("matmul_k_proj_layer{layer}"),
    );
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_matmul_k_proj_layer0() {
    run_real_matmul_k_proj(0);
}
#[test]
fn test_kernel_matmul_k_proj_layer8() {
    run_real_matmul_k_proj(8);
}
#[test]
fn test_kernel_matmul_k_proj_layer16() {
    run_real_matmul_k_proj(16);
}
#[test]
fn test_kernel_matmul_k_proj_layer24() {
    run_real_matmul_k_proj(24);
}
#[test]
fn test_kernel_matmul_k_proj_layer31() {
    run_real_matmul_k_proj(31);
}

fn run_real_matmul_v_proj(layer: usize) {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };
    let dir = format!("layer_{layer:02}");
    let input = load_ref(&prefill_ref(&format!("{dir}/post_attn_norm.bin")));
    let expected = load_ref(&prefill_ref(&format!("{dir}/v.bin")));
    let seq_len = input.shape[0];
    let kv_dim = config.num_kv_heads * config.head_dim;

    let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
    let dev_output = backend.alloc(&[seq_len, kv_dim], DType::FP16).unwrap();

    backend
        .matmul(&dev_input, &weights.layers[layer].v_proj, &dev_output)
        .unwrap();

    assert_kernel_close(
        &backend,
        &dev_output,
        &expected,
        RTOL,
        ATOL,
        &format!("matmul_v_proj_layer{layer}"),
    );
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_matmul_v_proj_layer0() {
    run_real_matmul_v_proj(0);
}
#[test]
fn test_kernel_matmul_v_proj_layer8() {
    run_real_matmul_v_proj(8);
}
#[test]
fn test_kernel_matmul_v_proj_layer16() {
    run_real_matmul_v_proj(16);
}
#[test]
fn test_kernel_matmul_v_proj_layer24() {
    run_real_matmul_v_proj(24);
}
#[test]
fn test_kernel_matmul_v_proj_layer31() {
    run_real_matmul_v_proj(31);
}

// ---------------------------------------------------------------------------
// MatMul — FFN projections (layer 0)
// ---------------------------------------------------------------------------

fn run_real_matmul_gate_proj(layer: usize) {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };
    let dir = format!("layer_{layer:02}");
    let input = load_ref(&prefill_ref(&format!("{dir}/post_ffn_norm.bin")));
    let expected = load_ref(&prefill_ref(&format!("{dir}/gate.bin")));
    let seq_len = input.shape[0];

    let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
    let dev_output = backend
        .alloc(&[seq_len, config.intermediate_size], DType::FP16)
        .unwrap();

    backend
        .matmul(&dev_input, &weights.layers[layer].gate_proj, &dev_output)
        .unwrap();

    assert_kernel_close(
        &backend,
        &dev_output,
        &expected,
        RTOL,
        ATOL,
        &format!("matmul_gate_proj_layer{layer}"),
    );
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_matmul_gate_proj_layer0() {
    run_real_matmul_gate_proj(0);
}
#[test]
fn test_kernel_matmul_gate_proj_layer8() {
    run_real_matmul_gate_proj(8);
}
#[test]
fn test_kernel_matmul_gate_proj_layer16() {
    run_real_matmul_gate_proj(16);
}
#[test]
fn test_kernel_matmul_gate_proj_layer24() {
    run_real_matmul_gate_proj(24);
}
#[test]
fn test_kernel_matmul_gate_proj_layer31() {
    run_real_matmul_gate_proj(31);
}

fn run_real_matmul_up_proj(layer: usize) {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };
    let dir = format!("layer_{layer:02}");
    let input = load_ref(&prefill_ref(&format!("{dir}/post_ffn_norm.bin")));
    let expected = load_ref(&prefill_ref(&format!("{dir}/up.bin")));
    let seq_len = input.shape[0];

    let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
    let dev_output = backend
        .alloc(&[seq_len, config.intermediate_size], DType::FP16)
        .unwrap();

    backend
        .matmul(&dev_input, &weights.layers[layer].up_proj, &dev_output)
        .unwrap();

    assert_kernel_close(
        &backend,
        &dev_output,
        &expected,
        RTOL,
        ATOL,
        &format!("matmul_up_proj_layer{layer}"),
    );
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_matmul_up_proj_layer0() {
    run_real_matmul_up_proj(0);
}
#[test]
fn test_kernel_matmul_up_proj_layer8() {
    run_real_matmul_up_proj(8);
}
#[test]
fn test_kernel_matmul_up_proj_layer16() {
    run_real_matmul_up_proj(16);
}
#[test]
fn test_kernel_matmul_up_proj_layer24() {
    run_real_matmul_up_proj(24);
}
#[test]
fn test_kernel_matmul_up_proj_layer31() {
    run_real_matmul_up_proj(31);
}

fn run_real_matmul_down_proj(layer: usize) {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };
    let dir = format!("layer_{layer:02}");
    let input = load_ref(&prefill_ref(&format!("{dir}/silu_mul.bin")));
    let expected = load_ref(&prefill_ref(&format!("{dir}/ffn_output.bin")));
    let seq_len = input.shape[0];

    let dev_input = upload_ref(&backend, &input, &[seq_len, config.intermediate_size]);
    let dev_output = backend
        .alloc(&[seq_len, config.hidden_size], DType::FP16)
        .unwrap();

    backend
        .matmul(&dev_input, &weights.layers[layer].down_proj, &dev_output)
        .unwrap();

    assert_kernel_close(
        &backend,
        &dev_output,
        &expected,
        RTOL,
        ATOL,
        &format!("matmul_down_proj_layer{layer}"),
    );
    backend.free(&dev_input).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_matmul_down_proj_layer0() {
    run_real_matmul_down_proj(0);
}
#[test]
fn test_kernel_matmul_down_proj_layer8() {
    run_real_matmul_down_proj(8);
}
#[test]
fn test_kernel_matmul_down_proj_layer16() {
    run_real_matmul_down_proj(16);
}
#[test]
fn test_kernel_matmul_down_proj_layer24() {
    run_real_matmul_down_proj(24);
}
#[test]
fn test_kernel_matmul_down_proj_layer31() {
    run_real_matmul_down_proj(31);
}

// ---------------------------------------------------------------------------
// SiLU × Mul
// ---------------------------------------------------------------------------

fn run_real_silu_mul(layer: usize) {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };
    let _ = weights; // only needed for model loading side-effect

    let dir = format!("layer_{layer:02}");
    let gate = load_ref(&prefill_ref(&format!("{dir}/gate.bin")));
    let up = load_ref(&prefill_ref(&format!("{dir}/up.bin")));
    let expected = load_ref(&prefill_ref(&format!("{dir}/silu_mul.bin")));
    let seq_len = gate.shape[0];

    let dev_gate = upload_ref(&backend, &gate, &[seq_len, config.intermediate_size]);
    let dev_up = upload_ref(&backend, &up, &[seq_len, config.intermediate_size]);
    let dev_output = backend
        .alloc(&[seq_len, config.intermediate_size], DType::FP16)
        .unwrap();

    backend.silu_mul(&dev_gate, &dev_up, &dev_output).unwrap();

    assert_kernel_close(
        &backend,
        &dev_output,
        &expected,
        RTOL,
        ATOL,
        &format!("silu_mul_layer{layer}"),
    );
    backend.free(&dev_gate).unwrap();
    backend.free(&dev_up).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_silu_mul_layer0() {
    run_real_silu_mul(0);
}
#[test]
fn test_kernel_silu_mul_layer8() {
    run_real_silu_mul(8);
}
#[test]
fn test_kernel_silu_mul_layer16() {
    run_real_silu_mul(16);
}
#[test]
fn test_kernel_silu_mul_layer24() {
    run_real_silu_mul(24);
}
#[test]
fn test_kernel_silu_mul_layer31() {
    run_real_silu_mul(31);
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

fn run_real_rope(layer: usize) {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };
    let _ = weights;

    let dir = format!("layer_{layer:02}");
    // Load pre-RoPE Q and K (flat projections), reshape to multi-head format
    let q_flat = load_ref(&prefill_ref(&format!("{dir}/q.bin")));
    let k_flat = load_ref(&prefill_ref(&format!("{dir}/k.bin")));
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
    let expected_q = load_ref(&prefill_ref(&format!("{dir}/q_rope.bin")));
    let expected_k = load_ref(&prefill_ref(&format!("{dir}/k_rope.bin")));

    // RoPE uses trig functions on FP16 values — slightly higher tolerance needed
    let rope_rtol = 0.01;
    let rope_atol = 0.025;
    assert_kernel_close(
        &backend,
        &dev_q,
        &expected_q,
        rope_rtol,
        rope_atol,
        &format!("rope_q_layer{layer}"),
    );
    assert_kernel_close(
        &backend,
        &dev_k,
        &expected_k,
        rope_rtol,
        rope_atol,
        &format!("rope_k_layer{layer}"),
    );
    backend.free(&dev_q).unwrap();
    backend.free(&dev_k).unwrap();
}

#[test]
fn test_kernel_rope_layer0() {
    run_real_rope(0);
}
#[test]
fn test_kernel_rope_layer8() {
    run_real_rope(8);
}
#[test]
fn test_kernel_rope_layer16() {
    run_real_rope(16);
}
#[test]
fn test_kernel_rope_layer24() {
    run_real_rope(24);
}
#[test]
fn test_kernel_rope_layer31() {
    run_real_rope(31);
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
// Attention kernel (prefill, layer 0)
// ---------------------------------------------------------------------------

fn run_real_attention(layer: usize) {
    let Some((backend, weights, config)) = setup_backend_and_weights() else {
        skip!("model or reference data not available");
    };

    let dir = format!("layer_{layer:02}");
    let q_rope_ref = load_ref(&prefill_ref(&format!("{dir}/q_rope.bin")));
    let k_rope_ref = load_ref(&prefill_ref(&format!("{dir}/k_rope.bin")));
    let v_ref = load_ref(&prefill_ref(&format!("{dir}/v.bin")));
    let expected = load_ref(&prefill_ref(&format!("{dir}/attn_output.bin")));

    // q_rope: [seq_len, num_q_heads, head_dim]
    // k_rope: [seq_len, num_kv_heads, head_dim]
    // v:      [seq_len, num_kv_heads, head_dim]
    let seq_len = q_rope_ref.shape[0];
    let num_q_heads = config.num_q_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let hidden = config.hidden_size;

    // Upload Q (post-RoPE) as [seq_len, num_q_heads, head_dim]
    let dev_q = upload_ref(&backend, &q_rope_ref, &[seq_len, num_q_heads, head_dim]);

    // Allocate KV cache tensors sized for the full prefill sequence.
    // copy_to_device loads data starting at offset 0, so the cache is fully
    // populated and start_pos=0 during the attention call.
    let dev_k_cache = backend
        .alloc(&[seq_len, num_kv_heads, head_dim], DType::FP16)
        .unwrap();
    let dev_v_cache = backend
        .alloc(&[seq_len, num_kv_heads, head_dim], DType::FP16)
        .unwrap();

    let k_bytes = ref_to_fp16_bytes(&k_rope_ref);
    let v_bytes = ref_to_fp16_bytes(&v_ref);
    backend.copy_to_device(&dev_k_cache, &k_bytes).unwrap();
    backend.copy_to_device(&dev_v_cache, &v_bytes).unwrap();

    // Allocate raw attention output [seq_len, num_q_heads, head_dim]
    let dev_attn_raw = backend
        .alloc(&[seq_len, num_q_heads, head_dim], DType::FP16)
        .unwrap();

    backend
        .attention(
            &dev_q,
            &dev_k_cache,
            &dev_v_cache,
            num_kv_heads,
            0, // start_pos = 0 (pure prefill, no prior context)
            &dev_attn_raw,
        )
        .unwrap();

    // Reshape [seq_len, num_q_heads, head_dim] -> [seq_len, hidden] for o_proj
    let dev_attn_flat = dev_attn_raw.reshape(vec![seq_len, hidden]).unwrap();

    // Apply output projection: [seq_len, hidden] x o_proj -> [seq_len, hidden]
    let dev_output = backend.alloc(&[seq_len, hidden], DType::FP16).unwrap();
    backend
        .matmul(&dev_attn_flat, &weights.layers[layer].o_proj, &dev_output)
        .unwrap();

    // attn_output.bin is the full self-attention block output (after o_proj),
    // which is what we compare against.
    assert_kernel_close(
        &backend,
        &dev_output,
        &expected,
        LOOSE_RTOL,
        LOOSE_ATOL,
        &format!("attention_layer{layer}"),
    );

    backend.free(&dev_q).unwrap();
    backend.free(&dev_k_cache).unwrap();
    backend.free(&dev_v_cache).unwrap();
    backend.free(&dev_attn_raw).unwrap();
    backend.free(&dev_output).unwrap();
}

#[test]
fn test_kernel_attention_layer0() {
    run_real_attention(0);
}
#[test]
fn test_kernel_attention_layer8() {
    run_real_attention(8);
}
#[test]
fn test_kernel_attention_layer16() {
    run_real_attention(16);
}
#[test]
fn test_kernel_attention_layer24() {
    run_real_attention(24);
}
#[test]
fn test_kernel_attention_layer31() {
    run_real_attention(31);
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

// ---------------------------------------------------------------------------
// Fixture-driven per-kernel tests (always-on, 4-layer tiny model)
// ---------------------------------------------------------------------------
//
// These tests use the committed `tests/fixtures/tiny-llama.gguf` and
// `tests/reference-fixture/` dumps. They do NOT skip when the real Llama
// model is unavailable — they only skip when CUDA itself is unavailable.

mod fixture {
    use super::*;
    use fracture_model_validation::{fixture_model_path, fixture_reference_dir};

    /// Reference path for the fixture, prompt 0 prefill.
    fn fixture_prefill_ref(relative: &str) -> String {
        fixture_reference_dir()
            .join("prompt_0")
            .join(relative)
            .to_str()
            .unwrap()
            .to_string()
    }

    /// Set up backend + fixture weights + config. Panics if fixture missing.
    /// Returns None only if CUDA is unavailable.
    fn setup_fixture_backend() -> Option<(CudaBackend, WeightStore, ModelConfig)> {
        let path = fixture_model_path();
        assert!(
            path.exists(),
            "fixture missing at {}; run scripts/build_fixture_model.py",
            path.display()
        );
        let mut backend = CudaBackend::new(0).ok()?;
        let weights = WeightStore::load(&path, &backend, None).expect("fixture load failed");
        let config = weights.config.clone();
        backend
            .precompute_rope_freqs(config.head_dim, config.rope_theta)
            .ok()?;
        Some((backend, weights, config))
    }

    macro_rules! cuda_or_skip {
        ($e:expr) => {
            match $e {
                Some(v) => v,
                None => {
                    eprintln!("skip: CUDA unavailable");
                    return;
                }
            }
        };
    }

    // -----------------------------------------------------------------------
    // Embedding (single test, not per-layer)
    // -----------------------------------------------------------------------

    #[test]
    fn fixture_kernel_embedding() {
        let (backend, weights, config) = cuda_or_skip!(setup_fixture_backend());
        let expected = load_ref(&fixture_prefill_ref("embeddings.bin"));
        let seq_len = expected.shape[0];
        let token_ids_ref = load_ref(&fixture_prefill_ref("token_ids.bin"));
        let token_ids: Vec<u32> = token_ids_ref
            .to_f32()
            .iter()
            .map(|&v| v as u32)
            .collect();
        let output = backend
            .alloc(&[seq_len, config.hidden_size], DType::FP16)
            .unwrap();
        backend
            .embedding(&token_ids, &weights.token_embedding, &output)
            .unwrap();
        assert_kernel_close(&backend, &output, &expected, RTOL, ATOL, "fixture_embedding");
        backend.free(&output).unwrap();
    }

    // -----------------------------------------------------------------------
    // RMSNorm (attn) — per layer
    // -----------------------------------------------------------------------

    fn run_fixture_rmsnorm_attn(layer: usize) {
        let (backend, weights, config) = cuda_or_skip!(setup_fixture_backend());
        let input = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/input_hidden.bin"
        )));
        let expected = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/post_attn_norm.bin"
        )));
        let seq_len = input.shape[0];
        let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
        let dev_output = backend
            .alloc(&[seq_len, config.hidden_size], DType::FP16)
            .unwrap();
        backend
            .rmsnorm(
                &dev_input,
                &weights.layers[layer].attn_norm,
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
            &format!("fixture_rmsnorm_attn_layer{layer}"),
        );
        backend.free(&dev_input).unwrap();
        backend.free(&dev_output).unwrap();
    }

    #[test]
    fn fixture_rmsnorm_attn_layer0() {
        run_fixture_rmsnorm_attn(0);
    }
    #[test]
    fn fixture_rmsnorm_attn_layer1() {
        run_fixture_rmsnorm_attn(1);
    }
    #[test]
    fn fixture_rmsnorm_attn_layer2() {
        run_fixture_rmsnorm_attn(2);
    }
    #[test]
    fn fixture_rmsnorm_attn_layer3() {
        run_fixture_rmsnorm_attn(3);
    }

    // -----------------------------------------------------------------------
    // RMSNorm (ffn) — per layer
    // -----------------------------------------------------------------------

    fn run_fixture_rmsnorm_ffn(layer: usize) {
        let (backend, weights, config) = cuda_or_skip!(setup_fixture_backend());
        let input = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/post_attn_residual.bin"
        )));
        let expected = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/post_ffn_norm.bin"
        )));
        let seq_len = input.shape[0];
        let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
        let dev_output = backend
            .alloc(&[seq_len, config.hidden_size], DType::FP16)
            .unwrap();
        backend
            .rmsnorm(
                &dev_input,
                &weights.layers[layer].ffn_norm,
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
            &format!("fixture_rmsnorm_ffn_layer{layer}"),
        );
        backend.free(&dev_input).unwrap();
        backend.free(&dev_output).unwrap();
    }

    #[test]
    fn fixture_rmsnorm_ffn_layer0() {
        run_fixture_rmsnorm_ffn(0);
    }
    #[test]
    fn fixture_rmsnorm_ffn_layer1() {
        run_fixture_rmsnorm_ffn(1);
    }
    #[test]
    fn fixture_rmsnorm_ffn_layer2() {
        run_fixture_rmsnorm_ffn(2);
    }
    #[test]
    fn fixture_rmsnorm_ffn_layer3() {
        run_fixture_rmsnorm_ffn(3);
    }

    // -----------------------------------------------------------------------
    // MatMul Q proj — per layer
    // -----------------------------------------------------------------------

    fn run_fixture_matmul_q_proj(layer: usize) {
        let (backend, weights, config) = cuda_or_skip!(setup_fixture_backend());
        let input = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/post_attn_norm.bin"
        )));
        let expected = load_ref(&fixture_prefill_ref(&format!("layer_{layer:02}/q.bin")));
        let seq_len = input.shape[0];
        let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
        let dev_output = backend
            .alloc(&[seq_len, config.hidden_size], DType::FP16)
            .unwrap();
        backend
            .matmul(&dev_input, &weights.layers[layer].q_proj, &dev_output)
            .unwrap();
        assert_kernel_close(
            &backend,
            &dev_output,
            &expected,
            RTOL,
            ATOL,
            &format!("fixture_matmul_q_proj_layer{layer}"),
        );
        backend.free(&dev_input).unwrap();
        backend.free(&dev_output).unwrap();
    }

    #[test]
    fn fixture_matmul_q_proj_layer0() {
        run_fixture_matmul_q_proj(0);
    }
    #[test]
    fn fixture_matmul_q_proj_layer1() {
        run_fixture_matmul_q_proj(1);
    }
    #[test]
    fn fixture_matmul_q_proj_layer2() {
        run_fixture_matmul_q_proj(2);
    }
    #[test]
    fn fixture_matmul_q_proj_layer3() {
        run_fixture_matmul_q_proj(3);
    }

    // -----------------------------------------------------------------------
    // MatMul K proj — per layer
    // -----------------------------------------------------------------------

    fn run_fixture_matmul_k_proj(layer: usize) {
        let (backend, weights, config) = cuda_or_skip!(setup_fixture_backend());
        let input = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/post_attn_norm.bin"
        )));
        let expected = load_ref(&fixture_prefill_ref(&format!("layer_{layer:02}/k.bin")));
        let seq_len = input.shape[0];
        let kv_dim = config.num_kv_heads * config.head_dim;
        let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
        let dev_output = backend.alloc(&[seq_len, kv_dim], DType::FP16).unwrap();
        backend
            .matmul(&dev_input, &weights.layers[layer].k_proj, &dev_output)
            .unwrap();
        assert_kernel_close(
            &backend,
            &dev_output,
            &expected,
            RTOL,
            ATOL,
            &format!("fixture_matmul_k_proj_layer{layer}"),
        );
        backend.free(&dev_input).unwrap();
        backend.free(&dev_output).unwrap();
    }

    #[test]
    fn fixture_matmul_k_proj_layer0() {
        run_fixture_matmul_k_proj(0);
    }
    #[test]
    fn fixture_matmul_k_proj_layer1() {
        run_fixture_matmul_k_proj(1);
    }
    #[test]
    fn fixture_matmul_k_proj_layer2() {
        run_fixture_matmul_k_proj(2);
    }
    #[test]
    fn fixture_matmul_k_proj_layer3() {
        run_fixture_matmul_k_proj(3);
    }

    // -----------------------------------------------------------------------
    // MatMul V proj — per layer
    // -----------------------------------------------------------------------

    fn run_fixture_matmul_v_proj(layer: usize) {
        let (backend, weights, config) = cuda_or_skip!(setup_fixture_backend());
        let input = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/post_attn_norm.bin"
        )));
        let expected = load_ref(&fixture_prefill_ref(&format!("layer_{layer:02}/v.bin")));
        let seq_len = input.shape[0];
        let kv_dim = config.num_kv_heads * config.head_dim;
        let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
        let dev_output = backend.alloc(&[seq_len, kv_dim], DType::FP16).unwrap();
        backend
            .matmul(&dev_input, &weights.layers[layer].v_proj, &dev_output)
            .unwrap();
        assert_kernel_close(
            &backend,
            &dev_output,
            &expected,
            RTOL,
            ATOL,
            &format!("fixture_matmul_v_proj_layer{layer}"),
        );
        backend.free(&dev_input).unwrap();
        backend.free(&dev_output).unwrap();
    }

    #[test]
    fn fixture_matmul_v_proj_layer0() {
        run_fixture_matmul_v_proj(0);
    }
    #[test]
    fn fixture_matmul_v_proj_layer1() {
        run_fixture_matmul_v_proj(1);
    }
    #[test]
    fn fixture_matmul_v_proj_layer2() {
        run_fixture_matmul_v_proj(2);
    }
    #[test]
    fn fixture_matmul_v_proj_layer3() {
        run_fixture_matmul_v_proj(3);
    }

    // -----------------------------------------------------------------------
    // MatMul gate proj — per layer
    // -----------------------------------------------------------------------

    fn run_fixture_matmul_gate_proj(layer: usize) {
        let (backend, weights, config) = cuda_or_skip!(setup_fixture_backend());
        let input = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/post_ffn_norm.bin"
        )));
        let expected = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/gate.bin"
        )));
        let seq_len = input.shape[0];
        let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
        let dev_output = backend
            .alloc(&[seq_len, config.intermediate_size], DType::FP16)
            .unwrap();
        backend
            .matmul(&dev_input, &weights.layers[layer].gate_proj, &dev_output)
            .unwrap();
        assert_kernel_close(
            &backend,
            &dev_output,
            &expected,
            RTOL,
            ATOL,
            &format!("fixture_matmul_gate_proj_layer{layer}"),
        );
        backend.free(&dev_input).unwrap();
        backend.free(&dev_output).unwrap();
    }

    #[test]
    fn fixture_matmul_gate_proj_layer0() {
        run_fixture_matmul_gate_proj(0);
    }
    #[test]
    fn fixture_matmul_gate_proj_layer1() {
        run_fixture_matmul_gate_proj(1);
    }
    #[test]
    fn fixture_matmul_gate_proj_layer2() {
        run_fixture_matmul_gate_proj(2);
    }
    #[test]
    fn fixture_matmul_gate_proj_layer3() {
        run_fixture_matmul_gate_proj(3);
    }

    // -----------------------------------------------------------------------
    // MatMul up proj — per layer
    // -----------------------------------------------------------------------

    fn run_fixture_matmul_up_proj(layer: usize) {
        let (backend, weights, config) = cuda_or_skip!(setup_fixture_backend());
        let input = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/post_ffn_norm.bin"
        )));
        let expected = load_ref(&fixture_prefill_ref(&format!("layer_{layer:02}/up.bin")));
        let seq_len = input.shape[0];
        let dev_input = upload_ref(&backend, &input, &[seq_len, config.hidden_size]);
        let dev_output = backend
            .alloc(&[seq_len, config.intermediate_size], DType::FP16)
            .unwrap();
        backend
            .matmul(&dev_input, &weights.layers[layer].up_proj, &dev_output)
            .unwrap();
        assert_kernel_close(
            &backend,
            &dev_output,
            &expected,
            RTOL,
            ATOL,
            &format!("fixture_matmul_up_proj_layer{layer}"),
        );
        backend.free(&dev_input).unwrap();
        backend.free(&dev_output).unwrap();
    }

    #[test]
    fn fixture_matmul_up_proj_layer0() {
        run_fixture_matmul_up_proj(0);
    }
    #[test]
    fn fixture_matmul_up_proj_layer1() {
        run_fixture_matmul_up_proj(1);
    }
    #[test]
    fn fixture_matmul_up_proj_layer2() {
        run_fixture_matmul_up_proj(2);
    }
    #[test]
    fn fixture_matmul_up_proj_layer3() {
        run_fixture_matmul_up_proj(3);
    }

    // -----------------------------------------------------------------------
    // MatMul down proj — per layer
    // -----------------------------------------------------------------------

    fn run_fixture_matmul_down_proj(layer: usize) {
        let (backend, weights, config) = cuda_or_skip!(setup_fixture_backend());
        let input = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/silu_mul.bin"
        )));
        let expected = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/ffn_output.bin"
        )));
        let seq_len = input.shape[0];
        let dev_input = upload_ref(&backend, &input, &[seq_len, config.intermediate_size]);
        let dev_output = backend
            .alloc(&[seq_len, config.hidden_size], DType::FP16)
            .unwrap();
        backend
            .matmul(&dev_input, &weights.layers[layer].down_proj, &dev_output)
            .unwrap();
        assert_kernel_close(
            &backend,
            &dev_output,
            &expected,
            RTOL,
            ATOL,
            &format!("fixture_matmul_down_proj_layer{layer}"),
        );
        backend.free(&dev_input).unwrap();
        backend.free(&dev_output).unwrap();
    }

    #[test]
    fn fixture_matmul_down_proj_layer0() {
        run_fixture_matmul_down_proj(0);
    }
    #[test]
    fn fixture_matmul_down_proj_layer1() {
        run_fixture_matmul_down_proj(1);
    }
    #[test]
    fn fixture_matmul_down_proj_layer2() {
        run_fixture_matmul_down_proj(2);
    }
    #[test]
    fn fixture_matmul_down_proj_layer3() {
        run_fixture_matmul_down_proj(3);
    }

    // -----------------------------------------------------------------------
    // SiLU × Mul — per layer
    // -----------------------------------------------------------------------

    fn run_fixture_silu_mul(layer: usize) {
        let (backend, weights, config) = cuda_or_skip!(setup_fixture_backend());
        let _ = weights;
        let gate = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/gate.bin"
        )));
        let up = load_ref(&fixture_prefill_ref(&format!("layer_{layer:02}/up.bin")));
        let expected = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/silu_mul.bin"
        )));
        let seq_len = gate.shape[0];
        let dev_gate = upload_ref(&backend, &gate, &[seq_len, config.intermediate_size]);
        let dev_up = upload_ref(&backend, &up, &[seq_len, config.intermediate_size]);
        let dev_output = backend
            .alloc(&[seq_len, config.intermediate_size], DType::FP16)
            .unwrap();
        backend.silu_mul(&dev_gate, &dev_up, &dev_output).unwrap();
        assert_kernel_close(
            &backend,
            &dev_output,
            &expected,
            RTOL,
            ATOL,
            &format!("fixture_silu_mul_layer{layer}"),
        );
        backend.free(&dev_gate).unwrap();
        backend.free(&dev_up).unwrap();
        backend.free(&dev_output).unwrap();
    }

    #[test]
    fn fixture_silu_mul_layer0() {
        run_fixture_silu_mul(0);
    }
    #[test]
    fn fixture_silu_mul_layer1() {
        run_fixture_silu_mul(1);
    }
    #[test]
    fn fixture_silu_mul_layer2() {
        run_fixture_silu_mul(2);
    }
    #[test]
    fn fixture_silu_mul_layer3() {
        run_fixture_silu_mul(3);
    }

    // -----------------------------------------------------------------------
    // RoPE — per layer
    // -----------------------------------------------------------------------

    fn run_fixture_rope(layer: usize) {
        let (backend, weights, config) = cuda_or_skip!(setup_fixture_backend());
        let _ = weights;
        let q_flat = load_ref(&fixture_prefill_ref(&format!("layer_{layer:02}/q.bin")));
        let k_flat = load_ref(&fixture_prefill_ref(&format!("layer_{layer:02}/k.bin")));
        let seq_len = q_flat.shape[0];
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
        let expected_q = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/q_rope.bin"
        )));
        let expected_k = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/k_rope.bin"
        )));
        // RoPE uses trig functions on FP16 — slightly higher tolerance,
        // matching the real-model rope test.
        let rope_rtol = 0.01;
        let rope_atol = 0.025;
        assert_kernel_close(
            &backend,
            &dev_q,
            &expected_q,
            rope_rtol,
            rope_atol,
            &format!("fixture_rope_q_layer{layer}"),
        );
        assert_kernel_close(
            &backend,
            &dev_k,
            &expected_k,
            rope_rtol,
            rope_atol,
            &format!("fixture_rope_k_layer{layer}"),
        );
        backend.free(&dev_q).unwrap();
        backend.free(&dev_k).unwrap();
    }

    #[test]
    fn fixture_rope_layer0() {
        run_fixture_rope(0);
    }
    #[test]
    fn fixture_rope_layer1() {
        run_fixture_rope(1);
    }
    #[test]
    fn fixture_rope_layer2() {
        run_fixture_rope(2);
    }
    #[test]
    fn fixture_rope_layer3() {
        run_fixture_rope(3);
    }

    // -----------------------------------------------------------------------
    // Attention (includes o_proj) — per layer
    // -----------------------------------------------------------------------

    fn run_fixture_attention(layer: usize) {
        let (backend, weights, config) = cuda_or_skip!(setup_fixture_backend());
        let q_rope_ref = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/q_rope.bin"
        )));
        let k_rope_ref = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/k_rope.bin"
        )));
        let v_ref = load_ref(&fixture_prefill_ref(&format!("layer_{layer:02}/v.bin")));
        let expected = load_ref(&fixture_prefill_ref(&format!(
            "layer_{layer:02}/attn_output.bin"
        )));
        let seq_len = q_rope_ref.shape[0];
        let num_q_heads = config.num_q_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let hidden = config.hidden_size;

        let dev_q = upload_ref(&backend, &q_rope_ref, &[seq_len, num_q_heads, head_dim]);
        let dev_k_cache = backend
            .alloc(&[seq_len, num_kv_heads, head_dim], DType::FP16)
            .unwrap();
        let dev_v_cache = backend
            .alloc(&[seq_len, num_kv_heads, head_dim], DType::FP16)
            .unwrap();
        let k_bytes = ref_to_fp16_bytes(&k_rope_ref);
        let v_bytes = ref_to_fp16_bytes(&v_ref);
        backend.copy_to_device(&dev_k_cache, &k_bytes).unwrap();
        backend.copy_to_device(&dev_v_cache, &v_bytes).unwrap();

        let dev_attn_raw = backend
            .alloc(&[seq_len, num_q_heads, head_dim], DType::FP16)
            .unwrap();
        backend
            .attention(
                &dev_q,
                &dev_k_cache,
                &dev_v_cache,
                num_kv_heads,
                0,
                &dev_attn_raw,
            )
            .unwrap();

        let dev_attn_flat = dev_attn_raw.reshape(vec![seq_len, hidden]).unwrap();
        let dev_output = backend.alloc(&[seq_len, hidden], DType::FP16).unwrap();
        backend
            .matmul(&dev_attn_flat, &weights.layers[layer].o_proj, &dev_output)
            .unwrap();

        assert_kernel_close(
            &backend,
            &dev_output,
            &expected,
            LOOSE_RTOL,
            LOOSE_ATOL,
            &format!("fixture_attention_layer{layer}"),
        );

        backend.free(&dev_q).unwrap();
        backend.free(&dev_k_cache).unwrap();
        backend.free(&dev_v_cache).unwrap();
        backend.free(&dev_attn_raw).unwrap();
        backend.free(&dev_output).unwrap();
    }

    #[test]
    fn fixture_attention_layer0() {
        run_fixture_attention(0);
    }
    #[test]
    fn fixture_attention_layer1() {
        run_fixture_attention(1);
    }
    #[test]
    fn fixture_attention_layer2() {
        run_fixture_attention(2);
    }
    #[test]
    fn fixture_attention_layer3() {
        run_fixture_attention(3);
    }

    // -----------------------------------------------------------------------
    // Final RMSNorm (output norm)
    // -----------------------------------------------------------------------

    #[test]
    fn fixture_rmsnorm_final() {
        let (backend, weights, config) = cuda_or_skip!(setup_fixture_backend());
        let last = config.num_layers - 1;
        let input = load_ref(&fixture_prefill_ref(&format!(
            "layer_{last:02}/output_hidden.bin"
        )));
        let expected = load_ref(&fixture_prefill_ref("final_norm.bin"));
        let seq_len = input.shape[0];
        let hidden = config.hidden_size;
        let dev_input = upload_ref(&backend, &input, &[seq_len, hidden]);
        let dev_output = backend.alloc(&[seq_len, hidden], DType::FP16).unwrap();
        backend
            .rmsnorm(
                &dev_input,
                &weights.output_norm,
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
            "fixture_rmsnorm_final",
        );
        backend.free(&dev_input).unwrap();
        backend.free(&dev_output).unwrap();
    }
}
