//! Fracture validation utilities.
//!
//! Standalone reference tensor loading and comparison for validating
//! Fracture inference outputs against PyTorch ground truth.
//!
//! This crate has **zero** dependencies on any Fracture crate.

pub mod tensor_compare;
pub mod golden_compare;

pub use tensor_compare::*;
pub use golden_compare::*;

/// Standard tolerance for FP16 compute operations.
/// Most FP16 matmuls should match within these bounds.
pub const FP16_RTOL: f32 = 1e-3;
pub const FP16_ATOL: f32 = 1e-3;

/// Looser tolerance for operations with known numerical sensitivity
/// (e.g., softmax, layer norm on large sequences).
pub const FP16_LOOSE_RTOL: f32 = 5e-3;
pub const FP16_LOOSE_ATOL: f32 = 5e-3;

/// Load a reference tensor, panicking with a clear message if the file is missing.
pub fn require_reference(path: &str) -> ReferenceTensor {
    load_reference_tensor(path).unwrap_or_else(|e| {
        panic!(
            "Failed to load reference tensor from '{}':\n  {}\n\n\
             Hint: Run `python scripts/dump_reference.py` to generate reference data.",
            path, e
        )
    })
}

/// Assert that an actual tensor matches a reference tensor within tolerance.
///
/// Panics with a full comparison report on mismatch.
pub fn assert_tensors_close(
    actual: &[u8],
    actual_dtype: DType,
    reference_path: &str,
    rtol: f32,
    atol: f32,
) {
    let reference = require_reference(reference_path);
    let result = compare_tensors(actual, actual_dtype, &reference, rtol, atol);
    if !result.matches {
        panic!(
            "Tensor mismatch against reference '{}':\n{}",
            reference_path, result
        );
    }
}
