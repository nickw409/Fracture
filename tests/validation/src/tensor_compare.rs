//! Reference tensor loading and element-wise comparison.

use half::f16;
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::Path;

/// Data type of a reference tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F16 = 0,
    F32 = 1,
    I32 = 2,
}

impl DType {
    /// Bytes per element for this dtype.
    pub fn element_size(self) -> usize {
        match self {
            DType::F16 => 2,
            DType::F32 => 4,
            DType::I32 => 4,
        }
    }

    fn from_u32(v: u32) -> Result<Self, String> {
        match v {
            0 => Ok(DType::F16),
            1 => Ok(DType::F32),
            2 => Ok(DType::I32),
            _ => Err(format!("Unknown dtype enum value: {}", v)),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F16 => write!(f, "float16"),
            DType::F32 => write!(f, "float32"),
            DType::I32 => write!(f, "int32"),
        }
    }
}

/// A reference tensor loaded from the binary format.
#[derive(Debug, Clone)]
pub struct ReferenceTensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<u8>,
}

impl ReferenceTensor {
    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Convert all elements to f32, regardless of stored dtype.
    pub fn to_f32(&self) -> Vec<f32> {
        bytes_to_f32(&self.data, self.dtype)
    }
}

/// Load a reference tensor from the Fracture binary format.
pub fn load_reference_tensor(path: &str) -> Result<ReferenceTensor, String> {
    let p = Path::new(path);
    if !p.exists() {
        return Err(format!("File not found: {}", path));
    }

    let data = fs::read(p).map_err(|e| format!("Failed to read '{}': {}", path, e))?;
    parse_reference_tensor(&data, path)
}

/// Parse a reference tensor from raw bytes (useful for testing).
pub fn parse_reference_tensor(data: &[u8], label: &str) -> Result<ReferenceTensor, String> {
    let mut cursor = io::Cursor::new(data);
    let mut buf4 = [0u8; 4];

    // ndim
    cursor.read_exact(&mut buf4).map_err(|e| format!("{}: failed to read ndim: {}", label, e))?;
    let ndim = u32::from_le_bytes(buf4) as usize;

    // shape
    let mut shape = Vec::with_capacity(ndim);
    for i in 0..ndim {
        cursor
            .read_exact(&mut buf4)
            .map_err(|e| format!("{}: failed to read shape[{}]: {}", label, i, e))?;
        shape.push(u32::from_le_bytes(buf4) as usize);
    }

    // dtype
    cursor
        .read_exact(&mut buf4)
        .map_err(|e| format!("{}: failed to read dtype: {}", label, e))?;
    let dtype = DType::from_u32(u32::from_le_bytes(buf4))
        .map_err(|e| format!("{}: {}", label, e))?;

    let num_elements: usize = shape.iter().product();
    let expected_bytes = num_elements * dtype.element_size();
    let header_bytes = 4 + ndim * 4 + 4;
    let remaining = data.len().saturating_sub(header_bytes);

    if remaining != expected_bytes {
        return Err(format!(
            "{}: expected {} data bytes (shape {:?}, dtype {}), got {}",
            label, expected_bytes, shape, dtype, remaining
        ));
    }

    let tensor_data = data[header_bytes..].to_vec();
    Ok(ReferenceTensor {
        shape,
        dtype,
        data: tensor_data,
    })
}

/// Convert raw bytes of a given dtype to f32 values.
pub fn bytes_to_f32(data: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F16 => data
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::I32 => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
    }
}

/// Detail about a single element mismatch.
#[derive(Debug, Clone)]
pub struct MismatchDetail {
    pub index: Vec<usize>,
    pub expected: f32,
    pub actual: f32,
    pub abs_error: f32,
}

/// Result of an element-wise tensor comparison.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub matches: bool,
    pub max_abs_error: f32,
    pub max_abs_error_index: Vec<usize>,
    pub mean_abs_error: f32,
    pub num_mismatches: usize,
    pub total_elements: usize,
    pub rtol: f32,
    pub atol: f32,
    pub first_mismatches: Vec<MismatchDetail>,
}

impl fmt::Display for ComparisonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.matches {
            write!(
                f,
                "MATCH: all {} elements within tolerance (rtol={}, atol={})\n\
                 Max absolute error: {:.6} at index {:?}\n\
                 Mean absolute error: {:.6}",
                self.total_elements,
                self.rtol,
                self.atol,
                self.max_abs_error,
                self.max_abs_error_index,
                self.mean_abs_error,
            )
        } else {
            write!(
                f,
                "MISMATCH: {} of {} elements exceed tolerance (rtol={}, atol={})\n\
                 Max absolute error: {:.6} at index {:?}\n\
                 Mean absolute error: {:.6}",
                self.num_mismatches,
                self.total_elements,
                self.rtol,
                self.atol,
                self.max_abs_error,
                self.max_abs_error_index,
                self.mean_abs_error,
            )?;
            if !self.first_mismatches.is_empty() {
                write!(f, "\n  First {} mismatches:", self.first_mismatches.len())?;
                for m in &self.first_mismatches {
                    write!(
                        f,
                        "\n    {:?}:  expected={:.6}  actual={:.6}  error={:.6}",
                        m.index, m.expected, m.actual, m.abs_error
                    )?;
                }
            }
            Ok(())
        }
    }
}

/// Convert a flat index to a multi-dimensional index given a shape.
fn flat_to_multi(flat: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut idx = vec![0usize; shape.len()];
    let mut remaining = flat;
    for i in (0..shape.len()).rev() {
        idx[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    idx
}

/// Compare two tensors element-wise.
///
/// Both are converted to f32 internally.
/// An element matches if: `|actual - expected| <= atol + rtol * |expected|`
pub fn compare_tensors(
    actual: &[u8],
    actual_dtype: DType,
    expected: &ReferenceTensor,
    rtol: f32,
    atol: f32,
) -> ComparisonResult {
    let actual_f32 = bytes_to_f32(actual, actual_dtype);
    let expected_f32 = expected.to_f32();

    let total = expected_f32.len();

    // Handle size mismatch
    if actual_f32.len() != total {
        return ComparisonResult {
            matches: false,
            max_abs_error: f32::INFINITY,
            max_abs_error_index: vec![],
            mean_abs_error: f32::INFINITY,
            num_mismatches: total,
            total_elements: total,
            rtol,
            atol,
            first_mismatches: vec![MismatchDetail {
                index: vec![],
                expected: 0.0,
                actual: 0.0,
                abs_error: f32::INFINITY,
            }],
        };
    }

    // Handle empty tensors
    if total == 0 {
        return ComparisonResult {
            matches: true,
            max_abs_error: 0.0,
            max_abs_error_index: vec![],
            mean_abs_error: 0.0,
            num_mismatches: 0,
            total_elements: 0,
            rtol,
            atol,
            first_mismatches: vec![],
        };
    }

    let mut max_abs_error: f32 = 0.0;
    let mut max_abs_error_flat: usize = 0;
    let mut sum_abs_error: f64 = 0.0;
    let mut num_mismatches: usize = 0;
    let mut first_mismatches: Vec<MismatchDetail> = Vec::new();

    for i in 0..total {
        let a = actual_f32[i];
        let e = expected_f32[i];
        let abs_err = (a - e).abs();

        sum_abs_error += abs_err as f64;

        if abs_err > max_abs_error {
            max_abs_error = abs_err;
            max_abs_error_flat = i;
        }

        let tol = atol + rtol * e.abs();
        if abs_err > tol {
            num_mismatches += 1;
            if first_mismatches.len() < 5 {
                first_mismatches.push(MismatchDetail {
                    index: flat_to_multi(i, &expected.shape),
                    expected: e,
                    actual: a,
                    abs_error: abs_err,
                });
            }
        }
    }

    ComparisonResult {
        matches: num_mismatches == 0,
        max_abs_error,
        max_abs_error_index: flat_to_multi(max_abs_error_flat, &expected.shape),
        mean_abs_error: (sum_abs_error / total as f64) as f32,
        num_mismatches,
        total_elements: total,
        rtol,
        atol,
        first_mismatches,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a binary tensor file in memory.
    fn build_tensor_bytes(shape: &[u32], dtype: DType, data: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        let ndim = shape.len() as u32;
        buf.extend_from_slice(&ndim.to_le_bytes());
        for &dim in shape {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf.extend_from_slice(&(dtype as u32).to_le_bytes());
        buf.extend_from_slice(data);
        buf
    }

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn f16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect()
    }

    fn i32_bytes(vals: &[i32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn test_load_reference_tensor_f32() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let raw = f32_bytes(&values);
        let blob = build_tensor_bytes(&[2, 3], DType::F32, &raw);

        let tensor = parse_reference_tensor(&blob, "test").unwrap();
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.dtype, DType::F32);
        assert_eq!(tensor.num_elements(), 6);

        let f32_data = tensor.to_f32();
        assert_eq!(f32_data, values);
    }

    #[test]
    fn test_load_reference_tensor_f16() {
        let values = [0.5f32, -1.0, 3.14];
        let raw = f16_bytes(&values);
        let blob = build_tensor_bytes(&[3], DType::F16, &raw);

        let tensor = parse_reference_tensor(&blob, "test").unwrap();
        assert_eq!(tensor.shape, vec![3]);
        assert_eq!(tensor.dtype, DType::F16);

        let f32_data = tensor.to_f32();
        for (a, e) in f32_data.iter().zip(values.iter()) {
            assert!((a - e).abs() < 0.01, "f16 round-trip: {} vs {}", a, e);
        }
    }

    #[test]
    fn test_load_reference_tensor_i32() {
        let values = [42i32, -7, 0, 100000];
        let raw = i32_bytes(&values);
        let blob = build_tensor_bytes(&[4], DType::I32, &raw);

        let tensor = parse_reference_tensor(&blob, "test").unwrap();
        assert_eq!(tensor.shape, vec![4]);
        assert_eq!(tensor.dtype, DType::I32);

        let f32_data = tensor.to_f32();
        for (a, e) in f32_data.iter().zip(values.iter()) {
            assert_eq!(*a, *e as f32);
        }
    }

    #[test]
    fn test_load_tensor_size_mismatch() {
        let raw = f32_bytes(&[1.0, 2.0]); // 2 elements
        let blob = build_tensor_bytes(&[3], DType::F32, &raw); // claims 3
        let result = parse_reference_tensor(&blob, "test");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("expected"));
    }

    #[test]
    fn test_compare_matching_tensors() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let raw = f32_bytes(&values);
        let blob = build_tensor_bytes(&[2, 2], DType::F32, &raw);
        let reference = parse_reference_tensor(&blob, "ref").unwrap();

        let actual = f32_bytes(&values);
        let result = compare_tensors(&actual, DType::F32, &reference, 1e-3, 1e-3);

        assert!(result.matches);
        assert_eq!(result.num_mismatches, 0);
        assert_eq!(result.total_elements, 4);
        assert_eq!(result.max_abs_error, 0.0);
        assert_eq!(result.mean_abs_error, 0.0);
    }

    #[test]
    fn test_compare_mismatching_tensors() {
        let expected_vals = [1.0f32, 2.0, 3.0, 4.0];
        let actual_vals = [1.0f32, 2.5, 3.0, 4.1]; // index 1 and 3 differ

        let raw = f32_bytes(&expected_vals);
        let blob = build_tensor_bytes(&[2, 2], DType::F32, &raw);
        let reference = parse_reference_tensor(&blob, "ref").unwrap();

        let actual = f32_bytes(&actual_vals);
        let result = compare_tensors(&actual, DType::F32, &reference, 1e-3, 1e-3);

        assert!(!result.matches);
        assert_eq!(result.num_mismatches, 2);
        assert_eq!(result.total_elements, 4);

        // Max error should be 0.5 at index [0, 1]
        assert!((result.max_abs_error - 0.5).abs() < 1e-6);
        assert_eq!(result.max_abs_error_index, vec![0, 1]);

        // Check first mismatch details
        assert_eq!(result.first_mismatches.len(), 2);
        assert_eq!(result.first_mismatches[0].index, vec![0, 1]);
        assert!((result.first_mismatches[0].expected - 2.0).abs() < 1e-6);
        assert!((result.first_mismatches[0].actual - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_compare_within_tolerance() {
        let expected_vals = [1.0f32, 2.0, 3.0];
        let actual_vals = [1.0005f32, 2.001, 3.002]; // within rtol=1e-3, atol=1e-3

        let raw = f32_bytes(&expected_vals);
        let blob = build_tensor_bytes(&[3], DType::F32, &raw);
        let reference = parse_reference_tensor(&blob, "ref").unwrap();

        let actual = f32_bytes(&actual_vals);
        let result = compare_tensors(&actual, DType::F32, &reference, 1e-3, 1e-3);

        assert!(result.matches, "Expected match but got: {}", result);
    }

    #[test]
    fn test_compare_f16_actual_vs_f32_reference() {
        let expected_vals = [1.0f32, 0.5, -0.25, 0.0];
        let raw = f32_bytes(&expected_vals);
        let blob = build_tensor_bytes(&[4], DType::F32, &raw);
        let reference = parse_reference_tensor(&blob, "ref").unwrap();

        // Actual is f16 — should match within f16 precision
        let actual = f16_bytes(&expected_vals);
        let result = compare_tensors(&actual, DType::F16, &reference, 1e-3, 1e-3);

        assert!(result.matches, "f16 vs f32 comparison failed: {}", result);
    }

    #[test]
    fn test_compare_empty_tensors() {
        let blob = build_tensor_bytes(&[0], DType::F32, &[]);
        let reference = parse_reference_tensor(&blob, "ref").unwrap();

        let result = compare_tensors(&[], DType::F32, &reference, 1e-3, 1e-3);
        assert!(result.matches);
        assert_eq!(result.total_elements, 0);
    }

    #[test]
    fn test_compare_single_element() {
        let raw = f32_bytes(&[42.0]);
        let blob = build_tensor_bytes(&[1], DType::F32, &raw);
        let reference = parse_reference_tensor(&blob, "ref").unwrap();

        let actual = f32_bytes(&[42.0]);
        let result = compare_tensors(&actual, DType::F32, &reference, 1e-6, 1e-6);
        assert!(result.matches);
        assert_eq!(result.total_elements, 1);
    }

    #[test]
    fn test_compare_all_zeros() {
        let raw = f32_bytes(&[0.0; 100]);
        let blob = build_tensor_bytes(&[10, 10], DType::F32, &raw);
        let reference = parse_reference_tensor(&blob, "ref").unwrap();

        let actual = f32_bytes(&[0.0; 100]);
        let result = compare_tensors(&actual, DType::F32, &reference, 0.0, 0.0);
        assert!(result.matches);
        assert_eq!(result.total_elements, 100);
        assert_eq!(result.max_abs_error, 0.0);
    }

    #[test]
    fn test_compare_size_mismatch_actual_vs_expected() {
        let raw = f32_bytes(&[1.0, 2.0, 3.0]);
        let blob = build_tensor_bytes(&[3], DType::F32, &raw);
        let reference = parse_reference_tensor(&blob, "ref").unwrap();

        // Actual has 2 elements instead of 3
        let actual = f32_bytes(&[1.0, 2.0]);
        let result = compare_tensors(&actual, DType::F32, &reference, 1e-3, 1e-3);
        assert!(!result.matches);
        assert_eq!(result.num_mismatches, 3); // all reported as mismatches
    }

    #[test]
    fn test_flat_to_multi_index() {
        assert_eq!(flat_to_multi(0, &[2, 3]), vec![0, 0]);
        assert_eq!(flat_to_multi(1, &[2, 3]), vec![0, 1]);
        assert_eq!(flat_to_multi(3, &[2, 3]), vec![1, 0]);
        assert_eq!(flat_to_multi(5, &[2, 3]), vec![1, 2]);

        // 3D
        assert_eq!(flat_to_multi(0, &[2, 3, 4]), vec![0, 0, 0]);
        assert_eq!(flat_to_multi(13, &[2, 3, 4]), vec![1, 0, 1]);
    }

    #[test]
    fn test_display_match() {
        let result = ComparisonResult {
            matches: true,
            max_abs_error: 0.0001,
            max_abs_error_index: vec![0, 5],
            mean_abs_error: 0.00005,
            num_mismatches: 0,
            total_elements: 100,
            rtol: 1e-3,
            atol: 1e-3,
            first_mismatches: vec![],
        };
        let s = format!("{}", result);
        assert!(s.contains("MATCH"));
        assert!(s.contains("100"));
    }

    #[test]
    fn test_display_mismatch() {
        let result = ComparisonResult {
            matches: false,
            max_abs_error: 0.5,
            max_abs_error_index: vec![0, 1],
            mean_abs_error: 0.1,
            num_mismatches: 3,
            total_elements: 10,
            rtol: 1e-3,
            atol: 1e-3,
            first_mismatches: vec![MismatchDetail {
                index: vec![0, 1],
                expected: 2.0,
                actual: 2.5,
                abs_error: 0.5,
            }],
        };
        let s = format!("{}", result);
        assert!(s.contains("MISMATCH"));
        assert!(s.contains("3 of 10"));
    }
}
