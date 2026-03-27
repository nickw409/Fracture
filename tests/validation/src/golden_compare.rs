//! Golden token sequence comparison.

use std::fmt;
use std::fs;
use std::path::Path;

/// Result of comparing two token sequences.
#[derive(Debug, Clone)]
pub struct TokenComparisonResult {
    pub matching_tokens: usize,
    pub total_expected: usize,
    pub total_actual: usize,
    pub divergence_index: Option<usize>,
    pub expected_token_at_divergence: Option<u32>,
    pub actual_token_at_divergence: Option<u32>,
}

impl TokenComparisonResult {
    pub fn matches(&self) -> bool {
        self.divergence_index.is_none() && self.total_actual == self.total_expected
    }
}

impl fmt::Display for TokenComparisonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.matches() {
            write!(
                f,
                "TOKEN MATCH: all {} tokens identical",
                self.total_expected
            )
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

/// Load a golden token sequence from the binary format.
///
/// The file uses the same format as other reference tensors:
/// `[ndim][shape...][dtype][data]` where dtype=2 (int32).
pub fn load_golden_tokens(path: &str) -> Result<Vec<u32>, String> {
    let p = Path::new(path);
    if !p.exists() {
        return Err(format!("Golden token file not found: {}", path));
    }

    let data = fs::read(p).map_err(|e| format!("Failed to read '{}': {}", path, e))?;

    if data.len() < 8 {
        return Err(format!("File too small: {} bytes", data.len()));
    }

    let ndim = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let mut offset = 4;

    let mut num_elements: usize = 1;
    for _ in 0..ndim {
        if offset + 4 > data.len() {
            return Err("Truncated shape".to_string());
        }
        let dim = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        num_elements *= dim;
        offset += 4;
    }

    if offset + 4 > data.len() {
        return Err("Truncated dtype".to_string());
    }
    let dtype_enum = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]);
    offset += 4;

    if dtype_enum != 2 {
        return Err(format!(
            "Expected dtype int32 (2) for token sequence, got {}",
            dtype_enum
        ));
    }

    let expected_bytes = num_elements * 4;
    let remaining = data.len() - offset;
    if remaining != expected_bytes {
        return Err(format!(
            "Data size mismatch: expected {} bytes, got {}",
            expected_bytes, remaining
        ));
    }

    let tokens: Vec<u32> = data[offset..]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok(tokens)
}

/// Compare two token sequences, reporting the first divergence point.
pub fn compare_token_sequences(actual: &[u32], expected: &[u32]) -> TokenComparisonResult {
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

    // All compared tokens match; check length
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matching_sequences() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 4, 5];
        let result = compare_token_sequences(&a, &b);
        assert!(result.matches());
        assert_eq!(result.matching_tokens, 5);
        assert_eq!(result.divergence_index, None);
    }

    #[test]
    fn test_diverging_sequences() {
        let actual = vec![1, 2, 99, 4, 5];
        let expected = vec![1, 2, 3, 4, 5];
        let result = compare_token_sequences(&actual, &expected);
        assert!(!result.matches());
        assert_eq!(result.matching_tokens, 2);
        assert_eq!(result.divergence_index, Some(2));
        assert_eq!(result.expected_token_at_divergence, Some(3));
        assert_eq!(result.actual_token_at_divergence, Some(99));
    }

    #[test]
    fn test_actual_shorter() {
        let actual = vec![1, 2, 3];
        let expected = vec![1, 2, 3, 4, 5];
        let result = compare_token_sequences(&actual, &expected);
        assert!(!result.matches());
        assert_eq!(result.matching_tokens, 3);
        assert_eq!(result.divergence_index, Some(3));
        assert_eq!(result.expected_token_at_divergence, Some(4));
        assert_eq!(result.actual_token_at_divergence, None);
    }

    #[test]
    fn test_actual_longer() {
        let actual = vec![1, 2, 3, 4, 5, 6];
        let expected = vec![1, 2, 3, 4, 5];
        let result = compare_token_sequences(&actual, &expected);
        assert!(!result.matches());
        assert_eq!(result.matching_tokens, 5);
        assert_eq!(result.divergence_index, Some(5));
        assert_eq!(result.expected_token_at_divergence, None);
        assert_eq!(result.actual_token_at_divergence, Some(6));
    }

    #[test]
    fn test_empty_sequences() {
        let result = compare_token_sequences(&[], &[]);
        assert!(result.matches());
        assert_eq!(result.matching_tokens, 0);
    }

    #[test]
    fn test_actual_empty_expected_nonempty() {
        let result = compare_token_sequences(&[], &[1, 2, 3]);
        assert!(!result.matches());
        assert_eq!(result.divergence_index, Some(0));
    }

    #[test]
    fn test_first_token_divergence() {
        let result = compare_token_sequences(&[99], &[1]);
        assert!(!result.matches());
        assert_eq!(result.matching_tokens, 0);
        assert_eq!(result.divergence_index, Some(0));
    }

    #[test]
    fn test_display_match() {
        let result = compare_token_sequences(&[1, 2, 3], &[1, 2, 3]);
        let s = format!("{}", result);
        assert!(s.contains("MATCH"));
    }

    #[test]
    fn test_display_mismatch() {
        let result = compare_token_sequences(&[1, 99], &[1, 2, 3]);
        let s = format!("{}", result);
        assert!(s.contains("MISMATCH"));
        assert!(s.contains("index 1"));
    }
}
