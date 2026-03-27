use serde::{Deserialize, Serialize};

/// Supported data types for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    FP16,
    FP32,
    BF16,
    INT8,
    INT4,
}

impl DType {
    /// Returns the size in bytes of a single element.
    /// INT4 returns 1 because elements are packed in pairs (0.5 bytes each),
    /// but the minimum addressable unit is 1 byte per 2 elements.
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::FP16 => 2,
            DType::FP32 => 4,
            DType::BF16 => 2,
            DType::INT8 => 1,
            DType::INT4 => 1, // 2 elements per byte, handled at allocation level
        }
    }

    /// Returns true if this type stores two elements per byte.
    pub fn is_packed(&self) -> bool {
        matches!(self, DType::INT4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_bytes() {
        assert_eq!(DType::FP16.size_bytes(), 2);
        assert_eq!(DType::FP32.size_bytes(), 4);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::INT8.size_bytes(), 1);
        assert_eq!(DType::INT4.size_bytes(), 1);
    }

    #[test]
    fn test_is_packed() {
        assert!(!DType::FP16.is_packed());
        assert!(!DType::FP32.is_packed());
        assert!(!DType::BF16.is_packed());
        assert!(!DType::INT8.is_packed());
        assert!(DType::INT4.is_packed());
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(format!("{}", DType::FP16), "fp16");
        assert_eq!(format!("{}", DType::FP32), "fp32");
        assert_eq!(format!("{}", DType::BF16), "bf16");
        assert_eq!(format!("{}", DType::INT8), "int8");
        assert_eq!(format!("{}", DType::INT4), "int4");
    }

    #[test]
    fn test_dtype_serde_roundtrip() {
        let variants = [DType::FP16, DType::FP32, DType::BF16, DType::INT8, DType::INT4];
        for dt in &variants {
            let json = serde_json::to_string(dt).unwrap();
            let back: DType = serde_json::from_str(&json).unwrap();
            assert_eq!(*dt, back, "roundtrip failed for {dt}");
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::FP16 => write!(f, "fp16"),
            DType::FP32 => write!(f, "fp32"),
            DType::BF16 => write!(f, "bf16"),
            DType::INT8 => write!(f, "int8"),
            DType::INT4 => write!(f, "int4"),
        }
    }
}
