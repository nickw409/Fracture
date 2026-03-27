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
