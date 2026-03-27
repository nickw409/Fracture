use fracture_core::{FractureError, ModelConfig, Result};

/// Parsed GGUF tensor descriptor.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: fracture_core::DType,
    pub offset: u64,
}

/// Parses GGUF v3 binary files.
pub struct GgufParser;

impl GgufParser {
    /// Parse a GGUF file, returning model config and tensor descriptors.
    pub fn parse(_path: &std::path::Path) -> Result<(ModelConfig, Vec<TensorInfo>)> {
        // TODO: Implement GGUF header, metadata, and tensor info parsing
        Err(FractureError::GgufParse("not yet implemented".into()))
    }
}
