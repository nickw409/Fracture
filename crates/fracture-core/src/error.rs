use thiserror::Error;

pub type Result<T> = std::result::Result<T, FractureError>;

#[derive(Debug, Error)]
pub enum FractureError {
    #[error("backend error: {0}")]
    Backend(String),

    #[error("out of device memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory { requested: usize, available: usize },

    #[error("invalid tensor shape: {0}")]
    InvalidShape(String),

    #[error("tensor not found: {0}")]
    TensorNotFound(String),

    #[error("model config error: {0}")]
    ModelConfig(String),

    #[error("GGUF parse error: {0}")]
    GgufParse(String),

    #[error("weight loading error: {0}")]
    WeightLoad(String),

    #[error("KV cache error: {0}")]
    KvCache(String),

    #[error("generation error: {0}")]
    Generation(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("server error: {0}")]
    Server(String),

    #[error("unsupported dtype: {0}")]
    UnsupportedDType(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_messages() {
        let err = FractureError::Backend("gpu fail".into());
        assert!(err.to_string().contains("gpu fail"));

        let err = FractureError::OutOfMemory {
            requested: 1024,
            available: 512,
        };
        let msg = err.to_string();
        assert!(msg.contains("1024"), "expected '1024' in: {msg}");
        assert!(msg.contains("512"), "expected '512' in: {msg}");

        let err = FractureError::InvalidShape("bad dims".into());
        assert!(err.to_string().contains("bad dims"));

        let err = FractureError::TensorNotFound("missing.weight".into());
        assert!(err.to_string().contains("missing.weight"));

        let err = FractureError::ModelConfig("bad config".into());
        assert!(err.to_string().contains("bad config"));

        let err = FractureError::GgufParse("corrupt header".into());
        assert!(err.to_string().contains("corrupt header"));

        let err = FractureError::WeightLoad("mmap fail".into());
        assert!(err.to_string().contains("mmap fail"));

        let err = FractureError::KvCache("out of slots".into());
        assert!(err.to_string().contains("out of slots"));

        let err = FractureError::Generation("eos not found".into());
        assert!(err.to_string().contains("eos not found"));

        let err = FractureError::Tokenizer("encode fail".into());
        assert!(err.to_string().contains("encode fail"));

        let err = FractureError::Server("bind failed".into());
        assert!(err.to_string().contains("bind failed"));

        let err = FractureError::UnsupportedDType("Q4_0".into());
        assert!(err.to_string().contains("Q4_0"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file gone");
        let fracture_err: FractureError = io_err.into();
        assert!(matches!(fracture_err, FractureError::Io(_)));
        assert!(fracture_err.to_string().contains("file gone"));
    }
}
