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
