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

    #[error("pipeline error: {0}")]
    Pipeline(String),

    #[error("protocol error: {0}")]
    Protocol(String),

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

        let err = FractureError::Pipeline("stage 2 failed".into());
        assert!(err.to_string().contains("stage 2 failed"));

        let err = FractureError::Protocol("CRC mismatch".into());
        assert!(err.to_string().contains("CRC mismatch"));

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

    #[test]
    fn test_error_context_propagation() {
        // Nested error context: layer -> kernel -> tensor detail
        let tensor_ctx = "tensor 'q_proj.weight' shape [4096, 4096]";
        let kernel_ctx = format!("kernel matmul failed: {tensor_ctx}");
        let layer_ctx = format!("layer 7: {kernel_ctx}");

        let err = FractureError::Backend(layer_ctx.clone());
        let display = err.to_string();

        // Full chain should be visible in Display output
        assert!(
            display.contains("layer 7"),
            "expected 'layer 7' in: {display}"
        );
        assert!(
            display.contains("kernel matmul failed"),
            "expected kernel context in: {display}"
        );
        assert!(
            display.contains("q_proj.weight"),
            "expected tensor name in: {display}"
        );
        assert!(
            display.contains("[4096, 4096]"),
            "expected tensor shape in: {display}"
        );

        // Also verify with KvCache variant for a different error path
        let kv_err = FractureError::KvCache(format!(
            "layer 12: cache overflow at pos 8192 for tensor '{tensor_ctx}'"
        ));
        let kv_display = kv_err.to_string();
        assert!(kv_display.contains("layer 12"), "expected layer in: {kv_display}");
        assert!(kv_display.contains("q_proj.weight"), "expected tensor in: {kv_display}");
    }

    #[test]
    fn test_error_context_contains_tensor_info() {
        let err = FractureError::TensorNotFound("tensor id 42".into());
        let display = err.to_string();
        assert!(
            display.contains("42"),
            "TensorNotFound should contain tensor id '42' in: {display}"
        );
    }

    #[test]
    fn test_error_context_contains_layer_info() {
        let err = FractureError::KvCache("invalid handle: 7".into());
        let display = err.to_string();
        assert!(
            display.contains("7"),
            "KvCache error should contain handle id '7' in: {display}"
        );
    }

    #[test]
    fn test_error_chain_preserves_context() {
        let err =
            FractureError::WeightLoad("blk.3.attn_q.weight: size mismatch".into());
        let display = err.to_string();
        assert!(
            display.contains("blk.3.attn_q.weight"),
            "WeightLoad should contain full tensor name in: {display}"
        );
        assert!(
            display.contains("size mismatch"),
            "WeightLoad should contain error detail in: {display}"
        );
    }

    #[test]
    fn test_no_panic_on_invalid_inputs() {
        use crate::{DeviceTensor, DType, ModelConfig, TensorId};

        // Zero-element shape: numel should be 0, size_bytes should be 0
        let t = DeviceTensor::new(TensorId(0), vec![0], DType::FP32);
        assert_eq!(t.numel(), 0);
        assert_eq!(t.size_bytes(), 0);

        // Empty shape: numel of empty product is 1 by convention (iter::product)
        let t2 = DeviceTensor::new(TensorId(1), vec![], DType::FP16);
        assert_eq!(t2.numel(), 1); // product of empty iterator = 1
        assert_eq!(t2.size_bytes(), 2); // 1 * 2 bytes

        // Multi-dim with a zero
        let t3 = DeviceTensor::new(TensorId(2), vec![3, 0, 4], DType::FP32);
        assert_eq!(t3.numel(), 0);
        assert_eq!(t3.size_bytes(), 0);

        // INT4 with zero elements
        let t4 = DeviceTensor::new(TensorId(3), vec![0], DType::INT4);
        assert_eq!(t4.numel(), 0);
        assert_eq!(t4.size_bytes(), 0);

        // ModelConfig::validate with zero fields returns Err, not panic
        let bad_config = ModelConfig {
            hidden_size: 0,
            num_layers: 0,
            num_q_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            intermediate_size: 0,
            vocab_size: 0,
            rope_theta: 0.0,
            rms_norm_eps: 0.0,
            max_seq_len: 0,
        };
        // Should return Err, not panic
        assert!(bad_config.validate().is_err());
    }
}
