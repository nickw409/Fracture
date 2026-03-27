use crate::{FractureError, Result};
use serde::{Deserialize, Serialize};

/// Model architecture configuration extracted from GGUF metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub max_seq_len: usize,
}

impl ModelConfig {
    /// Validates that the config is internally consistent.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(FractureError::ModelConfig("hidden_size must be > 0".into()));
        }
        if self.num_layers == 0 {
            return Err(FractureError::ModelConfig("num_layers must be > 0".into()));
        }
        if self.num_q_heads == 0 {
            return Err(FractureError::ModelConfig("num_q_heads must be > 0".into()));
        }
        if self.num_kv_heads == 0 {
            return Err(FractureError::ModelConfig("num_kv_heads must be > 0".into()));
        }
        if self.vocab_size == 0 {
            return Err(FractureError::ModelConfig("vocab_size must be > 0".into()));
        }

        let expected_head_dim = self.hidden_size / self.num_q_heads;
        if self.head_dim != expected_head_dim {
            return Err(FractureError::ModelConfig(format!(
                "head_dim ({}) != hidden_size ({}) / num_q_heads ({})",
                self.head_dim, self.hidden_size, self.num_q_heads
            )));
        }

        if self.num_q_heads % self.num_kv_heads != 0 {
            return Err(FractureError::ModelConfig(format!(
                "num_q_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_q_heads, self.num_kv_heads
            )));
        }

        Ok(())
    }

    /// Number of query heads per KV head (GQA group size).
    pub fn gqa_group_size(&self) -> usize {
        self.num_q_heads / self.num_kv_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama3_8b_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 4096,
            num_layers: 32,
            num_q_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 14336,
            vocab_size: 128256,
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            max_seq_len: 8192,
        }
    }

    #[test]
    fn test_valid_llama3_8b_config() {
        assert!(llama3_8b_config().validate().is_ok());
    }

    #[test]
    fn test_head_dim_mismatch_fails() {
        let mut cfg = llama3_8b_config();
        cfg.head_dim = 64; // should be 4096/32 = 128
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_gqa_divisibility_fails() {
        let mut cfg = llama3_8b_config();
        cfg.num_kv_heads = 7; // 32 % 7 != 0
        // head_dim would also be wrong, fix it to isolate the GQA check
        // Actually head_dim check comes first, so we need valid head_dim.
        // 32 heads, hidden=4096 => head_dim=128, that's fine.
        // But 32 % 7 != 0 should fail after head_dim passes.
        let result = cfg.validate();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("divisible"));
    }

    #[test]
    fn test_zero_hidden_size_fails() {
        let mut cfg = llama3_8b_config();
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_gqa_group_size() {
        let cfg = llama3_8b_config();
        assert_eq!(cfg.gqa_group_size(), 4);
    }
}
