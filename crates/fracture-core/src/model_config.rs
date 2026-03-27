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
