use fracture_core::{Backend, DeviceTensor, ModelConfig, Result};

/// Per-layer weight tensors on device.
pub struct LayerWeights {
    pub q_proj: DeviceTensor,
    pub k_proj: DeviceTensor,
    pub v_proj: DeviceTensor,
    pub o_proj: DeviceTensor,
    pub gate_proj: DeviceTensor,
    pub up_proj: DeviceTensor,
    pub down_proj: DeviceTensor,
    pub attn_norm: DeviceTensor,
    pub ffn_norm: DeviceTensor,
}

/// Holds all model weights on device, organized for fast access during inference.
pub struct WeightStore {
    pub config: ModelConfig,
    pub token_embedding: DeviceTensor,
    pub layers: Vec<LayerWeights>,
    pub output_norm: DeviceTensor,
    pub lm_head: DeviceTensor,
}

impl WeightStore {
    /// Load weights from a GGUF file onto the given backend.
    pub fn load<B: Backend>(
        _path: &std::path::Path,
        _backend: &B,
        _layer_range: Option<std::ops::Range<usize>>,
    ) -> Result<Self> {
        // TODO: Parse GGUF, allocate device tensors, copy weights
        Err(fracture_core::FractureError::WeightLoad(
            "not yet implemented".into(),
        ))
    }
}
