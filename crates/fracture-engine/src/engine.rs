use fracture_core::{Backend, DeviceTensor, ModelConfig, Result};
use crate::kv_cache::{CacheHandle, KvCacheManager};
use std::ops::Range;

/// The backend-agnostic transformer forward pass engine.
///
/// Generic over `B: Backend` — contains no CUDA or Metal imports.
/// Dispatches all GPU operations through Backend trait methods.
#[allow(dead_code)] // layer_range used in Phase 2
pub struct Engine<B: Backend> {
    backend: B,
    config: ModelConfig,
    layer_range: Range<usize>,
}

impl<B: Backend> Engine<B> {
    pub fn new(backend: B, config: ModelConfig, layer_range: Range<usize>) -> Self {
        Self {
            backend,
            config,
            layer_range,
        }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Run the forward pass.
    ///
    /// In Phase 1, layer_range is always [0, num_layers) and this accepts token IDs
    /// and returns logits. In Phase 2, a partial layer range accepts/returns activation tensors.
    pub fn forward(
        &self,
        _token_ids: &[u32],
        _cache: &mut KvCacheManager,
        _cache_handle: CacheHandle,
    ) -> Result<DeviceTensor> {
        // TODO: Implement transformer forward pass
        //   embed → for each layer in range: rmsnorm → QKV proj → rope →
        //   cache update → attention → output proj → residual → rmsnorm →
        //   gate/up proj → silu_mul → down proj → residual → final norm → lm_head
        Err(fracture_core::FractureError::Backend(
            "forward pass not yet implemented".into(),
        ))
    }
}
