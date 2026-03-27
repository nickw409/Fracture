use fracture_core::{Backend, DType, DeviceTensor, FractureError, Result};
use std::collections::HashMap;

/// Opaque handle to a sequence's KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheHandle(pub u64);

/// Per-layer cache entry for a single sequence.
struct LayerCache {
    k: DeviceTensor,
    v: DeviceTensor,
}

/// Per-sequence cache state.
struct SequenceCache {
    layers: Vec<LayerCache>,
    current_len: usize,
    max_len: usize,
}

/// Manages GPU memory for KV caches across all active sequences.
pub struct KvCacheManager {
    caches: HashMap<u64, SequenceCache>,
    next_id: u64,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
}

impl KvCacheManager {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            caches: HashMap::new(),
            next_id: 0,
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
        }
    }

    /// Allocate KV cache for a new sequence.
    pub fn alloc<B: Backend>(&mut self, backend: &B) -> Result<CacheHandle> {
        let id = self.next_id;
        self.next_id += 1;

        let mut layers = Vec::with_capacity(self.num_layers);
        for _ in 0..self.num_layers {
            let k = backend.alloc(
                &[self.max_seq_len, self.num_kv_heads, self.head_dim],
                DType::FP16,
            )?;
            let v = backend.alloc(
                &[self.max_seq_len, self.num_kv_heads, self.head_dim],
                DType::FP16,
            )?;
            layers.push(LayerCache { k, v });
        }

        self.caches.insert(
            id,
            SequenceCache {
                layers,
                current_len: 0,
                max_len: self.max_seq_len,
            },
        );

        Ok(CacheHandle(id))
    }

    /// Get the current sequence length for a cache.
    pub fn seq_len(&self, handle: CacheHandle) -> Result<usize> {
        self.caches
            .get(&handle.0)
            .map(|c| c.current_len)
            .ok_or_else(|| FractureError::KvCache(format!("invalid handle: {}", handle.0)))
    }

    /// Update the sequence length after appending tokens.
    pub fn set_seq_len(&mut self, handle: CacheHandle, new_len: usize) -> Result<()> {
        let cache = self
            .caches
            .get_mut(&handle.0)
            .ok_or_else(|| FractureError::KvCache(format!("invalid handle: {}", handle.0)))?;

        if new_len > cache.max_len {
            return Err(FractureError::KvCache(format!(
                "seq_len {} exceeds max_seq_len {}",
                new_len, cache.max_len
            )));
        }
        cache.current_len = new_len;
        Ok(())
    }

    /// Get the K cache tensor for a given layer and sequence.
    pub fn k_cache(&self, handle: CacheHandle, layer: usize) -> Result<&DeviceTensor> {
        let cache = self
            .caches
            .get(&handle.0)
            .ok_or_else(|| FractureError::KvCache(format!("invalid handle: {}", handle.0)))?;
        Ok(&cache.layers[layer].k)
    }

    /// Get the V cache tensor for a given layer and sequence.
    pub fn v_cache(&self, handle: CacheHandle, layer: usize) -> Result<&DeviceTensor> {
        let cache = self
            .caches
            .get(&handle.0)
            .ok_or_else(|| FractureError::KvCache(format!("invalid handle: {}", handle.0)))?;
        Ok(&cache.layers[layer].v)
    }

    /// Free all GPU memory for a completed sequence.
    pub fn free<B: Backend>(&mut self, handle: CacheHandle, backend: &B) -> Result<()> {
        let cache = self
            .caches
            .remove(&handle.0)
            .ok_or_else(|| FractureError::KvCache(format!("invalid handle: {}", handle.0)))?;

        for layer in &cache.layers {
            backend.free(&layer.k)?;
            backend.free(&layer.v)?;
        }
        Ok(())
    }
}
