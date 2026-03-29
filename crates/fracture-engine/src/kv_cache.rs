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
        if layer >= cache.layers.len() {
            return Err(FractureError::KvCache(format!(
                "layer index {} out of bounds (num_layers={})",
                layer, cache.layers.len()
            )));
        }
        Ok(&cache.layers[layer].k)
    }

    /// Get the V cache tensor for a given layer and sequence.
    pub fn v_cache(&self, handle: CacheHandle, layer: usize) -> Result<&DeviceTensor> {
        let cache = self
            .caches
            .get(&handle.0)
            .ok_or_else(|| FractureError::KvCache(format!("invalid handle: {}", handle.0)))?;
        if layer >= cache.layers.len() {
            return Err(FractureError::KvCache(format!(
                "layer index {} out of bounds (num_layers={})",
                layer, cache.layers.len()
            )));
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use fracture_core::{DeviceTimer, TensorId};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// MockBackend for KV cache tests. Supports alloc (with incrementing IDs) and free (no-op).
    struct MockBackend {
        next_id: AtomicU64,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                next_id: AtomicU64::new(1),
            }
        }
    }

    impl Backend for MockBackend {
        fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
        }

        fn free(&self, _tensor: &DeviceTensor) -> Result<()> {
            Ok(())
        }

        fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> {
            unimplemented!()
        }

        fn copy_to_host(&self, _src: &DeviceTensor, _dst: &mut [u8]) -> Result<()> {
            unimplemented!()
        }

        fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> {
            unimplemented!()
        }

        fn rmsnorm(&self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64, _out: &DeviceTensor) -> Result<()> {
            unimplemented!()
        }

        fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _positions: &[u32], _theta: f64, _head_dim: usize) -> Result<()> {
            unimplemented!()
        }

        fn attention(&self, _q: &DeviceTensor, _k_cache: &DeviceTensor, _v_cache: &DeviceTensor, _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor) -> Result<()> {
            unimplemented!()
        }

        fn silu_mul(&self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor) -> Result<()> {
            unimplemented!()
        }

        fn embedding(&self, _token_ids: &[u32], _embedding_table: &DeviceTensor, _out: &DeviceTensor) -> Result<()> {
            unimplemented!()
        }

        fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> {
            unimplemented!()
        }

        fn copy_rows(&self, _src: &DeviceTensor, _dst: &DeviceTensor, _src_offset: usize, _dst_offset: usize, _count: usize) -> Result<()> {
            unimplemented!()
        }

        fn device_name(&self) -> &str {
            "mock"
        }

        fn total_memory(&self) -> usize {
            1 << 30
        }

        fn available_memory(&self) -> usize {
            1 << 30
        }

        fn synchronize(&self) -> Result<()> {
            Ok(())
        }

        fn create_timer(&self) -> Result<DeviceTimer> {
            Ok(DeviceTimer(0))
        }

        fn start_timer(&self, _timer: &DeviceTimer) -> Result<()> {
            Ok(())
        }

        fn stop_timer(&self, _timer: &DeviceTimer) -> Result<f32> {
            Ok(0.0)
        }

        fn destroy_timer(&self, _timer: &DeviceTimer) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_kv_cache_invalid_handle() {
        let mgr = KvCacheManager::new(4, 2, 16, 512);
        let bad = CacheHandle(999);

        // seq_len
        let err = mgr.seq_len(bad).unwrap_err();
        assert!(matches!(err, FractureError::KvCache(_)));
        assert!(err.to_string().contains("999"), "error should contain handle ID: {err}");

        // k_cache
        let err = mgr.k_cache(bad, 0).unwrap_err();
        assert!(matches!(err, FractureError::KvCache(_)));
        assert!(err.to_string().contains("999"));

        // v_cache
        let err = mgr.v_cache(bad, 0).unwrap_err();
        assert!(matches!(err, FractureError::KvCache(_)));
        assert!(err.to_string().contains("999"));
    }

    #[test]
    fn test_kv_cache_invalid_handle_set_seq_len_and_free() {
        let mut mgr = KvCacheManager::new(4, 2, 16, 512);
        let backend = MockBackend::new();
        let bad = CacheHandle(999);

        // set_seq_len
        let err = mgr.set_seq_len(bad, 5).unwrap_err();
        assert!(matches!(err, FractureError::KvCache(_)));
        assert!(err.to_string().contains("999"));

        // free
        let err = mgr.free(bad, &backend).unwrap_err();
        assert!(matches!(err, FractureError::KvCache(_)));
        assert!(err.to_string().contains("999"));
    }

    #[test]
    fn test_kv_cache_alloc_creates_buffers() {
        let backend = MockBackend::new();
        let num_layers = 2;
        let num_kv_heads = 4;
        let head_dim = 32;
        let max_seq_len = 256;
        let mut mgr = KvCacheManager::new(num_layers, num_kv_heads, head_dim, max_seq_len);

        let handle = mgr.alloc(&backend).unwrap();

        // Verify k_cache and v_cache return tensors with correct shapes for each layer
        for layer in 0..num_layers {
            let k = mgr.k_cache(handle, layer).unwrap();
            assert_eq!(
                k.shape,
                vec![max_seq_len, num_kv_heads, head_dim],
                "k_cache shape mismatch at layer {layer}"
            );
            assert_eq!(k.dtype, DType::FP16);

            let v = mgr.v_cache(handle, layer).unwrap();
            assert_eq!(
                v.shape,
                vec![max_seq_len, num_kv_heads, head_dim],
                "v_cache shape mismatch at layer {layer}"
            );
            assert_eq!(v.dtype, DType::FP16);
        }
    }

    #[test]
    fn test_kv_cache_seq_len_lifecycle() {
        let backend = MockBackend::new();
        let mut mgr = KvCacheManager::new(2, 2, 16, 512);

        let handle = mgr.alloc(&backend).unwrap();

        // Initially 0
        assert_eq!(mgr.seq_len(handle).unwrap(), 0);

        // Set to 5
        mgr.set_seq_len(handle, 5).unwrap();
        assert_eq!(mgr.seq_len(handle).unwrap(), 5);

        // Set to 10
        mgr.set_seq_len(handle, 10).unwrap();
        assert_eq!(mgr.seq_len(handle).unwrap(), 10);
    }

    #[test]
    fn test_kv_cache_bounds_check() {
        let backend = MockBackend::new();
        let max_seq_len = 128;
        let mut mgr = KvCacheManager::new(2, 2, 16, max_seq_len);

        let handle = mgr.alloc(&backend).unwrap();

        // Setting seq_len beyond max_seq_len should fail
        let err = mgr.set_seq_len(handle, max_seq_len + 1).unwrap_err();
        assert!(
            matches!(err, FractureError::KvCache(_)),
            "expected KvCache error, got: {err}"
        );
        assert!(
            err.to_string().contains("exceeds max_seq_len"),
            "expected mention of exceeds max_seq_len: {err}"
        );

        // Exactly max_seq_len should succeed
        mgr.set_seq_len(handle, max_seq_len).unwrap();
        assert_eq!(mgr.seq_len(handle).unwrap(), max_seq_len);
    }

    #[test]
    fn test_kv_cache_free_releases() {
        let backend = MockBackend::new();
        let mut mgr = KvCacheManager::new(2, 2, 16, 512);

        let handle = mgr.alloc(&backend).unwrap();
        mgr.set_seq_len(handle, 5).unwrap();

        // Free the cache
        mgr.free(handle, &backend).unwrap();

        // All operations on freed handle should return errors
        assert!(mgr.seq_len(handle).is_err());
        assert!(mgr.k_cache(handle, 0).is_err());
        assert!(mgr.v_cache(handle, 0).is_err());
        assert!(mgr.set_seq_len(handle, 1).is_err());
        assert!(mgr.free(handle, &backend).is_err());
    }

    #[test]
    fn test_kv_cache_layer_access() {
        let backend = MockBackend::new();
        let num_layers = 4;
        let mut mgr = KvCacheManager::new(num_layers, 2, 16, 256);

        let handle = mgr.alloc(&backend).unwrap();

        // Verify k_cache and v_cache for all layers return valid tensors
        for layer in 0..num_layers {
            let k = mgr.k_cache(handle, layer).unwrap();
            assert_eq!(k.shape, vec![256, 2, 16]);

            let v = mgr.v_cache(handle, layer).unwrap();
            assert_eq!(v.shape, vec![256, 2, 16]);

            // Each tensor should have a unique ID
            assert_ne!(k.id, v.id, "k and v at layer {layer} should have different tensor IDs");
        }
    }

    #[test]
    fn test_kv_cache_error_contains_handle_id() {
        let cache = KvCacheManager::new(2, 2, 16, 128);
        let invalid_handle = CacheHandle(42);

        // seq_len should fail with an error containing "42"
        let err = cache.seq_len(invalid_handle).unwrap_err();
        let display = err.to_string();
        assert!(
            display.contains("42"),
            "KvCache error should contain handle id '42' in: {display}"
        );

        // k_cache should also fail with handle id in message
        let err = cache.k_cache(invalid_handle, 0).unwrap_err();
        let display = err.to_string();
        assert!(
            display.contains("42"),
            "KvCache k_cache error should contain handle id '42' in: {display}"
        );

        // v_cache should also fail with handle id in message
        let err = cache.v_cache(invalid_handle, 0).unwrap_err();
        let display = err.to_string();
        assert!(
            display.contains("42"),
            "KvCache v_cache error should contain handle id '42' in: {display}"
        );
    }

    #[test]
    fn test_kv_cache_initial_seq_len_zero() {
        let backend = MockBackend::new();
        let mut mgr = KvCacheManager::new(2, 2, 16, 512);

        let handle = mgr.alloc(&backend).unwrap();

        // Immediately after alloc, seq_len should be 0
        assert_eq!(
            mgr.seq_len(handle).unwrap(),
            0,
            "seq_len should be 0 immediately after alloc"
        );
    }

    /// FailingMockBackend: alloc fails after `fail_at` calls.
    struct FailingMockBackend {
        next_id: AtomicU64,
        alloc_count: AtomicU64,
        fail_at: u64,
    }

    impl FailingMockBackend {
        fn new(fail_at: u64) -> Self {
            Self {
                next_id: AtomicU64::new(1),
                alloc_count: AtomicU64::new(0),
                fail_at,
            }
        }
    }

    impl Backend for FailingMockBackend {
        fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
            let n = self.alloc_count.fetch_add(1, Ordering::SeqCst);
            if n >= self.fail_at {
                return Err(FractureError::Backend("OOM".into()));
            }
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
        }
        fn free(&self, _tensor: &DeviceTensor) -> Result<()> { Ok(()) }
        fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> { unimplemented!() }
        fn copy_to_host(&self, _src: &DeviceTensor, _dst: &mut [u8]) -> Result<()> { unimplemented!() }
        fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { unimplemented!() }
        fn rmsnorm(&self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64, _out: &DeviceTensor) -> Result<()> { unimplemented!() }
        fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _positions: &[u32], _theta: f64, _head_dim: usize) -> Result<()> { unimplemented!() }
        fn attention(&self, _q: &DeviceTensor, _k_cache: &DeviceTensor, _v_cache: &DeviceTensor, _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor) -> Result<()> { unimplemented!() }
        fn silu_mul(&self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { unimplemented!() }
        fn embedding(&self, _token_ids: &[u32], _embedding_table: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { unimplemented!() }
        fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { unimplemented!() }
        fn copy_rows(&self, _src: &DeviceTensor, _dst: &DeviceTensor, _src_offset: usize, _dst_offset: usize, _count: usize) -> Result<()> { unimplemented!() }
        fn device_name(&self) -> &str { "failing-mock" }
        fn total_memory(&self) -> usize { 1 << 30 }
        fn available_memory(&self) -> usize { 1 << 30 }
        fn synchronize(&self) -> Result<()> { Ok(()) }
        fn create_timer(&self) -> Result<DeviceTimer> { Ok(DeviceTimer(0)) }
        fn start_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }
        fn stop_timer(&self, _timer: &DeviceTimer) -> Result<f32> { Ok(0.0) }
        fn destroy_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }
    }

    #[test]
    fn test_kv_cache_alloc_oom() {
        // 2 layers × 2 tensors (k+v) = 4 allocs needed. Fail on the 3rd.
        let backend = FailingMockBackend::new(3);
        let mut mgr = KvCacheManager::new(2, 2, 16, 256);

        let result = mgr.alloc(&backend);
        assert!(result.is_err(), "alloc should fail when backend OOMs");
        let err = result.unwrap_err();
        assert!(
            matches!(err, FractureError::Backend(_)),
            "expected Backend error, got: {err:?}"
        );
    }

    #[test]
    fn test_kv_cache_layer_bounds_check() {
        let backend = MockBackend::new();
        let num_layers = 4;
        let mut mgr = KvCacheManager::new(num_layers, 2, 16, 256);

        let handle = mgr.alloc(&backend).unwrap();

        // Accessing layer at num_layers should fail
        let err = mgr.k_cache(handle, num_layers).unwrap_err();
        assert!(matches!(err, FractureError::KvCache(_)));
        assert!(
            err.to_string().contains("out of bounds"),
            "expected 'out of bounds' in: {err}"
        );

        let err = mgr.v_cache(handle, num_layers).unwrap_err();
        assert!(matches!(err, FractureError::KvCache(_)));
        assert!(
            err.to_string().contains("out of bounds"),
            "expected 'out of bounds' in: {err}"
        );

        // Accessing layer well beyond num_layers should also fail
        let err = mgr.k_cache(handle, 100).unwrap_err();
        assert!(matches!(err, FractureError::KvCache(_)));

        let err = mgr.v_cache(handle, 100).unwrap_err();
        assert!(matches!(err, FractureError::KvCache(_)));

        // Valid layer indices should still work
        for layer in 0..num_layers {
            assert!(mgr.k_cache(handle, layer).is_ok());
            assert!(mgr.v_cache(handle, layer).is_ok());
        }
    }

    #[test]
    fn test_kv_cache_prefill_plus_decode_seq_len_tracking() {
        let backend = MockBackend::new();
        let mut mgr = KvCacheManager::new(2, 2, 16, 512);

        let handle = mgr.alloc(&backend).unwrap();

        // Initially 0
        assert_eq!(mgr.seq_len(handle).unwrap(), 0);

        // Simulate prefill: append N=10 tokens
        let n = 10;
        mgr.set_seq_len(handle, n).unwrap();
        assert_eq!(mgr.seq_len(handle).unwrap(), n);

        // Simulate K=5 decode steps, each appending 1 token
        let k = 5;
        for step in 1..=k {
            mgr.set_seq_len(handle, n + step).unwrap();
            assert_eq!(
                mgr.seq_len(handle).unwrap(),
                n + step,
                "after prefill({n}) + {step} decode steps"
            );
        }

        // Final seq_len should be N+K
        assert_eq!(
            mgr.seq_len(handle).unwrap(),
            n + k,
            "final seq_len should be N+K = {}",
            n + k
        );
    }

    #[test]
    fn test_kv_cache_multi_sequence_independence() {
        let backend = MockBackend::new();
        let mut mgr = KvCacheManager::new(2, 2, 16, 512);

        let a = mgr.alloc(&backend).unwrap();
        let b = mgr.alloc(&backend).unwrap();
        assert_ne!(a, b, "handles should be distinct");

        // Set different seq_lens
        mgr.set_seq_len(a, 10).unwrap();
        mgr.set_seq_len(b, 20).unwrap();
        assert_eq!(mgr.seq_len(a).unwrap(), 10);
        assert_eq!(mgr.seq_len(b).unwrap(), 20);

        // Tensor IDs should be distinct between sequences
        let ka = mgr.k_cache(a, 0).unwrap().id;
        let kb = mgr.k_cache(b, 0).unwrap().id;
        assert_ne!(ka, kb, "different sequences should have different tensor IDs");

        // Free A, verify B is unaffected
        mgr.free(a, &backend).unwrap();
        assert!(mgr.seq_len(a).is_err(), "A should be freed");
        assert_eq!(mgr.seq_len(b).unwrap(), 20, "B should be unaffected by freeing A");
        assert!(mgr.k_cache(b, 0).is_ok(), "B's k_cache should still be accessible");

        mgr.free(b, &backend).unwrap();
    }
}
