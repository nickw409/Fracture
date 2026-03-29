use fracture_core::{Backend, DType, DeviceTensor, FractureError, Result};
use std::collections::HashMap;

use crate::kv_cache::CacheHandle;

/// Fixed number of tokens per block. Hardcoded — the attention kernel is tuned for this.
pub const BLOCK_SIZE: usize = 16;

/// Pre-allocated pool of KV cache blocks on GPU memory.
///
/// Each block stores K and V data for `BLOCK_SIZE` tokens for one layer.
/// Blocks are allocated at pool creation and returned to a free list when released.
pub struct BlockPool {
    /// K tensors: k_blocks[block_id][layer_idx] = [BLOCK_SIZE, num_kv_heads, head_dim] FP16
    k_blocks: Vec<Vec<DeviceTensor>>,
    /// V tensors: v_blocks[block_id][layer_idx] = [BLOCK_SIZE, num_kv_heads, head_dim] FP16
    v_blocks: Vec<Vec<DeviceTensor>>,
    /// Stack of free block IDs. Pop to allocate, push to free.
    free_list: Vec<usize>,
    /// Total blocks in the pool.
    capacity: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl BlockPool {
    /// Create a new block pool, pre-allocating all GPU memory.
    ///
    /// `num_blocks`: how many blocks to create. Calculated by the caller as:
    ///   (available_gpu_bytes) / (bytes_per_block_all_layers)
    pub fn new<B: Backend>(
        num_blocks: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        backend: &B,
    ) -> Result<Self> {
        let mut k_blocks = Vec::with_capacity(num_blocks);
        let mut v_blocks = Vec::with_capacity(num_blocks);

        for block_id in 0..num_blocks {
            let mut k_layers = Vec::with_capacity(num_layers);
            let mut v_layers = Vec::with_capacity(num_layers);

            for _ in 0..num_layers {
                let k = backend.alloc(&[BLOCK_SIZE, num_kv_heads, head_dim], DType::FP16)
                    .map_err(|e| FractureError::KvCache(format!(
                        "block pool alloc failed at block {block_id}: {e}"
                    )))?;
                let v = backend.alloc(&[BLOCK_SIZE, num_kv_heads, head_dim], DType::FP16)
                    .map_err(|e| FractureError::KvCache(format!(
                        "block pool alloc failed at block {block_id}: {e}"
                    )))?;
                k_layers.push(k);
                v_layers.push(v);
            }

            k_blocks.push(k_layers);
            v_blocks.push(v_layers);
        }

        // All blocks start free.
        let free_list: Vec<usize> = (0..num_blocks).rev().collect();

        Ok(Self {
            k_blocks,
            v_blocks,
            free_list,
            capacity: num_blocks,
            num_layers,
            num_kv_heads,
            head_dim,
        })
    }

    /// Allocate a block from the free list. Returns the block ID.
    fn alloc_block(&mut self) -> Result<usize> {
        self.free_list.pop().ok_or_else(|| {
            FractureError::OutOfMemory {
                requested: self.bytes_per_block(),
                available: 0,
            }
        })
    }

    /// Return a block to the free list.
    fn free_block(&mut self, block_id: usize) {
        debug_assert!(block_id < self.capacity, "block_id out of range");
        self.free_list.push(block_id);
    }

    /// Get the K tensor for a specific block and layer.
    pub fn k_tensor(&self, block_id: usize, layer: usize) -> &DeviceTensor {
        &self.k_blocks[block_id][layer]
    }

    /// Get the V tensor for a specific block and layer.
    pub fn v_tensor(&self, block_id: usize, layer: usize) -> &DeviceTensor {
        &self.v_blocks[block_id][layer]
    }

    /// Number of free blocks available.
    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    /// Total blocks in the pool.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes per block across all layers.
    pub fn bytes_per_block(&self) -> usize {
        // K + V, each [BLOCK_SIZE, num_kv_heads, head_dim] FP16, per layer
        BLOCK_SIZE * self.num_kv_heads * self.head_dim * 2 * 2 * self.num_layers
    }

    /// Free all GPU memory. Called on shutdown.
    pub fn destroy<B: Backend>(&self, backend: &B) -> Result<()> {
        for block in &self.k_blocks {
            for tensor in block {
                backend.free(tensor)?;
            }
        }
        for block in &self.v_blocks {
            for tensor in block {
                backend.free(tensor)?;
            }
        }
        Ok(())
    }
}

/// Per-sequence block allocation state.
struct SequenceBlocks {
    /// Logical block index → physical block_id.
    /// block_table[i] holds tokens [i*BLOCK_SIZE .. (i+1)*BLOCK_SIZE).
    block_table: Vec<usize>,
    /// Number of valid tokens in the last block (1..=BLOCK_SIZE).
    /// 0 means no blocks allocated yet (empty sequence).
    last_block_fill: usize,
    /// Total tokens stored across all blocks.
    current_len: usize,
}

/// Paged KV cache manager. Manages a block pool and per-sequence block tables.
///
/// The engine calls `append_kv` to write new KV data (which auto-grows blocks)
/// and reads via block tables passed to the paged attention kernel.
pub struct PagedKvCacheManager {
    pool: BlockPool,
    sequences: HashMap<u64, SequenceBlocks>,
    next_id: u64,
}

impl PagedKvCacheManager {
    /// Create a paged cache manager with a pre-allocated block pool.
    pub fn new<B: Backend>(
        num_blocks: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        backend: &B,
    ) -> Result<Self> {
        let pool = BlockPool::new(num_blocks, num_layers, num_kv_heads, head_dim, backend)?;
        Ok(Self {
            pool,
            sequences: HashMap::new(),
            next_id: 0,
        })
    }

    /// Allocate a new sequence. Allocates one initial block.
    pub fn alloc(&mut self) -> Result<CacheHandle> {
        let id = self.next_id;
        self.next_id += 1;

        let first_block = self.pool.alloc_block()?;

        self.sequences.insert(id, SequenceBlocks {
            block_table: vec![first_block],
            last_block_fill: 0,
            current_len: 0,
        });

        Ok(CacheHandle(id))
    }

    /// Append KV data for new tokens into the block pool.
    ///
    /// `num_new_tokens`: how many tokens to append (e.g., prompt length for prefill, 1 for decode).
    /// The caller has already computed K and V projections. This method writes them into
    /// the correct block positions using `backend.copy_rows()`.
    ///
    /// Automatically allocates new blocks when the current block is full.
    pub fn append_kv<B: Backend>(
        &mut self,
        handle: CacheHandle,
        layer: usize,
        keys: &DeviceTensor,   // [num_new_tokens, num_kv_heads, head_dim]
        values: &DeviceTensor, // same
        backend: &B,
    ) -> Result<()> {
        let seq = self.sequences.get_mut(&handle.0).ok_or_else(|| {
            FractureError::KvCache(format!("invalid handle: {}", handle.0))
        })?;

        let num_new = keys.shape[0];

        // On layer 0: allocate any new blocks needed and update metadata.
        // On layer > 0: blocks are already allocated, just write data.
        if layer == 0 {
            let mut remaining = num_new;
            let mut fill = seq.last_block_fill;

            while remaining > 0 {
                let slots = BLOCK_SIZE - fill;
                if slots == 0 {
                    let new_block = self.pool.alloc_block()?;
                    seq.block_table.push(new_block);
                    fill = 0;
                    continue;
                }
                let to_write = remaining.min(slots);
                fill += to_write;
                remaining -= to_write;
            }

            seq.last_block_fill = fill;
            seq.current_len += num_new;
        }

        // Write KV data into blocks for this layer.
        // Walk the block table starting from where the old data ended.
        let start_token = seq.current_len - num_new;
        let mut written = 0;

        while written < num_new {
            let global_pos = start_token + written;
            let block_idx = global_pos / BLOCK_SIZE;
            let offset_in_block = global_pos % BLOCK_SIZE;
            let block_id = seq.block_table[block_idx];

            let slots = BLOCK_SIZE - offset_in_block;
            let to_write = (num_new - written).min(slots);

            let k_dst = self.pool.k_tensor(block_id, layer);
            backend.copy_rows(keys, k_dst, written, offset_in_block, to_write)?;

            let v_dst = self.pool.v_tensor(block_id, layer);
            backend.copy_rows(values, v_dst, written, offset_in_block, to_write)?;

            written += to_write;
        }

        Ok(())
    }

    /// Get the block table for a sequence (physical block IDs).
    pub fn block_table(&self, handle: CacheHandle) -> Result<&[usize]> {
        let seq = self.sequences.get(&handle.0).ok_or_else(|| {
            FractureError::KvCache(format!("invalid handle: {}", handle.0))
        })?;
        Ok(&seq.block_table)
    }

    /// Total tokens stored for a sequence.
    pub fn seq_len(&self, handle: CacheHandle) -> Result<usize> {
        let seq = self.sequences.get(&handle.0).ok_or_else(|| {
            FractureError::KvCache(format!("invalid handle: {}", handle.0))
        })?;
        Ok(seq.current_len)
    }

    /// Number of valid tokens in the last block (needed by the attention kernel).
    pub fn last_block_tokens(&self, handle: CacheHandle) -> Result<usize> {
        let seq = self.sequences.get(&handle.0).ok_or_else(|| {
            FractureError::KvCache(format!("invalid handle: {}", handle.0))
        })?;
        Ok(seq.last_block_fill)
    }

    /// Free all blocks for a sequence, returning them to the pool.
    pub fn free(&mut self, handle: CacheHandle) -> Result<()> {
        let seq = self.sequences.remove(&handle.0).ok_or_else(|| {
            FractureError::KvCache(format!("invalid handle: {}", handle.0))
        })?;
        for block_id in &seq.block_table {
            self.pool.free_block(*block_id);
        }
        Ok(())
    }

    /// Number of free blocks available in the pool.
    pub fn num_free_blocks(&self) -> usize {
        self.pool.num_free()
    }

    /// Estimated number of additional tokens that can be stored.
    pub fn available_token_capacity(&self) -> usize {
        self.pool.num_free() * BLOCK_SIZE
    }

    /// Access the block pool (for the attention kernel to get tensor pointers).
    pub fn pool(&self) -> &BlockPool {
        &self.pool
    }

    /// Destroy the pool, freeing all GPU memory.
    pub fn destroy<B: Backend>(&self, backend: &B) -> Result<()> {
        self.pool.destroy(backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fracture_core::{DeviceTimer, TensorId};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    /// MockBackend that tracks alloc/free and supports copy_rows.
    struct MockBackend {
        next_id: AtomicU64,
        copy_rows_log: Mutex<Vec<(u64, u64, usize, usize, usize)>>, // (src_id, dst_id, src_off, dst_off, count)
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                next_id: AtomicU64::new(1),
                copy_rows_log: Mutex::new(Vec::new()),
            }
        }
    }

    impl Backend for MockBackend {
        fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
        }
        fn free(&self, _t: &DeviceTensor) -> Result<()> { Ok(()) }
        fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> { Ok(()) }
        fn copy_to_host(&self, _src: &DeviceTensor, _dst: &mut [u8]) -> Result<()> { Ok(()) }
        fn matmul(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn rmsnorm(&self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn rope(&self, _q: &DeviceTensor, _k: &DeviceTensor, _positions: &[u32], _theta: f64, _head_dim: usize) -> Result<()> { Ok(()) }
        fn attention(&self, _q: &DeviceTensor, _k_cache: &DeviceTensor, _v_cache: &DeviceTensor, _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn silu_mul(&self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn embedding(&self, _token_ids: &[u32], _embedding_table: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn add(&self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor) -> Result<()> { Ok(()) }
        fn copy_rows(&self, src: &DeviceTensor, dst: &DeviceTensor, src_offset: usize, dst_offset: usize, count: usize) -> Result<()> {
            self.copy_rows_log.lock().unwrap().push((src.id.0, dst.id.0, src_offset, dst_offset, count));
            Ok(())
        }
        fn device_name(&self) -> &str { "mock" }
        fn total_memory(&self) -> usize { 1 << 30 }
        fn available_memory(&self) -> usize { 1 << 30 }
        fn synchronize(&self) -> Result<()> { Ok(()) }
        fn create_timer(&self) -> Result<DeviceTimer> { Ok(DeviceTimer(0)) }
        fn start_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }
        fn stop_timer(&self, _timer: &DeviceTimer) -> Result<f32> { Ok(0.0) }
        fn destroy_timer(&self, _timer: &DeviceTimer) -> Result<()> { Ok(()) }
    }

    #[test]
    fn test_block_pool_creation() {
        let backend = MockBackend::new();
        let pool = BlockPool::new(10, 2, 8, 128, &backend).unwrap();
        assert_eq!(pool.capacity(), 10);
        assert_eq!(pool.num_free(), 10);
    }

    #[test]
    fn test_block_pool_alloc_and_free() {
        let backend = MockBackend::new();
        let mut pool = BlockPool::new(3, 2, 8, 128, &backend).unwrap();

        let b0 = pool.alloc_block().unwrap();
        let b1 = pool.alloc_block().unwrap();
        let b2 = pool.alloc_block().unwrap();
        assert_eq!(pool.num_free(), 0);

        // Pool is exhausted.
        assert!(pool.alloc_block().is_err());

        // Free one and re-alloc.
        pool.free_block(b1);
        assert_eq!(pool.num_free(), 1);
        let b3 = pool.alloc_block().unwrap();
        assert_eq!(b3, b1); // reuses the freed block
        assert_eq!(pool.num_free(), 0);

        pool.free_block(b0);
        pool.free_block(b2);
        pool.free_block(b3);
        assert_eq!(pool.num_free(), 3);
    }

    #[test]
    fn test_block_pool_tensor_shapes() {
        let backend = MockBackend::new();
        let pool = BlockPool::new(2, 4, 8, 128, &backend).unwrap();

        for block_id in 0..2 {
            for layer in 0..4 {
                let k = pool.k_tensor(block_id, layer);
                assert_eq!(k.shape, vec![BLOCK_SIZE, 8, 128]);
                assert_eq!(k.dtype, DType::FP16);

                let v = pool.v_tensor(block_id, layer);
                assert_eq!(v.shape, vec![BLOCK_SIZE, 8, 128]);
                assert_eq!(v.dtype, DType::FP16);
            }
        }
    }

    #[test]
    fn test_paged_cache_alloc_and_free() {
        let backend = MockBackend::new();
        let mut cache = PagedKvCacheManager::new(10, 2, 8, 128, &backend).unwrap();

        let h = cache.alloc().unwrap();
        assert_eq!(cache.seq_len(h).unwrap(), 0);
        assert_eq!(cache.last_block_tokens(h).unwrap(), 0);
        assert_eq!(cache.block_table(h).unwrap().len(), 1); // one initial block

        // Pool should have used 1 block.
        assert_eq!(cache.num_free_blocks(), 9);

        cache.free(h).unwrap();
        assert_eq!(cache.num_free_blocks(), 10);

        // Operations on freed handle fail.
        assert!(cache.seq_len(h).is_err());
    }

    #[test]
    fn test_paged_cache_append_single_token() {
        let backend = MockBackend::new();
        let num_layers = 2;
        let mut cache = PagedKvCacheManager::new(10, num_layers, 8, 128, &backend).unwrap();

        let h = cache.alloc().unwrap();

        // Simulate a decode step: append 1 token per layer.
        let k = DeviceTensor::new(TensorId(9000), vec![1, 8, 128], DType::FP16);
        let v = DeviceTensor::new(TensorId(9001), vec![1, 8, 128], DType::FP16);
        for layer in 0..num_layers {
            cache.append_kv(h, layer, &k, &v, &backend).unwrap();
        }

        assert_eq!(cache.seq_len(h).unwrap(), 1);
        assert_eq!(cache.last_block_tokens(h).unwrap(), 1);
        assert_eq!(cache.block_table(h).unwrap().len(), 1); // still fits in first block
    }

    #[test]
    fn test_paged_cache_append_fills_block_then_grows() {
        let backend = MockBackend::new();
        let num_layers = 1;
        let mut cache = PagedKvCacheManager::new(10, num_layers, 8, 128, &backend).unwrap();

        let h = cache.alloc().unwrap();

        // Append BLOCK_SIZE tokens one at a time — should stay in one block.
        let k = DeviceTensor::new(TensorId(9000), vec![1, 8, 128], DType::FP16);
        let v = DeviceTensor::new(TensorId(9001), vec![1, 8, 128], DType::FP16);
        for i in 0..BLOCK_SIZE {
            cache.append_kv(h, 0, &k, &v, &backend).unwrap();
            assert_eq!(cache.seq_len(h).unwrap(), i + 1);
        }
        assert_eq!(cache.block_table(h).unwrap().len(), 1);
        assert_eq!(cache.last_block_tokens(h).unwrap(), BLOCK_SIZE);

        // Append one more — should allocate a second block.
        cache.append_kv(h, 0, &k, &v, &backend).unwrap();
        assert_eq!(cache.seq_len(h).unwrap(), BLOCK_SIZE + 1);
        assert_eq!(cache.block_table(h).unwrap().len(), 2);
        assert_eq!(cache.last_block_tokens(h).unwrap(), 1);
    }

    #[test]
    fn test_paged_cache_prefill_bulk() {
        let backend = MockBackend::new();
        let num_layers = 1;
        // Need enough blocks: 50 tokens = ceil(50/16) = 4 blocks, plus 1 initial = 4 total.
        let mut cache = PagedKvCacheManager::new(10, num_layers, 8, 128, &backend).unwrap();

        let h = cache.alloc().unwrap();

        // Prefill 50 tokens at once.
        let k = DeviceTensor::new(TensorId(9000), vec![50, 8, 128], DType::FP16);
        let v = DeviceTensor::new(TensorId(9001), vec![50, 8, 128], DType::FP16);
        cache.append_kv(h, 0, &k, &v, &backend).unwrap();

        assert_eq!(cache.seq_len(h).unwrap(), 50);
        // 50 tokens: blocks [0-15], [16-31], [32-47], [48-49] = 4 blocks
        // But initial alloc gave us 1, so we needed 3 more = 4 total.
        assert_eq!(cache.block_table(h).unwrap().len(), 4);
        assert_eq!(cache.last_block_tokens(h).unwrap(), 2); // 50 - 48 = 2 in last block

        // Free and verify blocks returned.
        let free_before = cache.num_free_blocks();
        cache.free(h).unwrap();
        assert_eq!(cache.num_free_blocks(), free_before + 4);
    }

    #[test]
    fn test_paged_cache_oom() {
        let backend = MockBackend::new();
        // Only 2 blocks total.
        let mut cache = PagedKvCacheManager::new(2, 1, 8, 128, &backend).unwrap();

        let h = cache.alloc().unwrap(); // uses 1 block

        // Fill the first block.
        let k = DeviceTensor::new(TensorId(9000), vec![BLOCK_SIZE, 8, 128], DType::FP16);
        let v = DeviceTensor::new(TensorId(9001), vec![BLOCK_SIZE, 8, 128], DType::FP16);
        cache.append_kv(h, 0, &k, &v, &backend).unwrap();

        // Allocate a second sequence — uses the last block.
        let h2 = cache.alloc().unwrap();
        assert_eq!(cache.num_free_blocks(), 0);

        // Now appending more to h should fail (no blocks left).
        let k1 = DeviceTensor::new(TensorId(9002), vec![1, 8, 128], DType::FP16);
        let v1 = DeviceTensor::new(TensorId(9003), vec![1, 8, 128], DType::FP16);
        let result = cache.append_kv(h, 0, &k1, &v1, &backend);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), FractureError::OutOfMemory { .. }));

        cache.free(h).unwrap();
        cache.free(h2).unwrap();
    }

    #[test]
    fn test_paged_cache_multi_sequence_isolation() {
        let backend = MockBackend::new();
        let mut cache = PagedKvCacheManager::new(20, 1, 8, 128, &backend).unwrap();

        let a = cache.alloc().unwrap();
        let b = cache.alloc().unwrap();

        // Append different amounts.
        let k5 = DeviceTensor::new(TensorId(9000), vec![5, 8, 128], DType::FP16);
        let v5 = DeviceTensor::new(TensorId(9001), vec![5, 8, 128], DType::FP16);
        cache.append_kv(a, 0, &k5, &v5, &backend).unwrap();

        let k20 = DeviceTensor::new(TensorId(9002), vec![20, 8, 128], DType::FP16);
        let v20 = DeviceTensor::new(TensorId(9003), vec![20, 8, 128], DType::FP16);
        cache.append_kv(b, 0, &k20, &v20, &backend).unwrap();

        assert_eq!(cache.seq_len(a).unwrap(), 5);
        assert_eq!(cache.seq_len(b).unwrap(), 20);

        // Block tables should be independent.
        let a_blocks = cache.block_table(a).unwrap();
        let b_blocks = cache.block_table(b).unwrap();
        assert_eq!(a_blocks.len(), 1);  // 5 tokens fits in 1 block
        assert_eq!(b_blocks.len(), 2);  // 20 tokens = 2 blocks

        // No overlap in block IDs.
        for &ab in a_blocks {
            assert!(!b_blocks.contains(&ab), "block tables should not overlap");
        }

        // Free A, verify B unaffected.
        cache.free(a).unwrap();
        assert_eq!(cache.seq_len(b).unwrap(), 20);
        assert!(cache.seq_len(a).is_err());

        cache.free(b).unwrap();
    }

    #[test]
    fn test_paged_cache_reuse_after_free() {
        let backend = MockBackend::new();
        let mut cache = PagedKvCacheManager::new(5, 1, 8, 128, &backend).unwrap();

        let h1 = cache.alloc().unwrap();
        let k = DeviceTensor::new(TensorId(9000), vec![10, 8, 128], DType::FP16);
        let v = DeviceTensor::new(TensorId(9001), vec![10, 8, 128], DType::FP16);
        cache.append_kv(h1, 0, &k, &v, &backend).unwrap();

        let free_before = cache.num_free_blocks();
        cache.free(h1).unwrap();
        let free_after = cache.num_free_blocks();
        assert!(free_after > free_before);

        // Re-allocate — should succeed and reuse freed blocks.
        let h2 = cache.alloc().unwrap();
        cache.append_kv(h2, 0, &k, &v, &backend).unwrap();
        assert_eq!(cache.seq_len(h2).unwrap(), 10);

        cache.free(h2).unwrap();
    }

    #[test]
    fn test_paged_cache_available_token_capacity() {
        let backend = MockBackend::new();
        let cache = PagedKvCacheManager::new(100, 1, 8, 128, &backend).unwrap();
        assert_eq!(cache.available_token_capacity(), 100 * BLOCK_SIZE);
    }

    #[test]
    fn test_block_pool_bytes_per_block() {
        let backend = MockBackend::new();
        let pool = BlockPool::new(1, 32, 8, 128, &backend).unwrap();
        // K + V = 2 tensors, each BLOCK_SIZE * 8 * 128 * 2 bytes, × 32 layers
        let expected = BLOCK_SIZE * 8 * 128 * 2 * 2 * 32;
        assert_eq!(pool.bytes_per_block(), expected);
        // With BLOCK_SIZE=16: 16 * 8 * 128 * 2 * 2 * 32 = 2,097,152 = 2 MB
        assert_eq!(expected, 2 * 1024 * 1024);
    }
}
