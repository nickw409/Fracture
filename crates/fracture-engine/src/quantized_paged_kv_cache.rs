use fracture_core::{
    turboquant::{
        compute_codebook_tables, generate_rotation_matrix, rotation_seeds,
        TurboQuantConfig,
    },
    Backend, DType, DeviceTensor, FractureError, Result,
};
use std::collections::HashMap;

use crate::kv_cache::CacheHandle;
use crate::paged_kv_cache::BLOCK_SIZE;

/// A single quantized block for one layer (K or V).
///
/// Instead of `[BLOCK_SIZE, num_kv_heads, head_dim]` in FP16, this stores
/// bit-packed indices and per-head L2 norms.
struct QuantizedBlockTensors {
    /// Bit-packed quantized indices: `[BLOCK_SIZE, num_kv_heads * packed_dim_per_head]` INT8
    packed_indices: DeviceTensor,
    /// Per-head L2 norms: `[BLOCK_SIZE, num_kv_heads]` FP16
    norms: DeviceTensor,
}

/// Pre-allocated pool of quantized KV cache blocks on GPU memory.
///
/// Each block stores compressed K and V data for `BLOCK_SIZE` tokens across
/// all layers. Blocks with different layers may have different packed sizes
/// (protected layers use more bits).
pub struct QuantizedBlockPool {
    /// Quantized K blocks: k_blocks[block_id][layer_idx]
    k_blocks: Vec<Vec<QuantizedBlockTensors>>,
    /// Quantized V blocks: v_blocks[block_id][layer_idx]
    v_blocks: Vec<Vec<QuantizedBlockTensors>>,
    /// Stack of free block IDs
    free_list: Vec<usize>,
    /// Total blocks in the pool
    capacity: usize,

    // Dimensions
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,

    /// Per-layer effective bit widths
    layer_key_bits: Vec<u8>,
    layer_value_bits: Vec<u8>,

    // Precomputed tables (on device)
    /// K rotation matrix per layer: `[head_dim, head_dim]` FP32
    k_rotation_matrices: Vec<DeviceTensor>,
    /// V rotation matrix per layer: `[head_dim, head_dim]` FP32
    v_rotation_matrices: Vec<DeviceTensor>,
    /// Centroid tables per distinct bit-width: bits → `[2^bits]` FP32
    centroid_tables: HashMap<u8, DeviceTensor>,
}

impl QuantizedBlockPool {
    /// Create a new quantized block pool, pre-allocating all GPU memory.
    ///
    /// Allocates `num_blocks` physical blocks, each containing compressed K/V
    /// storage for all layers. Also uploads rotation matrices and codebooks.
    pub fn new<B: Backend>(
        num_blocks: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        config: &TurboQuantConfig,
        backend: &B,
    ) -> Result<Self> {
        // Compute per-layer bit widths
        let layer_key_bits: Vec<u8> = (0..num_layers)
            .map(|l| config.key_bits_for_layer(l, num_layers))
            .collect();
        let layer_value_bits: Vec<u8> = (0..num_layers)
            .map(|l| config.value_bits_for_layer(l, num_layers))
            .collect();

        // Allocate blocks
        let mut k_blocks = Vec::with_capacity(num_blocks);
        let mut v_blocks = Vec::with_capacity(num_blocks);

        for block_id in 0..num_blocks {
            let mut k_layers = Vec::with_capacity(num_layers);
            let mut v_layers = Vec::with_capacity(num_layers);

            for layer in 0..num_layers {
                let kb = layer_key_bits[layer];
                let vb = layer_value_bits[layer];
                let k_packed_dim = num_kv_heads * TurboQuantConfig::packed_dim_per_head(head_dim, kb);
                let v_packed_dim = num_kv_heads * TurboQuantConfig::packed_dim_per_head(head_dim, vb);

                let k_packed = backend
                    .alloc(&[BLOCK_SIZE, k_packed_dim], DType::INT8)
                    .map_err(|e| {
                        FractureError::KvCache(format!(
                            "quantized block pool K alloc failed at block {block_id} layer {layer}: {e}"
                        ))
                    })?;
                let k_norms = backend
                    .alloc(&[BLOCK_SIZE, num_kv_heads], DType::FP16)
                    .map_err(|e| {
                        FractureError::KvCache(format!(
                            "quantized block pool K norms alloc failed at block {block_id} layer {layer}: {e}"
                        ))
                    })?;
                k_layers.push(QuantizedBlockTensors {
                    packed_indices: k_packed,
                    norms: k_norms,
                });

                let v_packed = backend
                    .alloc(&[BLOCK_SIZE, v_packed_dim], DType::INT8)
                    .map_err(|e| {
                        FractureError::KvCache(format!(
                            "quantized block pool V alloc failed at block {block_id} layer {layer}: {e}"
                        ))
                    })?;
                let v_norms = backend
                    .alloc(&[BLOCK_SIZE, num_kv_heads], DType::FP16)
                    .map_err(|e| {
                        FractureError::KvCache(format!(
                            "quantized block pool V norms alloc failed at block {block_id} layer {layer}: {e}"
                        ))
                    })?;
                v_layers.push(QuantizedBlockTensors {
                    packed_indices: v_packed,
                    norms: v_norms,
                });
            }

            k_blocks.push(k_layers);
            v_blocks.push(v_layers);
        }

        let free_list: Vec<usize> = (0..num_blocks).rev().collect();

        // Generate rotation matrices and upload to device
        let mut k_rotation_matrices = Vec::with_capacity(num_layers);
        let mut v_rotation_matrices = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let (k_seed, v_seed) = rotation_seeds(config.seed, layer);

            let k_rot_host = generate_rotation_matrix(head_dim, k_seed);
            let k_rot_tensor = backend.alloc(&[head_dim, head_dim], DType::FP32)?;
            let k_rot_bytes: Vec<u8> = k_rot_host.iter().flat_map(|f| f.to_le_bytes()).collect();
            backend.copy_to_device(&k_rot_tensor, &k_rot_bytes)?;
            k_rotation_matrices.push(k_rot_tensor);

            let v_rot_host = generate_rotation_matrix(head_dim, v_seed);
            let v_rot_tensor = backend.alloc(&[head_dim, head_dim], DType::FP32)?;
            let v_rot_bytes: Vec<u8> = v_rot_host.iter().flat_map(|f| f.to_le_bytes()).collect();
            backend.copy_to_device(&v_rot_tensor, &v_rot_bytes)?;
            v_rotation_matrices.push(v_rot_tensor);
        }

        // Compute and upload codebook tables
        let codebooks = compute_codebook_tables(config, head_dim);
        let mut centroid_tables = HashMap::new();

        for (bits, codebook) in &codebooks {
            let n_levels = codebook.n_levels();
            let tensor = backend.alloc(&[n_levels], DType::FP32)?;
            let bytes: Vec<u8> = codebook
                .centroids
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            backend.copy_to_device(&tensor, &bytes)?;
            centroid_tables.insert(*bits, tensor);
        }

        Ok(Self {
            k_blocks,
            v_blocks,
            free_list,
            capacity: num_blocks,
            num_layers,
            num_kv_heads,
            head_dim,
            layer_key_bits,
            layer_value_bits,
            k_rotation_matrices,
            v_rotation_matrices,
            centroid_tables,
        })
    }

    /// Allocate a block from the free list. Returns the physical block ID.
    fn alloc_block(&mut self) -> Result<usize> {
        self.free_list.pop().ok_or_else(|| FractureError::OutOfMemory {
            requested: self.bytes_per_block(),
            available: 0,
        })
    }

    /// Return a block to the free list.
    fn free_block(&mut self, block_id: usize) {
        debug_assert!(block_id < self.capacity, "block_id out of range");
        self.free_list.push(block_id);
    }

    /// K packed indices tensor for a given block and layer.
    pub fn k_packed(&self, block_id: usize, layer: usize) -> &DeviceTensor {
        &self.k_blocks[block_id][layer].packed_indices
    }

    /// K norms tensor for a given block and layer.
    pub fn k_norms(&self, block_id: usize, layer: usize) -> &DeviceTensor {
        &self.k_blocks[block_id][layer].norms
    }

    /// V packed indices tensor for a given block and layer.
    pub fn v_packed(&self, block_id: usize, layer: usize) -> &DeviceTensor {
        &self.v_blocks[block_id][layer].packed_indices
    }

    /// V norms tensor for a given block and layer.
    pub fn v_norms(&self, block_id: usize, layer: usize) -> &DeviceTensor {
        &self.v_blocks[block_id][layer].norms
    }

    /// K rotation matrix for a given layer.
    pub fn k_rotation(&self, layer: usize) -> &DeviceTensor {
        &self.k_rotation_matrices[layer]
    }

    /// V rotation matrix for a given layer.
    pub fn v_rotation(&self, layer: usize) -> &DeviceTensor {
        &self.v_rotation_matrices[layer]
    }

    /// Centroid table for a given bit-width.
    pub fn centroids(&self, bits: u8) -> &DeviceTensor {
        &self.centroid_tables[&bits]
    }

    /// Effective key bit-width for a given layer.
    pub fn key_bits_for_layer(&self, layer: usize) -> u8 {
        self.layer_key_bits[layer]
    }

    /// Effective value bit-width for a given layer.
    pub fn value_bits_for_layer(&self, layer: usize) -> u8 {
        self.layer_value_bits[layer]
    }

    /// Number of free blocks.
    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    /// Total pool capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Average bytes per block across all layers.
    fn bytes_per_block(&self) -> usize {
        (0..self.num_layers)
            .map(|l| {
                TurboQuantConfig::bytes_per_block_layer(
                    self.num_kv_heads,
                    self.head_dim,
                    self.layer_key_bits[l],
                    self.layer_value_bits[l],
                )
            })
            .sum()
    }

    /// Free all GPU memory in the pool.
    pub fn destroy<B: Backend>(&self, backend: &B) -> Result<()> {
        for block in &self.k_blocks {
            for layer_data in block {
                backend.free(&layer_data.packed_indices)?;
                backend.free(&layer_data.norms)?;
            }
        }
        for block in &self.v_blocks {
            for layer_data in block {
                backend.free(&layer_data.packed_indices)?;
                backend.free(&layer_data.norms)?;
            }
        }
        for rot in &self.k_rotation_matrices {
            backend.free(rot)?;
        }
        for rot in &self.v_rotation_matrices {
            backend.free(rot)?;
        }
        for tensor in self.centroid_tables.values() {
            backend.free(tensor)?;
        }
        Ok(())
    }
}

/// Per-sequence tracking of quantized blocks.
struct QuantizedSequenceBlocks {
    /// Logical block index → physical block ID in the quantized pool.
    block_table: Vec<usize>,
    /// Fill level of the last block (1..=BLOCK_SIZE, 0 if empty after alloc).
    last_block_fill: usize,
    /// Total tokens stored in quantized blocks.
    current_len: usize,
}

/// Pre-allocated scratch tensors for compression (avoids per-call cudaMalloc).
struct CompressScratch {
    /// `[max_tokens, max_packed_dim]` INT8
    k_packed: DeviceTensor,
    /// `[max_tokens, num_kv_heads]` FP16
    k_norms: DeviceTensor,
    /// `[max_tokens, max_packed_dim]` INT8
    v_packed: DeviceTensor,
    /// `[max_tokens, num_kv_heads]` FP16
    v_norms: DeviceTensor,
}

/// Manages quantized KV cache sequences backed by a `QuantizedBlockPool`.
pub struct QuantizedKvCacheManager {
    pool: QuantizedBlockPool,
    sequences: HashMap<u64, QuantizedSequenceBlocks>,
    next_id: u64,
    #[allow(dead_code)] // Used when residual window support is added
    config: TurboQuantConfig,
    scratch: CompressScratch,
}

impl QuantizedKvCacheManager {
    /// Create a new quantized KV cache manager.
    ///
    /// `max_compress_tokens`: maximum tokens compressed in one call (e.g., max_prefill_tokens).
    pub fn new<B: Backend>(
        num_blocks: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_compress_tokens: usize,
        config: TurboQuantConfig,
        backend: &B,
    ) -> Result<Self> {
        let pool = QuantizedBlockPool::new(
            num_blocks, num_layers, num_kv_heads, head_dim, &config, backend,
        )?;

        // Allocate scratch tensors sized for the maximum packed dim across all layers
        let max_k_packed_dim = (0..num_layers)
            .map(|l| {
                let bits = config.key_bits_for_layer(l, num_layers);
                num_kv_heads * TurboQuantConfig::packed_dim_per_head(head_dim, bits)
            })
            .max()
            .unwrap_or(0);
        let max_v_packed_dim = (0..num_layers)
            .map(|l| {
                let bits = config.value_bits_for_layer(l, num_layers);
                num_kv_heads * TurboQuantConfig::packed_dim_per_head(head_dim, bits)
            })
            .max()
            .unwrap_or(0);

        let scratch = CompressScratch {
            k_packed: backend.alloc(&[max_compress_tokens, max_k_packed_dim], DType::INT8)?,
            k_norms: backend.alloc(&[max_compress_tokens, num_kv_heads], DType::FP16)?,
            v_packed: backend.alloc(&[max_compress_tokens, max_v_packed_dim], DType::INT8)?,
            v_norms: backend.alloc(&[max_compress_tokens, num_kv_heads], DType::FP16)?,
        };

        Ok(Self {
            pool,
            sequences: HashMap::new(),
            next_id: 0,
            config,
            scratch,
        })
    }

    /// Allocate a new sequence. Returns a cache handle.
    pub fn alloc(&mut self) -> Result<CacheHandle> {
        let id = self.next_id;
        self.next_id += 1;

        let first_block = self.pool.alloc_block()?;

        self.sequences.insert(
            id,
            QuantizedSequenceBlocks {
                block_table: vec![first_block],
                last_block_fill: 0,
                current_len: 0,
            },
        );

        Ok(CacheHandle(id))
    }

    /// Append compressed KV data for one layer.
    ///
    /// `keys` and `values` are `[N, num_kv_heads, head_dim]` FP16 tensors
    /// from the attention projection. They are compressed on-device via the
    /// Backend's `turboquant_compress` method and written into the block pool.
    pub fn append_kv<B: Backend>(
        &mut self,
        handle: CacheHandle,
        layer: usize,
        keys: &DeviceTensor,
        values: &DeviceTensor,
        backend: &B,
    ) -> Result<()> {
        let num_new = keys.shape[0];
        let layer_key_bits = self.pool.key_bits_for_layer(layer);
        let layer_value_bits = self.pool.value_bits_for_layer(layer);

        // Compress K
        let k_rotation = self.pool.k_rotation(layer);
        let k_centroids = self.pool.centroids(layer_key_bits);
        backend.turboquant_compress(
            keys,
            k_rotation,
            k_centroids,
            layer_key_bits,
            &self.scratch.k_packed,
            &self.scratch.k_norms,
        )?;

        // Compress V
        let v_rotation = self.pool.v_rotation(layer);
        let v_centroids = self.pool.centroids(layer_value_bits);
        backend.turboquant_compress(
            values,
            v_rotation,
            v_centroids,
            layer_value_bits,
            &self.scratch.v_packed,
            &self.scratch.v_norms,
        )?;

        // Get sequence state
        let seq = self.sequences.get_mut(&handle.0).ok_or_else(|| {
            FractureError::KvCache(format!("invalid handle: {}", handle.0))
        })?;

        // On layer 0: allocate any new blocks needed
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

        // Write compressed data into blocks for this layer
        let start_token = seq.current_len - num_new;
        let mut written = 0;

        while written < num_new {
            let global_pos = start_token + written;
            let block_idx = global_pos / BLOCK_SIZE;
            let offset_in_block = global_pos % BLOCK_SIZE;
            let block_id = seq.block_table[block_idx];

            let slots = BLOCK_SIZE - offset_in_block;
            let to_write = (num_new - written).min(slots);

            // Copy packed K indices into block
            let k_dst = self.pool.k_packed(block_id, layer);
            backend.copy_rows(&self.scratch.k_packed, k_dst, written, offset_in_block, to_write)?;

            // Copy K norms into block
            let k_norms_dst = self.pool.k_norms(block_id, layer);
            backend.copy_rows(&self.scratch.k_norms, k_norms_dst, written, offset_in_block, to_write)?;

            // Copy packed V indices into block
            let v_dst = self.pool.v_packed(block_id, layer);
            backend.copy_rows(&self.scratch.v_packed, v_dst, written, offset_in_block, to_write)?;

            // Copy V norms into block
            let v_norms_dst = self.pool.v_norms(block_id, layer);
            backend.copy_rows(&self.scratch.v_norms, v_norms_dst, written, offset_in_block, to_write)?;

            written += to_write;
        }

        Ok(())
    }

    /// Get the block table for a sequence.
    pub fn block_table(&self, handle: CacheHandle) -> Result<&[usize]> {
        let seq = self.sequences.get(&handle.0).ok_or_else(|| {
            FractureError::KvCache(format!("invalid handle: {}", handle.0))
        })?;
        Ok(&seq.block_table)
    }

    /// Get the current sequence length.
    pub fn seq_len(&self, handle: CacheHandle) -> Result<usize> {
        let seq = self.sequences.get(&handle.0).ok_or_else(|| {
            FractureError::KvCache(format!("invalid handle: {}", handle.0))
        })?;
        Ok(seq.current_len)
    }

    /// Free all blocks for a sequence.
    pub fn free(&mut self, handle: CacheHandle) -> Result<()> {
        let seq = self.sequences.remove(&handle.0).ok_or_else(|| {
            FractureError::KvCache(format!("invalid handle: {}", handle.0))
        })?;
        for block_id in &seq.block_table {
            self.pool.free_block(*block_id);
        }
        Ok(())
    }

    /// Number of free blocks available.
    pub fn num_free_blocks(&self) -> usize {
        self.pool.num_free()
    }

    /// Available token capacity.
    pub fn available_token_capacity(&self) -> usize {
        self.pool.num_free() * BLOCK_SIZE
    }

    /// Access the underlying pool (for attention kernel dispatch).
    pub fn pool(&self) -> &QuantizedBlockPool {
        &self.pool
    }

    /// Free all GPU memory.
    pub fn destroy<B: Backend>(&self, backend: &B) -> Result<()> {
        backend.free(&self.scratch.k_packed)?;
        backend.free(&self.scratch.k_norms)?;
        backend.free(&self.scratch.v_packed)?;
        backend.free(&self.scratch.v_norms)?;
        self.pool.destroy(backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_kv_cache::BLOCK_SIZE;
    use fracture_core::{DeviceTensor, TensorId};
    use std::sync::Mutex;

    /// Mock backend for testing block pool allocation without a real GPU.
    struct MockBackend {
        state: Mutex<MockState>,
    }

    struct MockState {
        next_id: u64,
        allocated: usize,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                state: Mutex::new(MockState {
                    next_id: 1,
                    allocated: 0,
                }),
            }
        }
    }

    impl Backend for MockBackend {
        fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
            let mut state = self.state.lock().unwrap();
            let id = state.next_id;
            state.next_id += 1;
            let numel: usize = shape.iter().product();
            state.allocated += numel * dtype.size_bytes();
            Ok(DeviceTensor::new(TensorId(id), shape.to_vec(), dtype))
        }

        fn free(&self, _tensor: &DeviceTensor) -> Result<()> {
            Ok(())
        }

        fn copy_to_device(&self, _dst: &DeviceTensor, _src: &[u8]) -> Result<()> {
            Ok(())
        }

        fn copy_to_host(&self, _src: &DeviceTensor, _dst: &mut [u8]) -> Result<()> {
            Ok(())
        }

        fn matmul(
            &self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor,
        ) -> Result<()> {
            Ok(())
        }

        fn rmsnorm(
            &self, _input: &DeviceTensor, _weight: &DeviceTensor, _eps: f64,
            _out: &DeviceTensor,
        ) -> Result<()> {
            Ok(())
        }

        fn rope(
            &self, _q: &DeviceTensor, _k: &DeviceTensor, _positions: &[u32],
            _theta: f64, _head_dim: usize,
        ) -> Result<()> {
            Ok(())
        }

        fn attention(
            &self, _q: &DeviceTensor, _k: &DeviceTensor, _v: &DeviceTensor,
            _num_kv_heads: usize, _start_pos: usize, _out: &DeviceTensor,
        ) -> Result<()> {
            Ok(())
        }

        fn silu_mul(
            &self, _gate: &DeviceTensor, _up: &DeviceTensor, _out: &DeviceTensor,
        ) -> Result<()> {
            Ok(())
        }

        fn embedding(
            &self, _token_ids: &[u32], _table: &DeviceTensor, _out: &DeviceTensor,
        ) -> Result<()> {
            Ok(())
        }

        fn add(
            &self, _a: &DeviceTensor, _b: &DeviceTensor, _out: &DeviceTensor,
        ) -> Result<()> {
            Ok(())
        }

        fn copy_rows(
            &self, _src: &DeviceTensor, _dst: &DeviceTensor, _src_offset: usize,
            _dst_offset: usize, _count: usize,
        ) -> Result<()> {
            Ok(())
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
        fn create_timer(&self) -> Result<fracture_core::DeviceTimer> {
            Ok(fracture_core::DeviceTimer(0))
        }
        fn start_timer(&self, _timer: &fracture_core::DeviceTimer) -> Result<()> {
            Ok(())
        }
        fn stop_timer(&self, _timer: &fracture_core::DeviceTimer) -> Result<f32> {
            Ok(0.0)
        }
        fn destroy_timer(&self, _timer: &fracture_core::DeviceTimer) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_pool_creation() {
        let backend = MockBackend::new();
        let config = TurboQuantConfig::default();
        let pool = QuantizedBlockPool::new(4, 2, 8, 128, &config, &backend).unwrap();

        assert_eq!(pool.capacity(), 4);
        assert_eq!(pool.num_free(), 4);
    }

    #[test]
    fn test_pool_alloc_free() {
        let backend = MockBackend::new();
        let config = TurboQuantConfig::default();
        let mut pool = QuantizedBlockPool::new(4, 2, 8, 128, &config, &backend).unwrap();

        let b0 = pool.alloc_block().unwrap();
        assert_eq!(pool.num_free(), 3);

        let b1 = pool.alloc_block().unwrap();
        assert_ne!(b0, b1);
        assert_eq!(pool.num_free(), 2);

        pool.free_block(b0);
        assert_eq!(pool.num_free(), 3);

        // Exhaust remaining
        pool.alloc_block().unwrap();
        pool.alloc_block().unwrap();
        pool.alloc_block().unwrap();
        assert_eq!(pool.num_free(), 0);

        // OOM
        assert!(pool.alloc_block().is_err());
    }

    #[test]
    fn test_pool_layer_bits() {
        let backend = MockBackend::new();
        let config = TurboQuantConfig {
            key_bits: 4,
            value_bits: 2,
            protected_bits: 8,
            protected_layers: 1,
            ..Default::default()
        };
        let pool = QuantizedBlockPool::new(2, 4, 8, 128, &config, &backend).unwrap();

        // Layer 0 protected
        assert_eq!(pool.key_bits_for_layer(0), 8);
        assert_eq!(pool.value_bits_for_layer(0), 8);
        // Layer 1 normal
        assert_eq!(pool.key_bits_for_layer(1), 4);
        assert_eq!(pool.value_bits_for_layer(1), 2);
        // Layer 2 normal
        assert_eq!(pool.key_bits_for_layer(2), 4);
        // Layer 3 protected (last 1)
        assert_eq!(pool.key_bits_for_layer(3), 8);
    }

    #[test]
    fn test_pool_rotation_matrices() {
        let backend = MockBackend::new();
        let config = TurboQuantConfig::default();
        let pool = QuantizedBlockPool::new(1, 2, 8, 128, &config, &backend).unwrap();

        // Should have rotation matrices for both layers
        let k_rot_0 = pool.k_rotation(0);
        let v_rot_0 = pool.v_rotation(0);
        assert_eq!(k_rot_0.shape, vec![128, 128]);
        assert_eq!(v_rot_0.shape, vec![128, 128]);

        // K and V rotations should be different tensors
        assert_ne!(k_rot_0.id, v_rot_0.id);
    }

    #[test]
    fn test_pool_centroid_tables() {
        let backend = MockBackend::new();
        let config = TurboQuantConfig::default(); // K4/V2, no protection
        let pool = QuantizedBlockPool::new(1, 1, 8, 128, &config, &backend).unwrap();

        let k_centroids = pool.centroids(4);
        assert_eq!(k_centroids.shape, vec![16]); // 2^4

        let v_centroids = pool.centroids(2);
        assert_eq!(v_centroids.shape, vec![4]); // 2^2
    }

    #[test]
    fn test_manager_alloc_free() {
        let backend = MockBackend::new();
        let config = TurboQuantConfig::default();
        let mut mgr =
            QuantizedKvCacheManager::new(4, 2, 8, 128, 16, config, &backend).unwrap();

        let h0 = mgr.alloc().unwrap();
        assert_eq!(mgr.seq_len(h0).unwrap(), 0);
        assert_eq!(mgr.num_free_blocks(), 3); // 1 block allocated for initial

        let h1 = mgr.alloc().unwrap();
        assert_eq!(mgr.num_free_blocks(), 2);

        mgr.free(h0).unwrap();
        assert_eq!(mgr.num_free_blocks(), 3);

        mgr.free(h1).unwrap();
        assert_eq!(mgr.num_free_blocks(), 4);
    }

    #[test]
    fn test_manager_invalid_handle() {
        let backend = MockBackend::new();
        let config = TurboQuantConfig::default();
        let mut mgr =
            QuantizedKvCacheManager::new(2, 1, 8, 128, 16, config, &backend).unwrap();

        let bad = CacheHandle(999);
        assert!(mgr.seq_len(bad).is_err());
        assert!(mgr.block_table(bad).is_err());
        assert!(mgr.free(bad).is_err());
    }

    #[test]
    fn test_manager_block_table() {
        let backend = MockBackend::new();
        let config = TurboQuantConfig::default();
        let mut mgr =
            QuantizedKvCacheManager::new(4, 1, 8, 128, 16, config, &backend).unwrap();

        let h = mgr.alloc().unwrap();
        let bt = mgr.block_table(h).unwrap();
        assert_eq!(bt.len(), 1); // initial block

        mgr.free(h).unwrap();
    }

    #[test]
    fn test_block_size_consistency() {
        assert_eq!(BLOCK_SIZE, 16, "quantized cache must use same BLOCK_SIZE as FP16 cache");
    }
}
