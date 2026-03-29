use crate::{DType, DeviceTensor, DeviceTimer, Result};

/// The GPU backend trait. All GPU operations go through this interface.
///
/// The engine is generic over `B: Backend` and never imports any backend crate directly.
/// Each backend (CUDA, Metal) implements this trait independently. Adding a new backend
/// requires zero changes to the engine, generation loop, server, or protocol.
pub trait Backend: Send + Sync {
    // ── Memory management ──────────────────────────────────────

    /// Allocate a tensor on the device with the given shape and dtype.
    fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor>;

    /// Free a device tensor. After this call the tensor id is invalid.
    fn free(&self, tensor: &DeviceTensor) -> Result<()>;

    /// Copy data from host memory to a device tensor.
    fn copy_to_device(&self, dst: &DeviceTensor, src: &[u8]) -> Result<()>;

    /// Copy data from a device tensor to host memory.
    fn copy_to_host(&self, src: &DeviceTensor, dst: &mut [u8]) -> Result<()>;

    // ── Compute operations ─────────────────────────────────────

    /// Matrix multiplication: C = A @ B
    /// A is [M, K], B is [K, N], C is [M, N]. All row-major.
    fn matmul(&self, a: &DeviceTensor, b: &DeviceTensor, out: &DeviceTensor) -> Result<()>;

    /// RMSNorm: output = (x / sqrt(mean(x^2) + eps)) * weight
    fn rmsnorm(
        &self,
        input: &DeviceTensor,
        weight: &DeviceTensor,
        eps: f64,
        out: &DeviceTensor,
    ) -> Result<()>;

    /// Apply Rotary Positional Embeddings to Q and K tensors.
    /// positions contains the sequence position for each token.
    fn rope(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        positions: &[u32],
        theta: f64,
        head_dim: usize,
    ) -> Result<()>;

    /// Scaled dot-product attention with causal masking and GQA.
    /// q: [N, num_q_heads, head_dim]
    /// k_cache: [seq_len, num_kv_heads, head_dim]
    /// v_cache: [seq_len, num_kv_heads, head_dim]
    /// out: [N, num_q_heads, head_dim]
    fn attention(
        &self,
        q: &DeviceTensor,
        k_cache: &DeviceTensor,
        v_cache: &DeviceTensor,
        num_kv_heads: usize,
        start_pos: usize,
        out: &DeviceTensor,
    ) -> Result<()>;

    /// Paged attention: scaled dot-product attention reading KV data from block tables.
    /// q: [N, num_q_heads, head_dim]
    /// block_table: physical block IDs for this sequence
    /// k_block_ptrs / v_block_ptrs: device pointers to each block's K/V data for this layer
    /// kv_len: total tokens across all blocks
    /// start_pos: tokens before this batch (for causal mask)
    /// out: [N, num_q_heads, head_dim]
    ///
    /// Default returns an error — backends must opt in by overriding.
    fn attention_paged(
        &self,
        _q: &DeviceTensor,
        _block_table: &[i32],
        _k_block_ptrs: &[*const std::ffi::c_void],
        _v_block_ptrs: &[*const std::ffi::c_void],
        _num_kv_heads: usize,
        _kv_len: usize,
        _start_pos: usize,
        _out: &DeviceTensor,
    ) -> Result<()> {
        Err(crate::FractureError::Backend(
            "paged attention not supported by this backend".into(),
        ))
    }

    /// Fused SiLU activation and elementwise multiply: output = silu(gate) * up
    fn silu_mul(
        &self,
        gate: &DeviceTensor,
        up: &DeviceTensor,
        out: &DeviceTensor,
    ) -> Result<()>;

    /// Token embedding lookup. Given token IDs, gather embedding rows.
    fn embedding(
        &self,
        token_ids: &[u32],
        embedding_table: &DeviceTensor,
        out: &DeviceTensor,
    ) -> Result<()>;

    /// Elementwise addition: out = a + b
    fn add(&self, a: &DeviceTensor, b: &DeviceTensor, out: &DeviceTensor) -> Result<()>;

    /// Copy a slice of a tensor along the first dimension.
    fn copy_rows(
        &self,
        src: &DeviceTensor,
        dst: &DeviceTensor,
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) -> Result<()>;

    // ── Device info ────────────────────────────────────────────

    /// Human-readable device name.
    fn device_name(&self) -> &str;

    /// Total device memory in bytes.
    fn total_memory(&self) -> usize;

    /// Currently available device memory in bytes.
    fn available_memory(&self) -> usize;

    /// Synchronize all pending device operations.
    fn synchronize(&self) -> Result<()>;

    // ── Profiling ────────────────────────────────────────────────

    /// Create a GPU timer for measuring kernel execution time.
    fn create_timer(&self) -> Result<DeviceTimer>;

    /// Record the start timestamp on the current stream.
    fn start_timer(&self, timer: &DeviceTimer) -> Result<()>;

    /// Record the stop timestamp, synchronize, and return elapsed milliseconds.
    fn stop_timer(&self, timer: &DeviceTimer) -> Result<f32>;

    /// Destroy a GPU timer and free its resources.
    fn destroy_timer(&self, timer: &DeviceTimer) -> Result<()>;

    /// Push a named profiling marker (e.g., NVTX range). Default no-op.
    fn marker_push(&self, _name: &str) {}

    /// Pop the most recent profiling marker. Default no-op.
    fn marker_pop(&self) {}
}
