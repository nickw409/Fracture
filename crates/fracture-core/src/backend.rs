use crate::{DType, DeviceTensor, Result};

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
}
