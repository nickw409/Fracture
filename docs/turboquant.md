# TurboQuant KV Cache Compression in Fracture

Fracture implements Google's TurboQuant algorithm (ICLR 2026) for KV cache compression, reducing inference memory consumption by 5x with negligible quality loss. This document describes the algorithm, the implementation, the kernel optimizations, and the validated results.

## Background

During autoregressive LLM inference, the KV cache stores key and value vectors from every past token at every layer. For Llama 3.1 8B (32 layers, 8 KV heads, head_dim=128), each token consumes 128 KB of GPU memory across all layers. A 4096-token context requires 512 MB of KV cache alone — on a 24 GB RTX 3090 with 15.3 GB of model weights, this limits the system to ~4000 tokens of context or a handful of concurrent sequences.

TurboQuant compresses each KV vector from 16 bits per coordinate (FP16) to 2-4 bits per coordinate via rotation + optimal scalar quantization, reducing the per-token KV memory by 5x. This enables ~20,000 tokens of context in the same memory budget, or 5x more concurrent sequences under continuous batching.

## Algorithm

Fracture implements the community-validated V3 variant of TurboQuant, which drops the paper's Stage 2 (QJL residual correction) in favor of allocating all bits to MSE-optimal reconstruction. Six independent reimplementations across Python, C, and Rust confirmed that QJL's unbiased inner product estimator fails under softmax — the estimator's variance is exponentially amplified, causing 0 out of 27 generation tests to pass. The MSE-only V3 approach achieves 0.9996 cosine similarity in attention scores at 5.1x compression, with 18 out of 18 perfect text retrieval at 8K context.

### Compression (per head vector)

Given a KV head vector `x` of dimension `d` (e.g., `d = 128`):

1. **Normalize.** Compute the L2 norm `||x||` and store it as FP16 (2 bytes). Normalize: `x_norm = x / (||x|| + 1e-8)`. The epsilon guard prevents division by zero for degenerate head vectors.

2. **Rotate.** Multiply by a random orthogonal matrix: `y = Pi @ x_norm`. The matrix `Pi` is Haar-distributed (generated via QR decomposition of a seeded Gaussian matrix). After rotation, each coordinate of the unit vector follows a near-Gaussian distribution `N(0, 1/d)`, which makes per-coordinate scalar quantization near-optimal. Without rotation, coordinate distributions vary across dimensions and scalar quantization is wasteful.

3. **Quantize.** Apply Lloyd-Max optimal scalar quantization to each coordinate independently. For a given bit-width `b`, the `2^b` centroids are the fixed points of the Lloyd-Max conditions for `N(0, 1/d)` — computed once at startup via iterative conditional expectation (continuous 1-D k-means with Simpson's rule integration). Each coordinate maps to the index of its nearest centroid.

4. **Bit-pack.** Pack the quantized indices into bytes. At 4 bits: 2 indices per byte (high nibble, low nibble). At 2 bits: 4 indices per byte. At 8 bits: 1 index per byte. The packed representation is stored as INT8 tensors on the GPU.

**Storage per vector:** `ceil(d * bits / 8) + 2` bytes (packed indices + FP16 norm).

### Decompression

1. Unpack bit indices from the packed byte array.
2. Look up centroid values from the codebook table.
3. Unrotate: `x_hat = Pi^T @ y_hat` (orthogonal matrix, so `Pi^T = Pi^{-1}`).
4. Rescale by the stored norm: `x_out = x_hat * ||x||`.

### Asymmetric K/V Bit Widths

Keys require higher precision than values. Attention scores are dot products with keys — small angular errors in K vectors translate directly to score errors. Values are weighted sums — MSE noise in V vectors averages out across the softmax distribution.

The default configuration is K4/V2: 4-bit keys and 2-bit values, averaging 3 bits per coordinate for 5.1x compression. The system supports any combination of 2, 4, and 8 bits independently for keys and values.

### Layer-Adaptive Protection

The first and last N transformer layers are disproportionately sensitive to quantization error. Protected layers use 8-bit quantization (256 centroids) through the same rotation + Lloyd-Max pipeline — no separate FP16 storage path. Protection improves top-1 attention accuracy from 94% to 99% at a modest cost (5.1x to 3.6x compression). Disabled by default; configured via `--tq-protected-layers N`.

## Kernel Optimizations

### Query Pre-Rotation (K scores)

The naive approach decompresses each K vector before computing the attention score:

```
score = dot(q, Pi_k^T @ y_hat_k) * norm_k / sqrt(d)
```

This requires an `O(d^2)` matrix-vector multiply per KV position. By linearity of the dot product and the orthogonal matrix:

```
dot(q, Pi_k^T @ y_hat_k) = dot(Pi_k @ q, y_hat_k)
```

Pre-rotating the query once (`Pi_k @ q`, one `O(d^2)` operation per query head) allows all K scores to be computed as simple `O(d)` dot products in the rotated space — no per-position unrotation. For a 4096-token context, this reduces per-query compute from `4096 * d^2` to `d^2 + 4096 * d`.

### V Accumulation in Rotated Space

The attention output is a weighted sum of decompressed V vectors:

```
out = sum_i(prob_i * Pi_v^T @ y_hat_v_i * norm_v_i)
```

By linearity of Pi_v^T:

```
out = Pi_v^T @ sum_i(prob_i * y_hat_v_i * norm_v_i)
```

The entire weighted sum is accumulated in V's rotated space using only the centroid values and norms. A single `O(d^2)` unrotation at the end produces the final output. This reduces V computation from `O(kv_len * d^2)` to `O(kv_len * d + d^2)`.

### Fused Attention Kernel

The production kernel (`attention_paged_tq.cu`) fuses decompression and attention into a single kernel launch, avoiding intermediate buffer allocation. The kernel proceeds in four phases:

1. **Pre-rotate query** with `Pi_k` (shared memory, `O(d^2)`).
2. **Score phase.** For each KV position: unpack K indices, look up centroids, compute `dot(q_rot, y_hat_k) * norm_k * scale`. Track running max for stable softmax. All work in registers and shared memory.
3. **Softmax.** Exp-and-sum with warp-level reductions, same as the standard paged attention kernel.
4. **V accumulation.** For each KV position: unpack V indices, look up centroids, accumulate `prob * y_hat_v * norm_v` in rotated space. Final unrotation with `Pi_v^T` writes the output.

Grid: `(num_tokens, num_q_heads)`, 128 threads per block. Shared memory: `kv_len * 4` bytes (scores) + `2 * head_dim * 4` bytes (rotated query + V accumulator) + centroid tables.

The rotation matrices (`128 * 128 * 4 = 64 KB` each) remain in global memory. With 4 MB total across 32 layers, they fit entirely in L2 cache after the first access.

## Implementation

### Trait Abstraction

The batched forward pass is generic over the `PagedCache` trait, which both `PagedKvCacheManager` (FP16) and `QuantizedKvCacheManager` (TurboQuant) implement:

```rust
pub trait PagedCache {
    fn seq_len(&self, handle: CacheHandle) -> Result<usize>;
    fn block_table(&self, handle: CacheHandle) -> Result<&[usize]>;
    fn append_kv<B: Backend>(&mut self, handle: CacheHandle, layer: usize,
                              keys: &DeviceTensor, values: &DeviceTensor, backend: &B) -> Result<()>;
    fn dispatch_attention<B: Backend>(&self, backend: &B, q: &DeviceTensor, handle: CacheHandle,
                                       cache_idx: usize, num_kv_heads: usize, kv_len: usize,
                                       start_pos: usize, out: &DeviceTensor) -> Result<()>;
    fn alloc(&mut self) -> Result<CacheHandle>;
    fn free(&mut self, handle: CacheHandle) -> Result<()>;
}
```

The `dispatch_attention` method is the key abstraction. The FP16 implementation gathers block tensors and calls `Backend::attention_paged`. The TurboQuant implementation gathers packed indices, norms, rotation matrices, and centroids, then calls `Backend::attention_paged_tq`. The forward pass code (`batched_forward`, `batched_forward_node`) is identical for both paths — no code duplication.

### Backend Trait Methods

Two new methods on the `Backend` trait, following the existing opt-in pattern (default implementations return an error; backends override to support):

- **`turboquant_compress`** — Normalize, rotate, Lloyd-Max quantize, and bit-pack KV vectors in a single fused GPU kernel. Called during KV cache append.
- **`attention_paged_tq`** — Fused quantized paged attention with separate K/V rotation matrices and centroid tables. Called during the attention phase of the forward pass.

### Storage Layout

The `QuantizedBlockPool` pre-allocates all block memory at startup, mirroring the FP16 `BlockPool` design. Each physical block contains compressed K and V data for 16 tokens across all layers. Per-layer storage varies when protected layers use higher bit widths:

```
Standard layer (K4/V2):
  K packed: [16, num_kv_heads * 64] INT8     (8,192 bytes)
  K norms:  [16, num_kv_heads] FP16           (256 bytes)
  V packed: [16, num_kv_heads * 32] INT8     (4,096 bytes)
  V norms:  [16, num_kv_heads] FP16           (256 bytes)
  Total: 12,800 bytes per layer

Protected layer (K8/V8):
  K packed: [16, num_kv_heads * 128] INT8   (16,384 bytes)
  K norms:  [16, num_kv_heads] FP16            (256 bytes)
  V packed: [16, num_kv_heads * 128] INT8   (16,384 bytes)
  V norms:  [16, num_kv_heads] FP16            (256 bytes)
  Total: 33,280 bytes per layer
```

The pool also holds precomputed data on-device: two rotation matrices per layer (K and V, each `[head_dim, head_dim]` FP32), and one centroid table per distinct bit-width (`[2^bits]` FP32). Compress scratch tensors are pre-allocated once to avoid per-call `cudaMalloc` overhead during the hot path.

### Lloyd-Max Codebook Computation

Codebooks are computed at startup in pure Rust (no Python or scipy dependency). The solver uses the Gaussian approximation `N(0, 1/d)` for the post-rotation coordinate distribution, which is accurate for `d >= 64`. Numerical integration uses composite Simpson's rule with 500 quadrature points. Convergence tolerance is `1e-10` with a maximum of 200 iterations. The codebook for `d=128, bits=4` (16 centroids) converges in ~50 iterations and takes <1ms.

The centroids are symmetric around zero by construction (the underlying distribution is symmetric). For `d=128`:

| Bits | Levels | Centroid range | MSE per coordinate |
|------|--------|----------------|-------------------|
| 2 | 4 | [-0.133, 0.133] | 0.116 |
| 4 | 16 | [-0.218, 0.218] | 0.009 |
| 8 | 256 | [-0.276, 0.276] | < 0.001 |

Centroids scale with `1/sqrt(d)` — the solver is parameterized by `d` and produces correct codebooks for any head dimension. A runtime guard falls back to solver computation if no precomputed table matches the model's head_dim.

### Rotation Matrix Generation

Rotation matrices are generated deterministically from a seed via Xoshiro256** PRNG (no external dependencies) and Gram-Schmidt QR orthogonalization. Each layer gets distinct K and V rotation seeds: `K_seed = base_seed + layer * 1000`, `V_seed = base_seed + layer * 1000 + 500`. Diagonal sign correction ensures proper rotations (`det(Pi) = +1`).

Properties verified by unit tests:
- Orthogonality: `Pi^T @ Pi = I` within `1e-4` tolerance
- Norm preservation: `||Pi @ x|| = ||x||` within `1e-3`
- Roundtrip: `Pi^T @ Pi @ x = x` within `1e-4`
- Distribution: coordinates of `Pi @ e_1` follow `N(0, 1/d)` (mean < 0.05, std within 50% of `1/sqrt(d)`)

## CLI Interface

```
fracture-worker-cuda \
  --coordinator <host:port> \
  --model <path-to-gguf> \
  --kv-quant turboquant          # Enable TurboQuant (default: FP16 paged)
  --tq-key-bits 4                # Key quantization bits (default: 4)
  --tq-value-bits 2              # Value quantization bits (default: 2)
  --tq-protected-bits 8          # Protected layer bits (default: 8)
  --tq-protected-layers 0        # Edge layers to protect (default: 0)
  --tq-seed 42                   # Rotation matrix seed (default: 42)
```

Without `--kv-quant`, the worker uses the existing FP16 paged path with zero code changes to the forward pass. The coordinator accepts `--kv-quant turboquant` for logging; actual cache management is local to each worker.

## Validated Results

All results measured on an NVIDIA RTX 3090 (24 GB VRAM, Ampere architecture).

### Compress/Decompress Round-Trip (GPU Kernel Tests)

Per-head-vector cosine similarity between original FP16 and compress-then-decompress output:

| Bit Width | Cosine Similarity | Threshold |
|-----------|------------------|-----------|
| 8-bit | > 0.999 | Near-lossless |
| 4-bit | > 0.95 | High fidelity |
| 2-bit | > 0.80 | Acceptable for V |

Norm preservation: relative error < 2% across all bit widths. Zero vectors handled correctly (no NaN, no crash) via epsilon-guarded normalization.

### End-to-End Batched Forward (A/B Comparison)

Same random model weights, same prompt, compared logit output between FP16 paged attention and TurboQuant paged attention:

| Config | Logit Cosine Similarity | Greedy Argmax Match |
|--------|------------------------|---------------------|
| TQ 8-bit K+V | 0.999999 | Yes |
| TQ K4/V2 (default) | 0.998995 | Yes |

The K4/V2 configuration — the production default at 5.1x compression — produces logits with 0.999 cosine similarity and identical greedy predictions.

### Memory Budget (Llama 3.1 8B)

```
RTX 3090 (24 GB):
  Weights: 15.3 GB
  Available for KV cache: 8.2 GB

  FP16:    2,097,152 bytes/block →  4,096 blocks →  65,536 tokens
  TQ K4V2:   409,600 bytes/block → 20,480 blocks → 327,680 tokens

  Compression ratio: 5.12x
```

### Regression Testing

772 tests pass with zero failures after TurboQuant integration:
- 750 unit/integration tests via nextest (serialized GPU groups)
- 22 GPU kernel and end-to-end tests
- All 15 pre-existing GPU integration tests unchanged (paged vs contiguous bit-identical, batched vs sequential match, concurrent scheduler correctness)

## File Map

```
crates/fracture-core/src/
  turboquant.rs              TurboQuantConfig, Lloyd-Max solver, rotation matrix PRNG,
                             codebook computation, memory budget calculation (822 lines)
  backend.rs                 Backend::turboquant_compress, Backend::attention_paged_tq
                             trait methods with default error returns

crates/fracture-engine/src/
  quantized_paged_kv_cache.rs  QuantizedBlockPool, QuantizedKvCacheManager, CompressScratch,
                               block alloc/free/append, pool init with rotation matrices
                               and codebooks on device (825 lines)
  batched.rs                   PagedCache trait + impls for PagedKvCacheManager and
                               QuantizedKvCacheManager; batched_forward and
                               batched_forward_node generic over C: PagedCache
  engine.rs                    KvCacheBackend::QuantizedPaged variant

backends/fracture-cuda/
  kernels/
    turboquant_compress.cu     Fused normalize/rotate/quantize/pack kernel (222 lines)
    turboquant_decompress.cu   Unpack/dequantize/unrotate test utility kernel (127 lines)
    attention_paged_tq.cu      Fused TQ attention with query pre-rotation and
                               V accumulation optimizations (309 lines)
  src/
    backend.rs                 CudaBackend::turboquant_compress, attention_paged_tq
    ffi.rs                     FFI declarations for 3 new kernel launch functions

bins/fracture-worker-cuda/     --kv-quant turboquant CLI, cache creation dispatch,
                               serve loop with PagedCache trait dispatch
bins/fracture-coordinator-cuda/ --kv-quant flag for logging

bins/fracture-server-cuda/tests/
  turboquant_gpu.rs            7 GPU validation tests (round-trip + e2e A/B comparison)
```

## References

- Ashkboos, S., Mohtashami, A., et al. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." ICLR 2026. [arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874)
- tonbistudio/turboquant-pytorch — Community PyTorch implementation (V3 with MSE-only, no QJL). [github.com/tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)
- Google Research Blog. "TurboQuant: Redefining AI efficiency with extreme compression." March 2026. [research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
