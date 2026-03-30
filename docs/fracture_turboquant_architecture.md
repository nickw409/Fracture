# Fracture: TurboQuant KV Cache Compression

**Depends on:** Phase 4 complete and validated (paged KV cache, continuous batching working)
**Goal:** Integrate TurboQuant (ICLR 2026) KV cache compression into the paged attention system, achieving ~5x memory reduction with no accuracy loss, while maintaining the existing FP16 path for A/B comparison.

---

## What Changes from Phase 4

Phase 4 established paged KV cache with 16-token FP16 blocks. Every token consumes 2 bytes per coordinate in the cache. For Llama 3.1 8B with 8 KV heads and head_dim=128, that's 4 KB per token per layer (K+V), or 128 KB per token across 32 layers. A 4096-token context consumes ~512 MB of KV cache alone.

TurboQuant compresses this by 5x, enabling ~20K context in the same memory, or fitting more concurrent sequences under the same memory budget.

| Component | Phase 4 | Phase 4.5 (TurboQuant) |
|---|---|---|
| KV cache precision | FP16 (16 bits/coordinate) | 2-4 bits/coordinate (asymmetric K/V) |
| Memory per token (8B model) | 128 KB across 32 layers | ~26 KB (K4/V2, 4.9x reduction) |
| Max context (RTX 3090, 24 GB) | ~4K tokens | ~20K tokens |
| Max concurrent sequences | Memory-limited | ~5x more sequences |
| Attention kernel | `attention_paged` | `attention_paged` (FP16) or `attention_paged_tq` (quantized) |
| Engine/server/protocol changes | N/A | Minimal — new `KvCacheBackend` variant + Backend trait methods |

**Critical constraint:** The existing FP16 paged path remains fully functional. TurboQuant is an opt-in mode selected via CLI flag (`--kv-quant turboquant`). Both paths coexist for A/B validation.

---

## Algorithm: TurboQuant V3 (Community-Validated)

The original TurboQuant paper proposes a two-stage pipeline: PolarQuant (MSE-optimal quantization) + QJL (1-bit residual correction for unbiased inner products). However, **6+ independent implementations confirmed that QJL fails for attention**: softmax exponentially amplifies the QJL estimator's variance, causing 0/27 generation tests to fail.

The community-validated V3 approach drops QJL entirely. All bits go to MSE reconstruction quality. This achieves 0.9996 cosine similarity in attention scores at 5.1x compression, with 18/18 perfect text retrieval at 8K context.

### Compression Pipeline (per KV vector)

```
Input: x ∈ R^d (one KV head vector, d = head_dim = 128)

1. Normalize:     x_norm = x / (||x||₂ + ε)    Store ||x||₂ as FP16 (2 bytes), ε = 1e-8
2. Rotate:        y = Pi @ x_norm              Pi is d×d random orthogonal matrix (Haar-distributed)
3. Quantize:      idx = LloydMax(y, bits)      Per-coordinate scalar quantization (bits ∈ {2,3,4})
4. Pack:          packed = bitpack(idx)         Pack indices into bytes (4 indices/byte at 2-bit)

Storage per vector: ceil(d * bits / 8) + 2 bytes (norm)
```

### Decompression Pipeline

```
1. Unpack:        idx = unpack(packed)
2. Lookup:        y_hat = centroids[idx]        Centroid table from Lloyd-Max
3. Unrotate:      x_hat = Pi^T @ y_hat          Pi^T = Pi^{-1} (orthogonal)
4. Rescale:       x_out = x_hat * ||x||₂
```

### Why Each Step Matters

**Random rotation (Pi):** After rotation by a Haar-distributed orthogonal matrix, each coordinate of a unit vector follows a near-Gaussian distribution N(0, 1/d). This makes per-coordinate scalar quantization near-optimal — without rotation, coordinate distributions vary wildly and scalar quantization is wasteful.

**Lloyd-Max quantization:** The optimal scalar quantizer for a known distribution. For N(0, 1/d), the centroids and decision boundaries are computed once offline via iterative conditional expectation (continuous 1-D k-means). For 4-bit (16 levels) the MSE distortion per coordinate is 0.009; for 2-bit (4 levels) it's 0.116.

**Norm preservation:** The L2 norm captures magnitude information lost by unit-sphere normalization. Stored in FP16 (2 bytes) — negligible overhead relative to the d coordinates.

### Asymmetric K/V Bit Widths

Keys require higher precision than values because attention scores are dot products with keys (sensitive to angular error), while values are weighted sums (tolerant of MSE). Community-validated configurations:

| Config | Key Bits | Value Bits | Avg Bits | Compression | Cosine Sim |
|---|---|---|---|---|---|
| K4/V2 | 4 | 2 | 3.0 | **5.1x** | 0.9996 |
| K4/V2 + protected | 4 (8 for edge layers) | 2 (8 for edge layers) | ~3.5 | 3.6x | 0.9997 |
| K3/V2 | 3 | 2 | 2.5 | 6.0x | 0.9990 |

**Default configuration: K4/V2** — best compression-to-quality tradeoff.

### Layer-Adaptive Protection

The first and last N layers of the transformer are disproportionately sensitive to quantization error. Protecting them with higher bit-width (8-bit quantization, still through the same quantized pipeline) improves top-1 attention accuracy from 94% to 99% at a modest compression cost (5.1x -> 3.6x). Protected layers use 8-bit K and 8-bit V via the same rotation + Lloyd-Max path — no separate FP16 storage needed, just a different codebook (256 centroids instead of 16 or 4).

**Default: `protected_layers = 0`** (no protection). Configurable via `--tq-protected-layers N`.

### Residual Window

The most recent tokens in the KV cache are kept in FP16 (uncompressed). This is critical for generation quality — the model's attention to the immediately preceding context must be precise.

**Default: `residual_tokens = 0`** (all tokens compressed). Configurable via `--tq-residual-tokens N`. When set, the last N tokens per sequence remain in standard FP16 paged blocks; only older tokens are stored in quantized blocks.

---

## Architecture

### New Types

```rust
// crates/fracture-core/src/turboquant.rs

/// TurboQuant configuration, passed through from CLI to cache manager.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    pub key_bits: u8,              // 2, 3, or 4 (default: 4)
    pub value_bits: u8,            // 2, 3, or 4 (default: 2)
    pub protected_bits: u8,        // Bit width for protected layers (default: 8)
    pub protected_layers: usize,   // First/last N layers use protected_bits (default: 0)
    pub residual_tokens: usize,    // Recent tokens kept in FP16 (default: 0)
    pub seed: u64,                 // Base seed for rotation matrices (default: 42)
}
```

### Precomputed Tables

Computed once at startup, stored on-device:

1. **Rotation matrices:** Two per layer (one for K, one for V — different seeds), each `[head_dim, head_dim]` in FP32. For Llama 3.1 8B (32 layers, head_dim=128): 32 × 2 × 128 × 128 × 4 = 4 MB. Generated via QR decomposition of seeded Gaussian matrices (K seed = `base_seed + layer * 1000`, V seed = `base_seed + layer * 1000 + 500`). Stored as device tensors. K and V must use distinct rotations because the attention kernel pre-rotates the query with Pi_k (for score computation) and unrotates the accumulated output with Pi_v^T — using the same matrix for both would corrupt the V reconstruction.

2. **Lloyd-Max codebooks:** One per distinct bit-width in use, stored in a `HashMap<u8, DeviceTensor>`. Centroids for N(0, 1/d):
   - 2-bit: 4 centroids (16 bytes FP32)
   - 4-bit: 16 centroids (64 bytes FP32)
   - 8-bit: 256 centroids (1024 bytes FP32) — for protected layers

   These fit in CUDA constant memory or kernel arguments. Precomputed at startup via the Lloyd-Max algorithm for the Gaussian approximation of the rotated coordinate distribution. Only the bit-widths actually in use are computed and uploaded (e.g., K4/V2 with no protection uploads only the 4-bit and 2-bit tables; with `protected_layers > 0`, the 8-bit table is added).

### Quantized Block Storage

A quantized block stores the same 16 tokens as a standard FP16 block, but in compressed form:

```
FP16 block (K or V, one layer):
  Shape: [16, num_kv_heads, head_dim]
  Size:  16 × 8 × 128 × 2 = 32,768 bytes = 32 KB

Quantized block (4-bit K, one layer):
  Packed indices: 16 × 8 × 128 × 4 / 8 = 8,192 bytes = 8 KB
  Norms (FP16):   16 × 8 × 2 = 256 bytes
  Total:          8,448 bytes ≈ 8.25 KB (3.87x compression per block)

Quantized block (2-bit V, one layer):
  Packed indices: 16 × 8 × 128 × 2 / 8 = 4,096 bytes = 4 KB
  Norms (FP16):   16 × 8 × 2 = 256 bytes
  Total:          4,352 bytes ≈ 4.25 KB (7.71x compression per block)

Combined K4/V2 per block per layer: 12,800 bytes ≈ 12.5 KB (5.12x vs 64 KB FP16)
```

### QuantizedBlockPool

```rust
// crates/fracture-engine/src/quantized_paged_kv_cache.rs

/// A block of quantized KV data for one layer.
/// Unlike FP16 blocks which are [BLOCK_SIZE, num_kv_heads, head_dim] DeviceTensors,
/// quantized blocks store bit-packed indices and per-head norms separately.
struct QuantizedBlockData {
    packed_indices: DeviceTensor,  // [BLOCK_SIZE, packed_dim] as INT8
    norms: DeviceTensor,          // [BLOCK_SIZE, num_kv_heads] as FP16
}

pub struct QuantizedBlockPool {
    // Quantized storage for K and V (separate bit widths, may differ per layer)
    k_blocks: Vec<Vec<QuantizedBlockData>>,   // k_blocks[block_id][layer_idx]
    v_blocks: Vec<Vec<QuantizedBlockData>>,   // v_blocks[block_id][layer_idx]
    free_list: Vec<usize>,
    capacity: usize,

    // Dimensions
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,

    // Per-layer bit widths (protected layers use higher bits, e.g. 8)
    layer_key_bits: Vec<u8>,                  // [num_layers] — key bits per layer
    layer_value_bits: Vec<u8>,                // [num_layers] — value bits per layer

    // Precomputed tables (on device)
    k_rotation_matrices: Vec<DeviceTensor>,   // [num_layers] each [head_dim, head_dim] FP32
    v_rotation_matrices: Vec<DeviceTensor>,   // [num_layers] each [head_dim, head_dim] FP32
    // One centroid table per distinct bit-width in use (typically 2-3 tables)
    centroid_tables: HashMap<u8, DeviceTensor>, // bits → [2^bits] FP32
}

impl QuantizedBlockPool {
    /// Returns the effective key bit-width for a given layer.
    pub fn key_bits_for_layer(&self, layer: usize) -> u8 {
        self.layer_key_bits[layer]
    }

    /// Returns the effective value bit-width for a given layer.
    pub fn value_bits_for_layer(&self, layer: usize) -> u8 {
        self.layer_value_bits[layer]
    }

    /// Returns the centroid table for a given bit-width.
    pub fn centroids(&self, bits: u8) -> &DeviceTensor {
        &self.centroid_tables[&bits]
    }
}
```

### QuantizedKvCacheManager

```rust
/// Pre-allocated scratch tensors for compression (avoids per-call cudaMalloc).
/// Sized for the maximum single-step append (prefill chunk or 1 decode token).
struct CompressScratch {
    k_packed: DeviceTensor,   // [max_prefill_tokens, max_packed_dim] INT8
    k_norms: DeviceTensor,    // [max_prefill_tokens, num_kv_heads] FP16
    v_packed: DeviceTensor,   // [max_prefill_tokens, max_packed_dim] INT8
    v_norms: DeviceTensor,    // [max_prefill_tokens, num_kv_heads] FP16
}

pub struct QuantizedKvCacheManager {
    pool: QuantizedBlockPool,
    sequences: HashMap<u64, QuantizedSequenceBlocks>,
    next_id: u64,
    config: TurboQuantConfig,
    compress_scratch: CompressScratch,
}

struct QuantizedSequenceBlocks {
    // Quantized blocks for older tokens
    quantized_block_table: Vec<usize>,       // Logical block → physical block ID
    quantized_tokens: usize,                 // Tokens in quantized blocks

    // FP16 residual window (reuses standard BlockPool mechanics)
    // Only populated when config.residual_tokens > 0
    fp16_block_table: Vec<usize>,
    fp16_tokens: usize,
    fp16_last_block_fill: usize,

    // Total
    current_len: usize,
}
```

The manager's `append_kv` method:
1. If `residual_tokens > 0` and fp16 window is not full: write to FP16 blocks (standard path).
2. If fp16 window is full and a new token arrives: compress the oldest fp16 block into a quantized block, shift the window forward.
3. If `residual_tokens == 0`: compress directly into quantized blocks on write.

Protected layers use the same quantized pipeline with `protected_bits` (default 8) — no separate FP16 storage needed. The `key_bits_for_layer()` and `value_bits_for_layer()` methods return the effective bit-width, which is `protected_bits` for edge layers and `key_bits`/`value_bits` for middle layers.

### KvCacheBackend Extension

```rust
// crates/fracture-engine/src/engine.rs

pub enum KvCacheBackend {
    Contiguous(KvCacheManager),
    Paged(PagedKvCacheManager),
    QuantizedPaged(QuantizedKvCacheManager),  // NEW
}
```

The engine dispatches based on variant:
- `Paged` → existing `attention_paged` kernel
- `QuantizedPaged` → new `attention_paged_tq` kernel for quantized blocks + existing `attention_paged` for FP16 residual window blocks

### Backend Trait Extensions

```rust
// crates/fracture-core/src/backend.rs — new methods with default error returns

/// Compress KV vectors using TurboQuant rotation + Lloyd-Max quantization.
///
/// input: [N, num_kv_heads, head_dim] FP16 — the K or V vectors to compress
/// rotation_matrix: [head_dim, head_dim] FP32 — layer-specific Haar rotation
/// centroids: [2^bits] FP32 — Lloyd-Max centroid table
/// packed_out: [N, packed_dim] INT8 — bit-packed quantized indices
/// norms_out: [N, num_kv_heads] FP16 — per-head L2 norms
fn turboquant_compress(
    &self,
    _input: &DeviceTensor,
    _rotation_matrix: &DeviceTensor,
    _centroids: &DeviceTensor,
    _bits: u8,
    _packed_out: &DeviceTensor,
    _norms_out: &DeviceTensor,
) -> Result<()> {
    Err(crate::FractureError::Backend(
        "TurboQuant compression not supported by this backend".into(),
    ))
}

/// Paged attention with TurboQuant-compressed KV blocks.
///
/// q: [N, num_q_heads, head_dim] FP16
/// block_table: physical block IDs for quantized blocks
/// k_packed / v_packed: per-block packed index tensors
/// k_norms / v_norms: per-block norm tensors
/// k_rotation / v_rotation: per-layer rotation matrices (distinct — K for query
///   pre-rotation, V for output unrotation)
/// k_centroids / v_centroids: centroid lookup tables
/// Additional params match attention_paged.
fn attention_paged_tq(
    &self,
    _q: &DeviceTensor,
    _block_table: &[i32],
    _k_packed: &[&DeviceTensor],
    _k_norms: &[&DeviceTensor],
    _v_packed: &[&DeviceTensor],
    _v_norms: &[&DeviceTensor],
    _k_rotation: &DeviceTensor,
    _v_rotation: &DeviceTensor,
    _k_centroids: &DeviceTensor,
    _v_centroids: &DeviceTensor,
    _key_bits: u8,
    _value_bits: u8,
    _num_kv_heads: usize,
    _kv_len: usize,
    _start_pos: usize,
    _out: &DeviceTensor,
) -> Result<()> {
    Err(crate::FractureError::Backend(
        "TurboQuant paged attention not supported by this backend".into(),
    ))
}
```

Both methods have default error implementations, matching the existing `attention_paged` pattern. Backends opt in by overriding.

---

## CUDA Kernels

### turboquant_compress.cu

Fuses normalize → rotate → quantize → pack for a batch of KV vectors.

```
Grid:  (num_tokens, num_kv_heads)    — one block per (token, head)
Block: 128 threads

Per block (token t, head h):
  1. Load x[t, h, :] (head_dim FP16 values) → shared memory as FP32
  2. Compute ||x||₂ via parallel reduction → store to norms_out[t, h]
  3. Normalize: x /= (||x||₂ + 1e-8)   // epsilon guard for zero vectors
  4. Rotate: y = Pi @ x
     - Each thread computes head_dim/128 output elements
     - Pi is in global memory (read-only, L2-cached — only 64 KB per layer)
  5. Quantize: for each y[i], binary search centroids → index
     - 16 centroids (4-bit): 4 comparisons
     - 4 centroids (2-bit): 2 comparisons
  6. Pack: combine indices into bytes, write to packed_out[t, packed_dim_offset]
```

**Performance:** The rotation (step 4) is a matrix-vector multiply, O(d²) per vector. For d=128, this is 16K FMAs — well within a single block's compute budget. The centroids fit in shared memory or registers.

### attention_paged_tq.cu

Fuses decompress → attention for quantized blocks. This is the critical kernel.

```
Grid:  (num_tokens, num_q_heads)     — one block per (token, q_head)
Block: 128 threads
Shared memory: kv_len * sizeof(float) + head_dim * sizeof(float) + centroids

Per block (token t, query head qh):
  kv_head = qh / group_size

  Phase 1: Compute attention scores + find max
    for kv_pos in [0, kv_len) strided by 128:
      // Decompress K[kv_pos, kv_head] on the fly:
      a. Load packed indices for this (kv_pos, kv_head) from k_packed[block_table[kv_pos/16]]
      b. Unpack bits → centroid indices
      c. Lookup centroids → rotated coordinates y_hat
      d. Unrotate: k_hat = Pi^T @ y_hat  (matmul in registers/shared)
      e. Rescale: k_hat *= norm[kv_pos, kv_head]
      f. Compute dot(q[t, qh], k_hat) / sqrt(head_dim) → score
      g. Track running max for stable softmax

  Phase 2: Softmax (exp + sum, same as standard kernel)

  Phase 3: Weighted V sum
    for kv_pos in [0, kv_len) strided by 128:
      // Decompress V[kv_pos, kv_head] on the fly (same as K but v_bits, v_centroids)
      a-e. Same decompress pipeline
      f. Accumulate: out += prob[kv_pos] * v_hat
```

**Optimization — skip full unrotation:** The attention score is `dot(q, Pi^T @ y_hat * norm)`. By linearity: `dot(q, Pi^T @ y_hat) * norm = dot(Pi @ q, y_hat) * norm`. Pre-rotating the query by Pi avoids per-KV-position unrotation entirely:

```
Optimized Phase 1:
  scale = 1.0 / sqrt(head_dim)
  q_rot = Pi_k @ q[t, qh]   // Rotate query ONCE with K rotation (head_dim² FMAs)
  for kv_pos:
    Unpack → lookup centroids → y_hat   // NO unrotation needed
    score = dot(q_rot, y_hat) * norm * scale   // Scaled dot in rotated space
```

This reduces per-KV-position work from O(d²) (unrotation) to O(d) (dot product). The single query rotation is amortized over all kv_len positions.

**For V summation**, the full unrotation cannot be avoided (we need the actual reconstructed vector, not just a dot product). However, V uses fewer bits (2-bit, 4 centroids), so decompression is cheaper.

**V unrotation batching:** Instead of unrotating each V independently, accumulate the weighted sum in rotated space and unrotate once at the end:

```
Optimized Phase 3:
  acc_rot = zeros(head_dim)       // Accumulate in V's rotated space
  for kv_pos:
    Unpack → lookup V centroids → y_hat_v
    acc_rot += prob[kv_pos] * y_hat_v * v_norm[kv_pos]
  out = Pi_v^T @ acc_rot          // Single unrotation with V rotation matrix
```

This reduces V unrotation from O(kv_len × d²) to O(d²) — a massive savings.

**Note:** K and V use different rotation matrices (different seeds per layer), so Pi_k and Pi_v are distinct.

### Shared Memory Budget

```
Standard attention_paged:
  scores: kv_len × 4 bytes
  reduction scratch: 64 × 4 bytes

TurboQuant attention_paged_tq:
  scores: kv_len × 4 bytes
  q_rot: head_dim × 4 bytes (rotated query, 512 bytes for d=128)
  centroids_k: 2^key_bits × 4 bytes (64 bytes for 4-bit)
  centroids_v: 2^value_bits × 4 bytes (16 bytes for 2-bit)
  acc_rot: head_dim × 4 bytes (accumulated rotated V, 512 bytes)
  reduction scratch: 64 × 4 bytes

Total additional: ~1.1 KB — negligible.
```

The rotation matrices Pi_k and Pi_v (128×128 FP32 = 64 KB each, 128 KB total per layer) are too large for shared memory. They remain in global memory (L2-cached). Each thread reads a row of Pi_k for the query rotation and a column of Pi_v^T for the final V unrotation. The L2 hit rate will be high since all thread blocks in a layer read the same Pi_k and Pi_v (~4 MB total for 32 layers fits comfortably in L2).

---

## Engine Integration

### Forward Pass Dispatch

In [batched.rs](crates/fracture-engine/src/batched.rs), the per-sequence attention dispatch gains a new branch:

```rust
// Existing: FP16 paged attention
match cache {
    KvCacheBackend::Paged(paged) => {
        // ... existing attention_paged call
    }
    KvCacheBackend::QuantizedPaged(qcache) => {
        // For quantized blocks: use attention_paged_tq
        // For FP16 residual window blocks: use attention_paged
        // Combine results (split attention across two ranges)
    }
    _ => { /* contiguous path */ }
}
```

### KV Append Path

When a new K/V is produced by the attention projection:

```rust
// QuantizedKvCacheManager::append_kv()
fn append_kv<B: Backend>(
    &mut self,
    handle: CacheHandle,
    layer: usize,
    keys: &DeviceTensor,    // [N, num_kv_heads, head_dim] FP16
    values: &DeviceTensor,
    backend: &B,
) -> Result<()> {
    let seq = self.sequences.get_mut(&handle.0)?;
    let is_protected = self.is_layer_protected(layer);

    if is_protected {
        // Protected layers: 8-bit quantized (same pipeline, higher-precision codebook)
        // Falls through to the direct compression path below with layer_key_bits=8, layer_value_bits=8
    }

    if self.config.residual_tokens > 0 && !is_protected {
        // Residual window mode: FP16 for recent, compress when window shifts
        self.residual_append(seq, layer, keys, values, backend)?;
    } else {
        // Direct compression: compress and store immediately
        let k_rotation = &self.pool.k_rotation_matrices[layer];
        let v_rotation = &self.pool.v_rotation_matrices[layer];
        let layer_key_bits = self.pool.key_bits_for_layer(layer);
        let layer_value_bits = self.pool.value_bits_for_layer(layer);
        let k_centroids = self.pool.centroids(layer_key_bits);
        let v_centroids = self.pool.centroids(layer_value_bits);

        // Use pre-allocated scratch tensors (avoid per-call cudaMalloc overhead)
        let k_packed = &self.compress_scratch.k_packed;
        let k_norms = &self.compress_scratch.k_norms;
        backend.turboquant_compress(keys, k_rotation, k_centroids, layer_key_bits, k_packed, k_norms)?;

        let v_packed = &self.compress_scratch.v_packed;
        let v_norms = &self.compress_scratch.v_norms;
        backend.turboquant_compress(values, v_rotation, v_centroids, layer_value_bits, v_packed, v_norms)?;

        // Copy into quantized block pool
        self.pool.write_quantized(seq, layer, k_packed, k_norms, v_packed, v_norms, backend)?;
    }
    Ok(())
}
```

### Split Attention (Quantized + FP16 Residual)

When a sequence has both quantized blocks and FP16 residual window blocks, attention is computed in two passes:

```
Pass 1: attention_paged_tq over quantized blocks → partial_scores_q, partial_out_q
Pass 2: attention_paged over FP16 blocks → partial_scores_f, partial_out_f
Merge:  Combine via log-sum-exp to get correct softmax normalization
```

This "split softmax" is a well-known technique (used in FlashAttention's multi-pass approach). Each pass returns the local max and sum-of-exp alongside the weighted output. The merge corrects the normalization:

```
max_combined = max(max_q, max_f)
sum_combined = sum_q * exp(max_q - max_combined) + sum_f * exp(max_f - max_combined)
out = (out_q * sum_q * exp(max_q - max_combined) + out_f * sum_f * exp(max_f - max_combined)) / sum_combined
```

This requires the attention kernels to output auxiliary `(max, sum_exp)` per (token, head) alongside the attention output. A new Backend trait method or kernel variant handles this.

**Simplification for `residual_tokens == 0`:** No split needed — all blocks are quantized, single-pass `attention_paged_tq`.

---

## Memory Budget

### Llama 3.1 8B on RTX 3090 (24 GB VRAM)

```
Weights: 15.3 GB
Scratch reserve: 512 MB
Compress scratch: ~64 KB (negligible — pre-allocated for max single-token compress)
Available for KV cache: 24 - 15.3 - 0.5 = 8.2 GB
Rotation matrices: 32 layers × 2 (K+V) × 128 × 128 × 4 = 4 MB (negligible)
Codebooks: ~400 bytes (negligible — one table per distinct bit-width)

FP16 paged (current):
  bytes_per_block = 16 × 8 × 128 × 2 × 2 × 32 = 2,097,152 bytes = 2 MB
  blocks = 8.2 GB / 2 MB = 4,096 blocks = 65,536 tokens

TurboQuant K4/V2, no protected layers:
  Per-layer per-block: 16 tokens × 8 heads × ((128 × key_bits/8 + 2) + (128 × val_bits/8 + 2))
  K: 16 × 8 × (64 + 2) = 8,448 bytes/layer
  V: 16 × 8 × (32 + 2) = 4,352 bytes/layer
  Total per block: (8,448 + 4,352) × 32 layers = 409,600 bytes ≈ 400 KB
  blocks = 8.2 GB / 400 KB = 20,480 blocks = 327,680 tokens
  Effective compression: 5.0x more tokens in the same memory

TurboQuant K4/V2, protected_layers=4 (8 protected layers use 8-bit K+V):
  compute_num_blocks() sums per-layer sizes:
    Protected layer (8-bit K, 8-bit V): 16 × 8 × ((128 + 2) + (128 + 2)) = 33,280 bytes
    Normal layer (4-bit K, 2-bit V): 16 × 8 × ((64 + 2) + (32 + 2)) = 12,800 bytes
    8 protected + 24 normal: 8 × 33,280 + 24 × 12,800 = 573,440 bytes ≈ 560 KB/block
  blocks = 8.2 GB / 560 KB = 14,628 blocks = 234,048 tokens
  Effective compression: 3.6x (matches reference benchmark)
```

### RTX 5090 (32 GB VRAM)

```
Available for KV cache: 32 - 15.3 - 0.5 = 16.2 GB

FP16: 8,192 blocks = 131,072 tokens
TurboQuant K4/V2 (no protection): 40,960 blocks = 655,360 tokens
TurboQuant K4/V2 (protected_layers=4): 29,257 blocks = 468,114 tokens
```

---

## Distributed Pipeline Compatibility

### Wire Protocol

No new message types needed. The existing `BatchedForward` (0x0C) and `BatchedForwardResult` (0x0D) messages carry raw activation tensors between pipeline stages — these are always FP16 regardless of KV cache compression. TurboQuant is local to each worker's KV cache.

### Worker Configuration

Workers receive TurboQuant config during registration (or via CLI flags). Each worker independently:
1. Generates rotation matrices for its assigned layers (deterministic from seed + layer_idx)
2. Computes Lloyd-Max codebooks for the configured bit widths
3. Allocates `QuantizedBlockPool` instead of `BlockPool`

### Heartbeat / Admission Control

`HeartbeatAckPayload.free_blocks` already reports available blocks. With TurboQuant, each block holds the same 16 tokens but uses less memory, so `compute_num_blocks()` yields more blocks from the same GPU memory. The coordinator's admission control (`PeerRegistry::min_free_blocks()`) works unchanged — it's token-capacity-aware via block count.

---

## A/B Testing Infrastructure

### CLI Interface

```
fracture-server-cuda \
  --model /path/to/model.gguf \
  --kv-quant turboquant \           # Enable TurboQuant (default: none = FP16)
  --tq-key-bits 4 \                 # Key quantization bits (default: 4)
  --tq-value-bits 2 \               # Value quantization bits (default: 2)
  --tq-protected-layers 0 \         # Protected edge layers (default: 0)
  --tq-protected-bits 8 \           # Bit width for protected layers (default: 8)
  --tq-residual-tokens 0 \          # FP16 residual window (default: 0)
  --tq-seed 42                      # Rotation matrix seed (default: 42)
```

Without `--kv-quant`, the server uses the existing FP16 paged path. No code path changes.

### Validation Tests

**Tier 1 — Kernel correctness:**
- `turboquant_compress`: Compress known FP16 vectors, decompress (via separate test kernel), verify round-trip error matches Lloyd-Max theoretical distortion.
- `attention_paged_tq`: Compare against `attention_paged` on the same KV data. Compress FP16 blocks, run TQ attention, verify cosine similarity > 0.999 with FP16 attention output.

**Tier 2 — End-to-end quality:**
- Generate 256 tokens with greedy decoding on FP16 path. Generate 256 tokens with TurboQuant K4/V2. Compare:
  - Token-level: expect identical tokens for at least 200/256 (minor divergence acceptable from quantization noise).
  - Perplexity: measure on a held-out prompt set, expect < 0.1% perplexity increase.

**Tier 3 — Memory validation:**
- Allocate TurboQuant cache, fill to capacity, verify block count matches theoretical `compute_num_blocks()`.
- Run concurrent sequences to verify no OOM until theoretical limit.

**Tier 4 — A/B throughput:**
- Benchmark tokens/second with FP16 vs TurboQuant under identical load.
- TurboQuant has higher per-token compute (compression + decompression) but higher batch capacity (more sequences fit in memory). Net throughput should improve under high concurrency.

---

## Lloyd-Max Codebook Computation

The codebooks are computed at startup (< 1ms). The algorithm:

```
For dimension d and bit-width b:
  n_levels = 2^b
  sigma = 1 / sqrt(d)
  Initialize centroids uniformly in [-3.5*sigma, 3.5*sigma]

  Repeat until convergence:
    1. Boundaries = midpoints between adjacent centroids
    2. For each partition [boundary_i, boundary_{i+1}]:
       centroid_i = E[X | X in partition] = integral(x * pdf(x)) / integral(pdf(x))
       where pdf(x) = N(0, sigma^2) = Gaussian approximation for d >= 64

  Output: centroids (2^b floats), boundaries (2^b - 1 floats)
```

For practical purposes, the codebooks for common (d, b) pairs can be hardcoded as const arrays, avoiding numerical integration at runtime:

```rust
// Precomputed for d=128 (Llama 3.1 8B head_dim)
const CENTROIDS_2BIT_D128: [f32; 4] = [-0.1326, -0.0443, 0.0443, 0.1326];
const CENTROIDS_3BIT_D128: [f32; 8] = [/* ... computed offline ... */];
const CENTROIDS_4BIT_D128: [f32; 16] = [/* ... computed offline ... */];
const CENTROIDS_8BIT_D128: [f32; 256] = [/* ... computed offline ... */];
```

For other head dimensions, compute at startup using the Gaussian approximation N(0, 1/d) (no scipy needed — the integrals are simple Gaussian CDFs computable with `erfc`). The centroids scale with `1/sqrt(d)`, so using the wrong dimension's codebook would produce significantly higher distortion.

**Dimension guard:** At `QuantizedBlockPool` construction, assert that const codebooks match the model's `head_dim`. If `head_dim` does not match any precomputed table, fall back to runtime Lloyd-Max computation and log a warning:

```rust
let centroids = match PRECOMPUTED_CODEBOOKS.get(&(head_dim, bits)) {
    Some(table) => table.clone(),
    None => {
        tracing::warn!(
            "no precomputed TurboQuant codebook for head_dim={head_dim}, bits={bits}; \
             computing at startup (this is fine but adds ~10ms)"
        );
        compute_lloyd_max_codebook(head_dim, bits)
    }
};
```

---

## Crate Structure

```
crates/fracture-core/src/
  turboquant.rs          # TurboQuantConfig, codebook computation, const tables

crates/fracture-engine/src/
  quantized_paged_kv_cache.rs   # QuantizedBlockPool, QuantizedKvCacheManager
  engine.rs                      # KvCacheBackend::QuantizedPaged variant
  batched.rs                     # Dispatch branch for quantized attention

backends/fracture-cuda/
  kernels/
    turboquant_compress.cu       # Rotation + quantization + packing kernel
    turboquant_decompress.cu     # Unpacking + dequantization + unrotation (test utility)
    attention_paged_tq.cu        # Fused quantized paged attention
  src/
    backend.rs                   # turboquant_compress, attention_paged_tq impls
    ffi.rs                       # FFI bindings for new kernels
```

No new crates. TurboQuant is an enhancement to existing crates, behind the `KvCacheBackend` enum.

---

## Implementation Order

| Step | Name | Depends On | Description |
|---|---|---|---|
| 4.5a | Core types + codebooks | Nothing | `TurboQuantConfig`, Lloyd-Max solver in Rust, const codebook tables for d=128 |
| 4.5b | Compression kernel | 4.5a | `turboquant_compress.cu`: rotate + quantize + pack. Unit tests against known vectors |
| 4.5c | Decompression kernel | 4.5b | `turboquant_decompress.cu`: unpack + dequantize + unrotate. Round-trip validation |
| 4.5d | Quantized block pool | 4.5a | `QuantizedBlockPool`, `QuantizedKvCacheManager`, block alloc/free/append |
| 4.5e | Quantized attention kernel | 4.5b, 4.5c | `attention_paged_tq.cu`: fused decompress + attention with query pre-rotation optimization |
| 4.5f | Backend trait + CUDA impl | 4.5b, 4.5e | `turboquant_compress`, `attention_paged_tq` in Backend trait + CudaBackend |
| 4.5g | Engine integration | 4.5d, 4.5f | `KvCacheBackend::QuantizedPaged`, dispatch in `batched.rs` and `engine.rs` |
| 4.5h | CLI + server wiring | 4.5g | `--kv-quant turboquant` flag, config propagation to cache manager |
| 4.5i | A/B validation | 4.5h | Tier 1-4 tests, quality comparison against FP16 baseline |
| 4.5j | Distributed support | 4.5h | Worker `--kv-quant` flag, quantized cache init in worker startup |

**Critical path:** 4.5a → 4.5b → 4.5e → 4.5f → 4.5g → 4.5h → 4.5i

Steps 4.5b/4.5c (kernels) and 4.5d (Rust cache manager) can proceed in parallel after 4.5a.

---

## Anticipated Challenges

### Rotation Matrix Memory Access Pattern

The query pre-rotation (`Pi @ q`) reads an entire row of Pi per output element. For head_dim=128, each thread reads 128 FP32 values from global memory. With 128 threads per block computing the full rotation, this is 128 × 128 × 4 = 64 KB of reads. The L2 cache (40-80 MB on modern GPUs) will hold all 32 layers' rotation matrices (~4 MB total), so this should be fast after the first access.

### Bit Packing/Unpacking in CUDA

Packing 4-bit indices: two indices per byte, straightforward shift-and-mask. Packing 2-bit indices: four indices per byte. Packing 3-bit indices: awkward — 8 indices per 3 bytes. Recommend avoiding 3-bit and supporting only 2-bit and 4-bit for the initial implementation. 3-bit can be added later with 4-bit storage and wasted capacity (simpler than true 3-bit packing).

### Split Attention Correctness

The log-sum-exp merge for split attention (quantized + FP16 residual) must be numerically precise. The merge formula requires the attention kernels to output max and sum-of-exp as auxiliary values. This adds complexity to both the quantized and FP16 kernels (new output tensors) and a merge kernel. **Recommendation:** Start with `residual_tokens = 0` (no split) and add residual window support as a follow-up.

### Lloyd-Max for Non-128 Head Dims

The const codebook tables are precomputed for d=128 (Llama family). Other models with different head_dim need runtime codebook computation. The `QuantizedBlockPool` constructor asserts that codebooks match the model's head_dim and falls back to runtime computation with a warning if no precomputed table exists. The Gaussian approximation makes runtime computation straightforward — the Lloyd-Max algorithm converges in < 100 iterations with simple numerical integration (~10ms at startup).

### Compress Scratch Tensor Sizing

The `CompressScratch` tensors are pre-allocated for the maximum single-step append (determined by `max_prefill_tokens`, default 512). The `packed_dim` used for allocation must be the **maximum** across all layers — i.e., the protected layer packed_dim (8-bit: `num_kv_heads * head_dim`) rather than the normal layer packed_dim (4-bit: `num_kv_heads * head_dim / 2`). Layers with smaller packed_dim simply use a prefix of the scratch buffer.

---

## What This Does NOT Include (Deferred)

- **QJL (Stage 2):** Dropped based on community validation. MSE-only V3 outperforms.
- **3-bit packing:** Only 2-bit and 4-bit supported initially. 3-bit uses 4-bit storage.
- **Residual window:** Deferred to follow-up. Initial implementation compresses all tokens.
- **Per-layer adaptive bits at runtime:** All non-protected layers use the same key_bits/value_bits. Dynamic adaptation based on layer sensitivity analysis is future work.
- **Metal backend TurboQuant:** Metal gets TurboQuant when the Metal backend (Phase 5) is implemented.
- **Weight quantization:** TurboQuant applies to KV cache only, not model weights.

---

## Success Criteria

1. `turboquant_compress` kernel produces correct packed indices matching reference Python implementation
2. `attention_paged_tq` output has cosine similarity > 0.999 vs FP16 `attention_paged` on same KV data
3. End-to-end greedy decode with TurboQuant K4/V2 matches FP16 output for >= 200/256 tokens
4. `QuantizedBlockPool` allocates 5x more blocks than `BlockPool` from the same memory budget
5. A/B test: `--kv-quant turboquant` vs default FP16 produces comparable output quality
6. No changes to existing FP16 code paths — all TurboQuant logic is additive
7. Distributed workers support `--kv-quant turboquant` with no protocol changes
