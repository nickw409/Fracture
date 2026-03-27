# Fracture Phase 1: Architecture Document
## Local GPU-Accelerated Inference Server

**Target model:** Llama 3 8B (GGUF format)  
**Runtime:** Rust + CUDA  
**Goal:** Load model, run inference on single GPU, serve OpenAI-compatible API

---

## Model Specifications (Llama 3 8B)

These numbers drive every design decision below. Reference them constantly.

| Parameter | Value |
|---|---|
| Hidden dimension (`d_model`) | 4096 |
| Number of layers | 32 |
| Attention heads (Q) | 32 |
| KV heads (GQA) | 8 |
| Head dimension (`d_head`) | 128 (= 4096 / 32) |
| GQA group size | 4 (= 32 Q heads / 8 KV heads) |
| FFN intermediate size | 14336 |
| Vocabulary size | 128256 |
| RoPE base frequency (θ) | 500000 |
| RMSNorm epsilon | 1e-5 |
| Max context length | 128K tokens (8K practical for Phase 1) |
| Activation function | SiLU (used in SwiGLU) |

Source: Llama 3.1 8B config from HuggingFace / Meta release.

---

## Memory Budget (FP16)

### Model Weights

| Component | Shape | Size (FP16) |
|---|---|---|
| Token embedding | [128256, 4096] | 1.00 GB |
| **Per transformer layer:** | | |
| → Q projection | [4096, 4096] | 32.0 MB |
| → K projection | [4096, 1024] | 8.0 MB |
| → V projection | [4096, 1024] | 8.0 MB |
| → Output projection | [4096, 4096] | 32.0 MB |
| → Gate projection (SwiGLU) | [4096, 14336] | 112.0 MB |
| → Up projection (SwiGLU) | [4096, 14336] | 112.0 MB |
| → Down projection (SwiGLU) | [14336, 4096] | 112.0 MB |
| → RMSNorm weights × 2 | [4096] × 2 | ~16 KB |
| **Layer subtotal** | | **~416 MB** |
| **32 layers total** | | **~13.3 GB** |
| Final RMSNorm | [4096] | ~8 KB |
| LM head | [128256, 4096] | 1.00 GB |
| **Total model weights** | | **~15.3 GB** |

**Implication:** FP16 Llama 3 8B fits on a 24GB GPU (RTX 3090/4090) with ~8.7GB remaining for KV cache, activations, and framework overhead. For a 16GB GPU, you need INT8 or INT4 quantization.

### KV Cache Per Token

| Component | Shape per token per layer | Size (FP16) |
|---|---|---|
| Key | [8 heads, 128 dim] = [1024] | 2 KB |
| Value | [8 heads, 128 dim] = [1024] | 2 KB |
| **Per layer total** | | **4 KB** |
| **All 32 layers** | | **128 KB** |

| Sequence length | Total KV cache |
|---|---|
| 512 tokens | 64 MB |
| 2048 tokens | 256 MB |
| 4096 tokens | 512 MB |
| 8192 tokens | 1.0 GB |

**Implication:** On a 24GB GPU with FP16 weights, you have ~8.7GB for KV cache. That supports ~69K tokens of context for a single sequence. Memory is not the bottleneck for single-request inference at Phase 1 — it becomes one with batching (Phase 4).

---

## Component Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     HTTP Server (axum)                     │
│              /v1/completions  /v1/chat/completions         │
│                  SSE streaming support                     │
└────────────────────────┬─────────────────────────────────┘
                         │ CompletionRequest
                         ▼
┌──────────────────────────────────────────────────────────┐
│                   Generation Loop                         │
│  tokenize → prefill → [decode loop] → detokenize         │
│  Owns: sampling params, stop conditions, token buffer     │
└──────┬───────────────────────────────────────┬───────────┘
       │ token_ids                              │ logits
       ▼                                        ▲
┌──────────────────────────────────────────────────────────┐
│                   Compute Engine                          │
│  forward(token_ids, seq_pos) → logits                    │
│  Operates on layer range [start, end) ← Phase 2 hook     │
│  Owns: weight tensors (GPU), kernel dispatch              │
└──────┬───────────────────────────────────────┬───────────┘
       │ cache_read/write                       │
       ▼                                        │
┌──────────────────────────────────────────────────────────┐
│                    KV Cache Manager                       │
│  alloc(seq_id) → cache_handle                            │
│  append(cache_handle, layer, K, V)                       │
│  get(cache_handle, layer) → (K_history, V_history)       │
│  free(seq_id)                                            │
│  Owns: GPU memory pool for KV storage                     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                    Weight Store                           │
│  load_gguf(path) → model_config + weight_map             │
│  get_layer_weights(layer_idx) → LayerWeights             │
│  Owns: GPU memory for model weights                       │
└──────────────────────────────────────────────────────────┘
```

### Why These Boundaries Matter

The **Compute Engine** accepts a layer range `[start, end)` even in Phase 1. In Phase 1, this is always `[0, 32)`. In Phase 2, a node might run `[0, 16)` and accept/return activation tensors instead of token IDs/logits. This abstraction costs nothing now and saves a rewrite later.

The **KV Cache Manager** is separate from the compute engine so that cache allocation policy can change independently. Phase 1 uses simple contiguous allocation. Phase 4 switches to paged allocation. The compute engine doesn't know or care.

The **Weight Store** handles format-specific parsing (GGUF) and presents a uniform interface. If you later support safetensors or add INT4 dequantization, only this module changes.

---

## Component 1: Weight Store

### Responsibilities
- Parse GGUF file header and metadata
- Extract model config (layer count, dims, head counts, etc.)
- Map weight tensors to GPU memory with correct alignment
- Support FP16 weights initially; design for quantized weight dequantization later

### GGUF Format Overview
GGUF is a binary format storing model metadata + weight tensors sequentially. Structure:
1. Magic number + version
2. Metadata key-value pairs (architecture, dims, etc.)
3. Tensor info table (name, shape, dtype, offset)
4. Tensor data (contiguous block, aligned)

### Weight Naming Convention (Llama GGUF)
```
token_embd.weight                          → [128256, 4096]
blk.{i}.attn_q.weight                     → [4096, 4096]
blk.{i}.attn_k.weight                     → [1024, 4096]
blk.{i}.attn_v.weight                     → [1024, 4096]
blk.{i}.attn_output.weight                → [4096, 4096]
blk.{i}.ffn_gate.weight                   → [14336, 4096]   (SwiGLU gate)
blk.{i}.ffn_up.weight                     → [14336, 4096]   (SwiGLU up)
blk.{i}.ffn_down.weight                   → [4096, 14336]   (SwiGLU down)
blk.{i}.attn_norm.weight                  → [4096]
blk.{i}.ffn_norm.weight                   → [4096]
output_norm.weight                         → [4096]
output.weight                              → [128256, 4096]  (LM head)
```

### GPU Memory Layout
Weights should be allocated as a single contiguous block per layer for cache-friendly access:
```
[Layer 0 weights | Layer 1 weights | ... | Layer 31 weights | Embedding | LM Head]
```

Each layer's weights are ordered by access pattern in the forward pass:
```
[attn_norm | Q | K | V | O | ffn_norm | gate | up | down]
```

All weight tensors must be 256-byte aligned for efficient GPU memory access.

### Interface
```rust
struct ModelConfig {
    hidden_size: usize,        // 4096
    num_layers: usize,         // 32
    num_q_heads: usize,        // 32
    num_kv_heads: usize,       // 8
    head_dim: usize,           // 128
    intermediate_size: usize,  // 14336
    vocab_size: usize,         // 128256
    rope_theta: f32,           // 500000.0
    rms_norm_eps: f32,         // 1e-5
    max_seq_len: usize,        // 8192 (Phase 1 practical limit)
}

struct LayerWeights {
    attn_norm: DeviceTensor,    // [4096]
    wq: DeviceTensor,           // [4096, 4096]
    wk: DeviceTensor,           // [1024, 4096]  (8 KV heads × 128)
    wv: DeviceTensor,           // [1024, 4096]
    wo: DeviceTensor,           // [4096, 4096]
    ffn_norm: DeviceTensor,     // [4096]
    w_gate: DeviceTensor,       // [14336, 4096]
    w_up: DeviceTensor,         // [14336, 4096]
    w_down: DeviceTensor,       // [4096, 14336]
}

struct WeightStore {
    config: ModelConfig,
    embedding: DeviceTensor,       // [128256, 4096]
    layers: Vec<LayerWeights>,     // 32 layers
    output_norm: DeviceTensor,     // [4096]
    lm_head: DeviceTensor,         // [128256, 4096]
}
```

---

## Component 2: KV Cache Manager

### Design Principles
1. **Contiguous per-layer allocation in Phase 1.** Each sequence gets a pre-allocated buffer per layer sized to `max_seq_len`. Simple, wasteful, but correct.
2. **Abstract the interface now.** The compute engine calls `append()` and `get()`, never touches raw pointers. This lets Phase 4 swap in paged allocation.
3. **Sequence lifecycle.** Cache is allocated when a request arrives, grown during generation, freed when the request completes.

### Memory Layout (Phase 1 — Contiguous)
For a single sequence with `max_seq_len = 4096`:
```
Per layer, per sequence:
  K buffer: [max_seq_len, num_kv_heads, head_dim] = [4096, 8, 128] FP16 → 8 MB
  V buffer: [max_seq_len, num_kv_heads, head_dim] = [4096, 8, 128] FP16 → 8 MB
  Per layer total: 16 MB

All 32 layers: 512 MB per sequence
```

This is the over-reservation problem discussed in the research. For Phase 1 with single-request serving, it's acceptable. Phase 4 replaces this with paged blocks.

### Interface
```rust
struct CacheHandle(u64);  // opaque ID

trait KVCacheManager {
    /// Allocate cache for a new sequence
    fn alloc(&mut self, seq_id: u64, max_len: usize) -> CacheHandle;

    /// Append new K,V tensors at given positions for a layer
    /// During prefill: positions = [0..prompt_len]
    /// During decode: positions = [current_pos]
    fn append(
        &mut self,
        handle: CacheHandle,
        layer: usize,
        positions: &[usize],
        keys: &DeviceTensor,    // [num_positions, num_kv_heads, head_dim]
        values: &DeviceTensor,  // [num_positions, num_kv_heads, head_dim]
    );

    /// Get cached K,V up to seq_len for attention computation
    fn get(
        &self,
        handle: CacheHandle,
        layer: usize,
        seq_len: usize,
    ) -> (DeviceTensor, DeviceTensor);  // K: [seq_len, 8, 128], V: [seq_len, 8, 128]

    /// Free cache when sequence completes
    fn free(&mut self, handle: CacheHandle);
}
```

### Phase 2 Hook
The `CacheHandle` abstraction means a distributed node only caches its assigned layers. A node running layers [16, 32) would only allocate cache for those 16 layers. The interface doesn't change.

---

## Component 3: Compute Engine

### Forward Pass — Full Sequence (Prefill and Decode)

For a single token at position `pos` (decode step), or a batch of tokens at positions `[0..n]` (prefill):

```
Input: token_ids [seq_len]  (prefill) or [1] (decode)

1. EMBEDDING LOOKUP
   hidden = embedding_table[token_ids]          → [seq_len, 4096]

2. FOR EACH LAYER i = 0..31:

   2a. PRE-ATTENTION RMSNORM
       normed = rmsnorm(hidden, attn_norm[i])   → [seq_len, 4096]

   2b. QKV PROJECTIONS (matmul)
       Q = normed @ Wq[i].T                     → [seq_len, 4096]  (32 heads × 128)
       K = normed @ Wk[i].T                     → [seq_len, 1024]  (8 heads × 128)
       V = normed @ Wv[i].T                     → [seq_len, 1024]  (8 heads × 128)

   2c. RESHAPE FOR MULTI-HEAD
       Q = Q.reshape(seq_len, 32, 128)
       K = K.reshape(seq_len, 8, 128)
       V = V.reshape(seq_len, 8, 128)

   2d. APPLY ROPE TO Q AND K
       Q = apply_rope(Q, positions, theta=500000)
       K = apply_rope(K, positions, theta=500000)

   2e. KV CACHE UPDATE
       cache.append(handle, i, positions, K, V)
       K_full, V_full = cache.get(handle, i, current_seq_len)
       // K_full: [current_seq_len, 8, 128]
       // V_full: [current_seq_len, 8, 128]

   2f. GROUPED QUERY ATTENTION
       // Expand K,V heads to match Q heads (repeat each KV head 4x)
       // Q:      [seq_len, 32, 128]
       // K_full: [current_seq_len, 8, 128] → expand to [current_seq_len, 32, 128]
       // V_full: [current_seq_len, 8, 128] → expand to [current_seq_len, 32, 128]
       //
       // For each head h:
       //   scores = Q[h] @ K_full[h].T / sqrt(128)    → [seq_len, current_seq_len]
       //   Apply causal mask (only for prefill; decode naturally causal)
       //   probs = softmax(scores)
       //   attn_out[h] = probs @ V_full[h]            → [seq_len, 128]
       //
       attn_out = concat_heads(attn_out)                → [seq_len, 4096]

   2g. OUTPUT PROJECTION
       attn_out = attn_out @ Wo[i].T                   → [seq_len, 4096]

   2h. RESIDUAL CONNECTION
       hidden = hidden + attn_out

   2i. PRE-FFN RMSNORM
       normed = rmsnorm(hidden, ffn_norm[i])            → [seq_len, 4096]

   2j. SWIGLU FFN
       gate = normed @ W_gate[i].T                      → [seq_len, 14336]
       up   = normed @ W_up[i].T                        → [seq_len, 14336]
       ffn_out = silu(gate) * up                         → [seq_len, 14336]
       ffn_out = ffn_out @ W_down[i].T                   → [seq_len, 4096]

   2k. RESIDUAL CONNECTION
       hidden = hidden + ffn_out

3. FINAL RMSNORM
   hidden = rmsnorm(hidden, output_norm)                → [seq_len, 4096]

4. LM HEAD
   logits = hidden @ lm_head.T                          → [seq_len, 128256]
   // Only last position's logits used for next token prediction
   logits = logits[-1]                                   → [128256]

Output: logits [128256]
```

### CUDA Kernels Required

| Kernel | Input shapes | Output shape | Notes |
|---|---|---|---|
| `rmsnorm` | x:[N, 4096], w:[4096] | [N, 4096] | Fuse with eps. N=1 during decode. |
| `rope` | Q:[N, 32, 128], K:[N, 8, 128], positions | same | Sin/cos lookup table, pre-computed. |
| `matmul` (general) | A:[M, K], B:[K, N] | [M, N] | Use cuBLAS initially. 6 matmuls per layer. |
| `attention` | Q, K_cache, V_cache | [N, heads, d_head] | Most complex kernel. Fuse softmax. |
| `silu_mul` | gate:[N, 14336], up:[N, 14336] | [N, 14336] | Fuse SiLU activation with elementwise multiply. |
| `embedding_lookup` | ids:[N], table:[V, D] | [N, D] | Simple gather. |
| `softmax_sample` | logits:[V], temp, top_p, top_k | token_id | CPU-side is fine for Phase 1. |

**Kernel priority order for implementation:**
1. Use cuBLAS for all matmuls initially — don't write custom GEMM kernels yet
2. Write custom `rmsnorm` — simple, good warmup kernel
3. Write custom `rope` — pre-compute sin/cos table on init
4. Write custom `silu_mul` — trivial fused kernel
5. Write custom `attention` — hardest kernel, start with naive implementation
6. [Stretch] Replace naive attention with FlashAttention-style tiled kernel

### Phase 2 Hook
```rust
trait ComputeEngine {
    /// Run forward pass on a layer range
    /// Phase 1: layer_range = 0..32, input = token embeddings, output = logits
    /// Phase 2: layer_range = N..M, input = activation tensor, output = activation tensor
    fn forward(
        &self,
        input: &DeviceTensor,
        layer_range: Range<usize>,
        cache: &mut dyn KVCacheManager,
        cache_handle: CacheHandle,
        positions: &[usize],
    ) -> DeviceTensor;
}
```

When `layer_range.start > 0`, the input is an activation tensor `[seq_len, 4096]` rather than token IDs.  
When `layer_range.end < 32`, the output is an activation tensor `[seq_len, 4096]` rather than logits.

---

## Component 4: Generation Loop

### Tokenizer
Use an existing BPE tokenizer crate (e.g., `tokenizers` from HuggingFace, or `tiktoken-rs`). Llama 3 uses a tiktoken-based tokenizer with 128256 tokens. Not worth implementing from scratch.

### Sampling
```
Input: logits [128256], temperature, top_k, top_p

1. Apply temperature: logits = logits / temperature
2. Top-K: keep only the K highest logits, set rest to -inf
3. Top-P (nucleus): sort by probability, keep smallest set whose cumulative prob >= top_p
4. Softmax over remaining logits → probabilities
5. Sample from distribution (or argmax if temperature = 0)

Output: token_id
```

### Generation State Machine
```
IDLE → receive request
  → PREFILL: process all prompt tokens in one forward pass
    → DECODE: generate one token per forward pass
      → check stop condition (EOS token, max_tokens, stop strings)
        → if stop: COMPLETE → free cache → IDLE
        → if not: emit token via SSE, loop DECODE
```

### Prefill vs Decode — Key Differences

| | Prefill | Decode |
|---|---|---|
| Tokens processed | All prompt tokens (e.g., 512) | 1 new token |
| Input shape | [prompt_len, 4096] | [1, 4096] |
| Attention pattern | Causal mask over full prompt | New token attends to all cached + self |
| Bottleneck | Compute-bound (matrix-matrix) | Memory-bandwidth-bound (load all weights for 1 token) |
| KV cache | Populated for all prompt positions | Appended with 1 new position |

---

## Component 5: HTTP Server

### Endpoints

**POST /v1/completions**
```json
{
  "model": "llama-3-8b",
  "prompt": "The meaning of life is",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "stream": true
}
```

**POST /v1/chat/completions**
```json
{
  "model": "llama-3-8b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 256,
  "stream": true
}
```

Chat endpoint applies the Llama 3 chat template to convert messages into a prompt string, then delegates to the same generation pipeline.

### Streaming (SSE)
For `stream: true`, emit `data: {"choices": [{"delta": {"content": "token"}}]}` as each token is generated. The generation loop sends tokens through a `tokio::sync::mpsc` channel that the HTTP handler reads from.

### Implementation
- Framework: `axum` + `tokio`
- Single-request serving in Phase 1 (no concurrent inference)
- Request queue for Phase 4 (continuous batching)

---

## Data Flow Summary

```
HTTP Request
    │
    ▼
Tokenizer: "Hello world" → [15496, 1917]
    │
    ▼
Generation Loop: PREFILL
    │  token_ids=[15496, 1917], positions=[0, 1]
    ▼
Compute Engine: forward(embed(token_ids), layers 0..32)
    │  KV cache populated for positions 0,1
    │  Returns logits[128256]
    ▼
Sampling: logits → token_id=382
    │
    ▼
Generation Loop: DECODE (repeat)
    │  token_ids=[382], positions=[2]
    ▼
Compute Engine: forward(embed([382]), layers 0..32)
    │  KV cache appended at position 2
    │  Returns logits[128256]
    ▼
Sampling: logits → next token
    │
    ▼
Detokenize → stream to client via SSE
```

---

## Validation Strategy

### Per-Kernel Numerical Validation
For each CUDA kernel, compare output against PyTorch reference:
1. Load Llama 3 8B in PyTorch (HuggingFace transformers)
2. Extract intermediate tensors at each step (hook into model forward pass)
3. Feed identical inputs to your CUDA kernel
4. Assert outputs match within FP16 tolerance (rtol=1e-3, atol=1e-3)

Priority order for validation:
1. RMSNorm — simplest, validate first
2. RoPE — compare against HF's `apply_rotary_pos_emb`
3. Single-layer forward pass — Q,K,V projections through attention output
4. Full model forward pass — compare final logits
5. End-to-end generation — compare generated text (temperature=0 for determinism)

### Build a Test Harness Early
Create a Python script that:
1. Loads Llama 3 8B in PyTorch
2. Runs a forward pass on a test prompt
3. Dumps every intermediate tensor to disk (activations after each layer, Q/K/V values, attention scores, etc.)
4. Your Rust test suite loads these tensors and compares against your CUDA output

**This harness is your single most important debugging tool. Build it before writing any CUDA kernels.**

---

## Proposed File Structure

```
fracture/
├── Cargo.toml
├── crates/
│   ├── fracture-core/          # Shared types, Backend trait, DeviceTensor
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── tensor.rs       # DeviceTensor, TensorId, DType
│   │       ├── backend.rs      # Backend trait (all GPU ops go through this)
│   │       ├── config.rs       # ModelConfig
│   │       └── error.rs        # Error types
│   ├── fracture-gguf/          # GGUF parser (backend-agnostic, returns host buffers)
│   │   └── src/
│   ├── fracture-engine/        # Compute engine + KV cache (uses Backend trait, no CUDA imports)
│   │   └── src/
│   ├── fracture-generate/      # Generation loop, sampling, tokenizer
│   │   └── src/
│   └── fracture-server/        # HTTP server (axum)
│       └── src/
├── backends/
│   └── fracture-cuda/          # CUDA backend (implements Backend trait)
│       ├── src/
│       │   ├── lib.rs          # CudaBackend struct + Backend impl
│       │   ├── memory.rs       # CUDA memory pool, TensorId → CUdeviceptr map
│       │   ├── cublas.rs       # cuBLAS wrapper (row-major trick lives here)
│       │   └── kernels.rs      # CUDA kernel dispatch
│       └── kernels/            # .cu files
│           ├── rmsnorm.cu
│           ├── rope.cu
│           ├── attention.cu
│           ├── silu_mul.cu
│           └── embedding.cu
├── bins/
│   └── fracture-server-cuda/   # Binary: wires CudaBackend into engine + server
│       └── src/main.rs
├── tests/
│   ├── reference/              # PyTorch reference tensor dumps
│   └── validation/             # Per-kernel numerical comparison tests
└── scripts/
    └── dump_reference.py       # PyTorch script to generate reference tensors
```

Workspace-level Cargo.toml with separate crates per component. `fracture-engine`
depends on `fracture-core` (for the `Backend` trait) but NEVER on `fracture-cuda`.
Only the final binary in `bins/` selects the backend. This enforces that all engine
logic is backend-agnostic, keeping the door open for a Metal backend in Phase 5+.

---

## Backend Abstraction (Cross-Platform Design)

Fracture Phase 1-3 targets CUDA only. Phase 5+ will add a Metal backend for Apple
Silicon. All decisions made now must avoid locking the engine to CUDA.

### The Rule

**No crate outside of `backends/` may import or depend on any GPU-specific type.**
`fracture-engine`, `fracture-generate`, `fracture-server`, `fracture-protocol` —
none of these know CUDA exists. They operate entirely through the `Backend` trait
defined in `fracture-core`.

### DeviceTensor: An Opaque Handle

`DeviceTensor` is NOT a pointer to GPU memory. It is an opaque identifier that
the `Backend` implementation maps to whatever internal representation it uses.
The engine never dereferences it, never casts it, never does pointer arithmetic.

```rust
/// Opaque handle to a tensor on a compute device.
/// The engine holds these. Only the Backend knows what's inside.
#[derive(Clone, Debug)]
pub struct DeviceTensor {
    pub id: TensorId,         // opaque u64 assigned by the backend
    pub shape: Vec<usize>,    // known to the engine for shape validation
    pub dtype: DType,         // FP16, FP32, INT8, INT4
}

#[derive(Clone, Copy, Debug)]
pub struct TensorId(pub u64);

#[derive(Clone, Copy, Debug)]
pub enum DType {
    FP16,
    FP32,
    BF16,
    INT8,
    INT4,
}
```

### The Backend Trait

This is the complete interface between the engine and any GPU backend.
Every operation the engine needs is a method on this trait.

```rust
pub trait Backend: Send + Sync {
    // ── Memory Management ──────────────────────────────────────
    /// Allocate a tensor on the device. Returns an opaque handle.
    fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor>;

    /// Free a previously allocated tensor.
    fn free(&self, tensor: &DeviceTensor) -> Result<()>;

    /// Copy data from host (CPU) buffer to device tensor.
    fn copy_to_device(&self, dst: &DeviceTensor, src: &[u8]) -> Result<()>;

    /// Copy data from device tensor to host (CPU) buffer.
    fn copy_to_host(&self, src: &DeviceTensor, dst: &mut [u8]) -> Result<()>;

    // ── Matrix Multiplication ──────────────────────────────────
    /// C = A @ B. All tensors are row-major.
    /// A: [M, K], B: [K, N], C: [M, N]
    /// Backend handles precision internally (FP32 accumulation, etc.)
    fn matmul(
        &self,
        a: &DeviceTensor,
        b: &DeviceTensor,
        out: &DeviceTensor,
    ) -> Result<()>;

    // ── Transformer Kernels ────────────────────────────────────
    /// RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * weight
    fn rmsnorm(
        &self,
        x: &DeviceTensor,       // [N, hidden_dim]
        weight: &DeviceTensor,   // [hidden_dim]
        eps: f32,
        out: &DeviceTensor,      // [N, hidden_dim]
    ) -> Result<()>;

    /// Apply RoPE to Q and K tensors in-place.
    fn rope(
        &self,
        q: &DeviceTensor,       // [N, num_q_heads, head_dim]
        k: &DeviceTensor,       // [N, num_kv_heads, head_dim]
        positions: &[usize],
        theta: f32,
        head_dim: usize,
    ) -> Result<()>;

    /// Grouped-query attention.
    /// Returns attention output [N, num_q_heads, head_dim].
    fn attention(
        &self,
        q: &DeviceTensor,           // [N, num_q_heads, head_dim]
        k_cache: &DeviceTensor,     // [seq_len, num_kv_heads, head_dim]
        v_cache: &DeviceTensor,     // [seq_len, num_kv_heads, head_dim]
        out: &DeviceTensor,         // [N, num_q_heads, head_dim]
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<()>;

    /// Fused SiLU(gate) * up.
    fn silu_mul(
        &self,
        gate: &DeviceTensor,    // [N, intermediate_dim]
        up: &DeviceTensor,      // [N, intermediate_dim]
        out: &DeviceTensor,     // [N, intermediate_dim]
    ) -> Result<()>;

    /// Embedding lookup: out[i] = table[ids[i]]
    fn embedding(
        &self,
        ids: &[u32],
        table: &DeviceTensor,   // [vocab_size, hidden_dim]
        out: &DeviceTensor,     // [N, hidden_dim]
    ) -> Result<()>;

    /// Element-wise add: out = a + b
    fn add(
        &self,
        a: &DeviceTensor,
        b: &DeviceTensor,
        out: &DeviceTensor,
    ) -> Result<()>;

    // ── Cache Operations ───────────────────────────────────────
    /// Copy a slice of src into dst at the given position offset.
    /// Used for KV cache append.
    fn copy_slice(
        &self,
        src: &DeviceTensor,
        dst: &DeviceTensor,
        dst_offset: usize,     // position offset in first dimension
    ) -> Result<()>;

    // ── Device Info ────────────────────────────────────────────
    fn device_name(&self) -> String;
    fn total_memory(&self) -> usize;
    fn available_memory(&self) -> usize;
}
```

### How the Engine Uses It

The compute engine is generic over the backend:

```rust
pub struct ComputeEngine<B: Backend> {
    backend: B,
    weights: WeightStore,  // contains DeviceTensors allocated via B
    // ...
}

impl<B: Backend> ComputeEngine<B> {
    pub fn forward_layer(
        &self,
        hidden: &DeviceTensor,
        layer: usize,
        // ...
    ) -> Result<DeviceTensor> {
        let w = &self.weights.layers[layer];

        // RMSNorm — calls backend, not CUDA
        let normed = self.backend.alloc(&hidden.shape, hidden.dtype)?;
        self.backend.rmsnorm(hidden, &w.attn_norm, 1e-5, &normed)?;

        // Q projection — calls backend matmul, not cuBLAS
        let q = self.backend.alloc(&[seq_len, 4096], DType::FP16)?;
        self.backend.matmul(&normed, &w.wq, &q)?;

        // ... rest of the layer
    }
}
```

**The engine never imports `fracture-cuda`.** It imports `fracture-core::Backend`.
The binary in `bins/fracture-server-cuda/` instantiates `CudaBackend` and passes
it to the engine:

```rust
// bins/fracture-server-cuda/src/main.rs
use fracture_cuda::CudaBackend;
use fracture_engine::ComputeEngine;

fn main() {
    let backend = CudaBackend::new(gpu_device)?;
    let engine = ComputeEngine::new(backend, model_config, weights)?;
    // ... start server
}
```

### What This Enables for Phase 5+

Adding a Metal backend means:
1. Create `backends/fracture-metal/` implementing `Backend` for Apple Silicon
2. Create `bins/fracture-server-metal/` wiring it into the engine
3. Zero changes to `fracture-engine`, `fracture-generate`, `fracture-server`, or `fracture-protocol`

A mixed cluster (CUDA + Metal nodes) works because the wire protocol transfers
raw tensor bytes. The coordinator doesn't care what backend produced the bytes.
Node A (CUDA) sends an activation tensor. Node B (Metal) receives it and loads
it onto its device. The Backend trait handles the device-specific memory management
on each side.

### What This Does NOT Abstract

Some things are intentionally left backend-specific:

- **Kernel launch configuration** (block size, grid size, shared memory) — CUDA and Metal have fundamentally different execution models. The Backend implementation handles this internally.
- **Memory layout optimizations** — cuBLAS prefers column-major, Metal might not. The Backend's matmul implementation handles the translation from the row-major convention the engine uses.
- **Quantization kernels** — INT4 dequantization is hardware-specific. Each backend implements its own dequant strategy.
- **Performance tuning** — FlashAttention on CUDA vs Metal's MPS attention have completely different implementations. The Backend trait's `attention` method hides this.

### Dependency Graph

```
fracture-core          (Backend trait, DeviceTensor, ModelConfig)
    ↑
fracture-engine        (uses Backend trait generically)
    ↑
fracture-generate      (uses engine, no GPU awareness)
    ↑
fracture-server        (HTTP layer, no GPU awareness)

fracture-cuda          (implements Backend for CUDA)
    ↑
fracture-server-cuda   (binary: plugs CudaBackend into server)

[Future]
fracture-metal         (implements Backend for Metal)
    ↑
fracture-server-metal  (binary: plugs MetalBackend into server)
```

The critical invariant: **the arrow from engine to core points UP, not sideways
to any backend.** If `fracture-engine` ever imports `fracture-cuda`, the abstraction
is broken.

---

## Implementation Order

1. **`fracture-core`** — Define `DeviceTensor`, `Backend` trait, `ModelConfig`, error types
2. **`fracture-gguf`** — Parse GGUF files, return host-side tensors (no GPU dependency)
3. **`scripts/dump_reference.py`** — Generate reference tensors from PyTorch
4. **`fracture-cuda: CudaBackend`** — Implement memory management (alloc, free, copy)
5. **`fracture-cuda: rmsnorm`** — First kernel, validate against reference
6. **`fracture-cuda: rope`** — Pre-compute sin/cos table
7. **`fracture-cuda: silu_mul`** — Fused SiLU + elementwise multiply
8. **`fracture-cuda: attention`** — Naive attention first, FlashAttention later
9. **`fracture-cuda: matmul`** — cuBLAS wrapper with row-major transpose trick
10. **`fracture-engine`** — Wire Backend trait calls, single-layer forward pass, validate
11. **`fracture-engine`** — Full 32-layer forward pass, validate end-to-end logits
12. **`fracture-engine`** — KV cache manager (contiguous allocation)
13. **`fracture-generate`** — Tokenizer integration, sampling, generation loop
14. **`fracture-server`** — HTTP endpoints, SSE streaming
15. **`bins/fracture-server-cuda`** — Wire CudaBackend into engine + server
16. **Benchmark** — tokens/sec vs llama.cpp on same hardware
17. **[Stretch] FlashAttention kernel** — Replace naive attention with tiled implementation

---

## Open Design Questions

These need answers before or during implementation:

1. **cuBLAS vs custom matmul kernels?** Start with cuBLAS. Profile later to decide if custom kernels for the specific shapes (e.g., [1, 4096] × [4096, 14336] during decode) are worth it.

2. **Tensor memory format: row-major vs column-major?** cuBLAS expects column-major. Rust/C naturally use row-major. Need to pick one convention and transpose where needed. Recommendation: store weights in the format cuBLAS expects (column-major) to avoid transpose overhead in the hot path.

3. **FP16 throughout or FP32 accumulation?** cuBLAS supports FP16 input with FP32 accumulation (`CUBLAS_COMPUTE_32F`). Use this — it's the standard approach and prevents precision degradation in deep networks.

4. **Async weight loading?** For a 15GB model, loading from disk to GPU takes several seconds. Consider memory-mapping the GGUF file and streaming weights to GPU asynchronously. Not critical for Phase 1 but nice for UX.

5. **Which tokenizer crate?** `tokenizers` (HuggingFace) is the most complete but pulls in heavy dependencies. `tiktoken-rs` is lighter. Need to verify Llama 3's tokenizer is supported.

---

## Success Criteria (Phase 1 Complete)

- [ ] Load Llama 3 8B from GGUF file and serve on a single GPU
- [ ] Numerical output matches PyTorch reference (temperature=0, identical prompt)
- [ ] OpenAI-compatible /v1/completions and /v1/chat/completions endpoints working
- [ ] SSE streaming delivers tokens as they're generated
- [ ] Benchmarked tokens/sec on at least one GPU (documented)
- [ ] Code structured so Phase 2 (layer-range execution) requires config change, not rewrite
