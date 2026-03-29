# Fracture Phase 4: Architecture Document
## Production Inference

**Depends on:** Phase 3 complete and validated (distributed inference working across machines)
**Goal:** Transform the distributed inference engine from one-request-at-a-time into a production system that serves multiple concurrent requests through continuous batching and paged KV cache.

---

## What Changes from Phase 3

Phase 3 proved correctness: two machines cooperate to run inference on a model that neither could handle alone, producing byte-identical output to single-node execution. But it processes one request at a time — the coordinator holds a Mutex during the entire generation (prefill + all decode steps), blocking all other requests. A 200-token generation at 30 tok/s takes ~7 seconds, during which every other request is queued.

Phase 4 removes that bottleneck.

| Component | Phase 3 | Phase 4 |
|---|---|---|
| Request handling | Mutex-serialized, one at a time | Continuous batching: requests enter/leave dynamically |
| KV cache allocation | Contiguous per-sequence (max_seq_len pre-allocated) | Paged blocks allocated on demand, freed individually |
| KV cache memory per sequence | 512 MB (4096 tokens × 32 layers, reserved upfront) | Grows from ~2 MB (first block) as tokens are generated |
| Max concurrent sequences | 1 | Limited by GPU memory (dozens to hundreds) |
| GPU utilization during decode | 1/N for N pipeline stages (~33% with 3 workers) | Near-100% with micro-batching |
| Worker failure | Active sequence dies, pipeline hangs | Affected sequences aborted, pipeline rebuilt for new requests |

---

## The Problem: Memory Waste and Pipeline Bubbles

### Memory Waste (Contiguous KV Cache)

On a 32GB RTX 5090 with Llama 3 8B (15.3 GB weights):

```
Available for KV cache: 32 GB - 15.3 GB - 1 GB overhead ≈ 15.7 GB

Phase 3 contiguous allocation (max_seq_len = 4096):
  Per layer per sequence:
    K: [4096, 8, 128] × 2 bytes = 8 MB
    V: [4096, 8, 128] × 2 bytes = 8 MB
    Per layer: 16 MB
  All 32 layers: 512 MB per sequence

  Max concurrent sequences: 15700 / 512 ≈ 30
```

But a typical request generates 50-200 tokens. A 50-token sequence allocates 512 MB of cache and uses 6.4 MB — wasting 98.75% of its allocation. With 30 concurrent 50-token requests, the GPU allocates 15.36 GB of cache holding 192 MB of actual data.

This matters because with continuous batching, we want to run as many sequences concurrently as possible to maximize GPU utilization. Every wasted MB of cache is a sequence that can't be admitted.

### Pipeline Bubbles (Sequential Execution)

In Phase 3, a single request flows through the pipeline sequentially:

```
Time →  ─── 1 decode step ───────────────────────────
Worker 0 (Head):   [Compute    ]
Worker 1 (Middle):              [Compute    ]
Worker 2 (Tail):                             [Compute    ]

Worker 0 utilization: 33% (idle while 1 and 2 work)
Worker 1 utilization: 33%
Worker 2 utilization: 33%
```

Each worker is idle (N-1)/N of the time. With 3 workers, 67% of GPU compute capacity is wasted on every decode step.

---

## Component 1: Paged KV Cache

### PagedAttention: Block Tables Instead of Contiguous Buffers

Replace the per-sequence contiguous buffer with a global pool of fixed-size blocks. Each sequence gets a **block table** — a list mapping logical block indices to physical block IDs in the pool.

```
Block Pool (GPU memory, pre-allocated at startup):
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ Block 5 │... │
│ 16 tok  │ 16 tok  │ 16 tok  │ 16 tok  │ 16 tok  │ 16 tok  │    │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴────┘

Sequence A (32 tokens, block_table = [0, 3]):
  Tokens 0-15  → Block 0
  Tokens 16-31 → Block 3

Sequence B (20 tokens, block_table = [1, 4]):
  Tokens 0-15  → Block 1
  Tokens 16-19 → Block 4 (partially filled, 12 slots unused)

Sequence C (5 tokens, block_table = [2]):
  Tokens 0-4   → Block 2 (partially filled, 11 slots unused)

Free list: [5, 6, 7, ...]
```

### Block Size Analysis

Each block stores K and V data for `block_size` tokens across one layer:

```
Per block, per layer (Llama 3 8B, FP16):
  K: [block_size, 8, 128] × 2 bytes
  V: [block_size, 8, 128] × 2 bytes

With block_size = 16:
  K per block per layer: 16 × 8 × 128 × 2 = 32,768 bytes = 32 KB
  V per block per layer: same = 32 KB
  Total per block per layer: 64 KB

  Total per block (all 32 layers): 64 KB × 32 = 2 MB
```

Why 16 tokens per block (hardcoded, not configurable — the attention kernel is tuned for this size):
- **Fragmentation:** Maximum waste is 15 tokens per sequence = 15 × 128 KB/token = 1.92 MB. Acceptable.
- **Block table size:** 4096-token context = 256 block table entries. At 4 bytes each = 1 KB per sequence. Negligible.
- **GPU alignment:** 32 KB (K data per block per layer) is a multiple of the 128-byte GPU cache line. Memory coalescing preserved.
- **Attention kernel:** Each block is contiguous — the inner loop of the attention kernel reads 16 sequential token positions with full coalescing, then jumps to the next block.

### Memory Budget (Paged)

```
Available GPU memory for blocks: ~15.7 GB (same as before)
Block size: 2 MB per block (all 32 layers)
Total blocks: 15700 / 2 ≈ 7850 blocks

Capacity in tokens: 7850 × 16 = 125,600 tokens across all sequences

Examples:
  50-token sequence:  ceil(50/16) = 4 blocks = 8 MB    (vs 512 MB contiguous — 64× less)
  200-token sequence: ceil(200/16) = 13 blocks = 26 MB  (vs 512 MB — 20× less)
  512-token sequence: 32 blocks = 64 MB                  (vs 512 MB — 8× less)
  4096-token sequence: 256 blocks = 512 MB               (same as contiguous — fully utilized)

Max concurrent sequences at various lengths:
  50 tokens:   15700 / 8 ≈ 1962 sequences
  200 tokens:  15700 / 26 ≈ 603 sequences
  512 tokens:  15700 / 64 ≈ 245 sequences
  4096 tokens: 15700 / 512 ≈ 30 sequences (same as contiguous)
```

The paged approach matches contiguous allocation for long sequences and dramatically improves utilization for the common case of short-to-medium requests.

### Distributed Paging

In distributed inference, each worker manages its own block pool for its assigned layers. A worker running layers [0, 16) has blocks sized for 16 layers, not 32:

```
Worker 0 (layers 0-15):
  Block size: 64 KB × 16 layers = 1 MB per block
  Available: ~14 GB → 14,000 blocks → 224,000 tokens

Worker 1 (layers 16-31):
  Block size: 64 KB × 16 layers = 1 MB per block
  Available: ~14 GB → 14,000 blocks → 224,000 tokens
```

The coordinator tracks the global block count across all workers. A sequence can only be admitted if every worker in the pipeline has enough free blocks. The constraining worker limits the total.

### Block Pool Manager Interface

```rust
/// Pre-allocated pool of KV cache blocks on GPU memory.
struct BlockPool {
    /// Per-block, per-layer GPU tensors. Indexed as blocks[block_id][layer_idx].
    /// Each tensor is [block_size, num_kv_heads, head_dim] FP16.
    k_blocks: Vec<Vec<DeviceTensor>>,  // k_blocks[block_id][layer]
    v_blocks: Vec<Vec<DeviceTensor>>,  // v_blocks[block_id][layer]

    /// Stack of free block IDs. Pop to allocate, push to free.
    free_list: Vec<usize>,

    block_size: usize,     // tokens per block (16)
    num_layers: usize,     // layers this pool covers
    num_kv_heads: usize,   // 8 for Llama 3
    head_dim: usize,       // 128 for Llama 3
}

/// Per-sequence block allocation state.
struct SequenceBlocks {
    /// Logical block index → physical block_id.
    /// block_table[i] holds tokens [i*block_size .. (i+1)*block_size).
    block_table: Vec<usize>,

    /// How many tokens are stored in the last block.
    /// Range: 1..=block_size. When equal to block_size, next append allocates a new block.
    last_block_fill: usize,

    /// Total tokens stored across all blocks.
    current_len: usize,
}
```

### PagedKvCacheManager Interface

The external interface stays close to the Phase 3 `KvCacheManager`. The engine calls `append` and retrieval methods without knowing about blocks.

```rust
struct PagedKvCacheManager {
    pool: BlockPool,
    sequences: HashMap<u64, SequenceBlocks>,  // keyed by CacheHandle ID
    next_id: u64,
}

impl PagedKvCacheManager {
    /// Create the block pool. Allocates all GPU memory upfront.
    ///
    /// total_gpu_bytes: how much memory to dedicate to the block pool.
    /// Computed as: available_memory - model_weights - safety_margin.
    fn new(
        total_gpu_bytes: usize,
        block_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        backend: &impl Backend,
    ) -> Result<Self>;

    /// Allocate initial block for a new sequence.
    /// Returns error if no free blocks available.
    fn alloc(&mut self) -> Result<CacheHandle>;

    /// Append K and V tensors for new positions.
    ///
    /// During prefill: positions has N entries (the full prompt).
    /// During decode: positions has 1 entry (the new token).
    ///
    /// Internally: writes into the current block's remaining slots.
    /// When the current block is full, allocates a new block from the free list.
    /// Returns OutOfMemory if the free list is empty when a new block is needed.
    fn append_kv(
        &mut self,
        handle: CacheHandle,
        layer: usize,
        keys: &DeviceTensor,    // [num_new_tokens, num_kv_heads, head_dim]
        values: &DeviceTensor,  // same shape
        backend: &impl Backend,
    ) -> Result<()>;

    /// Get the block table for a sequence (for paged attention).
    fn block_table(&self, handle: CacheHandle) -> Result<&[usize]>;

    /// Get the current sequence length (total tokens stored).
    fn seq_len(&self, handle: CacheHandle) -> Result<usize>;

    /// Get the fill level of the last block (needed by attention kernel
    /// to know how many valid tokens are in the final block).
    fn last_block_tokens(&self, handle: CacheHandle) -> Result<usize>;

    /// Free all blocks for a sequence, returning them to the free list.
    fn free(&mut self, handle: CacheHandle) -> Result<()>;

    /// Number of free blocks available.
    fn num_free_blocks(&self) -> usize;

    /// Estimated number of tokens that can still be stored across all sequences.
    fn available_token_capacity(&self) -> usize {
        self.num_free_blocks() * self.pool.block_size
    }
}
```

### Attention Kernel Changes

The attention kernel is the only compute kernel that changes. All other kernels (matmul, RMSNorm, RoPE, SiLU, embedding, add) operate on activation tensors and are unaffected by how KV data is stored.

**Phase 3 attention signature:**
```rust
fn attention(
    &self,
    q: &DeviceTensor,            // [N, num_q_heads, head_dim]
    k_cache: &DeviceTensor,      // [max_seq_len, num_kv_heads, head_dim] (contiguous)
    v_cache: &DeviceTensor,      // same
    num_kv_heads: usize,
    start_pos: usize,            // seq_len to attend over
    out: &DeviceTensor,
) -> Result<()>;
```

**Phase 4 paged attention signature:**
```rust
fn attention_paged(
    &self,
    q: &DeviceTensor,            // [N, num_q_heads, head_dim]
    block_table: &[usize],       // physical block IDs for this sequence
    seq_len: usize,              // total tokens across all blocks
    last_block_tokens: usize,    // valid tokens in final block (1..=block_size)
    num_kv_heads: usize,
    layer: usize,                // which layer's blocks to read
    out: &DeviceTensor,
) -> Result<()>;
```

The kernel receives the block table (copied to GPU as a small integer array) and iterates over blocks instead of a contiguous range. The paged attention kernel is written from scratch rather than adapted from an existing implementation (e.g., vLLM) — this follows the project's convention of implementing GPU kernels from first principles for learning purposes.

**Paged attention kernel pseudocode (decode, single query token):**
```
// One thread block per Q head.
// Each thread block computes attention for one head over all KV blocks.

__global__ void paged_attention_decode(
    float16* q,              // [num_q_heads, head_dim]
    int* block_table,        // [num_blocks]
    int num_blocks,
    int seq_len,
    int last_block_tokens,   // valid tokens in final block
    float16** k_block_ptrs,  // base pointers for K blocks [block_id][layer]
    float16** v_block_ptrs,  // base pointers for V blocks
    int layer,
    int block_size,          // 16
    float16* output          // [num_q_heads, head_dim]
) {
    int head = blockIdx.x;
    int kv_head = head / group_size;  // GQA mapping

    float max_score = -INFINITY;
    float score_sum = 0.0;
    float output_acc[HEAD_DIM] = {0};

    // Phase 1: compute attention scores over all blocks
    for (int b = 0; b < num_blocks; b++) {
        int block_id = block_table[b];
        int tokens_in_block = (b == num_blocks - 1) ? last_block_tokens : block_size;

        float16* k_ptr = k_block_ptrs[block_id * num_layers + layer];
        // k_ptr points to [block_size, num_kv_heads, head_dim]

        for (int t = 0; t < tokens_in_block; t++) {
            // Compute Q·K^T / sqrt(d) for this token
            float score = dot(q[head], k_ptr[t * num_kv_heads + kv_head]) * rsqrt_d;

            // Online softmax: track running max and sum
            float old_max = max_score;
            max_score = max(max_score, score);
            float exp_delta = exp(old_max - max_score);
            score_sum = score_sum * exp_delta + exp(score - max_score);

            // Accumulate weighted V
            float weight = exp(score - max_score);
            float16* v_ptr = v_block_ptrs[block_id * num_layers + layer];
            for (int d = 0; d < HEAD_DIM; d++) {
                output_acc[d] = output_acc[d] * exp_delta + weight * v_ptr[t * num_kv_heads + kv_head + d];
            }
        }
    }

    // Phase 2: normalize accumulated output
    for (int d = 0; d < HEAD_DIM; d++) {
        output[head * HEAD_DIM + d] = output_acc[d] / score_sum;
    }
}
```

The key insight: within each block, tokens are contiguous in memory, so the inner loop has good coalescing. The outer loop over blocks adds an indirection (block table lookup) but blocks are large enough (16 tokens) that the overhead is amortized.

**Prefill attention** is more complex because multiple query tokens attend to multiple key positions simultaneously. The approach for prefill is:

1. For the query's own prompt tokens (which are in the new blocks being appended), use the standard contiguous attention path — the new tokens haven't been scattered across blocks yet.
2. For prior tokens (in existing blocks from a chunked prefill), use the paged path.
3. In the common case (single-shot prefill, no prior cache), all tokens are in newly-appended contiguous blocks, so the prefill attention is identical to Phase 3.

### Keeping the Contiguous Path

The Phase 3 contiguous `KvCacheManager` is not deleted. Both implementations coexist behind the same enum, selected at startup. The contiguous path remains the default until paged attention is fully validated against reference outputs.

- A/B correctness validation: run identical prompts through both paths, compare token sequences
- Fallback if paging bugs surface during batching development
- The contiguous path may eventually be removed, but not until paged attention has been validated end-to-end on the full Llama 3 8B model with greedy golden output comparison

```rust
enum KvCacheBackend {
    Contiguous(KvCacheManager),
    Paged(PagedKvCacheManager),
}
```

---

## Component 2: Continuous Batching

### Overview

Continuous batching (also called iteration-level batching) processes multiple sequences in each forward pass. Sequences enter the batch when they begin (prefill) and leave when they finish (EOS or max_tokens). The batch composition changes every iteration.

This is fundamentally different from static batching (pad all sequences to the same length, process together). Static batching wastes compute on padding tokens and can't add new requests until the entire batch finishes. Continuous batching avoids both problems.

### Batched Forward Pass

The engine processes a batch by concatenating all sequences' tokens into a single tensor. For QKV projections, FFN, and other per-token operations, this is just a larger matmul. For attention, each sequence uses its own KV cache (block table).

**Example batch:**
```
Active sequences this iteration:
  Seq A: decode step 47, token_id=1234, position=93
  Seq B: decode step 12, token_id=5678, position=27
  Seq C: prefill, token_ids=[101,102,103,...,228], positions=[0..128]

Concatenated batch:
  token_ids: [1234, 5678, 101, 102, 103, ..., 228]  (130 tokens total)
  positions:  [93,   27,   0,   1,   2,  ..., 127]
  seq_ids:    [A,    A,    C,   C,   C,  ..., C  ]  (for routing outputs back)

  Batch tensor: [130, 4096] after embedding
```

**What batches vs. what doesn't:**

| Operation | Batched? | Shape | Notes |
|---|---|---|---|
| Embedding lookup | Yes | [total_tokens, 4096] | All tokens in one gather |
| RMSNorm | Yes | [total_tokens, 4096] | Per-token, no cross-sequence interaction |
| QKV projection (matmul) | Yes | [total_tokens, 4096] × [4096, 4096] | Larger M dimension → better GPU utilization |
| RoPE | Yes | [total_tokens, num_heads, 128] | Uses per-token positions array |
| Attention | **No** | Per-sequence | Each sequence has its own block table and seq_len |
| Output projection (matmul) | Yes | [total_tokens, 4096] | Same as QKV |
| SwiGLU FFN (3 matmuls + silu) | Yes | [total_tokens, 14336] | Per-token, no cross-sequence interaction |
| LM head | Yes | [total_tokens, 128256] | But only decode tokens need sampling |

Attention is the only operation that's per-sequence. Everything else benefits from the larger batch size: larger matmuls mean better GPU utilization (more thread blocks, better memory bandwidth saturation).

### Batched Engine Interface

```rust
/// A single sequence's contribution to a batch.
struct SequenceSlice {
    seq_id: u64,
    /// Token IDs for this iteration.
    /// Prefill: full prompt (or chunk of prompt). Decode: single token.
    token_ids: Vec<u32>,
    /// Absolute positions for RoPE.
    positions: Vec<u32>,
    /// Block table for paged attention (physical block IDs).
    block_table: Vec<usize>,
    /// Total tokens in KV cache (including new tokens from this iteration).
    cache_seq_len: usize,
    /// Valid tokens in the last block.
    last_block_tokens: usize,
}

/// Input to a batched forward pass.
struct BatchedForwardInput {
    sequences: Vec<SequenceSlice>,
}

/// Output from a batched forward pass.
struct BatchedForwardOutput {
    /// Per-sequence logits. Only the last token's logits for each sequence
    /// are included (that's what sampling needs).
    /// For a decode step: 1 logit vector per sequence.
    /// For a prefill: 1 logit vector (the last prompt token's prediction).
    logits: Vec<(u64, Vec<f32>)>,  // (seq_id, logits)
}
```

**Forward pass with batching (pseudocode):**
```
fn batched_forward(input: BatchedForwardInput) -> BatchedForwardOutput:
    // 1. Concatenate all token IDs and positions
    all_token_ids = concat(seq.token_ids for seq in input.sequences)
    all_positions = concat(seq.positions for seq in input.sequences)
    total_tokens = len(all_token_ids)

    // 2. Track which tokens belong to which sequence (for splitting outputs)
    seq_boundaries = []  // (start_idx, end_idx, seq_id)
    offset = 0
    for seq in input.sequences:
        seq_boundaries.push((offset, offset + len(seq.token_ids), seq.seq_id))
        offset += len(seq.token_ids)

    // 3. Embedding lookup — batched
    hidden = embedding_table[all_token_ids]  // [total_tokens, 4096]

    // 4. For each layer:
    for layer in 0..num_layers:
        // RMSNorm — batched
        normed = rmsnorm(hidden, attn_norm[layer])

        // QKV projections — batched (large matmul = good GPU utilization)
        Q = normed @ Wq[layer].T   // [total_tokens, 4096]
        K = normed @ Wk[layer].T   // [total_tokens, 1024]
        V = normed @ Wv[layer].T   // [total_tokens, 1024]

        // RoPE — batched (uses per-token positions)
        apply_rope(Q, K, all_positions)

        // KV cache append — per-sequence
        for (start, end, seq_id) in seq_boundaries:
            cache.append(seq_id, layer, K[start:end], V[start:end])

        // Attention — PER-SEQUENCE (each has its own block table)
        attn_out = zeros(total_tokens, 4096)
        for (start, end, seq_id) in seq_boundaries:
            seq = get_sequence(seq_id)
            attn_out[start:end] = paged_attention(
                Q[start:end], seq.block_table, seq.cache_seq_len,
                seq.last_block_tokens, layer
            )

        // Output projection — batched
        projected = attn_out @ Wo[layer].T

        // Residual + FFN — batched
        hidden = hidden + projected
        normed = rmsnorm(hidden, ffn_norm[layer])
        gate = normed @ W_gate[layer].T
        up = normed @ W_up[layer].T
        hidden = hidden + (silu(gate) * up) @ W_down[layer].T

    // 5. Final norm + LM head — batched, but only extract last token per sequence
    hidden = rmsnorm(hidden, output_norm)
    all_logits = hidden @ lm_head.T  // [total_tokens, 128256]

    // 6. Extract per-sequence logits (last token only)
    result = []
    for (start, end, seq_id) in seq_boundaries:
        result.push((seq_id, all_logits[end - 1]))  // last token's logits

    return BatchedForwardOutput { logits: result }
```

### Scheduler: Iteration-Level Batching

The coordinator runs a **batch scheduler** that decides which sequences to include in each pipeline iteration.

```rust
struct BatchScheduler {
    /// Requests waiting for their first prefill.
    prefill_queue: VecDeque<PendingRequest>,

    /// Sequences actively generating (have been prefilled, in decode phase).
    active_sequences: HashMap<u64, ActiveSequence>,

    /// Maximum sequences in a single batch.
    max_batch_size: usize,

    /// Maximum total tokens in a single forward pass (memory + compute bound).
    max_batch_tokens: usize,

    /// Maximum tokens to prefill in a single iteration (limits prefill impact on decode latency).
    max_prefill_tokens: usize,
}

struct PendingRequest {
    seq_id: u64,
    prompt_tokens: Vec<u32>,
    config: GenerationConfig,
    token_tx: mpsc::UnboundedSender<GenerationEvent>,
    enqueued_at: Instant,
}

struct ActiveSequence {
    seq_id: u64,
    config: GenerationConfig,
    current_pos: usize,
    generated_tokens: Vec<u32>,
    token_tx: mpsc::UnboundedSender<GenerationEvent>,
    /// Remaining prompt tokens for chunked prefill (empty if fully prefilled).
    remaining_prefill: Vec<u32>,
}

struct SchedulerDecision {
    /// Sequences to prefill this iteration (new requests or continued chunks).
    prefills: Vec<PrefillJob>,
    /// Sequences to decode this iteration (one token each).
    decodes: Vec<DecodeJob>,
    /// Total tokens in this batch (sum of all prefill tokens + number of decodes).
    total_tokens: usize,
}
```

**Scheduling algorithm (runs each iteration):**

```
fn schedule() -> SchedulerDecision:
    decision = SchedulerDecision::empty()

    // 1. Always include all active decodes (cheap — 1 token each).
    //    Skip sequences that are paused for chunked prefill.
    for seq in active_sequences where seq.remaining_prefill.is_empty():
        decision.decodes.push(DecodeJob { seq_id: seq.seq_id })
        decision.total_tokens += 1
        if decision.decodes.len() >= max_batch_size:
            return decision

    // 2. Continue chunked prefills for sequences that started but aren't done.
    for seq in active_sequences where !seq.remaining_prefill.is_empty():
        chunk_size = min(max_prefill_tokens, seq.remaining_prefill.len())
        if decision.total_tokens + chunk_size > max_batch_tokens:
            chunk_size = max_batch_tokens - decision.total_tokens
        if chunk_size == 0:
            break
        decision.prefills.push(PrefillJob {
            seq_id: seq.seq_id,
            tokens: seq.remaining_prefill[..chunk_size],
        })
        decision.total_tokens += chunk_size

    // 3. Admit new requests from the prefill queue (if capacity remains).
    while !prefill_queue.is_empty():
        req = prefill_queue.front()

        // Check memory: can we allocate blocks for this sequence?
        estimated_blocks = ceil(req.prompt_tokens.len() / block_size)
        if cache.num_free_blocks() < estimated_blocks:
            break  // no memory for new sequences

        // Check batch capacity
        chunk_size = min(max_prefill_tokens, req.prompt_tokens.len())
        if decision.total_tokens + chunk_size > max_batch_tokens:
            break
        if decision.prefills.len() + decision.decodes.len() >= max_batch_size:
            break

        req = prefill_queue.pop_front()
        cache.alloc(req.seq_id)  // allocate first block

        if chunk_size < req.prompt_tokens.len():
            // Chunked prefill: process first chunk, save rest
            decision.prefills.push(PrefillJob {
                seq_id: req.seq_id,
                tokens: req.prompt_tokens[..chunk_size],
            })
            active_sequences.insert(req.seq_id, ActiveSequence {
                remaining_prefill: req.prompt_tokens[chunk_size..],
                ...
            })
        else:
            // Full prefill in one shot
            decision.prefills.push(PrefillJob {
                seq_id: req.seq_id,
                tokens: req.prompt_tokens,
            })
            active_sequences.insert(req.seq_id, ActiveSequence {
                remaining_prefill: vec![],
                ...
            })

        decision.total_tokens += chunk_size

    return decision
```

### Prefill Chunking

A 2048-token prefill processes 2048× more tokens than a single decode step. Without chunking, a large prefill monopolizes an entire batch iteration, adding 50-100ms of latency to every active decode sequence's next token.

**Policy:** Split prefills into chunks of at most `max_prefill_tokens` (configurable, default: 512). A 2048-token prompt takes 4 iterations to fully prefill. During those iterations, active decode sequences continue generating tokens normally — they're included in the same batch. Operators can tune this via CLI flag (`--max-prefill-tokens`) to trade TTFT against decode latency for their workload.

The sequence transitions to decode after its entire prompt is prefilled. Until then, it doesn't produce any tokens.

```
Iteration 1: [Seq A: decode] [Seq B: decode] [Seq C: prefill chunk 0-511]
Iteration 2: [Seq A: decode] [Seq B: decode] [Seq C: prefill chunk 512-1023]
Iteration 3: [Seq A: decode] [Seq B: decode] [Seq C: prefill chunk 1024-1535]
Iteration 4: [Seq A: decode] [Seq B: decode] [Seq C: prefill chunk 1536-2047]
Iteration 5: [Seq A: decode] [Seq B: decode] [Seq C: decode]  ← C now active
```

---

## Component 3: Request Lifecycle

### Phase 3 (Current)

```
HTTP Request → Mutex lock → Prefill → Decode loop → Response → Mutex unlock
                (blocks until generation is complete)
```

Non-streaming: entire response buffered, returned as JSON.
Streaming: SSE events sent during generation, but the Mutex is still held.

### Phase 4

```
HTTP Request → Validate → Enqueue in prefill_queue → Return SSE stream handle
                                    │
                                    ▼
                           ┌──────────────────┐
                           │  Scheduler Loop   │  (dedicated tokio task)
                           │  runs every       │
                           │  iteration        │
                           └───────┬──────────┘
                                   │
    ┌──────────────────────────────┼──────────────────────────────┐
    │ 1. Build batch (decodes + prefills)                         │
    │ 2. Send batched Forward to pipeline                         │
    │ 3. Receive logits                                           │
    │ 4. Sample token per sequence                                │
    │ 5. Send tokens to SSE channels ──────────────── → Client A  │
    │ 6. Check stop conditions                        → Client B  │
    │ 7. Free completed sequences                     → Client C  │
    │ 8. Loop                                                     │
    └─────────────────────────────────────────────────────────────┘
```

The HTTP handler no longer blocks. It validates the request, enqueues it, and returns a streaming response connected to a channel. The scheduler loop is the sole consumer of the pipeline.

```rust
/// Sent from the HTTP handler to the scheduler.
struct InFlightRequest {
    seq_id: u64,
    prompt_tokens: Vec<u32>,
    config: GenerationConfig,
    /// Sender for streaming tokens back to the HTTP response.
    token_tx: mpsc::UnboundedSender<GenerationEvent>,
}

/// Events sent from the scheduler to the HTTP response stream.
enum GenerationEvent {
    /// A new token was generated.
    Token(u32),
    /// Generation finished. Includes the stop reason and final token count.
    Finished { stop_reason: StopReason, completion_tokens: usize },
    /// Generation failed mid-stream.
    Error(String),
}
```

**Non-streaming requests** create a oneshot channel instead. The scheduler collects all tokens into a `Vec<u32>` and sends the completed response through the oneshot when the sequence finishes. The HTTP handler awaits the oneshot.

**Cancellation** works the same as Phase 3: the `token_tx` channel's receiver is dropped when the client disconnects, and the scheduler detects the closed channel (`.send()` returns `Err`) and marks the sequence for cleanup.

### Admission Control

The scheduler must balance three pressures:

1. **Decode latency:** Active sequences need a token every iteration. Adding more sequences to the batch slows each iteration (larger matmuls, more attention work).
2. **Prefill throughput:** New requests should start generating quickly (low TTFT). But prefills are expensive and compete with decodes for batch capacity.
3. **Memory:** Each new sequence needs block pool capacity. Admitting too many sequences exhausts the pool, preventing existing sequences from growing.

**Tuning parameters:**

| Parameter | Default | CLI Flag | Description |
|---|---|---|---|
| `max_batch_size` | 64 | `--max-batch-size` | Maximum sequences in a batch |
| `max_batch_tokens` | 4096 | `--max-batch-tokens` | Maximum total tokens per iteration |
| `max_prefill_tokens` | 512 | `--max-prefill-tokens` | Maximum prefill tokens per iteration (tradeoff: higher = faster TTFT, lower = steadier decode latency) |
| `block_pool_reserve` | 10% | `--block-pool-reserve` | Reserve this fraction of blocks for active sequence growth |

The `block_pool_reserve` prevents a failure mode where all blocks are allocated to new prefills, leaving no room for active sequences to grow during decode. With 10% reserve: if the pool has 7850 blocks, 785 are reserved. New sequences are only admitted if `free_blocks - needed_blocks > reserved_blocks`.

---

## Component 4: Fault Tolerance

### Worker Failure Detection

Phase 3 already implements heartbeat monitoring with nonce-validated acks (5s interval, 3 missed = dead). Phase 4 extends this to handle failures gracefully rather than fatally.

### Failure Response

When a worker is detected as dead:

```
1. ABORT AFFECTED SEQUENCES
   For each active sequence:
     If the dead worker is in the pipeline:
       Send GenerationEvent::Error("worker failure") to the sequence's token_tx
       Free blocks on surviving workers (send CacheFree)
       Remove from active_sequences

2. ATTEMPT PIPELINE REBUILD
   remaining_workers = all workers except dead one
   new_assignments = scheduler.schedule(model_config, remaining_workers)

   If new_assignments succeeds:
     Send new RegisterAck to surviving workers with updated layer ranges
     Workers reload weights for their new layer ranges
     Rebuild DistributedPipeline with new worker set
     Resume accepting new requests (existing sequences already aborted)
     Log: "Pipeline rebuilt with {N-1} workers. {M} sequences aborted."

   If new_assignments fails (insufficient memory/compute):
     Enter degraded state: reject new requests with 503 Service Unavailable
     Log: "Pipeline cannot serve model with remaining workers. Waiting for reconnection."

3. AWAIT RECONNECTION
   Dead worker may restart and re-register
   When a new Register is received:
     Re-run scheduler with full worker set
     Rebuild pipeline
     Exit degraded state
```

### What Phase 4 Does NOT Do for Fault Tolerance

- **KV cache checkpointing:** No saving/restoring KV cache state across worker restarts. Affected sequences are lost.
- **Transparent failover:** No hiding the failure from clients. They get an error and must retry.
- **Coordinator redundancy:** Coordinator is single point of failure. Its death kills everything.
- **Partial pipeline serving:** Can't serve requests on a subset of layers. Either the full pipeline works or it doesn't.
- **Sequence migration:** When a worker dies, its sequences are aborted rather than migrated to surviving workers. Migration would require serializing KV cache blocks over the network and rebuilding block tables on the target worker — a viable future extension of the paged cache architecture but not worth the complexity in Phase 4.

These are all reasonable for the target use case (small cluster, dev/research workload). Sequence migration is the most natural future extension — the paged cache architecture makes it possible (blocks are self-contained, block tables can be rewritten), but the coordination logic and network transfer cost are non-trivial. Listed as a future possibility, not a Phase 4 goal.

---

## Component 5: Pipeline Micro-Batching

### The Problem

Even with continuous batching, the pipeline has bubbles. While worker 2 processes batch N, workers 0 and 1 are idle waiting for work:

```
Without micro-batching:
  Worker 0: [Batch1    ]            [Batch2    ]            [Batch3    ]
  Worker 1:             [Batch1    ]            [Batch2    ]            [Batch3    ]
  Worker 2:                         [Batch1    ]            [Batch2    ]
```

### The Solution

Send the next batch to an earlier stage before the current batch has finished the later stages. Each worker has a 1-deep input queue.

```
With micro-batching (steady state):
  Worker 0: [B1][B2][B3][B4][B5][B6]...
  Worker 1:     [B1][B2][B3][B4][B5]...
  Worker 2:         [B1][B2][B3][B4]...

  Pipeline fill time: 2 iterations (for 3 workers)
  Steady-state utilization: 100%
```

The coordinator manages this by tracking multiple in-flight batches. Each worker processes its current batch's Forward while the coordinator prepares the next batch with results from completed stages.

**Key constraint:** Each batch must use a distinct set of sequence IDs in the wire protocol frames so workers don't confuse responses. Since sequences are identified by `seq_id` and each sequence only appears in one batch per iteration, the existing wire protocol works without changes.

### Implementation

This is the final implementation step. The throughput gain is proportional to the number of pipeline stages (2× for 2 workers, 3× for 3 workers), and continuous batching already provides the majority of the improvement. However, implementing micro-batching is valuable as a learning exercise in pipeline scheduling and concurrent distributed coordination — understanding how to overlap stages translates to any system that pipelines work across machines.

---

## Distributed Batching

### Wire Protocol Changes

The wire protocol does not change structurally. Batched forward passes are sent as a single Forward message containing the concatenated batch tensor. The only addition is per-sequence metadata alongside the tensor data:

```rust
struct BatchedForwardPayload {
    /// Per-sequence metadata needed by the worker.
    sequences: Vec<SequenceMetadata>,
    /// Concatenated input tensor (token IDs or activations).
    tensor_data: Vec<u8>,
}

struct SequenceMetadata {
    seq_id: u64,
    /// Number of tokens for this sequence in this batch.
    num_tokens: usize,
    /// Absolute positions for RoPE.
    positions: Vec<u32>,
    /// Block table for paged attention.
    block_table: Vec<usize>,
    /// Total KV cache length (for attention).
    cache_seq_len: usize,
    /// Valid tokens in last block.
    last_block_tokens: usize,
}
```

This fits within the existing Forward (0x03) message type. The payload just carries richer metadata. Workers that receive batched Forward messages process the entire batch through their layer range and return batched activations or logits.

### Distributed Block Pool Coordination

Each worker manages its own block pool. The coordinator tracks capacity across all workers:

```
Coordinator block accounting:
  worker_0_free_blocks: 5000
  worker_1_free_blocks: 4800
  worker_2_free_blocks: 5200

  Effective free blocks = min(5000, 4800, 5200) = 4800
  (The constraining worker limits the system)
```

Block allocation is coordinated:
1. Coordinator decides to admit a new sequence
2. Sends CacheAlloc to all workers
3. Each worker allocates blocks from its local pool
4. If any worker fails (OOM), coordinator sends CacheFree rollback (already implemented in Phase 3)

Block growth during decode is implicit — each worker's `append_kv` allocates new blocks as needed from its local free list. The coordinator doesn't need to send explicit block allocation messages for each token.

### Worker Heartbeat Extension

Heartbeat messages now include block pool stats:

```rust
struct HeartbeatAckPayload {
    timestamp: u64,
    nonce: u64,
    gpu_memory_used: u64,
    gpu_memory_total: u64,
    free_blocks: u32,      // NEW: blocks available in this worker's pool
    active_sequences: u32, // NEW: sequences with allocated cache
}
```

The coordinator uses `free_blocks` from heartbeats to maintain its block accounting without explicit queries.

---

## Implementation Order

Phase 4 is large. Each step produces a working, testable system.

### Step 1: Paged KV Cache (Local Only)

Replace contiguous allocation with paged blocks in the single-node server. No batching — still one sequence at a time.

1. Implement `BlockPool` in `fracture-engine` — pre-allocate GPU memory blocks at startup
2. Implement `PagedKvCacheManager` — block table management, append with auto-grow, free returns blocks
3. Write `attention_paged` CUDA kernel — iterate over block table instead of contiguous range
4. Add `attention_paged` to the `Backend` trait alongside existing `attention` (both available)
5. Wire `PagedKvCacheManager` into the engine via the `KvCacheBackend` enum
6. Validate: greedy generation produces token-identical output to Phase 3
7. Benchmark: memory usage for 50, 200, 512, 4096 token sequences

**Validation strategy:** Run the same prompts through both contiguous and paged paths. Compare token sequences. Any divergence is a bug.

**Why local first:** The paged cache is the foundation. Getting it right before adding batching or distribution prevents debugging compound issues.

### Step 2: Batched Forward Pass (Local Only)

Add multi-sequence support to the engine. Still single-node, still synchronous.

1. Implement `BatchedForwardInput` / `BatchedForwardOutput` structs
2. Modify the engine's layer loop to handle concatenated batches (larger matmuls)
3. Attention dispatches per-sequence within the batch (loop over `seq_boundaries`)
4. Add a simple test harness that runs N sequences through the engine simultaneously
5. Validate: each sequence in the batch produces identical output to running it alone
6. Benchmark: throughput (tokens/sec) for batch sizes 1, 4, 8, 16, 32

### Step 3: Continuous Batching (Local Only)

Replace the Mutex-serialized server with async request handling.

1. Implement `BatchScheduler` with the decode-priority algorithm
2. Implement `GenerationEvent` channel-based streaming
3. Replace `spawn_blocking` + Mutex pattern with a dedicated scheduler loop (tokio task)
4. Add prefill chunking (`max_prefill_tokens`)
5. Wire SSE streaming to `GenerationEvent` channels
6. Validate: 10 concurrent HTTP requests produce correct, independent results
7. Benchmark: TTFT under load (1, 5, 10, 20 concurrent requests)
8. Benchmark: throughput (requests/sec) vs Phase 3

### Step 4: Distributed Batching

Extend batching to the distributed pipeline.

1. Add `BatchedForwardPayload` to the wire protocol
2. Modify the coordinator to send batched Forward messages with per-sequence metadata
3. Workers process batches using their local paged attention kernel
4. Each worker manages its own block pool; coordinator tracks global capacity
5. Extend heartbeat with block pool stats
6. Validate: distributed batched output matches local batched output
7. Benchmark: throughput with 2-3 workers under concurrent load

### Step 5: Fault Tolerance

1. Extend heartbeat failure path: abort affected sequences, send errors to SSE streams
2. Implement pipeline reconfiguration on worker death (re-run scheduler, redistribute layers)
3. Implement worker reconnection with re-scheduling
4. Test: kill a worker mid-generation, verify other sequences are unaffected, verify pipeline recovers

### Step 6: Pipeline Micro-Batching

Included as a learning exercise in pipeline scheduling, even though the throughput gain is modest for 2-3 worker setups. Understanding how to overlap pipeline stages is valuable for reasoning about distributed system performance.

1. Coordinator tracks multiple in-flight batches
2. Workers have a 1-deep input queue (process current batch while receiving next)
3. Measure utilization improvement with 2 and 3 workers
4. Compare measured vs theoretical speedup (should approach N× for N stages in steady state)

---

## What Phase 4 Does NOT Include (Deferred)

- **Sequence migration on worker failure** — Transferring KV cache blocks from a dead worker to survivors. The paged architecture makes this architecturally feasible (blocks are self-contained, block tables can be rewritten), but the coordination logic and network transfer cost are significant. Listed as the most natural future extension of Phase 4's fault tolerance.
- **Dynamic rebalancing** — Re-partitioning layers while inference is running requires live KV cache migration between workers. Hard distributed state problem. Deferred.
- **Mixed quantization** — Different precision per node. Requires dequantization kernels and quality calibration. Deferred to Phase 5 alongside Metal.
- **GPU-direct RDMA** — Bypassing CPU for activation transfer. Requires InfiniBand. Only matters at data center scale.
- **Speculative decoding** — Draft model predicts tokens verified by full model. Orthogonal to batching. Can be added independently.
- **Prefix caching** — Sharing KV cache blocks across requests with common prefixes (system prompts). Natural extension of paged cache. Deferred to Phase 4b.
- **Beam search** — Multiple candidate sequences per request. Requires per-beam block tables. Deferred.

---

## Success Criteria (Phase 4 Complete)

- [ ] Paged KV cache: greedy generation matches Phase 3 output token-for-token
- [ ] Paged memory: 50-token sequence uses < 20 MB KV cache (vs 512 MB contiguous)
- [ ] Batched forward: N sequences produce identical output to N sequential runs
- [ ] Continuous batching: 10 concurrent requests served correctly
- [ ] Throughput: >3× improvement over Phase 3 for concurrent request workload
- [ ] TTFT: first-token latency < 2× single-request latency under 10-request load
- [ ] Prefill chunking: large prompts don't stall active decode sequences
- [ ] Fault tolerance: worker death aborts only affected sequences; pipeline recovers for new requests
- [ ] Distributed batching: multi-machine pipeline serves batched requests correctly
- [ ] Block pool coordination: constraining worker correctly limits system-wide admission
