# Fracture Phase 4: Architecture Document
## Production Inference

**Depends on:** Phase 3 complete and validated (distributed inference working across machines)
**Goal:** Transform the distributed inference engine from one-request-at-a-time into a production-grade system that serves multiple concurrent requests efficiently through continuous batching, paged KV cache, and fault tolerance.

---

## What Changes from Phase 3

Phase 3 proved correctness: two machines cooperate to run inference on a model that neither could handle alone, producing byte-identical output to single-node execution. But it processes one request at a time — the coordinator holds a Mutex during the entire generation (prefill + all decode steps), blocking all other requests. A 200-token generation at 30 tok/s takes ~7 seconds, during which the GPU pipeline is 100% occupied by one user.

Phase 4 removes that bottleneck.

| Component | Phase 3 | Phase 4 |
|---|---|---|
| Request handling | Mutex-serialized, one at a time | Continuous batching: requests enter/leave dynamically |
| KV cache allocation | Contiguous per-sequence (max_seq_len pre-allocated) | Paged blocks allocated on demand |
| KV cache memory | 512 MB per sequence (4096 tokens × 32 layers) | ~2 KB per token per layer, grows as needed |
| Max concurrent sequences | 1 | Limited by GPU memory (dozens to hundreds) |
| Worker failure | Active sequence dies | Active sequences on failed worker are aborted; new requests avoid the dead worker |
| Layer assignment | Fixed at startup | Fixed at startup (dynamic rebalancing deferred to Phase 4b) |
| Pipeline utilization | 0% between requests, 33% during decode (1 of 3 stages active) | Near-continuous: micro-batching hides pipeline bubbles |

---

## Memory Budget: Why Paging Matters

### Phase 3 (Contiguous Allocation)

On a 32GB GPU with Llama 3 8B (15.3 GB weights):

```
Available for KV cache: 32 GB - 15.3 GB - 1 GB overhead = ~15.7 GB

Per-sequence cache (max_seq_len = 4096):
  K buffer: [4096, 8, 128] × 32 layers × 2 bytes = 256 MB
  V buffer: same = 256 MB
  Total per sequence: 512 MB

Max concurrent sequences: 15700 / 512 ≈ 30
```

But Phase 3 allocates all 512 MB upfront even if the sequence only generates 50 tokens. A 50-token request wastes 98% of its allocation. With one-at-a-time serving this doesn't matter. With batching it's a disaster — 30 concurrent requests would exhaust GPU memory even though their actual KV data totals only ~15 MB.

### Phase 4 (Paged Allocation)

```
Block size: 16 tokens × 8 KV heads × 128 head_dim × 2 bytes × 2 (K+V) = 64 KB per block per layer
                                                                        = 2 MB per block (all 32 layers)

A 50-token sequence uses: ceil(50/16) = 4 blocks = 8 MB  (vs 512 MB contiguous)
Memory efficiency: 64× improvement for short sequences

Max concurrent sequences (50 tokens each): 15700 / 8 ≈ 1962
Max concurrent sequences (512 tokens each): 15700 / 64 ≈ 245
Max concurrent sequences (4096 tokens each): 15700 / 512 = 30  (same as contiguous — fully utilized)
```

The paged approach matches contiguous allocation's maximum capacity for long sequences while dramatically improving utilization for the common case of short-to-medium requests.

---

## Component 1: Paged KV Cache

### Design: PagedAttention Block Table

Replace the per-sequence contiguous buffer with a global pool of fixed-size KV blocks. Each sequence maintains a block table mapping logical token positions to physical block indices.

```
Block Pool (GPU memory):
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ Block 5 │... │
│ 16 tok  │ 16 tok  │ 16 tok  │ 16 tok  │ 16 tok  │ 16 tok  │    │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴────┘
                                    ▲         ▲
                                    │         │
Sequence A (32 tokens):    block_table = [0, 3]
Sequence B (20 tokens):    block_table = [1, 4]
Sequence C (5 tokens):     block_table = [2]
                                          ▲
Free list: [5, 6, 7, ...]
```

### Block Layout

Each block holds KV data for a fixed number of tokens across one layer:

```rust
/// A single KV block for one layer.
/// Stores key and value data for `block_size` token positions.
///
/// K data: [block_size, num_kv_heads, head_dim] FP16
/// V data: [block_size, num_kv_heads, head_dim] FP16
struct KvBlock {
    k: DeviceTensor,  // [block_size, 8, 128] = 32 KB for Llama 3
    v: DeviceTensor,  // [block_size, 8, 128] = 32 KB for Llama 3
}
```

Block size is a tuning parameter. 16 tokens is the standard choice (matches PagedAttention paper):
- Small enough to avoid fragmentation waste (max 15 tokens wasted per sequence)
- Large enough that the block table stays small (256 entries for 4096-token context)
- Aligned to GPU memory access patterns (cache lines, warp sizes)

### Block Pool Manager

```rust
struct BlockPool {
    /// All blocks in GPU memory, indexed by block_id.
    blocks: Vec<Vec<KvBlock>>,  // blocks[block_id][layer_idx]
    /// Free block IDs available for allocation.
    free_list: Vec<usize>,
    /// Total number of blocks in the pool.
    capacity: usize,
    /// Tokens per block.
    block_size: usize,
}

struct SequenceBlocks {
    /// Maps logical block index → physical block_id.
    /// block_table[i] holds tokens [i*block_size .. (i+1)*block_size).
    block_table: Vec<usize>,
    /// Number of tokens currently stored.
    current_len: usize,
}
```

### Interface Change

The KvCacheManager interface changes minimally. The compute engine still calls `append()` and retrieval methods — it doesn't know about blocks. The internal implementation changes from contiguous indexing to block table lookups.

```rust
impl PagedKvCacheManager {
    /// Allocate initial block for a new sequence.
    fn alloc(&mut self) -> Result<CacheHandle>;

    /// Append new KV data. Allocates new blocks as needed.
    /// Returns error if block pool is exhausted.
    fn append_kv(
        &mut self,
        handle: CacheHandle,
        layer: usize,
        keys: &DeviceTensor,
        values: &DeviceTensor,
        backend: &impl Backend,
    ) -> Result<()>;

    /// Get K and V for attention. Returns a list of block references
    /// rather than a single contiguous tensor.
    fn get_kv_blocks(
        &self,
        handle: CacheHandle,
        layer: usize,
    ) -> Result<&[usize]>;  // block IDs

    fn free(&mut self, handle: CacheHandle, backend: &impl Backend) -> Result<()>;

    fn num_free_blocks(&self) -> usize;
}
```

### Attention Kernel Changes

The attention kernel must change to read KV data from non-contiguous blocks. This is the core algorithmic change of PagedAttention.

**Phase 3 attention (contiguous):**
```
For each query position:
  K_history = contiguous_k_cache[0..seq_len]  // single pointer + stride
  V_history = contiguous_v_cache[0..seq_len]
  scores = Q @ K_history^T / sqrt(d)
  output = softmax(scores) @ V_history
```

**Phase 4 attention (paged):**
```
For each query position:
  For each block in block_table:
    K_block = block_pool[block_id].k[0..tokens_in_block]
    scores[block_start..block_end] = Q @ K_block^T / sqrt(d)
  output = softmax(all_scores) @ V_blocks  // gather from blocks
```

The CUDA kernel receives the block table as an additional parameter. The inner loop iterates over blocks instead of a contiguous range. Each block access is still contiguous within the block (the 16 tokens are sequential), so GPU memory coalescing is preserved.

```rust
fn attention_paged(
    &self,
    q: &DeviceTensor,           // [N, num_q_heads, head_dim]
    block_table: &[usize],      // physical block IDs for this sequence
    num_kv_heads: usize,
    seq_len: usize,             // total tokens across all blocks
    layer: usize,
    out: &DeviceTensor,
) -> Result<()>;
```

---

## Component 2: Continuous Batching

### The Pipeline Bubble Problem

In Phase 3's sequential execution, a single request flows through the pipeline:

```
Time →
Worker 0 (Head):   [Prefill][Decode1][Decode2][Decode3]...
Worker 1 (Middle):          [Prefill][Decode1][Decode2][Decode3]...
Worker 2 (Tail):                    [Prefill][Decode1][Decode2][Decode3]...
```

Each worker is idle 66% of the time (waiting for the previous/next stage). With continuous batching, multiple sequences flow through simultaneously:

```
Time →
Worker 0: [A:Pf][B:Pf][A:D1][B:D1][C:Pf][A:D2][B:D2][C:D1]...
Worker 1:       [A:Pf][B:Pf][A:D1][B:D1][C:Pf][A:D2][B:D2]...
Worker 2:             [A:Pf][B:Pf][A:D1][B:D1][C:Pf][A:D2]...
```

Pipeline utilization jumps from 33% to near-100% once the pipeline is full.

### Scheduler: Iteration-Level Batching

The coordinator runs a **batch scheduler** that decides which sequences to process in each pipeline iteration. This is the core of continuous batching.

```rust
struct BatchScheduler {
    /// Sequences waiting for prefill.
    prefill_queue: VecDeque<PendingRequest>,
    /// Sequences in the decode phase (active generation).
    active_sequences: HashMap<u64, ActiveSequence>,
    /// Maximum sequences to batch together in one iteration.
    max_batch_size: usize,
    /// Maximum total tokens in a single batch (memory bound).
    max_batch_tokens: usize,
}

struct SchedulerDecision {
    /// Sequences to prefill this iteration (new requests entering).
    prefills: Vec<PrefillJob>,
    /// Sequences to decode this iteration (one token each).
    decodes: Vec<DecodeJob>,
}
```

Each iteration, the scheduler:

1. **Collects decode candidates:** All active sequences that aren't finished.
2. **Checks for new prefills:** If there's capacity (memory + batch size), admit new requests from the prefill queue.
3. **Builds the batch:** Concatenate all token IDs (1 per decode + N per prefill) into a single batch.
4. **Sends to pipeline:** One Forward message per worker, carrying the full batch.
5. **Processes results:** Sample tokens for each sequence, check stop conditions, free completed sequences.

### Batched Forward Pass

The engine's forward pass changes to process multiple sequences simultaneously. Each sequence has its own position and KV cache, but they share the weight matrices.

```rust
struct BatchedForwardRequest {
    /// Per-sequence data for this batch.
    sequences: Vec<SequenceSlice>,
}

struct SequenceSlice {
    seq_id: u64,
    /// Token IDs for this sequence in this iteration.
    /// Prefill: full prompt. Decode: single token.
    token_ids: Vec<u32>,
    /// Positions for RoPE.
    positions: Vec<u32>,
    /// Block table for paged attention.
    block_table: Vec<usize>,
    /// Current sequence length (for attention masking).
    seq_len: usize,
}
```

The compute engine processes the batch as a single large matmul (concatenated hidden states) but uses per-sequence block tables for attention. This is the standard approach used by vLLM, TGI, and other production engines:

- **Matmul (QKV projections, FFN):** Batched — all sequences concatenated into one `[total_tokens, hidden]` tensor. GPU utilization scales linearly with batch size.
- **Attention:** Per-sequence — each sequence has its own KV cache (block table). The paged attention kernel iterates over sequences in the batch.
- **Sampling:** Per-sequence — each sequence has its own temperature, top-k, top-p, and stop conditions.

### Distributed Batched Pipeline

The coordinator sends batched Forward messages to each worker. Each worker:

1. Receives batched token IDs + per-sequence metadata (positions, block tables)
2. Runs the forward pass on the batch
3. Returns batched activations (or logits for the tail worker)

The activation tensor between stages is now `[total_batch_tokens, hidden_size]` rather than `[seq_len, hidden_size]`. The wire protocol doesn't change — it already handles arbitrary tensor shapes.

Cache management becomes per-sequence within the batch:
- **CacheAlloc** is sent when a new sequence enters the batch (not when the HTTP request arrives — admission is deferred until the scheduler decides to admit it)
- **CacheFree** is sent when a sequence completes (EOS or max_tokens)
- Block allocation/deallocation happens on every append (the coordinator tracks block tables and sends them with Forward messages)

---

## Component 3: Request Lifecycle

### Phase 3 (Current)

```
HTTP Request → Mutex lock → Prefill → Decode loop → Response → Mutex unlock
```

### Phase 4

```
HTTP Request → Enqueue in prefill_queue → Return immediately (streaming: open SSE)
                                                     │
Scheduler loop (runs every iteration):               │
  1. Pick sequences for this batch                   │
  2. Prefill new sequences (if any)                  │
  3. Decode one step for active sequences            │
  4. Sample tokens                                   │
  5. Send tokens to SSE streams                      ▼
  6. Free completed sequences              ←── SSE: data: [DONE]
```

The HTTP handler no longer blocks on generation. It enqueues the request and returns a streaming response. The scheduler loop is the only thing that touches the pipeline — it runs on a dedicated tokio task.

```rust
struct InFlightRequest {
    seq_id: u64,
    prompt_tokens: Vec<u32>,
    config: GenerationConfig,
    /// Channel to send generated tokens to the SSE stream.
    token_tx: mpsc::UnboundedSender<GenerationEvent>,
}

enum GenerationEvent {
    Token(u32),
    Finished(StopReason),
    Error(String),
}
```

### Admission Control

The scheduler must decide when to admit new prefills vs. prioritizing active decodes. The tradeoff:

- **Prefills are expensive:** A 512-token prefill processes 512× more tokens than a decode step, temporarily consuming batch capacity and increasing latency for active sequences.
- **Decodes are latency-sensitive:** Each decode step produces one user-visible token. Delays compound — a 10ms slowdown per step adds 2 seconds to a 200-token generation.

**Policy: Decode-priority with prefill slots.**

1. Always schedule all active decodes first (they're cheap — 1 token each).
2. If total batch tokens < `max_batch_tokens`, admit prefills from the queue to fill remaining capacity.
3. Prefill chunking: split long prompts into chunks (e.g., 512 tokens at a time) to avoid monopolizing a batch iteration. The sequence accumulates KV cache over multiple iterations before entering the decode phase.

---

## Component 4: Fault Tolerance

### Worker Failure Detection

Phase 3 already has heartbeat monitoring with nonce-validated acks. Phase 4 extends this to handle failures gracefully:

1. **Detection:** Heartbeat timeout (configurable, default 15s = 3 missed beats at 5s interval). Already implemented.
2. **Active sequence abortion:** All sequences with active cache on the dead worker are aborted. Their SSE streams receive a `GenerationEvent::Error`. Cache on surviving workers is freed.
3. **Pipeline reconfiguration:** The dead worker is removed from the pipeline order. If the model can fit on the remaining workers (scheduler re-runs with fewer nodes), the pipeline is rebuilt and new requests are accepted. If not, the coordinator enters a degraded state that rejects new requests.
4. **Worker reconnection:** A restarted worker can re-register. The coordinator re-runs the scheduler with the new set of workers and rebuilds the pipeline (new requests only — in-flight sequences continue on the old pipeline until completion).

### Coordinator Failure

The coordinator is a single point of failure. Phase 4 does NOT implement coordinator redundancy (that's a distributed consensus problem that adds enormous complexity for marginal benefit in the target use case). If the coordinator dies:

- Workers detect via heartbeat timeout and shut down cleanly.
- All in-flight requests are lost.
- Restart the coordinator; workers reconnect and re-register.

---

## Component 5: Pipeline Micro-Batching

To hide pipeline latency, the coordinator can overlap pipeline stages by sending the next batch's data to an earlier stage while a later stage is still processing the current batch.

```
Without micro-batching (Phase 3):
  Worker 0: [Batch1    ]            [Batch2    ]
  Worker 1:             [Batch1    ]             [Batch2    ]
  Worker 2:                         [Batch1    ]             [Batch2    ]
  Utilization: ~33%

With micro-batching (Phase 4):
  Worker 0: [B1][B2][B3][B4][B5]...
  Worker 1:     [B1][B2][B3][B4]...
  Worker 2:         [B1][B2][B3]...
  Utilization: ~100% (after pipeline fill)
```

This requires the coordinator to manage multiple in-flight batches. Each worker processes its stage for batch N while receiving batch N+1. The wire protocol already supports this — sequence IDs distinguish which batch each Forward/ForwardResult belongs to.

---

## Implementation Order

Phase 4 is large. It should be implemented incrementally, with each step producing a working system.

### Step 1: Paged KV Cache (Local Only)

Change the single-node server to use paged allocation. No batching yet — one sequence at a time, but with paged blocks instead of contiguous.

1. Implement `BlockPool` and `PagedKvCacheManager` in `fracture-engine`
2. Modify the attention CUDA kernel to accept block tables
3. Wire the new cache manager into the single-node server
4. Validate: greedy generation produces identical output to Phase 3 (block boundaries must not affect results)
5. Benchmark: compare memory usage for various sequence lengths

**Why first:** The paged cache is a prerequisite for continuous batching. Getting it right (and validated against reference outputs) before adding batching complexity is critical.

### Step 2: Batched Forward Pass (Local Only)

Add multi-sequence support to the engine. Still single-node, no networking changes yet.

1. Implement `BatchedForwardRequest` and modify the engine's forward path to handle concatenated batches
2. Modify attention to handle per-sequence block tables within a batch
3. Add a simple batch scheduler that admits multiple requests
4. Validate: batched execution produces identical per-sequence output to sequential execution
5. Benchmark: measure throughput improvement (requests/sec) with increasing batch sizes

### Step 3: Continuous Batching (Local Only)

Replace the Mutex-serialized server with async request handling and iteration-level scheduling.

1. Implement `BatchScheduler` with decode-priority admission
2. Replace the `spawn_blocking` + Mutex pattern with a dedicated scheduler loop
3. Add prefill chunking for long prompts
4. Wire SSE streaming to the scheduler's token output channels
5. Validate: multiple concurrent HTTP requests produce correct, independent results
6. Benchmark: measure TTFT and throughput under load

### Step 4: Distributed Batching

Extend continuous batching to the distributed pipeline.

1. Modify the coordinator's pipeline orchestration to send batched Forward messages
2. Each Forward message carries per-sequence metadata (positions, block tables)
3. Workers process the batch using their paged attention kernel
4. Cache allocation/free messages are batched (one per sequence entering/leaving)
5. Validate: distributed batched output matches local batched output
6. Benchmark: measure throughput scaling with additional workers

### Step 5: Fault Tolerance

Add graceful worker failure handling.

1. Extend heartbeat failure path to abort affected sequences
2. Implement pipeline reconfiguration on worker death
3. Implement worker reconnection with re-scheduling
4. Test: kill a worker mid-generation, verify surviving sequences on other workers are unaffected, verify the pipeline recovers for new requests

### Step 6 (Stretch): Pipeline Micro-Batching

Overlap pipeline stages for maximum utilization.

1. Coordinator sends batch N+1 to worker 0 while worker 2 is still processing batch N
2. Each worker has a 1-deep input queue
3. Sequence IDs in the wire protocol disambiguate concurrent batches

---

## What Phase 4 Does NOT Include (Deferred to Phase 5+)

- **Dynamic rebalancing** — re-partitioning layers while inference is running. This requires live KV cache migration between workers, which is a hard distributed state problem. Deferred to Phase 4b or later.
- **Mixed quantization** — different precision per node. Requires dequantization kernels and calibration for quality impact. Deferred to Phase 5 alongside the Metal backend.
- **GPU-direct RDMA** — bypassing CPU for activation transfer. Requires specialized hardware (InfiniBand) and kernel-level integration. Only worthwhile at data center scale.
- **Speculative decoding** — using a smaller draft model to predict tokens verified by the full model. Orthogonal to batching; can be added independently.
- **Prefix caching** — sharing KV cache across requests with common prefixes (e.g., system prompts). Natural extension of paged cache (share blocks instead of copying). Deferred to Phase 4b.

---

## Success Criteria (Phase 4 Complete)

- [ ] Paged KV cache: greedy generation matches Phase 3 output token-for-token
- [ ] Continuous batching: 10 concurrent requests served without correctness regression
- [ ] Throughput: >3× improvement over Phase 3 for concurrent request workload
- [ ] TTFT: first-token latency < 2× single-request latency under load
- [ ] Memory efficiency: 50-token sequence uses < 20 MB KV cache (vs 512 MB contiguous)
- [ ] Fault tolerance: worker death aborts only affected sequences; pipeline recovers for new requests
- [ ] Distributed batching: multi-machine pipeline serves batched requests correctly
