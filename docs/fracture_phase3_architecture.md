# Fracture Phase 3: Architecture Document
## Distributed Inference

**Depends on:** Phase 2 complete and validated (split equivalence passing)
**Goal:** Multiple physical machines cooperate via a custom wire protocol to run pipeline-parallel inference on a model that no single machine could run alone.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    COORDINATOR NODE                          │
│                                                             │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │ HTTP API │  │  Generation  │  │    Pipeline           │ │
│  │ (axum)   │──│  Loop        │──│    Orchestrator       │ │
│  └──────────┘  └──────────────┘  └───────────┬───────────┘ │
│                                               │             │
│  ┌──────────────┐  ┌──────────────────────┐   │             │
│  │  Scheduler   │  │  Peer Registry       │   │             │
│  │  (layer      │  │  (node tracking,     │   │             │
│  │   assignment) │  │   health monitoring) │   │             │
│  └──────────────┘  └──────────────────────┘   │             │
└───────────────────────────────────────────────┼─────────────┘
                                                │
                    Wire Protocol (TCP/QUIC)     │
              ┌─────────────────┬───────────────┘
              │                 │
              ▼                 ▼
┌─────────────────────┐  ┌─────────────────────┐
│   WORKER NODE A     │  │   WORKER NODE B     │
│   Layers [0, 16)    │  │   Layers [16, 32)   │
│                     │  │                     │
│  ┌───────────────┐  │  │  ┌───────────────┐  │
│  │ ComputeNode   │  │  │  │ ComputeNode   │  │
│  │ (Phase 2 API) │  │  │  │ (Phase 2 API) │  │
│  └───────────────┘  │  │  └───────────────┘  │
│  ┌───────────────┐  │  │  ┌───────────────┐  │
│  │ Wire Protocol │  │  │  │ Wire Protocol │  │
│  │ Server        │  │  │  │ Server        │  │
│  └───────────────┘  │  │  └───────────────┘  │
└─────────────────────┘  └─────────────────────┘
```

### Key Architectural Decisions

**Pipeline parallelism, not tensor parallelism.** Tensor parallelism shards each layer across GPUs and requires all-reduce synchronization after every layer — demanding NVLink-class bandwidth. Pipeline parallelism splits by layers and only transfers activations at stage boundaries. As the research confirms: pipeline parallelism can tolerate slower cross-node interconnects (standard Ethernet or InfiniBand) while tensor parallelism needs fast intra-node links. Since Fracture targets commodity hardware across a network, pipeline parallelism is the right choice.

**Centralized coordinator, not fully decentralized.** A coordinator node manages scheduling, request routing, and pipeline orchestration. Worker nodes are stateless aside from weights and KV cache. This simplifies the protocol dramatically — workers don't need to know about each other, only about the coordinator.

**Coordinator can also be a worker.** The coordinator node can run a compute node alongside its coordination duties (e.g., run layers [0, 16) while coordinating the pipeline). This avoids wasting a machine.

---

## Wire Protocol

### Transport: TCP with Optional QUIC Upgrade

Start with TCP. It's universally available, well-understood, and sufficient for the activation tensor sizes involved. QUIC is a future optimization for reduced connection setup time and built-in multiplexing.

Connection is persistent — established once between coordinator and each worker, held open for the lifetime of the session.

### Framing

Every message on the wire follows this frame format:

```
┌──────────────────────────────────────────────────────┐
│  Magic (2 bytes): 0x4652 ("FR")                      │
│  Version (1 byte): 0x01                              │
│  Message Type (1 byte): see table below              │
│  Sequence ID (8 bytes): u64 big-endian               │
│  Payload Length (4 bytes): u32 big-endian             │
│  Payload (variable): type-specific binary data       │
│  Checksum (4 bytes): CRC32C of entire frame          │
└──────────────────────────────────────────────────────┘
```

Total header: 20 bytes. Minimal overhead relative to activation payloads (which are 32KB+ per transfer).

### Message Types

| Type ID | Name | Direction | Payload |
|---|---|---|---|
| 0x01 | `Register` | Worker → Coordinator | NodeInfo (capabilities, GPU spec) |
| 0x02 | `RegisterAck` | Coordinator → Worker | Assigned layer range, session config |
| 0x03 | `Forward` | Coordinator → Worker | ForwardRequest + serialized tensor |
| 0x04 | `ForwardResult` | Worker → Coordinator | ForwardResponse + serialized tensor |
| 0x05 | `Heartbeat` | Bidirectional | Timestamp + GPU memory stats |
| 0x06 | `HeartbeatAck` | Bidirectional | Timestamp echo |
| 0x07 | `CacheAlloc` | Coordinator → Worker | seq_id, max_len |
| 0x08 | `CacheFree` | Coordinator → Worker | seq_id |
| 0x09 | `Shutdown` | Coordinator → Worker | Graceful shutdown signal |
| 0x0A | `Error` | Bidirectional | Error code + message |

### Tensor Serialization Format

Activation tensors are the dominant payload. Format optimized for zero-copy where possible:

```
┌──────────────────────────────────────────────────────┐
│  NDim (2 bytes): u16                                 │
│  Shape (4 bytes × NDim): u32 per dimension           │
│  DType (1 byte): 0=FP16, 1=FP32, 2=BF16             │
│  Compression (1 byte): 0=None, 1=LZ4                 │
│  Data Length (4 bytes): u32 (compressed if applicable)│
│  Data: raw tensor bytes                              │
└──────────────────────────────────────────────────────┘
```

**Size analysis for Llama 3 8B activations:**

| Scenario | Tensor shape | Size (FP16) |
|---|---|---|
| Decode (1 token) | [1, 4096] | 8 KB |
| Prefill (128 tokens) | [128, 4096] | 1 MB |
| Prefill (512 tokens) | [512, 4096] | 4 MB |
| Prefill (2048 tokens) | [2048, 4096] | 16 MB |

Decode transfers are tiny — 8KB over even slow networks is sub-millisecond. Prefill transfers are larger but happen once per request. Compression (LZ4) is worth enabling for prefill but pointless for decode.

### Transfer Path: Device → Network → Device

```
Sending node:
  1. backend.copy_to_host(device_tensor, pinned_host_buffer)
  2. Write frame header + tensor header to socket
  3. Write tensor data from pinned_host_buffer to socket

Receiving node:
  1. Read frame header + tensor header from socket
  2. Read tensor data into pinned_host_buffer
  3. backend.copy_to_device(device_tensor, pinned_host_buffer)
```

The copy-to-host and copy-to-device operations are Backend trait methods. On CUDA,
these map to `cudaMemcpyAsync` with pinned memory. On Metal, these would use
`MTLBuffer` with shared storage mode. The wire protocol doesn't know or care —
it receives raw bytes from the sending backend and hands raw bytes to the receiving
backend. This means a CUDA node and a Metal node can be in the same pipeline.

**Pinned (page-locked) host memory** is critical for CUDA backends. Regular `malloc`
memory must be copied to a staging buffer by the driver before DMA. Pinned memory
skips this step, roughly doubling transfer bandwidth. Each backend manages its own
pinned buffer pool. Metal backends on Apple Silicon don't need this — unified memory
means no DMA copy at all, just a pointer handoff.

---

## Peer Discovery and Registration

### Discovery: Simple and Manual in Phase 3

Workers are configured with the coordinator's address and connect on startup. No automatic discovery, no gossip protocol, no mDNS. Manual configuration is appropriate for a project where you control all nodes.

**Worker startup:**
```
1. Read config: coordinator_address, model_path, gpu_device
2. Load GGUF file metadata (but don't load full weights yet)
3. Run calibration benchmark:
   a. Load weights for layer 0 only
   b. Run 20 single-layer forward passes (N=1) for decode timing
   c. Run 20 single-layer forward passes (N=128) for prefill timing
   d. Discard first 5 of each (warmup), average remaining 15
   e. Record decode_ms_per_layer and prefill_ms_per_layer
   f. Free the temporary layer weights
4. Connect to coordinator
5. Send Register message with NodeInfo:
   - node_id (hostname or UUID)
   - gpu_model (e.g. "NVIDIA RTX 3090")
   - gpu_memory_total (e.g. 24GB)
   - gpu_memory_available (total minus OS overhead)
   - compute_capability (e.g. 8.6)
   - decode_ms_per_layer (measured benchmark)
   - prefill_ms_per_layer (measured benchmark)
6. Wait for RegisterAck with layer assignment
7. Load assigned layer weights from GGUF
8. Begin serving Forward requests
```

**Coordinator startup:**
```
1. Parse pipeline config: expected worker count, model path
2. Listen for worker connections
3. Wait until all expected workers have registered
4. Run scheduler to assign layers (see below)
5. Send RegisterAck to each worker with their layer assignment
6. Build pipeline from registered nodes
7. Start HTTP server and begin accepting inference requests
```

### Node Configuration

**Worker config (`fracture-worker.toml`):**
```toml
coordinator = "192.168.1.100:9400"
model_path = "/models/llama-3-8b.gguf"
gpu_device = 0
listen_port = 9401       # for coordinator to connect back
```

**Coordinator config (`fracture-coordinator.toml`):**
```toml
listen_address = "0.0.0.0:9400"
model_path = "/models/llama-3-8b.gguf"
expected_workers = 2
http_port = 8080
max_seq_len = 4096

# Scheduling: "auto", "equal", or "manual"
scheduling_mode = "auto"

# Only for scheduling_mode = "manual"
# [manual_assignments]
# node_a = "0..16"
# node_b = "16..32"

gpu_device = 0
coordinator_layers = true
```

---

## Scheduler

The scheduler assigns layer ranges to workers by balancing two constraints: each node must have enough GPU memory to hold its assigned weights and KV cache, and each node's compute time per decode step should be approximately equal so no node sits idle waiting for a slower one (the "pipeline bubble" problem).

### The Heterogeneous Hardware Problem

Pipeline-parallel inference bottlenecks on the slowest node. If a 5090 finishes its 20 layers in 12ms but a 3060 takes 25ms for its 12 layers, the 5090 idles for 13ms every decode step — wasting 52% of its compute. The scheduler must account for both memory capacity and compute throughput.

Additionally, different GPU generations have different Tensor Core capabilities, memory bandwidth, and CUDA compute capabilities. A scheduler that only looks at VRAM will produce poor assignments on heterogeneous clusters.

### Calibration Step

Published TFLOPS specs are unreliable for predicting actual per-layer inference throughput (they don't account for memory bandwidth, kernel launch overhead, or driver differences). Instead, each worker benchmarks itself at registration time.

```
Worker startup (revised):
1. Read config
2. Load GGUF metadata (but not weights yet)
3. Load weights for a SINGLE layer (layer 0) temporarily
4. Run 20 single-layer forward passes (N=1, decode scenario)
5. Discard first 5 (warmup), average the remaining 15
6. Record: decode_ms_per_layer
7. Repeat with N=128 (prefill scenario)
8. Record: prefill_ms_per_layer
9. Free the temporary layer weights
10. Connect to coordinator and send Register with benchmark results
```

This takes ~2-3 seconds and gives the scheduler actual measured throughput on the exact hardware and driver combination present.

### Input

```rust
struct SchedulerInput {
    model_config: ModelConfig,
    workers: Vec<WorkerCapabilities>,
    coordinator_compute: Option<WorkerCapabilities>,
    scheduling_mode: SchedulingMode,
    max_seq_len: usize,  // needed for KV cache memory estimation
}

struct WorkerCapabilities {
    node_id: String,
    gpu_model: String,                // e.g. "NVIDIA RTX 5090"
    gpu_memory_available: usize,      // bytes usable for weights + cache
    compute_capability: (u32, u32),   // e.g. (8, 6) for 3090, (12, 0) for 5090
    decode_ms_per_layer: f32,         // measured: time for 1-token forward on 1 layer
    prefill_ms_per_layer_128: f32,    // measured: time for 128-token forward on 1 layer
}

enum SchedulingMode {
    /// Optimize for balanced compute time (production default)
    /// Assigns more layers to faster nodes so both finish in ~same wall time
    Auto,

    /// Force equal layer count per node (ignore compute speed)
    /// Useful for testing distributed correctness when one node is much faster
    EqualSplit,

    /// Explicit manual assignment — overrides all scheduling logic
    /// Useful for testing specific configurations
    Manual(Vec<ManualAssignment>),
}

struct ManualAssignment {
    node_id: String,
    layer_range: Range<usize>,
}
```

### Algorithm: Compute-Balanced Assignment (Auto mode)

```
0. PRUNE NODES THAT WOULD SLOW THE PIPELINE
   For each candidate node, estimate whether including it helps or hurts:

   a. Calculate pipeline_latency_with = max(per_node_time) + num_hops * hop_latency_ms
      where per_node_time is compute-balanced across all N candidates,
      and num_hops = N - 1

   b. Calculate pipeline_latency_without = same calculation with N-1 nodes
      (the candidate's layers redistributed to remaining nodes)

   c. If pipeline_latency_without < pipeline_latency_with:
      - Check: can the model still fit without this node?
      - If yes: EXCLUDE the node. Log:
        "Excluding {node_id}: removing it saves {delta}ms per token.
         Its {decode_ms}ms/layer compute + {hop_latency}ms hop overhead
         exceeds the benefit of offloading {N} layers from faster nodes."
      - If no (model doesn't fit without it): INCLUDE anyway. Log:
        "Including {node_id} despite slowdown: model requires {X}GB,
         remaining nodes only have {Y}GB. This node adds {Z}GB needed
         to fit the model."

   d. Repeat until no more nodes can be profitably removed.

   hop_latency_ms is estimated at 2.0 ms by default (configurable).
   For nodes on the same machine (localhost), use 0.1 ms.

1. Calculate per-layer memory (same as before):
   weight_memory_per_layer = ~416 MB (FP16)
   cache_memory_per_layer = max_seq_len * 4 KB
   total_per_layer = weight_memory + cache_memory

2. For each remaining worker, calculate hard memory ceiling:
   max_layers[i] = floor(
     (gpu_memory_available - role_overhead) / total_per_layer
   )
   where role_overhead:
     head node: +1.0 GB (embedding)
     tail node: +1.0 GB (LM head)
     middle node: 0

3. Calculate relative compute speed (using decode benchmark):
   speed[i] = 1.0 / decode_ms_per_layer[i]  // higher = faster
   total_speed = sum(speed)

4. Calculate ideal layer count (proportional to speed):
   ideal_layers[i] = round(speed[i] / total_speed * num_total_layers)

5. Clamp to memory ceiling:
   assigned_layers[i] = min(ideal_layers[i], max_layers[i])

6. Redistribute any unassigned layers to nodes with remaining capacity,
   prioritizing faster nodes.

7. Validate:
   sum(assigned_layers) == num_total_layers
   If not achievable → error: insufficient total capacity

8. Assign contiguous ranges in pipeline order.
   Order nodes by network topology (or registration order for Phase 3).

9. Calculate expected pipeline balance:
   per_node_decode_time[i] = assigned_layers[i] * decode_ms_per_layer[i]
   Report the imbalance ratio:
     max(per_node_decode_time) / min(per_node_decode_time)
   Log a warning if imbalance > 1.5x
```

### Algorithm: Equal Split Mode

```
1. Divide layers evenly: layers_per_node = num_total_layers / num_nodes
   Distribute remainder to nodes with more memory.

2. Validate each node can hold its assignment within memory ceiling.
   If not → error.

3. Log the expected imbalance ratio (it may be high — that's fine for testing).
```

### Algorithm: Manual Mode

```
1. Validate that manual assignments cover all layers exactly once.
2. Validate each node can hold its assignment within memory ceiling.
3. Apply as-is.
```

### Output

```rust
struct LayerAssignment {
    node_id: String,
    layer_range: Range<usize>,
    role: NodeRole,  // Head, Middle, Tail
    expected_decode_ms: f32,   // predicted per-step time for this node
    weight_memory_gb: f32,     // estimated GPU memory for weights
    cache_memory_gb: f32,      // estimated GPU memory for KV cache at max_seq_len
}

struct SchedulerResult {
    assignments: Vec<LayerAssignment>,
    excluded_nodes: Vec<ExcludedNode>,     // nodes pruned in step 0
    pipeline_decode_ms: f32,               // predicted total per-token latency
    imbalance_ratio: f32,                  // max_node_time / min_node_time (1.0 = perfect)
    bottleneck_node: String,               // which node is slowest
}

struct ExcludedNode {
    node_id: String,
    reason: ExclusionReason,
}

enum ExclusionReason {
    /// Removing this node made the pipeline faster
    SlowsDownPipeline {
        latency_with_ms: f32,
        latency_without_ms: f32,
    },
    // Future: could add InsufficientMemory, IncompatibleBackend, etc.
}
```

The `SchedulerResult` is logged at startup so you can immediately see if the pipeline is well-balanced.

### Example: RTX 3090 + RTX 5090, Llama 3 8B FP16

Hypothetical benchmarks (actual numbers will come from calibration):
```
3090: decode_ms_per_layer = 1.1 ms, gpu_memory_available = 22 GB
5090: decode_ms_per_layer = 0.5 ms, gpu_memory_available = 30 GB
```

**Auto mode (compute-balanced):**
```
Speed ratio: 5090 is 2.2x faster than 3090
Ideal split: 5090 gets 22 layers, 3090 gets 10 layers

5090 (head): layers [0, 22)  — 9.2 GB weights + 1.0 GB embed = 10.2 GB
  expected decode: 22 × 0.5 = 11.0 ms
3090 (tail): layers [22, 32) — 4.2 GB weights + 1.0 GB LM head = 5.2 GB
  expected decode: 10 × 1.1 = 11.0 ms
Imbalance ratio: 1.0 (perfectly balanced)
Pipeline decode latency: ~11.0 ms + network overhead
```

**Equal split mode (for testing):**
```
5090 (head): layers [0, 16)  — 6.6 GB weights + 1.0 GB embed = 7.6 GB
  expected decode: 16 × 0.5 = 8.0 ms
3090 (tail): layers [16, 32) — 6.6 GB weights + 1.0 GB LM head = 7.6 GB
  expected decode: 16 × 1.1 = 17.6 ms
Imbalance ratio: 2.2 (5090 idles 55% of the time — acceptable for testing)
Pipeline decode latency: ~17.6 ms + network overhead
```

**Manual mode (force minimum viable split):**
```
5090 (head): layers [0, 24)  — 10.0 GB weights + 1.0 GB embed
3090 (tail): layers [24, 32) — 3.3 GB weights + 1.0 GB LM head
```

### Your Testing Setup: 3090 + 5090

For **correctness testing** (does distributed inference produce the right output?):
Use `EqualSplit` or `Manual` mode. The pipeline will be imbalanced — the 5090 idles
waiting for the 3090 every step. That's fine. You're testing correctness, not performance.
The 3090 with 16 layers at ~1.1 ms/layer adds ~17.6 ms per step, which is totally
workable for testing.

For **performance testing** (does the scheduler actually balance the pipeline?):
Use `Auto` mode. Verify that the imbalance ratio is close to 1.0. Compare
per-token latency against the single-node 5090 baseline. The distributed version
will be slower (network overhead) but the question is how much slower.

For **testing models that don't fit on one GPU:**
Use a larger model (Llama 3 70B) or FP16 weights that exceed 32GB. This forces
a real split where neither node can run alone. This is the true distributed
inference scenario and the one that matters for your resume.

---

## Pipeline Orchestrator (Distributed)

Phase 2's `PipelineCoordinator` chained local `NodeService` calls. Phase 3 replaces local calls with network calls while preserving the same sequencing.

### Prefill Pipeline

```
Coordinator receives HTTP request
    │
    ▼
Tokenize prompt → token_ids, positions
    │
    ▼
Send CacheAlloc to all workers (seq_id, max_seq_len)
    │
    ▼
Send Forward{TokenIds} to head node (Node A)
    │  ... Node A computes layers [0, 16), returns activations
    ▼
Receive activations from Node A
    │
    ▼
Send Forward{Activations} to tail node (Node B)
    │  ... Node B computes layers [16, 32), returns logits
    ▼
Receive logits from Node B
    │
    ▼
Sample next token
```

### Decode Pipeline (Repeated)

```
Send Forward{TokenIds=[new_token]} to Node A
    │  ... Node A: embed → layers [0,16) → activation [1, 4096]
    ▼
Receive activation (8 KB transfer — sub-millisecond)
    │
    ▼
Send Forward{Activations} to Node B
    │  ... Node B: layers [16, 32) → logits
    ▼
Receive logits → sample → stream token to client → repeat
```

### Pipeline Latency Analysis

For decode (the latency-sensitive phase):

| Component | Estimated time |
|---|---|
| Node A compute (16 layers) | ~15-20 ms |
| Activation transfer A→Coordinator | <1 ms (8 KB over 1 Gbps) |
| Activation transfer Coordinator→B | <1 ms |
| Node B compute (16 layers) | ~15-20 ms |
| Logits transfer B→Coordinator | <1 ms (even logits are small when we only return top-K) |
| Sampling | <1 ms |
| **Total per token** | **~35-45 ms** |

Compare to single-node Phase 1: ~30-40 ms per token. The network overhead adds roughly 2-3 ms per token — minimal because activation tensors during decode are tiny (8 KB).

Prefill has larger transfers but only happens once per request, so the latency is amortized.

### Optimization: Logits Compression

The tail node doesn't need to send all 128256 logits (~256 KB at FP16) back to the coordinator. For sampling, only the top-K logits are needed. The tail node can:

1. Run sampling locally and return just the selected token ID (4 bytes), or
2. Return only the top-256 logit values + indices (~1 KB) for the coordinator to sample

Option 1 is simplest and eliminates the largest per-token transfer. The coordinator sends sampling parameters (temperature, top_k, top_p) to the tail node in the Forward request.

---

## Sequence State Management

The coordinator tracks the state of every active sequence across the pipeline:

```rust
struct SequenceState {
    seq_id: u64,
    status: SequenceStatus,
    current_pos: usize,          // next position to generate
    max_tokens: usize,
    sampling_params: SamplingParams,
    generated_tokens: Vec<u32>,
    cache_allocated_on: Vec<String>,  // node IDs with active cache
}

enum SequenceStatus {
    Prefilling,
    Decoding,
    Complete,
    Error(String),
}
```

### Cache Lifecycle

```
Request arrives:
  → Coordinator sends CacheAlloc to ALL workers in pipeline
  → Each worker pre-allocates cache for its layer range

Generation completes (or request cancelled):
  → Coordinator sends CacheFree to ALL workers
  → Each worker frees cache memory for that sequence
```

The coordinator is the single source of truth for which sequences are active. Workers never allocate or free cache on their own initiative.

---

## Health Monitoring

### Heartbeat Protocol

```
Every 5 seconds:
  Coordinator → Worker: Heartbeat { timestamp, nonce }
  Worker → Coordinator: HeartbeatAck { timestamp_echo, gpu_memory_used, active_sequences }
```

### Failure Detection

| Condition | Detection | Response |
|---|---|---|
| Worker misses 3 heartbeats | 15 seconds | Mark worker as dead. Abort all sequences using that worker. |
| Worker returns Error on Forward | Immediate | Abort the affected sequence. Log error. Worker remains in pipeline. |
| Coordinator crashes | Workers detect closed connection | Workers dump cache and exit. Require manual restart. |

**Phase 3 does not implement automatic recovery or failover.** If a worker dies mid-generation, the sequence fails. The coordinator logs the failure and continues serving new requests if the remaining workers can form a valid pipeline (they can't for a 2-node split — both are needed). Fault tolerance is Phase 4.

---

## Testing Strategy

### Unit Tests (No network)

| Test | What it validates |
|---|---|
| Frame serialization/deserialization | Wire format round-trips correctly |
| Tensor serialization | Tensor → bytes → tensor preserves all values within FP16 tolerance |
| Scheduler: auto mode | Layer assignments balance compute time, respects memory ceilings |
| Scheduler: node pruning | Very slow node excluded when model fits without it |
| Scheduler: node kept despite slowdown | Slow node included when its memory is needed to fit model |
| Scheduler: pruning log messages | Excluded and forced-include nodes log clear reasons |
| Scheduler: equal split mode | Even layer count regardless of speed, validates memory fits |
| Scheduler: manual mode | Applies explicit assignment, validates coverage and memory |
| Scheduler: imbalance reporting | Logs correct imbalance ratio and identifies bottleneck node |
| Scheduler: insufficient memory | Error when total cluster memory can't hold the model |
| Scheduler: calibration data | Measured ms/layer values are plausible (not zero, not absurd) |
| Sequence state machine | Status transitions are valid, no illegal states |

### Integration Tests (Two processes, localhost)

| Test | What it validates |
|---|---|
| Register → RegisterAck handshake | Worker connects, benchmarks, registers, receives layer assignment |
| Calibration benchmark | Worker produces plausible ms/layer values for local GPU |
| Auto scheduling (equal GPUs) | Two identical GPUs → roughly equal layer split |
| Auto scheduling (mixed GPUs) | Faster GPU gets more layers, imbalance ratio near 1.0 |
| Equal split scheduling | Forces 16/16 split regardless of GPU speed difference |
| Manual scheduling | Explicit layer ranges applied correctly |
| Single Forward round-trip | Coordinator sends Forward, receives ForwardResult, tensors intact |
| Full prefill pipeline | Multi-token prefill through 2-node split matches monolithic |
| Multi-step decode | 50 decode steps through network produce same output as Phase 1 |
| Cache alloc/free lifecycle | Workers allocate and free cache on coordinator command |
| Heartbeat monitoring | Worker reports health, coordinator detects simulated failure |

### End-to-End Tests (Two machines — manual)

| Test | What it validates |
|---|---|
| Cross-machine inference | Generate text with model split across two physical machines |
| Greedy output equivalence | temperature=0 output matches single-node Phase 1 |
| Long generation | 1000-token generation completes without drift or memory leak |
| Network interruption | Unplugging network cable → coordinator detects failure within 15s |

---

## New Crates

```
crates/
├── fracture-protocol/     # Wire protocol types, serialization, framing
│   └── src/
│       ├── frame.rs       # Frame encoding/decoding
│       ├── messages.rs    # Message types (Register, Forward, Heartbeat, etc.)
│       └── tensor.rs      # Tensor serialization with optional LZ4 compression
├── fracture-coordinator/  # Coordinator logic: scheduler, orchestrator, registry
│   └── src/
│       ├── scheduler.rs   # Layer assignment algorithm
│       ├── registry.rs    # Peer tracking and health monitoring
│       ├── pipeline.rs    # Distributed pipeline orchestration
│       └── state.rs       # Sequence state management
├── fracture-worker/       # Worker binary: connects to coordinator, serves compute
│   └── src/
│       └── main.rs
└── fracture-node/         # Coordinator binary: HTTP API + coordination
    └── src/
        └── main.rs
```

`fracture-protocol` is shared between coordinator and worker. `fracture-engine` (Phase 1) and `fracture-generate` (Phase 1) are used by the coordinator. `fracture-engine` is used by workers.

---

## Implementation Order

1. **`fracture-protocol`** — Frame encoding, message types, tensor serialization
2. **Protocol tests** — Round-trip serialization for all message types
3. **`fracture-coordinator/scheduler`** — Layer assignment algorithm
4. **`fracture-coordinator/registry`** — Peer registration and tracking
5. **`fracture-worker`** — Worker binary: connect, register, serve Forward requests
6. **`fracture-coordinator/pipeline`** — Replace Phase 2's local coordinator with network calls
7. **Integration test** — Two processes on localhost, split model, verify equivalence
8. **`fracture-coordinator/state`** — Sequence lifecycle management
9. **Heartbeat** — Health monitoring and failure detection
10. **`fracture-node`** — Coordinator binary with HTTP API
11. **End-to-end test** — Full generation through distributed pipeline
12. **Cross-machine test** — Two physical machines, verify everything works over real network

---

## What Phase 3 Does NOT Include (Deferred to Phase 4)

- **Fault tolerance / failover** — worker death kills active sequences
- **Dynamic rebalancing** — layer assignments are fixed at startup
- **Continuous batching** — one sequence at a time through the pipeline
- **Paged KV cache** — contiguous allocation from Phase 1
- **Mixed quantization** — all nodes run same precision
- **Pipeline interleaving** — no micro-batching to hide pipeline bubbles
- **GPU-direct RDMA** — activation transfer goes through CPU pinned memory

These are all valuable optimizations, but each adds significant complexity. Phase 3 proves that distributed inference works correctly. Phase 4 makes it fast.

---

## Success Criteria (Phase 3 Complete)

- [ ] Two physical machines collaboratively run Llama 3 8B inference
- [ ] Model does not fit on either machine alone (use FP16 on 12GB GPUs to force the split)
- [ ] Greedy generation (temperature=0) produces byte-identical output to Phase 1
- [ ] Activation transfer overhead is < 5ms per decode step on 1 Gbps Ethernet
- [ ] Worker registration, layer assignment, and pipeline setup complete in < 30 seconds
- [ ] Worker failure is detected within 15 seconds via heartbeat
- [ ] 1000-token generation completes without errors or memory leaks
