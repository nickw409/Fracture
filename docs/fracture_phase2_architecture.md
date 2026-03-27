# Fracture Phase 2: Architecture Document
## Node Abstraction

**Depends on:** Phase 1 complete and validated
**Goal:** Refactor the inference engine so a node operates on an arbitrary layer range, accepting and returning activation tensors. Validate by running two nodes on one machine splitting a model.

---

## What Changes from Phase 1

Phase 1 built the hooks: `layer_range` on the compute engine, `CacheHandle` abstraction on the KV cache manager. Phase 2 activates those hooks and adds the node-level API that Phase 3's networking layer will call.

| Component | Phase 1 | Phase 2 |
|---|---|---|
| Compute engine input | Always token IDs | Token IDs (first node) OR activation tensor (middle/last nodes) |
| Compute engine output | Always logits | Logits (last node) OR activation tensor (first/middle nodes) |
| Compute engine layer_range | Always [0, 32) | Configurable subset, e.g. [0, 16) or [16, 32) |
| KV cache | Allocated for all 32 layers | Allocated only for assigned layer range |
| Weight store | Loads all 32 layers | Loads only assigned layer range |
| Generation loop | Owns full pipeline | Exists only on the coordinator node |
| HTTP server | Serves requests | Exists only on the coordinator node |

---

## Node Roles

A distributed Fracture pipeline has three node types. In Phase 2, all run on one machine as separate processes. In Phase 3, they run on separate machines.

### Head Node (layers [0, N))
- Receives token IDs from the generation loop
- Runs embedding lookup
- Executes layers 0 through N-1
- Outputs activation tensor [seq_len, 4096]
- Manages KV cache for layers 0..N
- In a 2-node split: runs layers [0, 16)

### Tail Node (layers [M, 32))
- Receives activation tensor [seq_len, 4096]
- Executes layers M through 31
- Runs final RMSNorm
- Runs LM head to produce logits [seq_len, 128256]
- Manages KV cache for layers M..32
- Returns logits to coordinator for sampling
- In a 2-node split: runs layers [16, 32)

### Middle Node (layers [N, M)) — only in 3+ node pipelines
- Receives activation tensor
- Executes assigned layers
- Outputs activation tensor
- Manages KV cache for its layer range only

---

## Modified Compute Engine Interface

Phase 2's `ComputeNode` wraps Phase 1's `ComputeEngine<B: Backend>`. Like the engine,
it is generic over the backend — Phase 2 code lives in `fracture-engine` and has no
backend-specific imports.

```rust
/// Describes what a node is responsible for
struct NodeConfig {
    layer_range: Range<usize>,  // e.g. 0..16
    is_head: bool,              // true → owns embedding lookup
    is_tail: bool,              // true → owns final norm + LM head
}

enum NodeInput {
    /// Head node: receives token IDs from the generation loop
    TokenIds {
        ids: Vec<u32>,
        positions: Vec<usize>,
    },
    /// Middle/tail node: receives activation tensor from previous node
    Activations {
        hidden_states: DeviceTensor,   // [seq_len, 4096]
        positions: Vec<usize>,      // needed for RoPE in assigned layers
    },
}

enum NodeOutput {
    /// Tail node: returns logits for sampling
    Logits(DeviceTensor),              // [seq_len, 128256] or [1, 128256]
    /// Head/middle node: returns activation tensor for next node
    Activations(DeviceTensor),         // [seq_len, 4096]
}

trait ComputeNode {
    fn forward(
        &self,
        input: NodeInput,
        cache: &mut dyn KVCacheManager,
        cache_handle: CacheHandle,
    ) -> Result<NodeOutput>;

    fn config(&self) -> &NodeConfig;
}
```

The concrete implementation is `ComputeNodeImpl<B: Backend>` which holds a reference
to the `ComputeEngine<B>`. This means a CUDA node and a Metal node both implement
`ComputeNode` through the same engine code — only the backend differs.

### Implementation

The `ComputeNode::forward` implementation is a thin wrapper around Phase 1's compute engine:

```
fn forward(input, cache, handle):
    match input:
        TokenIds { ids, positions }:
            assert(self.config.is_head)
            hidden = embedding_lookup(ids)
        Activations { hidden_states, positions }:
            assert(!self.config.is_head)
            hidden = hidden_states

    for layer in self.config.layer_range:
        hidden = run_transformer_layer(hidden, layer, positions, cache, handle)

    if self.config.is_tail:
        hidden = rmsnorm(hidden, output_norm_weight)
        logits = hidden @ lm_head.T
        return Logits(logits)
    else:
        return Activations(hidden)
```

The transformer layer execution is identical to Phase 1. No kernel changes.

---

## Modified Weight Store

Phase 1's `WeightStore` loads everything. Phase 2 adds selective loading:

```rust
impl WeightStore {
    /// Load only the layers needed for this node's role
    fn load_for_node(path: &str, node_config: &NodeConfig) -> Result<Self> {
        let config = parse_gguf_metadata(path)?;
        let mut store = WeightStore::new(config);

        if node_config.is_head {
            store.embedding = load_tensor(path, "token_embd.weight")?;
        }

        for i in node_config.layer_range.clone() {
            store.layers[i] = load_layer_weights(path, i)?;
        }

        if node_config.is_tail {
            store.output_norm = load_tensor(path, "output_norm.weight")?;
            store.lm_head = load_tensor(path, "output.weight")?;
        }

        Ok(store)
    }
}
```

**Memory savings:** A 2-node split where each loads 16 layers reduces per-node weight memory from ~15.3 GB to ~7.65 GB + embedding/LM head overhead on head/tail nodes respectively. This lets each node fit comfortably on a 12GB GPU.

---

## Modified KV Cache Manager

No interface changes needed — the existing `KVCacheManager` trait works as-is. The only difference is that a node allocating cache for `layer_range = 16..32` creates buffers for 16 layers instead of 32, halving cache memory.

The implementation change is trivial: the `alloc` method respects the node's layer range.

```rust
fn alloc(&mut self, seq_id: u64, max_len: usize) -> CacheHandle {
    let num_layers = self.layer_range.len();  // 16 instead of 32
    // allocate buffers for num_layers instead of total model layers
    ...
}
```

Cache layer indices are local to the node. When a node owns layers [16, 32), its internal cache slot 0 corresponds to model layer 16.

---

## Node API (Local Process Communication)

Phase 2 uses local IPC to validate the split before Phase 3 adds networking. The node API is the same interface that Phase 3's wire protocol will call.

```rust
/// The API surface that Phase 3's network layer will wrap
trait NodeService {
    /// Process a forward pass and return the result
    fn forward(&self, request: ForwardRequest) -> Result<ForwardResponse>;

    /// Report node capabilities
    fn info(&self) -> NodeInfo;
}

struct ForwardRequest {
    seq_id: u64,
    input: NodeInput,
    is_prefill: bool,          // hint for cache behavior
}

struct ForwardResponse {
    seq_id: u64,
    output: NodeOutput,
}

struct NodeInfo {
    node_id: String,
    layer_range: Range<usize>,
    is_head: bool,
    is_tail: bool,
    gpu_memory_total: usize,
    gpu_memory_used: usize,
}
```

### Phase 2 IPC: Unix Domain Sockets

For single-machine validation, nodes communicate via Unix domain sockets using a simple framed protocol:

```
[4 bytes: message length (u32 big-endian)]
[N bytes: message payload (bincode-serialized ForwardRequest or ForwardResponse)]
```

Activation tensors are serialized as:
```
[4 bytes: ndim]
[4 bytes × ndim: shape]
[4 bytes: dtype (0=f16)]
[N bytes: raw tensor data, copied from GPU → CPU → socket → CPU → GPU]
```

The GPU→CPU→socket→CPU→GPU copy path is inefficient but correct. Phase 3 replaces this with a proper wire protocol that can later support GPU-direct RDMA.

---

## Pipeline Coordinator

In Phase 2, the coordinator runs in the same process as the head node. It orchestrates the pipeline:

```rust
struct PipelineCoordinator {
    nodes: Vec<Box<dyn NodeService>>,  // ordered by layer range
}

impl PipelineCoordinator {
    fn generate_step(&self, input: NodeInput, seq_id: u64) -> Result<DeviceTensor> {
        let mut current_input = input;

        for node in &self.nodes {
            let response = node.forward(ForwardRequest {
                seq_id,
                input: current_input,
                is_prefill: false,  // set correctly by caller
            })?;

            current_input = match response.output {
                NodeOutput::Activations(act) => NodeInput::Activations {
                    hidden_states: act,
                    positions: /* carried from input */,
                },
                NodeOutput::Logits(logits) => return Ok(logits),
            };
        }

        Err(Error::PipelineIncomplete)  // last node should have returned logits
    }
}
```

The generation loop from Phase 1 calls `coordinator.generate_step()` instead of directly calling the compute engine. For a single-node config (Phase 1 compatibility), the coordinator has exactly one node covering [0, 32).

---

## Validation Strategy

### The Critical Test: Split Equivalence

The single most important test in Phase 2:

```
1. Run full Phase 1 pipeline on test prompt, temperature=0 → get logits_full
2. Run 2-node Phase 2 pipeline on same prompt → get logits_split
3. Assert: logits_full == logits_split (within FP16 tolerance)
```

If this test passes, the abstraction is correct. If it fails, the activation tensor transfer or cache indexing has a bug.

Run this test for:
- 2-node split: [0, 16) + [16, 32)
- 3-node split: [0, 11) + [11, 22) + [22, 32) (uneven)
- Asymmetric split: [0, 8) + [8, 32) (simulates heterogeneous hardware)
- Single-node: [0, 32) (regression test — must match Phase 1 exactly)

### Additional Tests

| Test | What it validates |
|---|---|
| Selective weight loading | Node with [16, 32) doesn't load layers 0-15. GPU memory reflects this. |
| Partial KV cache | Node with [16, 32) allocates cache for 16 layers, not 32. |
| Multi-step decode | Run 50 decode steps through the split pipeline. Output matches monolithic. |
| Node info reporting | NodeInfo accurately reflects layer range, GPU memory usage. |

---

## Implementation Order

1. Add `NodeConfig` to compute engine, make `layer_range` functional
2. Implement `NodeInput`/`NodeOutput` enums and the `ComputeNode` trait
3. Modify weight store to support selective layer loading
4. Modify KV cache manager to respect layer range
5. Build `PipelineCoordinator` that chains nodes
6. Test: single-node coordinator matches Phase 1 output exactly
7. Build Unix socket IPC for multi-process communication
8. Test: 2-process split matches monolithic output
9. Test: 3-process split, asymmetric split
10. Test: 50-step decode through split pipeline

**Estimated scope:** Phase 2 is primarily a refactoring exercise. No new kernels. No new algorithms. The hard part is ensuring the activation tensor handoff preserves numerical equivalence. Expect 1-2 weeks if Phase 1 is solid.

---

## What Phase 2 Produces for Phase 3

Phase 3 needs exactly three things from Phase 2:

1. **`NodeService` trait** — The interface that Phase 3 wraps with network transport
2. **Activation tensor serialization** — The format that Phase 3 sends over the wire
3. **`PipelineCoordinator`** — The orchestration logic that Phase 3 extends with peer discovery and scheduling

All three are defined and tested in Phase 2. Phase 3 replaces the Unix socket transport with TCP/QUIC and adds the distributed coordination layer on top.
