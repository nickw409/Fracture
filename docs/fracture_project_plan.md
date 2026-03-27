# FRACTURE
## Distributed Cross-Platform LLM Inference Engine

*Project Plan & Architecture Overview*

---

## Overview

Fracture is a from-scratch LLM inference engine built in Rust with pluggable GPU backends (CUDA initially, Metal planned). The project begins as a single-node GPU-accelerated inference server capable of loading model weights, executing the transformer forward pass, managing KV cache, and serving completions over an HTTP API. It then extends into a distributed system where multiple nodes — potentially running different GPU backends and operating systems — split a model's layers across machines and cooperate via a custom binary protocol to run pipeline-parallel inference, enabling models that don't fit on a single GPU.

---

## Motivation & Resume Positioning

The project sits at the intersection of GPU systems programming and AI infrastructure. It fills a gap in an existing resume that is strong on GPU computing (Monte Carlo simulation, SIMD optimization, CUDA) and AI tooling (agentic workflows, LLM-powered agents) but lacks networking/protocol implementation, ML inference internals, and cross-platform GPU abstraction. Fracture adds:

- Binary protocol design and peer-to-peer networking
- Deep understanding of transformer inference mechanics (attention, KV cache, sampling)
- Distributed systems coordination (scheduling, fault tolerance, heterogeneous hardware)
- Cross-platform GPU backend abstraction (CUDA + Metal in a unified engine)
- A demonstrable, benchmarkable artifact (tokens/sec, multi-node scaling)

---

## Target Hardware

| Machine | GPU / Device | Device Memory | System RAM | PCIe | Role |
|---|---|---|---|---|---|
| Linux Desktop | NVIDIA RTX 3090 | 24 GB VRAM | 32 GB | Gen 3.0 | CUDA worker |
| Linux Desktop | NVIDIA RTX 5090 | 32 GB VRAM | 64-128 GB | Gen 5.0 | CUDA worker / coordinator |
| Mac Studio | Apple M2 Ultra | 64 GB unified | (shared) | N/A | Metal worker (Phase 5+) |
| **Total** | | **120 GB** | | | |

Key model fit at this cluster size:

| Model + Precision | Weight Size | Fits GPU-Resident? |
|---|---|---|
| Llama 3 8B FP16 | ~15 GB | Yes (single node) |
| Llama 3 70B INT4 | ~35 GB | Yes (2 CUDA nodes, forced split) |
| Llama 3 70B INT8 | ~70 GB | Yes (all 3 nodes in Phase 5) |
| Llama 3 70B FP16 | ~140 GB | Yes with all 3 nodes (tight) |

No offloading — if the model doesn't fit GPU-resident across available nodes, Fracture rejects the configuration. Quantization + more nodes is always preferred over RAM offloading.

---

## Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| Core runtime | Rust | Memory safety, async (tokio), zero-cost abstractions |
| Backend abstraction | `Backend` trait in fracture-core | Engine is generic over GPU backend; no CUDA/Metal imports in engine code |
| CUDA backend | CUDA + cuBLAS | Phase 1-3. Direct GPU control, Tensor Core acceleration |
| Metal backend | Metal Performance Shaders | Phase 5+. Apple Silicon unified memory, ~800 GB/s bandwidth |
| HTTP API | Rust (axum + tokio) | Async, OpenAI-compatible /v1/completions endpoint |
| Weight format | GGUF | Industry standard, supports quantized weights (INT4/INT8) |
| Wire protocol | Custom over TCP | Full control over tensor serialization, backend-agnostic byte transfer |
| Peer discovery | Centralized registry | Coordinator-based; workers register with capabilities on startup |

---

## Phased Architecture

### Phase 1: Local Inference Server

Build a complete single-node inference engine. This is the foundation that all subsequent phases build on and is a standalone deliverable.

- **Backend trait:** Define the cross-platform GPU abstraction in fracture-core. All operations (matmul, rmsnorm, rope, attention, silu_mul, embedding, memory management) are trait methods. The engine is generic over `B: Backend` and never imports any backend crate directly.
- **CUDA backend:** Implement the Backend trait using CUDA kernels and cuBLAS. This is the only backend in Phase 1, but the engine doesn't know that.
- **Weight loader:** Parse GGUF files, extract model architecture metadata, and load weights to device memory through the Backend trait's memory management methods.
- **Transformer forward pass:** Implement each operation through Backend trait calls: RMSNorm, RoPE, grouped-query attention with KV cache, SiLU activation, and feed-forward matmuls.
- **KV cache:** Allocate key/value tensors per layer per sequence. Design the memory layout to support future paged allocation (Phase 4). Track cache state per request.
- **Sampling & generation loop:** Logit processing (temperature, top-p, top-k), token sampling, KV cache update, loop until EOS or max tokens. Integrate an existing BPE tokenizer crate.
- **HTTP API:** Async server (axum/tokio) exposing /v1/completions and /v1/chat/completions. Support streaming via SSE for token-by-token output.

**MVP milestone:** Load Llama 3 8B in GGUF format, generate text on a single GPU, serve completions over HTTP. Benchmark tokens/sec against llama.cpp.

### Phase 2: Node Abstraction

Refactor the inference engine so it can operate on a subset of model layers rather than the full model. This is the architectural bridge to distribution.

- **Layer-range execution:** A node loads layers N through M and accepts an activation tensor as input rather than a text prompt. The forward pass runs only the assigned layers and outputs an activation tensor.
- **Node API:** Define the interface: NodeService trait with forward(), info(). This API becomes the wire protocol's local abstraction.
- **Selective weight loading:** Only load weights for assigned layers, reducing per-node memory usage.
- **Single-machine validation:** Run two node instances on one machine, each holding half the model. Verify output matches the monolithic Phase 1 server exactly (split equivalence test).

### Phase 3: Distribution

Add the networking layer that connects nodes across machines. This is where the project becomes a distributed system.

- **Wire protocol:** Custom binary protocol over TCP for tensor transfer. Backend-agnostic — tensors are serialized as raw bytes (shape + dtype + data). A CUDA node and a Metal node can be in the same pipeline because the protocol doesn't know what backend produced the bytes.
- **Peer discovery:** Centralized coordinator model. Workers register on startup with calibration benchmarks (measured ms/layer decode and prefill throughput).
- **Scheduler:** Given a model and registered nodes with heterogeneous hardware, determine optimal layer partitioning. Three modes: Auto (compute-balanced, with slow-node pruning), EqualSplit (for correctness testing), Manual (explicit override). The scheduler accounts for both memory capacity and measured compute speed. Nodes that would slow the pipeline are excluded unless their memory is needed.
- **Pipeline coordination:** Orchestrate sequential activation passing through the node pipeline. Coordinator tracks sequence state, manages cache lifecycle across all workers.
- **KV cache coherence:** Each node manages KV cache for its own layers. The coordinator is the single source of truth for which sequences are active.
- **Health monitoring:** Heartbeat protocol for failure detection. Phase 3 does not implement automatic recovery — worker death fails active sequences.

**MVP milestone:** Two physical machines collaboratively running inference on a model that neither could run alone. End-to-end text generation over the network.

### Phase 4: Production Concerns (Stretch)

- **Fault tolerance:** Detect node failures mid-generation and either reassign layers or gracefully degrade. Checkpoint KV cache state for recovery.
- **Continuous batching:** Slot new requests into the pipeline as old ones finish. Requires per-sequence KV cache lifecycle management across all nodes.
- **Paged KV cache:** Implement PagedAttention-style block allocation to eliminate memory fragmentation and enable memory sharing across sequences.
- **Dynamic rebalancing:** If a new node joins or one becomes slow, re-partition layers without stopping inference. Migrate cache state between nodes.
- **Mixed quantization:** Allow different nodes to run different quantization levels (FP16 on a beefy GPU, INT4 on a smaller one) while maintaining output quality.

### Phase 5: Cross-Platform Backends

- **Metal backend:** Implement the Backend trait for Apple Silicon using Metal Performance Shaders and custom Metal compute shaders. Zero changes to the engine, generation loop, server, or wire protocol.
- **Mixed-backend pipelines:** CUDA and Metal nodes in the same pipeline. The wire protocol already transfers raw bytes — each node's backend handles device-specific memory management independently.
- **Cross-platform testing:** Golden output comparison (each backend generates output for fixed prompts, compare token sequences). Backend equivalence validated per release.
- **Mac Studio integration:** With 64 GB unified memory, the M2 Ultra becomes the highest-memory node in the cluster, enabling larger models or more layers per node.

---

## Backend Abstraction (Cross-Platform Design)

The engine never imports any backend crate. All GPU operations go through the `Backend` trait defined in `fracture-core`. This is enforced by the crate dependency graph at compile time.

**Dependency graph:**
```
fracture-core           ← Backend trait, DeviceTensor, ModelConfig
    ↑
fracture-engine         ← uses Backend generically
    ↑
fracture-generate       ← uses engine, no GPU awareness
    ↑
fracture-server         ← HTTP layer, no GPU awareness

backends/fracture-cuda  ← implements Backend for CUDA
backends/fracture-metal ← implements Backend for Metal (Phase 5)

bins/fracture-server-cuda  ← binary: plugs CudaBackend into server
bins/fracture-server-metal ← binary: plugs MetalBackend into server (Phase 5)
```

**DeviceTensor** is an opaque handle (TensorId + shape + dtype). Only the backend knows how to map it to actual device memory. The engine never touches device pointers.

**Adding a new backend** means: implement the Backend trait, create a binary that wires it into the engine. Zero changes to fracture-engine, fracture-generate, fracture-server, or fracture-protocol.

---

## Critical Design Decisions

These decisions were made upfront because they compound through the entire project:

- **Backend trait over direct CUDA calls:** All GPU operations go through a trait. Costs nothing in Phase 1, enables Metal (and any future backend) without engine rewrites.
- **Row-major tensor convention everywhere:** Custom kernels use row-major naturally. cuBLAS column-major expectation is handled via the transpose trick inside the CUDA backend only. Other backends use whatever is native and translate internally.
- **FP16 storage, FP32 accumulation:** Weights and activations stored in FP16 for memory efficiency. Matrix multiplications and reductions accumulate in FP32 for numerical stability.
- **DeviceTensor as opaque handle:** The engine holds TensorIds, not device pointers. Backend manages the actual memory. This prevents CUDA types from leaking into engine code.
- **No offloading:** If the model doesn't fit GPU-resident across available nodes, Fracture rejects the configuration. Quantization + more nodes is preferred over RAM offloading, which is catastrophically slow on PCIe 3.0 hardware.
- **Wire protocol is backend-agnostic:** Tensor serialization is raw bytes (shape + dtype + data). A CUDA node's output and a Metal node's input are the same byte format.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| CUDA kernel debugging | High | Test each kernel in isolation against PyTorch reference. Build a numerical diff harness before writing kernels. |
| Scope creep (chasing llama.cpp perf) | Medium | Prioritize correctness and architecture over raw speed. Benchmark but don't optimize prematurely. |
| Distributed debugging complexity | High | Get Phase 1 rock-solid first. Use single-machine multi-process testing before going multi-node. |
| KV cache memory layout lock-in | High | Design the cache abstraction for paged allocation from the start, even if Phase 1 uses simple contiguous allocation. |
| Network bottleneck on activation transfer | Medium | Profile early. Activation tensors during decode are small (8-16 KB) — network is not the bottleneck. Prefill transfers are larger but one-time. |
| Backend trait too narrow or too wide | Medium | Design the trait around what the engine actually needs (the forward pass operations). Don't speculatively add operations. Extend the trait when the Metal backend reveals gaps. |
| Cross-platform numerical divergence | Low | Greedy decoding should match for 100+ tokens. Backend equivalence tested via golden output comparison per release. |

---

*Nicholas Wiley | github.com/nickw409 | Rust + CUDA + Metal | March 2026*
