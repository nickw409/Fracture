<div align="center">

# Fracture

**Distributed LLM inference engine in Rust with pluggable GPU backends**

Split large language models across multiple GPUs on multiple machines.\
Pipeline-parallel execution over TCP. OpenAI-compatible API. Zero engine changes to add new backends.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2024_Edition-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-green.svg)](https://developer.nvidia.com/cuda-toolkit)

[Architecture](#architecture) · [Features](#features) · [TurboQuant](#turboquant-kv-cache-compression) · [Quick Start](#quick-start) · [Distributed Inference](#distributed-inference) · [Network Resilience](#network-resilience) · [Testing](#testing) · [Roadmap](#roadmap)

</div>

---

Fracture is a from-scratch LLM inference engine (~41k lines of Rust, ~1300 lines of CUDA) designed around one principle: **the engine never knows what GPU it's running on**. All compute flows through a `Backend` trait, making the core engine, generation loop, HTTP server, and wire protocol completely backend-agnostic. Today that backend is CUDA; tomorrow it's Metal, and a single inference cluster can mix both.

The system runs Llama 3.1 8B with full numerical validation against PyTorch — greedy generation is token-for-token identical. It has been validated across heterogeneous hardware (RTX 5090 + RTX 3090) in multi-machine distributed inference. KV cache compression via [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) reduces memory consumption by 5x with 0.999 cosine similarity to FP16 output. The cluster is self-healing: workers reconnect after coordinator death, any worker can win a leader election, and nodes can join or leave without restarting the cluster.

## Architecture

```
                          ┌──────────────────────────────────────┐
                          │        Coordinator (HTTP API)         │
                          │  /v1/completions  /v1/chat  /health   │
                          └─────────────────┬────────────────────┘
                                            │
                          ┌─────────────────▼────────────────────┐
                          │        Distributed Pipeline           │
                          │  Continuous batching, paged KV cache, │
                          │  sequence state, cache lifecycle       │
                          └──┬──────────────┬───────────────┬────┘
                             │              │               │
                    ┌────────▼───┐  ┌───────▼────┐  ┌───────▼─────┐
                    │  Worker 0  │  │  Worker 1  │  │  Worker 2   │
                    │ Layers 0-9 │  │Layers 10-19│  │Layers 20-31 │
                    │   (Head)   │  │  (Middle)  │  │   (Tail)    │
                    └────────────┘  └────────────┘  └─────────────┘
                        GPU 0           GPU 1            GPU 2
```

Each worker runs the same backend-generic transformer forward pass over its assigned layer range. The coordinator chains activations through workers in order: the head worker embeds tokens, middle workers transform activations through their layers, and the tail worker produces logits. For single-node inference, a standalone server binary runs the full model on one GPU without the coordinator/worker overhead.

### Crate Map

The workspace enforces strict dependency boundaries — engine, server, and protocol crates never import a backend crate.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Binaries                                                           │
│  fracture-server-cuda · fracture-worker-cuda · fracture-coordinator │
└────────┬────────────────────┬──────────────────────┬────────────────┘
         │                    │                      │
┌────────▼─────────┐  ┌──────▼───────────┐  ┌───────▼──────────┐
│  fracture-server  │  │ fracture-generate │  │fracture-coordin. │
│  HTTP + SSE       │  │ Sampling + loop   │  │ Scheduling +     │
│                   │  │                   │  │ pipeline + HB    │
└────────┬──────────┘  └───────┬───────────┘  └───────┬──────────┘
         │                     │                      │
         │              ┌──────▼───────────┐  ┌───────▼──────────┐
         │              │  fracture-engine  │  │fracture-protocol │
         │              │  Forward pass,    │  │ Wire protocol,   │
         │              │  KV cache, batch  │  │ CRC32C, TCP      │
         │              └──────┬───────────┘  └──────────────────┘
         │                     │
┌────────▼─────────────────────▼──────────────────────────────────┐
│  fracture-core  ·  fracture-gguf                                │
│  Backend trait, DeviceTensor, ModelConfig  ·  GGUF parser/loader │
└─────────────────────────────────────────────────────────────────┘
         ▲
┌────────┴──────────┐
│  fracture-cuda    │
│  CUDA Backend     │
│  impl + kernels   │
└───────────────────┘
```

| Crate | Purpose |
|---|---|
| `fracture-core` | `Backend` trait, `DeviceTensor` (opaque GPU handle), `ModelConfig`, `TurboQuantConfig`, Lloyd-Max solver, error types, profiling |
| `fracture-engine` | Transformer forward pass, KV cache (contiguous + paged + quantized), `PagedCache` trait, batched forward, node abstraction, IPC, batch scheduler |
| `fracture-generate` | Sampling (temperature, top-k, top-p, seeded RNG), generation loop, cooperative cancellation |
| `fracture-server` | OpenAI-compatible HTTP API (axum), SSE streaming, batched + serialized modes |
| `fracture-gguf` | GGUF v3 file parser and weight loader (FP16/FP32/BF16) |
| `fracture-protocol` | Binary wire protocol over TCP (CRC32C integrity, 20 message types, 256 MB cap) |
| `fracture-coordinator` | Scheduler, peer registry, heartbeat, distributed pipeline (single + batched forward), rebalance orchestration |
| `fracture-election` | Priority-based leader election: `ElectionAgent`, `TermTracker`, bully-algorithm state machine |
| `fracture-cuda` | CUDA `Backend` implementation: 10 hand-written kernels (including TurboQuant compress/decompress/attention) + cuBLAS matmul |

## Features

### Inference Engine
- Full Llama 3.1 8B transformer: embedding, RMSNorm, GQA attention, RoPE, SwiGLU FFN
- FP16 storage with FP32 accumulation for numerical stability
- Greedy and stochastic sampling (temperature, top-k, top-p, seeded RNG)
- **Paged KV cache** — 16-token blocks allocated on demand (vs. contiguous pre-allocation), enabling dozens to hundreds of concurrent sequences
- **Batched forward pass** — processes multiple sequences in a single engine call
- **Continuous batching** — requests enter and leave the batch dynamically; decode-priority scheduling, prefill chunking (configurable, default 512 tokens)
- **[TurboQuant KV cache compression](#turboquant-kv-cache-compression)** — 5x memory reduction via rotation + Lloyd-Max quantization (ICLR 2026), opt-in A/B with FP16

### CUDA Backend
Hand-written kernels optimized for inference:
- RMSNorm, RoPE, SiLU×Mul, Embedding, Residual Add
- Contiguous and paged attention (paged reads K/V from block tables)
- TurboQuant fused compress (normalize → rotate → quantize → pack) and fused quantized attention (query pre-rotation + rotated-space V accumulation)
- cuBLAS FP16 GEMM with FP32 accumulation

### HTTP API (OpenAI-compatible)
- `POST /v1/completions` — text completion
- `POST /v1/chat/completions` — chat with Llama 3 chat template
- `GET /v1/models` · `GET /health`
- SSE streaming with `id`, `object`, `created`, `model`, `finish_reason`, `usage`
- `finish_reason`: `"stop"` (EOS) or `"length"` (max_tokens)
- Request validation, cooperative cancellation on client disconnect

### Distributed Inference
- Custom binary wire protocol with CRC32C integrity checks
- Compute-balanced layer scheduling with GPU calibration (5 warmup + 15 averaged forward passes)
- Three scheduling modes: **Auto** (performance-optimized), **EqualSplit** (testing), **Manual** (explicit)
- Heartbeat protocol with nonce-validated acks and dead-worker detection
- Cache lifecycle management: alloc with partial-failure rollback, reuse after free
- Distributed batched forward with per-worker paged cache and block pool admission control

### Network Resilience
- **Worker reconnection** — workers reconnect automatically after coordinator death; `ReRegister` protocol skips weight reload on reconnect
- **Leader election** — priority-based bully algorithm; any worker can become coordinator without human intervention; term numbers prevent split-brain
- **Seed node discovery** — new nodes query a known seed address (`WhoIsCoordinator` message) to find the current coordinator regardless of which node was originally elected
- **Dynamic join/leave** — workers join mid-run with deferred pipeline integration (`Pending` status); workers leave gracefully via drain (active requests complete before departure)
- **Crash recovery** — coordinator detects dead workers via heartbeat timeout, aborts in-flight sequences, and rebuilds the pipeline with surviving workers
- **`fracture.env` config file** — all CLI flags can be set via a config file; loaded at startup, CLI flags take precedence

### Numerical Validation
- Per-layer comparison against PyTorch reference tensors (rtol=1e-3, atol=1e-3)
- Greedy generation matches PyTorch token-for-token
- Batched output is bit-identical to sequential (max_diff=0.000000)
- Paged attention is bit-identical to contiguous (max_diff=0.000000)
- Cross-machine validation: RTX 5090 + RTX 3090

## Quick Start

### Requirements

- Rust (edition 2024)
- NVIDIA GPU with CUDA toolkit (`nvcc` on PATH)
- [`cargo-nextest`](https://nexte.st/) for running tests
- A GGUF model file (Llama 3 8B FP16) and its HuggingFace `tokenizer.json`

### Build

```bash
cargo check            # Verify workspace compiles
cargo clippy           # Lint
```

### Single-Node Server

```bash
cargo run --release -p fracture-server-cuda -- \
    --model /path/to/llama-3-8b.gguf \
    --tokenizer /path/to/tokenizer.json \
    --port 8080
```

Send a request:

```bash
# Text completion
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is", "max_tokens": 64, "temperature": 0}'

# Chat (streaming)
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 64, "stream": true}'
```

## Distributed Inference

Start the coordinator on machine A:

```bash
cargo run --release -p fracture-coordinator-cuda -- \
    --model /path/to/llama-3-8b.gguf \
    --tokenizer /path/to/tokenizer.json \
    --listen 0.0.0.0:9410 \
    --http-port 8080 \
    --min-workers 2
```

Start workers on machines B and C:

```bash
cargo run --release -p fracture-worker-cuda -- \
    --coordinator <machine-A-ip>:9410 \
    --model /path/to/llama-3-8b.gguf
```

The coordinator waits until `--min-workers` are registered, runs calibration, assigns layers based on measured GPU performance, and starts serving. The HTTP API is identical — clients don't know inference is distributed. Additional workers can join after startup and will be integrated dynamically.

### Config File

All flags can be set in a `fracture.env` file in the working directory (CLI flags take precedence):

```ini
FRACTURE_MODEL=/path/to/llama-3-8b.gguf
FRACTURE_LISTEN=0.0.0.0:9410
FRACTURE_HTTP_PORT=8080
FRACTURE_MIN_WORKERS=2
FRACTURE_SEEDS=192.168.1.10:9410,192.168.1.11:9410
```

See `fracture.env.example` for all documented keys.

### How Scheduling Works

Each worker runs calibration forward passes at registration. The scheduler uses these timings to assign layers proportional to each GPU's speed, then clamps assignments to fit in available memory. Slow workers can be pruned entirely. The result: a heterogeneous cluster (e.g., RTX 5090 + RTX 3090) where each GPU gets the workload it can handle.

## Network Resilience

### Leader Election

Any worker can become the coordinator. When the current coordinator dies, workers detect the timeout via heartbeat, then hold an election:

1. Each worker has an `--election-priority` (higher wins). Default derives from IP/port.
2. Workers exchange `ElectionStart` challenges; the highest-priority live node wins.
3. The winner broadcasts `Victory` and starts accepting connections as coordinator.
4. Term numbers are monotonically increasing — a coordinator with an older term is rejected.

```bash
# Worker that should become coordinator if the primary fails
cargo run --release -p fracture-worker-cuda -- \
    --coordinator <primary>:9410 \
    --model /path/to/model.gguf \
    --election-priority 100 \
    --peer-port 9411
```

### Seed Node Discovery

New nodes need to find the current coordinator. Pass one or more well-known seed addresses; nodes will query them with `WhoIsCoordinator` to resolve the actual coordinator address even after failover:

```bash
cargo run --release -p fracture-worker-cuda -- \
    --model /path/to/model.gguf \
    --seed 192.168.1.10:9410 \
    --seed 192.168.1.11:9411
```

### Graceful Leave and Dynamic Join

```bash
# Drain a worker before taking it offline (waits for active requests to finish)
curl -X POST http://<coordinator>:8080/admin/drain \
  -H "Content-Type: application/json" \
  -d '{"worker_id": "192.168.1.12:9412"}'

# View current cluster state
curl http://<coordinator>:8080/admin/cluster

# Trigger a manual pipeline rebalance
curl -X POST http://<coordinator>:8080/admin/rebalance
```

Workers that join after the pipeline is running are held in `Pending` state and integrated on the next rebalance without interrupting in-flight requests.

## TurboQuant KV Cache Compression

Fracture implements Google's [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) algorithm (ICLR 2026) for KV cache compression. The feature is opt-in — pass `--kv-quant turboquant` to a worker and the KV cache switches from FP16 to quantized storage. The FP16 path remains the default and is completely unaffected.

### What It Does

Each KV head vector (128 floats in FP16 = 256 bytes) is compressed to ~50 bytes via:

1. **Normalize** — store the L2 norm as FP16 (2 bytes), normalize to unit sphere
2. **Rotate** — multiply by a random orthogonal matrix so coordinates become near-Gaussian
3. **Quantize** — Lloyd-Max optimal scalar quantization per coordinate (4-bit keys, 2-bit values)
4. **Pack** — bit-pack indices into bytes (2 indices/byte at 4-bit, 4 indices/byte at 2-bit)

Decompression reverses the process: unpack → centroid lookup → unrotate → rescale. The attention kernel fuses decompression with the attention computation to avoid intermediate buffers.

### Why V3 (No QJL)

The original paper proposes a two-stage approach: MSE quantization + QJL residual correction for unbiased inner products. Six independent reimplementations confirmed that QJL fails under softmax — the estimator's variance is exponentially amplified, causing all generation tests to fail. Fracture implements the community-validated V3 variant: all bits go to MSE reconstruction quality, no QJL. This achieves 0.999 logit cosine similarity vs FP16 at 5x compression.

### Kernel Optimizations

The fused attention kernel (`attention_paged_tq.cu`) avoids the naive O(d^2) per-KV-position decompression cost:

- **Query pre-rotation**: `dot(q, Pi^T @ y_hat) = dot(Pi @ q, y_hat)`. Rotate the query once (O(d^2)), then every K score is an O(d) dot product in the rotated space.
- **V accumulation in rotated space**: accumulate the weighted V sum without unrotating each vector, then unrotate the final result once. Reduces V cost from O(kv_len × d^2) to O(kv_len × d + d^2).

### Memory Savings

On an RTX 3090 (24 GB) with Llama 3.1 8B:

| Mode | Bytes/block | Blocks in 8.2 GB | Token capacity |
|------|------------|-------------------|----------------|
| FP16 | 2,097,152 | 4,096 | 65,536 |
| TQ K4/V2 | 409,600 | 20,480 | 327,680 |

**5.12x more tokens** in the same memory. This means 5x longer context or 5x more concurrent sequences under continuous batching.

### Architecture

The batched forward pass is generic over a `PagedCache` trait that both `PagedKvCacheManager` (FP16) and `QuantizedKvCacheManager` (TurboQuant) implement. The trait's `dispatch_attention` method encapsulates the difference — FP16 gathers block tensors and calls `attention_paged`, TQ gathers packed indices, norms, rotation matrices, and centroids and calls `attention_paged_tq`. Zero code duplication in the forward pass.

```rust
// The same batched_forward code runs both FP16 and TurboQuant
pub fn batched_forward<B: Backend, C: PagedCache>(
    backend: &B, weights: &WeightStore, layer_range: &Range<usize>,
    cache: &mut C, sequences: &[SequenceSlice],
) -> Result<BatchedOutput>
```

### Usage

```bash
# Distributed worker with TurboQuant K4/V2 (5x compression)
cargo run --release -p fracture-worker-cuda -- \
    --coordinator <host>:9410 \
    --model /path/to/model.gguf \
    --kv-quant turboquant

# Custom bit widths
cargo run --release -p fracture-worker-cuda -- \
    --coordinator <host>:9410 \
    --model /path/to/model.gguf \
    --kv-quant turboquant \
    --tq-key-bits 4 \
    --tq-value-bits 2 \
    --tq-protected-layers 4    # First/last 4 layers use 8-bit for quality
```

### Validated Results

| Test | Result |
|------|--------|
| 8-bit TQ vs FP16 logit cosine similarity | 0.999999 |
| K4/V2 TQ vs FP16 logit cosine similarity | 0.998995 |
| K4/V2 greedy argmax matches FP16 | Yes |
| Compress → decompress round-trip (4-bit) | cosine > 0.95 |
| Compress → decompress round-trip (8-bit) | cosine > 0.999 |
| Zero vector handling | No NaN, no crash |

See [docs/turboquant.md](docs/turboquant.md) for the full technical writeup covering the algorithm, kernel optimizations, Lloyd-Max solver, rotation matrix generation, and storage layout.

## Testing

772 tests covering unit, GPU kernel, integration, model-validation, and end-to-end distributed scenarios.

```bash
cargo nextest run      # Run all tests
```

Tests are organized into thread groups via nextest config to manage GPU memory:
- **GPU-memory-sensitive** (max-threads=1): model-validation tests that load the full 15 GB model
- **E2E-distributed** (max-threads=1): tests that spawn coordinator + worker processes

| Category | What's Tested |
|---|---|
| Core types | Config validation, DType, tensor shape/reshape, error types |
| GGUF parser | Header, metadata, tensor info, weight loading, BF16 conversion |
| CUDA kernels | RMSNorm, RoPE, SiLU, attention (contiguous + paged + TurboQuant), embedding, add, matmul |
| Engine | Forward pass, node dispatch, pipeline splits, KV cache (contiguous + paged + quantized), batched forward, PagedCache trait, IPC |
| Sampling | Temperature, top-k, top-p, greedy, NaN/Inf handling, seeded RNG |
| Generation | Prefill/decode, stop conditions, metrics, cancellation, stop reason |
| Server | Request validation, chat template, response format, batched routes |
| Protocol | Frame encoding, message roundtrip, CRC integrity, 20 message types (including election + reconnect) |
| Coordinator | Scheduler, registry, state, heartbeat, distributed pipeline, rollback, rebalance, drain |
| Election | ElectionAgent term progression, TermTracker split-brain prevention, seed discovery |
| Batch scheduler | Decode priority, prefill chunking, admission control, block pool reserve |
| TurboQuant | Compress/decompress round-trip (2/4/8-bit), norm preservation, zero vector safety, e2e A/B vs FP16 |
| Model validation | PyTorch reference comparison, golden generation, kernel correctness |
| E2E distributed | Multi-process coordinator + worker inference (gated behind `FRACTURE_MODEL_PATH`) |

### Reference Tensor Harness

Generate PyTorch reference data for numerical validation:

```bash
python scripts/dump_reference.py \
    --model-path /path/to/llama-3-8b-hf \
    --output-dir tests/reference

python scripts/dump_reference.py --golden \
    --model-path /path/to/llama-3-8b-hf \
    --output-dir tests/golden
```

## Design Decisions

**Opaque tensors.** `DeviceTensor` is a handle (ID + shape + dtype), not a pointer. The engine never touches device memory directly — all reads and writes go through `Backend` methods. This is what makes backend-swapping possible without engine changes.

**Static pipeline, dynamic batching.** The worker-to-layer assignment is fixed at startup (after calibration), but requests flow in and out of the batch continuously. This gives predictable memory layout with flexible throughput.

**CRC32C everywhere.** Every wire protocol frame is integrity-checked. The distributed pipeline runs over TCP across machines — silent corruption would be catastrophic for inference correctness.

**Decode-priority scheduling.** Active decodes are always scheduled before new prefills. A decode step is one token per sequence (fast, latency-sensitive). A prefill can be hundreds of tokens (slow, throughput-oriented). Starving decodes would spike time-to-first-token for every in-flight request.

**Priority bully election, not Raft.** Distributed consensus (Raft, Paxos) is designed for replicated state machines that must agree on a log. Fracture's coordinator holds no persistent state that requires consensus — workers hold their own weights, and the pipeline assignment can be recomputed from scratch on failover. A priority-based bully algorithm is sufficient: the highest-priority reachable node wins, term numbers prevent split-brain from a rejoining stale coordinator. This keeps the election crate small and the failure path deterministic.

**TurboQuant: MSE-only, no QJL.** The TurboQuant paper proposes QJL residual correction for unbiased inner products. Community validation across 6+ independent implementations showed that softmax exponentially amplifies QJL's variance, producing worse results than MSE-only quantization. Fracture implements the V3 MSE-only approach. The `PagedCache` trait keeps the forward pass generic — TurboQuant required zero changes to `batched_forward`, only a new trait implementation.

## WSL2 Notes

On WSL2, the CUDA driver library lives in `/usr/lib/wsl/lib/`. This is handled automatically:
- `.cargo/config.toml` sets `LD_LIBRARY_PATH` for both standard and WSL2 paths
- `libcudart` is linked statically (the dynamic version segfaults due to WSL2 CUDA forwarding layer mismatch)
- cuBLAS remains dynamically linked

If CUDA tests segfault, verify `nvidia-smi` works and `.cargo/config.toml` is present.

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 1 | Single-node inference (Llama 3 8B, CUDA, OpenAI API) | **Complete** |
| 2 | Node abstraction (layer-range execution, pipeline splits, IPC) | **Complete** |
| 3 | Distribution (wire protocol, scheduling, heartbeat, multi-machine) | **Complete** |
| 4 | Production (paged KV cache, continuous batching, distributed batching) | **Complete** |
| 4.5 | [TurboQuant](docs/turboquant.md) KV cache compression (5x memory reduction, ICLR 2026) | **Complete** |
| 5 | Network resilience (leader election, worker reconnection, seed discovery, dynamic join/leave) | **Complete** |
| 6 | Cross-platform (Metal backend for Apple Silicon — heterogeneous clusters) | Planned |

## License

MIT
