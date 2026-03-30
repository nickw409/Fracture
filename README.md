<div align="center">

# Fracture

**Distributed LLM inference engine in Rust with pluggable GPU backends**

Split large language models across multiple GPUs on multiple machines.\
Pipeline-parallel execution over TCP. OpenAI-compatible API. Zero engine changes to add new backends.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2024_Edition-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-green.svg)](https://developer.nvidia.com/cuda-toolkit)

[Architecture](#architecture) · [Features](#features) · [Quick Start](#quick-start) · [Distributed Inference](#distributed-inference) · [Testing](#testing) · [Roadmap](#roadmap)

</div>

---

Fracture is a from-scratch LLM inference engine (~37k lines of Rust, ~650 lines of CUDA) designed around one principle: **the engine never knows what GPU it's running on**. All compute flows through a `Backend` trait, making the core engine, generation loop, HTTP server, and wire protocol completely backend-agnostic. Today that backend is CUDA; tomorrow it's Metal, and a single inference cluster can mix both.

The system runs Llama 3.1 8B with full numerical validation against PyTorch — greedy generation is token-for-token identical. It has been validated across heterogeneous hardware (RTX 5090 + RTX 3090) in multi-machine distributed inference.

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
| `fracture-core` | `Backend` trait, `DeviceTensor` (opaque GPU handle), `ModelConfig`, error types, profiling |
| `fracture-engine` | Transformer forward pass, contiguous + paged KV cache, batched forward, node abstraction, IPC, batch scheduler |
| `fracture-generate` | Sampling (temperature, top-k, top-p, seeded RNG), generation loop, cooperative cancellation |
| `fracture-server` | OpenAI-compatible HTTP API (axum), SSE streaming, batched + serialized modes |
| `fracture-gguf` | GGUF v3 file parser and weight loader (FP16/FP32/BF16) |
| `fracture-protocol` | Binary wire protocol over TCP (CRC32C integrity, 12 message types, 256 MB cap) |
| `fracture-coordinator` | Scheduler, peer registry, heartbeat, distributed pipeline (single + batched forward) |
| `fracture-cuda` | CUDA `Backend` implementation: 7 hand-written kernels + cuBLAS matmul |

## Features

### Inference Engine
- Full Llama 3.1 8B transformer: embedding, RMSNorm, GQA attention, RoPE, SwiGLU FFN
- FP16 storage with FP32 accumulation for numerical stability
- Greedy and stochastic sampling (temperature, top-k, top-p, seeded RNG)
- **Paged KV cache** — 16-token blocks allocated on demand (vs. contiguous pre-allocation), enabling dozens to hundreds of concurrent sequences
- **Batched forward pass** — processes multiple sequences in a single engine call
- **Continuous batching** — requests enter and leave the batch dynamically; decode-priority scheduling, prefill chunking (configurable, default 512 tokens)

### CUDA Backend
Hand-written kernels optimized for inference:
- RMSNorm, RoPE, SiLU×Mul, Embedding, Residual Add
- Contiguous and paged attention (paged reads K/V from block tables)
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
    --num-workers 2
```

Start a worker on machine B:

```bash
cargo run --release -p fracture-worker-cuda -- \
    --coordinator <machine-A-ip>:9410 \
    --model /path/to/llama-3-8b.gguf
```

The coordinator waits for all workers to register, runs calibration, assigns layers based on measured GPU performance, and starts serving. The HTTP API is identical — clients don't know inference is distributed.

### How Scheduling Works

Each worker runs calibration forward passes at registration. The scheduler uses these timings to assign layers proportional to each GPU's speed, then clamps assignments to fit in available memory. Slow workers can be pruned entirely. The result: a heterogeneous cluster (e.g., RTX 5090 + RTX 3090) where each GPU gets the workload it can handle.

## Testing

710 tests covering unit, GPU kernel, integration, model-validation, and end-to-end distributed scenarios.

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
| CUDA kernels | RMSNorm, RoPE, SiLU, attention (contiguous + paged), embedding, add, matmul |
| Engine | Forward pass, node dispatch, pipeline splits, KV cache, batched forward, IPC |
| Sampling | Temperature, top-k, top-p, greedy, NaN/Inf handling, seeded RNG |
| Generation | Prefill/decode, stop conditions, metrics, cancellation, stop reason |
| Server | Request validation, chat template, response format, batched routes |
| Protocol | Frame encoding, message roundtrip, CRC integrity, 12 message types |
| Coordinator | Scheduler, registry, state, heartbeat, distributed pipeline, rollback |
| Batch scheduler | Decode priority, prefill chunking, admission control, block pool reserve |
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
| 4 | Production (paged KV cache, continuous batching, distributed batching) | **In Progress** |
| 5 | Cross-platform (Metal backend for Apple Silicon — heterogeneous clusters) | Planned |

## License

MIT
