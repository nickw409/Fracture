# Fracture

Distributed cross-platform LLM inference engine in Rust with pluggable GPU backends.

Fracture splits large language models across multiple GPUs on multiple machines using pipeline parallelism. Each machine runs a worker process that owns a contiguous range of transformer layers. A coordinator process orchestrates activation passing between workers over TCP, schedules layer assignments based on measured GPU performance, and serves an OpenAI-compatible HTTP API.

The engine is backend-agnostic — all GPU operations go through a `Backend` trait, keeping the door open for Metal (Phase 5) and other backends without engine changes.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │       Coordinator (HTTP API)         │
                    │  /v1/completions  /v1/chat  /health  │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │      Distributed Pipeline            │
                    │  sequence state, cache lifecycle,    │
                    │  activation shape validation         │
                    └──┬───────────────┬──────────────┬───┘
                       │               │              │
              ┌────────▼───┐  ┌────────▼───┐  ┌──────▼─────┐
              │  Worker 0  │  │  Worker 1  │  │  Worker 2  │
              │ Layers 0-9 │  │ Layers 10-19│ │ Layers 20-31│
              │  (Head)    │  │  (Middle)  │  │  (Tail)    │
              └────────────┘  └────────────┘  └────────────┘
                   GPU 0           GPU 1           GPU 2
```

Each worker runs the same engine code — a backend-generic transformer forward pass over its assigned layer range. The coordinator chains activations through workers in order: the head worker embeds token IDs into activations, middle workers transform them through their layers, and the tail worker produces logits.

For single-node inference, a standalone server binary runs the full model on one GPU without the coordinator/worker overhead.

### Crate Structure

| Crate | Purpose |
|---|---|
| **Core** | |
| `crates/fracture-core` | Backend trait, DeviceTensor, ModelConfig, error types, profiling types |
| `crates/fracture-engine` | Transformer forward pass, KV cache, node abstraction, pipeline coordinator, IPC |
| `crates/fracture-generate` | Sampling (temperature/top-k/top-p/seeded), generation loop, cancellation |
| `crates/fracture-server` | OpenAI-compatible HTTP API with SSE streaming |
| `crates/fracture-gguf` | GGUF v3 parser and weight loader (FP16/FP32/BF16) |
| **Distributed** | |
| `crates/fracture-protocol` | Binary wire protocol (TCP, CRC32C, 10 message types) |
| `crates/fracture-coordinator` | Scheduler, peer registry, heartbeat, distributed pipeline |
| **Backend** | |
| `backends/fracture-cuda` | CUDA kernels + cuBLAS matmul |
| **Binaries** | |
| `bins/fracture-server-cuda` | Single-node server |
| `bins/fracture-worker-cuda` | Distributed worker (calibration, registration, forward serving) |
| `bins/fracture-coordinator-cuda` | Distributed coordinator (scheduling, HTTP API) |

**Critical invariant:** Engine, generate, server, and protocol crates never import backend crates. All GPU operations go through the `Backend` trait.

## Features

### Inference
- Full Llama 3 8B transformer: embedding, RMSNorm, GQA attention, RoPE, SwiGLU FFN
- FP16 storage with FP32 accumulation for numerical stability
- Greedy and stochastic sampling (temperature, top-k, top-p, seeded RNG)
- KV cache with contiguous per-sequence allocation
- Prefill + decode with cache consistency validated against PyTorch reference

### HTTP API (OpenAI-compatible)
- `POST /v1/completions` and `POST /v1/chat/completions` with Llama 3 chat template
- `GET /v1/models`, `GET /health`
- SSE streaming with `id`, `object`, `created`, `model`, `finish_reason`, `usage`
- `finish_reason`: `"stop"` (EOS token) vs `"length"` (max_tokens reached)
- Request validation: empty prompt, invalid temperature/top_p, unknown model, invalid chat roles
- Cooperative cancellation on client disconnect (frees GPU resources)

### Distributed Inference
- Custom binary wire protocol with CRC32C integrity and 256 MB payload limit
- Compute-balanced layer scheduling with slow-node pruning and memory-aware clamping
- Three scheduling modes: Auto (performance-optimized), EqualSplit (testing), Manual (explicit)
- Worker calibration: 20 single-layer forward passes (5 warmup + 15 averaged) for decode and prefill
- Heartbeat protocol with nonce-validated acks and dead-worker detection
- Cache lifecycle: alloc with partial-failure rollback, duplicate/unknown sequence detection, reuse after free
- Activation shape validation between pipeline stages

### Validation
- 542 tests: unit, GPU kernel, integration, model-validation, and e2e
- 229 specified behaviors in vexspec (1 deferred to Phase 4)
- Per-layer numerical validation against PyTorch reference tensors (rtol=1e-3, atol=1e-3)
- Golden output comparison: greedy generation matches PyTorch token-for-token
- Cross-machine inference validated with RTX 5090 + RTX 3090

## Requirements

- Rust (edition 2024)
- NVIDIA GPU with CUDA toolkit (`nvcc` on PATH)
- `cargo-nextest` for running tests
- GGUF model file (Llama 3 8B FP16 format)
- HuggingFace `tokenizer.json` for the model

## Build & Run

```bash
cargo check                    # Verify workspace compiles
cargo nextest run              # Run all 542 tests
cargo clippy                   # Lint
```

### Single-Node Server

```bash
cargo run --release -p fracture-server-cuda -- \
    --model /path/to/llama-3-8b.gguf \
    --tokenizer /path/to/tokenizer.json \
    --port 8080
```

```bash
# Completions
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is", "max_tokens": 64, "temperature": 0}'

# Chat (streaming)
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 64, "stream": true}'
```

### Distributed Inference (Multi-GPU)

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

The coordinator waits for all workers to register, runs the scheduler, assigns layers, and starts serving HTTP requests. The same API endpoints work — clients don't know inference is distributed.

## Test Coverage

| Category | Scope | Count |
|---|---|---|
| Core types | Config validation, DType, tensor shape/reshape, error types | 39 |
| GGUF parser | Header, metadata, tensor info, weight loading, BF16 conversion | 13 |
| CUDA kernels | RMSNorm, RoPE, SiLU, attention, embedding, add, matmul + shape validation | 101 |
| Engine | Forward pass, node dispatch, pipeline splits, KV cache, IPC | 65 |
| Sampling | Temperature, top-k, top-p, greedy, NaN/Inf, seeded, single-logit | 30 |
| Generation | Prefill/decode, stop conditions, metrics, cancellation, stop reason | 20 |
| Server | Request validation, chat template, response format | 24 |
| Protocol | Frame encoding, message roundtrip, CRC integrity, empty payloads | 17 |
| Coordinator | Scheduler, registry, state, heartbeat, distributed pipeline, rollback | 123 |
| Model validation | PyTorch reference comparison, golden generation, kernel correctness | 23 |
| GPU integration | End-to-end generation, memory lifecycle, streaming | 7 |
| E2E distributed | Multi-process coordinator + worker inference | 5 (ignored without model) |

Tests requiring the full model are gated behind `FRACTURE_MODEL_PATH` and the `#[ignore]` attribute for e2e tests.

### Reference Tensor Harness

```bash
# Dump PyTorch reference tensors
python scripts/dump_reference.py --model-path <path-to-llama3-8b-hf> --output-dir tests/reference

# Dump golden generation output
python scripts/dump_reference.py --golden --model-path <path-to-llama3-8b-hf> --output-dir tests/golden
```

## WSL2 Notes

On WSL2, the CUDA driver library lives in `/usr/lib/wsl/lib/` rather than the standard Linux location. This is handled automatically:

- `.cargo/config.toml` sets `LD_LIBRARY_PATH` for both paths
- `libcudart` is linked statically (the dynamic version segfaults on WSL2 due to a driver/runtime version mismatch in the CUDA forwarding layer)
- cuBLAS remains dynamically linked

If CUDA tests segfault, verify `nvidia-smi` works and `.cargo/config.toml` is present.

## Project Phases

| Phase | Goal | Status |
|---|---|---|
| 1 | Single-node inference (Llama 3 8B, CUDA, OpenAI API) | **Complete** |
| 2 | Node abstraction (layer-range execution, pipeline splits, IPC) | **Complete** |
| 3 | Distribution (wire protocol, scheduling, heartbeat, multi-machine) | **Complete** |
| 4 | Production (continuous batching, paged KV cache, fault tolerance) | Planned |
| 5 | Cross-platform (Metal backend for Apple Silicon) | Planned |

## License

MIT
