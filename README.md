# Fracture

Distributed cross-platform LLM inference engine in Rust with pluggable GPU backends.

**Phase 1** delivers single-node Llama 3 8B inference on a CUDA GPU, served over an OpenAI-compatible HTTP API. The engine is backend-agnostic — all GPU operations go through a `Backend` trait, keeping the door open for Metal (Phase 5+) and other backends without engine changes.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              HTTP Server (axum)                      │
│       /v1/completions    /v1/chat/completions        │
│                  SSE streaming                       │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Generation Loop                         │
│    prefill → [decode loop] → stream tokens           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Compute Engine                          │
│    forward(token_ids) → logits                       │
│    Generic over B: Backend — no CUDA imports         │
└──────┬───────────────────────────────┬──────────────┘
       │                               │
┌──────▼──────┐                 ┌──────▼──────┐
│  KV Cache   │                 │ Weight Store │
│  Manager    │                 │   (GGUF)     │
└─────────────┘                 └──────────────┘
```

### Crate Structure

| Crate | Purpose |
|---|---|
| `crates/fracture-core` | Backend trait, DeviceTensor, ModelConfig, error types |
| `crates/fracture-engine` | Backend-generic transformer forward pass, KV cache |
| `crates/fracture-generate` | Sampling, generation loop, chat template |
| `crates/fracture-server` | OpenAI-compatible HTTP API (axum) |
| `crates/fracture-gguf` | GGUF v3 file parser and weight loader |
| `backends/fracture-cuda` | CUDA Backend implementation (kernels + cuBLAS) |
| `bins/fracture-server-cuda` | Server binary wiring CUDA backend |

**Critical invariant:** Engine, generate, and server crates never import backend crates. All GPU ops go through the `Backend` trait in fracture-core.

## Requirements

- Rust (edition 2024)
- NVIDIA GPU with CUDA toolkit installed (`nvcc` on PATH)
- `cargo-nextest` for running tests

### WSL2 Notes

On WSL2, the CUDA driver library (`libcuda.so`) lives in `/usr/lib/wsl/lib/` rather than the standard Linux location. This project handles this automatically:

- **`.cargo/config.toml`** sets `LD_LIBRARY_PATH` to include both `/usr/local/cuda/lib64` and `/usr/lib/wsl/lib`, so `cargo nextest run` and `cargo run` work without manual environment setup.

- **Static cudart linking:** The CUDA runtime (`libcudart`) is linked statically (`libcudart_static.a`) rather than dynamically. On WSL2, the dynamic `libcudart.so` segfaults during initialization due to a driver/runtime version mismatch in the WSL2 CUDA forwarding layer. Static linking avoids this entirely. The cuBLAS libraries remain dynamically linked since they don't have this issue.

If you see segfaults in CUDA tests, verify that:
1. `nvidia-smi` works and shows your GPU
2. `/usr/lib/wsl/lib/libcuda.so` exists (WSL2) or `/usr/local/cuda/lib64/libcuda.so` exists (native Linux)
3. `.cargo/config.toml` is present with the `LD_LIBRARY_PATH` setting

## Build & Test

```bash
cargo check                    # Verify workspace compiles
cargo nextest run              # Run all tests (55 tests: unit + GPU kernel tests)
cargo nextest run -p fracture-cuda   # Run only CUDA GPU tests (17 tests)
cargo clippy                   # Lint
```

No manual `export LD_LIBRARY_PATH` is needed — `.cargo/config.toml` handles it.

### Test Coverage

| Tier | Scope | GPU Required | Count |
|---|---|---|---|
| Unit | Config validation, DType, tensor math | No | 13 |
| Unit | Sampling (temperature, top-k, top-p, greedy) | No | 7 |
| Unit | GGUF parsing (header, metadata, tensor info) | No | 5 |
| Unit | HTTP request validation | No | 9 |
| Unit | Chat template formatting | No | 4 |
| GPU | CUDA memory management (alloc/free/copy) | Yes | 4 |
| GPU | Kernel correctness (rmsnorm, rope, silu_mul, add, embedding) | Yes | 6 |
| GPU | cuBLAS matmul (M=1 decode + general) | Yes | 2 |
| GPU | Attention (single token + GQA head sharing) | Yes | 2 |
| GPU | GPU timers, synchronize, device info | Yes | 3 |

### Reference Tensor Harness

For numerical validation against PyTorch:

```bash
python scripts/dump_reference.py --model-path <path-to-llama3-8b-hf> --output-dir tests/reference
```

This dumps intermediate tensors (embeddings, per-layer activations, logits) that Tier 2 validation tests compare against.

## Running the Server

```bash
cargo run --release -p fracture-server-cuda -- \
    --model /path/to/llama-3-8b.gguf \
    --tokenizer /path/to/tokenizer.json \
    --port 8080
```

Then:
```bash
# Completions
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is", "max_tokens": 64, "temperature": 0}'

# Chat (streaming)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 64, "stream": true}'
```

## Conventions

- FP16 storage, FP32 accumulation for all compute
- Row-major tensor convention everywhere (cuBLAS column-major handled inside CUDA backend)
- DeviceTensor is an opaque handle — engine never touches device pointers
- All fallible operations return `Result<T, FractureError>` with context
- No panics in library crates
- Profiling is optional — zero overhead when disabled

## Project Phases

| Phase | Goal | Status |
|---|---|---|
| 1 | Single-node inference server (Llama 3 8B, CUDA) | **In progress** |
| 2 | Node abstraction (layer-range execution) | Planned |
| 3 | Distribution (wire protocol, peer discovery, scheduling) | Planned |
| 4 | Production (fault tolerance, continuous batching, paged KV cache) | Planned |
| 5 | Cross-platform (Metal backend for Apple Silicon) | Planned |
