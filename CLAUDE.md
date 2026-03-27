# Fracture

Distributed cross-platform LLM inference engine in Rust with pluggable GPU backends.

## Architecture

Rust workspace with strict crate boundaries enforcing backend-agnosticism:

- `crates/fracture-core` — Backend trait, DeviceTensor, ModelConfig, error types
- `crates/fracture-engine` — Backend-generic transformer forward pass
- `crates/fracture-generate` — Sampling, tokenization, generation loop
- `crates/fracture-server` — OpenAI-compatible HTTP API (axum)
- `crates/fracture-gguf` — GGUF file parser and weight loader
- `backends/fracture-cuda` — CUDA Backend trait implementation
- `bins/fracture-server-cuda` — Server binary wiring CUDA backend

**Critical invariant:** Engine, generate, server, and protocol crates never import backend crates. All GPU ops go through the `Backend` trait in fracture-core.

## Conventions

- FP16 storage, FP32 accumulation for all compute
- Row-major tensor convention everywhere (cuBLAS column-major handled inside CUDA backend only)
- DeviceTensor is an opaque handle (TensorId + shape + dtype) — engine never touches device pointers
- All fallible operations return `Result<T, FractureError>` with context
- No panics in library crates

## Build & Test

```bash
cargo check          # Verify workspace compiles
cargo test           # Run all tests
cargo clippy         # Lint
```

CUDA backend requires NVIDIA GPU + CUDA toolkit. The workspace compiles without CUDA for non-GPU crates.
