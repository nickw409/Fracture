# Fracture

Distributed cross-platform LLM inference engine in Rust with pluggable GPU backends.

## Architecture

Rust workspace with strict crate boundaries enforcing backend-agnosticism:

- `crates/fracture-core` — Backend trait, DeviceTensor, ModelConfig, profiling types, error types
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
- DeviceTimer is an opaque handle (u64 ID) — backend maps to GPU timer resources (e.g., CUDA events)
- All fallible operations return `Result<T, FractureError>` with context
- No panics in library crates
- Profiling is always optional — zero overhead when disabled (no timer creation, no timing calls)

## Build & Test

```bash
cargo check          # Verify workspace compiles
cargo nextest run    # Run all tests (55 tests: unit + GPU kernel tests)
cargo clippy         # Lint
```

CUDA backend requires NVIDIA GPU + CUDA toolkit. The workspace compiles without CUDA for non-GPU crates.

**No manual `export LD_LIBRARY_PATH` is needed.** The `.cargo/config.toml` sets it automatically for all cargo commands (including nextest). This is required on WSL2 where the CUDA driver lives in `/usr/lib/wsl/lib/`.

### WSL2 CUDA Linking

On WSL2, `libcudart.so` (dynamic) segfaults during initialization due to a version mismatch in the WSL2 CUDA forwarding layer. The build.rs links `libcudart_static.a` instead, which works correctly. cuBLAS remains dynamically linked. If CUDA tests segfault, check that `.cargo/config.toml` exists and `nvidia-smi` works.
