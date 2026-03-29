# Fracture

Distributed cross-platform LLM inference engine in Rust with pluggable GPU backends.

## Architecture

Rust workspace with strict crate boundaries enforcing backend-agnosticism:

### Core Crates
- `crates/fracture-core` — Backend trait, DeviceTensor, ModelConfig, profiling types, error types
- `crates/fracture-engine` — Backend-generic transformer forward pass, KV cache, node abstraction, pipeline coordinator, IPC transport
- `crates/fracture-generate` — Sampling (temperature/top-k/top-p/seeded), generation loop with StopReason, cooperative cancellation
- `crates/fracture-server` — OpenAI-compatible HTTP API (axum) with SSE streaming, finish_reason, usage stats
- `crates/fracture-gguf` — GGUF file parser and weight loader (FP16/FP32/BF16)

### Distributed Inference (Phase 3)
- `crates/fracture-protocol` — Binary wire protocol (TCP, CRC32C integrity, 10 message types)
- `crates/fracture-coordinator` — Scheduler, peer registry, sequence state, heartbeat, distributed pipeline

### Backend & Binaries
- `backends/fracture-cuda` — CUDA Backend trait implementation
- `bins/fracture-server-cuda` — Single-node server binary
- `bins/fracture-worker-cuda` — Distributed worker binary (calibration, registration, forward serving)
- `bins/fracture-coordinator-cuda` — Distributed coordinator binary (scheduling, pipeline orchestration, HTTP API)

**Critical invariant:** Engine, generate, server, and protocol crates never import backend crates. All GPU ops go through the `Backend` trait in fracture-core.

### Validation Infrastructure

- `scripts/dump_reference.py` — Dumps PyTorch reference tensors (Llama 3.1 8B) to `tests/reference/`
- `scripts/verify_reference.py` — Sanity-checks dumped tensors (shape, NaN/Inf, stats)
- `scripts/requirements.txt` — Python dependencies for the reference harness
- `tests/reference/` — Per-layer intermediate tensors from PyTorch (gitignored `.bin` files, regenerate locally)
- `tests/golden/` — Greedy generation golden outputs (gitignored `.bin` files)
- `tests/validation/` — Standalone Rust crate for loading and comparing reference tensors (not in workspace)
- `tests/model-validation/` — Integration tests comparing engine output against PyTorch reference (requires `FRACTURE_MODEL_PATH`)

### Test Organization

Tests are serialized by GPU memory sensitivity via `.config/nextest.toml`:
- `gpu-memory-sensitive` group (max-threads=1): model-validation tests (load full 15GB model), memory measurement tests, large matmul tests
- `e2e-distributed` group (max-threads=1): e2e tests that spawn coordinator + worker processes
- Model-validation tests use `max_seq_len=2048` (not the model's 128K) to avoid OOM

## Conventions

- FP16 storage, FP32 accumulation for all compute
- Row-major tensor convention everywhere (cuBLAS column-major handled inside CUDA backend only)
- DeviceTensor is an opaque handle (TensorId + shape + dtype) — engine never touches device pointers. Use `try_new()` for validated construction, `new()` for infallible (test/internal) use
- DeviceTimer is an opaque handle (u64 ID) — backend maps to GPU timer resources (e.g., CUDA events)
- All fallible operations return `Result<T, FractureError>` with context
- No panics in library crates
- Profiling is always optional — zero overhead when disabled (no timer creation, no timing calls)
- GenerationLoop returns `GenerationResult` with `StopReason::Stop` (EOS) or `StopReason::Length` (max_tokens)
- Cooperative cancellation via `Arc<AtomicBool>` — checked each decode iteration
- SSE streaming includes `id`, `object`, `created`, `model`, `finish_reason`, and `usage` in the final chunk
- Wire protocol uses CRC32C integrity checks; all payloads capped at 256 MB
- Distributed pipeline tracks cache lifecycle: alloc with rollback on partial failure, duplicate/unknown checks, reuse after free

## Build & Test

```bash
cargo check          # Verify workspace compiles
cargo nextest run    # Run all tests (542 tests: unit + GPU kernel + integration + e2e)
cargo clippy         # Lint
```

CUDA backend requires NVIDIA GPU + CUDA toolkit. The workspace compiles without CUDA for non-GPU crates.

**No manual `export LD_LIBRARY_PATH` is needed.** The `.cargo/config.toml` sets it automatically for all cargo commands (including nextest). This is required on WSL2 where the CUDA driver lives in `/usr/lib/wsl/lib/`.

### WSL2 CUDA Linking

On WSL2, `libcudart.so` (dynamic) segfaults during initialization due to a version mismatch in the WSL2 CUDA forwarding layer. The build.rs links `libcudart_static.a` instead, which works correctly. cuBLAS remains dynamically linked. If CUDA tests segfault, check that `.cargo/config.toml` exists and `nvidia-smi` works.

## Vex

[Vex](https://github.com/nickw409/vex) is a behavioral spec and test gap analysis tool. Install via `go install github.com/nickw409/vex@latest`.

- `.vex/vexspec.yaml` — The behavioral specification (229 implemented, 1 not-implemented, 2 not-tested). Organized into 30 sections covering core types, kernels, engine, server, protocol, coordinator, and e2e validation.
- `.vex/report.json` — Gap analysis output. Generated by `vex check`.
- `.vex/validation.json` — Validation status. Generated by `vex validate`.

```bash
vex validate         # Validate the vexspec is well-formed
vex check            # Check code against the spec (gap analysis)
vex check --drift=false  # Force full re-check of all sections
vex guide            # Print usage guide
```

The vexspec is the source of truth for what's implemented, what's tested, and what gaps remain. When adding new functionality or tests, update the spec to reflect the new status.

### Vexspec conventions
- Behaviors marked `not-implemented` or `not-tested` are tracked gaps
- `covered` overrides mark behaviors tested via cross-process boundaries (e2e binary spawns, IPC) that vex cannot trace
- `dismissed` entries suppress validate suggestions that were intentionally excluded (with reason)
- Sections are kept under 10 behaviors for parallel checking and fine-grained drift detection
