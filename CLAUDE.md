# Fracture

Distributed cross-platform LLM inference engine in Rust with pluggable GPU backends.

## Architecture

Rust workspace with strict crate boundaries enforcing backend-agnosticism:

### Core Crates
- `crates/fracture-core` — Backend trait, DeviceTensor, ModelConfig, StopReason, profiling types, error types, TurboQuantConfig, Lloyd-Max solver, rotation matrix generation
- `crates/fracture-engine` — Backend-generic transformer forward pass, KV cache (contiguous + paged + quantized), PagedCache trait, node abstraction, pipeline coordinator, IPC transport, batch scheduler, batched forward
- `crates/fracture-generate` — Sampling (temperature/top-k/top-p/seeded), generation loop with StopReason, cooperative cancellation
- `crates/fracture-server` — OpenAI-compatible HTTP API (axum) with SSE streaming, finish_reason, usage stats. Two modes: Phase 3 Mutex-serialized (`routes.rs`) and Phase 4 batched (`batched_routes.rs` + `scheduler_loop.rs`)
- `crates/fracture-gguf` — GGUF file parser and weight loader (FP16/FP32/BF16)

### Distributed Inference (Phase 3)
- `crates/fracture-protocol` — Binary wire protocol (TCP, CRC32C integrity, 12 message types including BatchedForward/Result)
- `crates/fracture-coordinator` — Scheduler, peer registry, sequence state, heartbeat, distributed pipeline (single + batched forward)

### Backend & Binaries
- `backends/fracture-cuda` — CUDA Backend trait implementation (paged attention, TurboQuant compress/decompress/attention kernels)
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

- **Architecture docs are the spec.** When `docs/fracture_phase*_architecture.md` specifies a design (trait, concurrency model, module boundary), implement that design. Do not substitute a simpler ad-hoc pattern because the correct approach is harder. If the specified approach feels too complex, flag it — don't silently downgrade.
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
- Paged KV cache uses 16-token blocks; block_size is hardcoded (kernel tuned for it)
- Both contiguous and paged KV cache coexist via `KvCacheBackend` enum; contiguous is default until paged is validated on production model
- BatchScheduler uses decode-priority policy: active decodes always scheduled before new prefills
- Prefill chunking splits prompts > `max_prefill_tokens` (default 512, configurable) across iterations
- TurboQuant uses separate K and V rotation matrices per layer (different seeds) — never share a single rotation for both
- TurboQuant codebook centroids scale with `1/sqrt(head_dim)` — codebooks are dimension-specific, not universal
- `PagedCache` trait abstracts FP16 paged and TurboQuant quantized caches; `batched_forward` is generic over it
- TurboQuant compress scratch tensors are pre-allocated (sized for max bit-width across layers) to avoid per-call cudaMalloc

## Build & Test

```bash
cargo check          # Verify workspace compiles
cargo nextest run    # Run all tests (772 tests: unit + GPU kernel + integration + e2e)
cargo clippy         # Lint
```

**Always use `cargo nextest run`, never `cargo test`.** Nextest enforces test groups in `.config/nextest.toml` that serialize GPU-memory-sensitive and e2e tests. `cargo test` ignores these groups and will cause OOM or port conflicts.

CUDA backend requires NVIDIA GPU + CUDA toolkit. The workspace compiles without CUDA for non-GPU crates.

**No manual `export LD_LIBRARY_PATH` is needed.** The `.cargo/config.toml` sets it automatically for all cargo commands (including nextest). This is required on WSL2 where the CUDA driver lives in `/usr/lib/wsl/lib/`.

### WSL2 CUDA Linking

On WSL2, `libcudart.so` (dynamic) segfaults during initialization due to a version mismatch in the WSL2 CUDA forwarding layer. The build.rs links `libcudart_static.a` instead, which works correctly. cuBLAS remains dynamically linked. If CUDA tests segfault, check that `.cargo/config.toml` exists and `nvidia-smi` works.

## Phase 4 Progress

Phase 4 adds production inference capabilities. See `docs/fracture_phase4_architecture.md` for full design.

### Completed (Steps 1-3)

**Step 1 — Paged KV Cache:**
- `BlockPool` (`paged_kv_cache.rs`): pre-allocated GPU memory blocks with free list
- `PagedKvCacheManager`: per-sequence block tables, auto-growing append, OOM detection
- `attention_paged` CUDA kernel (`attention_paged.cu`): reads K/V from block tables
- `Backend::attention_paged()` trait method with DeviceTensor-based interface
- `Engine::forward_paged()` / `forward_node_paged()`: paged engine forward path
- GPU validated: bit-identical to contiguous (max_diff=0.000000 across 35 tokens, 3 blocks)

**Step 2 — Batched Forward Pass:**
- `batched_forward()` (`batched.rs`): processes multiple sequences in one call
- Concatenates tokens for matmul/RMSNorm/RoPE/FFN, dispatches attention per-sequence
- GPU validated: batched = sequential (max_diff=0.000000 for each sequence)

**Step 3 — Continuous Batching:**
- `BatchScheduler` (`scheduler.rs`): decode-priority, prefill chunking, admission control, block pool reserve
- `scheduler_loop.rs`: tokio task driving batched_forward, per-sequence sampling, token streaming
- `batched_routes.rs`: async HTTP handlers that enqueue via SchedulerHandle (no Mutex)
- GPU validated: 3 concurrent requests produce identical tokens to sequential reference

### Completed (Step 4 — Distributed Batching)

- Wire protocol: `BatchedForward` (0x0C) and `BatchedForwardResult` (0x0D) message types
- `SequenceMetadataWire`: per-sequence positions, block_table, cache_seq_len
- `HeartbeatAckPayload.free_blocks`: block pool stats in heartbeat
- `DistributedPipeline::batched_forward()`: coordinator sends batched payloads through pipeline
- `batched_forward_node()` (`batched.rs`): pipeline-aware batched forward with head/middle/tail node roles
- Worker `PagedKvCacheManager` initialized at startup (sized from available GPU memory, 512MB scratch reserve)
- Worker `BatchedForward` handler: deserializes payload, builds SequenceSlice from wire metadata, calls `batched_forward_node()`, returns `BatchedForwardResultPayload`
- Worker heartbeat reports `free_blocks` from paged cache
- Worker `CacheAlloc`/`CacheFree` manage both contiguous and paged cache handles
- `WorkerEntry.free_blocks`: per-worker paged cache stats stored from heartbeat acks
- `PeerRegistry::min_free_blocks()`: bottleneck detection across pipeline workers for admission control
- `HeartbeatTracker::process_ack()` passes `free_blocks` through to registry
- `distributed_loop.rs`: distributed scheduler loop using `pipeline.batched_forward()` with admission control, chunked prefill, per-sequence sampling, token streaming via `GenerationEvent` channels
- Coordinator `--batched` flag switches from sequential `distributed_generate` to the batched scheduler loop with `create_batched_router()`
- E2e validated: distributed batched output matches sequential greedy output for multiple prompts
- E2e validated: concurrent requests through batched distributed pipeline produce correct independent outputs
- Throughput benchmark: measures tokens/second under concurrent load

### Completed (Step 4.5 — TurboQuant KV Cache Compression)

- Google's TurboQuant (ICLR 2026) integrated as opt-in `--kv-quant turboquant` mode
- Community-validated V3 algorithm: MSE-only (no QJL — fails under softmax), rotation + Lloyd-Max quantization
- `PagedCache` trait in `batched.rs`: both `PagedKvCacheManager` (FP16) and `QuantizedKvCacheManager` implement it; `batched_forward` and `batched_forward_node` are generic over `C: PagedCache`
- `TurboQuantConfig` (`turboquant.rs`): key_bits, value_bits, protected_bits, protected_layers, seed
- `QuantizedBlockPool` + `QuantizedKvCacheManager` (`quantized_paged_kv_cache.rs`): pre-allocated compressed blocks with per-layer bit widths, rotation matrices, and codebooks on device
- `CompressScratch`: pre-allocated scratch tensors to avoid per-call cudaMalloc
- Three CUDA kernels: `turboquant_compress.cu` (fused normalize/rotate/quantize/pack), `turboquant_decompress.cu` (test utility), `attention_paged_tq.cu` (fused decompress+attention with query pre-rotation optimization)
- `Backend::turboquant_compress()` and `Backend::attention_paged_tq()` trait methods with default error returns
- Worker/coordinator `--kv-quant turboquant` CLI flags; worker serve loop dispatches via PagedCache trait
- Asymmetric K/V bits (default K4/V2 = 5.1x compression), per-layer protection (8-bit edge layers)
- Separate K/V rotation matrices per layer (distinct seeds prevent V corruption during query pre-rotation)
- GPU validated: TQ K4/V2 logit cosine similarity = 0.999 vs FP16; identical greedy argmax
- See `docs/turboquant.md` for full technical documentation

### Planned (Steps 5-6)

- Step 5: Fault tolerance (worker failure → abort sequences → rebuild pipeline)
- Step 6: Pipeline micro-batching (overlap pipeline stages for ~100% utilization)

## Vex

[Vex](https://github.com/nickw409/vex) is a behavioral spec and test gap analysis tool. Install via `go install github.com/nickw409/vex@latest`.

- `.vex/vexspec.yaml` — The behavioral specification. Organized into 39 sections covering core types, kernels, engine, server, protocol, coordinator, paged cache, batched forward, distributed batching, and scheduler.
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
