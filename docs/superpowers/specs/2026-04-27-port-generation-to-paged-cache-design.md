# Port `GenerationLoop` and Golden Tests to Paged KV Cache

**Date:** 2026-04-27
**Status:** Approved design (pending spec self-review and user sign-off)
**Sub-project:** Architecture cleanup, sub-project 1a (precursor to 1b "remove stepping stones")

## Background

The Fracture codebase currently has two coexisting KV cache implementations:

- **Contiguous** (`KvCacheManager`) — Phase 1–3 design, default for single-sequence generation
- **Paged** (`PagedKvCacheManager` and `QuantizedKvCacheManager`) — Phase 4 design, required for continuous batching

`Engine::forward()` and `GenerationLoop::generate()` are still bound to the contiguous cache. The paged path is reachable only through `Engine::forward_paged()` and the `BatchScheduler` in `scheduler_loop.rs`.

The umbrella architecture-cleanup project will delete the contiguous cache as a stepping stone. Before that deletion can happen, the single-sequence golden test (`tests/model-validation/tests/golden.rs`) and the generation unit tests (`crates/fracture-generate/src/generation/generation_tests.rs`) must be ported to the paged path. This spec covers that port.

A separate `PagedCache` trait already exists in `crates/fracture-engine/src/batched.rs` and is implemented by both `PagedKvCacheManager` (FP16) and `QuantizedKvCacheManager` (TurboQuant). `batched_forward` is generic over it. Making `GenerationLoop` and `Engine::forward_paged` generic over the same trait is therefore near-zero abstraction cost.

## Goal

Get the golden generation tests and the `GenerationLoop` unit tests running on the paged cache infrastructure, so sub-project 1b can delete the contiguous cache without losing correctness coverage.

## Scope

### In scope

- Make `Engine::forward_paged` and `Engine::forward_node_paged` generic over `<C: PagedCache>`. Switch internal calls from concrete `PagedKvCacheManager` methods to the trait methods (`append_kv`, `dispatch_attention`).
- Make `GenerationLoop::generate`, `generate_with_cancel`, and `generate_inner` generic over `<B: Backend, C: PagedCache>`. Swap `engine.forward(...)` for `engine.forward_paged(...)`. Drop the backend argument from `cache.alloc(...)` and `cache.free(...)` (signature drift between `KvCacheManager` and `PagedCache`).
- Extend the local `MockBackend` and `FailingMockBackend` in `crates/fracture-generate/src/generation/generation_tests.rs` with no-op implementations of any `Backend` trait methods required by the paged path (precedent: `crates/fracture-engine/src/engine/engine_tests.rs`).
- Update the `make_cache(cfg)` helper in `generation_tests.rs` to construct a `PagedKvCacheManager` instead of `KvCacheManager`. Block count: `(cfg.max_seq_len.div_ceil(16)) + 2` (block size 16 is hardcoded per CLAUDE.md; +2 blocks of safety margin in case a test exercises one-token-past-limit error paths).
- Re-export `PagedCache` from `fracture_engine` (currently lives in `batched` module; visibility must allow external use).
- Create `bins/fracture-server-cuda/tests/golden.rs` with all helpers inlined. Test runs greedy generation through paged cache and compares against `tests/golden/prompt_*_greedy_50.bin`. Helpers required:
  - `golden_dir() -> PathBuf` — returns workspace-relative path to `tests/golden/`.
  - `load_golden_tokens(path: &str) -> std::io::Result<Vec<u32>>` — reads `.bin` of little-endian `u32` tokens.
  - `compare_token_sequences(actual: &[u32], expected: &[u32])` — asserts equality, logs first divergence index and surrounding tokens (lifted from `tests/validation/src/lib.rs::golden_compare`).
  - `setup_real_engine() -> Option<(Engine<CudaBackend>, ModelConfig)>` — reads `FRACTURE_MODEL_PATH`, returns `None` if unset; loads GGUF, constructs `CudaBackend::new(0)?` and `Engine::new(...)`.
  - `skip!` macro — local copy of the 5-line macro at `tests/model-validation/src/lib.rs:75`.

### Out of scope (handled in sub-project 1b)

- Removing `KvCacheManager`, `Engine::forward()`, `KvCacheBackend::Contiguous`
- Removing `routes.rs`, `gpu_integration.rs`, `tests/model-validation/`, `tests/reference/`, `tests/validation/`
- Removing the sequential distributed path or the coordinator's `--batched` flag
- Removing the Python reference scripts (`scripts/dump_reference.py`, `scripts/verify_reference.py`)
- Updating `vexspec.yaml`, `README.md`, or `CLAUDE.md` (docs still describe today's reality during 1a)

### Definition of done

- `cargo nextest run` is green across the full workspace.
- `cargo clippy --workspace --all-targets -- -D warnings` is clean.
- `bins/fracture-server-cuda/tests/golden.rs` runs and passes when `FRACTURE_MODEL_PATH` is set; skips gracefully when not.
- `KvCacheManager` and `Engine::forward` are still in-tree at the end of 1a, just with no callers from `fracture-generate` or the new golden test. Their removal is 1b's job.

## API changes

### `crates/fracture-engine/src/engine.rs`

```rust
// Before
pub fn forward_paged(
    &self,
    token_ids: &[u32],
    positions: &[u32],
    cache: &mut PagedKvCacheManager,
    cache_handle: CacheHandle,
) -> Result<Vec<f32>>

// After
pub fn forward_paged<C: PagedCache>(
    &self,
    token_ids: &[u32],
    positions: &[u32],
    cache: &mut C,
    cache_handle: CacheHandle,
) -> Result<Vec<f32>>
```

`forward_node_paged` receives the same `<C: PagedCache>` parameter. Internal calls switch from concrete `PagedKvCacheManager` methods to trait methods.

### `crates/fracture-generate/src/generation.rs`

```rust
// Before
pub fn generate<B: Backend>(
    engine: &Engine<B>,
    prompt_tokens: &[u32],
    config: &GenerationConfig,
    cache: &mut KvCacheManager,
    tx: &mpsc::UnboundedSender<u32>,
) -> Result<GenerationResult>

// After
pub fn generate<B: Backend, C: PagedCache>(
    engine: &Engine<B>,
    prompt_tokens: &[u32],
    config: &GenerationConfig,
    cache: &mut C,
    tx: &mpsc::UnboundedSender<u32>,
) -> Result<GenerationResult>
```

Same `<C: PagedCache>` addition for `generate_with_cancel` and `generate_inner`. Internal `engine.forward(...)` calls become `engine.forward_paged(...)`. `cache.alloc(engine.backend())?` becomes `cache.alloc()?`. `cache.free(handle, engine.backend())` becomes `cache.free(handle)`.

### Imports

- `generation.rs`: drop `use fracture_engine::KvCacheManager`, add `use fracture_engine::PagedCache`.
- `fracture_engine/src/lib.rs`: add `pub use batched::PagedCache;` if not already public.

### Surface that does not change

- `GenerationConfig`, `GenerationResult`, `StopReason`, `apply_chat_template`
- `CacheHandle` type
- mpsc streaming protocol (one `u32` per generated token)
- Cooperative cancellation via `Arc<AtomicBool>`

### Error handling on cache OOM

`PagedCache::alloc()` can fail if no free blocks remain (paged caches are bounded; contiguous wasn't). `GenerationLoop` surfaces the cache's existing error as-is — no new error variants, no wrapping.

## File-level changes

### Modified

- `crates/fracture-engine/src/engine.rs` — `forward_paged` and `forward_node_paged` become generic; route internals through trait methods.
- `crates/fracture-engine/src/lib.rs` — re-export `PagedCache`.
- `crates/fracture-generate/src/generation.rs` — generic parameter additions, `forward` → `forward_paged` swap, alloc/free signature drift fixes.
- `crates/fracture-generate/src/generation/generation_tests.rs` — extend `MockBackend` and `FailingMockBackend` with no-op paged Backend trait methods; update `make_cache` helper to build `PagedKvCacheManager`.

### Created

- `bins/fracture-server-cuda/tests/golden.rs` — relocated golden test with all helpers inlined. Estimated 150–250 lines. Reads from `tests/golden/` (data unchanged), uses real `CudaBackend`, depends on `FRACTURE_MODEL_PATH` and skips gracefully when missing.

### Unchanged but referenced

- `tests/golden/` — data files preserved as-is.
- `tests/model-validation/`, `tests/validation/`, `tests/reference/`, Python reference scripts — remain in-tree at end of 1a; deletion is 1b.

### Why inline helpers

`fracture_validation` and `fracture_model_validation` crates are slated for deletion in 1b. Depending on them from the new golden test would chain 1b's deletion into a multi-step port. Inlining now makes 1b a pure delete.

## Test strategy

### Discipline

This is a refactor; semantics are preserved. TDD here means:

- **Baseline first:** capture green test results on `main` before any change.
- **Per-step verification:** every modification gets `cargo nextest run` before moving to the next step. No batched compile-then-test cycles.
- **Three independent commits:** engine refactor, generate refactor, golden relocation. Each independently revertible.

### Execution plan

1. Baseline: `cargo nextest run` on main; confirm `tests/model-validation/tests/golden.rs` skips cleanly without `FRACTURE_MODEL_PATH`.
2. Engine refactor: make `forward_paged` and `forward_node_paged` generic. Run `cargo nextest run -p fracture-engine` and the GPU tests in `bins/fracture-server-cuda/tests/`. Expect green.
3. Generate refactor: add `<C: PagedCache>` to `GenerationLoop`, swap forward call, update `make_cache`, extend MockBackends. Run `cargo nextest run -p fracture-generate`. Expect green.
4. Golden relocation: create `bins/fracture-server-cuda/tests/golden.rs` with inlined helpers. Run with `FRACTURE_MODEL_PATH` set — must produce identical tokens to reference. Run without — must skip gracefully.
5. Final: `cargo nextest run` on full workspace + `cargo clippy --workspace --all-targets -- -D warnings`.

### Verification before completion

Five outputs must appear in the completion summary (no "I think it works"):

- `cargo nextest run` (full workspace) — green.
- `cargo clippy --workspace --all-targets -- -D warnings` — clean.
- `FRACTURE_MODEL_PATH=... cargo nextest run -p fracture-server-cuda --test golden` — passing.
- `cargo nextest run -p fracture-server-cuda --test golden` (no env var) — skips gracefully.
- `git status` — clean tree, no surprise modifications.

### Not tested in 1a

- TurboQuant golden test runs. The new `<C: PagedCache>` signature makes this trivially possible, but adding TQ-specific assertions is its own follow-up.
- Removal of contiguous path. That is 1b.
- Multi-GPU or concurrent-sequence generation (out of scope generally).

## Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|---|---|---|
| 1 | `forward_paged` trait-dispatch refactor changes numerics vs. concrete-method path | Low | High — silent corruption | Run paged GPU tests + new golden test before merge; logits must match pre-refactor bit-for-bit. |
| 2 | MockBackend extensions trigger latent test assumptions on paged outputs | Medium | Medium — easy-to-chase failures | No-op implementations are safe because generate-crate tests assert on sampling/positions/stop logic, not attention output values. |
| 3 | `cache.alloc()` / `cache.free()` signature drift cascades unexpectedly | Low | Low | Mechanical; compiler enforces every call site. |
| 4 | Inlined `golden_compare` differs subtly from original (endianness, alignment) | Low | Medium — false-positive regression | Read `tests/validation/src/lib.rs` carefully; copy verbatim where possible; sanity-check by hashing intermediate outputs. |
| 5 | `tests/model-validation/` removal during 1a accidentally breaks something | N/A in 1a | — | Not deleted in 1a; original `golden.rs` stays until 1b. |
| 6 | Block-pool sizing differs between contiguous and paged tests | Medium | Low | `make_cache` computes `(cfg.max_seq_len.div_ceil(16)) + 2` blocks; documented in helper. |
| 7 | `PagedCache` trait isn't `pub` re-exported from `fracture_engine` | Likely | Low | Add the re-export as part of the engine change. |

### Known unknowns

- Block count for unit tests. With `cfg.max_seq_len = 64` and 16-token blocks, 4 blocks per sequence + slack should suffice. Verify on first run.
- Paged-only `Backend` trait methods (e.g., `copy_rows` variants) that may need MockBackend implementations. Compiler will surface; treat as a discovered checklist.
- `Drop` ordering for paged cache when a generation test panics mid-test. Existing engine paged tests presumably handle this; verify nothing new is needed.

### Rollback strategy

- Three independent commits (engine, generate, golden) — any one revertible without affecting the others.
- No deletions in 1a. `KvCacheManager`, `Engine::forward`, `routes.rs`, full reference harness all still present at end of 1a. Worst case revert leaves contiguous path wired up identically to today.
- New golden test in `bins/fracture-server-cuda/tests/golden.rs` is purely additive — original `tests/model-validation/tests/golden.rs` is untouched and still runnable.

## Decision log

- **Approach: B (cache-generic via existing `PagedCache` trait)**, not A (direct swap to concrete `PagedKvCacheManager`). The trait already exists and is used by `batched_forward`; the abstraction cost is near-zero and we keep the option to run the same single-sequence test against `QuantizedKvCacheManager` later.
- **Golden test location: `bins/fracture-server-cuda/tests/`**, not `crates/fracture-generate/tests/`. The latter would force `fracture-cuda` to become a dev-dependency of `fracture-generate`, violating the project invariant "Engine, generate, server, and protocol crates never import backend crates" — even as a dev-dep it would require CUDA to test the generate crate.
- **Helpers: inlined**, not depended on. The `fracture_validation` and `fracture_model_validation` crates are slated for deletion in 1b; depending on them would turn 1b into a chain of helper ports. Inlining now makes 1b a pure delete.
- **GenerationLoop: rebuilt on paged**, not deleted. Single-sequence generation API has real value for unit testing (sampling, stop tokens, position tracking, cancellation) that would be much heavier to test through the async `BatchScheduler`. The contiguous binding is a stepping stone; the API itself is not.
- **OOM error handling: surface as-is.** `PagedCache::alloc()` already returns `Result<CacheHandle>`. No new error variants.
- **TurboQuant golden runs: deferred.** Free capability with the new generic signature; opt-in correctness assertions are a follow-up.
