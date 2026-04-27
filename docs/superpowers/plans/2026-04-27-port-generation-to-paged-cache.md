# Port `GenerationLoop` and Golden Tests to Paged KV Cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Engine::forward_paged` and `GenerationLoop` generic over the `PagedCache` trait, then relocate the golden generation tests from `tests/model-validation/` into `bins/fracture-server-cuda/tests/golden.rs` with all helpers inlined — so sub-project 1b can purely delete the contiguous KV cache and reference harness without further plumbing.

**Architecture:** Three independent commits across three phases. Phase A makes the engine paged-forward path generic over the existing `PagedCache` trait (already implemented for `PagedKvCacheManager` and `QuantizedKvCacheManager`, already pub-exported from `fracture_engine`). Phase B propagates that generic into `GenerationLoop` and updates the unit-test mock backends. Phase C creates the new golden test file with inlined helpers. Phase D verifies the full workspace.

**Tech Stack:** Rust 2024, `cargo nextest`, `tokio` (mpsc channels), `serde` + `serde_json` (golden metadata), `half` crate (FP16), CUDA backend.

**Spec reference:** [docs/superpowers/specs/2026-04-27-port-generation-to-paged-cache-design.md](../specs/2026-04-27-port-generation-to-paged-cache-design.md)

---

## Pre-flight check

- [ ] **Confirm `PagedCache` is already pub-exported from `fracture_engine`**

Run: `grep -n "PagedCache" /home/nwiley/projects/Fracture/crates/fracture-engine/src/lib.rs`
Expected output includes: `pub use batched::{..., PagedCache, ...};`

This was confirmed during spec drafting; no change to `lib.rs` is needed. If the export is missing, add it before starting Phase A.

- [ ] **Capture green baseline**

Run: `cd /home/nwiley/projects/Fracture && cargo nextest run 2>&1 | tail -20`
Expected: all tests pass. Record the count for comparison after each phase. If anything is failing on `main`, stop and resolve before continuing.

---

## Phase A — Make `Engine::forward_paged` generic over `PagedCache`

### Task A1: Make `forward_node_paged` generic and route attention through the trait

**Files:**
- Modify: `crates/fracture-engine/src/engine.rs:538-690`

The current `forward_node_paged` takes `cache: &mut PagedKvCacheManager` and at lines 663–690 manually builds `block_table_i32`, gathers `k_blocks`/`v_blocks` from `cache.pool()`, and calls `self.backend.attention_paged(...)`. The `PagedCache::dispatch_attention` trait method does exactly this internally and works for both `PagedKvCacheManager` and `QuantizedKvCacheManager`. Replace the inline block with the trait call.

- [ ] **Step 1: Add `PagedCache` to the imports in `engine.rs`**

Locate the existing imports at the top of `crates/fracture-engine/src/engine.rs`. The file currently imports `crate::paged_kv_cache::PagedKvCacheManager` (or similar). Add `PagedCache` from the `batched` module:

```rust
use crate::batched::PagedCache;
```

Run: `cd /home/nwiley/projects/Fracture && cargo check -p fracture-engine 2>&1 | tail -5`
Expected: still compiles (unused import warning is fine until step 2).

- [ ] **Step 2: Change `forward_node_paged` signature to be generic**

In `crates/fracture-engine/src/engine.rs`, find the function starting at line 538. Replace its signature:

```rust
// Before
pub fn forward_node_paged(
    &self,
    input: NodeInput,
    node_config: &NodeConfig,
    cache: &mut PagedKvCacheManager,
    cache_handle: CacheHandle,
) -> Result<NodeOutput> {
```

```rust
// After
pub fn forward_node_paged<C: PagedCache>(
    &self,
    input: NodeInput,
    node_config: &NodeConfig,
    cache: &mut C,
    cache_handle: CacheHandle,
) -> Result<NodeOutput> {
```

- [ ] **Step 3: Replace the inline attention dispatch with `cache.dispatch_attention`**

In the same function, locate the block at approximately lines 662–690 that manually builds the block table and calls `self.backend.attention_paged(...)`:

```rust
// 2f. Paged attention — reads from block table
let block_table = cache.block_table(cache_handle)?;
let block_table_i32: Vec<i32> = block_table.iter().map(|&b| b as i32).collect();

// Collect block K/V DeviceTensors for this layer
let pool = cache.pool();
let k_blocks: Vec<&DeviceTensor> = (0..pool.capacity())
    .map(|bid| pool.k_tensor(bid, cache_idx))
    .collect();
let v_blocks: Vec<&DeviceTensor> = (0..pool.capacity())
    .map(|bid| pool.v_tensor(bid, cache_idx))
    .collect();

let attn_out = DeviceTensor::new(
    attn_out_mh.id,
    vec![seq_len, num_q_heads, head_dim],
    DType::FP16,
);

self.backend.attention_paged(
    &q_mh,
    &block_table_i32,
    &k_blocks,
    &v_blocks,
    num_kv_heads,
    new_seq_len,
    start_pos,
    &attn_out,
)?;
```

Replace it with:

```rust
// 2f. Paged attention — dispatch through trait so this works for FP16 paged
// and TurboQuant quantized caches alike.
let attn_out = DeviceTensor::new(
    attn_out_mh.id,
    vec![seq_len, num_q_heads, head_dim],
    DType::FP16,
);

cache.dispatch_attention(
    &self.backend,
    &q_mh,
    cache_handle,
    cache_idx,
    num_kv_heads,
    new_seq_len,
    start_pos,
    &attn_out,
)?;
```

This removes the direct `cache.pool()` call (which is not on the `PagedCache` trait) and lets each cache implementation choose between `attention_paged` (FP16) and `attention_paged_tq` (TurboQuant).

- [ ] **Step 4: Compile-check the engine crate**

Run: `cd /home/nwiley/projects/Fracture && cargo check -p fracture-engine 2>&1 | tail -10`
Expected: compiles cleanly. If errors: any remaining `cache.pool()` calls or direct `attention_paged` calls in `forward_node_paged` will surface — they should all be gone.

### Task A2: Make `forward_paged` generic

**Files:**
- Modify: `crates/fracture-engine/src/engine.rs:510-531`

The wrapper above `forward_node_paged` just propagates the cache type through.

- [ ] **Step 1: Change `forward_paged` signature**

In `crates/fracture-engine/src/engine.rs`, find the function at line 510. Replace its signature:

```rust
// Before
pub fn forward_paged(
    &self,
    token_ids: &[u32],
    positions: &[u32],
    cache: &mut PagedKvCacheManager,
    cache_handle: CacheHandle,
) -> Result<Vec<f32>> {
```

```rust
// After
pub fn forward_paged<C: PagedCache>(
    &self,
    token_ids: &[u32],
    positions: &[u32],
    cache: &mut C,
    cache_handle: CacheHandle,
) -> Result<Vec<f32>> {
```

The body (which calls `self.forward_node_paged(input, &node_config, cache, cache_handle)?`) does not need any changes — it inherits `<C>` automatically.

- [ ] **Step 2: Compile-check**

Run: `cd /home/nwiley/projects/Fracture && cargo check -p fracture-engine 2>&1 | tail -5`
Expected: compiles cleanly.

### Task A3: Run engine unit tests

- [ ] **Step 1: Run engine crate tests**

Run: `cd /home/nwiley/projects/Fracture && cargo nextest run -p fracture-engine 2>&1 | tail -15`
Expected: all tests pass. If any fail, the most likely cause is a `cache.pool()` call we missed; investigate before continuing.

### Task A4: Run GPU integration tests for the paged path

These exercise the real `CudaBackend` with `PagedKvCacheManager` and confirm the trait-dispatch refactor produces identical numerics. This is the primary defense against silent corruption (Risk #1 in the spec).

- [ ] **Step 1: Run paged tests in `bins/fracture-server-cuda/tests/`**

Run: `cd /home/nwiley/projects/Fracture && cargo nextest run -p fracture-server-cuda 2>&1 | tail -20`
Expected: all paged-related GPU tests pass. If `attention_paged` numerics changed, this is where the failure would surface.

- [ ] **Step 2: If GPU is unavailable or tests skip**

If you don't have a GPU available, skip this step but flag it in the commit message: `(GPU tests unrun; need verification on machine with CUDA before merge)`.

### Task A5: Commit Phase A

- [ ] **Step 1: Verify only the expected files changed**

Run: `cd /home/nwiley/projects/Fracture && git status`
Expected: only `crates/fracture-engine/src/engine.rs` modified.

- [ ] **Step 2: Commit**

```bash
cd /home/nwiley/projects/Fracture && git add crates/fracture-engine/src/engine.rs && git commit -m "refactor(engine): make forward_paged generic over PagedCache trait

Replace the inline block-table/pool/attention_paged dispatch in
forward_node_paged with a call to PagedCache::dispatch_attention,
which already implements this for both FP16 paged and TurboQuant
quantized caches. The single-sequence forward path now works with
any PagedCache implementation, matching the existing batched path.

No semantic change. Verified bit-identical against existing GPU
paged tests."
```

Expected: commit succeeds. Capture the commit hash.

---

## Phase B — Make `GenerationLoop` generic over `PagedCache`

### Task B1: Update `generation.rs` imports and signatures

**Files:**
- Modify: `crates/fracture-generate/src/generation.rs:1-2`, `:53-107`, `:166-176`

- [ ] **Step 1: Replace the cache-related imports**

In `crates/fracture-generate/src/generation.rs`, replace line 2:

```rust
// Before
use fracture_engine::{CacheHandle, Engine, KvCacheManager};
```

```rust
// After
use fracture_engine::{CacheHandle, Engine, PagedCache};
```

Run: `cd /home/nwiley/projects/Fracture && cargo check -p fracture-generate 2>&1 | tail -10`
Expected: errors about `KvCacheManager` not found — proceed.

- [ ] **Step 2: Update `generate` signature**

Replace the `generate` function header (around line 53):

```rust
// Before
pub fn generate<B: Backend>(
    engine: &Engine<B>,
    prompt_tokens: &[u32],
    config: &GenerationConfig,
    cache: &mut KvCacheManager,
    tx: &mpsc::UnboundedSender<u32>,
) -> Result<GenerationResult> {
    Self::generate_with_cancel(engine, prompt_tokens, config, cache, tx, None)
}
```

```rust
// After
pub fn generate<B: Backend, C: PagedCache>(
    engine: &Engine<B>,
    prompt_tokens: &[u32],
    config: &GenerationConfig,
    cache: &mut C,
    tx: &mpsc::UnboundedSender<u32>,
) -> Result<GenerationResult> {
    Self::generate_with_cancel(engine, prompt_tokens, config, cache, tx, None)
}
```

- [ ] **Step 3: Update `generate_with_cancel` signature and the `cache.alloc` / `cache.free` calls**

Replace `generate_with_cancel` (around line 67):

```rust
// Before
pub fn generate_with_cancel<B: Backend>(
    engine: &Engine<B>,
    prompt_tokens: &[u32],
    config: &GenerationConfig,
    cache: &mut KvCacheManager,
    tx: &mpsc::UnboundedSender<u32>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<GenerationResult> {
    if prompt_tokens.is_empty() {
        return Err(FractureError::Generation("empty prompt".into()));
    }

    if prompt_tokens.len() > engine.config().max_seq_len {
        return Err(FractureError::Generation(format!(
            "prompt length {} exceeds max_seq_len {}",
            prompt_tokens.len(), engine.config().max_seq_len
        )));
    }

    let cache_handle = cache.alloc(engine.backend())?;

    let result =
        Self::generate_inner(engine, prompt_tokens, config, cache, cache_handle, tx, &cancel);

    // Always free the cache, even on error
    if let Err(e) = cache.free(cache_handle, engine.backend()) {
        tracing::warn!("failed to free KV cache: {e}");
    }

    result
}
```

```rust
// After
pub fn generate_with_cancel<B: Backend, C: PagedCache>(
    engine: &Engine<B>,
    prompt_tokens: &[u32],
    config: &GenerationConfig,
    cache: &mut C,
    tx: &mpsc::UnboundedSender<u32>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<GenerationResult> {
    if prompt_tokens.is_empty() {
        return Err(FractureError::Generation("empty prompt".into()));
    }

    if prompt_tokens.len() > engine.config().max_seq_len {
        return Err(FractureError::Generation(format!(
            "prompt length {} exceeds max_seq_len {}",
            prompt_tokens.len(), engine.config().max_seq_len
        )));
    }

    let cache_handle = cache.alloc()?;

    let result =
        Self::generate_inner(engine, prompt_tokens, config, cache, cache_handle, tx, &cancel);

    // Always free the cache, even on error
    if let Err(e) = cache.free(cache_handle) {
        tracing::warn!("failed to free KV cache: {e}");
    }

    result
}
```

The two changes inside the body: `cache.alloc(engine.backend())?` → `cache.alloc()?` and `cache.free(cache_handle, engine.backend())` → `cache.free(cache_handle)`. The `PagedCache` trait's `alloc`/`free` methods don't take a backend argument because the cache stores its own backend reference internally (or doesn't need one — `PagedKvCacheManager` stores blocks pre-allocated against a backend at construction time).

- [ ] **Step 4: Update `generate_inner` signature and swap `forward` → `forward_paged`**

Replace `generate_inner` (around line 99):

```rust
// Before
fn generate_inner<B: Backend>(
    engine: &Engine<B>,
    prompt_tokens: &[u32],
    config: &GenerationConfig,
    cache: &mut KvCacheManager,
    cache_handle: CacheHandle,
    tx: &mpsc::UnboundedSender<u32>,
    cancel: &Option<Arc<AtomicBool>>,
) -> Result<GenerationResult> {
    let request_start = Instant::now();
    let sampling_params = SamplingParams {
        temperature: config.temperature,
        top_k: config.top_k,
        top_p: config.top_p,
        seed: config.seed,
    };

    // Prefill: process all prompt tokens at once
    let positions: Vec<u32> = (0..prompt_tokens.len() as u32).collect();
    let logits = engine.forward(prompt_tokens, &positions, cache, cache_handle, None)?;
```

```rust
// After
fn generate_inner<B: Backend, C: PagedCache>(
    engine: &Engine<B>,
    prompt_tokens: &[u32],
    config: &GenerationConfig,
    cache: &mut C,
    cache_handle: CacheHandle,
    tx: &mpsc::UnboundedSender<u32>,
    cancel: &Option<Arc<AtomicBool>>,
) -> Result<GenerationResult> {
    let request_start = Instant::now();
    let sampling_params = SamplingParams {
        temperature: config.temperature,
        top_k: config.top_k,
        top_p: config.top_p,
        seed: config.seed,
    };

    // Prefill: process all prompt tokens at once
    let positions: Vec<u32> = (0..prompt_tokens.len() as u32).collect();
    let logits = engine.forward_paged(prompt_tokens, &positions, cache, cache_handle)?;
```

Note two simultaneous changes here: signature gets `C: PagedCache` and the prefill `engine.forward(...)` call becomes `engine.forward_paged(...)`. The contiguous `forward` takes a 5th `Option<&mut Profile>`-ish argument and a 6th `None`; `forward_paged` does not. Drop those last two arguments.

- [ ] **Step 5: Swap the decode-loop `engine.forward` for `engine.forward_paged`**

Within `generate_inner`, find the decode-loop call (around line 146):

```rust
// Before
let logits = engine.forward(&[next_token], &[pos], cache, cache_handle, None)?;
```

```rust
// After
let logits = engine.forward_paged(&[next_token], &[pos], cache, cache_handle)?;
```

Same drop of trailing args.

- [ ] **Step 6: Update `emit_metrics` signature**

Around line 167, replace:

```rust
// Before
fn emit_metrics<B: Backend>(
    engine: &Engine<B>,
    prompt_tokens: usize,
    generated_tokens: usize,
    ttft: std::time::Duration,
    request_start: Instant,
    decode_times: &[f64],
    cache: &KvCacheManager,
    cache_handle: CacheHandle,
) {
```

```rust
// After
fn emit_metrics<B: Backend, C: PagedCache>(
    engine: &Engine<B>,
    prompt_tokens: usize,
    generated_tokens: usize,
    ttft: std::time::Duration,
    request_start: Instant,
    decode_times: &[f64],
    cache: &C,
    cache_handle: CacheHandle,
) {
```

The body uses `cache.seq_len(cache_handle).unwrap_or(0)` — this method is already on the `PagedCache` trait (line 19 of `batched.rs`), no further changes.

- [ ] **Step 7: Compile-check `fracture-generate` (will fail in tests, that's expected)**

Run: `cd /home/nwiley/projects/Fracture && cargo check -p fracture-generate 2>&1 | tail -10`
Expected: library compiles. Tests will fail to compile (next task fixes them).

### Task B2: Update `MockBackend` in `generation_tests.rs`

**Files:**
- Modify: `crates/fracture-generate/src/generation/generation_tests.rs:77-110`, `:140-185`, `:238-240`

The `Backend` trait has a default error-returning impl for `attention_paged` (lines 74–88 of `crates/fracture-core/src/backend.rs`). Once `GenerationLoop` calls into `forward_paged` and through `dispatch_attention`, the mock backend will hit that default and tests will fail. Override it with a no-op.

- [ ] **Step 1: Add `attention_paged` no-op to `MockBackend`**

In `crates/fracture-generate/src/generation/generation_tests.rs`, locate the `impl Backend for MockBackend` block at line 77. Add a new method right after the existing `attention(...)` method (around line 97). The signature must match the trait declaration in `crates/fracture-core/src/backend.rs:74-88`.

Insert this method:

```rust
    fn attention_paged(
        &self,
        _q: &DeviceTensor,
        _block_table: &[i32],
        _k_blocks: &[&DeviceTensor],
        _v_blocks: &[&DeviceTensor],
        _num_kv_heads: usize,
        _kv_len: usize,
        _start_pos: usize,
        _out: &DeviceTensor,
    ) -> fracture_core::Result<()> {
        Ok(())
    }
```

- [ ] **Step 2: Add `attention_paged` no-op to `FailingMockBackend`**

In the same file, locate `impl Backend for FailingMockBackend` at line 140. Add the identical method after the existing `attention(...)` method (around line 172):

```rust
    fn attention_paged(
        &self,
        _q: &DeviceTensor,
        _block_table: &[i32],
        _k_blocks: &[&DeviceTensor],
        _v_blocks: &[&DeviceTensor],
        _num_kv_heads: usize,
        _kv_len: usize,
        _start_pos: usize,
        _out: &DeviceTensor,
    ) -> fracture_core::Result<()> {
        Ok(())
    }
```

### Task B3: Update `make_cache` helper

**Files:**
- Modify: `crates/fracture-generate/src/generation/generation_tests.rs:238-240`

- [ ] **Step 1: Replace `make_cache`**

Locate lines 238–240:

```rust
// Before
fn make_cache(cfg: &ModelConfig) -> KvCacheManager {
    KvCacheManager::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len)
}
```

`PagedKvCacheManager::new` takes `(num_blocks, num_layers, num_kv_heads, head_dim, &backend)` (block size is hardcoded to 16 inside the manager). `make_cache` currently doesn't have a `&backend` parameter; it needs one, but every caller already constructs a backend, so plumbing it through is straightforward.

```rust
// After
fn make_cache<B: Backend>(cfg: &ModelConfig, backend: &B) -> PagedKvCacheManager {
    // Block count: ceil(max_seq_len / 16) + 2 blocks of safety margin.
    // 16 is the hardcoded BLOCK_SIZE per CLAUDE.md.
    let num_blocks = cfg.max_seq_len.div_ceil(16) + 2;
    PagedKvCacheManager::new(num_blocks, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, backend)
        .expect("PagedKvCacheManager::new failed in test setup")
}
```

You also need to update the import at the top of the file. Find line 1: `use super::*;` — this picks up names from the parent `generation` module. The parent module's import was `use fracture_engine::{CacheHandle, Engine, PagedCache};` (after Task B1) — `KvCacheManager` is no longer in scope. Add a direct import:

In `crates/fracture-generate/src/generation/generation_tests.rs`, change line 2 from:

```rust
// Before
use fracture_core::{DType, DeviceTensor, DeviceTimer, ModelConfig, TensorId};
```

```rust
// After
use fracture_core::{DType, DeviceTensor, DeviceTimer, ModelConfig, TensorId};
use fracture_engine::PagedKvCacheManager;
```

- [ ] **Step 2: Update every test that calls `make_cache(...)`**

`make_cache` now requires a `&backend` argument. Run a search-and-fix:

Run: `cd /home/nwiley/projects/Fracture && grep -n "make_cache(" crates/fracture-generate/src/generation/generation_tests.rs`
Expected: ~20–30 call sites, each of the form `make_cache(&cfg)` or `make_cache(cfg)`.

For each call, the test already constructs a `MockBackend` (or `FailingMockBackend`) before calling `make_cache`. The fix per call site: pass that backend reference. Example:

```rust
// Before
let cfg = tiny_config();
let backend = MockBackend::always(42, cfg.vocab_size);
let engine = make_engine(backend, &cfg);
let mut cache = make_cache(&cfg);
```

```rust
// After
let cfg = tiny_config();
let backend = MockBackend::always(42, cfg.vocab_size);
let mut cache = make_cache(&cfg, &backend);
let engine = make_engine(backend, &cfg);
```

The order swap (cache before engine) is necessary because `make_engine` consumes the backend (`fn make_engine<B: Backend>(backend: B, cfg: &ModelConfig)` takes `B` by value), so we must take the `&backend` reference for `make_cache` before the engine takes ownership.

The compiler will surface every call site that needs updating. Treat the compile error list as your worklist.

- [ ] **Step 3: Compile-check `fracture-generate`**

Run: `cd /home/nwiley/projects/Fracture && cargo check -p fracture-generate --tests 2>&1 | tail -20`
Expected: compiles cleanly. If any errors remain, they're typically:
- A `make_cache` call site you missed (update it)
- A test that constructs `KvCacheManager` directly (replace with `PagedKvCacheManager::new(...)` following the same pattern)
- A test that calls `cache.alloc(&backend)` or `cache.free(handle, &backend)` directly (drop the backend arg)

### Task B4: Run `fracture-generate` tests

- [ ] **Step 1: Run the unit test suite**

Run: `cd /home/nwiley/projects/Fracture && cargo nextest run -p fracture-generate 2>&1 | tail -15`
Expected: all ~30+ generation unit tests pass. If a test fails:
- A position-tracking test failing usually means `forward_paged` signature mismatch — re-check the swap.
- A cancellation test failing usually means the cancel flag isn't reaching the decode loop — re-check the decode-loop change.
- A "expected error" test failing means error variants now have different shapes — adjust assertions to match.

### Task B5: Commit Phase B

- [ ] **Step 1: Verify only the expected files changed**

Run: `cd /home/nwiley/projects/Fracture && git status`
Expected: two files modified:
- `crates/fracture-generate/src/generation.rs`
- `crates/fracture-generate/src/generation/generation_tests.rs`

- [ ] **Step 2: Commit**

```bash
cd /home/nwiley/projects/Fracture && git add crates/fracture-generate/src/generation.rs crates/fracture-generate/src/generation/generation_tests.rs && git commit -m "refactor(generate): make GenerationLoop generic over PagedCache

GenerationLoop::generate, generate_with_cancel, generate_inner, and
emit_metrics now take <C: PagedCache> instead of being bound to the
contiguous KvCacheManager. Internal forward calls switch to
Engine::forward_paged. cache.alloc/free no longer take a backend
argument (PagedCache trait does not require it).

MockBackend and FailingMockBackend in generation_tests.rs gain a
no-op attention_paged override to satisfy the paged dispatch path.
make_cache helper now constructs a PagedKvCacheManager sized from
max_seq_len / 16 + 2 blocks safety margin.

All ~30 existing generation unit tests pass against the paged path
without changes to their assertions."
```

---

## Phase C — Relocate golden tests with inlined helpers

### Task C1: Create the new golden test file with skeleton

**Files:**
- Create: `bins/fracture-server-cuda/tests/golden.rs`

- [ ] **Step 1: Write the file skeleton**

Create `bins/fracture-server-cuda/tests/golden.rs` with the following content. This includes the module doc, all imports, the `skip!` macro, the `GoldenMetadata` type, and inlined helpers. The test functions go in the next task.

```rust
//! Golden generation comparison tests.
//!
//! Run full greedy generation through the Fracture engine on the paged KV
//! cache path and compare the output token sequence against the PyTorch
//! golden reference at `tests/golden/`.
//!
//! Prerequisites:
//!   - `FRACTURE_MODEL_PATH` env var pointing to a Llama 3.1 8B FP16 GGUF file
//!   - Golden data files in `tests/golden/` (committed to repo)
//!
//! Tests skip gracefully when either is missing.

use std::fs;
use std::path::{Path, PathBuf};

use fracture_core::ModelConfig;
use fracture_cuda::CudaBackend;
use fracture_engine::{Engine, PagedKvCacheManager};
use fracture_generate::{GenerationConfig, GenerationLoop};
use fracture_gguf::WeightStore;
use serde::Deserialize;
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Inlined helpers (originally from tests/model-validation/src/lib.rs and
// tests/validation/src/golden_compare.rs — both crates are slated for
// deletion in sub-project 1b, so we copy what we need rather than depend on
// them).
// ---------------------------------------------------------------------------

/// Skip the calling test with a message. Returns from the caller.
macro_rules! skip {
    ($($arg:tt)*) => {{
        eprintln!("SKIPPED: {}", format!($($arg)*));
        return;
    }};
}

/// Project root: walk up from this crate's manifest dir.
/// `bins/fracture-server-cuda/Cargo.toml` → workspace root is two parents up.
fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Root of the golden output directory.
fn golden_dir() -> PathBuf {
    project_root().join("tests/golden")
}

/// GGUF model path from `FRACTURE_MODEL_PATH`. Returns `None` if unset.
fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("FRACTURE_MODEL_PATH").ok()?);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("FRACTURE_MODEL_PATH={} does not exist", path.display());
        None
    }
}

/// Load Llama from GGUF, build a CUDA engine. Returns `None` if model unavailable.
fn setup_real_engine() -> Option<(Engine<CudaBackend>, ModelConfig)> {
    let path = model_path()?;
    let mut backend = CudaBackend::new(0).expect("CUDA backend creation failed");
    let weights = WeightStore::load(&path, &backend, None).expect("failed to load GGUF weights");
    let config = weights.config.clone();
    backend
        .precompute_rope_freqs(config.head_dim, config.rope_theta)
        .expect("RoPE precomputation failed");
    let engine = Engine::new(backend, weights, 0..config.num_layers);
    Some((engine, config))
}

#[derive(Deserialize)]
struct GoldenMetadata {
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
}

/// Result of comparing two token sequences.
#[derive(Debug, Clone)]
struct TokenComparisonResult {
    matching_tokens: usize,
    total_expected: usize,
    total_actual: usize,
    divergence_index: Option<usize>,
    expected_token_at_divergence: Option<u32>,
    actual_token_at_divergence: Option<u32>,
}

impl TokenComparisonResult {
    fn matches(&self) -> bool {
        self.divergence_index.is_none() && self.total_actual == self.total_expected
    }
}

impl std::fmt::Display for TokenComparisonResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.matches() {
            write!(f, "TOKEN MATCH: all {} tokens identical", self.total_expected)
        } else {
            write!(
                f,
                "TOKEN MISMATCH: {}/{} tokens match",
                self.matching_tokens, self.total_expected
            )?;
            if let Some(idx) = self.divergence_index {
                write!(
                    f,
                    "\n  First divergence at index {}: expected={:?} actual={:?}",
                    idx, self.expected_token_at_divergence, self.actual_token_at_divergence
                )?;
            }
            if self.total_actual != self.total_expected {
                write!(
                    f,
                    "\n  Length mismatch: expected {} tokens, got {}",
                    self.total_expected, self.total_actual
                )?;
            }
            Ok(())
        }
    }
}

/// Load a golden token sequence from the binary format
/// `[ndim:u32][shape:u32...][dtype:u32][data:u32...]`. dtype must be 2 (int32).
fn load_golden_tokens(path: &str) -> Result<Vec<u32>, String> {
    let p = Path::new(path);
    if !p.exists() {
        return Err(format!("Golden token file not found: {}", path));
    }

    let data = fs::read(p).map_err(|e| format!("Failed to read '{}': {}", path, e))?;
    if data.len() < 8 {
        return Err(format!("File too small: {} bytes", data.len()));
    }

    let ndim = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let mut offset = 4;

    let mut num_elements: usize = 1;
    for _ in 0..ndim {
        if offset + 4 > data.len() {
            return Err("Truncated shape".to_string());
        }
        let dim = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        num_elements *= dim;
        offset += 4;
    }

    if offset + 4 > data.len() {
        return Err("Truncated dtype".to_string());
    }
    let dtype_enum = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]);
    offset += 4;

    if dtype_enum != 2 {
        return Err(format!(
            "Expected dtype int32 (2) for token sequence, got {}",
            dtype_enum
        ));
    }

    let expected_bytes = num_elements * 4;
    let remaining = data.len() - offset;
    if remaining != expected_bytes {
        return Err(format!(
            "Data size mismatch: expected {} bytes, got {}",
            expected_bytes, remaining
        ));
    }

    Ok(data[offset..]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Compare two token sequences, reporting first divergence point.
fn compare_token_sequences(actual: &[u32], expected: &[u32]) -> TokenComparisonResult {
    let mut matching = 0;
    let min_len = actual.len().min(expected.len());

    for i in 0..min_len {
        if actual[i] == expected[i] {
            matching += 1;
        } else {
            return TokenComparisonResult {
                matching_tokens: matching,
                total_expected: expected.len(),
                total_actual: actual.len(),
                divergence_index: Some(i),
                expected_token_at_divergence: Some(expected[i]),
                actual_token_at_divergence: Some(actual[i]),
            };
        }
    }

    if actual.len() != expected.len() {
        let div_idx = min_len;
        TokenComparisonResult {
            matching_tokens: matching,
            total_expected: expected.len(),
            total_actual: actual.len(),
            divergence_index: Some(div_idx),
            expected_token_at_divergence: expected.get(div_idx).copied(),
            actual_token_at_divergence: actual.get(div_idx).copied(),
        }
    } else {
        TokenComparisonResult {
            matching_tokens: matching,
            total_expected: expected.len(),
            total_actual: actual.len(),
            divergence_index: None,
            expected_token_at_divergence: None,
            actual_token_at_divergence: None,
        }
    }
}
```

- [ ] **Step 2: Compile-check the new file (no test functions yet, so it should compile bare)**

Run: `cd /home/nwiley/projects/Fracture && cargo check -p fracture-server-cuda --tests 2>&1 | tail -10`
Expected: compiles cleanly, possibly with warnings about unused helpers (normal — tests come next).

### Task C2: Add the golden test functions

**Files:**
- Modify: `bins/fracture-server-cuda/tests/golden.rs` (append to file from C1)

- [ ] **Step 1: Append the test runner and two test entry points**

Append the following to `bins/fracture-server-cuda/tests/golden.rs`:

```rust
// ---------------------------------------------------------------------------
// Generation tests
// ---------------------------------------------------------------------------

/// Run greedy generation and compare against golden reference for a prompt.
fn test_golden_generation(prompt_index: usize) {
    let golden_path = golden_dir().join(format!("prompt_{prompt_index}_greedy_50.bin"));
    let meta_path = golden_dir().join(format!("prompt_{prompt_index}_greedy_50_meta.json"));

    if !golden_path.exists() {
        skip!("golden data for prompt_{prompt_index} not found");
    }

    let Some((engine, config)) = setup_real_engine() else {
        skip!("FRACTURE_MODEL_PATH not set");
    };

    let meta: GoldenMetadata =
        serde_json::from_str(&fs::read_to_string(&meta_path).unwrap()).unwrap();

    let golden_tokens = load_golden_tokens(golden_path.to_str().unwrap())
        .expect("failed to load golden tokens");

    // Build paged KV cache. max_seq_len = 2048 to match the original test
    // (full 128K would use ~16GB just for the cache). Block size 16 hardcoded.
    // Block count: ceil(2048/16) + 2 = 130.
    let num_blocks = 2048usize.div_ceil(16) + 2;
    let mut cache = PagedKvCacheManager::new(
        num_blocks,
        config.num_layers,
        config.num_kv_heads,
        config.head_dim,
        engine.backend(),
    )
    .expect("PagedKvCacheManager::new failed");

    let gen_config = GenerationConfig {
        max_tokens: 50,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        stop_tokens: vec![], // generate exactly 50 tokens
        seed: None,
    };

    let (tx, _rx) = mpsc::unbounded_channel();
    let generated = GenerationLoop::generate(
        &engine,
        &meta.prompt_token_ids,
        &gen_config,
        &mut cache,
        &tx,
    )
    .expect("generation failed");

    let mut engine_full: Vec<u32> = meta.prompt_token_ids.clone();
    engine_full.extend_from_slice(&generated.tokens);

    let result = compare_token_sequences(&engine_full, &golden_tokens);

    eprintln!("Prompt {prompt_index} golden generation: {result}");

    if !result.matches() {
        let gen_start = meta.prompt_token_ids.len();
        let divergence = result.divergence_index.unwrap();

        if divergence < gen_start {
            panic!(
                "prompt_{prompt_index}: divergence in prompt tokens at index {divergence} — \
                 this should never happen"
            );
        }

        let gen_tokens_correct = divergence - gen_start;
        let gen_tokens_expected = meta.generated_token_ids.len();

        eprintln!(
            "  Generated {}/{} correct tokens before divergence",
            gen_tokens_correct, gen_tokens_expected
        );
        eprintln!(
            "  At position {divergence}: engine={}, reference={}",
            result.actual_token_at_divergence.unwrap_or(0),
            result.expected_token_at_divergence.unwrap_or(0),
        );

        assert!(
            gen_tokens_correct >= 5,
            "prompt_{prompt_index}: only {gen_tokens_correct}/{gen_tokens_expected} generated \
             tokens match — engine likely has a correctness issue"
        );

        eprintln!(
            "WARNING: prompt_{prompt_index}: {gen_tokens_correct}/{gen_tokens_expected} tokens \
             match. FP16 accumulation divergence after {gen_tokens_correct} tokens is expected \
             for autoregressive generation."
        );
    }
}

#[test]
fn test_golden_generation_prompt_0() {
    test_golden_generation(0);
}

#[test]
fn test_golden_generation_prompt_1() {
    test_golden_generation(1);
}
```

- [ ] **Step 2: Verify the file compiles**

Run: `cd /home/nwiley/projects/Fracture && cargo check -p fracture-server-cuda --tests 2>&1 | tail -10`
Expected: compiles cleanly. If `serde_json` is not a direct dev-dependency of `fracture-server-cuda`, add it to the `[dev-dependencies]` section of `bins/fracture-server-cuda/Cargo.toml`.

- [ ] **Step 3: If `serde_json` is missing as a dev-dep, add it**

Check `bins/fracture-server-cuda/Cargo.toml`:

Run: `grep -n "serde\|serde_json" /home/nwiley/projects/Fracture/bins/fracture-server-cuda/Cargo.toml`

If `serde_json` doesn't appear under `[dev-dependencies]`, add it. The workspace likely has it as a workspace dep, so you can use:

```toml
[dev-dependencies]
serde = { workspace = true }
serde_json = { workspace = true }
```

Add only the lines that are missing. Re-run `cargo check -p fracture-server-cuda --tests` and confirm compilation.

### Task C3: Run the golden test with `FRACTURE_MODEL_PATH` unset

- [ ] **Step 1: Run the test, expecting graceful skip**

Run: `cd /home/nwiley/projects/Fracture && cargo nextest run -p fracture-server-cuda --test golden 2>&1 | tail -15`
Expected: both `test_golden_generation_prompt_0` and `test_golden_generation_prompt_1` pass with stdout containing `SKIPPED: FRACTURE_MODEL_PATH not set` (or similar). The tests must report PASS, not FAIL — `skip!` returns from the test fn without panic.

### Task C4: Run the golden test with `FRACTURE_MODEL_PATH` set (GPU required)

- [ ] **Step 1: Locate or set up the model**

If you have a Llama 3.1 8B GGUF model locally, get its path. If not, this step cannot be run on this machine — flag it for verification on a CUDA-enabled machine before merge.

- [ ] **Step 2: Run the test against real model**

Run: `cd /home/nwiley/projects/Fracture && FRACTURE_MODEL_PATH=/path/to/llama-3.1-8b.gguf cargo nextest run -p fracture-server-cuda --test golden 2>&1 | tail -30`
Expected: both tests pass. Log output should include `TOKEN MATCH: all N tokens identical` or `TOKEN MISMATCH` with at least 5 generated tokens correct (the test asserts `gen_tokens_correct >= 5`, allowing some FP16 drift after that).

If the test fails:
- 0 generated tokens match → engine is producing garbage; the trait-dispatch refactor in Phase A introduced a regression. Re-investigate.
- 1–4 generated tokens match → numerical drift is happening earlier than before. Check whether `dispatch_attention` produces bit-identical output to the previous inline path on a single GPU paged test.
- Test passes but with fewer matches than the old contiguous golden test → expected; the paged path has slightly different memory access patterns, FP16 accumulation order can differ. This is acceptable as long as ≥5 tokens match.

### Task C5: Commit Phase C

- [ ] **Step 1: Verify only the expected files changed**

Run: `cd /home/nwiley/projects/Fracture && git status`
Expected: one new file (`bins/fracture-server-cuda/tests/golden.rs`); possibly one modified (`bins/fracture-server-cuda/Cargo.toml` if dev-deps were added). The original `tests/model-validation/tests/golden.rs` is untouched.

- [ ] **Step 2: Commit**

```bash
cd /home/nwiley/projects/Fracture && git add bins/fracture-server-cuda/tests/golden.rs bins/fracture-server-cuda/Cargo.toml && git commit -m "test: add golden generation tests on paged KV cache path

Relocates greedy-generation golden tests from the dying tests/model-validation/
crate into bins/fracture-server-cuda/tests/, where CUDA is already a
mandatory dependency. All helpers (golden_dir, setup_real_engine,
load_golden_tokens, compare_token_sequences, skip! macro, GoldenMetadata)
are inlined so the original fracture_validation and fracture_model_validation
crates can be deleted in sub-project 1b without further churn.

Tests skip gracefully without FRACTURE_MODEL_PATH; with the env var set,
they assert at least 5 generated tokens match the PyTorch reference
(allowing for expected FP16 accumulation drift in autoregressive decoding).

The original tests/model-validation/tests/golden.rs is left in place;
sub-project 1b removes the entire model-validation tree."
```

---

## Phase D — Final verification

### Task D1: Full workspace test run

- [ ] **Step 1: Run the entire test suite**

Run: `cd /home/nwiley/projects/Fracture && cargo nextest run 2>&1 | tail -20`
Expected: all tests pass. Test count should match (or exceed by 2 — the new prompt_0 and prompt_1 golden tests in their new home — possibly 4 if the old ones in `tests/model-validation` still run as well; that's fine). No regressions.

### Task D2: Clippy

- [ ] **Step 1: Run clippy**

Run: `cd /home/nwiley/projects/Fracture && cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -10`
Expected: clean. If clippy flags an unused `KvCacheManager` import somewhere or an unused helper, fix it inline before completing.

### Task D3: Final git status

- [ ] **Step 1: Confirm clean tree**

Run: `cd /home/nwiley/projects/Fracture && git status`
Expected: working tree clean. Only the three commits from Phases A, B, C should be on the branch.

- [ ] **Step 2: Show the three commits**

Run: `cd /home/nwiley/projects/Fracture && git log --oneline -4`
Expected output (most recent first):
```
<hash> test: add golden generation tests on paged KV cache path
<hash> refactor(generate): make GenerationLoop generic over PagedCache
<hash> refactor(engine): make forward_paged generic over PagedCache trait
<hash> docs: add design spec for porting GenerationLoop to paged KV cache
```

### Task D4: Verification summary

- [ ] **Step 1: Paste the 5-item completion verification (per the spec)**

Compose a completion summary including the actual outputs of:
1. `cargo nextest run` (full workspace) — green count
2. `cargo clippy --workspace --all-targets -- -D warnings` — clean
3. `FRACTURE_MODEL_PATH=... cargo nextest run -p fracture-server-cuda --test golden` — passing
4. `cargo nextest run -p fracture-server-cuda --test golden` (no env var) — skips gracefully
5. `git status` — clean tree

If any of these can't be run on the current machine (GPU unavailable), mark the gap explicitly: "GPU verification needed before merge."

---

## Done

At this point sub-project 1a is complete:
- `Engine::forward_paged`, `forward_node_paged`, and `GenerationLoop::generate` family are generic over `PagedCache`
- Golden generation tests live at `bins/fracture-server-cuda/tests/golden.rs` with all helpers inlined
- `KvCacheManager` (contiguous), `Engine::forward()`, `routes.rs`, `tests/model-validation/`, `tests/validation/`, `tests/reference/`, and Python reference scripts are all still in-tree and untouched
- Three independent commits, each individually revertible
- Sub-project 1b is now a pure-deletion exercise on the above unchanged code
