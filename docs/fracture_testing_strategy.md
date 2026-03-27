# Fracture: Testing Strategy

## Testing Tiers

Fracture has three testing tiers with different requirements, speeds, and purposes.
Every behavior in the vexspec maps to one or more of these tiers.

### Tier 1: Unit Tests (No GPU)

**Runs on:** Any machine, CI without GPU
**Speed:** Milliseconds per test
**Scope:** Pure logic, parsing, data structures, sampling math

| Component | What's tested | Example |
|---|---|---|
| GGUF parser | Header parsing, metadata extraction, tensor info table | Feed hand-crafted binary buffers, assert parsed values |
| Sampling | Temperature scaling, top-K, top-P, greedy determinism | Known logit vectors → expected token selections |
| HTTP server | Request parsing, validation, response formatting, SSE framing | Construct requests → assert responses without running inference |
| Config | ModelConfig validation, GGUF name mapping | Invalid configs → expected errors |
| KV Cache (logic) | Handle lifecycle, bounds checking, sequence tracking | Mock backend memory, test allocation/free state machine |
| Chat template | Message formatting, special token insertion | Messages → expected prompt string |
| Tokenizer | Encode/decode round-trips, special tokens | Known text → expected token IDs |

**How to write these:**
Standard Rust `#[test]` functions in each crate's `tests/` or inline `#[cfg(test)]` modules.
No special infrastructure needed.

**Vex coverage:**
These cover: GGUF Parser (all behaviors), Sampling (all behaviors), HTTP Server (all behaviors),
KV Cache Manager (cache-lifecycle, cache-bounds), Generation Loop (tokenization, stop-conditions).

---

### Tier 2: Numerical Validation Tests (GPU Required)

**Runs on:** Machine with NVIDIA GPU + CUDA toolkit
**Speed:** Seconds per test (kernel launch + comparison)
**Scope:** Every Backend trait method implementation, individually

This is the most critical testing tier. A kernel that's "close but not right" produces
text that is subtly wrong in ways that are impossible to debug at higher levels.

#### The Reference Harness

`scripts/dump_reference.py` is the foundation of all Tier 2 tests.

**What it does:**
1. Loads Llama 3 8B in PyTorch (HuggingFace transformers, FP16)
2. Hooks into the model to capture intermediate tensors at every stage
3. Runs a forward pass on a set of test prompts
4. Dumps input/output tensor pairs for every operation to `tests/reference/`

**What it dumps (per layer, per test prompt):**

```
tests/reference/
├── prompt_0/                          # "The capital of France is"
│   ├── token_ids.bin                  # Input token IDs
│   ├── embeddings.bin                 # After embedding lookup
│   ├── layer_00/
│   │   ├── input_hidden.bin           # Input to layer 0
│   │   ├── post_attn_norm.bin         # After first RMSNorm
│   │   ├── q.bin                      # Q projection output
│   │   ├── k.bin                      # K projection output
│   │   ├── v.bin                      # V projection output
│   │   ├── q_rope.bin                 # Q after RoPE
│   │   ├── k_rope.bin                 # K after RoPE
│   │   ├── attn_scores.bin            # Raw attention scores (pre-softmax)
│   │   ├── attn_probs.bin             # Attention probabilities (post-softmax)
│   │   ├── attn_output.bin            # Attention output (post concat + Wo)
│   │   ├── post_attn_residual.bin     # After attention residual add
│   │   ├── post_ffn_norm.bin          # After second RMSNorm
│   │   ├── gate.bin                   # Gate projection output
│   │   ├── up.bin                     # Up projection output
│   │   ├── silu_mul.bin               # silu(gate) * up
│   │   ├── ffn_output.bin             # After down projection
│   │   └── output_hidden.bin          # After FFN residual add
│   ├── layer_01/
│   │   └── ...
│   ├── ...
│   ├── layer_31/
│   │   └── ...
│   ├── final_norm.bin                 # After output RMSNorm
│   ├── logits.bin                     # Final logits [seq_len, 128256]
│   └── sampled_token.bin              # Greedy-sampled token ID
├── prompt_1/                          # Second test prompt
│   └── ...
└── decode_step/                       # Single decode step after prompt_0
    ├── input_token.bin
    ├── position.bin
    ├── layer_00/
    │   ├── k_cache_full.bin           # Full K cache after append
    │   ├── v_cache_full.bin           # Full V cache after append
    │   └── ...                        # Same per-layer tensors as prefill
    ├── ...
    └── logits.bin
```

**Test prompts (chosen deliberately):**
- `"The capital of France is"` — short, common tokens, exercises basic path
- A 512-token prompt — exercises prefill at realistic length, stress-tests KV cache
- A prompt with rare/special tokens — exercises edge cases in tokenizer and embedding

**Important:** The reference harness dumps at FULL precision for intermediate values
(FP32 where PyTorch uses FP32 internally), then the test comparison uses the appropriate
tolerance based on what precision Fracture uses at that stage.

#### Per-Kernel Test Pattern

Every kernel test follows the same structure:

```rust
#[test]
fn test_rmsnorm_prefill() {
    // 1. Load reference input and expected output
    let input = load_reference_tensor("prompt_0/layer_00/input_hidden.bin");
    let weight = load_reference_tensor("prompt_0/layer_00/attn_norm_weight.bin");
    let expected = load_reference_tensor("prompt_0/layer_00/post_attn_norm.bin");

    // 2. Upload to device via Backend trait
    let backend = create_test_backend();  // CudaBackend or MetalBackend
    let dev_input = backend.alloc(&input.shape, input.dtype).unwrap();
    backend.copy_to_device(&dev_input, &input.data).unwrap();
    let dev_weight = backend.alloc(&weight.shape, weight.dtype).unwrap();
    backend.copy_to_device(&dev_weight, &weight.data).unwrap();
    let dev_output = backend.alloc(&input.shape, input.dtype).unwrap();

    // 3. Run through Backend trait — not CUDA directly
    backend.rmsnorm(&dev_input, &dev_weight, 1e-5, &dev_output).unwrap();

    // 4. Copy output back to host
    let mut output_data = vec![0u8; expected.data.len()];
    backend.copy_to_host(&dev_output, &mut output_data).unwrap();

    // 5. Assert closeness
    assert_tensors_close(&output_data, &expected.data, rtol=1e-3, atol=1e-3);
}
```

**Kernel tests to write (in implementation order):**

| Kernel | Test cases | Key edge cases |
|---|---|---|
| RMSNorm | Prefill (N=5), decode (N=1), layer 0, layer 31 | Zero input, large values |
| RoPE | Prefill positions [0..5], decode position [47] | Position 0, large positions |
| cuBLAS GEMM | All 4 shape families used in Llama 3 | M=1 (decode), large M (prefill) |
| SiLU × Mul | Prefill (N=5), decode (N=1) | Negative gate values, zeros |
| Attention (prefill) | Short prompt, 512-token prompt | Single token prompt |
| Attention (decode) | Various cache lengths: 1, 10, 512, 2048 | First decode step (cache len=prompt_len) |
| Embedding | Valid IDs, boundary IDs (0, 128255) | Out-of-range ID |

**Vex coverage:**
These cover: RMSNorm Kernel (all), RoPE Kernel (all), SiLU Multiply Kernel (all),
Attention Kernel (all), Embedding Lookup (all), Matrix Multiplication (all).

---

### Tier 3: Integration and End-to-End Tests (GPU + Model)

**Runs on:** Machine with GPU and Llama 3 8B GGUF weights on disk
**Speed:** Seconds to minutes
**Scope:** Multi-component integration, full generation pipeline

#### Integration Tests

These test component boundaries without running the full pipeline:

| Test | What it validates |
|---|---|
| GGUF load → single layer forward | Weight loading produces correct layer output |
| Full 32-layer forward (prefill) | Final logits match PyTorch reference |
| Full 32-layer forward (decode) | Logits match after KV cache is populated |
| Prefill/decode consistency | Decode step N+1 logits == prefill on full sequence |
| Layer-range forward | Layers [0, 16) produce correct intermediate activation |

#### End-to-End Tests

These test the complete pipeline:

| Test | What it validates |
|---|---|
| Greedy generation | Temperature=0 on known prompt produces expected text |
| Token streaming | Tokens arrive through channel in correct order |
| HTTP completions | POST request → correct JSON response |
| HTTP chat | Chat messages → correctly formatted response |
| HTTP SSE | Stream=true → correct SSE event sequence |
| Stop conditions | Generation stops at EOS, max_tokens, and stop strings |

**Greedy generation test is the most important.** If you run the same prompt with
temperature=0 in both PyTorch and Fracture and get different text, something is wrong.
This is the ultimate correctness gate before any release.

**Vex coverage:**
These cover: Compute Engine (all behaviors), Generation Loop (all behaviors),
KV Cache Manager (cache-append-prefill, cache-append-decode, cache-retrieval).

---

## Test Infrastructure

### Reference Tensor Format

Binary files with a simple header:

```
[4 bytes: ndim (u32)]
[4 bytes × ndim: shape dimensions (u32 each)]
[4 bytes: dtype enum (0=f16, 1=f32, 2=i32)]
[remaining: raw tensor data]
```

Rust helper to load:
```rust
struct ReferenceTensor {
    shape: Vec<usize>,
    dtype: DType,
    data: Vec<u8>,
}

fn load_reference_tensor(path: &str) -> ReferenceTensor { ... }
fn assert_tensors_close(actual: &[f16], expected: &[f16], rtol: f32, atol: f32) { ... }
```

The `assert_tensors_close` function should report:
- Max absolute error and where it occurs (index)
- Mean absolute error
- Percentage of elements exceeding tolerance
- First 5 mismatched values with their indices

This diagnostic output is critical for debugging kernel issues. A bare "assertion failed"
tells you nothing; knowing that "element [2, 17, 45] expected 0.0312 got 0.0298" points
you directly at the problem.

### Test Categories via Cargo Features

```toml
# Cargo.toml (workspace)
[features]
default = []
cuda-tests = []       # Tier 2: requires NVIDIA GPU with CUDA
metal-tests = []      # Tier 2: requires Apple Silicon with Metal
model-tests = []      # Tier 3: requires GPU + model weights on disk
```

```rust
#[test]
fn test_sampling_greedy() {
    // Always runs — no GPU needed
}

#[cfg(feature = "cuda-tests")]
#[test]
fn test_rmsnorm_matches_reference_cuda() {
    // Only runs with: cargo test --features cuda-tests
}

#[cfg(feature = "metal-tests")]
#[test]
fn test_rmsnorm_matches_reference_metal() {
    // Only runs with: cargo test --features metal-tests
}

#[cfg(feature = "model-tests")]
#[test]
fn test_e2e_greedy_generation() {
    // Only runs with: cargo test --features model-tests
}
```

This means `cargo test` runs fast Tier 1 tests everywhere (including CI without GPU),
while Tier 2/3 require explicit opt-in on capable hardware. CUDA and Metal tests
are separate features — a Linux machine runs `cuda-tests`, a Mac runs `metal-tests`,
and both validate against the same PyTorch reference tensors.

### Environment Variables for Test Config

```
FRACTURE_MODEL_PATH=/path/to/llama-3-8b.gguf    # For Tier 3 tests
FRACTURE_REFERENCE_DIR=tests/reference/           # For Tier 2 tests
FRACTURE_TEST_DEVICE=cuda:0                       # GPU device for Tier 2/3
```

---

## Test Implementation Order

Matches the implementation order from the architecture doc:

| Step | What to implement | What to test | Tier |
|---|---|---|---|
| 1 | fracture-core types | ModelConfig validation | 1 |
| 2 | fracture-gguf parser | Header, metadata, tensor info, weight names | 1 |
| 3 | dump_reference.py | Generate reference tensors (not a test itself) | — |
| 4 | Test infrastructure | load_reference_tensor, assert_tensors_close | 2 |
| 5 | RMSNorm kernel | rmsnorm-forward, rmsnorm-numerical-stability | 2 |
| 6 | RoPE kernel | rope-precomputation, rope-apply | 2 |
| 7 | cuBLAS GEMM wrapper | gemm-correctness, gemm-edge-cases | 2 |
| 8 | SiLU multiply kernel | silu-mul-forward | 2 |
| 9 | Attention kernel | attention-prefill, attention-decode, attention-gqa | 2 |
| 10 | Embedding lookup | embedding-gather | 2 |
| 11 | KV Cache manager | All cache behaviors | 1 + 2 |
| 12 | Compute engine (1 layer) | single-layer-forward | 2 |
| 13 | Compute engine (full) | full-forward-prefill, full-forward-decode | 2 |
| 14 | Compute engine (consistency) | prefill-decode-consistency | 2 |
| 15 | Compute engine (layer range) | layer-range-execution | 2 |
| 16 | Sampling | All sampling behaviors | 1 |
| 17 | Generation loop | tokenization, stop-conditions, streaming | 1 + 3 |
| 18 | HTTP server | All server behaviors | 1 + 3 |
| 19 | E2E greedy generation | greedy-generation | 3 |

---

## Running Vex

After implementing tests for a component:

```bash
# Check everything
vex check

# Check a single section after implementing its tests
vex check --section "RMSNorm Kernel"

# After stabilization, use drift for incremental checks
vex check --drift
```

Vex will audit whether the test files in each crate actually exercise the behaviors
described in the spec. Since Fracture's tests are split across tiers, Vex needs to
see tests from all tiers to confirm full coverage — run it on a machine where all
tiers can be inspected (even if Tier 2/3 tests can't execute without GPU).

---

## What "Done" Looks Like

Phase 1 is test-complete when:

1. `cargo test` passes (Tier 1 — all platforms)
2. `cargo test --features cuda-tests` passes (Tier 2 — CUDA machine)
3. `cargo test --features model-tests` passes (Tier 3 — GPU + model)
4. `vex check` reports zero gaps against the vexspec
5. Greedy generation on 3 test prompts produces byte-identical output to PyTorch
6. All CUDA kernels match PyTorch reference within FP16 tolerance on all test cases

---

## Cross-Platform Testing (Phase 5+)

When the Metal backend is added, testing expands to cover backend equivalence
and mixed-backend pipelines. See `fracture_crossplatform_testing.md` for the
full strategy. Summary:

- **Per-backend correctness:** Each backend independently validated against the same
  PyTorch reference tensors. CUDA tests run with `--features cuda-tests`, Metal
  tests with `--features metal-tests`.
- **Backend equivalence:** Golden output files generated by each backend for fixed
  prompts, committed to repo, compared by a platform-agnostic test. Greedy decoding
  should produce identical token sequences for 100+ tokens.
- **Mixed-pipeline testing:** Manual test with CUDA and Metal nodes in the same
  pipeline. Run when Backend trait, wire protocol, or tensor serialization changes.
- **Reference tensor distribution:** Generated independently on each platform by
  running the same `dump_reference.py` script (deterministic). No need to transfer
  large files between machines.
