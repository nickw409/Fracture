# Fracture: Cross-Platform Testing Strategy (Phase 5+ Addendum)

## The Problem

When Fracture has both CUDA and Metal backends, you need to verify three things:

1. **Each backend independently produces correct output** (matches PyTorch reference)
2. **Both backends produce the same output as each other** (backend equivalence)
3. **Mixed-backend pipelines work** (CUDA node sends activation to Metal node, result is correct)

These tests span two different operating systems (Linux for CUDA, macOS for Metal)
and two completely different GPU programming models. You cannot run both backends
on the same machine.

---

## Tier 1: Per-Backend Correctness (Same as Phase 1)

Each backend is validated independently against the PyTorch reference tensors.
The reference tensors are platform-agnostic — they're just binary files with shapes
and float values.

```
tests/reference/          ← Generated once by dump_reference.py (on any machine)
  prompt_0/
    layer_00/
      post_attn_norm.bin
      ...
```

**On the Linux/CUDA machine:**
```bash
cargo test --features cuda-tests    # runs all Tier 2 tests against CUDA backend
```

**On the Mac Studio:**
```bash
cargo test --features metal-tests   # runs all Tier 2 tests against Metal backend
```

Both test suites load the same reference tensors and assert closeness against
the same tolerances. If both pass, each backend is independently correct.

**Key: the reference tensors must be committed to the repo** (or stored in a
shared location both machines can access). They're generated once on any machine
with PyTorch and the model weights, then used everywhere.

---

## Tier 2: Backend Equivalence

Even if both backends match PyTorch within tolerance, they might disagree with
each other due to different floating-point behavior. CUDA and Metal have different
rounding modes, different fused multiply-add implementations, and different
approaches to FP16 ↔ FP32 conversion. Small differences accumulate over 32 layers.

**The test:** Run the same prompt with temperature=0 (greedy) on both backends.
Compare the generated token sequence.

This test cannot run on one machine. Two approaches:

### Option A: Golden Output Files (Recommended)

Each backend generates output for a fixed set of test prompts and writes the
results to a file:

```bash
# On Linux/CUDA machine:
cargo run --bin fracture-test-generate -- \
  --model /path/to/model.gguf \
  --prompts tests/prompts.json \
  --output tests/golden/cuda_output.json

# On Mac Studio:
cargo run --bin fracture-test-generate -- \
  --model /path/to/model.gguf \
  --prompts tests/prompts.json \
  --output tests/golden/metal_output.json
```

The golden output files are committed to the repo. A CI-agnostic test then
compares them:

```bash
# On any machine (no GPU needed):
cargo test --test backend_equivalence
# Loads cuda_output.json and metal_output.json, asserts token sequences match
```

**Acceptable divergence:** Greedy decoding should produce identical token sequences
for at least the first ~100 tokens. Beyond that, tiny floating-point differences
can compound and cause different token selections at low-confidence positions.
If the backends diverge at token 50 on a test prompt, there's likely a real bug.
If they diverge at token 500, it might just be numerical drift.

The test should report WHERE divergence starts, not just pass/fail:
```
Prompt "The capital of France is":
  CUDA:  [" Paris", ".", " It", " is", " known", ...]
  Metal: [" Paris", ".", " It", " is", " known", ...]
  Match: 256/256 tokens identical ✓

Prompt "Explain quantum computing":
  CUDA:  [" Quantum", " computing", " is", ...]
  Metal: [" Quantum", " computing", " is", ...]
  Divergence at token 187: CUDA=" the" Metal=" a"
  (187 matching tokens before divergence — likely numerical drift, not a bug)
```

### Option B: CI with Both Platforms

If you ever set up CI with access to both a CUDA GPU and a Mac runner (GitHub
Actions has macOS runners, but no GPU; self-hosted runners could have both):

```yaml
# Hypothetical CI workflow
jobs:
  test-cuda:
    runs-on: [self-hosted, cuda]
    steps:
      - run: cargo test --features cuda-tests
      - run: cargo run --bin fracture-test-generate -- --output golden/cuda.json
      - upload-artifact: golden/cuda.json

  test-metal:
    runs-on: [self-hosted, macos, metal]
    steps:
      - run: cargo test --features metal-tests
      - run: cargo run --bin fracture-test-generate -- --output golden/metal.json
      - upload-artifact: golden/metal.json

  compare:
    needs: [test-cuda, test-metal]
    runs-on: ubuntu-latest
    steps:
      - download-artifacts
      - run: cargo test --test backend_equivalence
```

This is ideal but requires self-hosted runners with GPUs. For a personal project,
Option A is more practical.

---

## Tier 3: Mixed-Backend Pipeline Testing

The hardest test: a CUDA node and a Metal node in the same pipeline, cooperating
over the wire protocol.

**What could go wrong:**
- Byte order / endianness differences (both are little-endian, so unlikely, but verify)
- FP16 representation differences (IEEE 754 is standard, but edge cases around NaN/Inf handling may differ)
- Tensor layout assumptions (row-major convention must be enforced by both backends)
- Serialization/deserialization mismatches

**The test setup:**
```
Mac Studio (Metal backend)              Linux box (CUDA backend)
  layers [0, 40)           ──network──    layers [40, 80)
  head node                               tail node
```

**Validation:**
1. Run the full model on CUDA-only (single node or two CUDA nodes) with temperature=0
2. Run the same model on the mixed pipeline (Metal head + CUDA tail) with temperature=0
3. Compare generated token sequences

If they match for 100+ tokens, the pipeline is working. If they diverge immediately,
there's a serialization or layout bug.

**This test is manual.** You run the coordinator on one machine, workers on both
machines, point them at each other, and compare output. It's not automatable in
CI without both machines online, but it only needs to be run when:
- The Backend trait changes
- The wire protocol tensor serialization changes
- A new backend is added or modified

---

## Test Matrix Summary

| Test | Runs on | Frequency | Automated? |
|---|---|---|---|
| CUDA kernel ↔ PyTorch reference | Linux + GPU | Every commit | Yes (self-hosted CI) |
| Metal kernel ↔ PyTorch reference | macOS + M2 Ultra | Every commit | Yes (self-hosted CI) |
| CUDA golden output generation | Linux + GPU | Per release | Semi (script) |
| Metal golden output generation | macOS + M2 Ultra | Per release | Semi (script) |
| Backend equivalence (golden compare) | Any machine | Per release | Yes (no GPU needed) |
| Mixed pipeline (CUDA + Metal) | Both machines on network | Major changes only | Manual |

---

## Practical Workflow

Day-to-day development on CUDA:
```bash
cargo test                          # Tier 1 (no GPU) — fast
cargo test --features cuda-tests    # Tier 2 (CUDA kernels) — seconds
cargo test --features model-tests   # Tier 3 (end-to-end) — minutes
```

When working on Metal backend:
```bash
cargo test                          # Tier 1 (no GPU) — same tests, same machine
cargo test --features metal-tests   # Tier 2 (Metal kernels) — seconds
cargo test --features model-tests   # Tier 3 (end-to-end) — minutes
```

Before a release:
```bash
# On CUDA machine:
cargo run --bin fracture-test-generate -- --output golden/cuda.json

# On Mac Studio:
cargo run --bin fracture-test-generate -- --output golden/metal.json

# On any machine:
cargo test --test backend_equivalence

# Manual: run mixed pipeline test if backend or protocol changed
```

---

## Reference Tensor Distribution

The PyTorch reference tensors (`tests/reference/`) are large (potentially
gigabytes for a full 32-layer dump). Options for sharing between machines:

1. **Git LFS** — store in the repo, both machines pull automatically
2. **Shared network drive** — NAS or NFS mount accessible from both machines
3. **Generate on each machine** — run dump_reference.py on both (requires
   PyTorch + model weights on both machines, but guarantees identical references)

Option 3 is the most reliable. The reference script is deterministic (same model,
same input, same output), so generating independently on both machines produces
identical files. This also eliminates endianness or filesystem concerns.
