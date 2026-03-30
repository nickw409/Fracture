# Fracture Phase 5: Architecture Document
## Cross-Platform Metal Backend

**Depends on:** Phase 4 complete and validated (paged KV cache, continuous batching working)
**Goal:** Implement the Backend trait for Apple Silicon using Metal, enabling the Mac Studio (M2 Ultra, 64 GB unified memory) to participate as a node in the distributed inference pipeline alongside CUDA nodes — without changing the engine, generation loop, server, or wire protocol.

---

## What Changes from Phase 4

Phase 4 made Fracture a production inference system with continuous batching and paged KV cache. But every node in the pipeline must be a CUDA GPU on Linux. Phase 5 removes that constraint.

| Component | Phase 4 | Phase 5 |
|---|---|---|
| GPU backends | CUDA only | CUDA + Metal |
| Supported platforms | Linux | Linux + macOS |
| Cluster composition | Homogeneous (all NVIDIA) | Heterogeneous (NVIDIA + Apple Silicon) |
| Max cluster memory | 56 GB (RTX 3090 + RTX 5090) | 120 GB (+ M2 Ultra 64 GB) |
| Largest model GPU-resident | Llama 3 70B INT4 (~35 GB) | Llama 3 70B FP16 (~140 GB, tight) |
| Engine/server/protocol changes | N/A | Zero |

The Backend trait abstraction — designed in Phase 1 specifically for this moment — means `MetalBackend` is a new crate implementing an existing interface. The engine is generic over `B: Backend` and never imports any backend crate. The wire protocol transfers raw bytes. A CUDA worker's activation output and a Metal worker's activation input are the same byte format.

---

## Metal Execution Model vs CUDA

Understanding the conceptual mapping between CUDA and Metal is essential for translating the backend:

| CUDA Concept | Metal Equivalent | Notes |
|---|---|---|
| CUDA stream | MTLCommandQueue + MTLCommandBuffer | Metal groups work into command buffers, not a persistent stream |
| cuBLAS | Metal Performance Shaders (MPS) | `MPSMatrixMultiplication` for gemm |
| CUDA kernel (`.cu`) | Compute shader (`.metal`) | MSL is C++-like, not CUDA C++ |
| `cudaMalloc` / `cudaFree` | `device.makeBuffer()` / drop | Ref-counted, no explicit free needed |
| `cudaMemcpy` H↔D | `memcpy` to/from buffer `contents()` | Unified memory — no PCIe transfer |
| Device memory + host memory | Unified memory (shared) | CPU and GPU see the same physical memory |
| `threadIdx.x` | `thread_position_in_threadgroup` | Same concept, different name |
| `blockIdx.x` | `threadgroup_position_in_grid` | Same concept |
| `__shared__` memory | `threadgroup` memory | Same concept |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` | Same concept |
| `__shfl_down_sync()` | `simd_shuffle_down()` | SIMD group width = 32 on Apple Silicon |
| Warp (32 threads) | SIMD group (32 threads) | Same width on M-series |
| Dynamic shared memory | `setThreadgroupMemoryLength:atIndex:` | Set at dispatch time |
| NVTX markers | `os_signpost` | macOS profiling API, visible in Instruments |
| `cudaEventRecord` / `cudaEventElapsedTime` | Command buffer `GPUStartTime`/`GPUEndTime` | Available after completion |

### Unified Memory: The Key Difference

CUDA separates host (CPU) and device (GPU) memory. Transfers go over PCIe (12-64 GB/s depending on gen). Weight loading for Llama 3 8B takes ~1-2 seconds on PCIe 3.0.

Apple Silicon uses unified memory. CPU and GPU share the same physical DRAM (~800 GB/s bandwidth on M2 Ultra). An `MTLBuffer` with `StorageModeShared` is accessible to both without any transfer. "Copying" weights to the device is just a `memcpy` into the buffer's address space — the GPU reads from the same physical pages.

This means:
- `copy_to_device` = `memcpy` (CPU-side, no GPU involvement)
- `copy_to_host` = `memcpy` (same)
- Weight loading is faster (no PCIe bottleneck)
- The 64 GB unified memory is shared between model weights, KV cache, activations, and the OS

The Backend trait's `copy_to_device`/`copy_to_host` API works naturally — the implementation just happens to be a memcpy on both sides.

---

## MetalBackend Struct Design

```rust
pub struct MetalBackend {
    device: metal::Device,                    // MTLDevice
    command_queue: metal::CommandQueue,        // MTLCommandQueue
    state: Mutex<MetalState>,                 // buffer registry + allocation tracking
    timer_manager: timers::TimerManager,
    device_name: String,
    total_memory: usize,

    // Pre-computed RoPE frequency table (same as CudaBackend::rope_freq_table)
    rope_freq_buffer: Option<metal::Buffer>,

    // Pipeline State Objects — compiled once, reused for every dispatch
    rmsnorm_pso: metal::ComputePipelineState,
    rope_pso: metal::ComputePipelineState,
    silu_mul_pso: metal::ComputePipelineState,
    attention_pso: metal::ComputePipelineState,
    attention_paged_pso: metal::ComputePipelineState,
    embedding_pso: metal::ComputePipelineState,
    add_pso: metal::ComputePipelineState,
    copy_rows_pso: metal::ComputePipelineState,
}

struct MetalState {
    buffers: HashMap<u64, metal::Buffer>,   // TensorId → MTLBuffer
    next_id: u64,
    allocated_bytes: usize,                 // Track for available_memory()
}
```

Mirroring the CUDA backend pattern:
- `CudaBackend.state` has `tensors: HashMap<u64, *mut c_void>` → `MetalState.buffers: HashMap<u64, metal::Buffer>`
- `CudaBackend.stream` (persistent) → `MetalBackend.command_queue` (creates command buffers per dispatch)
- `CudaBackend.cublas_handle` → MPS objects created per matmul call (stateless API)
- Pipeline State Objects (PSOs) are Metal's compiled kernel programs, analogous to keeping function pointers to CUDA kernels

### Constructor

```rust
impl MetalBackend {
    pub fn new() -> Result<Self> {
        let device = metal::Device::system_default()
            .ok_or(FractureError::Backend("No Metal device found".into()))?;

        let command_queue = device.new_command_queue();
        let total_memory = device.recommended_max_working_set_size() as usize;

        // Load compiled shader library from embedded metallib
        let library_data = include_bytes!(concat!(env!("OUT_DIR"), "/fracture_kernels.metallib"));
        let library = device.new_library_with_data(library_data)?;

        // Create PSOs for each kernel function
        let rmsnorm_pso = Self::make_pso(&device, &library, "rmsnorm_kernel")?;
        let rope_pso = Self::make_pso(&device, &library, "rope_kernel")?;
        // ... etc for all kernels

        Ok(Self {
            device,
            command_queue,
            state: Mutex::new(MetalState {
                buffers: HashMap::new(),
                next_id: 1,
                allocated_bytes: 0,
            }),
            timer_manager: timers::TimerManager::new(),
            device_name: device.name().to_string(),
            total_memory,
            rope_freq_buffer: None,
            rmsnorm_pso,
            rope_pso,
            // ...
        })
    }
}
```

---

## Crate Structure

```
backends/fracture-metal/
├── Cargo.toml
├── build.rs
├── src/
│   ├── lib.rs           # MetalBackend struct, Backend impl, buffer registry
│   ├── mps.rs           # FFI bindings to MPSMatrixMultiplication (matmul)
│   ├── timers.rs        # GPU timing via command buffer GPUStartTime/GPUEndTime
│   └── signpost.rs      # os_signpost profiling markers
└── kernels/
    ├── add.metal
    ├── silu_mul.metal
    ├── embedding.metal
    ├── rmsnorm.metal
    ├── rope.metal
    ├── attention.metal
    └── attention_paged.metal
```

### Build System (build.rs)

Metal shaders are compiled with Apple's `xcrun` toolchain:

```
For each kernels/*.metal:
  xcrun -sdk macosx metal -c -O2 -ffast-math <file>.metal -o <file>.air

Then link all .air into a single library:
  xcrun -sdk macosx metallib *.air -o fracture_kernels.metallib
```

The `.metallib` is written to `OUT_DIR` and embedded at runtime via `include_bytes!`. This is analogous to the CUDA backend's `build.rs` which compiles `.cu` files with `nvcc` and links them into a static library.

On non-macOS platforms, `build.rs` exits early (no `xcrun` available). The crate compiles but `MetalBackend::new()` returns an error.

### Dependencies

```toml
[package]
name = "fracture-metal"

[dependencies]
fracture-core = { workspace = true }
tracing = { workspace = true }
half = { workspace = true }

[target.'cfg(target_os = "macos")'.dependencies]
metal = "0.29"          # Safe wrappers around MTLDevice, MTLCommandQueue, etc.
objc = "0.2"            # Raw Obj-C runtime for MPS FFI
block = "0.1"           # Obj-C blocks for MPS callbacks
```

### Workspace Additions

Root `Cargo.toml`:
```toml
# Add to workspace members:
"backends/fracture-metal",
"bins/fracture-server-metal",
"bins/fracture-worker-metal",

# Add workspace dependency:
fracture-metal = { path = "backends/fracture-metal" }
```

---

## Backend Trait Implementation

### Memory Management

```rust
fn alloc(&self, shape: &[usize], dtype: DType) -> Result<DeviceTensor> {
    let size_bytes = shape.iter().product::<usize>() * dtype.size_bytes();
    let buffer = self.device.new_buffer(size_bytes as u64, MTLResourceOptions::StorageModeShared);

    let mut state = self.state.lock().unwrap();
    let id = state.next_id;
    state.next_id += 1;
    state.buffers.insert(id, buffer);
    state.allocated_bytes += size_bytes;

    DeviceTensor::try_new(TensorId(id), shape.to_vec(), dtype)
}

fn copy_to_device(&self, dst: &DeviceTensor, src: &[u8]) -> Result<()> {
    let state = self.state.lock().unwrap();
    let buffer = state.buffers.get(&dst.id().0).ok_or(/* ... */)?;
    // Unified memory: just memcpy into the buffer's CPU-accessible address
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), buffer.contents() as *mut u8, src.len());
    }
    Ok(())
}

fn copy_to_host(&self, src: &DeviceTensor, dst: &mut [u8]) -> Result<()> {
    // Same as copy_to_device but reversed — unified memory makes both directions a memcpy
    let state = self.state.lock().unwrap();
    let buffer = state.buffers.get(&src.id().0).ok_or(/* ... */)?;
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.contents() as *const u8, dst.as_mut_ptr(), dst.len());
    }
    Ok(())
}
```

### Kernel Dispatch Pattern

Every compute method follows the same pattern:

```rust
fn rmsnorm(&self, input: &DeviceTensor, weight: &DeviceTensor, eps: f64, out: &DeviceTensor) -> Result<()> {
    let state = self.state.lock().unwrap();
    let input_buf = state.buffers.get(&input.id().0).ok_or(/* ... */)?;
    let weight_buf = state.buffers.get(&weight.id().0).ok_or(/* ... */)?;
    let out_buf = state.buffers.get(&out.id().0).ok_or(/* ... */)?;

    let cmd_buf = self.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&self.rmsnorm_pso);
    encoder.set_buffer(0, Some(input_buf), 0);
    encoder.set_buffer(1, Some(weight_buf), 0);
    encoder.set_buffer(2, Some(out_buf), 0);

    // Scalar params via setBytes
    let params = RmsnormParams { rows: input.shape()[0] as u32, cols: input.shape()[1] as u32, eps: eps as f32 };
    encoder.set_bytes(3, std::mem::size_of::<RmsnormParams>() as u64, &params as *const _ as *const _);

    // Dispatch: one threadgroup per row, 256 threads per threadgroup
    let threadgroup_size = MTLSize::new(256, 1, 1);
    let grid_size = MTLSize::new(input.shape()[0] as u64, 1, 1);
    encoder.dispatch_threadgroups(grid_size, threadgroup_size);

    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    Ok(())
}
```

**Performance note:** Creating a new command buffer and waiting for completion per kernel call is suboptimal. Each `commit()` + `waitUntilCompleted()` has non-trivial overhead (~10-50 μs). During a single decode step, the engine calls ~10+ Backend methods per layer × 32 layers = ~320+ dispatch/wait cycles. This is correct but slow.

Optimization (post-initial implementation): Use a thread-local "active" command buffer that accumulates multiple kernel dispatches, only committing on `synchronize()` or when the engine needs results (e.g., before `copy_to_host`). This mirrors how the CUDA backend implicitly batches work on a stream and only synchronizes when needed.

### Matmul via MPS

The CUDA backend uses cuBLAS for matmul. The Metal equivalent is `MPSMatrixMultiplication` from Metal Performance Shaders:

```rust
fn matmul(&self, a: &DeviceTensor, b: &DeviceTensor, out: &DeviceTensor) -> Result<()> {
    // A: [M, K], B: [N, K] (row-major, B stored as N×K in GGUF)
    // Compute: out = A @ B^T → out: [M, N]
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[0];

    let state = self.state.lock().unwrap();
    let a_buf = state.buffers.get(&a.id().0).ok_or(/* ... */)?;
    let b_buf = state.buffers.get(&b.id().0).ok_or(/* ... */)?;
    let out_buf = state.buffers.get(&out.id().0).ok_or(/* ... */)?;

    // Create MPS matrix descriptors (FP16, row bytes = cols * 2)
    // MPSMatrixMultiplication: C = alpha * op(A) * op(B) + beta * C
    // With transpose_right=true: C = A @ B^T
    let gemm = MPSMatrixMultiplication::new(
        &self.device,
        false,   // transpose_left
        true,    // transpose_right
        m, n, k,
        1.0,     // alpha
        0.0,     // beta
    );

    let cmd_buf = self.command_queue.new_command_buffer();
    gemm.encode(cmd_buf, a_matrix, b_matrix, out_matrix);
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    Ok(())
}
```

MPS internally uses FP32 accumulation with FP16 inputs, matching the project's precision convention. The `MPSMatrixMultiplication` class is optimized for Apple Silicon's AMX (Apple Matrix eXtension) coprocessor, which provides hardware matrix multiply acceleration similar to NVIDIA Tensor Cores.

**MPS FFI:** The `metal` Rust crate does not wrap MPS. The `mps.rs` file provides targeted Objective-C FFI bindings for the 3 MPS classes needed: `MPSMatrixMultiplication`, `MPSMatrix`, `MPSMatrixDescriptor`. This follows the same pattern as `fracture-cuda/src/ffi.rs` — minimal, hand-written bindings for exactly the functionality needed.

---

## Kernel Translations (CUDA → Metal Shading Language)

Each CUDA `.cu` kernel maps to a `.metal` file with the same logic in MSL syntax. The kernels are ordered by implementation complexity.

### add.metal (Trivial)

Element-wise addition for residual connections. Direct port:

```metal
kernel void add_kernel(
    device const half* a      [[buffer(0)]],
    device const half* b      [[buffer(1)]],
    device half* out          [[buffer(2)]],
    constant uint& n          [[buffer(3)]],
    uint idx                  [[thread_position_in_grid]]
) {
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
```

### silu_mul.metal (Trivial)

Fused SiLU gating for FFN: `out = silu(gate) * up` with FP32 intermediate:

```metal
kernel void silu_mul_kernel(
    device const half* gate   [[buffer(0)]],
    device const half* up     [[buffer(1)]],
    device half* out          [[buffer(2)]],
    constant uint& n          [[buffer(3)]],
    uint idx                  [[thread_position_in_grid]]
) {
    if (idx < n) {
        float g = float(gate[idx]);
        float u = float(up[idx]);
        float silu_g = g / (1.0f + exp(-g));
        out[idx] = half(silu_g * u);
    }
}
```

### embedding.metal (Simple)

2D dispatch: one thread per (token, hidden_dim_element). Same out-of-vocab zero-fill behavior as CUDA:

```metal
kernel void embedding_kernel(
    device const uint* token_ids       [[buffer(0)]],
    device const half* embedding_table [[buffer(1)]],
    device half* out                   [[buffer(2)]],
    constant uint& num_tokens          [[buffer(3)]],
    constant uint& hidden_dim          [[buffer(4)]],
    constant uint& vocab_size          [[buffer(5)]],
    uint2 pos                          [[thread_position_in_grid]]
) {
    uint token = pos.y;
    uint dim = pos.x;
    if (token >= num_tokens || dim >= hidden_dim) return;

    uint id = token_ids[token];
    half val = (id < vocab_size) ? embedding_table[id * hidden_dim + dim] : half(0.0);
    out[token * hidden_dim + dim] = val;
}
```

### rmsnorm.metal (Moderate — threadgroup reduction)

One threadgroup per row. Requires warp-level reduction using `simd_shuffle_down` and threadgroup barrier:

```metal
kernel void rmsnorm_kernel(
    device const half* input   [[buffer(0)]],
    device const half* weight  [[buffer(1)]],
    device half* out           [[buffer(2)]],
    constant uint& rows        [[buffer(3)]],
    constant uint& cols        [[buffer(4)]],
    constant float& eps        [[buffer(5)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint tid                   [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]]
) {
    // Each thread accumulates partial sum-of-squares across its stride
    threadgroup float partial_sums[32]; // max 32 SIMD groups per threadgroup

    float sum_sq = 0.0f;
    for (uint i = tid; i < cols; i += THREADGROUP_SIZE) {
        float val = float(input[row * cols + i]);
        sum_sq += val * val;
    }

    // Warp-level reduction via simd_shuffle_down
    sum_sq = simd_sum(sum_sq);  // built-in SIMD reduction

    if (simd_lane == 0) partial_sums[simd_group] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across SIMD groups (first SIMD group only)
    // ... then rsqrt and multiply by weight
    float rms = rsqrt(total_sum / float(cols) + eps);
    for (uint i = tid; i < cols; i += THREADGROUP_SIZE) {
        out[row * cols + i] = half(float(input[row * cols + i]) * rms * float(weight[i]));
    }
}
```

### rope.metal (Moderate)

Rotary positional embeddings applied in-place to Q and K. Uses pre-computed frequency table stored in a Metal buffer (same as CUDA backend's `rope_freq_table`):

```metal
kernel void rope_kernel(
    device half* q             [[buffer(0)]],
    device half* k             [[buffer(1)]],
    device const uint* pos     [[buffer(2)]],  // position per token
    device const float* freqs  [[buffer(3)]],  // pre-computed frequency table
    constant uint& num_tokens  [[buffer(4)]],
    constant uint& num_q_heads [[buffer(5)]],
    constant uint& num_k_heads [[buffer(6)]],
    constant uint& head_dim    [[buffer(7)]],
    uint2 gid                  [[thread_position_in_grid]]
) {
    // gid.x = pair index within head (head_dim/2 pairs)
    // gid.y = (token * num_heads + head) linear index
    // Apply rotation: (x0, x1) → (x0*cos - x1*sin, x0*sin + x1*cos)
    // Same split-half convention as CUDA kernel and PyTorch/HuggingFace
}
```

### attention.metal (Complex — dynamic threadgroup memory)

Grouped-query attention with causal masking. Most complex kernel. One threadgroup per (token, query_head) pair:

```metal
kernel void attention_kernel(
    device const half* q          [[buffer(0)]],
    device const half* k_cache    [[buffer(1)]],
    device const half* v_cache    [[buffer(2)]],
    device half* out              [[buffer(3)]],
    constant AttentionParams& p   [[buffer(4)]],
    threadgroup float* scores     [[threadgroup(0)]], // dynamic threadgroup memory
    uint2 gid                     [[threadgroup_position_in_grid]],
    uint tid                      [[thread_position_in_threadgroup]]
) {
    uint token = gid.y;
    uint q_head = gid.x;
    uint kv_head = q_head / (p.num_q_heads / p.num_kv_heads); // GQA mapping

    // 1. Compute dot products: Q[token, q_head] · K[t, kv_head] for t in [0, kv_len)
    // 2. Apply causal mask (score = -inf for t > start_pos + token)
    // 3. Softmax in FP32 (online safe softmax with running max)
    // 4. Weighted sum over V[t, kv_head] → output
}
```

**Dynamic threadgroup memory:** The scores array size depends on `kv_len` (number of cached tokens), which varies per request. Metal supports this via `setThreadgroupMemoryLength:atIndex:` on the compute command encoder, set at dispatch time — directly analogous to CUDA's `<<<grid, block, shared_mem_bytes>>>` third parameter.

### attention_paged.metal (Complex)

Same as `attention.metal` but reads K/V from block tables instead of contiguous cache. The CUDA version passes device pointer arrays (`void**`). Metal handles this differently:

**Approach: Flat block pool with offset-based indexing.** Instead of passing an array of buffer pointers, pass a single buffer containing the entire block pool and a block table mapping logical blocks to byte offsets within that buffer. The kernel computes: `k_data = pool_buffer + block_table[logical_block] + offset_within_block`.

This avoids Metal's restrictions on pointer-to-pointer arguments and simplifies the dispatch. The block pool is already pre-allocated at startup (from Phase 4's `BlockPool`), so this is just a different way of indexing into the same memory.

---

## Binary Crates

### fracture-server-metal

Mirrors `bins/fracture-server-cuda/` exactly, replacing `CudaBackend` with `MetalBackend`:

```rust
// bins/fracture-server-metal/src/main.rs
use fracture_metal::MetalBackend;

fn main() -> anyhow::Result<()> {
    let backend = MetalBackend::new()?;
    // Identical to CUDA server from here: load model, create engine, start HTTP server
    // The engine, generation loop, and HTTP server are generic over Backend
}
```

### fracture-worker-metal

Mirrors `bins/fracture-worker-cuda/`:

```rust
// bins/fracture-worker-metal/src/main.rs
use fracture_metal::MetalBackend;

fn main() -> anyhow::Result<()> {
    let backend = MetalBackend::new()?;
    // Identical to CUDA worker: calibrate, register with coordinator, serve forward requests
}
```

**No Metal coordinator binary needed.** The coordinator (`bins/fracture-coordinator-cuda/`) orchestrates workers over the wire protocol and never touches GPU memory. It can coordinate Metal workers as-is.

---

## Mixed-Backend Distributed Pipeline

The wire protocol is already backend-agnostic. Tensor data is serialized as raw bytes (shape + dtype + data). A CUDA worker and a Metal worker produce and consume the same byte format because:

1. Both use FP16 (IEEE 754 half-precision) — same bit representation
2. Both use row-major tensor layout — same byte order
3. Both are little-endian architectures (x86-64 and ARM64)

```
Example: Llama 3 70B FP16 on 3-node cluster

Coordinator (any machine, no GPU needed)
    ↕ wire protocol
Worker 0 — Mac Studio M2 Ultra (Metal)
    layers [0, 40)     ← 64 GB unified memory, fits ~40 layers + cache
    ↕ wire protocol
Worker 1 — RTX 5090 (CUDA)
    layers [40, 60)    ← 32 GB VRAM
    ↕ wire protocol
Worker 2 — RTX 3090 (CUDA)
    layers [60, 80)    ← 24 GB VRAM

Total: 120 GB → fits Llama 3 70B FP16 (~140 GB weights)
Mac Studio carries the bulk load thanks to 64 GB unified memory.
```

### Scheduler Compatibility

The Phase 3 scheduler assigns layers based on measured compute speed (`decode_ms_per_layer`, `prefill_ms_per_layer`) and available memory. Metal workers report these same metrics during calibration. The scheduler's auto mode handles heterogeneous hardware by design — it was built for RTX 3090 + RTX 5090 asymmetry and extends naturally to CUDA + Metal.

The scheduler may need awareness that Metal's memory bandwidth (~800 GB/s on M2 Ultra) differs from CUDA's (936 GB/s on RTX 3090, 1.8 TB/s on RTX 5090). For memory-bandwidth-bound decode steps, the M2 Ultra sits between the 3090 and 5090 in throughput. Calibration benchmarks capture this automatically — no scheduler changes needed.

---

## Memory Budget (M2 Ultra, 64 GB Unified Memory)

```
Total unified memory: 64 GB
System reservation: ~8-10 GB (macOS + system processes)
Available for Fracture: ~54 GB (device.recommendedMaxWorkingSetSize)

Llama 3 8B FP16:
  Weights: 15.3 GB
  Available for KV cache: 54 - 15.3 - 1 (overhead) ≈ 37.7 GB
  Paged blocks (2 MB each): ~18,850 blocks → 301,600 tokens
  This is 2.4× more cache than the RTX 5090 (15.7 GB)

Llama 3 70B FP16 (40 layers on Metal worker):
  Weights for 40 layers: ~40 × 520 MB ≈ 20.8 GB
  Shared layers (embedding, LM head): ~2 GB
  Total weights: ~22.8 GB
  Available for KV cache: 54 - 22.8 - 1 ≈ 30.2 GB
  Paged blocks (40-layer, 1.25 MB each): ~24,160 blocks → 386,560 tokens
```

The M2 Ultra's large unified memory makes it the highest-capacity node in the cluster, able to hold more layers or more KV cache than any single NVIDIA card.

---

## Profiling and Timers

### GPU Timers (timers.rs)

Metal does not have CUDA-style event pairs (`cudaEventRecord`, `cudaEventElapsedTime`). Instead, use command buffer GPU timestamps:

```rust
struct TimerState {
    start_time: Option<f64>,  // captured from command buffer GPUStartTime
}

// start_timer: record that the next command buffer should capture start time
// stop_timer: commit current command buffer, wait, read GPUEndTime - GPUStartTime
```

After `commandBuffer.waitUntilCompleted()`, the properties `GPUStartTime` and `GPUEndTime` return `CFTimeInterval` (seconds). The elapsed time is `(GPUEndTime - GPUStartTime) * 1000.0` milliseconds.

### Profiling Markers (signpost.rs)

macOS equivalent of NVTX (NVIDIA Tools Extension):

```rust
use os_signpost::{SignpostInterval, SignpostLog};

pub fn marker_push(name: &str) {
    // os_signpost_interval_begin — visible in Instruments.app
}
pub fn marker_pop() {
    // os_signpost_interval_end
}
```

These markers appear in Xcode Instruments' Metal System Trace and GPU profiler, enabling the same workflow as CUDA's Nsight Systems with NVTX markers.

---

## Platform Gating

The `fracture-metal` crate and Metal binary crates must compile on all platforms (the workspace is checked on Linux too) but only function on macOS:

```rust
// backends/fracture-metal/src/lib.rs

#[cfg(target_os = "macos")]
mod metal_impl;
#[cfg(target_os = "macos")]
pub use metal_impl::MetalBackend;

#[cfg(not(target_os = "macos"))]
mod stub;
#[cfg(not(target_os = "macos"))]
pub use stub::MetalBackend;

// stub.rs: MetalBackend::new() returns Err("Metal backend requires macOS")
```

The `metal` crate dependency is platform-gated:
```toml
[target.'cfg(target_os = "macos")'.dependencies]
metal = "0.29"
```

This keeps `cargo check` working on Linux (CI, the main development machine) while Metal functionality is macOS-only.

---

## Testing Strategy

### Unit Tests (within fracture-metal)

Mirror the CUDA backend's test suite. Each kernel gets correctness tests against known reference values computed in Python/Rust:

- Memory management: alloc/free round-trip, copy round-trip, double-free error, available memory tracking
- add: verify element-wise addition
- silu_mul: verify against reference `silu(gate) * up`
- embedding: verify row lookup, out-of-vocab zero fill
- rmsnorm: verify against FP32 reference (allow FP16 tolerance 1e-2)
- rope: verify against reference RoPE output
- matmul: verify MPS result against reference A @ B^T
- attention: verify GQA with causal mask against reference (single head, multi-head, various kv_len)
- attention_paged: verify paged attention matches contiguous attention

### Tier 1: Metal Kernels vs PyTorch Reference

The existing `tests/reference/` infrastructure generates per-layer intermediate tensors from PyTorch. These binary files are platform-agnostic. Run the Metal backend against the same reference data:

```bash
# On Mac Studio:
cargo test -p fracture-metal   # unit tests + reference validation
```

### Tier 2: Backend Equivalence (Golden Output)

Per the existing cross-platform testing strategy (`docs/fracture_crossplatform_testing.md`):

```bash
# On Linux/CUDA machine:
cargo run --bin fracture-test-generate -- --output tests/golden/cuda_output.json

# On Mac Studio:
cargo run --bin fracture-test-generate -- --output tests/golden/metal_output.json

# On any machine (no GPU):
cargo test --test backend_equivalence
```

Greedy decoding should produce identical token sequences for 100+ tokens. Divergence before token 50 indicates a real bug. Divergence after token 100+ is likely numerical drift from FP16 FMA rounding differences between CUDA and Metal.

### Tier 3: Mixed-Backend Pipeline (Manual)

Run coordinator on Linux, CUDA worker on Linux, Metal worker on Mac Studio. Compare output against single-backend output:

1. Generate 256 tokens with greedy decoding on CUDA-only pipeline
2. Generate 256 tokens with same prompt on mixed CUDA+Metal pipeline
3. Assert token sequences match for 100+ tokens

This test is manual and only needs to run when:
- The Backend trait interface changes
- The wire protocol tensor serialization changes
- A Metal kernel is added or modified

---

## Implementation Order

| Step | Name | Depends On | Description |
|---|---|---|---|
| 5a | Crate skeleton + build.rs | Nothing | MetalBackend struct, stub trait impl, xcrun shader build |
| 5b | Memory + device info | 5a | alloc/free/copy, device_name/total_memory/available_memory, timers |
| 5c-1 | add.metal | 5b | Element-wise addition (simplest, establishes dispatch pattern) |
| 5c-2 | silu_mul.metal | 5b | SiLU gating (element-wise, FP32 intermediate) |
| 5c-3 | embedding.metal | 5b | Token embedding lookup (2D dispatch) |
| 5c-4 | rmsnorm.metal | 5b | RMSNorm (first threadgroup reduction kernel) |
| 5c-5 | rope.metal | 5b | Rotary embeddings (freq table buffer) |
| 5d | MPS matmul | 5b | MPSMatrixMultiplication FFI + Backend::matmul |
| 5c-6 | attention.metal | 5c-4 (test patterns) | GQA attention (most complex, dynamic threadgroup memory) |
| 5c-7 | attention_paged.metal | 5c-6 | Paged attention (flat pool + offset indexing) |
| 5e | Server + worker binaries | All of 5c, 5d | fracture-server-metal, fracture-worker-metal |
| 5f | Validation | 5e | Unit tests, Tier 1/2/3 testing, vexspec update |

**Critical path:** 5a → 5b → 5d (matmul) + 5c-4 (rmsnorm) → 5c-6 (attention) → 5c-7 (paged) → 5e → 5f

Start with the simplest kernels (add, silu_mul, embedding) to establish the Metal dispatch pattern and build confidence before tackling the reduction and attention kernels.

---

## Anticipated Challenges

### MPS FFI Bindings

The `metal` Rust crate wraps core Metal objects but not Metal Performance Shaders. Writing targeted Objective-C FFI for `MPSMatrixMultiplication` is required. This is bounded: only 3-4 MPS classes are needed, and the pattern matches the CUDA backend's `ffi.rs` approach.

### Paged Attention Memory Layout

CUDA passes device pointer arrays (`void**`) for block-table indexed attention. Metal's argument model doesn't support pointer-to-pointer naturally. Solution: use a flat block pool buffer with offset-based indexing. The kernel computes addresses as `pool_base + block_table[i] * block_stride`. This may require Phase 4's `BlockPool` to support a flat-buffer allocation mode in addition to the per-block `DeviceTensor` approach.

### FP16 Precision Divergence

CUDA and Metal implement FP16 FMA (fused multiply-add) differently. Small rounding differences accumulate over 32 layers of computation. Expected impact:
- Individual kernel outputs: within 1e-2 tolerance
- End-to-end greedy decode: identical tokens for ~100-200 tokens, possible divergence beyond that
- This is acceptable and documented in the cross-platform testing strategy

### Command Buffer Overhead

One command buffer per Backend method call means ~320+ commit/wait cycles per decode step. Initial implementation accepts this overhead for correctness. Optimization path: batch multiple kernel dispatches into a single command buffer, committing only on `synchronize()` or `copy_to_host()`. This could be implemented as an opt-in `begin_batch()`/`end_batch()` pattern or a thread-local accumulating command buffer.

### macOS-Only Development

All Metal work must happen on the Mac Studio. The non-Metal crates (engine, server, protocol) can still be developed on Linux. Recommended workflow: SSH + VS Code Remote from the Linux workstation, or develop on the Mac Studio directly for Metal-specific work.

---

## What Phase 5 Does NOT Include (Deferred)

- **Vulkan/WebGPU backend:** Only Metal. Other cross-platform GPU APIs are out of scope.
- **Mixed quantization across backends:** All nodes run the same quantization level.
- **Automatic backend selection:** Binaries are backend-specific (`fracture-server-cuda` vs `fracture-server-metal`). No runtime backend detection.
- **Shared command buffer batching:** Initial implementation uses one command buffer per dispatch. Batching is a performance optimization for later.
- **iOS/iPadOS support:** Metal exists on iOS but Fracture targets macOS only (server workloads).

---

## Success Criteria (Phase 5 Complete)

1. `MetalBackend` implements all 22 Backend trait methods
2. All Metal kernel unit tests pass on Mac Studio
3. Metal kernel outputs match PyTorch reference within FP16 tolerance
4. `fracture-server-metal` loads Llama 3 8B, generates correct text via HTTP
5. `fracture-worker-metal` registers with coordinator, participates in distributed pipeline
6. Greedy decode output matches CUDA backend for 100+ tokens (golden output comparison)
7. Mixed CUDA+Metal pipeline produces correct output over wire protocol
8. Workspace compiles on both Linux (CUDA) and macOS (Metal) via platform gating
