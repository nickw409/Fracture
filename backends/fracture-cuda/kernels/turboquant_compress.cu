#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// TurboQuant compression kernel: normalize → rotate → Lloyd-Max quantize → bit-pack.
//
// Input:       [N, num_kv_heads, head_dim] FP16 — K or V vectors
// rotation:    [head_dim, head_dim] FP32 — Haar-distributed orthogonal rotation matrix
// centroids:   [n_levels] FP32 — Lloyd-Max centroid table (2^bits levels)
// packed_out:  [N, num_kv_heads * packed_dim_per_head] INT8 — bit-packed quantized indices
// norms_out:   [N, num_kv_heads] FP16 — per-head L2 norms
//
// Grid:  (N, num_kv_heads) — one thread block per (token, head)
// Block: 128 threads
//
// Each block processes one head_dim-sized vector:
//   1. Load FP16 vector → shared memory as FP32
//   2. Compute L2 norm via parallel reduction
//   3. Normalize with epsilon guard
//   4. Rotate: y = Pi @ x_norm  (matrix-vector multiply)
//   5. Quantize: binary search centroids per coordinate
//   6. Bit-pack indices into bytes

#define TQ_THREADS 128
#define TQ_EPSILON 1e-8f

__global__ void turboquant_compress_kernel(
    const half* __restrict__ input,          // [N, num_kv_heads, head_dim]
    const float* __restrict__ rotation,      // [head_dim, head_dim] row-major
    const float* __restrict__ centroids,     // [n_levels]
    unsigned char* __restrict__ packed_out,   // [N, num_kv_heads * packed_dim_per_head]
    half* __restrict__ norms_out,            // [N, num_kv_heads]
    int num_kv_heads,
    int head_dim,
    int n_levels,       // 2^bits
    int bits,           // 2, 4, or 8
    int packed_dim_per_head  // ceil(head_dim * bits / 8)
) {
    int token = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;

    // Shared memory: input vector (FP32) + rotated vector (FP32)
    extern __shared__ float smem[];
    float* x_vec = smem;                      // [head_dim]
    float* y_vec = smem + head_dim;           // [head_dim]

    // Step 1: Load input vector from [token, head, :] into shared memory
    const half* in_ptr = input + (token * num_kv_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += TQ_THREADS) {
        x_vec[d] = __half2float(in_ptr[d]);
    }
    __syncthreads();

    // Step 2: Compute L2 norm via parallel reduction
    float local_sum_sq = 0.0f;
    for (int d = tid; d < head_dim; d += TQ_THREADS) {
        local_sum_sq += x_vec[d] * x_vec[d];
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    __shared__ float warp_sums[32];
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    if (lane == 0) warp_sums[warp_id] = local_sum_sq;
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        int num_warps = (TQ_THREADS + warpSize - 1) / warpSize;
        float val = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float norm = sqrtf(warp_sums[0]);

    // Store norm
    if (tid == 0) {
        norms_out[token * num_kv_heads + head] = __float2half(norm);
    }

    // Step 3: Normalize with epsilon guard
    float inv_norm = 1.0f / (norm + TQ_EPSILON);
    for (int d = tid; d < head_dim; d += TQ_THREADS) {
        x_vec[d] *= inv_norm;
    }
    __syncthreads();

    // Step 4: Rotate  y = Pi @ x_norm
    // Each thread computes a subset of output coordinates
    for (int i = tid; i < head_dim; i += TQ_THREADS) {
        float dot = 0.0f;
        const float* pi_row = rotation + i * head_dim;
        for (int j = 0; j < head_dim; j++) {
            dot += pi_row[j] * x_vec[j];
        }
        y_vec[i] = dot;
    }
    __syncthreads();

    // Step 5 + 6: Quantize each coordinate and pack into bytes
    // Output layout: packed_out[token, head * packed_dim_per_head + byte_offset]
    unsigned char* out_base = packed_out + (token * num_kv_heads + head) * packed_dim_per_head;

    if (bits == 4) {
        // 4-bit: 2 indices per byte
        int num_pairs = (head_dim + 1) / 2;
        for (int pair = tid; pair < num_pairs; pair += TQ_THREADS) {
            int d0 = pair * 2;
            int d1 = d0 + 1;

            // Quantize d0: binary search for nearest centroid
            float val0 = y_vec[d0];
            int idx0 = 0;
            float best_dist0 = fabsf(val0 - centroids[0]);
            for (int c = 1; c < n_levels; c++) {
                float dist = fabsf(val0 - centroids[c]);
                if (dist < best_dist0) {
                    best_dist0 = dist;
                    idx0 = c;
                }
            }

            int idx1 = 0;
            if (d1 < head_dim) {
                float val1 = y_vec[d1];
                float best_dist1 = fabsf(val1 - centroids[0]);
                for (int c = 1; c < n_levels; c++) {
                    float dist = fabsf(val1 - centroids[c]);
                    if (dist < best_dist1) {
                        best_dist1 = dist;
                        idx1 = c;
                    }
                }
            }

            // Pack: high nibble = idx0, low nibble = idx1
            out_base[pair] = (unsigned char)((idx0 << 4) | (idx1 & 0x0F));
        }
    } else if (bits == 2) {
        // 2-bit: 4 indices per byte
        int num_quads = (head_dim + 3) / 4;
        for (int quad = tid; quad < num_quads; quad += TQ_THREADS) {
            unsigned char packed = 0;
            for (int sub = 0; sub < 4; sub++) {
                int d = quad * 4 + sub;
                int idx = 0;
                if (d < head_dim) {
                    float val = y_vec[d];
                    float best_dist = fabsf(val - centroids[0]);
                    for (int c = 1; c < n_levels; c++) {
                        float dist = fabsf(val - centroids[c]);
                        if (dist < best_dist) {
                            best_dist = dist;
                            idx = c;
                        }
                    }
                }
                packed |= (unsigned char)(idx << (6 - sub * 2));
            }
            out_base[quad] = packed;
        }
    } else if (bits == 8) {
        // 8-bit: 1 index per byte, straight write
        for (int d = tid; d < head_dim; d += TQ_THREADS) {
            float val = y_vec[d];
            int idx = 0;
            float best_dist = fabsf(val - centroids[0]);
            for (int c = 1; c < n_levels; c++) {
                float dist = fabsf(val - centroids[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    idx = c;
                }
            }
            out_base[d] = (unsigned char)idx;
        }
    }
}

extern "C" cudaError_t launch_turboquant_compress(
    const void* input,
    const float* rotation,
    const float* centroids,
    void* packed_out,
    void* norms_out,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int n_levels,
    int bits,
    int packed_dim_per_head,
    cudaStream_t stream
) {
    dim3 grid(num_tokens, num_kv_heads);
    int threads = TQ_THREADS;

    // Shared memory: x_vec[head_dim] + y_vec[head_dim]
    size_t shared_size = 2 * head_dim * sizeof(float);

    turboquant_compress_kernel<<<grid, threads, shared_size, stream>>>(
        (const half*)input,
        rotation,
        centroids,
        (unsigned char*)packed_out,
        (half*)norms_out,
        num_kv_heads,
        head_dim,
        n_levels,
        bits,
        packed_dim_per_head
    );
    return cudaGetLastError();
}
