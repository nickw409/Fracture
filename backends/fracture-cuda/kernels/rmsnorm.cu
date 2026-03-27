#include <cuda_fp16.h>
#include <cuda_runtime.h>

// RMSNorm: output = (x / sqrt(mean(x^2) + eps)) * weight
// Input: [rows, cols] FP16, weight: [cols] FP16, output: [rows, cols] FP16
// Uses FP32 accumulation for sum-of-squares to prevent precision loss.
//
// One block per row. Warp-level reduction for the sum of squares.

__global__ void rmsnorm_kernel(
    half* __restrict__ output,
    const half* __restrict__ input,
    const half* __restrict__ weight,
    float eps,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const half* x = input + row * cols;
    half* out = output + row * cols;

    // Compute sum of squares in FP32
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float shared[32]; // max 32 warps per block
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum_sq = (lane < num_warps) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    // Broadcast the final sum to all threads
    __shared__ float rms_inv;
    if (threadIdx.x == 0) {
        float mean_sq = sum_sq / (float)cols;
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    // Apply normalization and weight
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = __half2float(x[i]);
        float w = __half2float(weight[i]);
        out[i] = __float2half(val * rms_inv * w);
    }
}

extern "C" cudaError_t launch_rmsnorm(
    void* output,
    const void* input,
    const void* weight,
    float eps,
    int rows,
    int cols,
    cudaStream_t stream
) {
    // One block per row, 256 threads
    int threads = (cols < 256) ? cols : 256;
    rmsnorm_kernel<<<rows, threads, 0, stream>>>(
        (half*)output, (const half*)input, (const half*)weight,
        eps, rows, cols
    );
    return cudaGetLastError();
}
