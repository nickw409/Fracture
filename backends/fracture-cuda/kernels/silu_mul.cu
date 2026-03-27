#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Fused SiLU activation and elementwise multiply: output = silu(gate) * up
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// All tensors are FP16, computation in FP32 for precision.

__global__ void silu_mul_kernel(
    half* __restrict__ output,
    const half* __restrict__ gate,
    const half* __restrict__ up,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);

    // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
    float silu_g = g / (1.0f + expf(-g));
    output[idx] = __float2half(silu_g * u);
}

extern "C" cudaError_t launch_silu_mul(
    void* output,
    const void* gate,
    const void* up,
    int n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads, 0, stream>>>(
        (half*)output, (const half*)gate, (const half*)up, n
    );
    return cudaGetLastError();
}
