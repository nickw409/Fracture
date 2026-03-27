#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Elementwise addition: output = a + b (all FP16)
__global__ void add_kernel(
    half* __restrict__ output,
    const half* __restrict__ a,
    const half* __restrict__ b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = __hadd(a[idx], b[idx]);
}

extern "C" cudaError_t launch_add(
    void* output,
    const void* a,
    const void* b,
    int n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, stream>>>(
        (half*)output, (const half*)a, (const half*)b, n
    );
    return cudaGetLastError();
}
