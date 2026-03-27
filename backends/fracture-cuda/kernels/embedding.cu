#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Embedding lookup: output[i] = table[token_ids[i]]
// table is [vocab_size, hidden_dim] FP16, output is [num_tokens, hidden_dim] FP16
__global__ void embedding_kernel(
    half* __restrict__ output,
    const half* __restrict__ table,
    const uint32_t* __restrict__ token_ids,
    int num_tokens,
    int hidden_dim,
    int vocab_size
) {
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (token_idx >= num_tokens || dim_idx >= hidden_dim) return;

    uint32_t token_id = token_ids[token_idx];
    // Bounds check: out-of-vocab IDs get zero
    if (token_id >= (uint32_t)vocab_size) {
        output[token_idx * hidden_dim + dim_idx] = __float2half(0.0f);
        return;
    }

    output[token_idx * hidden_dim + dim_idx] = table[token_id * hidden_dim + dim_idx];
}

extern "C" cudaError_t launch_embedding(
    void* output,
    const void* table,
    const uint32_t* token_ids,
    int num_tokens,
    int hidden_dim,
    int vocab_size,
    cudaStream_t stream
) {
    // Grid: one block row per token, multiple block columns to cover hidden_dim
    int threads = 256;
    dim3 grid(num_tokens, (hidden_dim + threads - 1) / threads);
    embedding_kernel<<<grid, threads, 0, stream>>>(
        (half*)output, (const half*)table, token_ids,
        num_tokens, hidden_dim, vocab_size
    );
    return cudaGetLastError();
}
