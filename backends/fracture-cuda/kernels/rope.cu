#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Rotary Positional Embeddings applied in-place to Q and K tensors.
//
// Q: [num_tokens, num_q_heads, head_dim] FP16
// K: [num_tokens, num_kv_heads, head_dim] FP16
// positions: [num_tokens] u32
// freq_table: [head_dim/2] FP32 — pre-computed inverse frequencies
//
// Uses the split-half convention (matching PyTorch/HuggingFace):
// For each (token, head, dimension d in 0..head_dim/2):
//   angle = positions[token] * freq_table[d]
//   cos_val = cos(angle), sin_val = sin(angle)
//   x0 = tensor[..., d],             x1 = tensor[..., d + head_dim/2]
//   tensor[..., d]              = x0 * cos_val - x1 * sin_val
//   tensor[..., d + head_dim/2] = x0 * sin_val + x1 * cos_val

__global__ void rope_kernel(
    half* __restrict__ tensor,
    const uint32_t* __restrict__ positions,
    const float* __restrict__ freq_table,
    int num_tokens,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total = num_tokens * num_heads * half_dim;
    if (idx >= total) return;

    int d = idx % half_dim;
    int remainder = idx / half_dim;
    int head = remainder % num_heads;
    int token = remainder / num_heads;

    float angle = (float)positions[token] * freq_table[d];
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Split-half: pair (d, d + half_dim) within each head
    int base = token * num_heads * head_dim + head * head_dim;
    float x0 = __half2float(tensor[base + d]);
    float x1 = __half2float(tensor[base + d + half_dim]);

    tensor[base + d]            = __float2half(x0 * cos_val - x1 * sin_val);
    tensor[base + d + half_dim] = __float2half(x0 * sin_val + x1 * cos_val);
}

extern "C" cudaError_t launch_rope(
    void* q,
    void* k,
    const uint32_t* positions,
    const float* freq_table,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream
) {
    int half_dim = head_dim / 2;
    int threads = 256;

    // Apply to Q
    int total_q = num_tokens * num_q_heads * half_dim;
    int blocks_q = (total_q + threads - 1) / threads;
    rope_kernel<<<blocks_q, threads, 0, stream>>>(
        (half*)q, positions, freq_table,
        num_tokens, num_q_heads, head_dim
    );

    // Apply to K
    int total_k = num_tokens * num_kv_heads * half_dim;
    int blocks_k = (total_k + threads - 1) / threads;
    rope_kernel<<<blocks_k, threads, 0, stream>>>(
        (half*)k, positions, freq_table,
        num_tokens, num_kv_heads, head_dim
    );

    return cudaGetLastError();
}
