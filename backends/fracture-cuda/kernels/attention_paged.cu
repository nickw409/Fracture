#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

// Paged Grouped-Query Attention with causal masking.
//
// Instead of contiguous K/V cache arrays, KV data is stored in fixed-size
// blocks (BLOCK_SIZE tokens each). A block_table maps logical block indices
// to physical block IDs in the block pool.
//
// Q:              [num_tokens, num_q_heads, head_dim] FP16
// block_table:    [num_blocks] int32 — maps logical block idx → physical block ID
// k_block_ptrs:   [total_pool_blocks] void* — base pointers for K data per block for this layer
// v_block_ptrs:   [total_pool_blocks] void* — base pointers for V data per block for this layer
// Output:         [num_tokens, num_q_heads, head_dim] FP16
//
// Each K/V block is [BLOCK_SIZE, num_kv_heads, head_dim] FP16.
//
// GQA: each group of (num_q_heads / num_kv_heads) Q heads shares one KV head.
// Causal mask: query at position (start_pos + q_idx) attends to keys at positions 0..start_pos+q_idx.
// Softmax computed in FP32 for numerical stability.

#define PAGED_BLOCK_SIZE 16

__global__ void paged_attention_kernel(
    half* __restrict__ output,
    const half* __restrict__ q,
    const int* __restrict__ block_table,       // [num_blocks_in_seq]
    const half** __restrict__ k_block_ptrs,    // [pool_capacity] → K block base ptrs
    const half** __restrict__ v_block_ptrs,    // [pool_capacity] → V block base ptrs
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int kv_len,              // total valid tokens across all blocks
    int start_pos,           // tokens before this batch (for causal mask)
    int num_blocks_in_seq    // number of blocks in block_table
) {
    int token_idx = blockIdx.x;
    int q_head = blockIdx.y;
    if (token_idx >= num_tokens || q_head >= num_q_heads) return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;

    int causal_len = start_pos + token_idx + 1;
    if (causal_len > kv_len) causal_len = kv_len;

    const half* q_vec = q + (token_idx * num_q_heads + q_head) * head_dim;
    float scale = rsqrtf((float)head_dim);

    // Phase 1: Compute attention scores over all blocks
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;

    float max_score = -FLT_MAX;

    for (int kv_pos = threadIdx.x; kv_pos < causal_len; kv_pos += blockDim.x) {
        // Map kv_pos to block and offset within block
        int logical_block = kv_pos / PAGED_BLOCK_SIZE;
        int offset_in_block = kv_pos % PAGED_BLOCK_SIZE;
        int physical_block = block_table[logical_block];

        // K data: k_block_ptrs[physical_block] points to [BLOCK_SIZE, num_kv_heads, head_dim]
        const half* k_block = k_block_ptrs[physical_block];
        const half* k_vec = k_block + (offset_in_block * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __half2float(q_vec[d]) * __half2float(k_vec[d]);
        }
        dot *= scale;

        scores[kv_pos] = dot;
        if (dot > max_score) max_score = dot;
    }
    __syncthreads();

    // Reduce max across threads (same pattern as contiguous attention)
    __shared__ float shared_max[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, max_score, offset);
        max_score = fmaxf(max_score, other);
    }
    if (lane == 0) shared_max[warp_id] = max_score;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        max_score = (lane < num_warps) ? shared_max[lane] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, max_score, offset);
            max_score = fmaxf(max_score, other);
        }
    }

    __shared__ float global_max;
    if (threadIdx.x == 0) global_max = max_score;
    __syncthreads();

    // Phase 2: exp(score - max) and sum
    float local_sum = 0.0f;
    for (int kv_pos = threadIdx.x; kv_pos < causal_len; kv_pos += blockDim.x) {
        float val = expf(scores[kv_pos] - global_max);
        scores[kv_pos] = val;
        local_sum += val;
    }
    __syncthreads();

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }

    __shared__ float global_sum;
    if (threadIdx.x == 0) global_sum = local_sum;
    __syncthreads();

    // Normalize
    float inv_sum = 1.0f / global_sum;
    for (int kv_pos = threadIdx.x; kv_pos < causal_len; kv_pos += blockDim.x) {
        scores[kv_pos] *= inv_sum;
    }
    __syncthreads();

    // Phase 3: Weighted sum of V vectors (reading from blocks)
    half* out_vec = output + (token_idx * num_q_heads + q_head) * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < causal_len; kv_pos++) {
            int logical_block = kv_pos / PAGED_BLOCK_SIZE;
            int offset_in_block = kv_pos % PAGED_BLOCK_SIZE;
            int physical_block = block_table[logical_block];

            const half* v_block = v_block_ptrs[physical_block];
            const half* v_vec = v_block + (offset_in_block * num_kv_heads + kv_head) * head_dim;
            acc += scores[kv_pos] * __half2float(v_vec[d]);
        }
        out_vec[d] = __float2half(acc);
    }
}

extern "C" cudaError_t launch_paged_attention(
    void* output,
    const void* q,
    const int* block_table,
    const void** k_block_ptrs,
    const void** v_block_ptrs,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int kv_len,
    int start_pos,
    int num_blocks_in_seq,
    cudaStream_t stream
) {
    dim3 grid(num_tokens, num_q_heads);
    int threads = 128;

    size_t shared_size = kv_len * sizeof(float) + 64 * sizeof(float);

    paged_attention_kernel<<<grid, threads, shared_size, stream>>>(
        (half*)output, (const half*)q,
        block_table,
        (const half**)k_block_ptrs, (const half**)v_block_ptrs,
        num_tokens, num_q_heads, num_kv_heads, head_dim,
        kv_len, start_pos, num_blocks_in_seq
    );
    return cudaGetLastError();
}
