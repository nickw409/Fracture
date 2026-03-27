#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

// Grouped-Query Attention with causal masking.
//
// Q:       [num_tokens, num_q_heads, head_dim] FP16
// K_cache: [max_seq_len, num_kv_heads, head_dim] FP16 (only first kv_len entries valid)
// V_cache: [max_seq_len, num_kv_heads, head_dim] FP16
// Output:  [num_tokens, num_q_heads, head_dim] FP16
//
// GQA: each group of (num_q_heads / num_kv_heads) Q heads shares one KV head.
// Causal mask: query at position (start_pos + q_idx) attends to keys at positions 0..start_pos+q_idx.
// Softmax computed in FP32 for numerical stability.
//
// Naive implementation: one block per (token, q_head) pair.
// This is correct but not optimal — FlashAttention-style tiling comes later.

__global__ void attention_kernel(
    half* __restrict__ output,
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int kv_len,
    int start_pos
) {
    // One block handles one (token, q_head) pair
    int token_idx = blockIdx.x;
    int q_head = blockIdx.y;
    if (token_idx >= num_tokens || q_head >= num_q_heads) return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;

    // Causal: this query at position (start_pos + token_idx) can attend to 0..start_pos+token_idx
    int causal_len = start_pos + token_idx + 1;
    if (causal_len > kv_len) causal_len = kv_len;

    // Pointer to this query vector: Q[token_idx, q_head, :]
    const half* q_vec = q + (token_idx * num_q_heads + q_head) * head_dim;

    float scale = rsqrtf((float)head_dim);

    // Phase 1: Compute attention scores and find max (for numerically stable softmax)
    // Using shared memory for scores
    extern __shared__ float shared_mem[];
    float* scores = shared_mem; // [causal_len] stored across thread iterations

    float max_score = -FLT_MAX;

    for (int kv_pos = threadIdx.x; kv_pos < causal_len; kv_pos += blockDim.x) {
        // K_cache[kv_pos, kv_head, :]
        const half* k_vec = k_cache + (kv_pos * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __half2float(q_vec[d]) * __half2float(k_vec[d]);
        }
        dot *= scale;

        scores[kv_pos] = dot;
        if (dot > max_score) max_score = dot;
    }
    __syncthreads();

    // Reduce max across threads
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

    // Phase 2: Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (int kv_pos = threadIdx.x; kv_pos < causal_len; kv_pos += blockDim.x) {
        float val = expf(scores[kv_pos] - global_max);
        scores[kv_pos] = val;
        local_sum += val;
    }
    __syncthreads();

    // Reduce sum across threads
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

    // Normalize scores to probabilities
    float inv_sum = 1.0f / global_sum;
    for (int kv_pos = threadIdx.x; kv_pos < causal_len; kv_pos += blockDim.x) {
        scores[kv_pos] *= inv_sum;
    }
    __syncthreads();

    // Phase 3: Weighted sum of V vectors
    // Output[token_idx, q_head, :] = sum_kv(prob[kv] * V_cache[kv, kv_head, :])
    half* out_vec = output + (token_idx * num_q_heads + q_head) * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < causal_len; kv_pos++) {
            const half* v_vec = v_cache + (kv_pos * num_kv_heads + kv_head) * head_dim;
            acc += scores[kv_pos] * __half2float(v_vec[d]);
        }
        out_vec[d] = __float2half(acc);
    }
}

extern "C" cudaError_t launch_attention(
    void* output,
    const void* q,
    const void* k_cache,
    const void* v_cache,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int kv_len,
    int start_pos,
    cudaStream_t stream
) {
    // One block per (token, q_head)
    dim3 grid(num_tokens, num_q_heads);
    int threads = 128; // Keep moderate for shared memory usage

    // Shared memory: scores array needs kv_len floats + reduction scratch
    size_t shared_size = kv_len * sizeof(float) + 64 * sizeof(float);

    attention_kernel<<<grid, threads, shared_size, stream>>>(
        (half*)output, (const half*)q,
        (const half*)k_cache, (const half*)v_cache,
        num_tokens, num_q_heads, num_kv_heads, head_dim,
        kv_len, start_pos
    );
    return cudaGetLastError();
}
