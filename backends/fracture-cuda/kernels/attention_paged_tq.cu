#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// TurboQuant Paged Attention: fused decompress + attention kernel.
//
// Instead of reading FP16 K/V blocks, this kernel reads bit-packed quantized
// blocks and decompresses on the fly during attention computation.
//
// Key optimization: Pre-rotate the query with Pi_k to compute attention scores
// directly in the rotated space, avoiding per-KV-position unrotation for K.
//   dot(q, Pi_k^T @ y_hat_k) * norm = dot(Pi_k @ q, y_hat_k) * norm
//
// V optimization: Accumulate the weighted sum in V's rotated space, then
// unrotate once at the end.
//   sum(prob_i * Pi_v^T @ y_hat_v_i * norm_i) = Pi_v^T @ sum(prob_i * y_hat_v_i * norm_i)
//
// Q:              [num_tokens, num_q_heads, head_dim] FP16
// block_table:    [num_blocks] int32
// k_packed_ptrs:  [pool_capacity] — packed K index tensors per block
// k_norms_ptrs:   [pool_capacity] — K norm tensors per block
// v_packed_ptrs:  [pool_capacity] — packed V index tensors per block
// v_norms_ptrs:   [pool_capacity] — V norm tensors per block
// k_rotation:     [head_dim, head_dim] FP32 — K rotation matrix
// v_rotation:     [head_dim, head_dim] FP32 — V rotation matrix
// k_centroids:    [2^key_bits] FP32
// v_centroids:    [2^value_bits] FP32
// Output:         [num_tokens, num_q_heads, head_dim] FP16
//
// Packed K block: [BLOCK_SIZE, num_kv_heads * k_packed_dim_per_head] INT8
// K norms block:  [BLOCK_SIZE, num_kv_heads] FP16
// (same layout for V with v_packed_dim_per_head)

#define TQ_ATTN_BLOCK_SIZE 16
#define TQ_ATTN_THREADS 128

// Unpack a single quantized index from a packed byte array.
__device__ __forceinline__ int unpack_index(
    const unsigned char* packed, int coord, int bits
) {
    if (bits == 4) {
        int byte_idx = coord / 2;
        int nibble = (coord % 2 == 0) ? ((packed[byte_idx] >> 4) & 0x0F)
                                       : (packed[byte_idx] & 0x0F);
        return nibble;
    } else if (bits == 2) {
        int byte_idx = coord / 4;
        int shift = 6 - (coord % 4) * 2;
        return (packed[byte_idx] >> shift) & 0x03;
    } else { // bits == 8
        return packed[coord];
    }
}

__global__ void attention_paged_tq_kernel(
    half* __restrict__ output,
    const half* __restrict__ q,
    const int* __restrict__ block_table,
    const unsigned char** __restrict__ k_packed_ptrs,  // [pool_cap] → packed K blocks
    const half** __restrict__ k_norms_ptrs,            // [pool_cap] → K norms blocks
    const unsigned char** __restrict__ v_packed_ptrs,   // [pool_cap] → packed V blocks
    const half** __restrict__ v_norms_ptrs,            // [pool_cap] → V norms blocks
    const float* __restrict__ k_rotation,              // [head_dim, head_dim]
    const float* __restrict__ v_rotation,              // [head_dim, head_dim]
    const float* __restrict__ k_centroids,             // [2^key_bits]
    const float* __restrict__ v_centroids,             // [2^value_bits]
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int kv_len,
    int start_pos,
    int key_bits,
    int value_bits,
    int k_packed_dim_per_head,
    int v_packed_dim_per_head
) {
    int token_idx = blockIdx.x;
    int q_head = blockIdx.y;
    if (token_idx >= num_tokens || q_head >= num_q_heads) return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;
    int tid = threadIdx.x;

    int causal_len = start_pos + token_idx + 1;
    if (causal_len > kv_len) causal_len = kv_len;

    float scale = rsqrtf((float)head_dim);

    // Shared memory layout:
    //   scores[kv_len]         — attention scores
    //   q_rot[head_dim]        — pre-rotated query (K space)
    //   acc_rot[head_dim]      — accumulated V in rotated space
    //   reduction_scratch[32]  — for warp reductions
    extern __shared__ float smem[];
    float* scores = smem;
    float* q_rot = smem + kv_len;
    float* acc_rot = q_rot + head_dim;

    // ── Step 0: Pre-rotate query with K rotation matrix ──
    // q_rot = Pi_k @ q[token, q_head, :]
    const half* q_vec = q + (token_idx * num_q_heads + q_head) * head_dim;
    for (int i = tid; i < head_dim; i += TQ_ATTN_THREADS) {
        float dot = 0.0f;
        const float* pi_row = k_rotation + i * head_dim;
        for (int j = 0; j < head_dim; j++) {
            dot += pi_row[j] * __half2float(q_vec[j]);
        }
        q_rot[i] = dot;
    }
    __syncthreads();

    // ── Phase 1: Attention scores in rotated K space ──
    float max_score = -FLT_MAX;

    for (int kv_pos = tid; kv_pos < causal_len; kv_pos += TQ_ATTN_THREADS) {
        int logical_block = kv_pos / TQ_ATTN_BLOCK_SIZE;
        int offset_in_block = kv_pos % TQ_ATTN_BLOCK_SIZE;
        int physical_block = block_table[logical_block];

        // Locate packed K data for this (kv_pos, kv_head)
        const unsigned char* k_packed_block = k_packed_ptrs[physical_block];
        const unsigned char* k_packed_head = k_packed_block
            + (offset_in_block * num_kv_heads + kv_head) * k_packed_dim_per_head;

        // Load K norm
        const half* k_norms_block = k_norms_ptrs[physical_block];
        float k_norm = __half2float(k_norms_block[offset_in_block * num_kv_heads + kv_head]);

        // Compute dot(q_rot, y_hat_k) directly — no unrotation needed
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            int idx = unpack_index(k_packed_head, d, key_bits);
            float centroid_val = k_centroids[idx];
            dot += q_rot[d] * centroid_val;
        }
        dot *= k_norm * scale;

        scores[kv_pos] = dot;
        if (dot > max_score) max_score = dot;
    }
    __syncthreads();

    // ── Max reduction across threads ──
    __shared__ float shared_max[32];
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, max_score, offset);
        max_score = fmaxf(max_score, other);
    }
    if (lane == 0) shared_max[warp_id] = max_score;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (TQ_ATTN_THREADS + warpSize - 1) / warpSize;
        max_score = (lane < num_warps) ? shared_max[lane] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, max_score, offset);
            max_score = fmaxf(max_score, other);
        }
    }

    __shared__ float global_max;
    if (tid == 0) global_max = max_score;
    __syncthreads();

    // ── Phase 2: Softmax (exp and sum) ──
    float local_sum = 0.0f;
    for (int kv_pos = tid; kv_pos < causal_len; kv_pos += TQ_ATTN_THREADS) {
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
        int num_warps = (TQ_ATTN_THREADS + warpSize - 1) / warpSize;
        local_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }

    __shared__ float global_sum;
    if (tid == 0) global_sum = local_sum;
    __syncthreads();

    // Normalize scores
    float inv_sum = 1.0f / global_sum;
    for (int kv_pos = tid; kv_pos < causal_len; kv_pos += TQ_ATTN_THREADS) {
        scores[kv_pos] *= inv_sum;
    }
    __syncthreads();

    // ── Phase 3: Weighted V sum in rotated space ──
    // acc_rot = sum_kv(prob[kv] * y_hat_v[kv] * v_norm[kv])
    // Then unrotate once: out = Pi_v^T @ acc_rot

    // Initialize accumulator
    for (int d = tid; d < head_dim; d += TQ_ATTN_THREADS) {
        acc_rot[d] = 0.0f;
    }
    __syncthreads();

    // Accumulate — each thread handles a subset of kv positions,
    // contributing to ALL dimensions (requires atomicAdd to shared)
    // Alternative: each thread handles a subset of dimensions for ALL kv positions.
    // The latter avoids atomics but iterates kv_len per thread.
    // For typical kv_len >> head_dim, the dimension-parallel approach is better.

    for (int d = tid; d < head_dim; d += TQ_ATTN_THREADS) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < causal_len; kv_pos++) {
            int logical_block = kv_pos / TQ_ATTN_BLOCK_SIZE;
            int offset_in_block = kv_pos % TQ_ATTN_BLOCK_SIZE;
            int physical_block = block_table[logical_block];

            const unsigned char* v_packed_block = v_packed_ptrs[physical_block];
            const unsigned char* v_packed_head = v_packed_block
                + (offset_in_block * num_kv_heads + kv_head) * v_packed_dim_per_head;

            const half* v_norms_block = v_norms_ptrs[physical_block];
            float v_norm = __half2float(v_norms_block[offset_in_block * num_kv_heads + kv_head]);

            int idx = unpack_index(v_packed_head, d, value_bits);
            float centroid_val = v_centroids[idx];

            acc += scores[kv_pos] * centroid_val * v_norm;
        }
        acc_rot[d] = acc;
    }
    __syncthreads();

    // ── Step 4: Unrotate output with V rotation matrix ──
    // out = Pi_v^T @ acc_rot
    half* out_vec = output + (token_idx * num_q_heads + q_head) * head_dim;

    for (int i = tid; i < head_dim; i += TQ_ATTN_THREADS) {
        float dot = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            // Pi_v^T[i, j] = Pi_v[j, i]
            dot += v_rotation[j * head_dim + i] * acc_rot[j];
        }
        out_vec[i] = __float2half(dot);
    }
}

extern "C" cudaError_t launch_attention_paged_tq(
    void* output,
    const void* q,
    const int* block_table,
    const void** k_packed_ptrs,
    const void** k_norms_ptrs,
    const void** v_packed_ptrs,
    const void** v_norms_ptrs,
    const float* k_rotation,
    const float* v_rotation,
    const float* k_centroids,
    const float* v_centroids,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int kv_len,
    int start_pos,
    int key_bits,
    int value_bits,
    int k_packed_dim_per_head,
    int v_packed_dim_per_head,
    cudaStream_t stream
) {
    dim3 grid(num_tokens, num_q_heads);
    int threads = TQ_ATTN_THREADS;

    // Shared memory: scores[kv_len] + q_rot[head_dim] + acc_rot[head_dim]
    size_t shared_size = kv_len * sizeof(float)
                       + head_dim * sizeof(float)
                       + head_dim * sizeof(float);

    attention_paged_tq_kernel<<<grid, threads, shared_size, stream>>>(
        (half*)output,
        (const half*)q,
        block_table,
        (const unsigned char**)k_packed_ptrs,
        (const half**)k_norms_ptrs,
        (const unsigned char**)v_packed_ptrs,
        (const half**)v_norms_ptrs,
        k_rotation,
        v_rotation,
        k_centroids,
        v_centroids,
        num_tokens, num_q_heads, num_kv_heads, head_dim,
        kv_len, start_pos,
        key_bits, value_bits,
        k_packed_dim_per_head, v_packed_dim_per_head
    );
    return cudaGetLastError();
}
