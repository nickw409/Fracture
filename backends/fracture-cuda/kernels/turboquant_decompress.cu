#include <cuda_fp16.h>
#include <cuda_runtime.h>

// TurboQuant decompression kernel (test utility).
//
// Unpacks bit-packed indices, looks up centroids, unrotates, and rescales.
// Used for round-trip validation — the production path fuses decompression
// into the attention kernel (attention_paged_tq.cu).
//
// packed_in:   [N, num_kv_heads * packed_dim_per_head] INT8
// norms_in:    [N, num_kv_heads] FP16
// rotation:    [head_dim, head_dim] FP32 (same matrix used for compression)
// centroids:   [n_levels] FP32
// output:      [N, num_kv_heads, head_dim] FP16
//
// Grid:  (N, num_kv_heads)
// Block: 128 threads

#define TQD_THREADS 128

__global__ void turboquant_decompress_kernel(
    const unsigned char* __restrict__ packed_in,   // [N, num_kv_heads * packed_dim_per_head]
    const half* __restrict__ norms_in,             // [N, num_kv_heads]
    const float* __restrict__ rotation,            // [head_dim, head_dim] row-major
    const float* __restrict__ centroids,           // [n_levels]
    half* __restrict__ output,                     // [N, num_kv_heads, head_dim]
    int num_kv_heads,
    int head_dim,
    int bits,
    int packed_dim_per_head
) {
    int token = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* y_hat = smem;            // [head_dim] — dequantized rotated vector
    float* x_hat = smem + head_dim; // [head_dim] — unrotated vector

    // Step 1: Unpack indices and lookup centroids
    const unsigned char* in_base = packed_in + (token * num_kv_heads + head) * packed_dim_per_head;

    if (bits == 4) {
        int num_pairs = (head_dim + 1) / 2;
        for (int pair = tid; pair < num_pairs; pair += TQD_THREADS) {
            unsigned char packed = in_base[pair];
            int idx0 = (packed >> 4) & 0x0F;
            int idx1 = packed & 0x0F;
            int d0 = pair * 2;
            int d1 = d0 + 1;
            y_hat[d0] = centroids[idx0];
            if (d1 < head_dim) {
                y_hat[d1] = centroids[idx1];
            }
        }
    } else if (bits == 2) {
        int num_quads = (head_dim + 3) / 4;
        for (int quad = tid; quad < num_quads; quad += TQD_THREADS) {
            unsigned char packed = in_base[quad];
            for (int sub = 0; sub < 4; sub++) {
                int d = quad * 4 + sub;
                if (d < head_dim) {
                    int idx = (packed >> (6 - sub * 2)) & 0x03;
                    y_hat[d] = centroids[idx];
                }
            }
        }
    } else if (bits == 8) {
        for (int d = tid; d < head_dim; d += TQD_THREADS) {
            int idx = in_base[d];
            y_hat[d] = centroids[idx];
        }
    }
    __syncthreads();

    // Step 2: Unrotate  x_hat = Pi^T @ y_hat
    for (int i = tid; i < head_dim; i += TQD_THREADS) {
        float dot = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            // Pi^T[i,j] = Pi[j,i] (transpose, since Pi is row-major)
            dot += rotation[j * head_dim + i] * y_hat[j];
        }
        x_hat[i] = dot;
    }
    __syncthreads();

    // Step 3: Rescale by stored norm
    float norm = __half2float(norms_in[token * num_kv_heads + head]);

    half* out_vec = output + (token * num_kv_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += TQD_THREADS) {
        out_vec[d] = __float2half(x_hat[d] * norm);
    }
}

extern "C" cudaError_t launch_turboquant_decompress(
    const void* packed_in,
    const void* norms_in,
    const float* rotation,
    const float* centroids,
    void* output,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int bits,
    int packed_dim_per_head,
    cudaStream_t stream
) {
    dim3 grid(num_tokens, num_kv_heads);
    int threads = TQD_THREADS;

    // Shared memory: y_hat[head_dim] + x_hat[head_dim]
    size_t shared_size = 2 * head_dim * sizeof(float);

    turboquant_decompress_kernel<<<grid, threads, shared_size, stream>>>(
        (const unsigned char*)packed_in,
        (const half*)norms_in,
        rotation,
        centroids,
        (half*)output,
        num_kv_heads,
        head_dim,
        bits,
        packed_dim_per_head
    );
    return cudaGetLastError();
}
