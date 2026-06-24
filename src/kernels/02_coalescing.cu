/*
 * Kernel 2: Global Memory Coalescing
 * ------------------------------------
 * Same amount of work as Kernel 1, but we change which thread is
 * responsible for which output element so that threads within a warp
 * read *consecutive* addresses from global memory.
 *
 * Background: threads execute in groups of 32 called a warp. If the
 * 32 threads of a warp access consecutive addresses in global memory,
 * the hardware can merge ("coalesce") those into a small number of
 * wide memory transactions (e.g. a single 128-byte load instead of 32
 * separate ones). If the accesses are scattered, the GPU has to issue
 * many narrow transactions instead, wasting most of the fetched bytes.
 *
 * The fix is a one-line change to how we compute (x, y) from
 * threadIdx: instead of using threadIdx.x for the row and threadIdx.y
 * for the column (kernel 1), we flip it so that consecutive threadIdx
 * values map to consecutive columns of B / consecutive output columns
 * in the same row of C. This means consecutive threads in a warp now
 * read consecutive floats from B, which the GPU can coalesce.
 *
 * Result: global memory throughput goes up dramatically even though
 * we are doing the exact same number of FLOPs and the exact same
 * number of bytes transferred per thread - it's purely about *how*
 * those transfers are issued.
 */

#include "gemm.cuh"

template <const int BLOCKSIZE>
__global__ void sgemm_coalescing(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta,
                                  float *C) {
  const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N) {
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
      acc += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * acc + beta * C[x * N + y];
  }
}

void runKernel2Coalescing(int M, int N, int K, float alpha, const float *A,
                           const float *B, float beta, float *C) {
  const uint BLOCKSIZE = 32;
  dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  sgemm_coalescing<BLOCKSIZE>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
