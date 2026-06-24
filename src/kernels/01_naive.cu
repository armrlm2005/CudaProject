/*
 * Kernel 1: Naive SGEMM
 * ----------------------
 * The simplest possible mapping: launch one thread per entry of C.
 * Each thread computes a full dot product between a row of A and a
 * column of B, with no caching or reuse of any kind.
 *
 * Why this is slow:
 *   - Every thread independently re-reads its row of A and column of B
 *     from global memory, with zero data reuse across threads.
 *   - Worse, because of how we assign (x, y) below, threads in the same
 *     warp end up reading non-contiguous addresses of A (each thread
 *     reads a different *row*, so consecutive threadIdx.x values jump
 *     K floats apart in memory). That means the GPU cannot coalesce
 *     these reads into wide memory transactions, and instead issues a
 *     separate narrow transaction per thread - a huge waste of memory
 *     bandwidth.
 *
 * This kernel exists purely as the performance floor / correctness
 * baseline that every later kernel is measured against.
 */

#include "gemm.cuh"

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
  // Compute the (x, y) entry of C this thread is responsible for.
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
      acc += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * acc + beta * C[x * N + y];
  }
}

void runKernel1Naive(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
