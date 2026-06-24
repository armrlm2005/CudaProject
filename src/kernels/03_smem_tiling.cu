/*
 * Kernel 3: Shared Memory Cache-Blocking
 * -----------------------------------------
 * Even with coalesced reads, kernel 2 still re-reads the same rows of A
 * and columns of B from slow global memory over and over across
 * different blocks. This kernel introduces shared memory (SMEM): a
 * small but very fast on-chip memory region shared by all threads in a
 * block.
 *
 * Strategy: tile A and B into BLOCKSIZE x BLOCKSIZE chunks. Each block
 * of threads cooperatively loads one chunk of A and one chunk of B
 * into SMEM, then every thread in the block reuses those cached values
 * to compute a partial dot product, before moving on to the next chunk
 * along the K dimension. This way, each value loaded from global
 * memory gets reused BLOCKSIZE times instead of once.
 *
 * Why this matters: shared memory has roughly an order of magnitude
 * higher bandwidth and lower latency than global memory, so shifting
 * reuse there directly reduces the number of slow GMEM transactions.
 *
 * __syncthreads() is required twice per loop iteration: once after
 * loading into SMEM (so no thread starts computing before the tile is
 * fully populated), and once after computing (so no thread overwrites
 * the SMEM tile for the next iteration before slower threads finish
 * using the current one).
 */

#include "gemm.cuh"

template <const int BLOCKSIZE>
__global__ void sgemm_smem_tiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  const uint threadRow = threadIdx.x / BLOCKSIZE;
  const uint threadCol = threadIdx.x % BLOCKSIZE;

  // Move A/B/C pointers to the start of the tiles this block owns.
  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  float acc = 0.0f;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Cooperative load: each thread brings in exactly one element of
    // the A-tile and one element of the B-tile. threadCol is the
    // consecutive index here too, to keep these loads coalesced.
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      acc += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }

    __syncthreads();
  }

  C[threadRow * N + threadCol] =
      alpha * acc + beta * C[threadRow * N + threadCol];
}

void runKernel3SmemTiling(int M, int N, int K, float alpha, const float *A,
                           const float *B, float beta, float *C) {
  const uint BLOCKSIZE = 32;
  dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  sgemm_smem_tiling<BLOCKSIZE>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
