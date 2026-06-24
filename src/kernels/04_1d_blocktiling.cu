/*
 * Kernel 4: 1D Block Tiling (multiple results per thread)
 * -----------------------------------------------------------
 * Kernel 3 is still bottlenecked on shared memory traffic: profiling
 * shows threads spending most of their time stalled waiting on SMEM
 * loads to return, rather than doing useful FMA work. The instruction
 * mix is dominated by loads, not arithmetic.
 *
 * The fix is to raise "arithmetic intensity" - the number of FLOPs we
 * do per byte loaded. Instead of one thread computing exactly one
 * output element, each thread now computes TM (e.g. 8) output elements
 * that all share the same column of B. This means we load one value of
 * B from SMEM and reuse it TM times in registers, instead of reloading
 * from SMEM for every single output.
 *
 * Concretely: each thread now holds a small per-thread accumulator
 * array `threadResults[TM]` in registers (not shared memory - this is
 * the fastest possible storage on the GPU), and the inner loop walks
 * down TM rows of the A-tile for a fixed column of the B-tile.
 */

#include "gemm.cuh"

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha,
                                      const float *A, const float *B,
                                      float beta, float *C) {
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // Each block has (BM * BN) / TM threads, one thread per output column
  // within a strip of TM rows.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // Indices used purely for the cooperative GMEM -> SMEM load step.
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  float threadResults[TM] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // Cache this B value once, reuse it TM times below.
      float Btmp = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
      }
    }
    __syncthreads();
  }

  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    int row = threadRow * TM + resIdx;
    C[row * N + threadCol] =
        alpha * threadResults[resIdx] + beta * C[row * N + threadCol];
  }
}

void runKernel4_1DBlocktiling(int M, int N, int K, float alpha,
                               const float *A, const float *B, float beta,
                               float *C) {
  const uint BM = 64, BN = 64, BK = 8, TM = 8;
  dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
  dim3 blockDim((BM * BN) / TM);
  sgemm_1d_blocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
