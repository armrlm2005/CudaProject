/*
 * Kernel 5: 2D Block Tiling
 * -----------------------------
 * Kernel 4 raised arithmetic intensity by having each thread compute a
 * 1D strip (column) of TM results. We can do better: have each thread
 * compute a 2D tile of TM x TN results instead. A square tile shares
 * even more of its inputs between the outputs it computes than a
 * 1D strip does, which raises arithmetic intensity further still.
 *
 * Implementation-wise, each thread now keeps two small register
 * arrays, regM[TM] and regN[TN], loaded fresh from SMEM on every
 * iteration of the K-dimension loop. We then do an outer product
 * between them (TM * TN multiply-adds) and accumulate into a
 * threadResults[TM * TN] register array.
 *
 * The loop order matters: we load regM/regN *before* the TM x TN
 * outer-product loop, so each SMEM element is read into a register
 * exactly once and reused TM (or TN) times from there, rather than
 * being re-read from SMEM on every inner iteration.
 */

#include "gemm.cuh"

template <const int BM, const int BN, const int BK, const int TM,
          const int TN>
__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha,
                                      const float *A, const float *B,
                                      float beta, float *C) {
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  const uint totalResultsBlocktile = BM * BN;
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint strideA = numThreadsBlocktile / BK;

  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  const uint strideB = numThreadsBlocktile / BN;

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      int row = threadRow * TM + resIdxM;
      int col = threadCol * TN + resIdxN;
      C[row * N + col] = alpha * threadResults[resIdxM * TN + resIdxN] +
                          beta * C[row * N + col];
    }
  }
}

void runKernel5_2DBlocktiling(int M, int N, int K, float alpha,
                               const float *A, const float *B, float beta,
                               float *C) {
  const uint BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
  dim3 blockDim((BM * BN) / (TM * TN));
  sgemm_2d_blocktiling<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
