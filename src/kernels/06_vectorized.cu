/*
 * Kernel 6: Vectorized Memory Access
 * --------------------------------------
 * Kernel 5 is much faster, but still spends a lot of cycles stalled on
 * memory pipeline congestion. Two changes attack this directly:
 *
 * 1. Vectorized loads: instead of issuing four separate 32-bit loads
 *    for four consecutive floats, we cast pointers to float4 and issue
 *    one 128-bit load instruction. This cuts the number of load
 *    instructions (and associated instruction-issue overhead) by 4x
 *    for both the GMEM->SMEM staging step and reading inputs from
 *    global memory.
 *
 * 2. Transposing As in shared memory: in kernel 5, reading a column of
 *    As (As[(threadRow*TM+i)*BK + dotIdx]) means consecutive `i` values
 *    are BK floats apart in memory - not vectorizable. By storing As
 *    transposed (swap the role of rows/cols when writing it into SMEM),
 *    the same logical access becomes contiguous, so we can use a
 *    float4 load there too.
 *
 * Both changes preserve every previous optimization (block tiling,
 * register caching, 2D thread tiling) - they're purely about how
 * existing memory traffic gets issued, not what data is moved.
 */

#include "gemm.cuh"

template <const int BM, const int BN, const int BK, const int TM,
          const int TN>
__global__ void sgemm_vectorized(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta,
                                  float *C) {
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // Indices for vectorized (float4) cooperative loading. Each thread
  // now loads 4 floats at once, so the index arithmetic divides by 4.
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Load 4 floats from A at once, then scatter them into the
    // *transposed* As tile (row/col swapped relative to kernel 5).
    float4 tmp = reinterpret_cast<const float4 *>(
        &A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    // B doesn't need transposing - just load it vectorized as-is.
    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[innerRowB * N + innerColB * 4])[0];

    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        // As is transposed, so this is now a contiguous read.
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
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

  // Write results back out 4-at-a-time, folding alpha/beta into the
  // vectorized store.
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      int row = threadRow * TM + resIdxM;
      int col = threadCol * TN + resIdxN;

      float4 existing = reinterpret_cast<float4 *>(&C[row * N + col])[0];
      float4 out;
      out.x = alpha * threadResults[resIdxM * TN + resIdxN + 0] + beta * existing.x;
      out.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * existing.y;
      out.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * existing.z;
      out.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * existing.w;
      reinterpret_cast<float4 *>(&C[row * N + col])[0] = out;
    }
  }
}

void runKernel6Vectorized(int M, int N, int K, float alpha, const float *A,
                           const float *B, float beta, float *C) {
  const uint BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
  dim3 blockDim((BM * BN) / (TM * TN));
  sgemm_vectorized<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
