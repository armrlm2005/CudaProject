/*
 * Kernel 7: Warp Tiling
 * --------------------------
 * So far we've organized work at the block level (BM x BN tiles) and
 * the thread level (TM x TN results per thread), but ignored the warp
 * as a unit. Warps matter because:
 *
 *   - Threads within a warp execute in lockstep, so organizing their
 *     memory accesses *as a group* (not just per-thread) lets us
 *     control shared memory bank-conflict patterns more precisely.
 *   - It gives us an extra tiling dimension to tune occupancy with,
 *     independent of the block-level and thread-level tile sizes.
 *
 * We introduce a warp tile of size WM x WN sitting between the block
 * tile (BM x BN) and the thread tile (TM x TN). Each warp is
 * responsible for one WM x WN sub-region of the block's output tile,
 * and within that, each thread still computes its own TM x TN
 * sub-tile, exactly as in kernel 6. To keep registers/SMEM traffic
 * manageable for large warp tiles, a warp may iterate over its WM x WN
 * region in WMITER x WNITER passes of a smaller WSUBM x WSUBN size.
 *
 * This kernel is structurally the most complex in the series - it's
 * the same ideas as kernels 5/6 (block tiling, register caching,
 * vectorized access) but with one more level of the hierarchy made
 * explicit and tunable.
 */

#include "gemm.cuh"

namespace wt {

template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                              float *As, float *Bs, int innerRowA,
                              int innerColA, int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }
  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

template <const int BM, const int BN, const int BK, const int WM,
          const int WN, const int WMITER, const int WNITER, const int WSUBM,
          const int WSUBN, const int TM, const int TN>
__device__ void processFromSmem(float *regM, float *regN,
                                 float *threadResults, const float *As,
                                 const float *Bs, const uint warpRow,
                                 const uint warpCol, const uint threadRowInWarp,
                                 const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[dotIdx * BM + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[dotIdx * BN + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) *
                              (WNITER * TN) +
                          wSubColIdx * TN + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace wt

template <const int BM, const int BN, const int BK, const int WM,
          const int WN, const int WNITER, const int TM, const int TN,
          const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemm_warptiling(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Which warp this thread belongs to, and where that warp sits inside
  // the block's BM x BN tile.
  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;

  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  float regM[WMITER * TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();

    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                         TN>(regM, regN, threadResults, As, Bs, warpRow,
                             warpCol, threadRowInWarp, threadColInWarp);

    A += BK;
    B += BK * N;
    __syncthreads();
  }

  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          int row = threadRowInWarp * TM + resIdxM;
          int col = threadColInWarp * TN + resIdxN;

          float4 existing =
              reinterpret_cast<float4 *>(&C_interim[row * N + col])[0];
          const uint i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                         wSubColIdx * TN + resIdxN;
          float4 out;
          out.x = alpha * threadResults[i + 0] + beta * existing.x;
          out.y = alpha * threadResults[i + 1] + beta * existing.y;
          out.z = alpha * threadResults[i + 2] + beta * existing.z;
          out.w = alpha * threadResults[i + 3] + beta * existing.w;
          reinterpret_cast<float4 *>(&C_interim[row * N + col])[0] = out;
        }
      }
    }
  }
}

void runKernel7Warptiling(int M, int N, int K, float alpha, const float *A,
                           const float *B, float beta, float *C) {
  // Tunable tile sizes - these defaults are a reasonable starting
  // point for matrices in the 1024-4096 range on Ampere-class GPUs.
  // Re-tune (or autotune) for your specific GPU if chasing maximum
  // throughput.
  const uint NUM_THREADS = 128;
  const uint BN = 128, BM = 128, BK = 16;
  const uint WN = 64, WM = 64;
  const uint WNITER = 4;
  const uint TN = 4, TM = 8;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

  sgemm_warptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
