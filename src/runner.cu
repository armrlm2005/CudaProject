#include "gemm.cuh"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

void runKernel(int kernelNum, int M, int N, int K, float alpha,
               const float *A, const float *B, float beta, float *C,
               void *cublasHandlePtr) {
  switch (kernelNum) {
  case 0: {
    // Reference: cuBLAS SGEMM. Note cuBLAS expects column-major
    // matrices; since our A/B/C are row-major, we exploit the identity
    // (A*B)^T = B^T * A^T to call cuBLAS with swapped operands/dims,
    // which gives us a row-major result without needing to transpose
    // any data ourselves.
    cublasHandle_t handle = *reinterpret_cast<cublasHandle_t *>(cublasHandlePtr);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
                &beta, C, N);
    break;
  }
  case 1:
    runKernel1Naive(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    runKernel2Coalescing(M, N, K, alpha, A, B, beta, C);
    break;
  case 3:
    runKernel3SmemTiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 4:
    runKernel4_1DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 5:
    runKernel5_2DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 6:
    runKernel6Vectorized(M, N, K, alpha, A, B, beta, C);
    break;
  case 7:
    runKernel7Warptiling(M, N, K, alpha, A, B, beta, C);
    break;
  default:
    fprintf(stderr, "Unknown kernel number: %d\n", kernelNum);
    exit(EXIT_FAILURE);
  }
}
