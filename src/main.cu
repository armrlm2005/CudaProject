/*
 * main.cu
 * --------
 * Usage:
 *   ./sgemm <kernel_num> <size> [iters]
 *
 *   kernel_num : 0 = cuBLAS, 1-7 = the custom kernels in src/kernels/
 *   size       : matrix dimension (square, size x size x size)
 *   iters      : number of timed repetitions (default 20)
 *
 * What this does:
 *   1. Allocates random M x K and K x N matrices on host + device.
 *   2. Runs the chosen kernel once and compares its output against
 *      cuBLAS's result (relative error check) to confirm correctness.
 *   3. Times `iters` repetitions of the kernel and reports GFLOPS,
 *      along with the equivalent cuBLAS GFLOPS for direct comparison.
 */

#include "gemm.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err));                                      \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

static void fillRandom(std::vector<float> &v, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &x : v)
    x = dist(rng);
}

// Returns true if, for every element, the absolute difference is within
// (atol + rtol * |ref|). This is the standard combined absolute+relative
// tolerance check (same idea as numpy.allclose) - pure relative error
// blows up for elements where the reference value happens to be close to
// zero, even when the absolute numerical difference is just ordinary
// floating-point accumulation noise from summing in a different order
// than cuBLAS does internally.
static bool verify(const std::vector<float> &ref, const std::vector<float> &got,
                    float rtol = 1e-2f, float atol = 1e-3f) {
  float maxAbsErr = 0.0f;
  size_t worstIdx = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    float absErr = std::fabs(ref[i] - got[i]);
    float allowed = atol + rtol * std::fabs(ref[i]);
    if (absErr > allowed && absErr > maxAbsErr) {
      maxAbsErr = absErr;
      worstIdx = i;
    }
  }
  if (maxAbsErr > 0.0f) {
    printf("  largest out-of-tolerance abs error: %f at index %zu (ref=%f, "
           "got=%f)\n",
           maxAbsErr, worstIdx, ref[worstIdx], got[worstIdx]);
    return false;
  }
  printf("  all elements within tolerance (atol=%f, rtol=%f)\n", atol, rtol);
  return true;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <kernel_num 0-7> <size> [iters]\n", argv[0]);
    return EXIT_FAILURE;
  }

  int kernelNum = std::atoi(argv[1]);
  int size = std::atoi(argv[2]);
  int iters = argc > 3 ? std::atoi(argv[3]) : 20;

  int M = size, N = size, K = size;
  float alpha = 1.0f, beta = 0.0f;

  std::mt19937 rng(42);
  std::vector<float> hA(M * K), hB(K * N), hC(M * N, 0.0f);
  fillRandom(hA, rng);
  fillRandom(hB, rng);

  float *dA, *dB, *dC, *dCRef;
  CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dCRef, M * N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, M * N * sizeof(float)));
  CUDA_CHECK(cudaMemset(dCRef, 0, M * N * sizeof(float)));

  cublasHandle_t handle;
  cublasCreate(&handle);

  // --- Correctness check (skip if benchmarking cuBLAS against itself) ---
  if (kernelNum != 0) {
    runKernel(0, M, N, K, alpha, dA, dB, beta, dCRef, &handle);
    runKernel(kernelNum, M, N, K, alpha, dA, dB, beta, dC, &handle);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hCRef(M * N), hCGot(M * N);
    CUDA_CHECK(cudaMemcpy(hCRef.data(), dCRef, M * N * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hCGot.data(), dC, M * N * sizeof(float),
                           cudaMemcpyDeviceToHost));

    printf("Kernel %d correctness check (size=%d):\n", kernelNum, size);
    printf("  first 5 ref  values: %f %f %f %f %f\n", hCRef[0], hCRef[1],
           hCRef[2], hCRef[3], hCRef[4]);
    printf("  first 5 got  values: %f %f %f %f %f\n", hCGot[0], hCGot[1],
           hCGot[2], hCGot[3], hCGot[4]);
    bool ok = verify(hCRef, hCGot);
    printf("  result: %s\n", ok ? "PASS" : "FAIL");
    if (!ok) {
      fprintf(stderr,
              "Correctness check failed - skipping benchmark. Check your "
              "tile-size template parameters against the matrix size.\n");
      return EXIT_FAILURE;
    }
  }

  // --- Reset C and warm up ---
  CUDA_CHECK(cudaMemset(dC, 0, M * N * sizeof(float)));
  runKernel(kernelNum, M, N, K, alpha, dA, dB, beta, dC, &handle);
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- Timed benchmark ---
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < iters; ++i) {
    runKernel(kernelNum, M, N, K, alpha, dA, dB, beta, dC, &handle);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  float avgMs = ms / iters;

  // FLOPs for an M x N x K GEMM with the C += beta*C term: 2*M*N*K (mul+add)
  double flops = 2.0 * double(M) * double(N) * double(K);
  double gflops = (flops / 1e9) / (avgMs / 1000.0);

  printf("Kernel %d | size=%d | avg time=%.4f ms | %.1f GFLOPS\n", kernelNum,
         size, avgMs, gflops);

  cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dCRef);

  return EXIT_SUCCESS;
}
