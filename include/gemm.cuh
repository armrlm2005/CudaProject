#pragma once

#include <cstdint>

// Ceiling division: how many tiles of size `b` are needed to cover `a`.
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------------
// Each kernel computes: C = alpha * (A @ B) + beta * C
// A is M x K, B is K x N, C is M x N. All matrices are row-major in
// global memory (C-style layout: A[row * K + col]).
// ---------------------------------------------------------------------

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

void runKernel1Naive(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C);

void runKernel2Coalescing(int M, int N, int K, float alpha, const float *A,
                           const float *B, float beta, float *C);

void runKernel3SmemTiling(int M, int N, int K, float alpha, const float *A,
                           const float *B, float beta, float *C);

void runKernel4_1DBlocktiling(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C);

void runKernel5_2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C);

void runKernel6Vectorized(int M, int N, int K, float alpha, const float *A,
                           const float *B, float beta, float *C);

void runKernel7Warptiling(int M, int N, int K, float alpha, const float *A,
                           const float *B, float beta, float *C);

// Dispatch helper used by main.cu / the benchmark harness.
// kernelNum: 1-7 selects a custom kernel, 0 selects cuBLAS as reference.
void runKernel(int kernelNum, int M, int N, int K, float alpha,
               const float *A, const float *B, float beta, float *C,
               void *cublasHandle);
