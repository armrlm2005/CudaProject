cmake_minimum_required(VERSION 3.18)
project(sgemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Adjust this to match your GPU's compute capability if needed.
# 75 = Turing (RTX 20xx, T4), 80/86 = Ampere (A100, RTX 30xx),
# 89 = Ada (RTX 40xx), 90 = Hopper (H100). 75;80;86;89 covers most
# Colab/Kaggle/consumer GPUs you're likely to test on.
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

find_package(CUDAToolkit REQUIRED)

add_executable(sgemm
  src/main.cu
  src/runner.cu
  src/kernels/01_naive.cu
  src/kernels/02_coalescing.cu
  src/kernels/03_smem_tiling.cu
  src/kernels/04_1d_blocktiling.cu
  src/kernels/05_2d_blocktiling.cu
  src/kernels/06_vectorized.cu
  src/kernels/07_warptiling.cu
)

target_include_directories(sgemm PRIVATE include)
target_link_libraries(sgemm PRIVATE CUDA::cublas CUDA::cudart)

target_compile_options(sgemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)
