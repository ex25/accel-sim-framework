#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err__));                                 \
      return 2;                                                                 \
    }                                                                           \
  } while (0)

__global__ void cais_add_kernel(const unsigned *lhs, const unsigned *rhs,
                                unsigned *output, int n, int iterations) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  unsigned value = lhs[tid];
  unsigned addend = rhs[tid];
#pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    // The PTX generator emits add.u32 here. The focused test rewrites only
    // this opcode to cais.add.u32 before GPGPU-Sim parses the file.
    asm volatile("add.u32 %0, %1, %2;"
                 : "=r"(value)
                 : "r"(value), "r"(addend));
  }
  output[tid] = value;
}

static unsigned input_value(int i) {
  return static_cast<unsigned>(0x10203040u + i * 17u);
}

int main(int argc, char **argv) {
  int n = argc > 1 ? std::atoi(argv[1]) : 4096;
  int iterations = argc > 2 ? std::atoi(argv[2]) : 32;
  if (n <= 0 || iterations <= 0) {
    std::fprintf(stderr, "usage: %s <elements> <iterations>\n", argv[0]);
    return 2;
  }

  std::vector<unsigned> lhs(n), rhs(n), output(n), expected(n);
  for (int i = 0; i < n; ++i) {
    lhs[i] = input_value(i);
    rhs[i] = static_cast<unsigned>(0x01020304u + i * 3u);
    expected[i] = lhs[i] + static_cast<unsigned>(iterations) * rhs[i];
  }

  unsigned *device_lhs = nullptr;
  unsigned *device_rhs = nullptr;
  unsigned *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_lhs, n * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&device_rhs, n * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(unsigned)));
  CUDA_CHECK(cudaMemcpy(device_lhs, lhs.data(), n * sizeof(unsigned),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_rhs, rhs.data(), n * sizeof(unsigned),
                        cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  cais_add_kernel<<<grid, block>>>(device_lhs, device_rhs, device_output, n,
                                    iterations);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, n * sizeof(unsigned),
                        cudaMemcpyDeviceToHost));

  unsigned max_error = 0;
  unsigned long long checksum = 0;
  for (int i = 0; i < n; ++i) {
    max_error = std::max(max_error, output[i] > expected[i]
                                       ? output[i] - expected[i]
                                       : expected[i] - output[i]);
    checksum += static_cast<unsigned long long>(output[i]) * ((i % 11) + 1);
  }
  bool pass = max_error == 0;
  std::printf("RESULT test=cais_add_u32 size=%d aux=%d checksum=%llu "
              "max_error=%u kernel_ms=%.6f status=%s\n",
              n, iterations, checksum, max_error, kernel_ms,
              pass ? "PASS" : "FAIL");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_lhs));
  CUDA_CHECK(cudaFree(device_rhs));
  CUDA_CHECK(cudaFree(device_output));
  return pass ? 0 : 1;
}
