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

__global__ void sm90_baseline_kernel(const float *input, float *output, int n,
                                     int iterations) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  float value = input[tid];
#pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    float scale = ((tid + i) & 1) ? 0.9997f : 1.0003f;
    value = fmaf(value, scale, 0.0002f * static_cast<float>((i & 3) + 1));
  }
  output[tid] = value;
}

static float input_value(int i) {
  return static_cast<float>((i % 37) - 18) * 0.03125f;
}

int main(int argc, char **argv) {
  int n = argc > 1 ? std::atoi(argv[1]) : 4096;
  int iterations = argc > 2 ? std::atoi(argv[2]) : 32;
  if (n <= 0 || iterations <= 0) {
    std::fprintf(stderr, "usage: %s <elements> <iterations>\n", argv[0]);
    return 2;
  }

  std::vector<float> input(n), output(n), expected(n);
  for (int i = 0; i < n; ++i) {
    input[i] = input_value(i);
    float value = input[i];
    for (int j = 0; j < iterations; ++j) {
      float scale = ((i + j) & 1) ? 0.9997f : 1.0003f;
      value = std::fma(value, scale,
                       0.0002f * static_cast<float>((j & 3) + 1));
    }
    expected[i] = value;
  }

  float *device_input = nullptr;
  float *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  sm90_baseline_kernel<<<grid, block>>>(device_input, device_output, n,
                                         iterations);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  double max_error = 0.0;
  double checksum = 0.0;
  for (int i = 0; i < n; ++i) {
    max_error = std::max(max_error, std::abs(static_cast<double>(
                                                  output[i] - expected[i])));
    checksum += static_cast<double>(output[i]) * ((i % 11) + 1);
  }
  bool pass = max_error <= 2.0e-4;
  std::printf("RESULT test=sm90_baseline size=%d aux=%d checksum=%.12e "
              "max_error=%.12e kernel_ms=%.6f status=%s\n",
              n, iterations, checksum, max_error, kernel_ms,
              pass ? "PASS" : "FAIL");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  return pass ? 0 : 1;
}
