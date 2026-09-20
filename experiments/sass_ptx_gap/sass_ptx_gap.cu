#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err__));                                  \
      std::exit(2);                                                             \
    }                                                                          \
  } while (0)

__global__ void fma_chain_kernel(const float *input, float *output, int n,
                                 int iterations) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  float x = input[tid];
#pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    x = fmaf(x, 1.0001f, 0.0001f * static_cast<float>((i & 7) + 1));
  }
  output[tid] = x;
}

__global__ void vec_add_kernel(const float *a, const float *b, float *c, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) c[tid] = a[tid] + b[tid];
}

__global__ void global_stride_kernel(const float *input, float *output, int n,
                                     int stride, int iterations) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  int span = n * stride;
  int index = (tid * stride) % span;
  float sum = 0.0f;
#pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    sum += input[(index + i * stride) % span];
  }
  output[tid] = sum;
}

__global__ void shared_reduce_kernel(const float *input, float *output) {
  __shared__ float scratch[256];
  int tid = threadIdx.x;
  scratch[tid] = input[tid];
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) scratch[tid] += scratch[tid + offset];
    __syncthreads();
  }
  if (tid == 0) output[0] = scratch[0];
}

__global__ void branch_diverge_kernel(const float *input, float *output, int n,
                                      int iterations) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  float value = input[tid];
#pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    if (((tid + i) & 1) == 0) {
      value = value * 1.0002f + 0.0003f;
    } else {
      value = value * 0.9998f - 0.0002f;
    }
  }
  output[tid] = value;
}

__global__ void atomic_hotspot_kernel(int *bins, int n, int bin_count,
                                      int iterations) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  int bin = (tid * 17) % bin_count;
#pragma unroll 1
  for (int i = 0; i < iterations; ++i) atomicAdd(&bins[bin], 1);
}

__global__ void gemv_kernel(const float *matrix, const float *vector,
                            float *output, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  float sum = 0.0f;
#pragma unroll 1
  for (int col = 0; col < n; ++col) {
    sum = fmaf(matrix[row * n + col], vector[col], sum);
  }
  output[row] = sum;
}

__global__ void gemm_kernel(const float *a, const float *b, float *c, int n) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= n || col >= n) return;
  float sum = 0.0f;
#pragma unroll 1
  for (int k = 0; k < n; ++k) {
    sum = fmaf(a[row * n + k], b[k * n + col], sum);
  }
  c[row * n + col] = sum;
}

static float value_for_index(size_t i) {
  return static_cast<float>(static_cast<int>(i % 29) - 14) * 0.03125f;
}

static double checksum(const std::vector<float> &values) {
  double result = 0.0;
  for (size_t i = 0; i < values.size(); ++i)
    result += static_cast<double>(values[i]) * static_cast<double>((i % 13) + 1);
  return result;
}

static void print_result(const std::string &test, int size, int aux,
                         double sum, double max_error, float kernel_ms,
                         bool pass) {
  std::printf(
      "RESULT test=%s size=%d aux=%d checksum=%.12e max_error=%.12e "
      "kernel_ms=%.6f status=%s\n",
      test.c_str(), size, aux, sum, max_error, kernel_ms,
      pass ? "PASS" : "FAIL");
}

template <typename Launch>
static float time_kernel(Launch launch) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms;
}

static int run_float_vector_case(const std::string &test, int size, int aux) {
  int n = size;
  std::vector<float> a(n), b(n), output(n), expected(n);
  for (int i = 0; i < n; ++i) {
    a[i] = value_for_index(i);
    b[i] = value_for_index(i + 7);
  }

  float *d_a = nullptr, *d_b = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  float elapsed = 0.0f;
  if (test == "fma_chain") {
    elapsed = time_kernel([&] {
      fma_chain_kernel<<<grid, block>>>(d_a, d_output, n, aux);
    });
    for (int tid = 0; tid < n; ++tid) {
      float x = a[tid];
      for (int i = 0; i < aux; ++i)
        x = std::fma(x, 1.0001f,
                     0.0001f * static_cast<float>((i & 7) + 1));
      expected[tid] = x;
    }
  } else if (test == "vec_add") {
    elapsed = time_kernel(
        [&] { vec_add_kernel<<<grid, block>>>(d_a, d_b, d_output, n); });
    for (int i = 0; i < n; ++i) expected[i] = a[i] + b[i];
  } else if (test == "branch_diverge") {
    elapsed = time_kernel([&] {
      branch_diverge_kernel<<<grid, block>>>(d_a, d_output, n, aux);
    });
    for (int tid = 0; tid < n; ++tid) {
      float value = a[tid];
      for (int i = 0; i < aux; ++i) {
        if (((tid + i) & 1) == 0)
          value = value * 1.0002f + 0.0003f;
        else
          value = value * 0.9998f - 0.0002f;
      }
      expected[tid] = value;
    }
  } else {
    std::fprintf(stderr, "unknown vector test: %s\n", test.c_str());
    return 2;
  }

  CUDA_CHECK(cudaMemcpy(output.data(), d_output, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double max_error = 0.0;
  for (int i = 0; i < n; ++i)
    max_error = std::max(max_error,
                         std::abs(static_cast<double>(output[i] - expected[i])));
  bool pass = max_error <= 2.0e-4;
  print_result(test, size, aux, checksum(output), max_error, elapsed, pass);
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_output));
  return pass ? 0 : 1;
}

static int run_stride(int n, int stride, int iterations) {
  int span = n * stride;
  std::vector<float> input(span), output(n), expected(n);
  for (int i = 0; i < span; ++i) input[i] = value_for_index(i);
  float *d_input = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, span * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), span * sizeof(float),
                        cudaMemcpyHostToDevice));
  dim3 block(256), grid((n + block.x - 1) / block.x);
  float elapsed = time_kernel([&] {
    global_stride_kernel<<<grid, block>>>(d_input, d_output, n, stride,
                                          iterations);
  });
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (int tid = 0; tid < n; ++tid) {
    int index = (tid * stride) % span;
    float sum = 0.0f;
    for (int i = 0; i < iterations; ++i)
      sum += input[(index + i * stride) % span];
    expected[tid] = sum;
  }
  double max_error = 0.0;
  for (int i = 0; i < n; ++i)
    max_error = std::max(max_error,
                         std::abs(static_cast<double>(output[i] - expected[i])));
  bool pass = max_error <= 2.0e-4;
  print_result("global_stride", n, stride, checksum(output), max_error, elapsed,
               pass);
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return pass ? 0 : 1;
}

static int run_shared_reduce() {
  constexpr int n = 256;
  std::vector<float> input(n);
  for (int i = 0; i < n; ++i) input[i] = value_for_index(i);
  float expected = 0.0f;
  for (float value : input) expected += value;
  float output = 0.0f;
  float *d_input = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));
  float elapsed = time_kernel(
      [&] { shared_reduce_kernel<<<1, n>>>(d_input, d_output); });
  CUDA_CHECK(cudaMemcpy(&output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
  double error = std::abs(static_cast<double>(output - expected));
  bool pass = error <= 2.0e-4;
  std::vector<float> values{output};
  print_result("shared_reduce", n, 0, checksum(values), error, elapsed, pass);
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return pass ? 0 : 1;
}

static int run_atomic(int n, int bin_count, int iterations) {
  std::vector<int> output(bin_count, 0);
  int *d_bins = nullptr;
  CUDA_CHECK(cudaMalloc(&d_bins, bin_count * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_bins, 0, bin_count * sizeof(int)));
  dim3 block(256), grid((n + block.x - 1) / block.x);
  float elapsed = time_kernel([&] {
    atomic_hotspot_kernel<<<grid, block>>>(d_bins, n, bin_count, iterations);
  });
  CUDA_CHECK(cudaMemcpy(output.data(), d_bins, bin_count * sizeof(int),
                        cudaMemcpyDeviceToHost));
  long long total = 0;
  for (int value : output) total += value;
  long long expected = static_cast<long long>(n) * iterations;
  bool pass = total == expected;
  print_result("atomic_hotspot", n, bin_count, static_cast<double>(total),
               static_cast<double>(std::llabs(total - expected)), elapsed, pass);
  CUDA_CHECK(cudaFree(d_bins));
  return pass ? 0 : 1;
}

static int run_gemv(int n) {
  std::vector<float> matrix(static_cast<size_t>(n) * n), vector(n), output(n),
      expected(n);
  for (size_t i = 0; i < matrix.size(); ++i) matrix[i] = value_for_index(i);
  for (int i = 0; i < n; ++i) vector[i] = value_for_index(i + 11);
  float *d_matrix = nullptr, *d_vector = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_matrix, matrix.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_vector, vector.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_matrix, matrix.data(), matrix.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vector, vector.data(), vector.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  dim3 block(128), grid((n + block.x - 1) / block.x);
  float elapsed = time_kernel(
      [&] { gemv_kernel<<<grid, block>>>(d_matrix, d_vector, d_output, n); });
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (int row = 0; row < n; ++row) {
    float sum = 0.0f;
    for (int col = 0; col < n; ++col)
      sum = std::fma(matrix[row * n + col], vector[col], sum);
    expected[row] = sum;
  }
  double max_error = 0.0;
  for (int i = 0; i < n; ++i)
    max_error = std::max(max_error,
                         std::abs(static_cast<double>(output[i] - expected[i])));
  bool pass = max_error <= 5.0e-4;
  print_result("gemv", n, 0, checksum(output), max_error, elapsed, pass);
  CUDA_CHECK(cudaFree(d_matrix));
  CUDA_CHECK(cudaFree(d_vector));
  CUDA_CHECK(cudaFree(d_output));
  return pass ? 0 : 1;
}

static int run_gemm(int n) {
  size_t elements = static_cast<size_t>(n) * n;
  std::vector<float> a(elements), b(elements), output(elements), expected(elements);
  for (size_t i = 0; i < elements; ++i) {
    a[i] = value_for_index(i);
    b[i] = value_for_index(i + 5);
  }
  float *d_a = nullptr, *d_b = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  dim3 block(16, 16), grid((n + 15) / 16, (n + 15) / 16);
  float elapsed = time_kernel(
      [&] { gemm_kernel<<<grid, block>>>(d_a, d_b, d_output, n); });
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, elements * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int k = 0; k < n; ++k)
        sum = std::fma(a[row * n + k], b[k * n + col], sum);
      expected[row * n + col] = sum;
    }
  }
  double max_error = 0.0;
  for (size_t i = 0; i < elements; ++i)
    max_error = std::max(max_error,
                         std::abs(static_cast<double>(output[i] - expected[i])));
  bool pass = max_error <= 1.0e-3;
  print_result("gemm", n, 0, checksum(output), max_error, elapsed, pass);
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_output));
  return pass ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <fma_chain|vec_add|global_stride|shared_reduce|"
                 "branch_diverge|atomic_hotspot|gemv|gemm> [size] [aux]\n",
                 argv[0]);
    return 2;
  }
  std::string test = argv[1];
  int size = argc > 2 ? std::atoi(argv[2]) : 4096;
  int aux = argc > 3 ? std::atoi(argv[3]) : 64;
  if (size <= 0 || aux < 0) return 2;

  if (test == "fma_chain" || test == "vec_add" ||
      test == "branch_diverge")
    return run_float_vector_case(test, size, aux);
  if (test == "global_stride") return run_stride(size, aux, 16);
  if (test == "shared_reduce") return run_shared_reduce();
  if (test == "atomic_hotspot") return run_atomic(size, aux, 4);
  if (test == "gemv") return run_gemv(size);
  if (test == "gemm") return run_gemm(size);

  std::fprintf(stderr, "unknown test: %s\n", test.c_str());
  return 2;
}
