#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(call)                                                         \
  do {                                                                      \
    cudaError_t error = (call);                                             \
    if (error != cudaSuccess) {                                             \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                 \
              cudaGetErrorString(error));                                   \
      return 2;                                                             \
    }                                                                       \
  } while (0)

__global__ void store_v2(uint32_t *data) {
  unsigned lane = threadIdx.x;
  uint32_t *address = data + 2 * lane;
  uint32_t value = 101 + 2 * lane;
  asm volatile("st.global.v2.b32 [%0], {%1, %2};"
               : : "l"(address), "r"(value), "r"(value + 1) : "memory");
}

__global__ void store_v4(uint32_t *data) {
  unsigned lane = threadIdx.x;
  uint32_t *address = data + 4 * lane;
  uint32_t value = 101 + 4 * lane;
  asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
               : : "l"(address), "r"(value), "r"(value + 1),
                   "r"(value + 2), "r"(value + 3) : "memory");
}

__global__ void load_v2(const uint32_t *data, uint32_t *output) {
  unsigned lane = threadIdx.x;
  const uint32_t *address = data + 2 * lane;
  uint32_t a, b;
  asm volatile("ld.global.v2.b32 {%0, %1}, [%2];"
               : "=r"(a), "=r"(b) : "l"(address));
  output[2 * lane] = a;
  output[2 * lane + 1] = b;
}

__global__ void load_v4(const uint32_t *data, uint32_t *output) {
  unsigned lane = threadIdx.x;
  const uint32_t *address = data + 4 * lane;
  uint32_t a, b, c, d;
  asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(address));
  output[4 * lane] = a;
  output[4 * lane + 1] = b;
  output[4 * lane + 2] = c;
  output[4 * lane + 3] = d;
}

int main(int argc, char **argv) {
  if (argc != 2) return 3;
  const bool store_first = strncmp(argv[1], "store_read_", 11) == 0;
  const bool aligned = strncmp(argv[1], "aligned_", 8) == 0;
  const bool v2 = strstr(argv[1], "v2") != nullptr;
  const bool v4 = strstr(argv[1], "v4") != nullptr;
  if ((!store_first && !aligned) || (v2 == v4)) return 3;
  const unsigned width = v2 ? 2 : 4;
  const unsigned count = 32 * width;
  std::vector<uint32_t> input(count), expected(count), actual(count);
  for (unsigned i = 0; i < count; ++i) {
    input[i] = i + 101;
    expected[i] = i + 101;
  }
  uint32_t *data = nullptr, *output = nullptr;
  CHECK(cudaMalloc(&data, count * sizeof(uint32_t)));
  CHECK(cudaMalloc(&output, count * sizeof(uint32_t)));
  if (store_first) {
    CHECK(cudaMemset(data, 0, count * sizeof(uint32_t)));
    if (v2)
      store_v2<<<1, 32>>>(data);
    else
      store_v4<<<1, 32>>>(data);
    CHECK(cudaDeviceSynchronize());
  } else {
    CHECK(cudaMemcpy(data, input.data(), count * sizeof(uint32_t),
                     cudaMemcpyHostToDevice));
  }
  if (v2)
    load_v2<<<1, 32>>>(data, output);
  else
    load_v4<<<1, 32>>>(data, output);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(actual.data(), output, count * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  unsigned errors = 0;
  for (unsigned i = 0; i < count; ++i)
    if (actual[i] != expected[i]) ++errors;
  printf("%s %s elements=%u errors=%u\n", errors ? "FAIL" : "PASS",
         argv[1], count, errors);
  CHECK(cudaFree(data));
  CHECK(cudaFree(output));
  return errors ? 2 : 0;
}
