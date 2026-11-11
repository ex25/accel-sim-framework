// A cold, read-only sector probe for separating demand and L2-fabric traffic.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA(call)                                                             \
  do {                                                                         \
    if ((call) != cudaSuccess)                                                 \
      std::exit(1);                                                            \
  } while (0)

__global__ void l2_probe(const unsigned *input, unsigned *output) {
  unsigned sum = 0;
  for (unsigned i = 0; i < 128; ++i) {
    unsigned value;
    const unsigned *p = input + threadIdx.x * 16384 + i * 32;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(p));
    sum += value;
  }
  output[threadIdx.x] = sum;
}

int main() {
  unsigned *input, *output, actual[32];
  CUDA(cudaMalloc(&input, 4 * 1024 * 1024));
  CUDA(cudaMalloc(&output, sizeof(actual)));
  CUDA(cudaMemset(input, 1, 4 * 1024 * 1024));
  l2_probe<<<1, 32>>>(input, output);
  CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost));
  for (unsigned value : actual)
    if (value != 0x80808080u)
      return 2;
  puts("PASS l2_probe: 4096 distinct demand sectors, 131072 bytes, 32 outputs");
}
