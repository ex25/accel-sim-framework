// Local GPU oracle for approximate arithmetic, including non-unit numerators.
#include "../../stage-1/host.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <iterator>
#include <vector>

__global__ void probe(const float *input, float *output, unsigned count) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  float a = input[2 * i], b = input[2 * i + 1];
  float *out = output + 9 * i;
  asm("ex2.approx.f32 %0,%1;" : "=f"(out[0]) : "f"(a));
  asm("ex2.approx.ftz.f32 %0,%1;" : "=f"(out[1]) : "f"(a));
  asm("rcp.approx.f32 %0,%1;" : "=f"(out[2]) : "f"(b));
  asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(out[3]) : "f"(b));
  asm("div.full.f32 %0,%1,%2;" : "=f"(out[4]) : "f"(a), "f"(b));
  asm("div.full.ftz.f32 %0,%1,%2;" : "=f"(out[5]) : "f"(a), "f"(b));
  asm("div.approx.f32 %0,%1,%2;" : "=f"(out[6]) : "f"(a), "f"(b));
  asm("div.approx.ftz.f32 %0,%1,%2;" : "=f"(out[7]) : "f"(a), "f"(b));
  asm("div.rn.f32 %0,%1,%2;" : "=f"(out[8]) : "f"(a), "f"(b));
}

int main(int argc, char **argv) {
  auto bytes = read(argv[1]);
  unsigned count = bytes.size() / 8;
  float *input, *output;
  CHECK(cudaMalloc(&input, bytes.size()));
  CHECK(cudaMalloc(&output, count * 9 * sizeof(float)));
  CHECK(cudaMemcpy(input, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
  probe<<<(count + 127) / 128, 128>>>(input, output, count);
  CHECK(cudaDeviceSynchronize());
  std::vector<float> result(count * 9);
  CHECK(cudaMemcpy(result.data(), output, result.size() * sizeof(float),
                   cudaMemcpyDeviceToHost));
  std::ofstream file(argv[2], std::ios::binary);
  file.write(reinterpret_cast<const char *>(result.data()),
             result.size() * sizeof(float));
  cudaDeviceProp device;
  CHECK(cudaGetDeviceProperties(&device, 0));
  printf("PASS device=%s inputs=%u\n", device.name, count);
}
