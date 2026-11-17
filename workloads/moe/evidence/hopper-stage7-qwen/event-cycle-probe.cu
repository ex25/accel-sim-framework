#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
__global__ void add_one(int *p) { if (threadIdx.x == 0) ++*p; }
int main(int argc, char **) {
  int *p, value = 0;
  assert(cudaMalloc(&p, sizeof(int)) == cudaSuccess);
  assert(cudaMemcpy(p, &value, sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
  add_one<<<1,32>>>(p);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  cudaStream_t stream = nullptr;
  if (argc > 1) assert(cudaStreamCreate(&stream) == cudaSuccess);
  cudaEvent_t begin, end;
  assert(cudaEventCreate(&begin) == cudaSuccess);
  assert(cudaEventCreate(&end) == cudaSuccess);
  for (int round = 0; round < 3; ++round) {
    assert(cudaEventRecord(begin, stream) == cudaSuccess);
    add_one<<<1,32,0,stream>>>(p);
    add_one<<<1,32,0,stream>>>(p);
    assert(cudaEventRecord(end, stream) == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    float ms;
    assert(cudaEventElapsedTime(&ms, begin, end) == cudaSuccess);
  }
  assert(cudaMemcpy(&value, p, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
  assert(value == 7);
  std::printf("EVENT_SMOKE value=%d stream=%s\n", value, argc > 1 ? "explicit" : "default");
  assert(cudaEventDestroy(begin) == cudaSuccess);
  assert(cudaEventDestroy(end) == cudaSuccess);
  if (stream) assert(cudaStreamDestroy(stream) == cudaSuccess);
  assert(cudaFree(p) == cudaSuccess);
}
