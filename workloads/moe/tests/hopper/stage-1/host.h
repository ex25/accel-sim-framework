#pragma once
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#define CHECK(call)                               \
  do {                                            \
    auto e = (call);                              \
    if (e != cudaSuccess) {                       \
      fprintf(stderr, "%s: %d\n", #call, int(e)); \
      exit(1);                                    \
    }                                             \
  } while (0)
extern "C" void **__cudaRegisterFatBinary(void *);
extern "C" void __cudaRegisterFatBinaryEnd(void **);
extern "C" void __cudaRegisterFunction(void **, const char *, char *,
                                       const char *, int, uint3 *, uint3 *,
                                       dim3 *, dim3 *, int *);
struct Wrapper {
  unsigned magic, version;
  const void *data;
  void *filename;
};
static void stamp(const char *s) {
  printf("HOST_TIME %s %.6f\n", s,
         std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
             .count());
  fflush(stdout);
}
__attribute__((constructor(101))) static void startup() {
  stamp("process_init_begin");
}
static std::vector<char> read(const char *file) {
  std::ifstream f(file, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}
