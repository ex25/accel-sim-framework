#include "host.h"
extern "C" const unsigned char module_a[], module_legacy[];
int main(int argc, char **argv) {
  stamp("main");
  Wrapper wrapper{0x466243b1, 1,
                  !strcmp(argv[1], "legacy") ? module_legacy : module_a,
                  nullptr};
  void **h = __cudaRegisterFatBinary(&wrapper);
  static char marker;
  __cudaRegisterFunction(h, &marker, argv[1], argv[1], -1, nullptr, nullptr,
                         nullptr, nullptr, nullptr);
  __cudaRegisterFatBinaryEnd(h);
  stamp("registration_done");
  auto input = read("input.bin");
  auto expected = read((std::string(argv[1]) + ".expected").c_str());
  unsigned threads = expected.size() / 256;
  unsigned *in, *out;
  CHECK(cudaMalloc(&in, input.size()));
  CHECK(cudaMalloc(&out, expected.size()));
  CHECK(cudaMemcpy(in, input.data(), input.size(), cudaMemcpyHostToDevice));
  std::vector<unsigned> result(expected.size() / 4, 0xdeadbeef);
  CHECK(
      cudaMemcpy(out, result.data(), expected.size(), cudaMemcpyHostToDevice));
  void *args[] = {&in, &out, &threads};
  stamp("kernel_begin");
  CHECK(cudaLaunchKernel(&marker, dim3(1), dim3(threads), args));
  CHECK(cudaDeviceSynchronize());
  stamp("kernel_done");
  CHECK(
      cudaMemcpy(result.data(), out, expected.size(), cudaMemcpyDeviceToHost));
  stamp("compare_begin");
  std::ofstream f(std::string(argv[1]) + ".actual", std::ios::binary);
  f.write((char *)result.data(), expected.size());
  auto want = (const unsigned *)expected.data();
  unsigned errors = 0;
  for (unsigned i = 0; i < result.size(); ++i)
    if (result[i] != want[i]) {
      if (errors < 12)
        printf("MISMATCH lane=%u slot=%u expected=%08x actual=%08x\n", i / 64,
               i % 64, want[i], result[i]);
      ++errors;
    }
  stamp("compare_end");
  printf("%s %s elements=%zu errors=%u\n", errors ? "FAIL" : "PASS", argv[1],
         result.size(), errors);
  return errors ? 2 : 0;
}
