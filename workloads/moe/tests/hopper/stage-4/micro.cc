#include "../stage-1/host.h"
extern "C" const unsigned char module_a[];

int main(int argc, char **argv) {
  stamp("main");
  const char *name = argv[1];
  unsigned delay = std::stoul(argv[5]);
  bool remote = !strcmp(name, "remote");
  Wrapper wrapper{0x466243b1, 1, module_a, nullptr};
  void **h = __cudaRegisterFatBinary(&wrapper);
  static char marker;
  __cudaRegisterFunction(h, &marker, const_cast<char *>(name), name, -1,
                         nullptr, nullptr, nullptr, nullptr, nullptr);
  __cudaRegisterFatBinaryEnd(h);
  stamp("registration_done");
  unsigned *out;
  std::vector<unsigned> result(128, 0xdeadbeef);
  CHECK(cudaMalloc(&out, result.size() * 4));
  unsigned errors = 0;
  for (unsigned rep = 0; rep < 2; ++rep) {
    result.assign(128, 0xdeadbeef);
    CHECK(cudaMemcpy(out, result.data(), result.size() * 4,
                     cudaMemcpyHostToDevice));
    void *args[] = {&out, &delay};
    stamp("kernel_begin");
    if (remote) {
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeClusterDimension;
      attr.val.clusterDim = {1, 2, 1};
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(2, 2);
      cfg.blockDim = dim3(64);
      cfg.dynamicSmemBytes = 2048;
      cfg.attrs = &attr;
      cfg.numAttrs = 1;
      CHECK(cudaLaunchKernelExC(&cfg, &marker, args));
    } else {
      CHECK(cudaLaunchKernel(&marker, dim3(2), dim3(64), args, 2048));
    }
    CHECK(cudaDeviceSynchronize());
    stamp("kernel_done");
    CHECK(cudaMemcpy(result.data(), out, result.size() * 4,
                     cudaMemcpyDeviceToHost));
    stamp("compare_begin");
    auto eq = [&](unsigned at, unsigned want) {
      if (result[at] != want) {
        printf("MISMATCH index=%u expected=%u actual=%u\n", at, want,
               result[at]);
        ++errors;
      }
    };
    for (unsigned cta = 0; cta < (remote ? 4 : 2); ++cta) {
      unsigned b = cta * 32;
      if (remote) {
        eq(b, cta < 2 ? 0 : 77);
        if (cta >= 2 && !result[b + 1])
          ++errors;
      } else if (!strcmp(name, "local") || !strcmp(name, "hint")) {
        for (unsigned r = 0; r < 3; ++r) {
          eq(b + r * 4, 0);
          eq(b + r * 4 + 1, (r + 1) * 11);
          if (!result[b + r * 4 + 2])
            ++errors;
          printf("POLLS cta=%u round=%u queries=%u\n", cta, r,
                 result[b + r * 4 + 2] + 1);
        }
      } else
        eq(b, 99);
    }
    std::ofstream f(std::string(name) + ".actual-" + std::to_string(rep),
                    std::ios::binary);
    f.write(reinterpret_cast<char *>(result.data()), result.size() * 4);
    stamp("compare_end");
  }
  printf("%s %s errors=%u\n", errors ? "FAIL" : "PASS", name, errors);
  CHECK(cudaFree(out));
  return errors ? 2 : 0;
}
