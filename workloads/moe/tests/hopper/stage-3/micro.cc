#include <set>

#include "../stage-1/host.h"
extern "C" const unsigned char module_a[];
extern "C" unsigned __cudaPushCallConfiguration(dim3, dim3, size_t,
                                                cudaStream_t);
extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *, dim3 *, size_t *,
                                                  void *);

static unsigned lcg(unsigned x, unsigned n) {
  while (n--) x = x * 1664525u + 1013904223u;
  return x;
}

int main(int argc, char **argv) {
  stamp("main");
  const char *name = argv[1];
  unsigned threads = std::stoul(argv[2]), grid_x = std::stoul(argv[3]);
  size_t shared = std::stoul(argv[4]);
  unsigned delay = std::stoul(argv[5]);
  unsigned repeats = std::stoul(argv[6]);
  bool plain_ex = !strcmp(name, "plain_ex");
  bool ordinary = !strcmp(name, "ordinary") || plain_ex;
  const char *entry = ordinary ? "ordinary" : name;
  bool barriers = !strcmp(name, "cluster_sync");
  bool sample = !strcmp(name, "address_sample");
  unsigned grid_y = ordinary ? 1 : 2;
  Wrapper wrapper{0x466243b1, 1, module_a, nullptr};
  void **h = __cudaRegisterFatBinary(&wrapper);
  static char marker, ordinary_marker;
  __cudaRegisterFunction(h, &marker, const_cast<char *>(entry), entry, -1,
                         nullptr, nullptr, nullptr, nullptr, nullptr);
  __cudaRegisterFunction(h, &ordinary_marker, const_cast<char *>("ordinary"),
                         "ordinary", -1, nullptr, nullptr, nullptr, nullptr,
                         nullptr);
  __cudaRegisterFatBinaryEnd(h);
  stamp("registration_done");
  CHECK(cudaFuncSetAttribute(
      &marker, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  size_t words = size_t(grid_x) * grid_y * threads * 32;
  unsigned *out;
  CHECK(cudaMalloc(&out, words * 4));
  if (reinterpret_cast<uintptr_t>(out) < 0x200000000ULL) return 3;
  std::vector<unsigned> result(words, 0xdeadbeef);
  void *args[] = {&out, &delay};
  unsigned errors = 0;
  auto check = [&](bool pass, size_t index, unsigned want) {
    if (!pass) {
      if (errors < 12)
        printf("MISMATCH index=%zu expected=%u actual=%u\n", index, want,
               result[index]);
      ++errors;
    }
  };
  for (unsigned rep = 0; rep < repeats; ++rep) {
    CHECK(cudaMemcpy(out, result.data(), words * 4, cudaMemcpyHostToDevice));
    // Ordinary -> ExC -> ordinary uses the same direct-launch binding path.
    if (!ordinary) {
      CHECK(cudaLaunchKernel(&ordinary_marker, dim3(1), dim3(32), args, 32,
                             stream));
      CHECK(cudaStreamSynchronize(stream));
    }
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {1, ordinary ? 1u : 2u, 1};
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(grid_x, grid_y, 1);
    cfg.blockDim = dim3(threads, 1, 1);
    cfg.dynamicSmemBytes = shared;
    cfg.stream = stream;
    cfg.attrs = &attr;
    cfg.numAttrs = plain_ex && rep == 1 ? 0 : 1;
    __cudaPushCallConfiguration(dim3(3, 1, 1), dim3(64, 1, 1), 96, stream);
    stamp("kernel_begin");
    if (ordinary && !plain_ex)
      CHECK(cudaLaunchKernel(&marker, cfg.gridDim, cfg.blockDim, args, shared,
                             stream));
    else
      CHECK(cudaLaunchKernelExC(&cfg, &marker, args));
    dim3 saved_grid, saved_block;
    size_t saved_shared;
    cudaStream_t saved_stream;
    CHECK(__cudaPopCallConfiguration(&saved_grid, &saved_block, &saved_shared,
                                     &saved_stream));
    if (saved_grid.x != 3 || saved_block.x != 64 || saved_shared != 96 ||
        saved_stream != stream)
      return 4;
    CHECK(cudaStreamSynchronize(stream));
    stamp("kernel_done");
    CHECK(cudaMemcpy(result.data(), out, words * 4, cudaMemcpyDeviceToHost));
    stamp("compare_begin");
    std::set<unsigned> sms;
    unsigned offsets[] = {0, 4088, 4092};
    if (!strcmp(name, "mapa") || barriers) {
      offsets[1] = 4096;
      offsets[2] = 8188;
    } else if (!strcmp(name, "addresses") || sample) {
      offsets[1] = 98304;
      offsets[2] = 215036;
    } else if (ordinary) {
      offsets[1] = 24;
      offsets[2] = 28;
    }
    for (unsigned y = 0; y < grid_y; ++y)
      for (unsigned x = 0; x < grid_x; ++x)
        for (unsigned lane = 0; lane < (sample ? 1 : threads); ++lane) {
          size_t base = ((size_t(y) * grid_x + x) * threads + lane) * 32;
          unsigned rank = ordinary ? 0 : y;
          unsigned tag = 100 * x + 10 * rank + 1;
          unsigned want[] = {x,   y,   rank, 0, rank, 0, result[base + 6],
                             tag, lane};
          for (unsigned j = 0; j < 9; ++j)
            check(result[base + j] == want[j], base + j, want[j]);
          unsigned sm = result[base + 6];
          sms.insert(sm);
          check(sm < 132, base + 6, 131);
          for (unsigned j = 0; j < 3; ++j) {
            check(result[base + 9 + j] == tag, base + 9 + j, tag);
            check(result[base + 12 + j] == offsets[j], base + 12 + j,
                  offsets[j]);
            check(result[base + 25 + j] == 1, base + 25 + j, 1);
          }
          check(result[base + 15] == 0, base + 15, 0);
          check(result[base + 30] == 1, base + 30, 1);
          check(result[base + 31] == 1, base + 31, 1);
          uint64_t generic =
              uint64_t(result[base + 28]) | (uint64_t(result[base + 29]) << 32);
          uint64_t expected_generic = 0x200000000ULL - 132 * 262144ULL +
                                      uint64_t(sm) * 262144 + offsets[2];
          check(generic == expected_generic, base + 28, expected_generic);
          if (!strcmp(name, "mapa")) {
            check(result[base + 16] == 4096, base + 16, 4096);
            check(result[base + 17] == 0x01001000, base + 17, 0x01001000);
          }
          if (barriers)
            for (unsigned round = 0; round < 3; ++round) {
              unsigned n = (x == 0 && rank == 1) ? delay * (round + 1) : 0;
              unsigned value = lcg(7, n + 8);
              check(result[base + 16 + round] == value, base + 16 + round,
                    value);
              check(result[base + 22 + round] > result[base + 19 + round],
                    base + 22 + round, result[base + 19 + round] + 1);
            }
        }
    if (!ordinary) {
      for (unsigned x = 0; x < grid_x; ++x)
        check(result[x * threads * 32 + 6] !=
                  result[(grid_x + x) * threads * 32 + 6],
              x * threads * 32 + 6, ~0u);
    }
    if (grid_x == 66 && !ordinary)
      for (unsigned sm : {0u, 79u, 80u, 131u}) check(sms.count(sm), 6, sm);
    std::ofstream f(std::string(name) + ".actual-" + std::to_string(rep),
                    std::ios::binary);
    f.write(reinterpret_cast<char *>(result.data()), words * 4);
    printf("SM_SAMPLES repeat=%u count=%zu", rep, sms.size());
    for (unsigned sm : sms) printf(" %u", sm);
    puts("");
    stamp("compare_end");
    if (!ordinary) {
      CHECK(cudaLaunchKernel(&ordinary_marker, dim3(1), dim3(32), args, 32,
                             stream));
      CHECK(cudaStreamSynchronize(stream));
      unsigned probe[32];
      CHECK(cudaMemcpy(probe, out, sizeof(probe), cudaMemcpyDeviceToHost));
      check(probe[2] == 0 && probe[4] == 0 && probe[9] == 1, 2, 0);
    }
  }
  printf("%s %s elements=%zu repeats=%u errors=%u\n", errors ? "FAIL" : "PASS",
         name, words, repeats, errors);
  CHECK(cudaFree(out));
  CHECK(cudaStreamDestroy(stream));
  return errors ? 2 : 0;
}
