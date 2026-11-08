#include "../stage-1/host.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cuda.h>
extern "C" const unsigned char module_a[];
using Encode = CUresult (*)(CUtensorMap *, CUtensorMapDataType, cuuint32_t,
                            void *, const cuuint64_t *, const cuuint64_t *,
                            const cuuint32_t *, const cuuint32_t *,
                            CUtensorMapInterleave, CUtensorMapSwizzle,
                            CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static unsigned physical(unsigned base, unsigned row, unsigned k, unsigned w) {
  static const unsigned blocks[8][8] = {
      {0, 1, 2, 3, 4, 5, 6, 7}, {1, 0, 3, 2, 5, 4, 7, 6},
      {2, 3, 0, 1, 6, 7, 4, 5}, {3, 2, 1, 0, 7, 6, 5, 4},
      {4, 5, 6, 7, 0, 1, 2, 3}, {5, 4, 7, 6, 1, 0, 3, 2},
      {6, 7, 4, 5, 2, 3, 0, 1}, {7, 6, 5, 4, 3, 2, 1, 0}};
  unsigned linear = base + row * w + k * 2;
  return (linear / 128) * 128 +
         blocks[(linear / 128) % (w / 16)][(linear % 128) / 16] * 16 +
         linear % 16;
}
static uint16_t bf(float f) {
  uint32_t bits;
  memcpy(&bits, &f, 4);
  return bits >> 16;
}
int main(int argc, char **argv) {
  stamp("main");
  std::string name = argv[1];
  unsigned threads = std::stoul(argv[2]);
  unsigned shared_bytes = std::stoul(argv[4]);
  Wrapper wrapper{0x466243b1, 1, module_a, nullptr};
  auto h = __cudaRegisterFatBinary(&wrapper);
  static char marker;
  std::string entry = "test_" + name;
  __cudaRegisterFunction(h, &marker, const_cast<char *>(entry.c_str()),
                         entry.c_str(), -1, nullptr, nullptr, nullptr, nullptr,
                         nullptr);
  __cudaRegisterFatBinaryEnd(h);
  stamp("registration_done");
  bool pipeline = name == "pipeline" || name == "serial";
  bool io = name == "ldmatrix" || name == "stmatrix";
  unsigned width = name.find("64") != std::string::npos ? 64 : 128;
  unsigned base = name.find("offset") == 0
                      ? (name.find("low") != std::string::npos ? 1296 : 1280)
                      : 0;
  std::vector<unsigned char> input(81920, 0);
  auto put = [&](unsigned addr, uint16_t v) {
    memcpy(input.data() + addr, &v, 2);
  };
  for (unsigned m = 0; m < 64; ++m)
    for (unsigned k = 0; k < 16; ++k)
      put(physical(base, m, k, width), bf(int(m % 7) - 3 + int(k % 3)));
  for (unsigned n = 0; n < 128; ++n)
    for (unsigned k = 0; k < 16; ++k)
      put(physical(base + 16384, n, k, width),
          bf(int(n % 11) - 5 - int(k % 2)));
  for (unsigned t = 0; t < 128; ++t)
    for (unsigned j = 0; j < 64; ++j) {
      unsigned m = (t / 32) * 16 + (t % 32) / 4 + (j / 2 % 2) * 8;
      unsigned n = (j / 4) * 8 + (t % 4) * 2 + j % 2;
      float d = int(m % 5) - int(n % 3);
      memcpy(input.data() + 40960 + 4 * (t * 64 + j), &d, 4);
    }
  if (io)
    for (unsigned q = 0; q < 4; ++q)
      for (unsigned r = 0; r < 8; ++r)
        for (unsigned c = 0; c < 8; ++c) {
          uint16_t v = 1000 * q + 10 * r + c + 1;
          unsigned address = name == "ldmatrix"
                                 ? q * 128 + r * 16 + c * 2
                                 : (4 * r + c / 2) * 16 + q * 4 + (c % 2) * 2;
          put(address, v);
        }
  if (name == "quota")
    for (unsigned j = 0; j < 200; ++j) {
      float value = j + 1;
      memcpy(input.data() + 32768 + 4 * j, &value, 4);
    }
  if (name == "legacy8")
    for (unsigned j = 0; j < 256; ++j) {
      float value = 3 * j + 1;
      memcpy(input.data() + 4 * j, &value, 4);
    }
  void *in, *out, *ma = nullptr, *mb = nullptr, *mc = nullptr, *gc = nullptr;
  void *ga = nullptr, *gb = nullptr;
  CHECK(cudaMalloc(&in, input.size()));
  CHECK(cudaMalloc(&out, 65536));
  CHECK(cudaMemcpy(in, input.data(), input.size(), cudaMemcpyHostToDevice));
  if (name == "integration" || pipeline) {
    Encode encode = nullptr;
    cudaDriverEntryPointQueryResult status;
    CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",
                                           (void **)&encode, 12000,
                                           cudaEnableDefault, &status));
    assert(encode);
    std::vector<uint16_t> a(32 * 128), b(64 * 256);
    for (unsigned m = 0; m < 64; ++m)
      for (unsigned k = 0; k < 16; ++k)
        a[m * 32 + k] = bf(int(m % 7) - 3 + int(k % 3));
    for (unsigned n = 0; n < 128; ++n)
      for (unsigned k = 0; k < 16; ++k)
        b[n * 64 + k] = bf(int(n % 11) - 5 - int(k % 2));
    if (pipeline) {
      for (unsigned m = 0; m < 64; ++m)
        for (unsigned k = 0; k < 16; ++k)
          a[(m + 64) * 32 + k] = bf(int(m % 7) - 2 + int(k % 3));
      for (unsigned n = 0; n < 128; ++n)
        for (unsigned k = 0; k < 16; ++k)
          b[(n + 128) * 64 + k] = bf(int(n % 11) - 3 - int(k % 2));
    }
    CHECK(cudaMalloc(&ga, a.size() * 2));
    CHECK(cudaMalloc(&gb, b.size() * 2));
    CHECK(cudaMemcpy(ga, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gb, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
    auto map = [&](void *data, unsigned rows, void **ptr) {
      CUtensorMap desc;
      unsigned cols = rows == 64 ? 32 : 64;
      cuuint64_t dims[] = {cols, rows * (pipeline ? 2u : 1u)},
                 stride[] = {cols * 2};
      cuuint32_t box[] = {cols, rows}, element[] = {1, 1};
      assert(encode(&desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, data, dims,
                    stride, box, element, CU_TENSOR_MAP_INTERLEAVE_NONE,
                    rows == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                               : CU_TENSOR_MAP_SWIZZLE_128B,
                    CU_TENSOR_MAP_L2_PROMOTION_NONE,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
      CHECK(cudaMalloc(ptr, 128));
      CHECK(cudaMemcpy(*ptr, &desc, 128, cudaMemcpyHostToDevice));
    };
    map(ga, 64, &ma);
    map(gb, 128, &mb);
    if (pipeline) {
      CHECK(cudaMalloc(&gc, 32768));
      CHECK(cudaMalloc(&mc, 128));
      CUtensorMap desc;
      cuuint64_t dims[] = {128, 128}, stride[] = {256};
      cuuint32_t box[] = {64, 64}, element[] = {1, 1};
      assert(encode(&desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gc, dims,
                    stride, box, element, CU_TENSOR_MAP_INTERLEAVE_NONE,
                    CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
      CHECK(cudaMemcpy(mc, &desc, 128, cudaMemcpyHostToDevice));
    }
  }
  if (name == "quota") {
    cudaFuncAttributes attributes{};
    CHECK(cudaFuncGetAttributes(&attributes, &marker));
    assert(attributes.numRegs == 168);
    printf("QUOTA initial_regs=%d total=64512\n", attributes.numRegs);
  }
  CHECK(cudaFuncSetAttribute(
      &marker, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  unsigned errors = 0;
  for (unsigned rep = 0; rep < 2; ++rep) {
    if (name == "integration") {
      std::vector<uint16_t> a(32 * 64), b(64 * 128);
      for (unsigned m = 0; m < 64; ++m)
        for (unsigned k = 0; k < 16; ++k)
          a[m * 32 + k] = bf(int(m % 7) - 3 + int(k % 3) + int(rep));
      for (unsigned n = 0; n < 128; ++n)
        for (unsigned k = 0; k < 16; ++k)
          b[n * 64 + k] = bf(int(n % 11) - 5 - int(k % 2) + int(2 * rep));
      CHECK(cudaMemcpy(ga, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
      CHECK(cudaMemcpy(gb, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
    }
    std::vector<float> output(16384, NAN);
    if (pipeline) {
      std::vector<uint16_t> sentinel(16384, 0xabcd);
      CHECK(cudaMemcpy(gc, sentinel.data(), 32768, cudaMemcpyHostToDevice));
    }
    CHECK(cudaMemcpy(out, output.data(), 65536, cudaMemcpyHostToDevice));
    void *args[] = {&in, &out, &ma, &mb, &mc};
    stamp("kernel_begin");
    if (name == "integration") {
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeClusterDimension;
      attr.val.clusterDim = {1, 2, 1};
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(1, 2);
      cfg.blockDim = dim3(threads);
      cfg.dynamicSmemBytes = shared_bytes;
      cfg.attrs = &attr;
      cfg.numAttrs = 1;
      CHECK(cudaLaunchKernelExC(&cfg, &marker, args));
    } else {
      CHECK(cudaLaunchKernel(&marker, dim3(1), dim3(threads), args,
                             shared_bytes, nullptr));
    }
    CHECK(cudaDeviceSynchronize());
    stamp("kernel_end");
    CHECK(cudaMemcpy(output.data(), out, 65536, cudaMemcpyDeviceToHost));
    stamp("compare_begin");
    auto check = [&](unsigned i, float expected) {
      if (output[i] != expected) {
        if (errors < 12)
          printf("MISMATCH i=%u actual=%g expected=%g\n", i, output[i],
                 expected);
        ++errors;
      }
    };
    if (pipeline) {
      std::vector<uint16_t> values(16384);
      CHECK(cudaMemcpy(values.data(), gc, 32768, cudaMemcpyDeviceToHost));
      for (unsigned stage = 0; stage < 2; ++stage)
        for (unsigned m = 0; m < 64; ++m)
          for (unsigned n = 0; n < 128; ++n) {
            float c = 0;
            for (unsigned k = 0; k < 16; ++k)
              c += (int(m % 7) - 3 + int(k % 3) + int(stage)) *
                   (int(n % 11) - 5 - int(k % 2) + int(2 * stage));
            uint32_t bits;
            memcpy(&bits, &c, 4);
            uint16_t expected = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
            unsigned i = stage * 8192 + m * 128 + n;
            if (values[i] != expected) {
              if (errors < 12)
                printf("PIPE_MISMATCH i=%u actual=%u expected=%u\n", i,
                       values[i], expected);
              ++errors;
            }
          }
      std::ofstream raw(name + "-bf16-" + std::to_string(rep) + ".bin",
                        std::ios::binary);
      raw.write((char *)values.data(), 32768);
    } else if (name == "quota") {
      for (unsigned t = 0; t < 384; ++t)
        check(t, 42);
      for (unsigned j = 0; j < 200; ++j) {
        float expected;
        memcpy(&expected, input.data() + 32768 + 4 * j, 4);
        check(1024 + j, expected);
      }
    } else if (name == "legacy8") {
      for (unsigned j = 0; j < 256; ++j)
        check(j, 3 * j + 1);
    } else if (io) {
      auto *bits = reinterpret_cast<uint16_t *>(output.data());
      for (unsigned q = 0; q < 4; ++q)
        for (unsigned r = 0; r < 8; ++r)
          for (unsigned c = 0; c < 8; ++c) {
            unsigned i = name == "ldmatrix"
                             ? (4 * r + c / 2) * 8 + q * 2 + c % 2
                             : q * 64 + r * 8 + c;
            if (bits[i] != 1000 * q + 10 * r + c + 1)
              ++errors;
          }
    } else
      for (unsigned t = 0; t < 128; ++t)
        for (unsigned j = 0; j < 64; ++j) {
          // Written as eight-column strips and row pairs, independent of the
          // implementation.
          unsigned row = (t / 32) * 16 + (t % 32) / 4 + (j / 2 % 2) * 8;
          unsigned col = (j / 4) * 8 + (t % 4) * 2 + j % 2;
          float expected = 0;
          unsigned variant = name == "integration" ? rep : 0;
          for (unsigned k = 0; k < 16; ++k)
            expected += (int(row % 7) - 3 + int(k % 3) + int(variant)) *
                        (int(col % 11) - 5 - int(k % 2) + int(2 * variant));
          if (name == "chain")
            expected *= 2;
          if (name == "initial")
            expected += int(row % 5) - int(col % 3);
          check(t * 64 + j, expected);
          if (name == "groups")
            check(8192 + t * 64 + j, 2 * expected);
        }
    std::ofstream file(name + "-" + std::to_string(rep) + ".bin",
                       std::ios::binary);
    file.write((char *)output.data(), 65536);
    stamp("compare_end");
  }
  printf("%s errors=%u %s\n", name.c_str(), errors, errors ? "FAIL" : "PASS");
  return errors ? 1 : 0;
}
