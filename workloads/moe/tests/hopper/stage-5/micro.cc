#include "../stage-1/host.h"
#include <array>
#include <cassert>
#include <cuda.h>
extern "C" const unsigned char module_a[];
using Encode = CUresult (*)(CUtensorMap *, CUtensorMapDataType, cuuint32_t,
                            void *, const cuuint64_t *, const cuuint64_t *,
                            const cuuint32_t *, const cuuint32_t *,
                            CUtensorMapInterleave, CUtensorMapSwizzle,
                            CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
// Independent block-table reference: each row lists the physical 16B blocks.
// No simulator header or bitwise swizzle mapping is used here.
static unsigned physical(unsigned q, unsigned swizzle, unsigned base = 1280) {
  static const unsigned table[8][8] = {
      {0, 1, 2, 3, 4, 5, 6, 7}, {1, 0, 3, 2, 5, 4, 7, 6},
      {2, 3, 0, 1, 6, 7, 4, 5}, {3, 2, 1, 0, 7, 6, 5, 4},
      {4, 5, 6, 7, 0, 1, 2, 3}, {5, 4, 7, 6, 1, 0, 3, 2},
      {6, 7, 4, 5, 2, 3, 0, 1}, {7, 6, 5, 4, 3, 2, 1, 0}};
  unsigned address = base + q;
  unsigned phase = (address / 128) % (swizzle == 64 ? 4 : 8);
  return (address / 128) * 128 + table[phase][(address % 128) / 16] * 16 +
         q % 16 - base;
}
int main(int argc, char **argv) {
  stamp("main");
  std::string test = argv[1];
  bool expert = test.find("expert") == 0;
  bool remote = test.find("mask") == 0;
  bool output = test.find("output") == 0;
  bool rotate = test == "rotate";
  unsigned bx = 32, by = 64, swizzle = 64;
  if (test == "expert64") {
    bx = 32;
    by = 64;
    swizzle = 64;
  } else if (test.find("128") != std::string::npos) {
    bx = 64;
    by = 128;
    swizzle = 128;
  } else if (test.find("64") != std::string::npos) {
    bx = 64;
    swizzle = 128;
  }
  unsigned bytes = bx * by * 2;
  unsigned x = test.find("edge") != std::string::npos ? 112
               : test.find("oob") == 0                ? 256
                                                      : 0;
  unsigned y = test.find("edge") != std::string::npos ? 144 : 0;
  unsigned mask = remote ? test.back() - '0' : 1;
  const char *entry = expert  ? test.c_str()
                      : remote   ? "multicast"
                      : output ? "output"
                      : rotate ? "rotate"
                               : "input";
  Wrapper wrapper{0x466243b1, 1, module_a, nullptr};
  void **h = __cudaRegisterFatBinary(&wrapper);
  static char marker;
  __cudaRegisterFunction(h, &marker, const_cast<char *>(entry), entry, -1,
                         nullptr, nullptr, nullptr, nullptr, nullptr);
  __cudaRegisterFatBinaryEnd(h);
  stamp("registration_done");
  Encode encode = nullptr;
  cudaDriverEntryPointQueryResult status;
  CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",
                                         reinterpret_cast<void **>(&encode),
                                         12000, cudaEnableDefault, &status));
  assert(status == cudaDriverEntryPointSuccess && encode);
  unsigned short *source, *second, *dest0, *dest1, *out;
  void *published, *destmap, *prepared;
  constexpr unsigned elements = 160 * 160;
  CHECK(cudaMalloc(&source, elements * 2));
  CHECK(cudaMalloc(&second, elements * 2));
  CHECK(cudaMalloc(&dest0, elements * 2));
  CHECK(cudaMalloc(&dest1, elements * 2));
  CHECK(cudaMalloc(&out, expert ? 2 * 33088 : 4 * 32768));
  CHECK(cudaMalloc(&published, expert ? 2 * 128 : 4 * 128));
  CHECK(cudaMalloc(&destmap, 2 * 128));
  CHECK(cudaMalloc(&prepared, bytes));
  cuuint64_t dims[2] = {1, 1}, strides[1] = {0};
  cuuint32_t box[2] = {bx, by}, elem[2] = {1, 1};
  auto make = [&](void *base) {
    CUtensorMap map;
    assert(encode(&map,
                  swizzle == 64 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
                                : CU_TENSOR_MAP_DATA_TYPE_UINT16,
                  2, base, dims, strides, box, elem,
                  CU_TENSOR_MAP_INTERLEAVE_NONE,
                  swizzle == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                                : CU_TENSOR_MAP_SWIZZLE_128B,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
    // Check the host ABI itself, before using its bytes as the copy reference.
    std::array<unsigned char, 128> expected{};
    auto field = [&](unsigned offset, const auto &value) {
      memcpy(expected.data() + offset, &value, sizeof(value));
    };
    field(0, reinterpret_cast<uint64_t>(base));
    field(8, uint64_t(strides[0]));
    for (unsigned i = 0; i < 2; ++i) {
      field(40 + 4 * i, uint32_t(dims[i]));
      field(60 + 4 * i, uint32_t(box[i]));
      field(80 + 4 * i, uint32_t(elem[i]));
    }
    const uint32_t flags[] = {
        2,
        uint32_t(swizzle == 64 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
                               : CU_TENSOR_MAP_DATA_TYPE_UINT16),
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        uint32_t(swizzle == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                               : CU_TENSOR_MAP_SWIZZLE_128B),
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE};
    memcpy(expected.data() + 100, flags, sizeof(flags));
    assert(memcmp(&map, expected.data(), expected.size()) == 0);
    puts("ABI_MAP all 128 bytes PASS");
    return map;
  };
  CUtensorMap initial = make(expert ? second : source);
  std::array<unsigned char, 128> original;
  memcpy(original.data(), &initial, 128);
  dims[0] = 128;
  dims[1] = 160;
  strides[0] = 320;
  CUtensorMap outputs[2] = {make(dest0), make(dest1)};
  CHECK(cudaMemcpy(destmap, outputs, sizeof(outputs), cudaMemcpyHostToDevice));
  unsigned errors = 0;
  auto check = [&](unsigned actual, unsigned want, unsigned at) {
    if (actual != want) {
      if (errors < 12)
        printf("MISMATCH at=%u expected=%u actual=%u\n", at, want, actual);
      ++errors;
    }
  };
  if (expert) {
    constexpr unsigned shared_marker = 0xa5;
    unsigned transfer_bytes = test == "expert64" ? 4096 : 32768;
    unsigned dump_bytes = transfer_bytes + 256;
    unsigned cta_output_bytes = dump_bytes + 64;
    unsigned ctas = test == "expert64" ? 1 : 2;
    unsigned stages = test == "expert64" ? 1 : 3;
    CHECK(cudaFuncSetAttribute(&marker, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               215040));
    std::vector<unsigned short> data(elements, 0);
    for (unsigned j = 0; j < 160; ++j)
      for (unsigned i = 0; i < 128; ++i)
        data[j * 160 + i] = 1000 + i + 128 * j;
    CHECK(cudaMemcpy(second, data.data(), elements * 2, cudaMemcpyHostToDevice));
    for (unsigned stage = 0; stage < stages; ++stage) {
      CHECK(cudaMemset(out, 0, ctas * cta_output_bytes));
      CHECK(cudaMemset(published, 0, ctas * 128));
      unsigned x = 0, y = 0;
      void *args[] = {&out, &initial, &published, &second, &x, &y, &stage};
      stamp("kernel_begin");
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeClusterDimension;
      attr.val.clusterDim = {1, ctas, 1};
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(1, ctas);
      cfg.blockDim = dim3(32);
      cfg.dynamicSmemBytes = 215040;
      cfg.attrs = &attr;
      cfg.numAttrs = 1;
      CHECK(cudaLaunchKernelExC(&cfg, &marker, args));
      CHECK(cudaDeviceSynchronize());
      stamp("kernel_done");

      std::vector<unsigned char> got(ctas * cta_output_bytes);
      CHECK(cudaMemcpy(got.data(), out, got.size(), cudaMemcpyDeviceToHost));
      std::vector<unsigned char> maps(ctas * 128);
      CHECK(cudaMemcpy(maps.data(), published, maps.size(),
                       cudaMemcpyDeviceToHost));
      stamp("compare_begin");
      unsigned swizzle_phase = swizzle == 64 ? 4 : 8;
      std::array<unsigned char, 128> expected_map = original;
      uint64_t global_base = reinterpret_cast<uint64_t>(second);
      uint64_t stride = 320;
      memcpy(expected_map.data(), &global_base, 8);
      memcpy(expected_map.data() + 8, &stride, 8);
      uint32_t dims[5] = {128, 160, 1, 1, 1};
      memcpy(expected_map.data() + 40, dims, sizeof(dims));
      for (unsigned cta = 0; cta < ctas; ++cta) {
        unsigned local_base = test == "expert64" ? 196608
                              : test == "expert128" ? stage * 32768 + cta * 8192
                              : 98304 + stage * 32768;
        unsigned delivery_base = test == "expert128"
                                     ? stage * 32768 + 8192
                                     : local_base;
        unsigned barrier = test == "expert64" ? 213680
                                               : 213632 + stage * 8;
        unsigned header[9];
        memcpy(header, got.data() + cta * cta_output_bytes + dump_bytes,
               sizeof(header));
        check(header[0], 0, cta * 9);
        check(header[1], local_base, cta * 9 + 1);
        check(header[2], delivery_base, cta * 9 + 2);
        check(header[3], barrier, cta * 9 + 3);
        check(header[4], cta, cta * 9 + 4);
        check(header[5], stage, cta * 9 + 5);
        check(header[6], 0, cta * 9 + 6);
        check(header[7], 0, cta * 9 + 7);
        check(header[8], transfer_bytes, cta * 9 + 8);
        check((delivery_base / 128) % swizzle_phase, 0,
              cta * 9 + 9);
        for (unsigned byte = 0; byte < 128; ++byte)
          check(maps[cta * 128 + byte], expected_map[byte], cta * 128 + byte);

        std::vector<unsigned char> expected(dump_bytes, shared_marker);
        unsigned tile_count = test == "expert64" ? 1 : 2;
        for (unsigned tile = 0; tile < tile_count; ++tile) {
          unsigned tile_base = delivery_base + tile * 16384;
          for (unsigned v = 0; v < by; ++v)
            for (unsigned u = 0; u < bx; ++u) {
              unsigned logical = 2 * (v * bx + u);
              unsigned physical_offset = physical(logical, swizzle, tile_base);
              unsigned short value = data[v * 160 + tile * 64 + u];
              memcpy(expected.data() + 128 + tile * 16384 + physical_offset,
                     &value, 2);
            }
        }
        unsigned char *actual = got.data() + cta * cta_output_bytes;
        for (unsigned at = 0; at < dump_bytes; ++at)
          check(actual[at], expected[at], cta * dump_bytes + at);
        std::ofstream dump(test + ".stage-" + std::to_string(stage) +
                               "-rank-" + std::to_string(cta) + ".shared",
                           std::ios::binary);
        dump.write(reinterpret_cast<char *>(actual), dump_bytes);
        printf("EXPERT_SAMPLE test=%s stage=%u rank=%u shared_symbol=%u local=%u delivered=%u barrier=%u swizzle=%u phase=%u\n",
               test.c_str(), stage, cta, header[0], header[1], header[2],
               header[3], swizzle, (delivery_base / 128) % swizzle_phase);
      }
      check(memcmp(&initial, original.data(), 128), 0, 0);
      stamp("compare_end");
    }
    printf("%s %s errors=%u\n", errors ? "FAIL" : "PASS", test.c_str(),
           errors);
    return errors ? 2 : 0;
  }
  for (unsigned rep = 0; rep < 2; ++rep) {
    std::vector<unsigned short> data(elements, 0), old(elements, 999);
    for (unsigned j = 0; j < 160; ++j)
      for (unsigned i = 0; i < 128; ++i)
        data[j * 160 + i] = 1000 + rep * 19000 + i + 128 * j;
    CHECK(cudaMemcpy(source, old.data(), elements * 2, cudaMemcpyHostToDevice));
    CHECK(
        cudaMemcpy(second, data.data(), elements * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(published, 0, 4 * 128));
    CHECK(cudaMemset(dest0, 0xee, elements * 2));
    CHECK(cudaMemset(dest1, 0xee, elements * 2));
    // Repeated all-OOB and unselected destinations must write their result too.
    CHECK(cudaMemset(out, 0xcc, 4 * 32768));
    if (output) {
      std::vector<unsigned short> physical_data(bytes / 2, 0);
      for (unsigned v = 0; v < by; ++v)
        for (unsigned u = 0; u < bx; ++u)
          if (x + u < 128 && y + v < 160)
            physical_data[physical(2 * (v * bx + u), swizzle) / 2] =
                data[(y + v) * 160 + x + u];
      CHECK(cudaMemcpy(prepared, physical_data.data(), bytes,
                       cudaMemcpyHostToDevice));
    }
    void *args[] = {&out, &initial, &published, &second, &destmap,
                    &x,   &y,       &bytes,     &mask,   &prepared};

    stamp("kernel_begin");
    if (remote) {
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeClusterDimension;
      attr.val.clusterDim = {1, 2, 1};
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(2, 2);
      cfg.blockDim = dim3(32);
      cfg.dynamicSmemBytes = 40000;
      cfg.attrs = &attr;
      cfg.numAttrs = 1;
      CHECK(cudaLaunchKernelExC(&cfg, &marker, args));
    } else
      CHECK(cudaLaunchKernel(&marker, dim3(1), dim3(32), args, 40000));
    CHECK(cudaDeviceSynchronize());
    stamp("kernel_done");
    std::vector<unsigned short> got(4 * 16384);
    CHECK(cudaMemcpy(got.data(), out, got.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<unsigned char> maps(4 * 128);
    CHECK(cudaMemcpy(maps.data(), published, maps.size(),
                     cudaMemcpyDeviceToHost));
    stamp("compare_begin");
    unsigned ctas = remote ? 4 : 1;
    for (unsigned c = 0; c < ctas; ++c) {
      std::array<unsigned char, 128> expected = original;
      uint64_t base = reinterpret_cast<uint64_t>(rotate ? source : second),
               stride = 320, inactive = 160;
      memcpy(expected.data(), &base, 8);
      memcpy(expected.data() + 8, &stride, 8);
      for (unsigned d = 1; d < 4; ++d)
        memcpy(expected.data() + 8 + 8 * d, &inactive, 8);
      uint32_t dd[5] = {128, 160, 1, 1, 1};
      memcpy(expected.data() + 40, dd, 20);
      for (unsigned b = 0; b < 128; ++b)
        check(maps[c * 128 + b], expected[b], c * 128 + b);
      for (unsigned v = 0; v < by; ++v)
        for (unsigned u = 0; u < bx; ++u) {
          unsigned want = 0;
          if (remote && !(mask & (1 << (c / 2))))
            want = 43981;
          else if (output)
            want = 12345;
          else if (x + u < 128 && y + v < 160)
            want = data[(y + v) * 160 + x + u];
          unsigned at = c * 16384 + physical(2 * (v * bx + u), swizzle) / 2;
          check(got[at], want, at);
        }
    }
    if (rotate) {
      for (unsigned v = 0; v < by; ++v)
        for (unsigned u = 0; u < bx; ++u) {
          unsigned q = 2 * (v * bx + u);
          check(got[4096 + physical(q, swizzle, 9472) / 2], 999, 4096 + q / 2);
          check(got[8192 + physical(q, swizzle, 17664) / 2], data[v * 160 + u],
                8192 + q / 2);
        }
      uint64_t saved;
      memcpy(&saved, maps.data() + 128, 8);
      check(saved == reinterpret_cast<uint64_t>(second), 1, 128);
    }
    check(memcmp(&initial, original.data(), 128), 0, 0);
    if (output)
      for (unsigned which = 0; which < 2; ++which) {
        std::vector<unsigned short> dst(elements);
        CHECK(cudaMemcpy(dst.data(), which ? dest1 : dest0, elements * 2,
                         cudaMemcpyDeviceToHost));
        for (unsigned j = 0; j < 160; ++j)
          for (unsigned i = 0; i < 160; ++i) {
            bool inside =
                i < 128 && i >= x && i < x + bx && j >= y && j < y + by;
            check(dst[j * 160 + i],
                  inside ? (which ? 12345 : data[j * 160 + i]) : 0xeeee,
                  j * 160 + i);
          }
      }
    std::ofstream f(test + ".actual-" + std::to_string(rep), std::ios::binary);
    f.write(reinterpret_cast<char *>(got.data()), got.size() * 2);
    stamp("compare_end");
  }
  printf("%s %s errors=%u\n", errors ? "FAIL" : "PASS", test.c_str(), errors);
  return errors ? 2 : 0;
}
