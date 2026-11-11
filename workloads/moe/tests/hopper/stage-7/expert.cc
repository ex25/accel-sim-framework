#include "../stage-1/host.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <string>

extern "C" const unsigned char expert_module[];
static char marker;
struct Problem {
  int m, n, k;
};
struct MapSpec {
  unsigned type, rank;
  uint64_t dims[2], stride;
  unsigned box[2], element[2], flags[4];
};
template <class T> static T *device(const std::vector<T> &v) {
  T *p;
  CHECK(cudaMalloc(&p, v.size() * sizeof(T)));
  CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
static uint16_t bf16(float f) {
  uint32_t b;
  memcpy(&b, &f, 4);
  return (b + 0x7fff + ((b >> 16) & 1)) >> 16;
}
static int av(unsigned e, int m, int k) { return int((e + 2 * m + k) % 3) - 1; }
static int bv(unsigned e, int n, int k) {
  return int((2 * e + n + 3 * k) % 5) - 2;
}

int main(int argc, char **argv) {
  const std::string test = argv[1];
  const bool real = std::string(argv[2]) == "real";
  const unsigned tile_rounds = argc > 3 ? std::stoul(argv[3]) : 1;
  const int active_group = argc > 4 ? std::stoi(argv[4]) : -1;
  stamp("main");
  Wrapper wrapper{0x466243b1, 1, expert_module, nullptr};
  auto handle = __cudaRegisterFatBinary(&wrapper);
  const char *entry = (test == "loading" || test == "streams") ? "loading"
                      : test == "fp32"                         ? "fp32"
                      : test == "normalize" ? "moe_kernel_45ff4fce81e9bcc8"
                                            : "moe_kernel_df3e0c49e4650c1f";
  __cudaRegisterFunction(handle, &marker, const_cast<char *>(entry), entry, -1,
                         nullptr, nullptr, nullptr, nullptr, nullptr);
  __cudaRegisterFatBinaryEnd(handle);
  stamp("registered");
  if (test == "fp32") {
    std::vector<float> result(128 * 16);
    auto out = device(result);
    void *args[] = {&out};
    CHECK(cudaLaunchKernel(&marker, dim3(1), dim3(128), args, 0, nullptr));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(result.data(), out, result.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    for (float value : result)
      if (value != 128)
        return 1;
    puts("PASS expert fp32 independent FMA accumulators elements=2048");
    return 0;
  }
  if (test == "normalize") {
    std::vector<int64_t> selected(512);
    std::vector<float> weights(512), expected(128 * 64, 0);
    std::vector<int> mask_expected(128 * 64, 0);
    const float pattern[] = {1, 2, 1, 4};
    for (unsigned t = 0; t < 128; ++t)
      for (unsigned j = 0; j < 4; ++j) {
        unsigned e = (t + 13 * j) % 64;
        selected[t * 4 + j] = e;
        weights[t * 4 + j] = pattern[j];
        expected[t * 64 + e] = pattern[j] / 8;
        mask_expected[t * 64 + e] = 1;
      }
    auto indices = device(selected);
    auto values = device(weights);
    auto output = device(std::vector<float>(128 * 64, 0));
    auto mask = device(std::vector<int>(128 * 64, 0));
    unsigned tokens = 128;
    void *args[] = {&indices, &values, &output, &mask, &tokens};
    CHECK(cudaLaunchKernel(&marker, dim3(4), dim3(128), args, 0, nullptr));
    CHECK(cudaDeviceSynchronize());
    std::vector<float> actual(expected.size());
    std::vector<int> actual_mask(mask_expected.size());
    CHECK(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(actual_mask.data(), mask, actual_mask.size() * sizeof(int),
                     cudaMemcpyDeviceToHost));
    if (actual != expected || actual_mask != mask_expected)
      return 1;
    puts("PASS expert normalize original module75 weights=8192 mask=8192");
    return 0;
  }
  if (test == "loading" || test == "streams") {
    std::vector<unsigned> result(32);
    auto out = device(result);
    void *args[] = {&out};
    cudaStream_t streams[3] = {};
    if (test == "streams") {
      CHECK(cudaStreamCreate(&streams[1]));
      CHECK(cudaStreamCreate(&streams[2]));
    }
    const unsigned order[] = {0, 1, 2, 1, 0, 2};
    for (unsigned i = 0; i < (test == "streams" ? 6u : 1u); ++i) {
      printf("STREAM_CHECK launch=%u slot=%u\n", i, order[i]);
      CHECK(cudaLaunchKernel(&marker, dim3(1), dim3(32), args, 0,
                             streams[order[i]]));
      CHECK(cudaDeviceSynchronize());
      CHECK(cudaMemcpy(result.data(), out, 128, cudaMemcpyDeviceToHost));
      for (unsigned t = 0; t < 32; ++t) {
        if (result[t] != 3 * t + 7 + 0x3340)
          return 1;
      }
    }
    if (test == "streams") {
      CHECK(cudaStreamDestroy(streams[1]));
      CHECK(cudaStreamDestroy(streams[2]));
    }
    puts("PASS expert loading scoped labels hex and decimal U (all launches)");
    return 0;
  }
  auto raw = read((test + ".param.raw").c_str());
  alignas(64) std::array<unsigned char, 1344> param;
  memcpy(param.data(), raw.data(), param.size());
  auto set = [&](unsigned off, auto value) {
    memcpy(param.data() + off, &value, sizeof(value));
  };
  auto shape_bytes = read((test + ".shapes").c_str());
  std::vector<Problem> shapes(shape_bytes.size() / sizeof(Problem));
  memcpy(shapes.data(), shape_bytes.data(), shape_bytes.size());
  std::vector<uint16_t *> ap, bp, cp;
  std::vector<uint64_t> as, bs, cs;
  std::vector<std::vector<uint16_t>> expected;
  for (unsigned e = 0; e < shapes.size(); ++e) {
    auto &s = shapes[e];
    if (!real) {
      s.n = (test == "fc1" ? 256 : 128) * tile_rounds;
      s.k = test == "fc1" ? 128 : 256;
    }
    std::vector<uint16_t> a(s.m * s.k), b(s.n * s.k),
        c(s.m * s.n + 128, 0x7f7f);
    for (int m = 0; m < s.m; ++m)
      for (int k = 0; k < s.k; ++k)
        a[m * s.k + k] = bf16(av(e, m, k));
    for (int n = 0; n < s.n; ++n)
      for (int k = 0; k < s.k; ++k)
        b[n * s.k + k] = bf16(bv(e, n, k));
    std::vector<uint16_t> ref(s.m * s.n, 0x7f7f);
    if (active_group < 0 || int(e) == active_group) {
      for (int m = 0; m < s.m; ++m)
        for (int n = 0; n < s.n; ++n) {
          int sum = 0;
          for (int k = 0; k < s.k; ++k)
            sum += av(e, m, k) * bv(e, n, k);
          ref[m * s.n + n] = bf16(sum);
        }
    }
    expected.push_back(std::move(ref));
    ap.push_back(device(a));
    bp.push_back(device(b));
    cp.push_back(device(c));
    as.push_back(s.k);
    bs.push_back(s.k);
    cs.push_back(s.n);
  }
  // Keep real N/K and the full group table, but restrict this diagnostic's
  // work. Inactive outputs retain their original allocation and sentinel.
  if (active_group >= 0)
    for (unsigned e = 0; e < shapes.size(); ++e)
      if (int(e) != active_group) shapes[e].m = 0;
  auto problem = device(shapes);
  uint64_t tiles = 0;
  for (const auto &s : shapes)
    tiles += ((s.m + 127) / 128) * ((s.n + 255) / 256) * 2;
  set(1208, tiles);
  set(16, problem);
  set(1240, problem);
  set(464, device(ap));
  set(480, device(bp));
  set(472, device(as));
  set(488, device(bs));
  set(1040, device(cs));
  set(1048, device(cp));
  set(1056, device(cs));
  std::vector<unsigned char> workspace(132 * 4 * 128);
  auto work = device(workspace);
  set(456, work);
  set(1024, work + 132 * 2 * 128);
  set(1280, work + 132 * 2 * 128);
  using Encode = CUresult (*)(CUtensorMap *, CUtensorMapDataType, cuuint32_t,
                              void *, const cuuint64_t *, const cuuint64_t *,
                              const cuuint32_t *, const cuuint32_t *,
                              CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  Encode encode = nullptr;
  CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",
                                         reinterpret_cast<void **>(&encode),
                                         12000, cudaEnableDefault));
  auto map_bytes = read("maps.raw");
  const unsigned offsets[] = {64, 256, 640, 832};
  for (unsigned i = 0; i < 4; ++i) {
    MapSpec spec;
    memcpy(&spec, map_bytes.data() + i * sizeof(spec), sizeof(spec));
    alignas(64) CUtensorMap map;
    auto result = encode(&map, CUtensorMapDataType(spec.type), 2, nullptr,
                         spec.dims, &spec.stride, spec.box, spec.element,
                         CUtensorMapInterleave(spec.flags[0]),
                         CUtensorMapSwizzle(spec.flags[1]),
                         CUtensorMapL2promotion(spec.flags[2]),
                         CUtensorMapFloatOOBfill(spec.flags[3]));
    if (result != CUDA_SUCCESS)
      return 2;
    memcpy(param.data() + offsets[i], &map, 128);
  }
  std::ofstream params(test + ".param.actual", std::ios::binary);
  params.write(reinterpret_cast<char *>(param.data()), param.size());
  params.close();
  std::ofstream problems(test + ".problems.actual", std::ios::binary);
  problems.write(reinterpret_cast<char *>(shapes.data()),
                 shapes.size() * sizeof(Problem));
  problems.close();
  CHECK(cudaFuncSetAttribute(
      &marker, cudaFuncAttributeMaxDynamicSharedMemorySize, 215040));
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {1, 2, 1};
  cudaLaunchConfig_t config{};
  config.gridDim = {66, 2, 1};
  config.blockDim = {384, 1, 1};
  config.dynamicSmemBytes = 215040;
  config.attrs = &attr;
  config.numAttrs = 1;
  void *args[] = {param.data()};
  stamp("simulate_begin");
  CHECK(cudaLaunchKernelExC(&config, &marker, args));
  CHECK(cudaDeviceSynchronize());
  stamp("simulate_end");
  size_t count = 0, active_count = 0, failed = 0, guard_failed = 0;
  std::ofstream actual(test + ".actual", std::ios::binary),
      ref(test + ".expected", std::ios::binary);
  for (unsigned group = 0; group < shapes.size(); ++group) {
    const unsigned e = group;
    std::vector<uint16_t> c(expected[e].size() + 128);
    CHECK(cudaMemcpy(c.data(), cp.at(group), c.size() * 2,
                     cudaMemcpyDeviceToHost));
    for (unsigned j = 0; j < expected[e].size(); ++j) {
      if (c[j] != expected[e][j]) {
        if (failed < 10)
          printf("MISMATCH e=%u j=%u actual=%04x expected=%04x\n", e, j, c[j],
                 expected[e][j]);
        ++failed;
      }
    }
    for (unsigned j = expected[e].size(); j < c.size(); ++j)
      guard_failed += c[j] != 0x7f7f;
    count += expected[e].size();
    if (active_group < 0 || int(e) == active_group)
      active_count += expected[e].size();
    actual.write(reinterpret_cast<char *>(c.data()), expected[e].size() * 2);
    ref.write(reinterpret_cast<char *>(expected[e].data()),
              expected[e].size() * 2);
  }
  stamp("compare_end");
  printf("%s expert %s real=%u elements=%zu active_group=%d "
         "active_elements=%zu inactive_elements=%zu failed=%zu "
         "guard_failed=%zu\n",
         failed || guard_failed ? "FAIL" : "PASS", test.c_str(), real, count,
         active_group, active_count, count - active_count, failed, guard_failed);
  return failed || guard_failed ? 1 : 0;
}
