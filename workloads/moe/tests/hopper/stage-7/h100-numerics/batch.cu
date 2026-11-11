// Isolated frozen kernels; no changes to the simulator or workload artifacts.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    auto status = (call);                                                      \
    if (status != cudaSuccess)                                                 \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               cudaGetErrorString(status));                    \
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
#include "modules.inc"
struct Case {
  const char *name;
  unsigned h, width, experts, topk, rows, activation, normalize;
  int exponent;
  unsigned normalize_shape[10], exponent_shape[10];
};
#include "cases.inc"
static std::map<std::string, char *> functions;
static std::array<char, 40> markers;
static std::vector<Wrapper> wrappers;
static double started;
static double now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
static void stamp(const std::string &s) {
  printf("BATCH %.6f %s\n", now() - started, s.c_str());
  fflush(stdout);
}
static void register_module(const char *key, const char *entry,
                            const unsigned char *data) {
  wrappers.push_back({0x466243b1, 1, data, nullptr});
  auto handle = __cudaRegisterFatBinary(&wrappers.back());
  char *marker = &markers[functions.size()];
  __cudaRegisterFunction(handle, marker, const_cast<char *>(entry), entry, -1,
                         nullptr, nullptr, nullptr, nullptr, nullptr);
  __cudaRegisterFatBinaryEnd(handle);
  functions[key] = marker;
}
template <class T> static std::vector<T> read(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("Cannot read " + path);
  auto bytes = file.tellg();
  if (size_t(bytes) % sizeof(T))
    throw std::runtime_error("Invalid size " + path);
  std::vector<T> result(size_t(bytes) / sizeof(T));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(result.data()), bytes);
  return result;
}
template <class T> static T *allocate(size_t count) {
  T *result;
  CHECK(cudaMalloc(&result, count * sizeof(T)));
  return result;
}
template <class T> static T *upload(const std::vector<T> &data) {
  auto result = allocate<T>(data.size());
  CHECK(cudaMemcpy(result, data.data(), data.size() * sizeof(T),
                   cudaMemcpyHostToDevice));
  return result;
}
template <class T>
static void save(const std::string &path, T *device, size_t count) {
  std::vector<T> result(count);
  CHECK(cudaMemcpy(result.data(), device, count * sizeof(T),
                   cudaMemcpyDeviceToHost));
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<char *>(result.data()), count * sizeof(T));
  if (!file)
    throw std::runtime_error("Cannot write " + path);
}
static void launch(const std::string &key, unsigned grid, unsigned block,
                   void **args, unsigned smem = 0) {
  CHECK(cudaLaunchKernel(functions.at(key), dim3(grid), dim3(block), args, smem,
                         nullptr));
  CHECK(cudaDeviceSynchronize());
}
__device__ unsigned mix(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  return x ^ (x >> 16);
}
__global__ void initialize(__nv_bfloat16 *out, size_t count, unsigned seed,
                           float scale, uint64_t offset) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    uint64_t index = offset + i;
    unsigned key = unsigned(index) ^ mix(unsigned(index >> 32)) ^ seed;
    out[i] = __float2bfloat16_rn((int(mix(key) % 17) - 8) * scale);
  }
}
static void fill(__nv_bfloat16 *out, size_t count, unsigned seed, float scale,
                 uint64_t offset) {
  initialize<<<1024, 256>>>(out, count, seed, scale, offset);
  CHECK(cudaGetLastError());
}
__global__ void make_pairs(const __nv_bfloat16 *fc1, float2 *pairs,
                           unsigned count, unsigned width) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    unsigned row = i / width, col = i % width;
    pairs[i] =
        make_float2(__bfloat162float(fc1[row * 2 * width + col]),
                    __bfloat162float(fc1[row * 2 * width + width + col]));
  }
}
struct Problem {
  int m, n, k;
};
struct MapSpec {
  unsigned type, rank;
  uint64_t dims[2], stride;
  unsigned box[2], element[2], flags[4];
};
static void grouped(const Case &c, unsigned fc, const std::string &mode,
                    __nv_bfloat16 *a, __nv_bfloat16 *out,
                    const std::vector<int> &rows, const std::string &prefix) {
  const unsigned n = fc == 1 ? 2 * c.width : c.h;
  const unsigned k = fc == 1 ? c.h : c.width;
  auto raw =
      read<unsigned char>(prefix + ".fc" + std::to_string(fc) + ".param");
  alignas(64) std::array<unsigned char, 1344> param;
  if (raw.size() != param.size())
    throw std::runtime_error("Parameter size");
  memcpy(param.data(), raw.data(), raw.size());
  auto set = [&](unsigned off, auto value) {
    memcpy(param.data() + off, &value, sizeof(value));
  };
  std::vector<Problem> shapes;
  std::vector<__nv_bfloat16 *> ap, bp, cp;
  std::vector<uint64_t> as, bs, cs;
  unsigned start = 0;
  uint64_t tiles = 0;
  while (start < c.rows) {
    unsigned expert = rows[start * 2 + 1], end = start + 1;
    while (end < c.rows && unsigned(rows[end * 2 + 1]) == expert)
      ++end;
    shapes.push_back({int(end - start), int(n), int(k)});
    ap.push_back(a + size_t(start) * k);
    cp.push_back(out + size_t(start) * n);
    auto weight = allocate<__nv_bfloat16>(size_t(n) * k);
    fill(weight, size_t(n) * k, fc == 1 ? 2028 : 2029, 1.f / 256,
         uint64_t(expert) * n * k);
    bp.push_back(weight);
    as.push_back(k);
    bs.push_back(k);
    cs.push_back(n);
    tiles += ((end - start + 127) / 128) * ((n + 255) / 256) * 2;
    start = end;
  }
  auto problem = upload(shapes);
  auto dap = upload(ap), dbp = upload(bp), dcp = upload(cp);
  auto das = upload(as), dbs = upload(bs), dcs = upload(cs);
  auto work = allocate<unsigned char>(132 * 4 * 128);
  CHECK(cudaMemset(work, 0, 132 * 4 * 128));
  CHECK(cudaMemset(out, 0xad, size_t(c.rows) * n * 2));
  set(16, problem);
  set(1240, problem);
  set(456, work);
  set(464, dap);
  set(472, das);
  set(480, dbp);
  set(488, dbs);
  set(1024, work + 132 * 2 * 128);
  set(1032, dcp);
  set(1040, dcs);
  set(1048, dcp);
  set(1056, dcs);
  set(1208, tiles);
  set(1232, unsigned(shapes.size()));
  set(1280, work + 132 * 2 * 128);
  auto maps = read<MapSpec>(prefix + ".fc" + std::to_string(fc) + ".maps");
  using Encode = CUresult (*)(CUtensorMap *, CUtensorMapDataType, cuuint32_t,
                              void *, const cuuint64_t *, const cuuint64_t *,
                              const cuuint32_t *, const cuuint32_t *,
                              CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  Encode encode = nullptr;
  CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",
                                         reinterpret_cast<void **>(&encode),
                                         12000, cudaEnableDefault));
  const unsigned offsets[] = {64, 256, 640, 832};
  for (unsigned i = 0; i < 4; ++i) {
    const auto &spec = maps.at(i);
    alignas(64) CUtensorMap map;
    auto status = encode(&map, CUtensorMapDataType(spec.type), 2, nullptr,
                         spec.dims, &spec.stride, spec.box, spec.element,
                         CUtensorMapInterleave(spec.flags[0]),
                         CUtensorMapSwizzle(spec.flags[1]),
                         CUtensorMapL2promotion(spec.flags[2]),
                         CUtensorMapFloatOOBfill(spec.flags[3]));
    if (status != CUDA_SUCCESS)
      throw std::runtime_error("TMA encoding");
    memcpy(param.data() + offsets[i], &map, 128);
  }
  auto marker = functions.at("16." + mode);
  CHECK(cudaFuncSetAttribute(
      marker, cudaFuncAttributeMaxDynamicSharedMemorySize, 215040));
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
  CHECK(cudaLaunchKernelExC(&config, marker, args));
  CHECK(cudaDeviceSynchronize());
  for (auto p : bp)
    CHECK(cudaFree(p));
  CHECK(cudaFree(problem));
  CHECK(cudaFree(dap));
  CHECK(cudaFree(dbp));
  CHECK(cudaFree(dcp));
  CHECK(cudaFree(das));
  CHECK(cudaFree(dbs));
  CHECK(cudaFree(dcs));
  CHECK(cudaFree(work));
}
static void case_run(const Case &c, const std::string &mode,
                     const std::string &dest, bool activation_only) {
  const std::string prefix = c.name;
  auto fc1 = upload(read<__nv_bfloat16>(prefix + ".fc1.bf16"));
  auto cpuact = upload(read<__nv_bfloat16>(prefix + ".activation.bf16"));
  auto act = allocate<__nv_bfloat16>(size_t(c.rows) * c.width);
  auto rows = read<int>(prefix + ".rows.i32");
  const std::string out =
      dest + "/" + prefix + "." + mode +
      (activation_only && mode != "a800" ? ".cpuinput" : "");
  if (!activation_only) {
    auto x = allocate<__nv_bfloat16>(size_t(c.rows) * c.h);
    for (unsigned row = 0; row < c.rows; ++row)
      fill(x + size_t(row) * c.h, c.h, 2026, 1.f / 16,
           uint64_t(rows[row * 2]) * c.h);
    grouped(c, 1, mode, x, fc1, rows, prefix);
    save(out + ".fc1.bf16", fc1, size_t(c.rows) * 2 * c.width);
    CHECK(cudaFree(x));
    stamp(prefix + " " + mode + " FC1 saved");
  }
  unsigned count = c.rows * c.width;
  if (count % 1024)
    throw std::runtime_error("Activation tile size");
  void *actargs[] = {&fc1, &act, &count};
  launch(std::to_string(c.activation) + "." + mode, count / 1024, 128, actargs);
  save(out + ".activation.bf16", act, count);
  auto pairs = allocate<float2>(count);
  make_pairs<<<(count + 255) / 256, 256>>>(fc1, pairs, count, c.width);
  CHECK(cudaGetLastError());
  auto stages = allocate<unsigned>(size_t(count) * 8);
  void *probeargs[] = {&pairs, &stages};
  launch("probe." + mode, count / 32, 32, probeargs);
  save(out + ".stages.u32", stages, size_t(count) * 8);
  CHECK(cudaFree(stages));
  CHECK(cudaFree(pairs));
  auto indices = upload(read<int64_t>(prefix + ".selected.i64"));
  auto scores = upload(read<float>(prefix + ".scores.f32"));
  auto weights = allocate<float>(128 * c.experts);
  auto mask = allocate<int>(128 * c.experts);
  CHECK(cudaMemset(weights, 0, 128 * c.experts * sizeof(float)));
  CHECK(cudaMemset(mask, 0, 128 * c.experts * sizeof(int)));
  unsigned tokens = 128;
  if (c.exponent >= 0) {
    auto exponents = allocate<float>(128 * c.topk);
    void *args[] = {&scores, &exponents, &tokens};
    launch(std::to_string(c.exponent) + "." + mode, c.exponent_shape[0],
           c.exponent_shape[3], args, c.exponent_shape[9]);
    CHECK(cudaFree(scores));
    scores = exponents;
  }
  void *routeargs[] = {&indices, &scores, &weights, &mask, &tokens};
  launch(std::to_string(c.normalize) + "." + mode, c.normalize_shape[0],
         c.normalize_shape[3], routeargs, c.normalize_shape[9]);
  save(out + ".weights.f32", weights, 128 * c.experts);
  save(out + ".mask.i32", mask, 128 * c.experts);
  CHECK(cudaFree(indices));
  CHECK(cudaFree(scores));
  CHECK(cudaFree(weights));
  CHECK(cudaFree(mask));
  stamp(prefix + " " + mode + " activation/probe/routing saved");
  if (!activation_only) {
    auto result = allocate<__nv_bfloat16>(size_t(c.rows) * c.h);
    grouped(c, 2, mode, cpuact, result, rows, prefix);
    save(out + ".fc2-cpuactivation.bf16", result, size_t(c.rows) * c.h);
    grouped(c, 2, mode, act, result, rows, prefix);
    save(out + ".fc2-nativeactivation.bf16", result, size_t(c.rows) * c.h);
    CHECK(cudaFree(result));
    stamp(prefix + " " + mode + " FC2 variants saved");
  }
  CHECK(cudaFree(fc1));
  CHECK(cudaFree(cpuact));
  CHECK(cudaFree(act));
}
int main(int argc, char **argv) {
  try {
    started = now();
    if (argc != 3)
      throw std::runtime_error("Usage: batch OUTPUT a800|h100");
    bool activation_only = !strcmp(argv[2], "a800");
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    if (!activation_only && prop.major != 9)
      throw std::runtime_error("Requires H100 compute capability 9.0");
    int runtime = 0, driver = 0;
    CHECK(cudaRuntimeGetVersion(&runtime));
    CHECK(cudaDriverGetVersion(&driver));
    printf("DEVICE %s cc=%d.%d runtime=%d driver=%d\n", prop.name, prop.major,
           prop.minor, runtime, driver);
    wrappers.reserve(40);
#define REGISTER(key, entry, data) register_module(key, entry, data)
#include "register.inc"
    auto input = read<float2>("pairs.f32");
    auto pairs = upload(input);
    auto stages = allocate<unsigned>(input.size() * 8);
    auto opsinput = read<float2>("ops.f32");
    auto opairs = upload(opsinput);
    auto ops = allocate<unsigned>(opsinput.size() * 5);
    std::vector<std::string> modes =
        activation_only ? std::vector<std::string>{"a800"}
                        : std::vector<std::string>{"jit", "cubin"};
    bool failed = false;
    for (const auto &mode : modes) {
      void *args[] = {&pairs, &stages};
      launch("probe." + mode, input.size() / 32, 32, args);
      save(std::string(argv[1]) + "/probe." + mode + ".u32", stages,
           input.size() * 8);
      void *oargs[] = {&opairs, &ops};
      launch("ops." + mode, opsinput.size() / 32, 32, oargs);
      save(std::string(argv[1]) + "/ops." + mode + ".u32", ops,
           opsinput.size() * 5);
      stamp(mode + " unique-pair and math sweeps saved");
      // Collect the A800-prechecked measurements first, before H100-only
      // grouped GEMM. An ABI failure must not lose the math evidence.
      for (const auto &c : cases)
        case_run(c, mode, argv[1], true);
    }
    if (!activation_only) {
      for (const auto &mode : modes) {
        for (const auto &c : cases) {
          try {
            case_run(c, mode, argv[1], false);
          } catch (const std::exception &error) {
            failed = true;
            fprintf(stderr, "FAILED_GROUP %s %s: %s\n", c.name, mode.c_str(),
                    error.what());
          }
        }
      }
    }
    stamp(failed ? "PARTIAL_FAILURE" : "COMPLETE");
    return failed ? 1 : 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
