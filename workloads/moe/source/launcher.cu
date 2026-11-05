// Standalone host launcher for fixed Megatron-TE forwards. Device computation
// comes from the exported framework PTX, not a replacement MoE implementation.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t status = (call);                            \
    if (status != cudaSuccess)                              \
      throw std::runtime_error(std::string(#call) + ": " +  \
                               cudaGetErrorString(status)); \
  } while (0)

extern "C" void **__cudaRegisterFatBinary(void *);
extern "C" void __cudaRegisterFatBinaryEnd(void **);
extern "C" void __cudaRegisterFunction(void **, const char *, char *,
                                       const char *, int, uint3 *, uint3 *,
                                       dim3 *, dim3 *, int *);
extern const unsigned char _binary_fatbins_bin_start[];
struct Relocation {
  unsigned argument, offset;
};
struct Module {
  size_t offset;
  const char *entry;
  const Relocation *relocations;
  size_t relocation_count;
};
// CUDA compiler registration ABI from fatbinary_section.h. A distinct type
// avoids redeclaring NVCC's own wrapper in its generated host stub.
struct FatBinary {
  uint32_t magic, version;
  const unsigned long long *data;
  void *filename;
};
#include "modules.inc"

struct Reader {
  std::ifstream stream;
  explicit Reader(const std::string &path) : stream(path, std::ios::binary) {
    if (!stream) throw std::runtime_error("Cannot read " + path);
  }
  template <class T>
  T read() {
    T value;
    stream.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!stream) throw std::runtime_error("Truncated forward description");
    return value;
  }
  std::vector<unsigned char> bytes(size_t count) {
    std::vector<unsigned char> value(count);
    stream.read(reinterpret_cast<char *>(value.data()), count);
    if (!stream) throw std::runtime_error("Truncated forward payload");
    return value;
  }
  std::vector<unsigned char> bytes() { return bytes(read<uint32_t>()); }
};

struct Buffer {
  uint64_t size;
  unsigned char *data = nullptr;
};
std::map<uint64_t, Buffer> buffers;

unsigned char *relocate(uint64_t address) {
  auto it = buffers.upper_bound(address);
  if (it == buffers.begin()) return nullptr;
  --it;
  if (address - it->first >= it->second.size) return nullptr;
  return it->second.data + (address - it->first);
}

struct Map {
  std::array<unsigned char, 128> original, encoded;
  uint64_t address;
  uint32_t type, rank;
  uint64_t dims[5]{}, strides[5]{};
  uint32_t box[5]{}, element[5]{}, flags[4]{};
};
std::vector<Map> tensor_maps;

void patch(std::vector<unsigned char> &data,
           const std::vector<size_t> &offsets) {
  std::vector<std::pair<size_t, size_t>> opaque;
  for (const auto &map : tensor_maps) {
    auto begin = data.begin();
    while (true) {
      auto found = std::search(begin, data.end(), map.original.begin(),
                               map.original.end());
      if (found == data.end()) break;
      size_t offset = found - data.begin();
      opaque.emplace_back(offset, offset + 128);
      std::copy(map.encoded.begin(), map.encoded.end(), found);
      begin = found + 128;
    }
  }
  for (size_t offset : offsets) {
    if (std::any_of(opaque.begin(), opaque.end(), [&](const auto &range) {
          return offset >= range.first && offset < range.second;
        }))
      continue;
    uint64_t original;
    std::memcpy(&original, data.data() + offset, 8);
    auto *pointer = relocate(original);
    if (pointer) std::memcpy(data.data() + offset, &pointer, 8);
  }
}

struct Action {
  uint32_t tag = 0;
  uint64_t address = 0, destination = 0, count = 0, function = 0;
  uint32_t launch[10]{}, value = 0, width = 0;
  std::vector<unsigned char> data;
  std::vector<size_t> pointer_offsets;
  std::vector<std::vector<unsigned char>> arguments;
  std::vector<void *> pointers;
};
struct Recipe {
  uint64_t address, count;
  uint32_t seed;
  float scale;
  uint64_t offset;
};

uint32_t mix_bits(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  return x ^ (x >> 16);
}

void initialize(const Recipe &recipe) {
  constexpr size_t chunk_size = 1 << 20;
  std::vector<__nv_bfloat16> values(
      std::min<uint64_t>(recipe.count, chunk_size));
  for (uint64_t start = 0; start < recipe.count; start += chunk_size) {
    uint64_t count = std::min<uint64_t>(chunk_size, recipe.count - start);
    for (uint64_t i = 0; i < count; ++i) {
      uint64_t index = recipe.offset + start + i;
      uint32_t key =
          uint32_t(index) ^ mix_bits(uint32_t(index >> 32)) ^ recipe.seed;
      float value = (int(mix_bits(key) % 17) - 8) * recipe.scale;
      values[i] = __float2bfloat16_rn(value);
    }
    CUDA_CHECK(cudaMemcpy(relocate(recipe.address) + start * 2, values.data(),
                          count * 2, cudaMemcpyHostToDevice));
  }
}

__global__ void setup_anchor(int *scratch) { *scratch = 0; }

int main(int argc, char **argv) {
  try {
    std::string input, output;
    int warmup = 0, iterations = 1, repeats = 1;
    bool simulation = false;
    for (int i = 1; i < argc; ++i) {
      std::string option = argv[i];
      if (option == "--simulation")
        simulation = true;
      else if (option == "--input")
        input = argv[++i];
      else if (option == "--output")
        output = argv[++i];
      else if (option == "--warmup")
        warmup = std::stoi(argv[++i]);
      else if (option == "--iterations")
        iterations = std::stoi(argv[++i]);
      else if (option == "--repeats")
        repeats = std::stoi(argv[++i]);
      else
        throw std::runtime_error("Unknown option " + option);
    }
    if (input.empty() || output.empty() || iterations < 1 || repeats < 1 ||
        warmup < 0)
      throw std::runtime_error(
          "Require --input and --output with positive iterations");
    if (simulation && (iterations != 1 || repeats != 1 || warmup != 0))
      throw std::runtime_error("Simulation uses one forward without warmup");
    Reader reader(input);
    if (reader.read<uint32_t>() != 0x4d4f4550 || reader.read<uint32_t>() != 3)
      throw std::runtime_error("Unsupported forward format");
    std::vector<Action> actions, constants;
    std::vector<Recipe> recipes;
    uint64_t output_address = 0, output_count = 0;
    while (reader.stream.peek() != EOF) {
      Action action;
      action.tag = reader.read<uint32_t>();
      if (action.tag == 1) {
        uint64_t address = reader.read<uint64_t>();
        uint64_t size = reader.read<uint64_t>();
        buffers[address].size = std::max(buffers[address].size, size);
      } else if (action.tag == 2) {
        reader.bytes();  // Provenance name; the numeric ID selects the module.
        action.function = reader.read<uint64_t>();
        for (auto &value : action.launch) value = reader.read<uint32_t>();
        uint32_t count = reader.read<uint32_t>();
        for (unsigned i = 0; i < count; ++i)
          action.arguments.push_back(reader.bytes());
        actions.push_back(std::move(action));
      } else if (action.tag == 3 || action.tag == 10) {
        action.address = reader.read<uint64_t>();
        action.data = reader.bytes();
        if (action.tag == 3) {
          unsigned count = reader.read<uint32_t>();
          for (unsigned i = 0; i < count; ++i)
            action.pointer_offsets.push_back(reader.read<uint32_t>());
        }
        (action.tag == 10 ? constants : actions).push_back(std::move(action));
      } else if (action.tag == 4) {
        action.address = reader.read<uint64_t>();
        action.count = reader.read<uint64_t>();
        action.data.resize(action.count);
        actions.push_back(std::move(action));
      } else if (action.tag == 5) {
        Map map{};
        auto data = reader.bytes(128);
        std::copy(data.begin(), data.end(), map.original.begin());
        map.address = reader.read<uint64_t>();
        map.type = reader.read<uint32_t>();
        map.rank = reader.read<uint32_t>();
        for (unsigned i = 0; i < map.rank; ++i)
          map.dims[i] = reader.read<uint64_t>();
        for (unsigned i = 0; i + 1 < map.rank; ++i)
          map.strides[i] = reader.read<uint64_t>();
        for (unsigned i = 0; i < map.rank; ++i)
          map.box[i] = reader.read<uint32_t>();
        for (unsigned i = 0; i < map.rank; ++i)
          map.element[i] = reader.read<uint32_t>();
        for (auto &flag : map.flags) flag = reader.read<uint32_t>();
        tensor_maps.push_back(map);
      } else if (action.tag == 6) {
        action.address = reader.read<uint64_t>();
        action.value = reader.read<uint32_t>();
        action.width = reader.read<uint32_t>();
        action.count = reader.read<uint64_t>();
        if (action.width != 1 && action.value != 0)
          throw std::runtime_error(
              "This forward requires nonzero word memset support");
        actions.push_back(std::move(action));
      } else if (action.tag == 7) {
        Recipe recipe;
        recipe.address = reader.read<uint64_t>();
        recipe.count = reader.read<uint64_t>();
        recipe.seed = reader.read<uint32_t>();
        recipe.scale = reader.read<float>();
        recipe.offset = reader.read<uint64_t>();
        recipes.push_back(recipe);
      } else if (action.tag == 8) {
        output_address = reader.read<uint64_t>();
        output_count = reader.read<uint64_t>();
      } else if (action.tag == 9) {
        action.destination = reader.read<uint64_t>();
        action.address = reader.read<uint64_t>();
        action.count = reader.read<uint64_t>();
        actions.push_back(std::move(action));
      } else
        throw std::runtime_error("Unknown forward record");
    }
    for (auto &entry : buffers) {
      CUDA_CHECK(cudaMalloc(&entry.second.data, entry.second.size));
      CUDA_CHECK(cudaMemset(entry.second.data, 0, entry.second.size));
    }
    for (const auto &recipe : recipes) initialize(recipe);
    for (auto &constant : constants) {
      CUDA_CHECK(cudaMemcpy(relocate(constant.address), constant.data.data(),
                            constant.data.size(), cudaMemcpyHostToDevice));
    }
    using Encode = CUresult (*)(
        CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *,
        const cuuint64_t *, const cuuint64_t *, const cuuint32_t *,
        const cuuint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle,
        CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    Encode encode = nullptr;
    if (!tensor_maps.empty()) {
      CUDA_CHECK(cudaGetDriverEntryPointByVersion(
          "cuTensorMapEncodeTiled", reinterpret_cast<void **>(&encode), 12000,
          cudaEnableDefault));
    }
    for (auto &map : tensor_maps) {
      alignas(64) CUtensorMap encoded;
      CUresult status = encode(
          &encoded, CUtensorMapDataType(map.type), map.rank,
          relocate(map.address), map.dims, map.strides, map.box, map.element,
          CUtensorMapInterleave(map.flags[0]), CUtensorMapSwizzle(map.flags[1]),
          CUtensorMapL2promotion(map.flags[2]),
          CUtensorMapFloatOOBfill(map.flags[3]));
      if (status != CUDA_SUCCESS)
        throw std::runtime_error("Tensor map encoding failed");
      std::memcpy(map.encoded.data(), &encoded, 128);
    }
    std::vector<char> markers(std::size(modules));
    std::vector<FatBinary> wrappers(std::size(modules));
    for (size_t i = 0; i < std::size(modules); ++i) {
      wrappers[i] = {0x466243b1, 1,
                     reinterpret_cast<const unsigned long long *>(
                         _binary_fatbins_bin_start + modules[i].offset),
                     nullptr};
      void **handle = __cudaRegisterFatBinary(&wrappers[i]);
      __cudaRegisterFunction(
          handle, &markers[i], const_cast<char *>(modules[i].entry),
          modules[i].entry, -1, nullptr, nullptr, nullptr, nullptr, nullptr);
      __cudaRegisterFatBinaryEnd(handle);
    }
    for (auto &action : actions) {
      if (action.tag == 2) {
        const auto &module = modules[action.function];
        for (size_t index = 0; index < action.arguments.size(); ++index) {
          auto &argument = action.arguments[index];
          std::vector<size_t> offsets;
          for (size_t i = 0; i < module.relocation_count; ++i)
            if (module.relocations[i].argument == index)
              offsets.push_back(module.relocations[i].offset);
          patch(argument, offsets);
          action.pointers.push_back(argument.data());
        }
        if (action.launch[9])
          CUDA_CHECK(cudaFuncSetAttribute(
              &markers[action.function],
              cudaFuncAttributeMaxDynamicSharedMemorySize, action.launch[9]));
      } else if (action.tag == 3) {
        patch(action.data, action.pointer_offsets);
      }
    }
    const bool trace = std::getenv("MOE_TRACE_LAUNCHES") != nullptr;
    auto forward = [&]() {
      for (auto &action : actions) {
        if (action.tag == 2) {
          if (trace)
            std::cerr << "Launching " << action.function << " "
                      << modules[action.function].entry << "\n";
          auto *p = action.launch;
          dim3 grid(p[0], p[1], p[2]), block(p[3], p[4], p[5]);
          if (p[6] * p[7] * p[8] > 1) {
            cudaLaunchAttribute attr{};
            attr.id = cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim = {p[6], p[7], p[8]};
            cudaLaunchConfig_t config{};
            config.gridDim = grid;
            config.blockDim = block;
            config.dynamicSmemBytes = p[9];
            config.attrs = &attr;
            config.numAttrs = 1;
            CUDA_CHECK(cudaLaunchKernelExC(&config, &markers[action.function],
                                           action.pointers.data()));
          } else {
            CUDA_CHECK(cudaLaunchKernel(&markers[action.function], grid, block,
                                        action.pointers.data(), p[9], nullptr));
          }
        } else if (action.tag == 3) {
          CUDA_CHECK(cudaMemcpyAsync(relocate(action.address),
                                     action.data.data(), action.data.size(),
                                     cudaMemcpyHostToDevice));
        } else if (action.tag == 4) {
          CUDA_CHECK(cudaMemcpy(action.data.data(), relocate(action.address),
                                action.count, cudaMemcpyDeviceToHost));
        } else if (action.tag == 6) {
          CUDA_CHECK(cudaMemsetAsync(relocate(action.address), action.value,
                                     action.count * action.width));
        } else if (action.tag == 9) {
          CUDA_CHECK(cudaMemcpyAsync(relocate(action.destination),
                                     relocate(action.address), action.count,
                                     cudaMemcpyDeviceToDevice));
        }
        if (trace) CUDA_CHECK(cudaDeviceSynchronize());
      }
    };
    int *scratch;
    CUDA_CHECK(cudaMalloc(&scratch, sizeof(int)));
    setup_anchor<<<1, 1>>>(scratch);
    for (int i = 0; i < warmup; ++i) forward();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t begin{}, end{};
    if (!simulation) {
      CUDA_CHECK(cudaEventCreate(&begin));
      CUDA_CHECK(cudaEventCreate(&end));
    }
    std::vector<float> samples;
    for (int repeat = 0; repeat < repeats; ++repeat) {
      std::cout << "MOE_ROI_BEGIN\n" << std::flush;
      if (!simulation) CUDA_CHECK(cudaEventRecord(begin));
      for (int i = 0; i < iterations; ++i) forward();
      if (!simulation) CUDA_CHECK(cudaEventRecord(end));
      CUDA_CHECK(cudaDeviceSynchronize());
      std::cout << "MOE_ROI_END\n" << std::flush;
      if (!simulation) {
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
        samples.push_back(milliseconds / iterations);
      }
    }
    std::vector<__nv_bfloat16> bf16(output_count);
    CUDA_CHECK(cudaMemcpy(bf16.data(), relocate(output_address),
                          output_count * 2, cudaMemcpyDeviceToHost));
    std::vector<float> values(output_count);
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = __bfloat162float(bf16[i]);
    std::ofstream file(output + ".f32", std::ios::binary);
    file.write(reinterpret_cast<char *>(values.data()), values.size() * 4);
    std::ofstream result(output + ".json");
    result << "{\"cuda_event_ms\":";
    if (simulation)
      result << "null";
    else {
      auto sorted = samples;
      std::sort(sorted.begin(), sorted.end());
      result << (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) /
                    2;
    }
    result << ",\"cuda_event_ms_samples\":[";
    for (size_t i = 0; i < samples.size(); ++i)
      result << (i ? "," : "") << samples[i];
    result << "]";
    result << ",\"iterations\":" << iterations << ",\"warmup\":" << warmup
           << ",\"repeats\":" << repeats << ",\"framework_kernels\":true}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << "\n";
    return 1;
  }
}
