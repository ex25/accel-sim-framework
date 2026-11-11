#include "../../stage-1/host.h"
extern "C" const unsigned char activation_module[], ops_module[],
    matrix_module[];

int main(int argc, char **argv) {
  std::string mode = argv[1];
  bool matrix = mode == "single128" || mode == "chain";
  const unsigned char *module = matrix          ? matrix_module
                                : mode == "ops" ? ops_module
                                                : activation_module;
  std::string entry = matrix          ? "test_" + mode
                      : mode == "ops" ? "ops_probe"
                                      : "activation_probe";
  Wrapper wrapper{0x466243b1, 1, module, nullptr};
  auto handle = __cudaRegisterFatBinary(&wrapper);
  static char marker;
  __cudaRegisterFunction(handle, &marker, const_cast<char *>(entry.c_str()),
                         entry.c_str(), -1, nullptr, nullptr, nullptr, nullptr,
                         nullptr);
  __cudaRegisterFatBinaryEnd(handle);
  auto input = read((mode + ".input").c_str());
  auto expected = read((mode + ".expected").c_str());
  void *in, *out, *unused = nullptr;
  CHECK(cudaMalloc(&in, input.size()));
  CHECK(cudaMalloc(&out, expected.size()));
  CHECK(cudaMemcpy(in, input.data(), input.size(), cudaMemcpyHostToDevice));
  unsigned shared = matrix ? 49152 : 0;
  if (matrix)
    CHECK(cudaFuncSetAttribute(
        &marker, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  void *args[] = {&in, &out, &unused, &unused, &unused};
  CHECK(cudaLaunchKernel(&marker, dim3(1), dim3(matrix ? 128 : 32), args,
                         shared));
  CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> actual(expected.size() / 4);
  CHECK(
      cudaMemcpy(actual.data(), out, expected.size(), cudaMemcpyDeviceToHost));
  std::ofstream file(mode + ".actual", std::ios::binary);
  file.write(reinterpret_cast<const char *>(actual.data()), expected.size());
  unsigned errors = 0;
  for (size_t i = 0; i < actual.size(); ++i)
    errors +=
        actual[i] != reinterpret_cast<const uint32_t *>(expected.data())[i];
  printf("%s %s elements=%zu errors=%u\n", errors ? "FAIL" : "PASS",
         mode.c_str(), actual.size(), errors);
  return errors ? 2 : 0;
}
