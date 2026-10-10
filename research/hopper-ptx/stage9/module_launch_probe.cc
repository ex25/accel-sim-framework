#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

static int checks = 0, failures = 0;
static void expect(const char *label, CUresult got, CUresult want) {
  const char *name = "<unset>";
  cuGetErrorName(got, &name);
  ++checks;
  if (got != want) ++failures;
  std::printf("RUNTIME_CHECK label=%s got=%d name=%s expected=%d\n", label,
              int(got), name, int(want));
}
static void state(const char *label, bool valid) {
  ++checks;
  failures += !valid;
  std::printf("RUNTIME_STATE label=%s status=%s\n", label, valid ? "PASS" : "FAIL");
}
int main(int argc, char **argv) {
  if (argc != 3) return 2;
  std::ifstream input(argv[1]);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  std::string text = buffer.str();
  if (text.empty()) return 2;
  CUmodule sentinel = (CUmodule)uintptr_t(0x1234), module = sentinel;
  expect("before_init", cuModuleLoadData(&module, text.c_str()), CUDA_ERROR_NOT_INITIALIZED);
  state("failure_preserves_output", module == sentinel);
  expect("bad_init", cuInit(1), CUDA_ERROR_INVALID_VALUE);
  expect("init", cuInit(0), CUDA_SUCCESS);
  expect("null_output", cuModuleLoadData(nullptr, text.c_str()), CUDA_ERROR_INVALID_VALUE);
  expect("null_image", cuModuleLoadData(&module, nullptr), CUDA_ERROR_INVALID_VALUE);
  expect("missing_file", cuModuleLoad(&module, "/nonexistent/hopper-stage9.ptx"), CUDA_ERROR_FILE_NOT_FOUND);
  expect("invalid_header", cuModuleLoadData(&module, ".version 8.0\n"), CUDA_ERROR_INVALID_PTX);
  expect("invalid_syntax", cuModuleLoadData(&module,
      ".version 8.0\n.target sm_90a\n.address_size 64\n.visible .entry broken() { ??? }\n"), CUDA_ERROR_INVALID_PTX);
  expect("static_global_rejected", cuModuleLoadData(&module,
      ".version 8.0\n.target sm_90a\n.address_size 64\n.global .u32 unsupported = 9;\n"), CUDA_ERROR_NOT_SUPPORTED);
  state("bad_load_preserves_output", module == sentinel);
  const unsigned char elf[] = {0x7f, 'E', 'L', 'F'};
  expect("cubin_rejected", cuModuleLoadData(&module, elf), CUDA_ERROR_NOT_SUPPORTED);
  expect("fatbinary_rejected", cuModuleLoadFatBinary(&module, elf), CUDA_ERROR_NOT_SUPPORTED);
  CUjit_option option = CU_JIT_MAX_REGISTERS;
  void *option_value = (void *)uintptr_t(32);
  expect("jit_options_rejected", cuModuleLoadDataEx(&module, text.c_str(), 1, &option, &option_value), CUDA_ERROR_NOT_SUPPORTED);
  expect("file_load", cuModuleLoad(&module, argv[1]), CUDA_SUCCESS);
  CUmodule module2 = nullptr;
  std::string variant = text;
  expect("data_ex_load", cuModuleLoadDataEx(&module2, variant.c_str(), 0, nullptr, nullptr), CUDA_SUCCESS);
  CUfunction first = nullptr, second = nullptr, repeat = nullptr;
  expect("query_first", cuModuleGetFunction(&first, module, "runtime_kernel"), CUDA_SUCCESS);
  expect("query_second", cuModuleGetFunction(&second, module2, "runtime_kernel"), CUDA_SUCCESS);
  expect("repeat_query", cuModuleGetFunction(&repeat, module, "runtime_kernel"), CUDA_SUCCESS);
  state("module_function_isolation", first != second && first == repeat && module != module2);
  expect("missing_function", cuModuleGetFunction(&repeat, module, "missing"), CUDA_ERROR_NOT_FOUND);
  state("failed_query_preserves_output", repeat == first);
  expect("invalid_module", cuModuleGetFunction(&repeat, sentinel, "runtime_kernel"), CUDA_ERROR_INVALID_HANDLE);
  expect("null_function_output", cuModuleGetFunction(nullptr, module, "runtime_kernel"), CUDA_ERROR_INVALID_VALUE);
  int value = -123;
  expect("ptx_version", cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_PTX_VERSION, first), CUDA_SUCCESS);
  state("ptx_version_value", value == 80);
  expect("static_shared", cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, first), CUDA_SUCCESS);
  state("static_shared_value", value == 0);
  expect("dynamic_shared_limit", cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, first), CUDA_SUCCESS);
  state("dynamic_shared_limit_value", value == 0);
  value = -123;
  expect("unknown_register_allocation", cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_NUM_REGS, first), CUDA_ERROR_NOT_SUPPORTED);
  state("metadata_output_preserved", value == -123);
  expect("dynamic_attribute_rejected", cuFuncSetAttribute(first, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 1024), CUDA_ERROR_NOT_SUPPORTED);
  expect("negative_attribute", cuFuncSetAttribute(first, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, -1), CUDA_ERROR_INVALID_VALUE);
  CUcontext context_sentinel = (CUcontext)uintptr_t(0x9876);
  expect("context_rejected", cuCtxGetCurrent(&context_sentinel), CUDA_ERROR_NOT_SUPPORTED);
  state("context_output_preserved", context_sentinel == (CUcontext)uintptr_t(0x9876));
  auto launch = [&](CUfunction f, unsigned grid, unsigned block, unsigned shared,
                    CUstream stream, void **arguments, void **extra) {
    return cuLaunchKernel(f, grid, 1, 1, block, 1, 1, shared, stream, arguments, extra);
  };
  void *extra[] = {CU_LAUNCH_PARAM_END};
  void *dummy_params[] = {nullptr};
  expect("invalid_function", launch(nullptr, 2, 32, 0, nullptr, nullptr, nullptr), CUDA_ERROR_INVALID_HANDLE);
  expect("empty_grid", launch(first, 0, 32, 0, nullptr, nullptr, nullptr), CUDA_ERROR_INVALID_VALUE);
  expect("large_block", launch(first, 2, 1025, 0, nullptr, nullptr, nullptr), CUDA_ERROR_INVALID_VALUE);
  expect("both_param_forms", launch(first, 2, 32, 0, nullptr, dummy_params, extra), CUDA_ERROR_INVALID_VALUE);
  expect("packed_extra_rejected", launch(first, 2, 32, 0, nullptr, nullptr, extra), CUDA_ERROR_NOT_SUPPORTED);
  expect("dynamic_shared_rejected", launch(first, 2, 32, 16, nullptr, nullptr, nullptr), CUDA_ERROR_NOT_SUPPORTED);
  expect("invalid_stream_rejected", launch(first, 2, 32, 0, (CUstream)uintptr_t(0xdead), nullptr, nullptr), CUDA_ERROR_NOT_SUPPORTED);
  CUlaunchAttribute cluster = {};
  cluster.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  cluster.value.clusterDim.x = 2;
  cluster.value.clusterDim.y = cluster.value.clusterDim.z = 1;
  CUlaunchConfig config = {};
  config.gridDimX = 2; config.gridDimY = config.gridDimZ = 1;
  config.blockDimX = 32; config.blockDimY = config.blockDimZ = 1;
  config.attrs = &cluster; config.numAttrs = 1;
  expect("cluster_launch_rejected", cuLaunchKernelEx(&config, first, nullptr, nullptr), CUDA_ERROR_NOT_SUPPORTED);
  value = -123;
  expect("cluster_occupancy_rejected", cuOccupancyMaxActiveClusters(&value, first, &config), CUDA_ERROR_NOT_SUPPORTED);
  state("occupancy_output_preserved", value == -123);
  alignas(64) CUtensorMap descriptor;
  std::memset(&descriptor, 0xa5, sizeof(descriptor));
  unsigned char old_descriptor[sizeof(descriptor)];
  std::memcpy(old_descriptor, &descriptor, sizeof(descriptor));
  cuuint64_t dimensions[] = {64, 64}, strides[] = {256};
  cuuint32_t boxes[] = {16, 16}, elements[] = {1, 1};
  expect("tensor_map_rejected", cuTensorMapEncodeTiled(&descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT32,
      2, (void *)uintptr_t(0x1000), dimensions, strides, boxes, elements, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), CUDA_ERROR_NOT_SUPPORTED);
  state("descriptor_preserved", std::memcmp(&descriptor, old_descriptor, sizeof(descriptor)) == 0);
  expect("tensor_address_rejected", cuTensorMapReplaceAddress(&descriptor, (void *)uintptr_t(0x1000)), CUDA_ERROR_NOT_SUPPORTED);
  state("replace_descriptor_preserved", std::memcmp(&descriptor, old_descriptor, sizeof(descriptor)) == 0);
  expect("launch_first", launch(first, 2, 32, 0, nullptr, nullptr, nullptr), CUDA_SUCCESS);
  expect("sync_first", cuCtxSynchronize(), CUDA_SUCCESS);
  config.numAttrs = 0; config.attrs = nullptr;
  expect("launch_ex_base", cuLaunchKernelEx(&config, second, nullptr, nullptr), CUDA_SUCCESS);
  expect("unload_inflight", cuModuleUnload(module2), CUDA_SUCCESS);
  expect("stale_launch", launch(second, 2, 32, 0, nullptr, nullptr, nullptr), CUDA_ERROR_INVALID_HANDLE);
  expect("stale_query", cuModuleGetFunction(&repeat, module2, "runtime_kernel"), CUDA_ERROR_INVALID_HANDLE);
  expect("unload_twice", cuModuleUnload(module2), CUDA_ERROR_INVALID_HANDLE);
  CUmodule reloaded = nullptr;
  expect("reload", cuModuleLoadData(&reloaded, text.c_str()), CUDA_SUCCESS);
  state("handles_not_reused", reloaded != module2);
  expect("stale_after_reload", launch(second, 2, 32, 0, nullptr, nullptr, nullptr), CUDA_ERROR_INVALID_HANDLE);
  expect("unload_reload", cuModuleUnload(reloaded), CUDA_SUCCESS);
  expect("unload_first", cuModuleUnload(module), CUDA_SUCCESS);
  std::ifstream numeric_input(argv[2]);
  std::ostringstream numeric_buffer;
  numeric_buffer << numeric_input.rdbuf();
  std::string numeric_text = numeric_buffer.str(), numeric_variant = numeric_text;
  size_t bias = numeric_variant.find("%r3, %r3, 100;");
  if (bias == std::string::npos) return 2;
  numeric_variant.replace(bias, std::strlen("%r3, %r3, 100;"), "%r3, %r3, 200;");
  CUmodule numeric_modules[2] = {};
  CUfunction numeric_functions[2] = {};
  expect("numeric_load_first", cuModuleLoadData(&numeric_modules[0], numeric_text.c_str()), CUDA_SUCCESS);
  expect("numeric_load_second", cuModuleLoadData(&numeric_modules[1], numeric_variant.c_str()), CUDA_SUCCESS);
  expect("numeric_query_first", cuModuleGetFunction(&numeric_functions[0], numeric_modules[0], "runtime_numeric"), CUDA_SUCCESS);
  expect("numeric_query_second", cuModuleGetFunction(&numeric_functions[1], numeric_modules[1], "runtime_numeric"), CUDA_SUCCESS);
  void *device = nullptr;
  state("malloc", cudaMalloc(&device, 64 * sizeof(uint32_t)) == cudaSuccess);
  uint8_t small = 3;
  alignas(16) unsigned char record[16] = {};
  uint32_t record_value = 7, tail = 11;
  std::memcpy(record + 8, &record_value, sizeof(record_value));
  void *params[] = {&device, &small, record, &tail};
  void *invalid_params[] = {&device, nullptr, record, &tail};
  expect("missing_params", launch(numeric_functions[0], 2, 32, 0, nullptr, nullptr, nullptr), CUDA_ERROR_INVALID_VALUE);
  expect("null_argument", launch(numeric_functions[0], 2, 32, 0, nullptr, invalid_params, nullptr), CUDA_ERROR_INVALID_VALUE);
  for (unsigned module_index = 0; module_index < 2; ++module_index) {
    expect("numeric_launch", launch(numeric_functions[module_index], 2, 32, 0, nullptr, params, nullptr), CUDA_SUCCESS);
    expect("numeric_sync", cuCtxSynchronize(), CUDA_SUCCESS);
    uint32_t output[64] = {};
    state("numeric_copy", cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost) == cudaSuccess);
    bool correct = true;
    for (unsigned lane = 0; lane < 64; ++lane)
      correct &= output[lane] == 121 + 100 * module_index + lane;
    state(module_index ? "second_module_numeric" : "first_module_numeric", correct);
    std::printf("DRIVER_NUMERIC module=%u first=%u last=%u expected_first=%u\n", module_index, output[0], output[63], 121 + 100 * module_index);
    expect("numeric_unload", cuModuleUnload(numeric_modules[module_index]), CUDA_SUCCESS);
  }
  state("free", cudaFree(device) == cudaSuccess);
  const char *error_text = nullptr;
  expect("error_string", cuGetErrorString(CUDA_ERROR_NOT_SUPPORTED, &error_text), CUDA_SUCCESS);
  state("error_string_value", error_text && std::strcmp(error_text, "CUDA_ERROR_NOT_SUPPORTED") == 0);
  std::printf("RUNTIME_RESULT status=%s checks=%d failures=%d numeric_values=128\n", failures ? "FAIL" : "PASS", checks, failures);
  return failures ? 1 : 0;
}
