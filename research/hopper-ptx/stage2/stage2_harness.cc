#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "abstract_hardware_model.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "libcuda/gpgpu_context.h"

static void die(const char *message) {
  std::fprintf(stderr, "stage2_harness: %s\n", message);
  std::exit(2);
}

int main(int argc, char **argv) {
  if (argc != 2) die("usage: stage2_harness KERNEL.ptx");
  const char *ptx_path = argv[1];
  const unsigned n = 128;
  const unsigned block = 128;
  setenv("PTX_SIM_KERNELFILE", ptx_path, 1);

  gpgpu_context context;
  context.gpgpu_ptx_sim_init_perf();
  symbol_table *symbols = context.gpgpu_ptx_sim_load_ptx_from_filename(ptx_path);
  function_info *entry = symbols->lookup_function("stage2_kernel");
  if (entry == nullptr) die("stage2_kernel was not found in parsed PTX");

  gpgpu_t *gpu = context.the_gpgpusim->g_the_gpu;
  std::vector<uint32_t> input(n), output(n, 0);
  for (unsigned i = 0; i < n; ++i) input[i] = 1000 + 3 * i;
  void *input_device = gpu->gpu_malloc(input.size() * sizeof(uint32_t));
  void *output_device = gpu->gpu_malloc(output.size() * sizeof(uint32_t));
  if (input_device == nullptr || output_device == nullptr) die("gpu_malloc failed");
  gpu->memcpy_to_gpu(reinterpret_cast<size_t>(input_device), input.data(),
                     input.size() * sizeof(uint32_t));

  gpgpu_ptx_sim_arg_list_t args;
  // gpgpu_opencl_ptx_sim_init_grid consumes this list in reverse order.
  args.push_back(gpgpu_ptx_sim_arg(&n, sizeof(n), 16));
  args.push_back(gpgpu_ptx_sim_arg(&output_device, sizeof(output_device), 8));
  args.push_back(gpgpu_ptx_sim_arg(&input_device, sizeof(input_device), 0));
  dim3 grid(1, 1, 1), threads(block, 1, 1);
  kernel_info_t *kernel = context.func_sim->gpgpu_opencl_ptx_sim_init_grid(
      entry, args, grid, threads, gpu);
  std::printf("HARNESS kernel=stage2_kernel grid=1,1,1 block=%u params=3 input=%p output=%p\n",
              block, input_device, output_device);
  context.func_sim->gpgpu_cuda_ptx_sim_main_func(*kernel, true);
  gpu->memcpy_from_gpu(output.data(), reinterpret_cast<size_t>(output_device),
                       output.size() * sizeof(uint32_t));

  uint64_t checksum = 0;
  for (unsigned i = 0; i < n; ++i) {
    const uint32_t expected = input[i] + 7 + i;
    if (output[i] != expected) {
      std::fprintf(stderr, "MISMATCH index=%u expected=%u actual=%u\n", i,
                   expected, output[i]);
      return 1;
    }
    checksum += output[i];
  }
  std::printf("RESULT status=PASS n=%u checksum=%llu first=%u last=%u\n", n,
              static_cast<unsigned long long>(checksum), output.front(),
              output.back());
  return 0;
}
