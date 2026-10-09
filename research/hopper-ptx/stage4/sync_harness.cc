#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "libcuda/gpgpu_context.h"

static void die(const char *msg) { std::fprintf(stderr, "stage4_harness: %s\n", msg); std::exit(2); }

int main(int argc, char **argv) {
  if (argc != 2) die("usage: sync_harness KERNEL.ptx");
  const char *ptx = argv[1];
  const unsigned n = 32;
  setenv("PTX_SIM_KERNELFILE", ptx, 1);
  gpgpu_context context;
  context.gpgpu_ptx_sim_init_perf();
  symbol_table *symbols = context.gpgpu_ptx_sim_load_ptx_from_filename(ptx);
  function_info *entry = symbols->lookup_function("mbarrier_kernel");
  if (!entry) die("mbarrier_kernel was not parsed");
  gpgpu_t *gpu = context.the_gpgpusim->g_the_gpu;
  std::vector<uint32_t> output(n, 0);
  void *output_device = gpu->gpu_malloc(output.size() * sizeof(uint32_t));
  if (!output_device) die("gpu_malloc failed");
  gpgpu_ptx_sim_arg_list_t args;
  args.push_back(gpgpu_ptx_sim_arg(&output_device, sizeof(output_device), 0));
  dim3 grid(1, 1, 1), threads(n, 1, 1);
  kernel_info_t *kernel = context.func_sim->gpgpu_opencl_ptx_sim_init_grid(
      entry, args, grid, threads, gpu);
  context.func_sim->gpgpu_cuda_ptx_sim_main_func(*kernel, true);
  gpu->memcpy_from_gpu(output.data(), reinterpret_cast<size_t>(output_device),
                       output.size() * sizeof(uint32_t));
  for (unsigned i = 0; i < n; ++i) {
    if (output[i] != 10) {
      std::fprintf(stderr, "MISMATCH lane=%u expected=10 actual=%u\n", i, output[i]);
      return 1;
    }
  }
  std::printf("SYNC_RESULT status=PASS n=%u first=%u last=%u\n", n,
              output.front(), output.back());
  return 0;
}
