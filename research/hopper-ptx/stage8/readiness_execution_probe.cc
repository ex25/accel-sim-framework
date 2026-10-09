#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "gpgpu-sim/gpu-sim.h"
#include "libcuda/gpgpu_context.h"

// Run an unsupported instruction before a store: a diagnostic and intact memory
// are required in both modes. Returning normally would permit a silent no-op.
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  setenv("PTX_SIM_KERNELFILE", argv[1], 1);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  gpgpu_context context;
  auto *gpu = context.gpgpu_ptx_sim_init_perf();
  auto *symbols = context.gpgpu_ptx_sim_load_ptx_from_filename(argv[1]);
  auto *entry = symbols->lookup_function("readiness_kernel");
  if (!entry) return 2;
  entry->do_pdom(); entry->set_pdom();
  const unsigned guard = 0xdeadbeef;
  const bool scoped_labels = std::string(argv[3]) == "scoped_labels";
  void *device = gpu->gpu_malloc(sizeof(guard));
  gpu->memcpy_to_gpu(reinterpret_cast<size_t>(device), &guard, sizeof(guard));
  gpgpu_ptx_sim_arg_list_t args;
  args.push_back(gpgpu_ptx_sim_arg(&device, sizeof(device), 0));
  auto *kernel = context.func_sim->gpgpu_opencl_ptx_sim_init_grid(
      entry, args, dim3(1, 1, 1), dim3(128, 1, 1), gpu);
  try {
    if (std::string(argv[2]) == "functional")
      context.func_sim->gpgpu_cuda_ptx_sim_main_func(*kernel, true);
    else {
      gpu->init(); gpu->launch(kernel);
      unsigned cycles = 0;
      while (gpu->active() && cycles++ < 100000) gpu->cycle();
      if (gpu->active()) throw std::runtime_error("unexpected kernel timeout");
    }
  } catch (const std::exception &error) {
    unsigned after = 0;
    gpu->memcpy_from_gpu(&after, reinterpret_cast<size_t>(device), sizeof(after));
    if (std::string(error.what()).find(argv[3]) == std::string::npos || after != guard) {
      std::fprintf(stderr, "UNEXPECTED_FAILURE %s guard=%x\n", error.what(), after);
      return 1;
    }
    std::printf("STRICT_EXECUTION_RESULT status=PASS mode=%s guard=unchanged diagnostic=%s\n",
                argv[2], error.what());
    return 0;
  }
  if (scoped_labels) {
    unsigned after = 0;
    gpu->memcpy_from_gpu(&after, reinterpret_cast<size_t>(device), sizeof(after));
    if (after != 3) {
      std::fprintf(stderr, "SCOPED_LABEL_FAILURE actual=%u expected=3\n", after);
      return 1;
    }
    std::printf("SCOPED_LABEL_RESULT status=PASS mode=%s value=3\n", argv[2]);
    return 0;
  }
  std::fprintf(stderr, "UNEXPECTED_SUCCESS unsupported instruction executed\n");
  return 1;
}
