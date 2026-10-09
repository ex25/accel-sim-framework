#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "libcuda/gpgpu_context.h"

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) return 2;
  setenv("PTX_SIM_KERNELFILE", argv[1], 1);
  gpgpu_context context;
  context.gpgpu_ptx_sim_init_perf();
  symbol_table *symbols = context.gpgpu_ptx_sim_load_ptx_from_filename(argv[1]);
  function_info *entry = symbols->lookup_function("tma_kernel");
  if (!entry) return 2;
  gpgpu_t *gpu = context.the_gpgpusim->g_the_gpu;
  std::vector<uint32_t> input{11, 12, 13, 21, 22, 23};
  void *source = gpu->gpu_malloc(24);
  void *descriptor_address = gpu->gpu_malloc(128);
  void *out = gpu->gpu_malloc(32 * 4);
  gpu->memcpy_to_gpu(reinterpret_cast<size_t>(source), input.data(), 24);
  ptx_tma_descriptor descriptor;
  descriptor.base = reinterpret_cast<uint64_t>(source);
  descriptor.rank = 2;
  descriptor.element_bytes = 4;
  descriptor.dimensions = {3, 2, 0, 0, 0};
  descriptor.byte_strides = {4, 12, 0, 0, 0};
  descriptor.box = {2, 2, 0, 0, 0};
  if (argc == 2)
    gpu->register_ptx_tma_descriptor(reinterpret_cast<uint64_t>(descriptor_address), descriptor);
  gpgpu_ptx_sim_arg_list_t args;
  // The loader consumes the argument list in reverse order.
  args.push_back(gpgpu_ptx_sim_arg(&descriptor_address, sizeof(descriptor_address), 8));
  args.push_back(gpgpu_ptx_sim_arg(&out, sizeof(out), 0));
  kernel_info_t *kernel = context.func_sim->gpgpu_opencl_ptx_sim_init_grid(
      entry, args, dim3(1, 1, 1), dim3(32, 1, 1), gpu);
  context.func_sim->gpgpu_cuda_ptx_sim_main_func(*kernel, true);
  std::vector<uint32_t> output(32);
  gpu->memcpy_from_gpu(output.data(), reinterpret_cast<size_t>(out), 32 * 4);
  const unsigned expected[] = {13, 0, 23, 0};
  for (unsigned i = 0; i < 32; ++i) {
    if (output[i] != expected[i % 4]) {
      std::fprintf(stderr, "MISMATCH lane=%u expected=%u actual=%u\n", i, expected[i % 4], output[i]);
      return 1;
    }
  }
  const auto &stats = gpu->ptx_tma_stats;
  const auto &metadata = gpu->last_ptx_tma_metadata;
  const uint64_t base = reinterpret_cast<uint64_t>(source);
  if (stats.issue != 1 || stats.completion != 1 || stats.global_reads != 2 ||
      stats.shared_writes != 1 || stats.global_bytes != 8 || stats.shared_bytes != 16 ||
      stats.oob_bytes != 8 || stats.wait_failure != 32 || stats.wait_success != 32 ||
      metadata.addresses != std::vector<uint64_t>({base + 8, base + 20}) ||
      metadata.element_bytes != 4 || metadata.in_bounds_bytes != 8 ||
      metadata.oob_bytes != 8 || metadata.barrier != 16 || metadata.mask != 1) {
    std::fprintf(stderr, "TMA metadata/statistics mismatch\n");
    return 1;
  }
  std::printf("TMA_RESULT status=PASS tile=13,0,23,0 bytes=16 oob=8 wait_before=0 wait_after=1\n");
  std::printf("PTX_EVENT_RESULT status=PASS issue=1 completion=1 global_reads=2 shared_writes=1 wait_failure=32 wait_success=32 metadata=PASS\n");
}
