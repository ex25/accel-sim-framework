#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include "abstract_hardware_model.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "libcuda/gpgpu_context.h"
#include "gpgpu-sim/gpu-sim.h"

// This harness only feeds simulator memory and reads it back. Numerical
// validation lives in run_acceptance.py, independently of the instruction.
int main(int argc, char **argv) {
  if (argc != 6) return 2;  // PTX input.bin output.bin block mode
  const unsigned block = std::strtoul(argv[4], nullptr, 10);
  const bool functional = std::strtoul(argv[5], nullptr, 10) == 1;
  std::vector<uint16_t> input(4096);
  std::ifstream source(argv[2], std::ios::binary);
  if (!source.read(reinterpret_cast<char *>(input.data()), input.size() * 2))
    return 2;
  setenv("PTX_SIM_KERNELFILE", argv[1], 1);
  gpgpu_context context;
  context.gpgpu_ptx_sim_init_perf();
  symbol_table *symbols = context.gpgpu_ptx_sim_load_ptx_from_filename(argv[1]);
  function_info *entry = symbols->lookup_function("matrix_load");
  if (!entry || entry->get_ptx_version().target_string() != "sm_90a") return 2;
  entry->do_pdom();
  entry->set_pdom();
  unsigned loads = 0;
  for (unsigned i = 0; i < entry->get_function_size(); ++i) {
    const auto *insn = entry->get_instruction(entry->get_start_PC() + i * 8);
    if (!insn || insn->ptx_matrix.kind != ptx_matrix_metadata::LDMATRIX) continue;
    const auto &m = insn->ptx_matrix;
    if (m.m != 8 || m.n != 8 || !m.functional_supported || m.timing_supported ||
        insn->dst().get_vect_nelem() != m.matrices) return 2;
    ++loads;
    std::printf("LOAD_METADATA opcode=%s matrices=%u trans=%u functional=1 timing=0\n",
                m.opcode.c_str(), m.matrices, unsigned(m.transpose));
  }
  if (loads != 1) return 2;
  gpgpu_sim *gpu = context.the_gpgpusim->g_the_gpu;
  std::vector<uint32_t> output(2 * 64 * 4, 0xdeadbeef);
  void *in = gpu->gpu_malloc(input.size() * 2);
  void *out = gpu->gpu_malloc(output.size() * 4);
  gpu->memcpy_to_gpu(reinterpret_cast<size_t>(in), input.data(), input.size() * 2);
  gpu->memcpy_to_gpu(reinterpret_cast<size_t>(out), output.data(), output.size() * 4);
  gpgpu_ptx_sim_arg_list_t args;
  args.push_back(gpgpu_ptx_sim_arg(&out, sizeof(out), 8));
  args.push_back(gpgpu_ptx_sim_arg(&in, sizeof(in), 0));
  kernel_info_t *kernel = context.func_sim->gpgpu_opencl_ptx_sim_init_grid(
      entry, args, dim3(2, 1, 1), dim3(block, 1, 1), gpu);
  if (functional) {
    // Match stream_manager's functional launch/finish lifecycle; the kernel
    // interpreter alone does not set the GPU mode or functional warp IDs.
    gpu->functional_launch(kernel);
    context.func_sim->gpgpu_cuda_ptx_sim_main_func(*kernel, true);
    gpu->finish_functional_sim(kernel);
  }
  else {
    gpu->init();
    gpu->launch(kernel);
    unsigned cycles = 0;
    while (gpu->active() && cycles++ < 100000) gpu->cycle();
    if (gpu->active() || !kernel->done()) return 1;
  }
  gpu->memcpy_from_gpu(output.data(), reinterpret_cast<size_t>(out), output.size() * 4);
  std::ofstream sink(argv[3], std::ios::binary);
  sink.write(reinterpret_cast<const char *>(output.data()), output.size() * 4);
  if (!sink) return 2;
  std::printf("EXECUTION grid=2 block=%u mode=%s output_words=%zu\n",
              block, functional ? "functional" : "detailed", output.size());
  return 0;
}
