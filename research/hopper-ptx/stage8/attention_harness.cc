#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "gpgpu-sim/gpu-sim.h"
#include "libcuda/gpgpu_context.h"

int main(int argc, char **argv) {
  if (argc != 9) { fprintf(stderr,"usage: harness PTX MODE CASE SEQ ROWS HEADS CYCLE_LIMIT OUTPUT\n"); return 2; }
  try {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const std::string mode = argv[2], dir = argv[3];
    const unsigned seq = std::stoul(argv[4]), rows = std::stoul(argv[5]), heads = std::stoul(argv[6]);
    const uint64_t limit = std::stoull(argv[7]);
    if ((mode != "functional" && mode != "detailed") || !rows || rows > seq || !heads || heads > 8 ||
        (seq != 64 && seq != 128 && seq != 256)) throw std::runtime_error("invalid run dimensions/mode");
    setenv("PTX_SIM_KERNELFILE", argv[1], 1);
    gpgpu_context context;
    auto *gpu = context.gpgpu_ptx_sim_init_perf();
    auto *symbols = context.gpgpu_ptx_sim_load_ptx_from_filename(argv[1]);
    auto *entry = symbols->lookup_function("attention_kernel");
    if (!entry) throw std::runtime_error("missing attention kernel");
    entry->do_pdom(); entry->set_pdom();
    const size_t bytes = seq * 8 * 128 * 2;
    std::vector<uint16_t> data(bytes / 2, 0x7e00);
    void *output = gpu->gpu_malloc(bytes);
    gpu->memcpy_to_gpu(reinterpret_cast<size_t>(output), data.data(), bytes);
    void *input[3];
    for (unsigned i = 0; i < 3; ++i) {
      std::ifstream file(dir + "/" + std::string(1, "qkv"[i]) + ".f16le", std::ios::binary);
      if (!file.read(reinterpret_cast<char *>(data.data()), bytes) || file.peek() != EOF)
        throw std::runtime_error("input length mismatch");
      input[i] = gpu->gpu_malloc(bytes);
      gpu->memcpy_to_gpu(reinterpret_cast<size_t>(input[i]), data.data(), bytes);
    }
    gpgpu_ptx_sim_arg_list_t args;
    args.push_back(gpgpu_ptx_sim_arg(&seq, sizeof(seq), 32));
    for (int i = 2; i >= 0; --i) args.push_back(gpgpu_ptx_sim_arg(&input[i], sizeof(input[i]), (i + 1) * 8));
    args.push_back(gpgpu_ptx_sim_arg(&output, sizeof(output), 0));
    auto *kernel = context.func_sim->gpgpu_opencl_ptx_sim_init_grid(entry, args, dim3(rows, heads, 1), dim3(std::max(128u, seq),1,1),gpu);
    uint64_t cycles = 0;
    if (mode == "functional") context.func_sim->gpgpu_cuda_ptx_sim_main_func(*kernel, true);
    else {
      gpu->init(); gpu->launch(kernel);
      while (gpu->active() && cycles < limit) { gpu->cycle(); ++cycles; }
      if (gpu->active() || !kernel->done()) throw std::runtime_error("attention simulation exceeded cycle limit");
    }
    gpu->memcpy_from_gpu(data.data(), reinterpret_cast<size_t>(output), bytes);
    std::ofstream file(argv[8], std::ios::binary);
    file.write(reinterpret_cast<char *>(data.data()), bytes);
    if (!file) throw std::runtime_error("output write failed");
    printf("PTX_ATTENTION_RESULT status=PASS mode=%s seq=%u rows=%u heads=%u cycles=%llu\n",mode.c_str(),seq,rows,heads,(unsigned long long)cycles);
    return 0;
  } catch (const std::exception &error) { fprintf(stderr,"PTX_ATTENTION_FAILURE %s\n",error.what()); return 1; }
}
