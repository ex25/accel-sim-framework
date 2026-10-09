#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "gpgpu-sim/gpu-sim.h"
#include "libcuda/gpgpu_context.h"

int main(int argc, char **argv) {
  // PTX mode grid.x grid.y grid.z cluster.x cluster.y cluster.z explicit
  if (argc != 10) return 2;
  setvbuf(stdout, nullptr, _IOLBF, 0);
  setenv("PTX_SIM_KERNELFILE", argv[1], 1);
  const std::string mode = argv[2];
  const dim3 grid(std::stoul(argv[3]), std::stoul(argv[4]), std::stoul(argv[5]));
  const dim3 cluster(std::stoul(argv[6]), std::stoul(argv[7]), std::stoul(argv[8]));
  const bool explicit_launch = std::stoul(argv[9]);
  try {
    gpgpu_context context;
    auto *gpu = context.gpgpu_ptx_sim_init_perf();
    auto *symbols = context.gpgpu_ptx_sim_load_ptx_from_filename(argv[1]);
    auto *entry = symbols->lookup_function("cluster_kernel");
    if (!entry) return 2;
    entry->do_pdom(); entry->set_pdom();
    const unsigned ctas = grid.x * grid.y * grid.z, n = cluster.x * cluster.y * cluster.z;
    std::vector<uint32_t> out(ctas * 32 * 20, 0xdeadbeef);
    void *device = gpu->gpu_malloc(out.size() * 4);
    gpu->memcpy_to_gpu(reinterpret_cast<size_t>(device), out.data(), out.size() * 4);
    gpgpu_ptx_sim_arg_list_t args;
    args.push_back(gpgpu_ptx_sim_arg(&device, sizeof(device), 0));
    auto *kernel = context.func_sim->gpgpu_opencl_ptx_sim_init_grid(
        entry, args, grid, dim3(32, 1, 1), gpu);
    if (explicit_launch) gpu->register_ptx_cluster_launch(kernel->get_uid(), cluster);
    unsigned cycles = 0;
    if (mode == "functional") context.func_sim->gpgpu_cuda_ptx_sim_main_func(*kernel, true);
    else {
      gpu->init(); gpu->launch(kernel);
      while (gpu->active() && cycles < 100000) {gpu->cycle(); ++cycles;}
      if (gpu->active() || !kernel->done()) throw std::runtime_error("detailed kernel did not finish");
    }
    gpu->memcpy_from_gpu(out.data(), reinterpret_cast<size_t>(device), out.size() * 4);
    const bool metadata = std::string(argv[1]).find("metadata.ptx") != std::string::npos;
    for (unsigned z = 0; z < grid.z; ++z) for (unsigned y = 0; y < grid.y; ++y)
      for (unsigned x = 0; x < grid.x; ++x) for (unsigned lane = 0; lane < 32; ++lane) {
        const unsigned cta = x + grid.x * (y + grid.y * z);
        const unsigned rank = x % cluster.x + cluster.x * (y % cluster.y + cluster.y * (z % cluster.z));
        const unsigned target_rank = (rank + 1) % n, source_rank = (rank + n - 1) % n;
        auto peer = [&](unsigned r) {
          unsigned px = x / cluster.x * cluster.x + r % cluster.x;
          unsigned py = y / cluster.y * cluster.y + r / cluster.x % cluster.y;
          unsigned pz = z / cluster.z * cluster.z + r / (cluster.x * cluster.y);
          return px + grid.x * (py + grid.y * pz);
        };
        // Independent CPU reference in logical grid coordinates.
        const std::vector<uint32_t> expected = {rank, n, x/cluster.x, y/cluster.y, z/cluster.z,
          grid.x/cluster.x, grid.y/cluster.y, grid.z/cluster.z,
          x%cluster.x, y%cluster.y, z%cluster.z, cluster.x, cluster.y, cluster.z,
          unsigned(explicit_launch), target_rank, peer(target_rank)*1000+lane,
          peer(source_rank)*1000+lane, peer(target_rank)*1000+lane, target_rank};
        for (unsigned field = 0; field < (metadata ? 15 : 20); ++field)
          if (out[(cta*32+lane)*20+field] != expected[field]) {
            fprintf(stderr, "MISMATCH cta=%u lane=%u field=%u actual=%u expected=%u\n", cta,lane,field,out[(cta*32+lane)*20+field],expected[field]);return 1;
          }
      }
    const auto &s = gpu->ptx_cluster_stats;
    if (!metadata && (s.mapa != ctas*32*2 || s.barrier_arrive != ctas*32*3 ||
                     s.barrier_wait != s.barrier_arrive || s.proxy_fence != ctas*32))
      throw std::runtime_error("cluster execution statistics mismatch");
    printf("CLUSTER_RESULT status=PASS mode=%s grid=%u,%u,%u cluster=%u,%u,%u explicit=%u values=%u barriers=%llu wait_checks=%llu cycles=%u\n",
      mode.c_str(),grid.x,grid.y,grid.z,cluster.x,cluster.y,cluster.z,explicit_launch,
      ctas*32*(metadata?15:20),(unsigned long long)s.barrier_wait,(unsigned long long)s.wait_checks,cycles);
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "CLUSTER_FAILURE %s\n", error.what());return 3;
  }
}
