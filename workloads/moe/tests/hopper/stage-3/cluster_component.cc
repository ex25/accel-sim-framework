// PTX mapa encodings are checked against real, distinct shared objects here.
// This is a component check, not a remote PTX load/store test.
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <unordered_map>

#include "libcuda/gpgpu_context.h"
#include "src/cuda-sim/memory.h"
#include "src/cuda-sim/ptx_ir.h"
#include "src/gpgpu-sim/shader.h"

class FixtureFunction : public function_info {
 public:
  FixtureFunction(gpgpu_context *ctx, unsigned regs) : function_info(1, ctx) {
    m_kernel_info = {};
    m_kernel_info.regs = regs;
    set_maxnt_id(0);
  }
};

int main(int argc, char **argv) {
  ptx_version version(8.7, 0);
  version.set_target("sm_90a", NULL, NULL);
  assert(version.target() == 90);
  version.set_target("sm_70", NULL, NULL);
  assert(version.target() == 70);
  auto *ctx = GPGPU_Context();
  unsigned regs = std::stoul(argv[2]);
  assert(regs == 168);  // Independent H100 resource record for module 16.
  FixtureFunction entry(ctx, regs);
  kernel_info_t kernel(dim3(66, 2, 1), dim3(384), &entry, {}, {});
  kernel.set_cluster_dim(dim3(1, 2, 1));
  kernel.set_dynamic_shared_memory(215040);
  kernel.set_driver_shared_memory(1024);
  assert(kernel.shared_memory_per_cta() == 216064);
  assert(384 * ((regs + 3) & ~3) == 64512);
  shader_core_config config(ctx);
  config.warp_size = 32;
  config.n_thread_per_shader = 2048;
  config.gpgpu_shmem_size = 233472;
  config.gpgpu_shader_registers = 65536;
  config.max_cta_per_core = 32;
  config.n_simt_clusters = 132;
  config.n_simt_cores_per_cluster = 1;
  config.adaptive_cache_config = false;
  config.gpgpu_ignore_resources_limitation = false;
  assert(config.max_cta(kernel) == 1);
  printf(
      "PASS expert metadata: regs=%u registers/CTA=64512 "
      "PTX static=0 dynamic=215040 driver=1024 total=216064 capacity=233472\n",
      regs);
  assert(STATIC_ALLOC_LIMIT > 0 && STATIC_ALLOC_LIMIT < SHARED_GENERIC_START);
  printf(
      "ADDRESS_RANGES static_limit=%llx local_start=%llx shared_start=%llx "
      "heap=%llx\n",
      STATIC_ALLOC_LIMIT, LOCAL_GENERIC_START, SHARED_GENERIC_START,
      GLOBAL_HEAP_START);
  std::ifstream f(argv[1], std::ios::binary);
  std::vector<char> data{std::istreambuf_iterator<char>(f), {}};
  assert(data.size() == 4 * 32 * 128);
  const auto *words = reinterpret_cast<const uint32_t *>(data.data());
  for (unsigned x = 0; x < 2; ++x)
    for (unsigned rank = 0; rank < 2; ++rank) {
      unsigned value = 100 * x + 10 * rank + 1;
      auto *shared = kernel.create_cluster_shared(dim3(x, rank, 0));
      shared->write(4096, 4, &value, NULL, NULL);
    }
  for (unsigned x = 0; x < 2; ++x)
    for (unsigned rank = 0; rank < 2; ++rank) {
      dim3 source(x, rank, 0);
      auto local = kernel.resolve_local_shared(source, 4096);
      for (unsigned target = 0; target < 2; ++target) {
        uint32_t encoding = words[((rank * 2 + x) * 32) * 32 + 16 + target];
        auto resolved = kernel.resolve_cluster_shared(source, encoding);
        assert(resolved.cta.x == x && resolved.cta.y == target);
        assert(resolved.offset == 4096);
        assert((resolved.shared == local.shared) == (rank == target));
        unsigned actual = 0;
        resolved.shared->read(resolved.offset, 4, &actual);
        assert(actual == 100 * x + 10 * target + 1);
      }
    }
  kernel.retire_cluster_cta(dim3(0, 0, 0));
  auto retained = kernel.resolve_cluster_shared(dim3(0, 1, 0), 4096);
  unsigned actual = 0;
  retained.shared->read(retained.offset, 4, &actual);
  assert(actual == 1);
  kernel_info_t next(dim3(2, 2), dim3(32), &entry, {}, {});
  next.set_cluster_dim(dim3(1, 2, 1));
  auto *fresh = next.create_cluster_shared(dim3(0, 0, 0));
  assert(fresh != retained.shared);
  kernel.retire_cluster_cta(dim3(0, 1, 0));
  puts(
      "PASS PTX mapa -> kernel/cluster/rank/shared/offset; "
      "local rank explicit; live peer retained; kernels isolated");
}
