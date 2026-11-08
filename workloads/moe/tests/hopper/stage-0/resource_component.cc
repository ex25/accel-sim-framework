// The stream-ID constructor is also used by trace_kernel_info_t (SASS).
#include "libcuda/gpgpu_context.h"
#include "src/cuda-sim/ptx_ir.h"
#include "src/gpgpu-sim/shader.h"

#include <cassert>
#include <cstdio>

class FixtureFunction : public function_info {
 public:
  explicit FixtureFunction(gpgpu_context *ctx) : function_info(1, ctx) {
    m_kernel_info.smem = 32768;
    set_maxnt_id(0);
    m_kernel_info.maxthreads = 0;
  }
  unsigned base_shared() const { return m_kernel_info.smem; }
};

int main() {
  gpgpu_context *ctx = GPGPU_Context();
  FixtureFunction entry(ctx);
  kernel_info_t trace(dim3(32), dim3(32), &entry, 0ULL);
  kernel_info_t ptx(dim3(32), dim3(32), &entry, {}, {});
  assert(trace.dynamic_shared_memory() == 0);
  assert(trace.shared_memory_per_cta() == 32768);
  assert(ptx.dynamic_shared_memory() == 0);
  ptx.set_dynamic_shared_memory(32768);
  assert(ptx.shared_memory_per_cta() == 65536);
  assert(trace.shared_memory_per_cta() == 32768);
  assert(entry.base_shared() == 32768);
  shader_core_config config(ctx);
  config.warp_size = 32;
  config.n_thread_per_shader = 2048;
  config.gpgpu_shmem_size = 233472;
  config.gpgpu_shader_registers = 65536;
  config.max_cta_per_core = 32;
  config.n_simt_clusters = 1;
  config.n_simt_cores_per_cluster = 1;
  config.adaptive_cache_config = false;
  config.gpgpu_ignore_resources_limitation = false;
  assert(config.max_cta(trace) == 7);
  assert(config.max_cta(ptx) == 3);
  assert(config.max_cta(trace) == 7);
  puts("PASS resource component: SASS constructor D=0, B+D, isolated CTA limits 7/3/7");
}
