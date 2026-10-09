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
  if (argc != 3) return 2;
  const std::string scenario = argv[2];
  setvbuf(stdout, nullptr, _IOLBF, 0);
  setenv("PTX_SIM_KERNELFILE", argv[1], 1);
  gpgpu_context context;
  auto *gpu = context.gpgpu_ptx_sim_init_perf();
  symbol_table *symbols = context.gpgpu_ptx_sim_load_ptx_from_filename(argv[1]);
  auto *entry = symbols->lookup_function("tma_detailed");
  if (!entry) return 2;
  entry->do_pdom(); entry->set_pdom();
  const bool one_d = scenario == "one_d";
  const unsigned ctas = scenario == "single" || scenario == "local" || scenario == "all_oob" || one_d ? 1 :
      scenario == "not_started" ? 3 : 4;
  unsigned mask = ctas == 1 ? 1 : 5;
  if (scenario == "empty") mask = 0;
  if (scenario == "unknown") mask = 17;
  if (scenario == "not_started") mask = 9;
  auto *source = gpu->gpu_malloc(256);
  auto *descriptor_address = gpu->gpu_malloc(128);
  auto *out = gpu->gpu_malloc(80);
  std::vector<uint32_t> input(64);
  input[2] = 13; input[10] = 23;
  gpu->memcpy_to_gpu(reinterpret_cast<size_t>(source), input.data(), 256);
  ptx_tma_descriptor descriptor;
  descriptor.base = reinterpret_cast<uint64_t>(source);
  descriptor.rank = 2; descriptor.element_bytes = 4;
  descriptor.dimensions = {3,2,0,0,0}; descriptor.byte_strides = {4,32,0,0,0};
  descriptor.box = {2,2,0,0,0};
  if (scenario == "all_oob") descriptor.dimensions = {1,2,0,0,0};
  if (one_d) { descriptor.rank = 1; descriptor.box = {4,0,0,0,0}; }
  if (scenario == "bad_descriptor") descriptor.rank = 3;
  if (scenario == "high_rank") {
    descriptor.rank = 3; descriptor.dimensions[2] = 1;
    descriptor.box[2] = 1; descriptor.byte_strides[2] = 64;
  }
  if (scenario == "overlap") {
    descriptor.dimensions[0] = 4;
    descriptor.byte_strides[1] = 4;
  }
  if (scenario != "unregistered")
    gpu->register_ptx_tma_descriptor(reinterpret_cast<uint64_t>(descriptor_address), descriptor);
  gpgpu_ptx_sim_arg_list_t args;
  args.push_back(gpgpu_ptx_sim_arg(&mask, sizeof(mask), 16));
  args.push_back(gpgpu_ptx_sim_arg(&descriptor_address, sizeof(descriptor_address), 8));
  args.push_back(gpgpu_ptx_sim_arg(&out, sizeof(out), 0));
  auto *kernel = context.func_sim->gpgpu_opencl_ptx_sim_init_grid(entry, args, dim3(ctas,1,1), dim3(32,1,1), gpu);
  try {
    for (unsigned cta = 0; cta < (scenario == "not_started" ? 4 : ctas); ++cta) {
      ptx_tma_cta_binding binding;
      binding.kernel = kernel->get_uid(); binding.cta = cta;
      binding.cluster = 0; binding.rank = cta;
      binding.shared_address = 0; binding.barrier_offset = 16; binding.capacity = 16;
      if (scenario == "offset" && cta == 2) binding.barrier_offset = 24;
      gpu->register_ptx_tma_cta(binding);
      if (scenario == "duplicate" && cta == 0) gpu->register_ptx_tma_cta(binding);
    }
    gpu->init(); gpu->launch(kernel);
    unsigned cycles = 0;
    while (gpu->active() && cycles < 200000) { gpu->cycle(); ++cycles; }
    if (gpu->active() || !kernel->done()) throw std::runtime_error("detailed simulation did not finish");
    std::vector<uint32_t> output(20);
    gpu->memcpy_from_gpu(output.data(), reinterpret_cast<size_t>(out), output.size()*4);
    for (unsigned cta = 0; cta < ctas; ++cta) {
      bool selected = mask & (1u << cta);
      const unsigned expected[] = {13,0,one_d ? 0u : 23u,0};
      auto &table = kernel->get_cluster_mbarrier_lookup_table(dim3(0,0,0));
      auto *bar = table.lookup_clustermbar((cta << 24) | 16);
      for (unsigned e = 0; e < 4; ++e) {
        unsigned value = selected ? (scenario == "all_oob" ? 0 : expected[e]) : 0xcccccccc;
        if (output[cta*4+e] != value) {
          fprintf(stderr,"MISMATCH rank=%u e=%u actual=%x expected=%x\n",cta,e,output[cta*4+e],value); return 1;
        }
      }
      if (bar->get_phase() != int(selected) || bar->get_tx_count() != (selected ? 0 : 16) ||
          bar->get_pending_thread_count() != (selected ? 1u : 0u)) return 1;
      printf("CTA_RESULT status=PASS rank=%u selected=%u phase=%d tx=%d pending=%u\n",
          cta,selected,bar->get_phase(),bar->get_tx_count(),bar->get_pending_thread_count());
    }
    const auto &stats = gpu->ptx_tma_stats;
    unsigned targets = ctas == 1 ? 1 : 2;
    unsigned requests = one_d ? 1 : 2;
    unsigned global_bytes = one_d ? 4 : 8;
    const auto &metadata = gpu->last_ptx_tma_metadata;
    if (metadata.element_bytes != 4 || metadata.barrier != 16 || metadata.mask != mask ||
        metadata.in_bounds_bytes != (scenario == "all_oob" ? 0u : global_bytes) ||
        metadata.oob_bytes != 16 - metadata.in_bounds_bytes ||
        metadata.addresses.size() * 4 != metadata.in_bounds_bytes) return 1;
    if (output[16] != (scenario == "all_oob" ? 0u : 0xcccccccc) ||
        output[17] != (scenario == "all_oob" ? 1u : 0u)) {
      fprintf(stderr, "SNAPSHOT_FAILURE value=%x wait=%u\n",output[16],output[17]); return 1;
    }
    if (stats.detailed_issue != 1 || stats.issue != 1 || stats.completion != targets ||
        stats.shared_writes != targets || stats.global_requests != stats.global_responses ||
        (scenario != "all_oob" && (stats.global_requests != requests || stats.global_bytes != global_bytes ||
          stats.oob_completions != targets || stats.in_bounds_completions != targets)) ||
        (scenario == "all_oob" && (stats.global_requests != 0 || stats.global_bytes != 0 ||
          stats.oob_completions != 1 || stats.in_bounds_completions != 0)) ||
        !stats.wait_failure || !stats.wait_success) return 1;
    printf("VISIBILITY_RESULT status=PASS before=%x wait_before=%u wait_after=1\n", output[16],output[17]);
    printf("METADATA_RESULT status=PASS tile_bytes=16 in_bounds=%zu oob=%zu barrier=%x mask=%x\n",
        metadata.in_bounds_bytes,metadata.oob_bytes,metadata.barrier,metadata.mask);
    printf("DETAILED_RESULT status=PASS scenario=%s mask=%x cycles=%u issue=%llu requests=%llu responses=%llu writes=%llu completions=%llu waits_failed=%llu waits_success=%llu\n",
        scenario.c_str(),mask,cycles,(unsigned long long)stats.issue,
        (unsigned long long)stats.global_requests,(unsigned long long)stats.global_responses,
        (unsigned long long)stats.shared_writes,(unsigned long long)stats.completion,
        (unsigned long long)stats.wait_failure,(unsigned long long)stats.wait_success);
  } catch (const std::exception &error) {
    fprintf(stderr,"TMA_NEGATIVE scenario=%s issue=%llu reads=%llu writes=%llu diagnostic=%s\n",
        scenario.c_str(),(unsigned long long)gpu->ptx_tma_stats.issue,
        (unsigned long long)gpu->ptx_tma_stats.global_reads,(unsigned long long)gpu->ptx_tma_stats.shared_writes,error.what());
    return 3;
  }
}
