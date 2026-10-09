// Controlled trace-decoder oracle. Synthetic dynamic fields are NOT a capture.
#include <cstdio>
#include <stdexcept>
#include <string>
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "gpgpu-sim/gpu-sim.h"
#include "libcuda/gpgpu_context.h"
#include "option_parser.h"
#include "../../../gpu-simulator/trace-driven/trace_driven.h"
#include "../../../gpu-simulator/ISA_Def/hopper_opcode.h"

static void require(bool okay, const char *message) {
  if (!okay) throw std::runtime_error(message);
}
static void base(inst_trace_t &trace, const char *opcode, unsigned mask = 0xffffffffU) {
  trace.opcode = opcode; trace.mask = mask; trace.m_pc = 0;
  trace.cta_id = dim3(1,0,0); trace.cluster_cta_id = dim3(1,0,0);
  trace.cluster_id = dim3(0,0,0); trace.cluster_rank = 1;
  trace.reg_dsts_num = trace.reg_srcs_num = 0; trace.is_gmma_commit_group = 0;
}
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  try {
    gpgpu_context context; auto *gpu = context.gpgpu_ptx_sim_init_perf();
    trace_config config; auto options = option_parser_create();
    config.reg_options(options); config.parse_config();
    setenv("PTX_SIM_KERNELFILE",argv[1],1);
    auto *symbols = context.gpgpu_ptx_sim_load_ptx_from_filename(argv[1]);
    auto *entry = symbols->lookup_function("wgmma_kernel"); require(entry,"missing PTX WGMMA fixture");
    entry->do_pdom(); entry->set_pdom();
    unsigned matches = 0;
    for (unsigned i = 0; i < entry->get_function_size(); ++i) {
      const auto *ptx = entry->get_instruction(entry->get_start_PC() + i * 8);
      if (!ptx || ptx->ptx_matrix.wgmma != ptx_matrix_metadata::MMA) continue;
      const auto n = ptx->ptx_matrix.n;
      inst_trace_t trace; const auto opcode = "HGMMA.64x" + std::to_string(n) + "x16.F32";
      base(trace,opcode.c_str()); trace.is_gmma_commit_group = 1;
      trace_warp_inst_t decoded(gpu->getShaderCoreConfig());
      require(decoded.parse_from_trace_struct(trace,&Hopper_OpcodeMap,&config,nullptr),"SASS GMMA decode failed");
      require(decoded.op == ptx->op && decoded.latency == ptx->latency &&
              decoded.initiation_interval == ptx->initiation_interval &&
              decoded.m_is_gmma_commit_group,"PTX/SASS GMMA metadata differs");
      printf("SASS_ORACLE_RESULT status=PASS origin=synthetic_decoder opcode=%s n=%u latency=%u ii=%u op=%u commit=1\n",
             opcode.c_str(),n,decoded.latency,decoded.initiation_interval,unsigned(decoded.op));
      ++matches;
    }
    require(matches == 3,"expected three PTX MMA operations");
    for (unsigned limit : {0u,1u,7u}) {
      inst_trace_t trace; base(trace,"WARPGROUP.DEPBAR.LE"); trace.imm = limit;
      trace_warp_inst_t decoded(gpu->getShaderCoreConfig());
      decoded.parse_from_trace_struct(trace,&Hopper_OpcodeMap,&config,nullptr);
      require(decoded.m_is_depbar && decoded.m_depbar_group_no == limit,"SASS wait limit differs");
      printf("SASS_WAIT_RESULT status=PASS origin=synthetic_decoder limit=%u\n",limit);
    }
    for (unsigned mask : {0xffffffffU,0u}) {
      inst_trace_t trace; base(trace,"UTMALDG.1D.S.G",mask);
      trace.tma_memadd_info.reset(new tma_inst_memaddr_info_t);
      trace.tma_memadd_info->width = 4;
      trace.tma_memadd_info->addrs = {0x1000,0x1004,0x1008,0x100c};
      trace.tma_mbar_addr = 256; trace.tma_byte_count = 16; trace.tma_oob_byte_count = 4;
      trace.tma_is_multicast = true; trace.tma_multicast_cta_mask = 3;
      trace_warp_inst_t decoded(gpu->getShaderCoreConfig());
      decoded.parse_from_trace_struct(trace,&Hopper_OpcodeMap,&config,nullptr);
      require(decoded.is_tma_load() && decoded.get_tma_byte_count() == (mask ? 16u : 0u) &&
              decoded.get_tma_oob_byte_count() == (mask ? 4u : 0u) &&
              decoded.get_tma_mbar_addr() == (mask ? 256u : 0u) &&
              decoded.get_tma_multicast_cta_mask() == (mask ? 3u : 0u) &&
              decoded.get_cuda_cluster_rank() == 1,"SASS TMA dynamic fields differ");
      printf("SASS_TMA_RESULT status=PASS origin=synthetic_decoder mask=%u bytes=%zu oob=%zu mbar=%u multicast_mask=%u\n",
             mask,decoded.get_tma_byte_count(),decoded.get_tma_oob_byte_count(),decoded.get_tma_mbar_addr(),decoded.get_tma_multicast_cta_mask());
    }
    return 0;
  } catch (const std::exception &error) { fprintf(stderr,"SASS_ORACLE_FAILURE %s\n",error.what()); return 1; }
}
