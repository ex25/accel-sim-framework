#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "gpgpu-sim/gpu-sim.h"
#include "libcuda/gpgpu_context.h"

// Logical matrices are independent of the simulator's fragment/descriptor decoder.
static float a(unsigned row, unsigned k) { return 0.25f*float(int((row*3+k*5)%11)-5); }
static float b(unsigned k, unsigned col) { return 0.5f*float(int((k*7+col*2)%13)-6); }
static uint16_t fp16(float value) {
  half_float::half h(value);
  uint16_t bits; memcpy(&bits, &h, 2); return bits;
}
static size_t offset(unsigned row, unsigned k, unsigned mn, bool trans) {
  if (trans) return 2*(row%8) + 128*(row/8) + 16*(k%8) + mn*16*(k/8);
  return 16*(row%8) + 256*(row/8) + 2*(k%8) + 128*(k/8);
}
int main(int argc, char **argv) {
  if (argc < 4 || argc > 5) return 2;
  const std::string mode = argv[2], name = argv[3];
  const unsigned n = name.find("176") != std::string::npos ? 176 : 128;
  const bool trans = name == "ss176" || name.find("rs176") == 0;
  const bool registers = name.find("rs") == 0;
  const unsigned count = name == "empty" ? 0 : name.find("timing") == 0 ? 3 : name == "ss176" || name == "rs128" ? 2 : 1;
  const bool scale_d = name.find("ss128") != 0 && name != "rs176_zero";
  const float sign = name == "rs128" ? -1 : 1;
  const unsigned payload = (64+n)*32;
  setvbuf(stdout, nullptr, _IOLBF, 0);
  setenv("PTX_SIM_KERNELFILE", argv[1], 1);
  try {
    gpgpu_context context;
    auto *gpu = context.gpgpu_ptx_sim_init_perf();
    auto *symbols = context.gpgpu_ptx_sim_load_ptx_from_filename(argv[1]);
    auto *entry = symbols->lookup_function("wgmma_kernel");
    if (!entry) return 2;
    entry->do_pdom(); entry->set_pdom();
    unsigned mma_count=0;
    for (unsigned i=0; i<entry->get_function_size(); ++i) {
      const auto *insn=entry->get_instruction(entry->get_start_PC()+i*8);
      if (!insn || insn->ptx_matrix.kind != ptx_matrix_metadata::WGMMA) continue;
      const auto &m=insn->ptx_matrix;
      if (!m.functional_supported || !m.timing_supported) {
        fprintf(stderr,"METADATA_FAILURE supported=%u,%u kind=%u\n",m.functional_supported,m.timing_supported,unsigned(m.wgmma));return 1;
      }
      if (m.wgmma==ptx_matrix_metadata::MMA) {
        if (m.n != n || m.accumulator_registers.size() != n/2 ||
            insn->latency != n/2 || insn->initiation_interval != n/2 || !insn->is_gmma()) {
          fprintf(stderr,"METADATA_FAILURE n=%u acc=%zu latency=%u ii=%u op=%u\n",m.n,m.accumulator_registers.size(),insn->latency,insn->initiation_interval,unsigned(insn->op)); return 1;
        }
        ++mma_count;
      }
      if (m.wgmma==ptx_matrix_metadata::COMMIT && !insn->m_is_gmma_commit_group) {
        fprintf(stderr,"METADATA_FAILURE commit flag missing\n");return 1;
      }
    }
    printf("FRONTEND_RESULT status=PASS n=%u mma=%u latency=%u ii=%u\n",n,mma_count,n/2,n/2);
    if (mode == "frontend") return 0;
    std::vector<uint16_t> input((payload+2048)/2);
    for (unsigned row=0; row<64; ++row)
      for (unsigned k=0; k<16; ++k) input[offset(row,k,64,trans)/2] = fp16(a(row,k));
    for (unsigned col=0; col<n; ++col)
      for (unsigned k=0; k<16; ++k) input[(2048+offset(col,k,n,trans))/2] = fp16(b(k,col));
    for (unsigned tid=0; tid<128; ++tid)
      for (unsigned e=0; e<8; ++e) {
        unsigned row=16*(tid/32)+(tid%32)/4+8*((e/2)%2);
        unsigned k=2*(tid%4)+e%2+8*(e/4);
        input[payload/2+tid*8+e]=fp16(a(row,k));
      }
    auto *source = gpu->gpu_malloc(input.size()*2);
    auto *output = gpu->gpu_malloc(64*n*4);
    gpu->memcpy_to_gpu(reinterpret_cast<size_t>(source),input.data(),input.size()*2);
    std::vector<float> sentinel(64*n,-99999);
    gpu->memcpy_to_gpu(reinterpret_cast<size_t>(output),sentinel.data(),sentinel.size()*4);
    gpgpu_ptx_sim_arg_list_t args;
    args.push_back(gpgpu_ptx_sim_arg(&source,sizeof(source),8));
    args.push_back(gpgpu_ptx_sim_arg(&output,sizeof(output),0));
    unsigned threads=argc==5 ? std::stoul(argv[4]) : 128;
    auto *kernel=context.func_sim->gpgpu_opencl_ptx_sim_init_grid(entry,args,dim3(1,1,1),dim3(threads,1,1),gpu);
    unsigned cycles=0;
    if (mode == "functional") context.func_sim->gpgpu_cuda_ptx_sim_main_func(*kernel,true);
    else {
      gpu->init(); gpu->launch(kernel);
      while (gpu->active() && cycles<200000) {gpu->cycle(); ++cycles;}
      if (gpu->active() || !kernel->done()) throw std::runtime_error("WGMMA detailed simulation did not finish");
    }
    std::vector<float> actual(64*n);
    gpu->memcpy_from_gpu(actual.data(),reinterpret_cast<size_t>(output),actual.size()*4);
    // Reconstruct each logical output from its fragment; independent scalar GEMM.
    float max_error=0;
    for (unsigned row=0; row<64; ++row) for (unsigned col=0; col<n; ++col) {
      float dot=0;
      for (unsigned k=0; k<16; ++k) dot+=a(row,k)*b(k,col);
      float expected=(scale_d ? 0.5f : 0.f)+count*sign*dot;
      unsigned tid=(row/16)*32+(row%8)*4+(col%8)/2;
      unsigned reg=(col/8)*4+(row%16)/8*2+col%2;
      float error=std::fabs(actual[tid*(n/2)+reg]-expected);
      max_error=std::max(max_error,error);
      if (error || !std::isfinite(actual[tid*(n/2)+reg])) {
        fprintf(stderr,"MISMATCH row=%u col=%u actual=%g expected=%g\n",row,col,actual[tid*(n/2)+reg],expected);return 1;
      }
    }
    const auto &s=gpu->ptx_wgmma_stats;
    if (s.mma != count || (mode != "functional" && s.completion != count*4)) return 1;
    printf("MATRIX_RESULT status=PASS case=%s mode=%s elements=%u max_error=%g source=%s\n",name.c_str(),mode.c_str(),64*n,max_error,registers ? "register/shared" : "shared/shared");
    printf("TIMING_RESULT status=PASS case=%s cycles=%u mma=%llu commit=%llu wait=%llu completion=%llu blocked_checks=%llu\n",name.c_str(),cycles,(unsigned long long)s.mma,(unsigned long long)s.commit,(unsigned long long)s.wait,(unsigned long long)s.completion,(unsigned long long)s.wait_checks);
  } catch (const std::exception &e) {
    fprintf(stderr,"WGMMA_NEGATIVE diagnostic=%s\n",e.what());return 3;
  }
}
