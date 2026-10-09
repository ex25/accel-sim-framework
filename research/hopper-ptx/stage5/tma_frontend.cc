#include <cstdio>
#include <cstdlib>
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "libcuda/gpgpu_context.h"
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  setenv("PTX_SIM_KERNELFILE", argv[1], 1);
  gpgpu_context context;
  context.gpgpu_ptx_sim_init_perf();
  symbol_table *symbols = context.gpgpu_ptx_sim_load_ptx_from_filename(argv[1]);
  function_info *entry = symbols->lookup_function("tma_frontend");
  if (!entry) return 1;
  unsigned found = 0;
  for (unsigned i = 0; i < entry->get_function_size(); ++i) {
    const ptx_instruction *insn = entry->get_instruction(entry->get_start_PC() + i * 8);
    if (!insn || insn->ptx_tma.kind == ptx_tma_op::NONE) continue;
    ++found;
    std::printf("TMA index=%u opcode=%s kind=%u rank=%d\n", i,
                insn->ptx_tma.opcode.c_str(),
                static_cast<unsigned>(insn->ptx_tma.kind),
                static_cast<int>(insn->ptx_tma.rank));
  }
  if (found != 4) return 1;
  std::printf("FRONTEND_RESULT status=PASS tma_instructions=%u\n", found);
  return 0;
}
