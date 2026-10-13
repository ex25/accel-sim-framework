#include <cstdio>
#include <cstdlib>
#include <string>

#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "libcuda/gpgpu_context.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: frontend_probe PTX KERNEL\n");
    return 2;
  }
  setenv("PTX_SIM_KERNELFILE", argv[1], 1);
  gpgpu_context context;
  context.gpgpu_ptx_sim_init_perf();
  symbol_table *symbols = context.gpgpu_ptx_sim_load_ptx_from_filename(argv[1]);
  function_info *entry = symbols->lookup_function(argv[2]);
  if (!entry) {
    std::fprintf(stderr, "missing kernel %s\n", argv[2]);
    return 1;
  }
  std::printf("FRONTEND status=PASS target=%s target_number=%u instructions=%u\n",
              entry->get_ptx_version().target_string().c_str(),
              entry->get_ptx_version().target(), entry->get_function_size());
  for (unsigned i = 0; i < entry->get_function_size(); ++i) {
    const ptx_instruction *insn = entry->get_instruction(entry->get_start_PC() + i * 8);
    if (!insn) continue;
    std::printf("INSN index=%u opcode=%s operands=%u source=%s:%u\n", i,
                insn->get_opcode_cstr(), insn->get_num_operands(),
                insn->source_file(), insn->source_line());
    if (insn->ptx_matrix.kind != ptx_matrix_metadata::NONE) {
      const auto &m = insn->ptx_matrix;
      std::printf("MATRIX opcode=%s m=%u n=%u k=%u matrices=%u trans=%u functional=%u timing=%u",
                  m.opcode.c_str(), m.m, m.n, m.k, m.matrices,
                  unsigned(m.transpose), unsigned(m.functional_supported),
                  unsigned(m.timing_supported));
      for (unsigned p = 0; p < insn->get_num_operands(); ++p) {
        const auto &op = insn->operand_lookup(p);
        if (op.is_vector()) {
          std::printf(" tuple%u=%u", p, op.get_vect_nelem());
          for (unsigned r = 0; r < op.get_vect_nelem(); ++r)
            std::printf(" %s", op.vec_symbol(r)->name().c_str());
        }
      }
      std::puts("");
    }
  }
  return 0;
}
