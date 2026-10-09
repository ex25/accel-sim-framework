#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>
#include "cuda-sim/memory.h"

#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "libcuda/gpgpu_context.h"

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--overflow") {
    try {
      function_info::checked_param_offset(
          std::numeric_limits<unsigned>::max() - 3, 8, 64);
    } catch (const std::overflow_error &) {
      std::puts("PARAM_OVERFLOW status=PASS");
      return 0;
    }
    return 1;
  }
  if (argc != 3) return 2;
  setenv("PTX_SIM_KERNELFILE", argv[1], 1);
  gpgpu_context context;
  context.gpgpu_ptx_sim_init_perf();
  symbol_table *symbols = context.gpgpu_ptx_sim_load_ptx_from_filename(argv[1]);
  function_info *entry = symbols->lookup_function(argv[2]);
  if (!entry) return 1;
  std::printf("PARAM_LAYOUT total=%u count=%u\n", entry->get_args_aligned_size(),
              entry->num_args());
  const unsigned total = entry->get_args_aligned_size();
  memory_space_impl<32> memory("stage8-params", 4);
  std::vector<unsigned char> expected(total + 8, 0xcd);
  memory.write(0, expected.size(), expected.data(), NULL, NULL);
  std::vector<std::vector<unsigned char> > data(entry->num_args());
  for (unsigned i = 0; i < entry->num_args(); ++i) {
    const symbol *arg = entry->get_arg(i);
    const std::pair<size_t, unsigned> config = entry->get_param_config(i);
    std::printf("PARAM index=%u name=%s size=%zu offset=%u address=%llu\n", i,
                arg->name().c_str(), config.first, config.second,
                static_cast<unsigned long long>(arg->get_address()));
    // Deliberately truncate a block and supply wrong host offsets. Neither
    // may move the next formal argument or overwrite padding bytes.
    size_t copied = config.first > 4 ? config.first - 1 : config.first;
    data[i].assign(copied, 0x20 + i);
    gpgpu_ptx_sim_arg value(data[i].data(), copied, 0);
    entry->add_param_data(i, &value);
    for (size_t j = 0; j < copied; ++j)
      expected[config.second + j] = 0x20 + i;
  }
  entry->finalize(&memory);
  std::vector<unsigned char> actual(expected.size());
  memory.read(0, actual.size(), actual.data());
  if (actual != expected) return 1;
  std::puts("PARAM_COPY status=PASS guard_bytes=8 truncated=true");
  return 0;
}
