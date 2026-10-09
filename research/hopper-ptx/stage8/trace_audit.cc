// Audit a real NVBit v5 text trace with the repository's address decoder.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include "../../../gpu-simulator/trace-parser/trace_parser.h"

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  try {
    std::ifstream file(argv[1]); if (!file) throw std::runtime_error("missing trace");
    std::string line; uint64_t instructions=0, scalar=0, reads=0, writes=0, sectors=0, barriers=0;
    bool version=false, binary=false, grid=false, block=false;
    std::map<std::string,uint64_t> opcodes;
    while (std::getline(file,line)) {
      if (line == "-accelsim tracer version = 5") version=true;
      if (line == "-binary version = 80") binary=true;
      if (line == "-grid dim = (1,1,1)") grid=true;
      if (line == "-block dim = (128,1,1)") block=true;
      if (line.empty() || line[0]=='-' || line[0]=='#' || line.find('=') != std::string::npos) continue;
      inst_trace_t trace;
      if (!trace.parse_from_string(line,5,0,dim3(0,0,0),dim3(0,0,0),dim3(0,0,0),0))
        throw std::runtime_error("trace parse failed");
      const auto opcode=trace.opcode.substr(0,trace.opcode.find('.'));
      const unsigned lanes=__builtin_popcount(trace.mask);
      ++instructions; scalar+=lanes; opcodes[opcode]+=lanes;
      if (opcode=="BAR" && lanes) ++barriers;
      if ((opcode=="LDG" || opcode=="STG") && lanes) {
        if (!trace.memadd_info || trace.memadd_info->width <= 0) throw std::runtime_error("missing global memory addresses");
        const auto width=trace.memadd_info->width;
        (opcode=="LDG" ? reads : writes) += uint64_t(lanes)*width;
        std::set<uint64_t> transaction_sectors;
        for (unsigned lane=0; lane<32; ++lane) if (trace.mask & (1u<<lane)) {
          const uint64_t address=trace.memadd_info->addrs[lane];
          for (auto sector=address/32; sector<=(address+width-1)/32; ++sector) transaction_sectors.insert(sector);
        }
        sectors+=transaction_sectors.size();
      }
    }
    // One row/head of the frozen attention: Q, K, V reads and FP16 output.
    if (!version || !binary || !grid || !block || reads!=65792 || writes!=256 || barriers!=12)
      throw std::runtime_error("captured trace does not match frozen one-CTA workload");
    printf("A800_TRACE_RESULT status=PASS origin=hardware_capture warp_instructions=%llu scalar_instructions=%llu global_read_bytes=%llu global_write_bytes=%llu requested_32B_sectors=%llu barrier_warp_events=%llu\n",
      (unsigned long long)instructions,(unsigned long long)scalar,(unsigned long long)reads,(unsigned long long)writes,(unsigned long long)sectors,(unsigned long long)barriers);
    for (const auto &item:opcodes) printf("A800_TRACE_OPCODE opcode=%s active_scalar_count=%llu\n",item.first.c_str(),(unsigned long long)item.second);
    return 0;
  } catch (const std::exception &error) { fprintf(stderr,"A800_TRACE_FAILURE %s\n",error.what());return 1; }
}
