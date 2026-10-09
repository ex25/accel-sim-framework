ROOT := $(abspath ../../..)
SIM := $(ROOT)/gpu-simulator/gpgpu-sim
SIM_RUNTIME ?= /tmp/hopper-stage3-build/libcudart.so
CXX := /usr/bin/g++
NVCC := /usr/local/cuda/bin/nvcc
CPPFLAGS := -I$(SIM)/src -I$(SIM) -I$(SIM)/libcuda -I/usr/local/cuda/include -I$(ROOT)/util/tracer_nvbit/tracer_tool -I$(ROOT)/gpu-simulator/trace-driven -I$(ROOT)/gpu-simulator/trace-parser -I.
CXXFLAGS := -std=c++17 -O2 -ffunction-sections -fdata-sections
LIBS := $(SIM_RUNTIME) -Wl,-rpath,$(dir $(SIM_RUNTIME)) -Wl,--gc-sections -lz -lzstd -lGL -lpthread -ldl -lrt -lm
all: attention attention_harness param_layout_probe readiness_execution_probe sass_oracle trace_sim trace_audit
attention: attention.cu
	$(NVCC) -ccbin $(CXX) -std=c++17 -O3 -arch=sm_80 -Xptxas=-v $< -o $@
attention_harness: attention_harness.cc $(SIM_RUNTIME)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LIBS)
param_layout_probe: param_layout_probe.cc $(SIM_RUNTIME)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LIBS)
readiness_execution_probe: readiness_execution_probe.cc $(SIM_RUNTIME)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LIBS)
trace_driven.o: $(ROOT)/gpu-simulator/trace-driven/trace_driven.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
trace_parser.o: $(ROOT)/gpu-simulator/trace-parser/trace_parser.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
sass_oracle: sass_oracle.cc trace_driven.o trace_parser.o $(SIM_RUNTIME)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< trace_driven.o trace_parser.o -o $@ $(LIBS)
trace_audit: trace_audit.cc trace_parser.o $(SIM_RUNTIME)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< trace_parser.o -o $@ $(LIBS)
accelsim_version.h:
	python3 -c 'import subprocess; from pathlib import Path; outer=subprocess.check_output(["git","-C","$(ROOT)","rev-parse","HEAD"],text=True).strip(); inner=subprocess.check_output(["git","-C","$(SIM)","rev-parse","HEAD"],text=True).strip(); Path("$@").write_text("const char *g_accelsim_version=\"stage8_outer_"+outer+"_inner_"+inner+"\";\n")'
trace_sim: $(ROOT)/gpu-simulator/accel-sim.cc $(ROOT)/gpu-simulator/main.cc trace_driven.o trace_parser.o accelsim_version.h $(SIM_RUNTIME)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(ROOT)/gpu-simulator/accel-sim.cc $(ROOT)/gpu-simulator/main.cc trace_driven.o trace_parser.o -o $@ $(LIBS)
clean:
	rm -f attention attention_harness param_layout_probe readiness_execution_probe sass_oracle trace_sim trace_audit trace_driven.o trace_parser.o accelsim_version.h
.PHONY: all clean
