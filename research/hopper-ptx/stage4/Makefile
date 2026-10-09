ROOT := $(abspath ../../..)
SIM := $(ROOT)/gpu-simulator/gpgpu-sim
SIM_RUNTIME ?= /tmp/hopper-stage3-build/libcudart.so
CXX := /usr/bin/g++
CXXFLAGS := -std=c++17 -O2 -I$(SIM)/src -I$(SIM) -I$(SIM)/libcuda -I/usr/local/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include
LIBS := $(SIM_RUNTIME) -Wl,-rpath,$(dir $(SIM_RUNTIME)) -lz -lGL -lpthread -ldl -lrt -lm

all: sync_harness mbarrier_frontend state_probe

sync_harness: sync_harness.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

mbarrier_frontend: mbarrier_frontend.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

state_probe: state_probe.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

clean:
	rm -f sync_harness mbarrier_frontend state_probe
