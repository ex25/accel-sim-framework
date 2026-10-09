ROOT := $(abspath ../../..)
SIM := $(ROOT)/gpu-simulator/gpgpu-sim
SIM_RUNTIME ?= /tmp/hopper-stage3-build/libcudart.so
CXX := /usr/bin/g++
CXXFLAGS := -std=c++17 -O2 -I$(SIM)/src -I$(SIM) -I$(SIM)/libcuda -I/usr/local/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include
LIBS := $(SIM_RUNTIME) -Wl,-rpath,$(dir $(SIM_RUNTIME)) -lz -lGL -lpthread -ldl -lrt -lm

all: tma_harness tma_frontend tma_reference tma_detailed

tma_harness: tma_harness.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

tma_frontend: tma_frontend.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

tma_reference: tma_reference.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

tma_detailed: tma_detailed.cc $(SIM_RUNTIME)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

clean:
	rm -f tma_harness tma_frontend tma_reference tma_detailed
