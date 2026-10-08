ROOT := $(abspath ../../..)
SIM := $(ROOT)/gpu-simulator
BUILD := $(SIM)/build/release
SIM_RUNTIME ?= $(BUILD)/gpgpu-sim/libcudart.so
CXX := g++
CXXFLAGS := -std=c++17 -O2 -I$(SIM)/gpgpu-sim/src -I$(SIM)/gpgpu-sim -I$(SIM)/gpgpu-sim/libcuda -I/usr/local/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include
LIBS := $(SIM_RUNTIME) -Wl,-rpath,$(dir $(SIM_RUNTIME)) -lz -lGL -lpthread -ldl -lrt -lm

all: stage2_harness

stage2_harness: stage2_harness.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

clean:
	rm -f stage2_harness
