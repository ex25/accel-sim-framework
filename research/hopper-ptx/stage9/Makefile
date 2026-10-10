ROOT := $(abspath ../../..)
SIM := $(ROOT)/gpu-simulator/gpgpu-sim
SIM_RUNTIME ?= /tmp/hopper-stage3-build/libcudart.so
CXX := /usr/bin/g++
CXXFLAGS := -std=c++17 -O2 -I/usr/local/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include
LIBS := $(SIM_RUNTIME) -Wl,-rpath,$(dir $(SIM_RUNTIME)) -lpthread -ldl -lrt -lm
all: module_launch_probe
module_launch_probe: module_launch_probe.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)
clean:
	rm -f module_launch_probe
