NVCC ?= nvcc
BUILD_DIR ?= build
TARGET := $(BUILD_DIR)/sm90_ptx_baseline

NVCCFLAGS := -O3 -std=c++17 --cudart shared \
	-gencode arch=compute_90,code=compute_90

.PHONY: all clean inspect

all: $(TARGET)

$(TARGET): sm90_ptx_baseline.cu
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

inspect: $(TARGET)
	cuobjdump --dump-ptx $(TARGET) | head -40
	ldd $(TARGET)

clean:
	rm -rf $(BUILD_DIR)
