NVCC ?= nvcc
BUILD_DIR ?= build
TARGET := $(BUILD_DIR)/sass_ptx_gap

NVCCFLAGS := -O3 -std=c++17 --cudart shared \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_80,code=compute_80

.PHONY: all clean inspect

all: $(TARGET)

$(TARGET): sass_ptx_gap.cu
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Xptxas=-v $< -o $@

inspect: $(TARGET)
	cuobjdump --dump-ptx $(TARGET) >/dev/null
	cuobjdump --dump-sass $(TARGET) >/dev/null
	ldd $(TARGET)

clean:
	rm -rf $(BUILD_DIR)
