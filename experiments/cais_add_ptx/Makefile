NVCC ?= nvcc
PYTHON ?= python3
BUILD_DIR ?= build
TARGET := $(BUILD_DIR)/cais_add_ptx

NVCCFLAGS := -O3 -std=c++17 --cudart shared \
	-gencode arch=compute_90,code=compute_90

.PHONY: all clean inspect ptx

all: $(TARGET) $(BUILD_DIR)/control.ptx $(BUILD_DIR)/custom.ptx

$(TARGET): cais_add_ptx.cu
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/control.ptx $(BUILD_DIR)/custom.ptx: cais_add_ptx.cu make_custom_ptx.py
	mkdir -p $(BUILD_DIR)
	$(PYTHON) make_custom_ptx.py $< $(BUILD_DIR)

ptx: $(BUILD_DIR)/control.ptx $(BUILD_DIR)/custom.ptx

inspect: $(TARGET) ptx
	cuobjdump --dump-ptx $(TARGET) | head -40
	ldd $(TARGET)

clean:
	rm -rf $(BUILD_DIR)
