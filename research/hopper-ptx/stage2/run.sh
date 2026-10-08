#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_DIR=${1:-"$HERE/artifacts/run"}
mkdir -p "$RUN_DIR"
cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config" "$RUN_DIR/gpgpusim.config"
cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt" "$RUN_DIR/config_hopper_islip.icnt"
sed -i 's#^-gpgpu_ptx_sim_mode[[:space:]].*#-gpgpu_ptx_sim_mode 1#' "$RUN_DIR/gpgpusim.config"
make -C "$HERE" all
export CUDA_INSTALL_PATH=${CUDA_INSTALL_PATH:-/usr/local/cuda}
export GPGPUSIM_ROOT="$ROOT/gpu-simulator/gpgpu-sim"
export GPGPUSIM_CONFIG=gcc-13.4.0/cuda-12080/release
export LD_LIBRARY_PATH="$GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG:${LD_LIBRARY_PATH:-}"
unset PTX_SIM_MODE_FUNC
cd "$RUN_DIR"
"$HERE/stage2_harness" "$HERE/stage2_kernel.ptx" 2>&1 | tee run.log
