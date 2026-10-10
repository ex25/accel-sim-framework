#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=${1:?provide a fresh run directory}
MODE=${2:-1}
[[ "$MODE" == 0 || "$MODE" == 1 ]]
ulimit -c 0
mkdir -p "$RUN"
make -C "$HERE" clean all > "$RUN/build.log" 2>&1
cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/"{gpgpusim.config,config_hopper_islip.icnt} "$RUN/"
sed -i "s#^-gpgpu_ptx_sim_mode[[:space:]].*#-gpgpu_ptx_sim_mode $MODE#" "$RUN/gpgpusim.config"
export CUDA_INSTALL_PATH=${CUDA_INSTALL_PATH:-/usr/local/cuda}
export GPGPUSIM_ROOT="$ROOT/gpu-simulator/gpgpu-sim"
export GPGPUSIM_CONFIG=gcc-13.4.0/cuda-12080/release
export LD_LIBRARY_PATH="/tmp/hopper-stage3-build:$GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG:${LD_LIBRARY_PATH:-}"
(cd "$RUN" && "$HERE/module_launch_probe" "$HERE/runtime_kernel.ptx" "$HERE/runtime_numeric.ptx" > runtime.log 2>&1)
grep -E '^RUNTIME_RESULT status=PASS' "$RUN/runtime.log"
