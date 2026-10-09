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
export LD_LIBRARY_PATH="/tmp/hopper-stage3-build:$GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG:${LD_LIBRARY_PATH:-}"
cd "$RUN_DIR"
"$HERE/sync_harness" "$HERE/mbarrier_kernel.ptx" 2>&1 | tee sync.log
"$HERE/mbarrier_frontend" "$HERE/mbarrier_frontend.ptx" 2>&1 | tee frontend.log
"$HERE/state_probe" 2>&1 | tee state.log

for bad in mbarrier_bad_modifier.ptx mbarrier_bad_operands.ptx mbarrier_cluster_wait.ptx; do
  if "$HERE/mbarrier_frontend" "$HERE/$bad" >/dev/null 2>"$RUN_DIR/$bad.log"; then
    echo "NEGATIVE_FAILURE file=$bad"
    exit 1
  fi
  echo "NEGATIVE_RESULT status=PASS file=$bad"
done
