#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_DIR=${1:-/tmp/hopper-stage5-detailed}
mkdir -p "$RUN_DIR"
cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config" "$RUN_DIR/gpgpusim.config"
cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt" "$RUN_DIR/config_hopper_islip.icnt"
sed -i 's#^-gpgpu_ptx_sim_mode[[:space:]].*#-gpgpu_ptx_sim_mode 0#; s#^-gpgpu_max_cycle[[:space:]].*#-gpgpu_max_cycle 200000#' "$RUN_DIR/gpgpusim.config"
make -C "$HERE" tma_detailed
export LD_LIBRARY_PATH=/tmp/hopper-stage3-build
for scenario in single local one_d all_oob; do
  ptx="$HERE/tma_detailed.ptx"
  if [[ "$scenario" == one_d ]]; then ptx="$HERE/tma_detailed_1d.ptx"; fi
  if [[ "$scenario" == local ]]; then ptx="$HERE/tma_detailed_local.ptx"; fi
  (cd "$RUN_DIR"; "$HERE/tma_detailed" "$ptx" "$scenario" 2>&1 | tee "detailed-$scenario.log" | grep -E 'TMA_DETAIL event=|CTA_RESULT|VISIBILITY_RESULT|METADATA_RESULT|DETAILED_RESULT')
done
