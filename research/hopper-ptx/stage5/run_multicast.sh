#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_DIR=${1:-/tmp/hopper-stage5-multicast}
mkdir -p "$RUN_DIR"
cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config" "$RUN_DIR/gpgpusim.config"
cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt" "$RUN_DIR/config_hopper_islip.icnt"
sed -i 's#^-gpgpu_ptx_sim_mode[[:space:]].*#-gpgpu_ptx_sim_mode 0#; s#^-gpgpu_max_cycle[[:space:]].*#-gpgpu_max_cycle 200000#' "$RUN_DIR/gpgpusim.config"
make -C "$HERE" tma_detailed
export LD_LIBRARY_PATH=/tmp/hopper-stage3-build
(cd "$RUN_DIR"; "$HERE/tma_detailed" "$HERE/tma_detailed.ptx" multicast 2>&1 | tee multicast.log | grep -E 'TMA_DETAIL event=|CTA_RESULT|VISIBILITY_RESULT|METADATA_RESULT|DETAILED_RESULT')
for bad in empty unknown not_started offset duplicate unregistered bad_descriptor overlap high_rank prefetch; do
  ptx="$HERE/tma_detailed.ptx"
  if [[ "$bad" == high_rank ]]; then ptx="$HERE/tma_detailed_3d.ptx"; fi
  if [[ "$bad" == prefetch ]]; then ptx="$HERE/tma_detailed_prefetch.ptx"; fi
  if (cd "$RUN_DIR"; "$HERE/tma_detailed" "$ptx" "$bad") >"$RUN_DIR/$bad.out" 2>"$RUN_DIR/$bad.log"; then
    echo "NEGATIVE_FAILURE scenario=$bad"; exit 1
  fi
  grep -Eq 'empty or invalid multicast mask|unknown multicast rank|not started/initialized|barrier offset mismatch|duplicate PTX TMA target|descriptor address is not registered|descriptor rank/element size|overlapping descriptor strides|only 1D/2D|prefetch timing is not implemented' "$RUN_DIR/$bad.out" "$RUN_DIR/$bad.log"
  if grep -Eq 'event=(issue|global_request|global_response|shared_write|completion)' "$RUN_DIR/$bad.out" "$RUN_DIR/$bad.log"; then
    echo "NEGATIVE_FAILURE scenario=$bad wrote-before-failure"; exit 1
  fi
  echo "NEGATIVE_RESULT status=PASS scenario=$bad"
done
