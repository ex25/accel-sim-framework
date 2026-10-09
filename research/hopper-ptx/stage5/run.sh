#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_DIR=${1:-/tmp/hopper-stage5-run}
mkdir -p "$RUN_DIR"
cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config" "$RUN_DIR/gpgpusim.config"
cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt" "$RUN_DIR/config_hopper_islip.icnt"
sed -i 's#^-gpgpu_ptx_sim_mode[[:space:]].*#-gpgpu_ptx_sim_mode 1#' "$RUN_DIR/gpgpusim.config"
make -C "$HERE" all
export LD_LIBRARY_PATH=/tmp/hopper-stage3-build
(cd "$RUN_DIR"; "$HERE/tma_frontend" "$HERE/tma_frontend.ptx" 2>&1 | tee frontend.log | grep -E 'TMA index|FRONTEND_RESULT')
(cd "$RUN_DIR"; "$HERE/tma_harness" "$HERE/tma_kernel.ptx" 2>&1 | tee functional.log | grep -E 'TMA_RESULT|PTX_EVENT_RESULT')
(cd "$RUN_DIR"; "$HERE/tma_reference" 2>&1 | tee reference.log | grep -E 'REFERENCE_RESULT|EVENT_RESULT')
for bad in tma_bad_rank.ptx tma_bad_store.ptx tma_bad_rank_operands.ptx; do
  if (cd "$RUN_DIR"; "$HERE/tma_frontend" "$HERE/$bad") >"$RUN_DIR/$bad.out" 2>"$RUN_DIR/$bad.log"; then
    echo "NEGATIVE_FAILURE file=$bad"; exit 1
  fi
  grep -Eq 'unsupported TMA opcode/modifier|TMA coordinate rank mismatch' "$RUN_DIR/$bad.out" "$RUN_DIR/$bad.log"
  echo "NEGATIVE_RESULT status=PASS file=$bad"
done
if (cd "$RUN_DIR"; "$HERE/tma_harness" "$HERE/tma_kernel.ptx" unregistered) >"$RUN_DIR/unregistered.out" 2>"$RUN_DIR/unregistered.log"; then
  echo "NEGATIVE_FAILURE file=unregistered"; exit 1
fi
grep -q 'TMA descriptor address is not registered' "$RUN_DIR/unregistered.log"
echo "NEGATIVE_RESULT status=PASS file=unregistered"
