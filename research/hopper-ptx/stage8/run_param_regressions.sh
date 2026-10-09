#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
RUN=${1:?provide a fresh regression directory}
PYTHON_BIN=${PYTHON_BIN:-/home/wangjz/miniconda3/envs/in-switch/bin/python3}
mkdir -p "$RUN/frontend"
make -C "$ROOT/research/hopper-ptx/stage3" clean all > "$RUN/frontend-build.log" 2>&1
make -C "$HERE" param_layout_probe > "$RUN/layout-build.log" 2>&1
"$PYTHON_BIN" "$HERE/param_layout_probe.py" "$RUN/layout" > "$RUN/layout.log" 2>&1
cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/"{gpgpusim.config,config_hopper_islip.icnt} "$RUN/frontend/"
export LD_LIBRARY_PATH=/tmp/hopper-stage3-build
for case in types ldmatrix wgmma; do
  (cd "$RUN/frontend"; "$ROOT/research/hopper-ptx/stage3/frontend_probe" "$ROOT/research/hopper-ptx/stage3/frontend_$case.ptx" "frontend_$case" > "$case.log" 2>&1)
  rg '^FRONTEND status=PASS' "$RUN/frontend/$case.log"
done
for case in bad_operand bad_modifier; do
  code=0
  (cd "$RUN/frontend"; "$ROOT/research/hopper-ptx/stage3/frontend_probe" "$ROOT/research/hopper-ptx/stage3/frontend_$case.ptx" frontend_bad > "$case.log" 2>&1) || code=$?
  [[ $code != 0 ]]
  rg -q 'matrix register tuple length mismatch|unsupported matrix opcode/modifier' "$RUN/frontend/$case.log"
done
for stage in 4 5 6 7; do
  # These harness Makefiles do not all track simulator header dependencies.
  make -C "$ROOT/research/hopper-ptx/stage$stage" clean all > "$RUN/stage$stage-build.log" 2>&1
  bash "$ROOT/research/hopper-ptx/stage$stage/run.sh" "$RUN/stage$stage" > "$RUN/stage$stage.log" 2>&1
  echo "STAGE_REGRESSION status=PASS stage=$stage"
done
bash "$ROOT/research/hopper-ptx/stage5/run_detailed.sh" "$RUN/stage5-detailed" > "$RUN/stage5-detailed.log" 2>&1
bash "$ROOT/research/hopper-ptx/stage5/run_multicast.sh" "$RUN/stage5-multicast" > "$RUN/stage5-multicast.log" 2>&1
cat "$RUN/layout.log"
echo 'PARAM_REGRESSIONS status=PASS stages=3,4,5,6,7'
