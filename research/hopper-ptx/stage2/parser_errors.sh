#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
RUN_DIR=${1:-"$HERE/artifacts/parser-errors"}
mkdir -p "$RUN_DIR"
cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config" "$RUN_DIR/gpgpusim.config"
cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt" "$RUN_DIR/config_hopper_islip.icnt"
sed -i 's#^-gpgpu_ptx_sim_mode[[:space:]].*#-gpgpu_ptx_sim_mode 1#' "$RUN_DIR/gpgpusim.config"
make -C "$HERE" all >/dev/null
export GPGPUSIM_ROOT="$ROOT/gpu-simulator/gpgpu-sim"
export GPGPUSIM_CONFIG=gcc-13.4.0/cuda-12080/release
export LD_LIBRARY_PATH="$GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG:${LD_LIBRARY_PATH:-}"
unset PTX_SIM_MODE_FUNC
for fixture in bad_opcode bad_modifier bad_type; do
  set +e
  (cd "$RUN_DIR" && "$HERE/stage2_harness" "$HERE/$fixture.ptx") >"$RUN_DIR/$fixture.log" 2>&1
  rc=$?
  set -e
  test "$rc" -ne 0
  rg -q "$HERE/$fixture.ptx:[0-9]+" "$RUN_DIR/$fixture.log"
done
echo "PARSER_ERRORS status=PASS directory=$RUN_DIR"
