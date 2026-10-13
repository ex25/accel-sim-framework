#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_DIR=${1:-/tmp/hopper-stage6-run}
mkdir -p "$RUN_DIR/functional" "$RUN_DIR/detailed" "$RUN_DIR/ptx"
"${PYTHON_BIN:-python3}" "$HERE/generate.py" "$RUN_DIR/ptx"
for mode in functional detailed; do
  cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config" "$RUN_DIR/$mode/"
  cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt" "$RUN_DIR/$mode/"
  value=1; if [[ "$mode" == detailed ]]; then value=0; fi
  sed -i "s/^-gpgpu_ptx_sim_mode .*/-gpgpu_ptx_sim_mode $value/" "$RUN_DIR/$mode/gpgpusim.config"
  # Reuse the Hopper trace GMMA specialized-unit configuration for PTX.
  printf '\n' >> "$RUN_DIR/$mode/gpgpusim.config"
  rg '^-specialized_unit_[1-5] ' "$ROOT/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config" >> "$RUN_DIR/$mode/gpgpusim.config"
done
make -C "$HERE" all
export LD_LIBRARY_PATH="$(dirname "${SIM_RUNTIME:-/tmp/hopper-stage3-build/libcudart.so}")"
for case in ss128 ss128_hex ss176 rs128 rs176 rs176_zero empty; do
  for mode in functional detailed; do
    (cd "$RUN_DIR/$mode"; timeout 45 "$HERE/wgmma_harness" "$RUN_DIR/ptx/$case.ptx" "$mode" "$case" > "$case.log" 2>&1)
    grep -E 'FRONTEND_RESULT|MATRIX_RESULT|TIMING_RESULT' "$RUN_DIR/$mode/$case.log"
  done
done
for case in timing128 timing176; do
  (cd "$RUN_DIR/detailed"; timeout 45 "$HERE/wgmma_harness" "$RUN_DIR/ptx/$case.ptx" detailed "$case" > "$case.log" 2>&1)
  grep -E 'MATRIX_RESULT|TIMING_RESULT' "$RUN_DIR/detailed/$case.log"
done
"${PYTHON_BIN:-python3}" "$HERE/verify_timing.py" "$RUN_DIR/detailed/timing128.log" "$RUN_DIR/detailed/timing176.log"
negative() {
  local mode=$1 case=$2 pattern=$3 threads=${4:-128}
  local log="$RUN_DIR/$mode/negative-$case-$threads.log"
  local result=0
  (cd "$RUN_DIR/$mode"; timeout 45 "$HERE/wgmma_harness" "$RUN_DIR/ptx/$case.ptx" "$mode" ss128 "$threads") > "$log" 2>&1 || result=$?
  if [[ $result == 0 || $result == 124 || $result == 137 ]]; then
    echo "NEGATIVE_FAILURE case=$case mode=$mode exit=$result"; exit 1
  fi
  grep -Eq "$pattern" "$log"
  echo "NEGATIVE_RESULT status=PASS case=$case mode=$mode threads=$threads"
}
for case in bad_tuple duplicate_tuple bad_type bad_descriptor_width bad_modifier bad_scale float_scale bad_scale64 bad_wait bad_wait64 predicated literal_overflow hex_overflow; do
  negative functional "$case" 'matrix register tuple length mismatch|unsupported matrix opcode/modifier|immediate out of range|immediate must be 0..7|predicated wgmma|duplicate register|specified bit width/type|matrix register width mismatch|integer literal exceeds 64 bits|requires an immediate'
done
for mode in functional detailed; do
  negative "$mode" ss128 'incomplete WGMMA warpgroup' 64
  negative "$mode" no_fence 'MMA requires wgmma.fence'
  negative "$mode" after_fence 'accumulator modified after fence'
  negative "$mode" source_overwrite 'source overwrite before wait_group'
  negative "$mode" wait1_access 'accumulator access before wait_group'
  negative "$mode" no_wait 'accumulator access before wait_group'
  negative "$mode" swizzle 'unsupported WGMMA descriptor swizzle'
  negative "$mode" oob 'overlaps or exceeds declared shared memory'
  negative "$mode" nonuniform 'nonuniform WGMMA descriptor/scale-d'
  negative "$mode" divergent 'incomplete WGMMA warpgroup'
  negative "$mode" partial 'incomplete WGMMA active warp|incomplete WGMMA warpgroup'
done

mkdir -p "$RUN_DIR/no_gmma"
cp "$RUN_DIR/detailed/config_hopper_islip.icnt" "$RUN_DIR/no_gmma/"
sed '/^-specialized_unit_/d' "$RUN_DIR/detailed/gpgpusim.config" > "$RUN_DIR/no_gmma/gpgpusim.config"
negative no_gmma ss128 'requires enabled specialized_unit_1..5'
