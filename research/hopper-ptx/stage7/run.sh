#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
RUN_DIR=${1:-/tmp/hopper-stage7-acceptance}
mkdir -p "$RUN_DIR/functional" "$RUN_DIR/detailed" "$RUN_DIR/ptx"
python3 "$HERE/generate.py" "$RUN_DIR/ptx"
make -C "$HERE" all
for mode in functional detailed; do
  cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config" "$RUN_DIR/$mode/"
  cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt" "$RUN_DIR/$mode/"
  value=1; if [[ "$mode" == detailed ]]; then value=0; fi
  sed -i "s/^-gpgpu_ptx_sim_mode .*/-gpgpu_ptx_sim_mode $value/" "$RUN_DIR/$mode/gpgpusim.config"
done
export LD_LIBRARY_PATH=/tmp/hopper-stage3-build
run() {
  local name=$1 mode=$2 ptx=$3; shift 3
  (cd "$RUN_DIR/$mode"; timeout 45 "$HERE/cluster_harness" "$RUN_DIR/ptx/$ptx.ptx" "$mode" "$@" > "$name.log" 2>&1)
  rg '^CLUSTER_RESULT' "$RUN_DIR/$mode/$name.log"
}
for mode in functional detailed; do
  run implicit_meta "$mode" metadata 1 1 1 1 1 1 0
  run explicit1_meta "$mode" metadata 1 1 1 1 1 1 1
  run metadata2 "$mode" metadata 2 1 1 2 1 1 1
  run metadata3d "$mode" metadata 4 2 2 2 2 2 1
done
run implicit functional cluster 1 1 1 1 1 1 0
run explicit1 functional cluster 1 1 1 1 1 1 1
run pair functional cluster 2 1 1 2 1 1 1
run four functional cluster 4 1 1 4 1 1 1
run twod functional cluster 2 2 1 2 2 1 1
run threed functional cluster 2 2 2 2 2 2 1
run multicluster functional cluster 4 2 1 2 2 1 1
# A failure counts only when the expected diagnostic is present; timeouts don't.
negative() {
  local name=$1 mode=$2 expected=$3; shift 3
  local code=0
  (cd "$RUN_DIR/$mode"; timeout 45 "$HERE/cluster_harness" "$RUN_DIR/ptx/$name.ptx" "$mode" "$@") > "$RUN_DIR/$mode/$name.log" 2>&1 || code=$?
  if [[ $code == 0 || $code == 124 || $code == 137 ]]; then
    echo "unexpected negative result $name code=$code"; exit 1
  fi
  rg "$expected" "$RUN_DIR/$mode/$name.log" | tail -1
  echo "CLUSTER_NEGATIVE status=PASS mode=$mode case=$name exit=$code"
}
for item in 'bad_modifier:unsupported cluster opcode' 'bad_width:registers of the declared width' 'bad_operands:operand count mismatch' 'bad_barrier:unsupported cluster opcode' 'predicated_barrier:aligned cluster barrier' 'setmax_modifier:unsupported cluster opcode' 'setmax_count:multiple of 8' 'bad_sreg:dimension modifier mismatch'; do
  negative "${item%%:*}" functional "${item#*:}" 2 1 1 2 1 1 1
done
for item in 'bad_rank:mapa rank/address' 'bad_address:mapa rank/address' 'bad_access:misaligned' 'cta_access:mapped address requires' 'wait_without_arrive:wait without arrive' 'duplicate_arrive:arrive before previous wait' 'missing_arrival:cluster deadlock' 'divergent_barrier:full active warp' 'setmax_inc:register-pool' 'setmax_dec:register-pool'; do
  negative "${item%%:*}" functional "${item#*:}" 2 1 1 2 1 1 1
done
for item in 'setmax_inc:register-pool' 'setmax_dec:register-pool'; do
  negative "${item%%:*}" detailed "${item#*:}" 2 1 1 2 1 1 1
done
cp "$RUN_DIR/ptx/cluster.ptx" "$RUN_DIR/ptx/remote_timing.ptx"
sed '/barrier.cluster/d' "$RUN_DIR/ptx/cluster.ptx" > "$RUN_DIR/ptx/remote_timing.ptx"
negative remote_timing detailed 'DSMEM remote timing' 2 1 1 2 1 1 1
cp "$RUN_DIR/ptx/cluster.ptx" "$RUN_DIR/ptx/barrier_timing.ptx"
negative barrier_timing detailed 'cluster barrier timing' 2 1 1 2 1 1 1
cp "$RUN_DIR/ptx/metadata.ptx" "$RUN_DIR/ptx/bad_launch.ptx"
negative bad_launch functional 'invalid PTX cluster dimensions' 2 1 1 9 1 1 1
cp "$RUN_DIR/ptx/metadata.ptx" "$RUN_DIR/ptx/bad_grid.ptx"
negative bad_grid functional 'must be divisible' 3 1 1 2 1 1 1
sed -n '1,36p' "$RUN_DIR/ptx/metadata.ptx" > "$RUN_DIR/ptx/fence_timing.ptx"
printf 'fence.proxy.async.shared::cluster;\nret;\n}\n' >> "$RUN_DIR/ptx/fence_timing.ptx"
negative fence_timing detailed 'cluster proxy fence timing' 2 1 1 2 1 1 1
