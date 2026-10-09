#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
RUN=${1:-/tmp/hopper-stage8}
PYTHON_BIN=${PYTHON_BIN:-/home/wangjz/miniconda3/envs/in-switch/bin/python3}
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 2; }
mkdir -p "$RUN/functional" "$RUN/detailed" "$RUN/wgmma"
make -C "$HERE" all
make -C "$ROOT/research/hopper-ptx/stage3" all
make -C "$ROOT/research/hopper-ptx/stage6" all
"$PYTHON_BIN" "$HERE/param_layout_probe.py" "$RUN/parameter-layout" > "$RUN/parameter-layout.log"
"$PYTHON_BIN" "$HERE/test_ncu_csv.py" "$RUN/counter-contracts" > "$RUN/counter-contracts.log"
"$PYTHON_BIN" "$HERE/reference.py" prepare "$ROOT/research/hopper-ptx/stage1/artifacts/inputs" "$RUN/inputs" > "$RUN/reference.log"
"$PYTHON_BIN" "$ROOT/research/hopper-ptx/stage6/generate.py" "$RUN/wgmma"
"$PYTHON_BIN" - "$ROOT" "$RUN" <<'PY'
from pathlib import Path
import sys
root, run = map(Path, sys.argv[1:])
source = root / 'gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100'
special = '\n'.join(l for l in (root/'gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config').read_text().splitlines() if l.startswith('-specialized_unit_'))
for mode, value in [('functional',1),('detailed',0)]:
    text = source.joinpath('gpgpusim.config').read_text()
    text = '\n'.join(f'-gpgpu_ptx_sim_mode {value}' if l.startswith('-gpgpu_ptx_sim_mode ') else l for l in text.splitlines())
    (run/mode/'gpgpusim.config').write_text(text+'\n'+special+'\n')
    (run/mode/'config_hopper_islip.icnt').write_bytes((source/'config_hopper_islip.icnt').read_bytes())
PY
"$PYTHON_BIN" "$HERE/frontend_contracts.py" "$ROOT" "$RUN/frontend-contracts" "$RUN/detailed" > "$RUN/frontend-contracts.log" 2>&1
/usr/local/cuda/bin/ptxas -arch=sm_90a -v "$HERE/attention.ptx" -o "$RUN/attention-sm90a.cubin" > "$RUN/ptxas-sm90a.log" 2>&1
/usr/local/cuda/bin/ptxas -arch=sm_90a -v "$ROOT/research/hopper-ptx/stage1/artifacts/frozen_fwd.ptx" -o "$RUN/frozen-sm90a.cubin" > "$RUN/ptxas-frozen.log" 2>&1
"$PYTHON_BIN" "$HERE/sass_inventory.py" "$RUN/frozen-sm90a.cubin" "$RUN/frozen-sass.json"
export LD_LIBRARY_PATH=/tmp/hopper-stage3-build
(cd "$RUN/functional"; timeout 900 "$HERE/attention_harness" "$HERE/attention.ptx" functional "$RUN/inputs/128" 128 128 8 10000000 "$RUN/ptx-functional-128.f16le" > attention128.log 2>&1)
"$PYTHON_BIN" "$HERE/reference.py" compare "$RUN/inputs/128" "$RUN/ptx-functional-128.f16le" > "$RUN/ptx-functional-128-comparison.log"
for seq in 64 128 256; do
  (cd "$RUN/detailed"; timeout 300 "$HERE/attention_harness" "$HERE/attention.ptx" detailed "$RUN/inputs/$seq" "$seq" 1 1 10000000 "$RUN/ptx-detailed-$seq.f16le" > "attention$seq.log" 2>&1)
  "$PYTHON_BIN" "$HERE/reference.py" compare "$RUN/inputs/$seq" "$RUN/ptx-detailed-$seq.f16le" --rows 1 --heads 1 > "$RUN/ptx-detailed-$seq-comparison.log"
done
for n in 128 176; do
  (cd "$RUN/detailed"; "$HERE/sass_oracle" "$RUN/wgmma/timing$n.ptx" > "$RUN/sass-oracle$n.log" 2>&1)
  (cd "$RUN/detailed"; timeout 45 "$ROOT/research/hopper-ptx/stage6/wgmma_harness" "$RUN/wgmma/timing$n.ptx" detailed "timing$n" > "timing$n.log" 2>&1)
done
"$PYTHON_BIN" "$ROOT/research/hopper-ptx/stage6/verify_timing.py" "$RUN/detailed/timing128.log" "$RUN/detailed/timing176.log" > "$RUN/wgmma-timing-check.log"
bash "$ROOT/research/hopper-ptx/stage5/run_detailed.sh" "$RUN/tma" > "$RUN/tma.log" 2>&1
# The immutable frozen module must pass the actual frontend; no expected-failure bypass.
"$PYTHON_BIN" "$HERE/probe_frozen.py" "$ROOT" "$RUN"
rg '^PTX_ATTENTION_RESULT' "$RUN/functional/attention128.log" "$RUN"/detailed/attention{64,128,256}.log
cat "$RUN/wgmma-timing-check.log"
