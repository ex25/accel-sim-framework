#!/usr/bin/env bash
# Run outside the sandbox where the NVIDIA driver is accessible.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
RUN=${1:-/tmp/hopper-stage8}
PYTHON_BIN=${PYTHON_BIN:-/home/wangjz/miniconda3/envs/in-switch/bin/python3}
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 2; }
mkdir -p "$RUN/a800-traces"
make -C "$HERE" all
"$PYTHON_BIN" "$HERE/reference.py" prepare "$ROOT/research/hopper-ptx/stage1/artifacts/inputs" "$RUN/inputs" > "$RUN/reference.log"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv > "$RUN/a800-device.csv"
for seq in 64 128 256; do
  "$HERE/attention" "$RUN/inputs/$seq" "$seq" 30 "$RUN/a800-$seq.f16le" > "$RUN/a800-$seq.log"
  "$PYTHON_BIN" "$HERE/reference.py" compare "$RUN/inputs/$seq" "$RUN/a800-$seq.f16le" > "$RUN/a800-$seq-comparison.log"
done
# Counter permission failures are recorded explicitly, never replaced by estimates.
code=0
/usr/local/cuda/bin/ncu --target-processes all --launch-count 1 --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread --csv --log-file "$RUN/ncu128.csv" "$HERE/attention" "$RUN/inputs/128" 128 1 "$RUN/a800-profile-128.f16le" > "$RUN/ncu128.log" 2>&1 || code=$?
"$PYTHON_BIN" "$HERE/check_ncu_csv.py" "$RUN/ncu128.csv" "$RUN/profiler-readiness.json" || true
"$PYTHON_BIN" - "$RUN/profiler-readiness.json" "$code" <<'PYCODE'
import json, sys
from pathlib import Path
path=Path(sys.argv[1]); code=int(sys.argv[2]); data=json.loads(path.read_text())
data['exit']=code
if code:
    data['status']='BLOCKED'
    data['errors'].append('ncu_exit='+str(code))
path.write_text(json.dumps(data,indent=2)+'\n')
PYCODE
# Keep LD_PRELOAD inside env, so NVBit is loaded into the CUDA program only.
timeout 120 env USER_DEFINED_FOLDERS=1 TRACES_FOLDER="$RUN/a800-traces" DYNAMIC_KERNEL_RANGE=1 ALLOW_REG_VAL_TRACING=0 TRACE_FILE_COMPRESS=0 LD_PRELOAD="$ROOT/util/tracer_nvbit/tracer_tool/tracer_tool.so" "$HERE/attention" "$RUN/inputs/128" 128 1 "$RUN/a800-traced-128.f16le" 1 1 > "$RUN/a800-trace.log" 2>&1
"$ROOT/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing" "$RUN/a800-traces/traces" --text > "$RUN/trace-postprocess.log"
traces=("$RUN"/a800-traces/traces/*.traceg)
if [[ ${#traces[@]} != 1 || ! -f ${traces[0]} ]]; then echo 'expected exactly one captured trace'; exit 1; fi
"$HERE/trace_audit" "${traces[0]}" > "$RUN/a800-trace-audit.log"
"$PYTHON_BIN" "$HERE/reference.py" compare "$RUN/inputs/128" "$RUN/a800-traced-128.f16le" --rows 1 --heads 1 > "$RUN/a800-traced-128-comparison.log"
mkdir -p "$RUN/a800-sass-model"
cp "$ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config" "$RUN/a800-sass-model/"
cp "$ROOT/gpu-simulator/configs/tested-cfgs/SM80_A100/trace.config" "$RUN/a800-sass-model/"
(cd "$RUN/a800-sass-model"; LD_LIBRARY_PATH=/tmp/hopper-stage3-build timeout 120 "$HERE/trace_sim" -config gpgpusim.config -config trace.config -trace "$RUN/a800-traces/traces/kernelslist.g" > "$RUN/a800-sass-model.log" 2>&1)
rg '^A800_RESULT' "$RUN"/a800-{64,128,256}.log
head -1 "$RUN/a800-trace-audit.log"
