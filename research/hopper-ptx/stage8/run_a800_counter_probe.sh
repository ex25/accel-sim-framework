#!/usr/bin/env bash
set -euo pipefail

# Run on the A800 host. This script is read-only with respect to the driver:
# it never calls sudo or changes module/driver configuration.
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
BASE_RUN=${1:-/tmp/hopper-stage8-counter-probe-$(date -u +%Y%m%dT%H%M%SZ)-$$}
if [[ -e "$BASE_RUN" ]]; then
  RUN="$BASE_RUN/attempt-$(date -u +%Y%m%dT%H%M%SZ)-$$"
else
  RUN="$BASE_RUN"
fi
INPUTS=${2:-$RUN/inputs}
CUDA_BIN=${CUDA_BIN:-/usr/local/cuda/bin}
PYTHON_BIN=${PYTHON_BIN:-/home/wangjz/miniconda3/envs/in-switch/bin/python3}
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=$(command -v python3)
mkdir -p "$RUN" "$RUN/permission-check"
exec > >(tee "$RUN/counter-probe.log") 2>&1

echo "workspace=$ROOT"
echo "run_dir=$RUN"
echo "timestamp=$(date -Is)"
echo "user=$(id -u):$(id -g)"
echo "host=$(hostname)"
echo "python=$PYTHON_BIN"
if [[ -r /proc/self/status ]]; then
  grep -E '^Cap(Inh|Prm|Eff|Bnd|Amb):' /proc/self/status | tee "$RUN/permission-check/capabilities.txt"
fi
"$PYTHON_BIN" --version

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable; run this script on the A800 host." >&2
  exit 2
fi
if [[ ! -x "$CUDA_BIN/ncu" || ! -x "$CUDA_BIN/nvcc" ]]; then
  echo "ERROR: expected ncu and nvcc under $CUDA_BIN." >&2
  exit 2
fi
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv \
  | tee "$RUN/permission-check/device.csv"
"$CUDA_BIN/ncu" --version | tee "$RUN/permission-check/ncu-version.txt"
"$CUDA_BIN/nvcc" --version | tee "$RUN/permission-check/nvcc-version.txt"
if [[ -r /proc/driver/nvidia/params ]]; then
  grep -E 'RestrictProfiling|RmProfiling' /proc/driver/nvidia/params \
    | tee "$RUN/permission-check/nvidia-params.txt" || true
else
  echo "WARNING: /proc/driver/nvidia/params is not readable" \
    | tee "$RUN/permission-check/nvidia-params.txt"
fi

"$PYTHON_BIN" - <<'PY' | tee "$RUN/permission-check/python-environment.txt"
import sys, numpy
print(f"executable={sys.executable}")
print(f"version={sys.version.replace(chr(10), ' ')}")
print(f"numpy={numpy.__version__}")
PY
make -C "$HERE" attention
"$PYTHON_BIN" "$HERE/reference.py" prepare \
  "$ROOT/research/hopper-ptx/stage1/artifacts/inputs" "$INPUTS" > "$RUN/reference.log"

set +e
"$CUDA_BIN/ncu" --target-processes all --launch-count 1 \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread --csv --log-file "$RUN/ncu128.csv" \
  "$HERE/attention" "$INPUTS/128" 128 1 "$RUN/a800-profile-128.f16le" \
  > "$RUN/ncu128.stdout" 2> "$RUN/ncu128.stderr"
status=$?
set -e
echo "ncu_exit=$status"
"$PYTHON_BIN" "$HERE/check_ncu_csv.py" "$RUN/ncu128.csv" \
  "$RUN/profiler-readiness.json" || true
"$PYTHON_BIN" - "$RUN/profiler-readiness.json" "$status" <<'PY'
import json, sys
path, code = sys.argv[1], int(sys.argv[2])
with open(path) as stream:
    data = json.load(stream)
data["exit"] = code
if code:
    data["status"] = "BLOCKED"
with open(path, "w") as stream:
    stream.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
if [[ $status -ne 0 ]]; then
  "$PYTHON_BIN" - "$RUN/profiler-readiness.json" "$status" <<'PY'
import json, os, sys
path, code = sys.argv[1], int(sys.argv[2])
data = json.loads(open(path).read()) if os.path.exists(path) else {}
data["exit"] = code
data["status"] = "BLOCKED"
data["permission_error"] = any("nvgpuctrperm" in e.lower() for e in data.get("errors", []))
data["next_action"] = "Ask an administrator to grant profiling capability or run on an authorized A800 host; do not substitute estimates."
open(path, "w").write(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
  echo "PROFILER_RESULT status=BLOCKED; no driver or module settings were changed." >&2
  exit 3
fi
if ! "$PYTHON_BIN" - "$RUN/profiler-readiness.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    sys.exit(0 if json.load(stream).get("status") == "PASS" else 1)
PY
then
  echo "PROFILER_RESULT status=BLOCKED; required counter values are unavailable." >&2
  exit 3
fi
"$PYTHON_BIN" "$HERE/reference.py" compare "$INPUTS/128" \
  "$RUN/a800-profile-128.f16le" > "$RUN/profile-numerical.log"
for seq in 64 128 256; do
  "$HERE/attention" "$INPUTS/$seq" "$seq" 30 "$RUN/a800-$seq.f16le" \
    > "$RUN/a800-$seq.log"
  "$PYTHON_BIN" "$HERE/reference.py" compare "$INPUTS/$seq" "$RUN/a800-$seq.f16le" \
    > "$RUN/a800-$seq-comparison.log"
done
echo "PROFILER_RESULT status=PASS csv=$RUN/ncu128.csv"
