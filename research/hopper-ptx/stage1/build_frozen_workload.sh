#!/usr/bin/env bash
set -euo pipefail

# Rebuild the exact FA3 source revision used by this stage.
ROOT=$(cd "$(dirname "$0")" && pwd)
SRC=${FA3_SRC:-/tmp/flash-attention-stage1}
OUT=${FA3_OUT:-"$ROOT/artifacts/frozen_fwd.ptx"}
FA3_COMMIT=94e22c906678e5483fa0e9e24d8e787bc2c0ed4c
CUTLASS_COMMIT=7127592069c2fe01b041e174ba4345ef9b279671

if [[ ! -d "$SRC/.git" ]]; then
  git clone https://github.com/Dao-AILab/flash-attention.git "$SRC"
fi
git -C "$SRC" fetch --depth 1 origin "$FA3_COMMIT"
git -C "$SRC" checkout --detach "$FA3_COMMIT"
git -C "$SRC" -c submodule.fetchJobs=1 submodule update --init csrc/cutlass
test "$(git -C "$SRC/csrc/cutlass" rev-parse HEAD)" = "$CUTLASS_COMMIT"

mkdir -p "$(dirname "$OUT")"
/usr/local/cuda/bin/nvcc -ccbin /usr/bin/g++ -std=c++17 -O3 --use_fast_math \
  --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG \
  -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DCUTLASS_ENABLE_GDC_FOR_SM90 \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 -arch=sm_90a -ptx \
  -I"$SRC/hopper" -I"$SRC/csrc/cutlass/include" "$ROOT/frozen_fwd.cu" -o "$OUT"
ptxas -v -arch=sm_90a -o /tmp/fa3-stage1.cubin "$OUT"
