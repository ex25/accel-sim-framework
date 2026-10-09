#!/usr/bin/env python3
"""Exercise explicit parameter alignments and strict negative contracts."""
from pathlib import Path
import os
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/hopper-stage8-param-layout")
OUT.mkdir(parents=True, exist_ok=True)
config_dir = ROOT / "gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100"
for name in ("gpgpusim.config", "config_hopper_islip.icnt"):
    source = config_dir / name
    if source.exists():
        (OUT / name).write_bytes(source.read_bytes())

HEADER = ".version 8.7\n.target sm_90\n.address_size 64\n"
BODY = ".reg .b32 %r<2>;\nret;\n"
cases = {
    "align32_64": HEADER + ".visible .entry layout(.param .align 32 .b8 a[32], .param .align 64 .b8 b[64]) {\n" + BODY + "}\n",
    "scalar_vector": HEADER + ".visible .entry layout(.param .u8 a, .param .u16 b, .param .v4 .b32 c, .param .u64 d) {\n" + BODY + "}\n",
    "array_natural": HEADER + ".visible .entry layout(.param .b8 a[3], .param .u64 b) {\n" + BODY + "}\n",
    "frozen_abi": HEADER + ".visible .entry layout(.param .align 64 .b8 a[2368]) {\n" + BODY + "}\n",
    "align_non_power_two": HEADER + ".visible .entry bad(.param .align 48 .b8 a[32]) {\n" + BODY + "}\n",
    "align_zero": HEADER + ".visible .entry bad(.param .align 0 .b8 a[32]) {\n" + BODY + "}\n",
    "align_too_large": HEADER + ".visible .entry bad(.param .align 2097152 .b8 a[32]) {\n" + BODY + "}\n",
    "align_integer_overflow": HEADER + ".visible .entry bad(.param .align 4294967296 .b8 a[32]) {\n" + BODY + "}\n",
    "offset_overflow": HEADER + ".visible .entry bad(.param .b64 a[536870911], .param .align 64 .b8 b[64]) {\n" + BODY + "}\n",
}
for name, text in cases.items():
    (OUT / f"{name}.ptx").write_text(text)

env = {**os.environ, "LD_LIBRARY_PATH": "/tmp/hopper-stage3-build"}
for name, expected in [("align32_64", [0, 64]), ("scalar_vector", [0, 2, 16, 32]),
                       ("array_natural", [0, 8]), ("frozen_abi", [0])]:
    positive = subprocess.run([str(HERE / "param_layout_probe"), str(OUT / f"{name}.ptx"), "layout"],
                              cwd=OUT, env=env, text=True, capture_output=True, timeout=45)
    (OUT / f"{name}.log").write_text(positive.stdout + positive.stderr)
    lines = [line for line in positive.stdout.splitlines() if line.startswith("PARAM index=")]
    offsets = [int(re.search(r"offset=(\d+)", line).group(1)) for line in lines]
    if positive.returncode or offsets != expected or "PARAM_COPY status=PASS" not in positive.stdout:
        raise ValueError(f"positive {name}: exit={positive.returncode} offsets={offsets}")
for name in ("align_non_power_two", "align_zero", "align_too_large", "align_integer_overflow", "offset_overflow"):
    result = subprocess.run([str(ROOT / "research/hopper-ptx/stage3/frontend_probe"),
                             str(OUT / f"{name}.ptx"), "bad"],
                            cwd=OUT, env=env, text=True, capture_output=True, timeout=45)
    (OUT / f"{name}.log").write_text(result.stdout + result.stderr)
    if result.returncode == 0 or not any(p in result.stdout + result.stderr for p in
                                        (".align must be", "Syntax error", "parameter offset/size overflow")):
        print(f"PARAM_LAYOUT_RESULT status=FAIL negative={name}", file=sys.stderr)
        sys.exit(1)
(OUT / "result.json").write_text(json.dumps({"status": "PASS", "positive_cases": 4,
                                           "negative_cases": 5, "copy_guard": "PASS"}, indent=2) + "\n")
print("PARAM_LAYOUT_RESULT status=PASS positive=4 negatives=5 copy_guard=PASS")
