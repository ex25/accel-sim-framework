#!/usr/bin/env python3
"""Run directed arithmetic through PTX execution, with before/after cycle checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
SIM = ROOT / "gpu-simulator/gpgpu-sim"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("capture", type=Path)
p.add_argument("--matrix-ptx", type=Path, required=True)
p.add_argument("--baseline-library", type=Path, required=True)
p.add_argument("--run-dir", type=Path, required=True)
a = p.parse_args()
run, cap = a.run_dir.resolve(), a.capture.resolve()
run.mkdir(parents=True, exist_ok=False)
cuda = Path(os.environ.get("CUDA_INSTALL_PATH", "/usr/local/cuda"))
commands = []
def call(cmd):
    commands.append(list(map(str, cmd)))
    subprocess.run(cmd, check=True)
asm = []
for name, source in [("activation", cap / "package/probe.ptx"), ("ops", cap / "package/ops.ptx"), ("matrix", a.matrix_ptx.resolve())]:
    shutil.copy2(source, run / (name + ".ptx"))
    call([cuda / "bin/fatbinary", "--64", "--create=" + str(run / (name + ".fatbin")),
          "--image=profile=compute_90a,file=" + str(run / (name + ".ptx"))])
    asm.append(f'.section .nv_fatbin,"a"\n.balign 8\n.global {name}_module\n{name}_module:\n.incbin "{run / (name + ".fatbin")}"\n')
(run / "embed.S").write_text("".join(asm) + '.section .note.GNU-stack,"",@progbits\n')
call(["gcc", "-c", run / "embed.S", "-o", run / "embed.o"])
call(["g++", "-std=c++17", "-O2", "-I" + str(cuda / "include"), HERE / "small.cc", run / "embed.o",
      "-L" + str(cuda / "lib64"), "-lcudart", "-o", run / "small"])
# Original H100 observations, selecting an actual BF16 boundary and other rows.
pairs = np.fromfile(cap / "package/pairs.f32", np.float32).reshape(-1, 2)
indices = [int(np.flatnonzero((pairs[:, 0] == .78515625) & (pairs[:, 1] == .87109375))[0])]
indices += np.linspace(0, len(pairs) - 1, 31, dtype=int).tolist()
fixtures = {"activation": (pairs[indices], np.fromfile(cap / "h100/probe.cubin.u32", np.uint32).reshape(-1, 8)[indices])}
ops = np.fromfile(cap / "package/ops.f32", np.float32).reshape(-1, 2)
observed = np.fromfile(cap / "h100/ops.cubin.u32", np.uint32).reshape(-1, 5)
idx = np.flatnonzero(observed[:, 2] != observed[:, 3])[:16].tolist() + np.linspace(0, len(ops)-1, 16, dtype=int).tolist()
fixtures["ops"] = ops[idx], observed[idx]
# Each output is +/- (1 + 2^-24 + 2^-24), exactly 1 FP32 ulp from +/-1.
# A sequential RN FMA loses each half-ulp. The H100 block accumulator retains
# both. Expected values are specified arithmetically, not with model code.
x = np.zeros(40960 // 2, np.uint16)
def put(base, row, k, value):
    linear = base + row * 128 + 2 * k
    address = linear ^ (((linear // 128) % 8) * 16)
    x[address // 2] = np.array([value], np.float32).view(np.uint32)[0] >> 16
for m in range(64):
    for k, value in enumerate((1., 2.**-24, 2.**-24)):
        put(0, m, k, value * (-1 if m % 2 else 1))
for n in range(128):
    for k in range(3):
        put(16384, n, k, -1 if n % 2 else 1)
for mode, factor in [("single128", 1), ("chain", 2)]:
    expected = np.empty((128, 64), np.float32)
    for t in range(128):
        for j in range(64):
            m = 16 * (t // 32) + (t % 32) // 4 + 8 * ((j % 4) // 2)
            n = 8 * (j // 4) + 2 * (t % 4) + j % 2
            expected[t, j] = factor * (1 + 2.**-23) * (-1 if (m+n) % 2 else 1)
    fixtures[mode] = x, expected
baseline = run / "baseline/gpgpu-sim"
baseline.mkdir(parents=True)
shutil.copy2(a.baseline_library, baseline / "libcudart.so.12")
base_env = os.environ.copy()
for key in ("PTX_SIM_USE_PTX_FILE", "PTX_SIM_KERNELFILE", "CUOBJDUMP_SIM_FILE"):
    base_env.pop(key, None)
base_env["PTX_SIM_MODE_FUNC"] = "0"
config_dir = SIM / "configs/tested-cfgs/SM90_H100"
report = dict(commands=commands, variants={})
for variant in ("before", "after"):
    directory = run / variant
    directory.mkdir()
    for mode, (inputs, expected) in fixtures.items():
        inputs.tofile(directory / (mode + ".input"))
        expected.tofile(directory / (mode + ".expected"))
    shutil.copy2(config_dir / "gpgpusim.config", directory)
    shutil.copy2(config_dir / "config_hopper_islip.icnt", directory)
    env = base_env.copy()
    if variant == "before":
        env["LD_LIBRARY_PATH"] = str(baseline) + ":" + env.get("LD_LIBRARY_PATH", "")
    libs = subprocess.check_output(["ldd", run / "small"], env=env, text=True)
    path = Path(next(line.split("=>")[1].split("(")[0].strip() for line in libs.splitlines() if "libcudart" in line))
    records = {}
    for mode in fixtures:
        start = time.monotonic()
        with (directory / (mode + ".log")).open("w") as log:
            result = subprocess.run([run / "small", mode], cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
        text = (directory / (mode + ".log")).read_text()
        records[mode] = dict(returncode=result.returncode, passed="PASS " + mode in text,
                             cycles=list(map(int, re.findall(r"^gpu_sim_cycle = (\d+)", text, re.M))),
                             instructions=list(map(int, re.findall(r"^gpu_sim_insn = (\d+)", text, re.M))),
                             wall_seconds=time.monotonic()-start)
        print(variant, mode, records[mode], flush=True)
    report["variants"][variant] = dict(library=str(path), library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), tests=records)
    (run / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
assert all(r["passed"] for r in report["variants"]["after"]["tests"].values())
assert all(report["variants"]["before"]["tests"][name]["cycles"] == row["cycles"] for name, row in report["variants"]["after"]["tests"].items())
