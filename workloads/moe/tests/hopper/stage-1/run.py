#!/usr/bin/env python3
"""Stage 1 self-tests, using stage 0's fatbinary registration and H100 setup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from generate_micro import generate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SIM = ROOT / "gpu-simulator/gpgpu-sim"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run-dir", type=Path, required=True)
p.add_argument("--test", default="all")
p.add_argument("--shuffle-latency", type=int, default=4)
p.add_argument("--shuffle-initiation", type=int, default=2)
a = p.parse_args()
run = a.run_dir.resolve()
run.mkdir(parents=True, exist_ok=True)
commands = []


def call(cmd, **kw):
    commands.append([str(x) for x in cmd])
    subprocess.run(cmd, check=True, **kw)


cuda = Path(os.environ.get("CUDA_INSTALL_PATH", "/usr/local/cuda"))
entries = generate(run)
asm = []
for name, target, symbol in [
    ("micro", "sm_90a", "module_a"),
    ("legacy", "sm_60", "module_legacy"),
]:
    call(
        [
            cuda / "bin/ptxas",
            "-arch=" + target,
            run / (name + ".ptx"),
            "-o",
            run / (name + ".cubin"),
        ]
    )
    call(
        [
            cuda / "bin/fatbinary",
            "--64",
            "--create=" + str(run / (name + ".fatbin")),
            "--image=profile="
            + target.replace("sm_", "compute_")
            + ",file="
            + str(run / (name + ".ptx")),
        ]
    )
    asm.append(
        '.section .nv_fatbin,"a"\n.balign 8\n.global '
        + symbol
        + "\n"
        + symbol
        + ':\n.incbin "'
        + str(run / (name + ".fatbin"))
        + '"\n'
    )
(run / "embed.S").write_text("".join(asm) + '.section .note.GNU-stack,"",@progbits\n')
call(["gcc", "-c", run / "embed.S", "-o", run / "embed.o"])
call(
    [
        "g++",
        "-std=c++17",
        "-g",
        "-O0",
        "-I" + str(cuda / "include"),
        HERE / "micro.cc",
        run / "embed.o",
        "-L" + str(cuda / "lib64"),
        "-lcudart",
        "-o",
        run / "micro",
    ]
)
libraries = subprocess.check_output(["ldd", run / "micro"], text=True)
if not any(
    "libcudart" in line and "gpgpu-sim" in line for line in libraries.splitlines()
):
    raise RuntimeError(libraries)
config = (SIM / "configs/tested-cfgs/SM90_H100/gpgpusim.config").read_text()
config = config.replace("-gpgpu_n_clusters 132", "-gpgpu_n_clusters 1").replace(
    "-gpgpu_n_chiplet_partition 2", "-gpgpu_n_chiplet_partition 1"
)
config = config.replace(
    "-ptx_opcode_latency_int 4,4,4,4,21,4",
    f"-ptx_opcode_latency_int 4,4,4,4,21,{a.shuffle_latency}",
)
config = config.replace(
    "-ptx_opcode_initiation_int 2,2,2,2,2,2",
    f"-ptx_opcode_initiation_int 2,2,2,2,2,{a.shuffle_initiation}",
)
(run / "gpgpusim.config").write_text(config)
shutil.copy2(SIM / "configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt", run)
env = os.environ.copy()
for key in ["PTX_SIM_USE_PTX_FILE", "PTX_SIM_KERNELFILE", "CUOBJDUMP_SIM_FILE"]:
    env.pop(key, None)
env["PTX_SIM_MODE_FUNC"] = "0"
source_diff = subprocess.check_output(["git", "diff", "--binary"], cwd=SIM)
(run / "inner.diff").write_bytes(source_diff)
library_path = Path(
    next(
        line.split("=>", 1)[1].split("(", 1)[0].strip()
        for line in libraries.splitlines()
        if "libcudart" in line
    )
)
library_sha256 = hashlib.sha256(library_path.read_bytes()).hexdigest()
records = []
for test in entries if a.test == "all" else a.test.split(","):
    cmd = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(run / (test + ".time")),
        str(run / "micro"),
        test,
    ]
    commands.append(cmd)
    with (run / (test + ".log")).open("w") as log:
        result = subprocess.run(
            cmd, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180
        )
    passed = result.returncode == 0 and "PASS" in (run / (test + ".log")).read_text()
    records.append(dict(test=test, returncode=result.returncode, passed=passed))
    print(records[-1], flush=True)
metadata = dict(
    library_sha256=library_sha256,
    source_diff_sha256=hashlib.sha256(source_diff).hexdigest(),
    binary_sha256=hashlib.sha256((run / "micro").read_bytes()).hexdigest(),
    commands=commands,
    libraries=libraries,
    tests=records,
    config_sha256=hashlib.sha256((run / "gpgpusim.config").read_bytes()).hexdigest(),
    outer=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip(),
    inner=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SIM, text=True
    ).strip(),
    cuda=subprocess.check_output([cuda / "bin/nvcc", "--version"], text=True),
)
(run / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
raise SystemExit(0 if all(x["passed"] for x in records) else 1)
