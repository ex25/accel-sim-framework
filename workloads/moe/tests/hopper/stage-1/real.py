#!/usr/bin/env python3
"""Run selected unmodified Qwen64/128 modules with synthetic CPU references."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from read_forward import launches

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SIM = ROOT / "gpu-simulator/gpgpu-sim"
ART = ROOT / "workloads/moe/artifacts"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run-dir", type=Path, required=True)
p.add_argument("--test", default="all")
p.add_argument("--full-config", action="store_true")
a = p.parse_args()
run = a.run_dir.resolve()
run.mkdir(parents=True, exist_ok=True)
cuda = Path(os.environ.get("CUDA_INSTALL_PATH", "/usr/local/cuda"))
commands = []


def call(cmd, **kw):
    commands.append(list(map(str, cmd)))
    subprocess.run(cmd, check=True, **kw)


modules = json.loads((ART / "workloads.json").read_text())["modules"]
selected = [70, 3, 4, 77, 10, 38, 78, 79, 80, 81, 82]
inc = []
asm = []
meta = []
for index in selected:
    module = modules[index]
    spec = next(x for x in launches if x["module"] == index)
    call(
        [
            cuda / "bin/fatbinary",
            "--64",
            "--create=" + str(run / f"{index}.fatbin"),
            "--image=profile=compute_90a,file=" + str(ART / module["ptx"]),
        ]
    )
    asm.append(
        f'.section .nv_fatbin,"a"\n.balign 8\n.global real_{index}\nreal_{index}:\n.incbin "{run}/{index}.fatbin"\n'
    )
    inc.append(f'extern "C" const unsigned char real_{index}[];\n')
    args = (
        "{"
        + ",".join(
            "{" + ",".join(str(b) for b in bytes.fromhex(arg)) + "}"
            for arg in spec["args"]
        )
        + "}"
    )
    ptr = (
        "{"
        + ",".join(
            "{" + ",".join(map(str, where)) + "}" for where in module["parameter_words"]
        )
        + "}"
    )
    shape = "{" + ",".join(map(str, spec["shape"])) + "}"
    meta.append(
        "{"
        + str(index)
        + ',{"'
        + module["entry"]
        + '",real_'
        + str(index)
        + ","
        + shape
        + ","
        + args
        + ","
        + ptr
        + "}}"
    )
inc.append("static const std::map<unsigned,Launch> specs={" + ",".join(meta) + "};\n")
(run / "real_launches.inc").write_text("".join(inc))
(run / "launches.json").write_text(
    json.dumps([x for x in launches if x["module"] in selected], indent=2) + "\n"
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
        "-I" + str(run),
        HERE / "real.cc",
        run / "embed.o",
        "-L" + str(cuda / "lib64"),
        "-lcudart",
        "-o",
        run / "real",
    ]
)
libraries = subprocess.check_output(["ldd", run / "real"], text=True)
if not any("libcudart" in x and "gpgpu-sim" in x for x in libraries.splitlines()):
    raise RuntimeError(libraries)
config = (SIM / "configs/tested-cfgs/SM90_H100/gpgpusim.config").read_text()
if not a.full_config:
    config = config.replace("-gpgpu_n_clusters 132", "-gpgpu_n_clusters 1").replace(
        "-gpgpu_n_chiplet_partition 2", "-gpgpu_n_chiplet_partition 1"
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
for test in (
    ["convert", "route", "mapping", "swiglu"] if a.test == "all" else a.test.split(",")
):
    cmd = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(run / (test + ".time")),
        str(run / "real"),
        test,
    ]
    commands.append(cmd)
    with (run / (test + ".log")).open("w") as log:
        result = subprocess.run(
            cmd, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600
        )
    passed = (
        result.returncode == 0 and "PASS real" in (run / (test + ".log")).read_text()
    )
    records.append(dict(test=test, returncode=result.returncode, passed=passed))
    print(records[-1], flush=True)
metadata = dict(
    library_sha256=library_sha256,
    source_diff_sha256=hashlib.sha256(source_diff).hexdigest(),
    binary_sha256=hashlib.sha256((run / "real").read_bytes()).hexdigest(),
    commands=commands,
    libraries=libraries,
    tests=records,
    full_config=a.full_config,
    config_sha256=hashlib.sha256((run / "gpgpusim.config").read_bytes()).hexdigest(),
    ptx_sha256={
        str(i): hashlib.sha256((ART / modules[i]["ptx"]).read_bytes()).hexdigest()
        for i in selected
    },
    forward_sha256=hashlib.sha256(
        (ART / "cases/qwen64-t128/forward.bin").read_bytes()
    ).hexdigest(),
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
