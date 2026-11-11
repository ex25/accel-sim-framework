#!/usr/bin/env python3
"""Run frozen gating alone, optionally restricting it to one CTA/K tile."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from read_forward import load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SIM = ROOT / "gpu-simulator/gpgpu-sim"
ART = ROOT / "workloads/moe/artifacts"
case_launches = {
    name: load(ART / "cases" / name / "forward.bin")
    for name in ["qwen64-t128", "phi-t128", "mixtral-t128"]
}
launches = [x for v in case_launches.values() for x in v]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run-dir", type=Path, required=True)
p.add_argument("--test", default="all")
p.add_argument("--full-config", action="store_true")
p.add_argument("--qwen-blocks", type=int, choices=[1, 8], default=8)
p.add_argument("--qwen-iterations", type=int, choices=[1, 16], default=16)
p.add_argument("--single-cta", action="store_true")
p.add_argument("--single-iteration", action="store_true")
a = p.parse_args()
run = a.run_dir.resolve()
run.mkdir(parents=True, exist_ok=True)
cuda = Path(os.environ.get("CUDA_INSTALL_PATH", "/usr/local/cuda"))
commands = []


def call(cmd, **kw):
    commands.append(list(map(str, cmd)))
    subprocess.run(cmd, check=True, **kw)


modules = json.loads((ART / "workloads.json").read_text())["modules"]
selected = [71, 41, 2]
inc = []
asm = []
meta = []
embedded_ptx = {}
for index in selected:
    module = modules[index]
    spec = next(x for x in launches if x["module"] == index)
    ptx = ART / module["ptx"]
    if index == 71 and (a.qwen_iterations == 1 or a.single_iteration):
        # Keep matrix strides and shared access patterns; execute only the
        # first K=128 tile. This copy is diagnostic, not a hardware baseline.
        text = ptx.read_text()
        assert text.count("%rd93, 7680;") == text.count("%rd93, 8192;") == 1
        text = text.replace("%rd93, 7680;", "%rd93, 0;")
        text = text.replace("%rd93, 8192;", "%rd93, 512;")
        ptx = run / "qwen-one-loop.ptx"
        ptx.write_text(text)
    elif index in (2, 41) and a.single_iteration:
        # Preserve all strides, pipeline setup and shared address expressions.
        text = ptx.read_text()
        assert text.count(", 16384;") == 2
        # The other 16384 immediate is the shared weight-buffer base.
        import re
        text, n = re.subn(r"(setp.ne.s64[^\n]*), 16384;", r"\1, 128;", text)
        assert n == 1
        ptx = run / f"{index}-one-loop.ptx"
        ptx.write_text(text)
    embedded_ptx[str(index)] = hashlib.sha256(ptx.read_bytes()).hexdigest()
    call(
        [
            cuda / "bin/fatbinary",
            "--64",
            "--create=" + str(run / f"{index}.fatbin"),
            "--image=profile=compute_90a,file=" + str(ptx),
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
for test in (["qwen", "phi", "mixtral"] if a.test == "all" else a.test.split(",")):
    cmd = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(run / (test + ".time")),
        str(run / "real"),
        test,
        str(a.qwen_blocks),
        str(a.qwen_iterations),
        str(int(a.single_cta)),
        str(int(a.single_iteration)),
    ]
    commands.append(cmd)
    with (run / (test + ".log")).open("w") as log:
        result = subprocess.run(
            cmd, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600
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
    qwen_blocks=a.qwen_blocks,
    qwen_iterations=a.qwen_iterations,
    single_cta=a.single_cta,
    single_iteration=a.single_iteration,
    embedded_ptx_sha256=embedded_ptx,
    config_sha256=hashlib.sha256((run / "gpgpusim.config").read_bytes()).hexdigest(),
    ptx_sha256={
        str(i): hashlib.sha256((ART / modules[i]["ptx"]).read_bytes()).hexdigest()
        for i in selected
    },
    forward_sha256={
        name: hashlib.sha256(
            (ART / "cases" / name / "forward.bin").read_bytes()
        ).hexdigest()
        for name in case_launches
    },
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
