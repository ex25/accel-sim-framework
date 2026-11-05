#!/usr/bin/env python3
"""Run a fixed Megatron-TE forward on Hopper or through the PTX simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
REPOSITORY = ROOT.parent.parent


def workload(case, tokens):
    manifest = json.loads((ARTIFACTS / "workloads.json").read_text())
    selected = next(r for r in manifest["cases"] if r["name"] == case + "-t" + str(tokens))
    path = ARTIFACTS / "cases" / selected["name"] / "forward.bin"
    if hashlib.sha256(path.read_bytes()).hexdigest() != selected["forward_sha256"]:
        raise ValueError("Launch input differs from the fixed workload")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["mixtral", "qwen64", "phi"])
    parser.add_argument("--tokens", type=int, choices=[128, 4096, 8192], default=128)
    parser.add_argument("--mode", choices=["native", "sim"], default="native")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.repeats < 1:
        parser.error("--warmup must be nonnegative; --iterations/--repeats must be positive")
    if args.list:
        manifest = json.loads((ARTIFACTS / "workloads.json").read_text())
        for name in ("mixtral", "qwen64", "phi"):
            c = next(case["framework"]["parameters"] for case in manifest["cases"]
                     if case["name"] == name + "-t128")
            print(name, "H=", c["hidden"], "I=", c["intermediate"],
                  "E=", c["experts"], "K=", c["top_k"], "tokens=128/4096/8192")
        return
    if not args.case:
        parser.error("--case is required")
    if args.mode == "sim" and (args.warmup != 0 or args.iterations != 1 or args.repeats != 1):
        parser.error("Simulation uses exactly one forward without warmup")
    selected = workload(args.case, args.tokens)
    binary = ARTIFACTS / "moe_layer"
    record = json.loads((ARTIFACTS / "workloads.json").read_text())
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if digest != record["binary_sha256"]:
        parser.error("Executable differs from the validated workload")
    libraries = subprocess.check_output(["ldd", str(binary)], text=True)
    simulated = any("libcudart" in line and "gpgpu-sim" in line for line in libraries.splitlines())
    if (args.mode == "sim") != simulated:
        parser.error("ldd must resolve libcudart to gpgpu-sim only for --mode sim")
    if args.run_dir:
        directory = args.run_dir.resolve()
        directory.mkdir(parents=True, exist_ok=False)
    else:
        (ROOT / "runs").mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=selected["name"] + "-" + args.mode + "-", dir=ROOT / "runs"))
    if args.mode == "sim":
        config = REPOSITORY / "gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100"
        for name in ("gpgpusim.config", "config_hopper_islip.icnt"):
            shutil.copy2(config / name, directory / name)
    command = [str(binary), "--input", str(ARTIFACTS / "cases" / selected["name"] / "forward.bin"),
               "--output", str(directory / "output"), "--warmup", str(args.warmup),
               "--iterations", str(args.iterations), "--repeats", str(args.repeats)]
    if args.mode == "sim":
        command.append("--simulation")
    manifest = {"workload": selected["name"], "mode": args.mode,
                "parameters": selected["framework"]["parameters"], "warmup": args.warmup,
                "iterations": args.iterations, "repeats": args.repeats, "binary_sha256": digest,
                "forward_sha256": selected["forward_sha256"],
                "loaded_libraries": libraries, "command": command, "state": "running"}
    path = directory / "run.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    with (directory / "stdout.log").open("w") as log:
        process = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(exit_code=process.returncode, state="completed" if not process.returncode else "failed")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Results:", directory)
    print((directory / "stdout.log").read_text()[-3000:])
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
