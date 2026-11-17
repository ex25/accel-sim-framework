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
    parser.add_argument("--config-dir", type=Path,
                        help="Simulator configuration directory (default: SM90_H100)")
    parser.add_argument("--sim-events", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Record simulator CUDA-event device cycles through GDB "
                             "(default: enabled in sim mode)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.sim_events is None:
        args.sim_events = args.mode == "sim"
    if args.sim_events and args.mode != "sim":
        parser.error("--sim-events requires --mode sim")
    if args.config_dir and args.mode != "sim":
        parser.error("--config-dir requires --mode sim")
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
    if args.mode == "sim" and args.iterations != 1:
        parser.error("Simulation records one forward per event interval; use --iterations 1")
    if args.mode == "sim" and not args.sim_events and (args.warmup or args.repeats != 1):
        parser.error("Multiple simulator forwards require CUDA-event collection")
    selected = workload(args.case, args.tokens)
    binary = ARTIFACTS / "moe_layer"
    record = json.loads((ARTIFACTS / "workloads.json").read_text())
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if digest != record["binary_sha256"]:
        parser.error("Executable differs from the validated workload")
    libraries = subprocess.check_output(["ldd", str(binary)], text=True)
    runtime_library = next((Path(line.split("=>")[1].split("(")[0].strip())
                            for line in libraries.splitlines()
                            if "libcudart" in line and "gpgpu-sim" in line), None)
    simulated = runtime_library is not None
    if (args.mode == "sim") != simulated:
        parser.error("ldd must resolve libcudart to gpgpu-sim only for --mode sim")
    if args.run_dir:
        directory = args.run_dir.resolve()
        directory.mkdir(parents=True, exist_ok=False)
    else:
        (ROOT / "runs").mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=selected["name"] + "-" + args.mode + "-", dir=ROOT / "runs"))
    if args.mode == "sim":
        config = (args.config_dir or REPOSITORY / "gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100").resolve()
        for name in ("gpgpusim.config", "config_hopper_islip.icnt"):
            shutil.copy2(config / name, directory / name)
    # The frozen launcher only brackets repeats with events. In simulation, run
    # warmups through that same loop so their device timestamps are saved too.
    launcher_warmup = 0 if args.sim_events else args.warmup
    launcher_repeats = args.warmup + args.repeats if args.sim_events else args.repeats
    command = [str(binary), "--input", str(ARTIFACTS / "cases" / selected["name"] / "forward.bin"),
               "--output", str(directory / "output"), "--warmup", str(launcher_warmup),
               "--iterations", str(args.iterations), "--repeats", str(launcher_repeats)]
    if args.mode == "sim" and not args.sim_events:
        command.append("--simulation")
    if args.sim_events:
        observer = ROOT / "tests/hopper/stage-7/event_timing.py"
        command = ["gdb", "--quiet", "--batch", "--return-child-result",
                   "-ex", "source " + str(observer), "-ex", "run", "--args"] + command
    manifest = {"workload": selected["name"], "mode": args.mode,
                "parameters": selected["framework"]["parameters"], "warmup": args.warmup,
                "iterations": args.iterations, "repeats": args.repeats, "binary_sha256": digest,
                "forward_sha256": selected["forward_sha256"],
                "loaded_libraries": libraries, "command": command, "state": "running",
                "simulator_event_observer": args.sim_events}
    manifest["launcher_warmup"] = launcher_warmup
    manifest["launcher_repeats"] = launcher_repeats
    if args.mode == "sim":
        manifest["configuration_source"] = str(config)
        manifest["configuration_sha256"] = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("gpgpusim.config", "config_hopper_islip.icnt")}
        manifest["runtime_library"] = str(runtime_library)
        manifest["runtime_library_sha256"] = hashlib.sha256(
            runtime_library.read_bytes()).hexdigest()
    path = directory / "run.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    with (directory / "stdout.log").open("w") as log:
        process = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(exit_code=process.returncode, state="completed" if not process.returncode else "failed")
    if args.sim_events and process.returncode == 0:
        timing = json.loads((directory / "event-timing.json").read_text())
        samples = timing["samples"]
        assert len(samples) == args.warmup + args.repeats
        for i, sample in enumerate(samples):
            sample["phase"] = "warmup" if i < args.warmup else "measured"
            sample["phase_index"] = i if i < args.warmup else i - args.warmup
        (directory / "event-timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        manifest["simulated_forward_event_ms"] = sum(
            s["device_elapsed_ms"] for s in samples[args.warmup:]) / args.repeats
        manifest["simulation_event_samples"] = samples
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Results:", directory)
    print((directory / "stdout.log").read_text()[-3000:])
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
