#!/usr/bin/env python3
"""Check ordinary global v2/v4 byte coverage and aligned sector requests."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SIM = ROOT / "gpu-simulator/gpgpu-sim"
TESTS = ("aligned_v2", "aligned_v4", "store_read_v2", "store_read_v4")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def last_int(pattern, log):
    values = re.findall(pattern, log, re.MULTILINE)
    return int(values[-1]) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, default=SIM)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--test", default="all")
    args = parser.parse_args()
    sim = args.sim_root.resolve()
    library = args.library.resolve()
    if library.is_dir():
        library /= "libcudart.so"
    library = library.resolve()
    config = (args.config or sim / "configs/tested-cfgs/SM90_H100/"
              "gpgpusim.config").resolve()
    icnt = sim / "configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt"
    tests = TESTS if args.test == "all" else tuple(args.test.split(","))
    if not tests or any(test not in TESTS for test in tests):
        parser.error("--test must name aligned_v2/v4 or store_read_v2/v4")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    cuda = Path(os.environ.get("CUDA_INSTALL_PATH", "/usr/local/cuda"))
    binary = run / "vector_memory"
    build = [str(cuda / "bin/nvcc"), "-std=c++17", "-O2", "-arch=sm_90a",
             "--cudart", "shared", str(HERE / "vector_memory.cu"), "-o",
             str(binary)]
    with (run / "build.log").open("w") as log:
        subprocess.run(build, check=True, stdout=log, stderr=subprocess.STDOUT)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(library.parent) + ":" + env.get(
        "LD_LIBRARY_PATH", "")
    env["PTX_SIM_MODE_FUNC"] = "0"
    for key in ("PTX_SIM_USE_PTX_FILE", "PTX_SIM_KERNELFILE",
                "CUOBJDUMP_SIM_FILE", "MOE_TRACE_LAUNCHES"):
        env.pop(key, None)
    ldd = subprocess.check_output(["ldd", str(binary)], env=env, text=True)
    loaded = next(line.split("=>", 1)[1].split("(", 1)[0].strip()
                  for line in ldd.splitlines() if "libcudart" in line)
    if Path(loaded).resolve() != library:
        raise RuntimeError("unexpected libcudart:\n" + ldd)
    records = []
    for test in tests:
        directory = run / test
        directory.mkdir()
        shutil.copy2(config, directory / "gpgpusim.config")
        shutil.copy2(icnt, directory / icnt.name)
        command = ["/usr/bin/time", "-v", "-o", str(directory / "time.log"),
                   str(binary), test]
        with (directory / "stdout.log").open("w") as log:
            result = subprocess.run(command, cwd=directory, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=180)
        output = (directory / "stdout.log").read_text()
        width = 2 if test.endswith("v2") else 4
        sectors = width * 4
        cycles = [int(x) for x in re.findall(r"^gpu_sim_cycle = (\d+)$",
                                                 output, re.MULTILINE)]
        read_requests = last_int(
            r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]"
            r"\[TOTAL_ACCESS\] = (\d+)", output)
        record = dict(test=test, command=command, returncode=result.returncode,
                      cycles=cycles, read_requests=read_requests,
                      expected_read_requests=sectors,
                      output_sha256=sha(directory / "stdout.log"),
                      config_sha256=sha(directory / "gpgpusim.config"))
        record["passed"] = (result.returncode == 0 and
                            f"PASS {test} elements={width * 32} errors=0" in output
                            and len(cycles) == (2 if test.startswith("store") else 1)
                            and read_requests == sectors)
        if test.startswith("store"):
            record["l2_read_hit"] = last_int(
                r"L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\] = (\d+)",
                output)
            record["l2_read_sector_miss"] = last_int(
                r"L2_cache_stats_breakdown\[GLOBAL_ACC_R\]"
                r"\[SECTOR_MISS\] = (\d+)", output)
            record["passed"] &= (record["l2_read_hit"] == sectors and
                                  record["l2_read_sector_miss"] == 0)
        records.append(record)
        print(record, flush=True)
    source_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"],
                                          cwd=sim)
    (run / "inner.diff").write_bytes(source_diff)
    metadata = dict(build_command=build, tests=records, ldd=ldd,
                    loaded_library=str(library), library_sha256=sha(library),
                    binary_sha256=sha(binary),
                    source_sha256=sha(HERE / "vector_memory.cu"),
                    runner_sha256=sha(HERE / "vector_memory.py"),
                    source_diff_sha256=hashlib.sha256(source_diff).hexdigest(),
                    config_source=str(config), config_sha256=sha(config),
                    icnt_source=str(icnt), icnt_sha256=sha(icnt),
                    simulator_root=str(sim),
                    outer=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                  cwd=ROOT, text=True).strip(),
                    inner=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                  cwd=sim, text=True).strip())
    (run / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0 if all(record["passed"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
