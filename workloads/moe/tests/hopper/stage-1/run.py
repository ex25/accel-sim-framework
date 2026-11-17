#!/usr/bin/env python3
"""Stage 1 self-tests, using stage 0's fatbinary registration and H100 setup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
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
p.add_argument("--config", type=Path)
p.add_argument("--sim-root", type=Path, default=SIM)
a = p.parse_args()
sim = a.sim_root.resolve()
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
config_source = a.config or sim / "configs/tested-cfgs/SM90_H100/gpgpusim.config"
config = config_source.read_text()
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
shutil.copy2(sim / "configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt", run)
env = os.environ.copy()
for key in ["PTX_SIM_USE_PTX_FILE", "PTX_SIM_KERNELFILE", "CUOBJDUMP_SIM_FILE"]:
    env.pop(key, None)
env["PTX_SIM_MODE_FUNC"] = "0"
source_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=sim)
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
for record in records:
    if not record["test"].startswith(
        ("global_load_completion_v", "global_hit_writeback_")
    ):
        continue
    log = (run / (record["test"] + ".log")).read_text()
    distribution = re.search(
        r"\ndistro:\n([0-9,\s]+)\ngpgpu_n_tot_thrd_icount", log)[1]
    issued = sum(map(int, re.findall(r"\d+", distribution)))
    completed = int(re.search(r"gpgpu_n_tot_w_icount = (\d+)", log)[1])
    record.update(issued_warp_instructions=issued,
                  completed_warp_instructions=completed)
    record["passed"] &= issued == completed
    print(f"{record['test']} issued={issued} completed={completed}", flush=True)
coalescing_expected = {
    "shared_interleaved_independent": 3,
    "shared_interleaved_early_use": 0,
    "shared_interleaved_base_change": 0,
    "shared_interleaved_barrier": 0,
    "shared_interleaved_store": 0,
    "shared_interleaved_branch": 0,
}
for record in records:
    test = record["test"]
    if test not in coalescing_expected:
        continue
    log = (run / (test + ".log")).read_text()
    matches = re.findall(r"coalesced (\d+) shared loads in '" + test + "'", log)
    count = sum(map(int, matches))
    record["coalesced_shared_loads"] = count
    record["passed"] &= count == coalescing_expected[test]
    print(f"{test} coalesced={count} expected={coalescing_expected[test]}",
          flush=True)
schedule_expected = {
    "shared_prefetch_independent": 1,
    "shared_prefetch_early_use": 0,
    "shared_prefetch_base_write": 0,
    "shared_prefetch_barrier": 0,
    "shared_prefetch_store": 0,
    "shared_prefetch_branch": 0,
}
for record in records:
    test = record["test"]
    if test not in schedule_expected:
        continue
    log = (run / (test + ".log")).read_text()
    matches = re.findall(r"scheduled (\d+) shared vector reads in '" + test +
                         "'", log)
    count = sum(map(int, matches))
    record["scheduled_shared_reads"] = count
    record["passed"] &= count == schedule_expected[test]
    print(f"{test} scheduled={count} expected={schedule_expected[test]}",
          flush=True)
bank_deltas = {
    "large_a": 0, "large_b": 24, "small_a": 0, "small_b": 16,
    "contiguous": 0, "broadcast": 1, "partial_broadcast": 0,
    "interleaved_broadcast": 2, "alternating_broadcast": 1,
    "odd_run_broadcast": 2,
    "conflict": 0, "active_half": 0,
}
by_name = {record["test"]: record for record in records}
for test, expected in {"shared_store_broadcast_v2": 1,
                       "shared_store_broadcast_v4": 3,
                       "shared_atomic_same_word": 31}.items():
    if test not in by_name:
        continue
    log = (run / (test + ".log")).read_text()
    conflicts = int(re.search(r"gpgpu_n_shmem_bkconflict = (\d+)", log)[1])
    by_name[test].update(shared_extra_service=conflicts,
                         expected_extra_service=expected)
    by_name[test]["passed"] &= conflicts == expected
    print(f"{test} extra_service={conflicts} expected={expected}", flush=True)
hit_writeback_check = None
hit_pair = ("global_hit_writeback_same", "global_hit_writeback_spread")
if all(test in by_name for test in hit_pair):
    cycles, accesses, hits = [], [], []
    for test in hit_pair:
        log = (run / (test + ".log")).read_text()
        cycles.append(int(re.search(r"^gpu_sim_cycle = (\d+)$", log,
                                    re.MULTILINE)[1]))
        for counts, status in ((accesses, "TOTAL_ACCESS"), (hits, "HIT")):
            counts.append(int(re.search(
                r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[" +
                status + r"\] = (\d+)", log)[1]))
    # H100 configuration: one write per RF bank per cycle. Four hot
    # destinations in one bank require three more cycles than distinct banks.
    # Equal cache counters also catch accidentally repeating cache access while
    # waiting for the write port. The old HIT bypass gives a zero cycle delta.
    passed = cycles[0] - cycles[1] == 3 and accesses == [2, 2] and hits == [1, 1]
    hit_writeback_check = dict(cycles=cycles, delta=cycles[0] - cycles[1],
                               expected_delta=3, read_accesses=accesses,
                               read_hits=hits, passed=passed)
    for test in hit_pair:
        by_name[test]["passed"] &= passed
    print(f"global_hit_writeback {hit_writeback_check}", flush=True)
bank_checks = {}
for pattern, expected in bank_deltas.items():
    scalar = f"shared_bank_{pattern}_scalar"
    vector = f"shared_bank_{pattern}_v4"
    if scalar not in by_name or vector not in by_name:
        continue
    counts = []
    for test in (scalar, vector):
        log = (run / (test + ".log")).read_text()
        match = re.search(r"gpgpu_n_shmem_bkconflict = (\d+)", log)
        counts.append(int(match.group(1)) if match else None)
    delta = counts[1] - counts[0] if None not in counts else None
    passed = delta == expected
    bank_checks[pattern] = dict(scalar=counts[0], vector=counts[1],
                                delta=delta, expected=expected, passed=passed)
    by_name[vector]["passed"] &= passed
    print(f"shared_bank_{pattern} delta={delta} expected={expected}",
          flush=True)
# v2 expectations are independent H100 wavefront deltas, not predictions
# generated from the simulator's request partition implementation.
v2_deltas = {"large_a": 0, "large_b": 8, "small_a": 0, "small_b": 16,
             "contiguous": 0, "broadcast": 0, "partial_broadcast": 0,
             "interleaved_broadcast": 2, "alternating_broadcast": 0,
             "odd_run_broadcast": 0, "conflict": 0, "active_half": 0}
for pattern, expected in v2_deltas.items():
    scalar, vector = (f"shared_bank_{pattern}_{suffix}"
                      for suffix in ("scalar", "v2"))
    if scalar not in by_name or vector not in by_name:
        continue
    counts = [int(re.search(r"gpgpu_n_shmem_bkconflict = (\d+)",
                           (run / (test + ".log")).read_text())[1])
              for test in (scalar, vector)]
    passed = counts[1] - counts[0] == expected
    bank_checks[pattern + "_v2"] = dict(scalar=counts[0], vector=counts[1],
                                        delta=counts[1] - counts[0],
                                        expected=expected, passed=passed)
    by_name[vector]["passed"] &= passed
    print(f"shared_bank_{pattern}_v2 {bank_checks[pattern + '_v2']}", flush=True)
metadata = dict(
    library_sha256=library_sha256,
    source_diff_sha256=hashlib.sha256(source_diff).hexdigest(),
    simulator_root=str(sim),
    binary_sha256=hashlib.sha256((run / "micro").read_bytes()).hexdigest(),
    commands=commands,
    libraries=libraries,
    tests=records,
    bank_checks=bank_checks,
    hit_writeback_check=hit_writeback_check,
    config_sha256=hashlib.sha256((run / "gpgpusim.config").read_bytes()).hexdigest(),
    config_source=str(config_source.resolve()),
    outer=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip(),
    inner=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=sim, text=True
    ).strip(),
    cuda=subprocess.check_output([cuda / "bin/nvcc", "--version"], text=True),
)
(run / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
raise SystemExit(0 if all(x["passed"] for x in records) else 1)
