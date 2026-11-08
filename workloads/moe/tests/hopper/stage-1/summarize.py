#!/usr/bin/env python3
"""Check the stage 1 delivery logs and collect device counts and host costs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("runs", type=Path)
a = p.parse_args()
root = a.runs.resolve()
selections = {
    "micro-delivery": [
        "numeric",
        "collectives",
        "bits_loads",
        "warp_full",
        "warp_partial",
        "chain32",
        "chain64",
        "throughput",
        "legacy",
    ],
    "collectives-runtime": ["collectives"],
    "final-warp-mask": ["warp_full", "warp_partial", "collectives", "numeric"],
    "latency16-delivery": ["chain32", "chain64"],
    "initiation2": ["throughput"],
    "initiation8": ["throughput"],
    "stage0-regression": [
        "launch",
        "work",
        "modules",
        "parameters",
        "shared",
        "normal",
        "assert",
        "trap",
    ],
    "real-corrected": ["convert"],
    "mapping-delivery": ["mapping"],
    "real-final": ["swiglu"],
    "route-delivery": ["route"],
}
summary = {}
for folder, tests in selections.items():
    records = json.loads((root / folder / "run.json").read_text())["tests"]
    by_name = {x["test"]: x for x in records}
    for test in tests:
        assert by_name[test]["passed"], (folder, test, by_name[test])
        log = (root / folder / (test + ".log")).read_text()
        assert "Syntax error" not in log and "requires split arrival" not in log
        stamps = defaultdict(list)
        for phase, value in re.findall(r"HOST_TIME (\S+) ([0-9.]+)", log):
            stamps[phase].append(float(value))
        timing = (root / folder / (test + ".time")).read_text()
        rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", timing)[1])
        wall = re.search(r"Elapsed \(wall clock\) time .*?: ([^\n]+)", timing)[1]
        data = dict(
            passed=True,
            rss_kib=rss,
            process_wall=wall,
            per_kernel_cycles_including_launch=list(
                map(int, re.findall(r"\bgpu_sim_cycle\s*=\s*(\d+)", log))
            ),
            per_kernel_thread_instructions=list(
                map(int, re.findall(r"\bgpu_sim_insn\s*=\s*(\d+)", log))
            ),
            cumulative_cycles=list(
                map(int, re.findall(r"gpu_tot_sim_cycle\s*=\s*(\d+)", log))
            ),
            results=[x for x in log.splitlines() if x.startswith("PASS")],
        )
        if stamps["kernel_begin"]:
            data["initialization_and_input_preparation_s"] = (
                stamps["kernel_begin"][0] - stamps["process_init_begin"][0]
            )
            data["launch_to_sync_s"] = [
                end - begin
                for begin, end in zip(stamps["kernel_begin"], stamps["kernel_done"])
            ]
        data["comparison_loops_s"] = sum(
            end - begin
            for begin, end in zip(stamps["compare_begin"], stamps["compare_end"])
        )
        summary[folder + "/" + test] = data


def observations(path):
    text = path.read_text()
    assert "Python Exception" not in text and "exited normally" in text, path
    return [json.loads(x[4:]) for x in text.splitlines() if x.startswith("OBS ")]


for folder, interval in [("initiation2", 2), ("initiation8", 8)]:
    units = defaultdict(list)
    for row in observations(root / folder / "observe-unit.log"):
        if row["kind"] == "unit_accept":
            units[row["unit"]].append(row["cycle"])
    gaps = [[b - a for a, b in zip(v, v[1:])] for v in units.values()]
    assert len(units) == 4 and all(min(x) == interval for x in gaps)
    summary[folder + "/unit_accept"] = {
        "minimum_intervals": [min(x) for x in gaps],
        "samples": sum(map(len, units.values())),
    }
for test in ["chain32", "chain64"]:
    lo = summary["micro-delivery/" + test]["cumulative_cycles"][-1]
    hi = summary["latency16-delivery/" + test]["cumulative_cycles"][-1]
    assert hi - lo == int(test[5:]) * 12
rows = observations(root / "micro-final/observe-warp.log")
shared_done = next(
    x["cycle"]
    for x in rows
    if x["kind"] == "complete" and x.get("opcode") == "st" and x["warp"] == 0
)
release = next(x["cycle"] for x in rows if x["kind"] == "warp_barrier_release")
barrier_pc = next(
    x["pc"] for x in rows if x["kind"] == "issue" and x.get("opcode") == "bar"
)
load = next(
    x["cycle"]
    for x in rows
    if x["kind"] == "issue" and x.get("opcode") == "ld" and x["pc"] > barrier_pc
)
assert shared_done <= release < load
summary["warp_memory_order"] = dict(
    shared_store_complete=shared_done,
    barrier_release=release,
    following_load_issue=load,
)
partial = observations(root / "micro-delivery/observe-partial.log")
assert any(
    x["kind"] == "collective" and x["mask"] == x["arrived"] == 0xFFFF0000
    for x in partial
)
assert any(x["kind"] == "warp_barrier_release" for x in partial)
final_partial = observations(root / "final-warp-mask/observe-partial.log")
assert any(
    x["kind"] == "decode"
    and x["opcode"] == "bar"
    and x["sources"][0] > 0
    and not any(x["destinations"])
    for x in final_partial
)
rows = observations(root / "collectives-runtime/observe.log")
assert any(
    x["kind"] == "decode" and x["opcode"] == "shfl" and all(x["sources"][:4])
    for x in rows
)
assert any(
    x["kind"] == "decode"
    and x["opcode"] == "elect"
    and x["destinations"][0] == 0
    and x["destinations"][1] > 0
    for x in rows
)
assert any(
    x["kind"] == "collective" and x["mask"] == x["arrived"] == 0xFFFF0000 for x in rows
)
rows = observations(root / "micro-delivery/observe-numeric.log")
assert any(
    x["kind"] == "decode" and x["opcode"] == "cvt" and all(x["sources"][:2])
    for x in rows
)
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("PASS delivery evidence", len(summary), "records")
