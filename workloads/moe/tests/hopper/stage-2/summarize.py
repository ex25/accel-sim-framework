"""Assert saved self-test evidence and collect cycles/host costs (not acceptance)."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run_dir", type=Path)
a = p.parse_args()
base = a.run_dir.resolve()


def events(directory, name):
    s = (directory / ("events-" + name + ".log")).read_text()
    assert "exited normally" in s and "Python Exception" not in s
    return [
        json.loads(line.split("OBS ", 1)[1])
        for line in s.splitlines()
        if "OBS " in line
    ]


def program(directory, name):
    s = (directory / (name + ".log")).read_text()
    assert "PASS" in s and not re.search(r"^FAIL\b", s, re.M)

    def number(label):
        v = re.findall(re.escape(label) + r"\s*=\s*([0-9]+)", s)
        return int(v[-1]) if v else 0

    marks = {k: float(v) for k, v in re.findall(r"HOST_TIME (\w+) ([\d.]+)", s)}
    t = (directory / (name + ".time")).read_text()
    return dict(
        cycles_including_launch=number("gpu_sim_cycle"),
        thread_instructions=number("gpu_sim_insn"),
        global_read_transactions=number("gpgpu_n_mem_read_global"),
        l2_global_read_sectors=number(
            "L2_cache_stats_breakdown[GLOBAL_ACC_R][TOTAL_ACCESS]"
        ),
        l2_global_write_sectors=number(
            "L2_cache_stats_breakdown[GLOBAL_ACC_W][TOTAL_ACCESS]"
        ),
        dram_reads=number("total dram reads"),
        dram_writes=number("total dram writes"),
        initialization_seconds=marks["kernel_begin"] - marks["process_init_begin"],
        launch_sync_seconds=marks["kernel_done"] - marks["kernel_begin"],
        comparison_seconds=marks["compare_end"] - marks["compare_begin"],
        max_rss_kib=int(
            re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)[1]
        ),
        process_wall=re.search(
            r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)", t
        )[1],
    )


summary = {"self_test_only": True}
for d in [
    "delivery-micro",
    "delivery-latency1237",
    "delivery-lanes",
    "release-micro",
    "stage0-regression",
    "stage1-delivery",
]:
    m = json.loads((base / d / "run.json").read_text())
    assert all(t["passed"] for t in m["tests"]), d
    summary[d] = (
        m["tests"]
        if d == "stage0-regression"
        else {t["test"]: program(base / d, t["test"]) for t in m["tests"]}
    )
# Check all real request-return counts, unique completion and data ordering.
summary["events"] = {}
for d, names in [
    (
        "delivery-micro",
        ["copy_mix", "zero", "groups6", "identity", "reuse", "overlap", "double"],
    ),
    ("delivery-latency1237", ["copy_mix", "overlap", "double"]),
    ("delivery-lanes", ["lane_groups"]),
]:
    summary["events"][d] = {}
    for name in names:
        es = events(base / d, name)
        copies = {e["uid"]: e for e in es if e["kind"] == "issue" and e["op"] == 1}
        done = {e["uid"]: e for e in es if e["kind"] == "shared_delivered"}
        returns = Counter(e["uid"] for e in es if e["kind"] == "request_return")
        assert copies.keys() == done.keys(), (d, name)
        assert sum(e["kind"] == "shared_delivered" for e in es) == len(copies)
        for uid, issue in copies.items():
            assert returns[uid] == issue["requests"], (d, name, uid)
            assert done[uid]["cycle"] >= issue["cycle"]
            rs = [
                e["cycle"]
                for e in es
                if e["kind"] == "request_return" and e["uid"] == uid
            ]
            if rs:
                assert done[uid]["cycle"] >= max(rs)
        summary["events"][d][name] = dict(
            copies=len(copies),
            requests=sum(returns.values()),
            issue_cycles=[e["cycle"] for e in copies.values()],
            delivery_cycles=[e["cycle"] for e in done.values()],
            consumer_cycles=[e["cycle"] for e in es if e["kind"] == "shared_consume"],
            wait_transitions=[e for e in es if e["kind"] == "wait_state"],
        )
        if name in ["copy_mix", "zero"]:
            assert len(copies) == 1
            issue = next(iter(copies.values()))
            delivery = next(iter(done.values()))
            consumes = [e for e in es if e["kind"] == "shared_consume"]
            assert len(consumes) == 1 and consumes[0]["cycle"] > delivery["cycle"]
            assert issue["requests"] == (8 if name == "copy_mix" else 0)
            if name == "copy_mix":
                assert len(issue["copies"]) == 24
                assert Counter(c["size"] for c in issue["copies"]) == {16: 16, 0: 8}
            for dec in [e for e in es if e["kind"] == "decode"]:
                assert sum(v > 0 for v in dec["sources"]) == 3
                assert not any(dec["outputs"]) and dec["data_size"] == 16
                assert dec["cache"] == 6  # CACHE_GLOBAL (.cg)
        if name == "lane_groups":
            assert len(copies) == 1
            delivered = next(iter(done.values()))
            assert [c["group"] for c in delivered["copies"]] == [0] * 16 + [1] * 16
            commits = [e for e in es if e["kind"] == "issue" and e["op"] == 2]
            assert [e["mask"] for e in commits] == [0xFFFF0000, 0xFFFFFFFF]
        if name == "identity":
            x, y = copies.values()
            assert x["pc"] == y["pc"] and x["uid"] != y["uid"]
            assert y["cycle"] < min(e["cycle"] for e in done.values())
            assert x["copies"][0]["src"] == y["copies"][0]["src"]
            assert x["copies"][0]["dst"] != y["copies"][0]["dst"]
        if name == "reuse":
            assert len(copies) == 8
            assert len({e["copies"][0]["shared"] for e in copies.values()}) == 2
        if name == "overlap":
            issue = next(iter(copies.values()))
            delivery = next(iter(done.values()))
            for warp in [0, 1]:
                assert any(
                    e["kind"] == "arithmetic"
                    and e["warp"] == warp
                    and issue["cycle"] < e["cycle"] < delivery["cycle"]
                    for e in es
                ), (d, warp)
        if name == "double":
            # Next tile is already in flight when the previous tile is read.
            issues = list(copies.values())
            assert len(issues) == 4
            assert any(
                issues[1]["cycle"] < e["cycle"] < done[issues[1]["uid"]]["cycle"]
                for e in es
                if e["kind"] == "shared_consume"
            )
for name in ["copy_mix", "overlap"]:
    x = summary["events"]["delivery-micro"][name]
    y = summary["events"]["delivery-latency1237"][name]
    assert y["delivery_cycles"][0] > x["delivery_cycles"][0]
    assert y["consumer_cycles"][0] > x["consumer_cycles"][0]
for d in ["delivery-micro", "delivery-latency1237"]:
    assert (
        summary[d]["double"]["cycles_including_launch"]
        < summary[d]["single"]["cycles_including_launch"]
    )
    assert summary[d]["zero"]["global_read_transactions"] == 0
    assert summary[d]["copy_mix"]["global_read_transactions"] == 8
for name, result in summary["delivery-micro"].items():
    assert (
        result["cycles_including_launch"]
        == summary["release-micro"][name]["cycles_including_launch"]
    )
assert (
    summary["release-micro"]["lane_groups"]["cycles_including_launch"]
    == summary["delivery-lanes"]["lane_groups"]["cycles_including_launch"]
)
real_run = json.loads((base / "real-release/run.json").read_text())
assert all(t["passed"] for t in real_run["tests"])
summary["real"] = {
    name: program(base / "real-release", name) for name in ["qwen", "mixtral"]
}
for name, n in [("qwen", 8192), ("mixtral", 1024)]:
    actual = (base / "real-release" / (name + ".actual")).read_bytes()
    expected = (base / "real-release" / (name + ".expected")).read_bytes()
    assert len(actual) == 4 * n and actual == expected
    summary["real"][name]["exact_elements"] = n
(base / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("PASS stage2 saved self-test evidence; summary.json written")
