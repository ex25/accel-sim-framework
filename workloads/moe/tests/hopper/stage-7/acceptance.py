#!/usr/bin/env python3
"""Independently audit completed frozen forwards; never change tolerances."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

HERE = Path(__file__).resolve().parent
MOE = HERE.parents[4] / "workloads/moe"
sys.path.insert(0, str(HERE.parent / "stage-2"))
from read_forward import load


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run):
    manifest = json.loads((MOE / "artifacts/workloads.json").read_text())
    actual_run = json.loads((run / "run.json").read_text())
    name = actual_run["workload"]
    frozen = next(c for c in manifest["cases"] if c["name"] == name)
    reference = MOE / "artifacts/cases" / name
    assert sha(reference / "forward.bin") == frozen["forward_sha256"]
    assert sha(reference / "reference.bf16") == frozen["reference_sha256"]
    assert sha(MOE / "artifacts/moe_layer") == manifest["binary_sha256"]
    assert actual_run["binary_sha256"] == manifest["binary_sha256"]
    assert actual_run["forward_sha256"] == frozen["forward_sha256"]
    assert actual_run["parameters"] == frozen["framework"]["parameters"]
    assert actual_run["mode"] == "sim"
    assert (actual_run["warmup"], actual_run["iterations"],
            actual_run["repeats"]) == (0, 1, 1)
    assert actual_run["state"] == "completed" and actual_run["exit_code"] == 0
    assert "gpgpu-sim" in actual_run["loaded_libraries"]
    build = json.loads((run.parent / (name + ".build.json")).read_text())
    assert sha(run / "gpgpusim.config") == build["configuration_sha256"]
    log = (run / "stdout.log").read_text()
    assert "mode=performance simulation" in log
    assert re.search(r"simulation mode 0\b", log)
    assert log.count("MOE_ROI_BEGIN") == log.count("MOE_ROI_END") == 1
    registered = set(re.findall(r"__cudaRegisterFunction (moe_kernel_\w+)", log))
    assert registered == {m["entry"] for m in manifest["modules"]}
    # The frozen sm_90a modules have no ptxas system-shared allocation.
    # Compare compiler resource metadata, not a simulator layout function.
    static_shared = {}
    for metadata in run.glob("*.ptxas"):
        text = metadata.read_text()
        entry = re.search(r"Compiling entry function '([^']+)'", text)
        if entry:
            assert "system" not in text.lower()
            smem = re.search(r"(\d+) bytes smem", text)
            static_shared[entry[1]] = int(smem[1]) if smem else 0
    roi = log.split("MOE_ROI_BEGIN", 1)[1].split("MOE_ROI_END", 1)[0]
    events = [dict(token.split("=", 1) for token in line.split())
              for line in re.findall(r"^SIM_KERNEL (.*)$", roi, re.M)]
    launches = load(reference / "forward.bin")
    hardware = json.loads((MOE / "evidence/hardware-counters.json").read_text())
    hardware_case = next(c for c in hardware["cases"] if c["case"] == name)
    assert len(events) == len(launches) == len(hardware_case["launches"])
    assert [x["module"] for x in launches] == frozen["module_ids"]
    rows = []
    previous_end = -1
    for i, (event, launch, hw) in enumerate(zip(events, launches,
                                              hardware_case["launches"])):
        assert event["entry"] == manifest["modules"][launch["module"]]["entry"]
        assert event["entry"] == hw["entry"]
        for key, dimensions in (("grid", launch["shape"][:3]),
                                ("block", launch["shape"][3:6]),
                                ("cluster", launch["shape"][6:9])):
            assert tuple(map(int, event[key].split(","))) == tuple(dimensions)
        expected_shared = launch["shape"][9] + static_shared[event["entry"]] + 1024
        assert int(event["shared"]) == expected_shared
        start, end = int(event["start"]), int(event["end"])
        assert previous_end <= start < end
        previous_end = end
        milliseconds = (end - start) / float(event["core_MHz"]) / 1000
        hw_ms = hw["metrics"]["gpu__time_duration.sum"] / 1e6
        rows.append(dict(launch_index=i, module=launch["module"],
                         role=hw["role"], sim_ms=milliseconds, ncu_ms=hw_ms,
                         relative_error=milliseconds / hw_ms - 1,
                         wgmma=int(event["wgmma"]),
                         wall_seconds=float(event["wall_end"]) -
                         float(event["wall_start"])))
    expert_calls = [r for r in rows if r["module"] == 16]
    assert len(expert_calls) == 2 and all(r["wgmma"] > 0 for r in expert_calls)
    expected_count = actual_run["parameters"]["tokens"] * actual_run["parameters"]["hidden"]
    raw_reference = (reference / "reference.bf16").read_bytes()
    raw_actual = (run / "output.f32").read_bytes()
    assert len(raw_reference) == expected_count * 2
    assert len(raw_actual) == expected_count * 4
    failures = []
    different = nonfinite = not_bf16 = 0
    maximum = squared_error = squared_norm = 0.0
    for i in range(expected_count):
        # Decode the saved BF16 bits directly, independently of compare.values.
        bits = struct.unpack_from("<H", raw_reference, 2 * i)[0]
        ref = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
        value = struct.unpack_from("<f", raw_actual, 4 * i)[0]
        actual_bits = struct.unpack_from("<I", raw_actual, 4 * i)[0]
        finite = math.isfinite(ref) and math.isfinite(value)
        nonfinite += not finite
        not_bf16 += bool(actual_bits & 0xffff)
        different += value != ref
        delta = abs(value - ref)
        tolerance = 1e-5 + .02 * abs(ref)
        if not finite or delta > tolerance:
            failures.append(dict(index=i, token=i // actual_run["parameters"]["hidden"],
                                 channel=i % actual_run["parameters"]["hidden"],
                                 reference=ref, actual=value, absolute_error=delta,
                                 tolerance=tolerance, reference_bf16=hex(bits),
                                 actual_float_bits=hex(actual_bits)))
        if finite:
            maximum = max(maximum, delta)
            squared_error += delta * delta
            squared_norm += ref * ref
    numerical = dict(passed=not failures, elements=expected_count,
                     failed_elements=len(failures), different_elements=different,
                     nonfinite_elements=nonfinite, non_bf16_elements=not_bf16,
                     max_absolute_error=maximum,
                     relative_l2_error=math.sqrt(squared_error / max(squared_norm, 1e-30)))
    original = json.loads((run / "comparison.json").read_text())
    for key in ("passed", "elements", "failed_elements", "different_elements",
                "max_absolute_error", "relative_l2_error"):
        assert numerical[key] == original[key], (key, numerical[key], original[key])
    assert not not_bf16
    sim_sum, ref_sum = sum(r["sim_ms"] for r in rows), sum(r["ncu_ms"] for r in rows)
    major = [r for r in rows if r["module"] == 16 or "gating" in r["role"]
             or r["ncu_ms"] >= .05 * ref_sum]
    result = dict(case=name, execution_checks_passed=True, registered_entries=len(registered),
                  launches=len(events), numerical=numerical,
                  kernel_sum_sim_ms=sim_sum, kernel_sum_ncu_ms=ref_sum,
                  kernel_sum_relative_error=sim_sum / ref_sum - 1,
                  major_calls=major, rows=rows,
                  performance_within_20_percent=(abs(sim_sum / ref_sum - 1) <= .2
                                                and all(abs(r["relative_error"]) <= .2
                                                        for r in major)),
                  timing_scope="First CTA admission to last kernel completion; not CUDA-event end-to-end time.",
                  ncu_conditions=hardware["conditions"],
                  original_comparison_agrees=True,
                  validator_sha256=sha(Path(__file__)))
    (run / "independent-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    (run / "independent-acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(name, json.dumps(numerical), "kernel sum error", result["kernel_sum_relative_error"])
    return numerical["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    passed = [audit(run.resolve()) for run in args.runs]
    raise SystemExit(0 if all(passed) else 1)
