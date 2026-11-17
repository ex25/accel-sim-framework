#!/usr/bin/env python3
"""Compare the complete output with saved framework or same-binary GPU output."""
import argparse
import array
import json
import math
from pathlib import Path


def values(directory):
    for name in ("output.f32", "reference.f32", "reference.bf16"):
        path = directory / name
        if path.exists():
            result = array.array("f")
            if path.suffix == ".bf16":
                raw = path.read_bytes()
                expanded = bytearray(2 * len(raw))
                expanded[2::4] = raw[0::2]
                expanded[3::4] = raw[1::2]
                result.frombytes(expanded)
            else:
                result.frombytes(path.read_bytes())
            return result
    raise FileNotFoundError("No saved MoE output in " + str(directory))


def timing(directory):
    path = directory / "run.json"
    if not path.exists():
        return None
    run = json.loads(path.read_text())
    if run["mode"] == "native":
        return json.loads((directory / "output.json").read_text())["cuda_event_ms"]
    # Missing device events remain unmeasured. ROI cycle checkpoints are a
    # different measurement and must not stand in for CUDA-event E2E.
    event_path = directory / "event-timing.json"
    if not event_path.exists():
        return None
    event = json.loads(event_path.read_text())
    samples = event.get("samples", [event])
    assert len(samples) == run["warmup"] + run["repeats"]
    return sum(s["device_elapsed_ms"] for s in samples[run["warmup"]:]) / run["repeats"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=.02)
    args = parser.parse_args()
    runs = [json.loads((p / "run.json").read_text()) if (p / "run.json").exists() else None
            for p in (args.reference, args.candidate)]
    if all(runs) and any(r["mode"] == "sim" for r in runs):
        if (runs[0]["binary_sha256"] != runs[1]["binary_sha256"] or
                runs[0]["forward_sha256"] != runs[1]["forward_sha256"] or
                runs[0]["parameters"] != runs[1]["parameters"]):
            parser.error("Simulator comparison requires the same binary, input and parameters")
    expected, actual = values(args.reference), values(args.candidate)
    if len(expected) != len(actual):
        raise ValueError("Output shapes differ")
    failed = different = 0
    maximum = error = norm = 0.0
    for ref, value in zip(expected, actual):
        delta = value - ref
        different += value != ref
        failed += not (math.isfinite(value) and math.isfinite(ref) and abs(delta) <= args.atol + args.rtol * abs(ref))
        maximum = max(maximum, abs(delta))
        error += delta * delta
        norm += ref * ref
    result = {"passed": failed == 0, "elements": len(actual), "different_elements": different,
              "bitwise_equal": expected.tobytes() == actual.tobytes(),
              "failed_elements": failed, "max_absolute_error": maximum,
              "relative_l2_error": math.sqrt(error / max(norm, 1e-30)),
              "reference_ms": timing(args.reference), "candidate_ms": timing(args.candidate)}
    matched_protocol = all(runs) and all(
        runs[0][key] == runs[1][key] for key in ("warmup", "iterations"))
    result["timing_protocol_matches"] = bool(matched_protocol)
    result["timing_scope"] = (
        "CUDA-event timing only. Simulator DMA duration and host submission gaps "
        "remain unmodeled. Final-output numerical comparison is independent of warmup count.")
    if matched_protocol and result["reference_ms"] and result["candidate_ms"] is not None:
        result["roi_time_relative_difference"] = (result["candidate_ms"] / result["reference_ms"]) - 1
    (args.candidate / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
