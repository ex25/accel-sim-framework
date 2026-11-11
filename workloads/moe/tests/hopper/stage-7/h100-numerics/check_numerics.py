#!/usr/bin/env python3
"""Check production arithmetic offline against a downloaded H100 evidence batch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from prepare import recipe

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
SIM = ROOT / "gpu-simulator/gpgpu-sim"


def expand(x):
    return (np.asarray(x, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def rounded(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--literature", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    capture, run, literature = [p.resolve() for p in (args.capture, args.run_dir, args.literature)]
    run.mkdir(parents=True, exist_ok=True)
    library = run / "model_probe.so"
    command = ["g++", "-O3", "-shared", "-fPIC", "-std=c++17", "-ffp-contract=off",
               "-I" + str(SIM / "src/cuda-sim"), str(HERE / "numerics.cc"), "-o", str(library)]
    subprocess.run(command, check=True)
    lib = ctypes.CDLL(str(library))
    ptr = ctypes.c_void_p
    for name in ("math_ops", "activation_stages"):
        getattr(lib, name).argtypes = [ctypes.c_uint, ptr, ptr]
    lib.blocks.argtypes = [ctypes.c_uint] + [ptr] * 4
    lib.dot.argtypes = [ctypes.c_uint] * 2 + [ptr] * 4
    report = dict(command=command, header_sha256={name: hashlib.sha256(
        (SIM / "src/cuda-sim" / name).read_bytes()).hexdigest()
        for name in ("nvidia_math.h", "hopper_math.h")})
    # Both outputs and inputs are the original downloaded files, not generated
    # golden values from the production arithmetic under test.
    for name, inputs, expected, columns in (
        ("math_ops", "ops.f32", "ops.cubin.u32", 5),
        ("activation_stages", "pairs.f32", "probe.cubin.u32", 8),
    ):
        x = np.fromfile(capture / "package" / inputs, np.float32).reshape(-1, 2)
        want = np.fromfile(capture / "h100" / expected, np.uint32).reshape(-1, columns)
        actual = np.empty_like(want)
        getattr(lib, name)(len(x), x.ctypes.data, actual.ctypes.data)
        report[name] = dict(inputs=len(x), different_per_column=np.count_nonzero(actual != want, axis=0).tolist())
        assert np.array_equal(actual, want), report[name]
    # Independent author-published H100 inputs, C accumulators and FP32 outputs.
    def read(name, base):
        return np.array([int(x, base) for x in (literature / name).read_text().split()], np.uint32)
    a = (read("a_H100_bf16.txt", 16) >> 16).astype(np.uint16)
    b = (read("b_H100_bf16.txt", 16) >> 16).astype(np.uint16)
    c = read("c_H100_fp32.txt", 2).view(np.float32)
    want = read("d_H100_fp32.txt", 2)
    actual = np.empty_like(c)
    lib.blocks(len(c), a.ctypes.data, b.ctypes.data, c.ctypes.data, actual.ctypes.data)
    report["published_h100_bf16"] = dict(inputs=len(c), different=int(np.count_nonzero(actual.view(np.uint32) != want)))
    assert np.array_equal(actual.view(np.uint32), want), report["published_h100_bf16"]
    meta = json.loads((capture / "experiment.json").read_text())
    report["cases"] = {}
    for case in meta["cases"]:
        start = time.monotonic()
        name, width, h = case["name"], case["width"], case["hidden"]
        rows = case["rows"]
        fc1 = expand(np.fromfile(capture / "package" / f"{name}.fc1.bf16", np.uint16)).reshape(len(rows), 2 * width)
        pairs = np.stack([fc1[:, :width].ravel(), fc1[:, width:].ravel()], axis=1)
        stages = np.empty((len(pairs), 8), np.uint32)
        lib.activation_stages(len(pairs), pairs.ctypes.data, stages.ctypes.data)
        act = expand(stages[:, 6]).reshape(len(rows), width)
        expected_act = np.fromfile(capture / "h100" / f"{name}.cubin.cpuinput.activation.bf16", np.uint16)
        assert np.array_equal(stages[:, 6], expected_act)
        actual = np.empty((len(rows), h), np.float32)
        raw = np.empty_like(actual)
        def expert(e):
            indices = [i for i, (_, selected) in enumerate(rows) if selected == e]
            for n in range(0, h, 256):
                count = min(256, h - n)
                weight = recipe(np.arange(e * h * width + n * width,
                                          e * h * width + (n + count) * width, dtype=np.uint64),
                                2029, np.float32(1 / 256)).reshape(count, width)
                for i in indices:
                    lib.dot(count, width, weight.ctypes.data, act[i].ctypes.data,
                            actual[i, n:].ctypes.data, raw[i, n:].ctypes.data)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(expert, sorted({e for _, e in rows})))
        hardware = expand(np.fromfile(capture / "h100" / f"{name}.cubin.fc2-nativeactivation.bf16", np.uint16)).reshape(actual.shape)
        differences = int(np.count_nonzero(actual != hardware))
        actual.tofile(run / f"{name}.fc2.f32")
        # Replay the unchanged simulator's separate FP32 multiply and add in
        # expert order, retaining its original routing weights and tolerance.
        row_index = {tuple(row): i for i, row in enumerate(rows)}
        outputs = []
        for ti, token in enumerate(case["tokens"]):
            value = np.zeros(h, np.float32)
            for e in sorted(case["chosen"][ti]):
                j = case["chosen"][ti].index(e)
                value = value + actual[row_index[token, e]] * np.float32(case["weights"][ti][j])
            outputs.append(rounded(value))
        outputs = np.stack(outputs)
        golden = expand(np.fromfile(ROOT / "workloads/moe/artifacts/cases" / f"{name}-t128/reference.bf16", np.uint16)).reshape(128, h)[case["tokens"]]
        failed = int(np.count_nonzero(np.abs(outputs - golden) > 1e-5 + .02 * np.abs(golden)))
        outputs.tofile(run / f"{name}.combined.f32")
        report["cases"][name] = dict(activation_elements=len(pairs), activation_different=0,
                                     fc2_elements=actual.size, fc2_different=differences,
                                     combined_elements=outputs.size, original_failed=len(case["failures"]),
                                     combined_failed=failed, max_abs=float(np.abs(outputs-golden).max()),
                                     seconds=time.monotonic()-start)
        print(name, json.dumps(report["cases"][name]), flush=True)
        (run / "offline-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        assert differences == 0 and failed == 0, report["cases"][name]
    report["scope"] = "Production arithmetic, all captured expert outputs and affected tokens; offline replay, not a complete simulator forward."
    (run / "offline-comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
