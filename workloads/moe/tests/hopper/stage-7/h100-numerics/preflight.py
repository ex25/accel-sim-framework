#!/usr/bin/env python3
"""Validate the already-run local A800 preflight before a paid invocation."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    package = run / "package"
    meta = json.loads((run / "experiment.json").read_text())
    previous = run.parent / "numerical-diagnosis-20261110/native-a800-batch"
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in package.iterdir() if p.is_file() and p.name != "sha256.json"}
    (package / "sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    report = dict(passed=True, cases={}, manifest_sha256=hashlib.sha256(
        (package / "sha256.json").read_bytes()).hexdigest(),
        scope="A800 original kernels, probes, routing and launch ABI. H100 WGMMA cannot execute on A800.")
    for c in meta["cases"]:
        name, n = c["name"], len(c["rows"]) * c["width"]
        out = run / "a800-preflight"
        a = np.fromfile(out / f"{name}.a800.activation.bf16", dtype=np.uint16)
        stages = np.fromfile(out / f"{name}.a800.stages.u32", dtype=np.uint32).reshape(n, 8)
        cpu = np.fromfile(package / f"{name}.activation.bf16", dtype=np.uint16)
        assert len(a) == n and np.array_equal(a, stages[:, 6])
        assert np.isfinite((a.astype(np.uint32) << 16).view(np.float32)).all()
        row = dict(values=n, original_vs_probe_bf16_differences=0,
                   vs_cpu_bf16_differences=int(np.count_nonzero(a != cpu)))
        if name != "qwen64":
            old = json.loads((previous / "inputs.json").read_text())[name]
            old_output = np.fromfile(previous / f"{name}-original.actual", dtype=np.uint16).reshape(-1, c["width"])
            index = {key: i for i, key in enumerate(old["keys"])}
            reordered = np.stack([old_output[index[f"fc1_t{t}_e{e}"]] for t, e in c["rows"]]).ravel()
            assert np.array_equal(a, reordered)
            row["vs_previous_valid_a800_differences"] = 0
        weights = np.fromfile(out / f"{name}.a800.weights.f32", dtype=np.float32).reshape(128, c["experts"])
        selected = np.fromfile(package / f"{name}.selected.i64", dtype=np.int64).reshape(128, c["topk"])
        mask = np.fromfile(out / f"{name}.a800.mask.i32", dtype=np.int32).reshape(weights.shape)
        expected = np.zeros_like(mask)
        np.put_along_axis(expected, selected, 1, axis=1)
        assert np.array_equal(mask, expected)
        assert np.isfinite(weights).all() and np.max(np.abs(weights.sum(axis=1)-1)) < 5e-7
        row["routing_mask_matches"] = True
        for fc in (1, 2):
            raw = (package / f"{name}.fc{fc}.param").read_bytes()
            assert len(raw) == 1344
            assert struct.unpack_from("<f", raw, 512)[0] == 0
            assert struct.unpack_from("<f", raw, 552)[0] == 1
            assert len((package / f"{name}.fc{fc}.maps").read_bytes()) == 256
        report["cases"][name] = row
    log = (run / "a800-preflight/batch.log").read_text()
    assert "COMPLETE" in log
    report["a800_log"] = log
    (run / "preflight.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
