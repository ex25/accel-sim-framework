#!/usr/bin/env python3
"""Analyze downloaded H100 evidence without allocating a GPU."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
DIAG = ROOT / "workloads/moe/runs/hopper-stage-7/numerical-diagnosis-20261110"
sys.path.insert(0, str(HERE))
from prepare import recipe


def expand(x):
    return (np.asarray(x, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def rounded(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    result, package = run / "h100", run / "package"
    status = json.loads((result / "run.json").read_text())
    for name, digest in status["outputs_sha256"].items():
        assert hashlib.sha256((result / name).read_bytes()).hexdigest() == digest
    meta = json.loads((run / "experiment.json").read_text())
    lib = ctypes.CDLL(str(DIAG / "math_probe.so"))
    ptr = ctypes.POINTER(ctypes.c_float)
    lib.dot.argtypes = [ctypes.c_uint, ctypes.c_uint, ptr, ptr, ptr]
    lib.madd.argtypes = [ctypes.c_float] * 3 + [ctypes.c_int]
    lib.madd.restype = ctypes.c_float
    cp = lambda x: x.ctypes.data_as(ptr)
    report = dict(device=status["device"], remote_status=status["returncode"],
                  remote_seconds=status["execution_and_validation_seconds"], cases={})
    modes = ("jit", "cubin")
    for c in meta["cases"]:
        name, width, h = c["name"], c["width"], c["hidden"]
        cpu_fc1 = np.fromfile(package / f"{name}.fc1.bf16", dtype=np.uint16)
        cpu_act = np.fromfile(package / f"{name}.activation.bf16", dtype=np.uint16)
        a800 = np.fromfile(run / "a800-preflight" / f"{name}.a800.activation.bf16", dtype=np.uint16)
        a800_stages = np.fromfile(run / "a800-preflight" / f"{name}.a800.stages.u32", dtype=np.uint32).reshape(-1, 8)
        gold = expand(np.fromfile(ROOT / "workloads/moe/artifacts/cases" / (name+"-t128") / "reference.bf16", dtype=np.uint16)).reshape(128, h)
        row_index = {tuple(row): i for i, row in enumerate(c["rows"])}
        mode_reports = {}
        for mode in modes:
            prefix = result / f"{name}.{mode}"
            stages = np.fromfile(str(prefix)+".cpuinput.stages.u32", dtype=np.uint32).reshape(-1, 8)
            act = np.fromfile(str(prefix)+".cpuinput.activation.bf16", dtype=np.uint16)
            weights = np.fromfile(str(prefix)+".cpuinput.weights.f32", dtype=np.float32).reshape(128, c["experts"])
            assert np.array_equal(act, stages[:, 6])
            native_fc1_file = Path(str(prefix)+".fc1.bf16")
            m = dict(activation_vs_cpu=int(np.count_nonzero(act != cpu_act)),
                     activation_vs_a800=int(np.count_nonzero(act != a800)),
                     stage_vs_a800_differences=[int(np.count_nonzero(stages[:, i] != a800_stages[:, i])) for i in range(8)])
            if native_fc1_file.exists():
                native_fc1 = np.fromfile(native_fc1_file, dtype=np.uint16)
                m["fc1_vs_cpu_differences"] = int(np.count_nonzero(native_fc1 != cpu_fc1))
            # Independent CPU FC2 with H100-measured activation: isolates math
            # from the H100 matrix accumulation, for all original failures.
            actf = expand(act).reshape(len(c["rows"]), width)
            contributions = {}
            for i, (token, expert) in enumerate(c["rows"]):
                channels = sorted({f["channel"] for f in c["failures"] if f["token"] == token})
                w = np.stack([recipe(np.arange(expert*h*width+n*width, expert*h*width+(n+1)*width, dtype=np.uint64), 2029, np.float32(1/256)) for n in channels])
                x = actf[i].copy()
                out = np.empty(len(channels), dtype=np.float32)
                lib.dot(len(channels), width, cp(w), cp(x), cp(out))
                for n, value in zip(channels, out):
                    contributions[token, expert, n] = float(value)
            failed_rows = []
            for failure in c["failures"]:
                t, n = failure["token"], failure["channel"]
                index = c["tokens"].index(t)
                value = 0.
                for e in sorted(c["chosen"][index]):
                    j = c["chosen"][index].index(e)
                    value = lib.madd(contributions[t, e, n], c["weights"][index][j], value, 0)
                value = float(rounded([value])[0])
                ref = failure["hardware"]
                failed_rows.append(dict(token=t, channel=n, simulator=failure["simulator"], golden=ref,
                                        h100_activation_cpu_replay=value,
                                        still_fails=abs(value-ref) > 1e-5+.02*abs(ref)))
            m["activation_cpu_replay_remaining_failures"] = sum(r["still_fails"] for r in failed_rows)
            m["failed_coordinate_replay"] = failed_rows
            # Inspect all outputs of affected tokens, not just old failures.
            for variant in ("cpuactivation", "nativeactivation"):
                f = Path(str(prefix)+f".fc2-{variant}.bf16")
                if not f.exists():
                    continue
                fc2 = expand(np.fromfile(f, dtype=np.uint16)).reshape(len(c["rows"]), h)
                for native_weights in (False, True):
                    outputs = []
                    for ti, token in enumerate(c["tokens"]):
                        out = np.zeros(h, dtype=np.float32)
                        for e in sorted(c["chosen"][ti]):
                            j = c["chosen"][ti].index(e)
                            w = weights[token, e] if native_weights else c["weights"][ti][j]
                            y = fc2[row_index[token, e]]
                            for n in range(h):
                                out[n] = lib.madd(y[n], w, out[n], 1)
                        outputs.append(rounded(out))
                    outputs = np.stack(outputs)
                    reference = gold[c["tokens"]]
                    diff = np.abs(outputs-reference)
                    label = variant + ("_nativeweights" if native_weights else "_cpuweights")
                    np.save(result / f"{name}.{mode}.{label}.npy", outputs)
                    m[label] = dict(elements=outputs.size, different=int(np.count_nonzero(outputs!=reference)),
                                    failed=int(np.count_nonzero(diff>1e-5+.02*np.abs(reference))), max_abs=float(diff.max()))
            mode_reports[mode] = m
        report["cases"][name] = mode_reports
    report["scope"] = "Isolated kernels for previously failing tokens; CPU combines captured FC2 and routing weights. Not a complete forward acceptance or simulator repair."
    (run / "analysis.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({**report, "cases": {case: {mode: {k:v for k,v in row.items() if k!='failed_coordinate_replay'} for mode,row in modes.items()} for case,modes in report['cases'].items()}}, indent=2))


if __name__ == "__main__":
    main()
