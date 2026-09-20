#!/usr/bin/env python3
"""Collect comparable metrics from the A800 SASS/PTX gap experiment.

The simulator emits one summary block per run.  This collector deliberately
keeps simulator counters separate from the native CUDA event time: PTX replay
uses simulated CUDA timing, while the native run uses the real A800.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CASES = [
    "fma_chain_1024_128",
    "vec_add_4096_0",
    "global_stride_4096_16",
    "shared_reduce_256_0",
    "branch_diverge_1024_64",
    "atomic_hotspot_1024_1",
    "gemv_64_0",
    "gemv_128_0",
    "gemv_256_0",
    "gemm_16_0",
    "gemm_32_0",
    "gemm_64_0",
]


def first_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no {pattern} below {root}")
    return matches[0]


def number(text: str, key: str) -> float | int | None:
    match = re.search(rf"^{re.escape(key)}\s*=\s*([0-9.eE+-]+)", text, re.MULTILINE)
    if not match:
        return None
    value = float(match.group(1))
    return int(value) if value.is_integer() else value


def result_fields(text: str) -> dict[str, str]:
    match = re.findall(r"^RESULT (.+)$", text, re.MULTILINE)
    if not match:
        return {}
    return dict(item.split("=", 1) for item in match[-1].split() if "=" in item)


def cache_access(text: str, kind: str) -> int | None:
    pattern = rf"Total_core_cache_stats_breakdown\[GLOBAL_ACC_{kind}\]\[TOTAL_ACCESS\] = (\d+)"
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def sim_metrics(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    result = result_fields(text)
    return {
        "sim_log": str(path),
        "sim_cycle": number(text, "gpu_tot_sim_cycle"),
        "sim_insn": number(text, "gpu_tot_sim_insn"),
        "sim_ipc": number(text, "gpu_tot_ipc"),
        "sim_cta": number(text, "gpu_tot_issued_cta"),
        "global_read": cache_access(text, "R"),
        "global_write": cache_access(text, "W"),
        "result_status": result.get("status"),
        "checksum": result.get("checksum"),
        "max_error": result.get("max_error"),
    }


def trace_metrics(trace_dir: Path) -> dict[str, object]:
    stats = first_file(trace_dir, "stats_ctx_*").read_text(errors="replace")
    rows = [line for line in stats.splitlines() if line.startswith("kernel-")]
    if not rows:
        return {"trace_insts": None, "trace_reported_insts": None, "trace_blocks": None}
    fields = [field.strip() for field in rows[-1].split(",")]
    return {
        "trace_insts": int(fields[-2]),
        "trace_reported_insts": int(fields[-1]),
        "trace_blocks": int(fields[5]),
    }


def pct(ptx: object, sass: object) -> object:
    if not isinstance(ptx, (int, float)) or not isinstance(sass, (int, float)) or sass == 0:
        return None
    return 100.0 * (ptx - sass) / sass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sass-root", type=Path, required=True)
    parser.add_argument("--ptx-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for case in CASES:
        sass_log = first_file(args.sass_root / "sass_ptx_gap" / case / "A100-SASS", "gpgpu-sim-out_*.txt")
        ptx_log = args.ptx_root / "sass_ptx_gap" / case / "A100-PTX" / "direct-run.log"
        trace_dir = args.trace_root / case / "traces"
        sass = sim_metrics(sass_log)
        ptx = sim_metrics(ptx_log)
        trace = trace_metrics(trace_dir)
        row: dict[str, object] = {"case": case, **trace}
        for key, value in sass.items():
            if key != "sim_log":
                row[f"sass_{key}"] = value
        for key, value in ptx.items():
            if key != "sim_log":
                row[f"ptx_{key}"] = value
        for key in ("sim_cycle", "sim_insn", "sim_ipc", "global_read", "global_write"):
            row[f"{key}_delta_pct"] = pct(row.get(f"ptx_{key}"), row.get(f"sass_{key}"))
        rows.append(row)

    fieldnames = list(rows[0])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row.get(field, "")) for field in fieldnames))


if __name__ == "__main__":
    main()
