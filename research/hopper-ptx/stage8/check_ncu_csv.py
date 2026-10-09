#!/usr/bin/env python3
"""Validate that an Nsight Compute CSV contains finite requested counters."""
import csv
import json
import math
import re
import sys
from pathlib import Path

REQUIRED = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__registers_per_thread",
]
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    csv_path, json_path = map(Path, sys.argv[1:])
    result = {"status": "BLOCKED", "scope": "hardware_counters", "csv": str(csv_path),
              "required_metrics": REQUIRED, "metrics": {}, "errors": []}
    if not csv_path.is_file():
        result["errors"].append("missing_csv")
    else:
        rows = list(csv.reader(csv_path.open(newline="", errors="replace")))
        permission_error = False
        for row in rows:
            joined = " ".join(row)
            if "ERR_NVGPUCTRPERM" in joined:
                permission_error = True
                result["errors"].append("ERR_NVGPUCTRPERM")
            elif "==ERROR==" in joined:
                result["errors"].append("ncu_error")
        result["permission_error"] = permission_error
        header = next((row for row in rows if "Metric Name" in row and "Metric Value" in row), None)
        values = {metric: [] for metric in REQUIRED}
        if header:
            name_index, value_index = header.index("Metric Name"), header.index("Metric Value")
            for row in rows:
                if len(row) <= max(name_index, value_index) or row[name_index] not in values:
                    continue
                raw = row[value_index].strip().replace(",", "")
                value = float(raw) if NUMBER.fullmatch(raw) else float("nan")
                values[row[name_index]].append(value)
        for metric in REQUIRED:
            numbers = values[metric]
            if not numbers or any(not math.isfinite(v) or v < 0 for v in numbers):
                result["errors"].append(f"missing_or_nonfinite:{metric}")
            else:
                result["metrics"][metric] = numbers
        if not result["errors"]:
            result["status"] = "PASS"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PROFILER_RESULT status=" + result["status"] +
          " metrics=" + str(len(result["metrics"])) +
          " errors=" + str(len(result["errors"])))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
