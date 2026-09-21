#!/usr/bin/env python3
"""Check functional and timing equivalence of add.u32 and cais.add.u32."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Metrics:
    cycles: int
    instructions: int
    ipc: float
    checksum: int
    max_error: int
    status: str


def parse_log(path: Path) -> Metrics:
    text = path.read_text()
    if "GPGPU-Sim PTX: using external PTX file" not in text:
        raise SystemExit(f"{path}: external PTX override marker is missing")

    def last(pattern: str) -> str:
        matches = re.findall(pattern, text)
        if not matches:
            raise SystemExit(f"{path}: missing pattern {pattern!r}")
        return matches[-1]

    return Metrics(
        cycles=int(last(r"gpu_tot_sim_cycle\s*=\s*(\d+)")),
        instructions=int(last(r"gpu_tot_sim_insn\s*=\s*(\d+)")),
        ipc=float(last(r"gpu_tot_ipc\s*=\s*([0-9.]+)")),
        checksum=int(last(r"checksum=(\d+)")),
        max_error=int(last(r"max_error=(\d+)")),
        status=last(r"status=(PASS|FAIL)"),
    )


def compare_pair(control_path: Path, custom_path: Path) -> Metrics:
    control = parse_log(control_path)
    custom = parse_log(custom_path)
    if control.status != "PASS" or custom.status != "PASS":
        raise SystemExit(f"functional failure: control={control} custom={custom}")
    if control.max_error != 0 or custom.max_error != 0:
        raise SystemExit(f"non-zero error: control={control} custom={custom}")
    if control.checksum != custom.checksum:
        raise SystemExit(f"checksum mismatch: control={control} custom={custom}")
    if control.cycles != custom.cycles or control.instructions != custom.instructions:
        raise SystemExit(f"timing/count mismatch: control={control} custom={custom}")
    if not math.isclose(control.ipc, custom.ipc, rel_tol=0.0, abs_tol=1e-6):
        raise SystemExit(f"IPC mismatch: control={control} custom={custom}")
    return custom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control_iter1", type=Path)
    parser.add_argument("custom_iter1", type=Path)
    parser.add_argument("control_iter32", type=Path)
    parser.add_argument("custom_iter32", type=Path)
    args = parser.parse_args()

    iter1 = compare_pair(args.control_iter1, args.custom_iter1)
    iter32 = compare_pair(args.control_iter32, args.custom_iter32)
    if iter32.cycles <= iter1.cycles or iter32.instructions <= iter1.instructions:
        raise SystemExit(f"dependent chain did not scale: iter1={iter1} iter32={iter32}")
    print(
        "CAIS_ADD_TEST_PASS "
        f"iter1_cycles={iter1.cycles} iter32_cycles={iter32.cycles} "
        f"iter1_insn={iter1.instructions} iter32_insn={iter32.instructions}"
    )


if __name__ == "__main__":
    main()
