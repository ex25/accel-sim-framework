#!/usr/bin/env python3
"""Run the P1 PTX synchronization smoke test in an external output directory."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/booksim_nvlink"))
from communication_events import CommunicationEvent  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("output directory must be outside the repository")
    output.mkdir(parents=True, exist_ok=False)

    app = ROOT / "experiments/cais_add_ptx/build/cais_add_ptx"
    source_ptx = ROOT / "experiments/cais_add_ptx/build/control.ptx"
    config_dir = ROOT / "gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100"
    activation = ROOT.parent / "activate-accel-sim.sh"
    for path in (app, source_ptx, config_dir / "gpgpusim.config",
                 config_dir / "config_hopper_islip.icnt", activation):
        if not path.exists():
            raise FileNotFoundError(f"missing prerequisite: {path}")

    ptx = source_ptx.read_text()
    if "{\n" not in ptx:
        raise ValueError("expected kernel opening brace not found in control PTX")
    (output / "sync.ptx").write_text(
        ptx.replace("{\n", "{\n\tmembar.cta;\n\tbar.sync 0;\n", 1)
    )
    (output / "gpgpusim.config").write_bytes((config_dir / "gpgpusim.config").read_bytes())
    (output / "config_hopper_islip.icnt").write_bytes(
        (config_dir / "config_hopper_islip.icnt").read_bytes()
    )
    command = (
        f"source {shlex.quote(str(activation))} >/dev/null && "
        "PTX_SIM_USE_PTX_FILE=1 "
        f"PTX_SIM_KERNELFILE={shlex.quote(str(output / 'sync.ptx'))} "
        f"PTX_SIM_EVENT_FILE={shlex.quote(str(output / 'events.jsonl'))} "
        "PTX_SIM_RANK=0 "
        f"{shlex.quote(str(app))} 256 1"
    )
    result = subprocess.run(
        ["bash", "-c", command], cwd=output, capture_output=True, text=True,
        timeout=120, check=False,
    )
    (output / "run.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0 or "status=PASS" not in result.stdout:
        print(f"PTX_SYNC_SMOKE_FAIL exit={result.returncode} log={output / 'run.log'}")
        return 1

    events = [
        CommunicationEvent.from_dict(json.loads(line))
        for line in (output / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    counts = Counter(event.kind.value for event in events)
    expected = {
        "memory_release": 8,
        "memory_acquire": 8,
        "local_barrier_arrive": 8,
        "barrier_wakeup": 1,
    }
    if counts != expected or any(dict(event.metadata).get("time_domain") != "sim_cycle"
                                 for event in events):
        print(f"PTX_SYNC_SMOKE_FAIL unexpected events={dict(counts)}")
        return 1
    if [event.event_id for event in events] != [f"gpu0-sync-{i}" for i in range(25)]:
        print("PTX_SYNC_SMOKE_FAIL event IDs are not sequential")
        return 1
    summary = {"status": "PASS", "events": len(events), "counts": dict(counts),
               "time_domain": "sim_cycle", "output_dir": str(output)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PTX_SYNC_SMOKE_PASS events={len(events)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
