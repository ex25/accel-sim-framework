#!/usr/bin/env python3
"""Check byte/flit accounting and monotonic trends in a BookSim sweep.

This is deliberately an analytical sanity check.  It does not calibrate
NVLink, infer a physical clock, or claim that BookSim latency is hardware
latency.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path, flit_bytes: int, link_flits_per_cycle: float) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"link_latency", "packet_size", "latency_cycles", "accepted_rate", "sim_cycles"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
    result = []
    for row in rows:
        packet_flits = int(row["packet_size"])
        if packet_flits <= 0:
            raise ValueError("packet_size must be positive")
        latency = float(row["latency_cycles"])
        if latency <= 0:
            raise ValueError("latency_cycles must be positive")
        result.append(
            {
                "link_latency": float(row["link_latency"]),
                "packet_flits": float(packet_flits),
                "message_bytes": float(packet_flits * flit_bytes),
                "serialization_cycles": float(math.ceil(packet_flits / link_flits_per_cycle)),
                "latency_cycles": latency,
                "accepted_rate": float(row["accepted_rate"]),
                "sim_cycles": float(row["sim_cycles"]),
            }
        )
    return result


def check(rows: list[dict[str, float]]) -> list[str]:
    errors: list[str] = []
    by_link: dict[float, list[dict[str, float]]] = defaultdict(list)
    by_packet: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        # A packet cannot complete in fewer cycles than the number of flits it
        # must serialize on a one-flit-per-cycle link.
        if row["latency_cycles"] + 1e-9 < row["serialization_cycles"]:
            errors.append(
                f"latency {row['latency_cycles']} < serialization lower bound "
                f"{row['serialization_cycles']} for {row}"
            )
        by_link[row["link_latency"]].append(row)
        by_packet[row["packet_flits"]].append(row)

    for link, group in by_link.items():
        group.sort(key=lambda row: row["packet_flits"])
        for previous, current in zip(group, group[1:]):
            if current["latency_cycles"] + 1e-9 < previous["latency_cycles"]:
                errors.append(f"latency is not monotonic with packet size at link_latency={link}")
    for packet, group in by_packet.items():
        group.sort(key=lambda row: row["link_latency"])
        for previous, current in zip(group, group[1:]):
            if current["latency_cycles"] + 1e-9 < previous["latency_cycles"]:
                errors.append(f"latency is not monotonic with link latency at packet_size={packet}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--flit-bytes", type=int, default=32)
    parser.add_argument("--link-flits-per-cycle", type=float, default=1.0)
    args = parser.parse_args()
    if args.flit_bytes <= 0 or args.link_flits_per_cycle <= 0:
        parser.error("flit and link rates must be positive")
    try:
        rows = load_rows(args.csv, args.flit_bytes, args.link_flits_per_cycle)
    except (OSError, ValueError) as exc:
        print(f"ANALYTICAL_CHECK_FAIL: {exc}", file=sys.stderr)
        return 2
    errors = check(rows)
    for row in rows:
        effective = row["message_bytes"] / row["latency_cycles"]
        print(
            "link_latency={:.6g} packet_flits={} message_bytes={} "
            "serialization_cycles={} latency_cycles={:.6g} effective_bytes_per_cycle={:.6g}".format(
                row["link_latency"],
                int(row["packet_flits"]),
                int(row["message_bytes"]),
                int(row["serialization_cycles"]),
                row["latency_cycles"],
                effective,
            )
        )
    if errors:
        for error in errors:
            print(f"ANALYTICAL_CHECK_FAIL: {error}", file=sys.stderr)
        return 1
    print(
        f"ANALYTICAL_CHECK_PASS rows={len(rows)} flit_bytes={args.flit_bytes} "
        f"link_flits_per_cycle={args.link_flits_per_cycle}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
