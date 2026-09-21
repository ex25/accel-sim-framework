#!/usr/bin/env python3
"""Generate control and proposed-opcode PTX fixtures from the CUDA kernel."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    source = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = output_dir / "_generated.ptx"
    subprocess.run(
        ["nvcc", "-ptx", "-O3", "-std=c++17", "-arch=compute_90", str(source),
         "-o", str(generated)],
        check=True,
    )
    raw = generated.read_text()
    count = raw.count("add.u32")
    if count != 1:
        raise SystemExit(f"expected exactly one marker add.u32, found {count}")
    (output_dir / "control.ptx").write_text(raw)
    (output_dir / "custom.ptx").write_text(raw.replace("add.u32", "cais.add.u32"))
    generated.unlink()
    print(f"PTX_FIXTURE_PASS marker_count={count} output={output_dir}")


if __name__ == "__main__":
    main()
