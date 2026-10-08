#!/usr/bin/env python3
"""Write portable little-endian FP16 Q/K/V without an RNG/library dependency."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, length, offset in (("q", 128, 17), ("k", 128, 53), ("v", 128, 101)):
        # i indexes contiguous BSHD. Dyadic values are exact in binary16.
        data = b"".join(struct.pack("<e", ((i * 73 + offset) % 257 - 128) / 128)
                        for i in range(length * 8 * 128))
        filename = f"{name}.f16le"
        (output / filename).write_bytes(data)
        records[name] = {"file": filename, "shape": [1, length, 8, 128],
                         "dtype": "float16", "layout": "BSHD", "bytes": len(data),
                         "offset": offset, "sha256": hashlib.sha256(data).hexdigest()}
    result = {"formula": "((i * 73 + offset) % 257 - 128) / 128",
              "encoding": "IEEE754 binary16 little-endian, contiguous BSHD",
              "inputs": records}
    (output / "inputs.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(generate(parser.parse_args().output), indent=2))
