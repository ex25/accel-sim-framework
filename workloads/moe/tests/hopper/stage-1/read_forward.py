import struct, json
from pathlib import Path

p = Path(__file__).resolve().parents[3] / "artifacts/cases/qwen64-t128/forward.bin"
f = p.open("rb")


def rd(fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def blob():
    return f.read(rd("I")[0])


assert rd("II") == (0x4D4F4550, 3)
launches = []
alloc = []
while f.peek(1):
    tag = rd("I")[0]
    if tag == 1:
        alloc.append(rd("QQ"))
    elif tag == 2:
        name = blob().decode()
        mod = rd("Q")[0]
        shape = rd("10I")
        args = [blob() for _ in range(rd("I")[0])]
        launches.append(
            dict(module=mod, name=name, shape=shape, args=[a.hex() for a in args])
        )
    elif tag in [3, 10]:
        rd("Q")
        blob()
        if tag == 3:
            f.read(rd("I")[0] * 4)
    elif tag == 4:
        rd("QQ")
    elif tag == 5:
        f.read(128)
        _, _, rank = rd("QII")
        f.read(rank * 8 + (rank - 1) * 8 + rank * 4 * 2 + 16)
    elif tag == 6:
        rd("QIIQ")
    elif tag == 7:
        rd("QQIfQ")
    elif tag == 8:
        rd("QQ")
    elif tag == 9:
        rd("QQQ")
    else:
        raise ValueError(tag)
f.close()
