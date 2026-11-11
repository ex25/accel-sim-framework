#!/usr/bin/env python3
"""Prepare and compile one offline H100 numerical investigation bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
ART = ROOT / "workloads/moe/artifacts"
DIAG = ROOT / "workloads/moe/runs/hopper-stage-7/numerical-diagnosis-20261110"
sys.path.insert(0, str(HERE.parent.parent / "stage-2"))
from read_forward import load


def recipe(indices, seed, scale):
    def mix(x):
        x = np.asarray(x, dtype=np.uint32).copy()
        x ^= x >> 16
        x *= np.uint32(0x7feb352d)
        x ^= x >> 15
        x *= np.uint32(0x846ca68b)
        return x ^ (x >> 16)
    key = indices.astype(np.uint32) ^ mix((indices >> 32).astype(np.uint32))
    return ((mix(key ^ np.uint32(seed)) % 17).astype(np.int32) - 8).astype(np.float32) * scale


def bf16(x):
    return (np.ascontiguousarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def maps_in(path):
    maps = []
    with path.open("rb") as f:
        def rd(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
        def blob():
            return f.read(rd("I")[0])
        assert rd("II") == (0x4d4f4550, 3)
        while f.peek(1):
            tag = rd("I")[0]
            if tag == 1:
                rd("QQ")
            elif tag == 2:
                blob(); rd("Q10I")
                for _ in range(rd("I")[0]):
                    blob()
            elif tag in (3, 10):
                rd("Q"); blob()
                if tag == 3:
                    f.read(rd("I")[0] * 4)
            elif tag in (4, 8):
                rd("QQ")
            elif tag == 5:
                raw = f.read(128)
                _, dtype, rank = rd("QII")
                assert rank == 2
                dims, stride = rd("2Q"), rd("Q")[0]
                box, element, flags = rd("2I"), rd("2I"), rd("4I")
                maps.append((raw, struct.pack("<II3Q8I", dtype, rank, *dims, stride, *box, *element, *flags)))
            elif tag == 6:
                rd("QIIQ")
            elif tag == 7:
                rd("QQIfQ")
            elif tag == 9:
                rd("QQQ")
            else:
                raise ValueError(tag)
    return maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    package = run / "package"
    package.mkdir()
    manifest = json.loads((ART / "workloads.json").read_text())
    cases, pairs, denominators = [], [], []
    for case, act_module, normalize, exp_module in [
        ("qwen64", 82, 75, 72), ("phi", 52, 44, -1), ("mixtral", 17, 7, -1)
    ]:
        params = next(c["framework"]["parameters"] for c in manifest["cases"] if c["name"] == case + "-t128")
        h, width, experts, topk = [params[k] for k in ("hidden", "intermediate", "experts", "top_k")]
        source = np.load(DIAG / f"{case}-failed-coordinates.npz")
        def row_key(key):
            return tuple(map(int, re.fullmatch(r"fc1_t(\d+)_e(\d+)", key).groups()))
        keys = sorted((k for k in source if k.startswith("fc1_t")), key=lambda k: row_key(k)[::-1])
        rows = [row_key(k) for k in keys]
        fc1 = np.stack([source[k] for k in keys])
        activation = np.stack([source[k.replace("fc1_", "activation_")] for k in keys])
        bf16(fc1).tofile(package / f"{case}.fc1.bf16")
        bf16(activation).tofile(package / f"{case}.activation.bf16")
        np.array(rows, dtype=np.int32).tofile(package / f"{case}.rows.i32")
        gate, up = fc1[:, :width].ravel(), fc1[:, width:].ravel()
        pairs.append(np.stack([gate, up], axis=1))
        x = -gate * np.float32(float.fromhex("0x1.715476p+0"))
        denominators.append(np.unique(np.float32(1) + np.exp2(x)))
        tokens = source["tokens"].tolist()
        chosen, weights = source["chosen"], source["weights"]
        inputs = recipe(np.arange(128*h, dtype=np.uint64), 2026, np.float32(1/16)).reshape(128, h)
        router = recipe(np.arange(experts*h, dtype=np.uint64), 2027, np.float32(1/128)).reshape(experts, h)
        scores = inputs @ router.T
        selected = np.argsort(-scores, axis=1, kind="stable")[:, :topk]
        assert np.array_equal(selected[tokens], chosen)
        selected.astype(np.int64).tofile(package / f"{case}.selected.i64")
        np.take_along_axis(scores, selected, axis=1).tofile(package / f"{case}.scores.f32")
        forward = ART / "cases" / (case + "-t128") / "forward.bin"
        calls = load(forward)
        gemms = [c for c in calls if c["module"] == 16]
        for i, call in enumerate(gemms, 1):
            raw = bytes.fromhex(call["args"][0])
            (package / f"{case}.fc{i}.param").write_bytes(raw)
            desc = bytearray()
            for offset in (64, 256, 640, 832):
                desc += next(encoded for original, encoded in maps_in(forward) if original == raw[offset:offset+128])
            (package / f"{case}.fc{i}.maps").write_bytes(desc)
        cases.append(dict(name=case, hidden=h, width=width, experts=experts, topk=topk,
                          act_module=act_module, normalize=normalize, exp_module=exp_module,
                          rows=rows, tokens=tokens, chosen=chosen.tolist(), weights=weights.tolist(),
                          failures=json.loads((DIAG / f"{case}-failed-coordinates.json").read_text())["rows"],
                          normalize_shape=next(c["shape"] for c in calls if c["module"] == normalize),
                          exp_shape=next((c["shape"] for c in calls if c["module"] == exp_module), [])))
    # Every actual gate/up pair, plus all BF16 gates in [-16,16] at up=1.
    gates = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
    gates = gates[np.isfinite(gates) & (np.abs(gates) <= 16)]
    pairs.append(np.stack([gates, np.ones_like(gates)], axis=1))
    all_pairs, inverse = np.unique(np.concatenate(pairs).view(np.uint64), return_inverse=True)
    all_pairs = all_pairs.view(np.float32).reshape(-1, 2)
    actual_pair_count = sum(p.shape[0] for p in pairs[:-1])
    np.savez(run / "probe-mapping.npz", inverse=inverse[:actual_pair_count], pairs=all_pairs)
    # Probe uses whole warps; preserve the unpadded length in metadata.
    padded = np.pad(all_pairs, ((0, -len(all_pairs) % 32), (0, 0)), mode="edge")
    padded.tofile(package / "pairs.f32")
    den = np.unique(np.concatenate(denominators)).view(np.uint32)
    # Actual denominators and their neighbours, and a mantissa sweep, permit
    # checking a general reciprocal model offline instead of token patches.
    den = np.unique(np.concatenate([den + np.uint32(i) if i >= 0 else den - np.uint32(-i) for i in range(-4, 5)] + [np.arange(0x3f800000, 0x40000000, 256, dtype=np.uint32)]))
    op_input = np.stack([np.zeros(len(den), dtype=np.float32), den.view(np.float32)], axis=1)
    # x for ex2 is also sampled at every real gate value.
    exp_x = np.unique(-all_pairs[:, 0] * np.float32(float.fromhex("0x1.715476p+0")))
    op_input = np.concatenate([op_input, np.stack([exp_x, np.ones_like(exp_x)], axis=1)])
    op_input = np.pad(op_input, ((0, -len(op_input) % 32), (0, 0)), mode="edge")
    op_input.tofile(package / "ops.f32")
    probe = (DIAG / "native-a800-batch/probe.ptx").read_text().replace(".target sm_80", ".target sm_90a")
    (package / "probe.ptx").write_text(probe)
    ops = probe[:probe.index(".visible .entry")] + '''.visible .entry ops_probe(.param .u64 input, .param .u64 output) {
  .reg .b32 %r<4>;
  .reg .b64 %rd<7>;
  .reg .f32 %f<8>;
  ld.param.u64 %rd0, [input];
  ld.param.u64 %rd1, [output];
  mov.u32 %r0, %tid.x;
  mov.u32 %r2, %ctaid.x;
  mad.lo.u32 %r0, %r2, 32, %r0;
  mul.wide.u32 %rd2, %r0, 8;
  add.u64 %rd3, %rd0, %rd2;
  ld.global.f32 %f0, [%rd3];
  ld.global.f32 %f1, [%rd3+4];
  ex2.approx.f32 %f2, %f0;
  rcp.approx.f32 %f3, %f1;
  div.full.f32 %f4, 0f3f800000, %f1;
  div.rn.f32 %f5, 0f3f800000, %f1;
  div.approx.f32 %f6, 0f3f800000, %f1;
  mul.wide.u32 %rd4, %r0, 20;
  add.u64 %rd5, %rd1, %rd4;
  st.global.f32 [%rd5], %f2;
  st.global.f32 [%rd5+4], %f3;
  st.global.f32 [%rd5+8], %f4;
  st.global.f32 [%rd5+12], %f5;
  st.global.f32 [%rd5+16], %f6;
  ret;
}
'''
    (package / "ops.ptx").write_text(ops)
    cuda = Path("/usr/local/cuda")
    commands = []
    def call(command):
        command = list(map(str, command))
        commands.append(command)
        subprocess.run(command, check=True, cwd=package)
    modules = {str(i): (ART / manifest["modules"][i]["ptx"], manifest["modules"][i]["entry"])
               for i in (16, 17, 52, 82, 7, 44, 72, 75)}
    modules.update(probe=(package / "probe.ptx", "activation_probe"), ops=(package / "ops.ptx", "ops_probe"))
    asm, inc = [], []
    for key, (ptx, entry) in modules.items():
        target = package / f"{key}.ptx"
        if ptx != target:
            shutil.copy2(ptx, target)
        for mode in ("jit", "cubin", "a800"):
            if mode == "a800" and key == "16":
                continue
            arch = "80" if mode == "a800" else "90a"
            source = target
            if mode == "a800":
                source = package / f"{key}.sm80.ptx"
                source.write_text(target.read_text().replace(".target sm_90a", ".target sm_80"))
            symbol = f"module_{key}_{mode}"
            fatbin = package / f"{key}.{mode}.fatbin"
            if mode != "jit":
                cubin = package / f"{key}.{mode}.cubin"
                call([cuda/"bin/ptxas", f"-arch=sm_{arch}", source, "-o", cubin])
                call([cuda/"bin/fatbinary", "--64", "--create="+str(fatbin), f"--image=profile=sm_{arch},file={cubin}"])
                sass = subprocess.check_output([str(cuda/"bin/cuobjdump"), "-sass", str(cubin)], text=True)
                (run / f"{key}.{mode}.sass").write_text(sass)
            else:
                call([cuda/"bin/fatbinary", "--64", "--create="+str(fatbin), f"--image=profile=compute_{arch},file={source}"])
            asm.append(f'.section .nv_fatbin,"a"\n.balign 8\n.global {symbol}\n{symbol}:\n.incbin "{fatbin}"\n')
            inc.append(f'extern "C" const unsigned char {symbol}[];\n')
            inc.append(f'REGISTER("{key}.{mode}", "{entry}", {symbol});\n')
    # Declarations and registration statements occupy separate scopes.
    (package / "modules.inc").write_text("".join(x for x in inc if x.startswith("extern")))
    (package / "register.inc").write_text("".join(x for x in inc if not x.startswith("extern")))
    (package / "embed.S").write_text("".join(asm)+'.section .note.GNU-stack,"",@progbits\n')
    (package / "cases.inc").write_text("static const Case cases[] = {\n" + ",\n".join(
        '{"%s",%d,%d,%d,%d,%d,%d,%d,%d,{%s},{%s}}' % (
            c["name"],c["hidden"],c["width"],c["experts"],c["topk"],len(c["rows"]),c["act_module"],c["normalize"],c["exp_module"],
            ",".join(map(str,c["normalize_shape"])), ",".join(map(str,c["exp_shape"] or [0]*10))) for c in cases)+"\n};\n")
    shutil.copy2(HERE / "batch.cu", package / "batch.cu")
    call(["gcc", "-c", "embed.S", "-o", "embed.o"])
    call([cuda/"bin/nvcc", "-std=c++17", "-O2", "-ccbin", "/usr/bin/g++", "-gencode=arch=compute_80,code=sm_80", "-gencode=arch=compute_90a,code=sm_90a", "-Xcompiler=-static-libstdc++,-static-libgcc", "batch.cu", "embed.o", "-o", "batch"])
    record = dict(cases=cases, probe_count=len(padded), unique_probe_count=len(all_pairs), actual_pair_count=actual_pair_count,
                  ops_count=len(op_input), commands=commands,
                  scope="Only failing-token expert groups and isolated math/routing kernels; no whole MoE forward or simulator run.",
                  budget="One H100 function, no retries, no deployment; 60s subprocess cap, 120s function timeout.")
    (run / "experiment.json").write_text(json.dumps(record, indent=2)+"\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.iterdir() if p.is_file()}
    (package / "sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    print(json.dumps(dict(package=str(package), bytes=sum(p.stat().st_size for p in package.iterdir()), probe_count=len(padded), ops_count=len(op_input), rows={c['name']:len(c['rows']) for c in cases}), indent=2))


if __name__ == "__main__":
    main()
