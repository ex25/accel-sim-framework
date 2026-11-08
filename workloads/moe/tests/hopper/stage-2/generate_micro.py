"""Legal cp.async programs; expectations use coordinates, never simulator maps."""

import json
import struct


def generate(run):
    cases = {}
    # 2 CTAs, 8 source rounds, each lane has four tagged words.
    values = [
        100000 * c + 10000 * g + 100 * l + j
        for c in range(2)
        for g in range(8)
        for l in range(32)
        for j in range(4)
    ]
    (run / "input.bin").write_bytes(struct.pack("<" + "I" * len(values), *values))

    def make(name, body, expected, threads=32, grid=1):
        cases[name] = dict(threads=threads, grid=grid)
        want = [0xDEADBEEF] * (grid * threads * 64)
        for c in range(grid):
            for lane in range(threads):
                for slot, v in expected(c, lane).items():
                    want[(c * threads + lane) * 64 + slot] = v & 0xFFFFFFFF
        (run / (name + ".expected")).write_bytes(
            struct.pack("<" + "I" * len(want), *want)
        )
        entries.append(f"""
.visible .entry {name}(.param .u64 input, .param .u64 output,
                       .param .u32 threads) {{
.reg .b32 %r<32>;
.reg .b64 %rd<16>;
.reg .pred %p<8>;
.shared .align 16 .b8 buf[4096];
ld.param.u64 %rd0, [input];
ld.param.u64 %rd1, [output];
mov.u32 %r0, %tid.x;
mov.u32 %r1, %ctaid.x;
and.b32 %r2, %r0, 31;
mul.lo.u32 %r3, %r2, 16;
mov.u32 %r4, buf;
add.u32 %r4, %r4, %r3;
mad.lo.u32 %r5, %r1, 4096, %r3;
cvt.u64.u32 %rd2, %r5;
add.u64 %rd2, %rd0, %rd2;
mad.lo.u32 %r5, %r1, {threads}, %r0;
mul.wide.u32 %rd3, %r5, 256;
add.u64 %rd3, %rd1, %rd3;
mov.u32 %r6, 16;
{body}
ret;
}}
""")

    def copy(dst="%r4", src="%rd2", prefix=""):
        return f"{prefix}cp.async.cg.shared.global [{dst}], [{src}], 16, %r6;\n"

    def consume(dst="%r4", slot=0):
        return f"""ld.shared.v4.u32 {{%r20,%r21,%r22,%r23}}, [{dst}];
st.global.v4.u32 [%rd3+{slot*4}], {{%r20,%r21,%r22,%r23}};\n"""

    def pattern(c, l, g=0):
        return {j: 100000 * c + 10000 * g + 100 * l + j for j in range(4)}

    entries = []
    for name in ["copy_mix", "zero"]:
        body = "mov.u32 %r24, 0xa5a5a5a5;\nst.shared.v4.u32 [%r4], {%r24,%r24,%r24,%r24};\nbar.sync 0;\n"
        if name == "copy_mix":
            body += "setp.lt.u32 %p0, %r0, 24;\nsetp.lt.u32 %p1, %r0, 16;\nselp.u32 %r6, 16, 0, %p1;\n"
            body += copy(prefix="@%p0 ")
            expected = lambda c, l: {
                j: (100 * l + j if l < 16 else 0 if l < 24 else 0xA5A5A5A5)
                for j in range(4)
            }
        else:
            body += "mov.u32 %r6, 0;\n" + copy()
            expected = lambda c, l: dict.fromkeys(range(4), 0)
        make(
            name,
            body
            + "cp.async.commit_group;\ncp.async.wait_group 0;\nbar.sync 0;\n"
            + consume(),
            expected,
        )

    for n in [0, 2, 4, 6]:
        count = n + 2
        body = (
            """mov.u32 %r24, 0xa5a5a5a5;\nst.shared.v4.u32 [%r4], {%r24,%r24,%r24,%r24};
bar.sync 0;
setp.ge.u32 %p0, %r0, 16;
@!%p0 bra JOIN;
cp.async.commit_group;
JOIN:
mov.u32 %r7, 0;
mov.u32 %r8, %r4;
mov.u64 %rd4, %rd2;
LOOP:
"""
            + copy("%r8", "%rd4", prefix="@%p0 ")
            + f"""
cp.async.commit_group;
add.u32 %r8, %r8, 512;
add.u64 %rd4, %rd4, 512;
add.u32 %r7, %r7, 1;
setp.lt.u32 %p1, %r7, {count};
@%p1 bra LOOP;
cp.async.wait_group {n};
bar.sync 0;
"""
        )
        # Predicate-false lanes keep only first initialized slot. Do not read
        # the other uninitialized slots.
        body += "@!%p0 bra EMPTY;\n"
        for g in range(2):
            body += f"add.u32 %r8, %r4, {512*g};\n" + consume("%r8", 4 * g)
        body += "EMPTY:\ncp.async.wait_group 0;\nbar.sync 0;\n@!%p0 bra DONE;\n"
        for g in range(count):
            body += f"add.u32 %r8, %r4, {512*g};\n" + consume("%r8", 8 + 4 * g)
        body += "DONE:\n"

        def expected(c, l, count=count):
            if l < 16:
                return {}
            d = {}
            for g in range(2):
                d.update({4 * g + j: v for j, v in pattern(c, l, g).items()})
            for g in range(count):
                d.update({8 + 4 * g + j: v for j, v in pattern(c, l, g).items()})
            return d

        make("groups" + str(n), body, expected)

    # Same PC and same source, distinct destinations: both can be outstanding.
    body = (
        "mov.u32 %r7, 0;\nmov.u32 %r8, %r4;\nLOOP:\n"
        + copy("%r8")
        + """cp.async.commit_group;
add.u32 %r8, %r8, 512;
add.u32 %r7, %r7, 1;
setp.lt.u32 %p0, %r7, 2;
@%p0 bra LOOP;
cp.async.wait_group 0;
bar.sync 0;
"""
        + consume()
        + "add.u32 %r8, %r4, 512;\n"
        + consume("%r8", 4)
    )
    make(
        "identity",
        body,
        lambda c, l: {4 * g + j: v for g in range(2) for j, v in pattern(c, l).items()},
    )

    # Four rounds alternate two slots, cross-lane consumption after CTA sync.
    body = (
        "mov.u32 %r7, 0;\nLOOP:\nand.b32 %r8, %r7, 1;\nshl.b32 %r8, %r8, 9;\nadd.u32 %r8, %r4, %r8;\n"
        + copy("%r8")
        + """cp.async.commit_group;
cp.async.wait_group 0;
bar.sync 0;
xor.b32 %r9, %r2, 1;
shl.b32 %r9, %r9, 4;
sub.u32 %r8, %r8, %r3;
add.u32 %r8, %r8, %r9;
"""
        + consume("%r8")
        + """bar.sync 0;
add.u64 %rd3, %rd3, 16;
add.u64 %rd2, %rd2, 512;
add.u32 %r7, %r7, 1;
setp.lt.u32 %p0, %r7, 4;
@%p0 bra LOOP;
"""
    )
    make(
        "reuse",
        body,
        lambda c, l: {
            4 * g + j: v for g in range(4) for j, v in pattern(c, l ^ 1, g).items()
        },
        grid=2,
    )

    # Independent chain and another warp proceed while copy is outstanding.
    work = (
        "mov.u32 %r10, 1;\n"
        + "mad.lo.u32 %r10, %r10, 3, 1;\n" * 128
        + "st.global.u32 [%rd3+16], %r10;\n"
    )
    expected = lambda c, l: {**pattern(c, l % 32), 4: ((3**129 - 1) // 2) & 0xFFFFFFFF}
    make(
        "overlap",
        "setp.lt.u32 %p0, %r0, 32;\n@!%p0 bra WORK;\n"
        + copy()
        + """cp.async.commit_group;
WORK:
"""
        + work
        + """cp.async.wait_group 0;
bar.sync 0;
"""
        + consume(),
        expected,
        threads=64,
    )

    # Legal single vs double-buffer pipeline, same four tiles and arithmetic.
    for double in [False, True]:
        body = copy() + "cp.async.commit_group;\n"
        for g in range(4):
            slot = g % 2 if double else 0
            body += "cp.async.wait_group 0;\nbar.sync 0;\n"
            if double and g < 3:
                body += (
                    f"add.u32 %r8, %r4, {512*(1-slot)};\nadd.u64 %rd4, %rd2, {512*(g+1)};\n"
                    + copy("%r8", "%rd4")
                    + "cp.async.commit_group;\n"
                )
            body += (
                f"add.u32 %r8, %r4, {512*slot};\n"
                + consume("%r8", 4 * g)
                + work
                + "bar.sync 0;\n"
            )
            if not double and g < 3:
                body += (
                    f"add.u64 %rd4, %rd2, {512*(g+1)};\n"
                    + copy("%r4", "%rd4")
                    + "cp.async.commit_group;\n"
                )
        # work uses slot4, so place visible work elsewhere.
        body = body.replace("[%rd3+16], %r10", "[%rd3+128], %r10")
        make(
            "double" if double else "single",
            body,
            lambda c, l: {
                **{4 * g + j: v for g in range(4) for j, v in pattern(c, l, g).items()},
                32: ((3**129 - 1) // 2) & 0xFFFFFFFF,
            },
        )

    # Both halves copy at the same PC after only high lanes commit an empty
    # group. This exposes accidental warp-wide group numbering directly.
    make(
        "lane_groups",
        """setp.ge.u32 %p0, %r0, 16;
@!%p0 bra JOIN;
cp.async.commit_group;
JOIN:
"""
        + copy()
        + """cp.async.commit_group;
cp.async.wait_group 0;
bar.sync 0;
"""
        + consume(),
        pattern,
    )

    (run / "micro.ptx").write_text(
        "// Stage 2 microtests\n.version 8.4\n.target sm_90a\n.address_size 64\n"
        + "".join(entries)
    )
    (run / "cases.json").write_text(json.dumps(cases, indent=2))
    return cases
