"""Directed PTX inputs with expected results specified independently by lane."""

import struct
from pathlib import Path

SENTINEL = 0xDEADBEEF
STRIDE = 64


def generate(run):
    entries = {}
    patterns = [
        0,
        0x80000000,
        0x3F800000,
        0xC0000000,
        0x3F807FFF,
        0x3F808000,
        0x3F808001,
        0x3F818000,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
        0x00010000,
    ]
    rounded = [
        0,
        0x8000,
        0x3F80,
        0xC000,
        0x3F80,
        0x3F80,
        0x3F81,
        0x3F82,
        0x7F80,
        0xFF80,
        0x7FFF,
        0x0001,
    ]
    inputs = [patterns[i % len(patterns)] for i in range(64)]
    (run / "input.bin").write_bytes(struct.pack("<64I", *inputs))

    def new(name, threads=64):
        out = [[SENTINEL] * STRIDE for _ in range(threads)]
        code = [
            "ld.param.u64 %rd0, [input];",
            "ld.param.u64 %rd1, [output];",
            "mov.u32 %r0, %tid.x;",
            "and.b32 %r1, %r0, 31;",
            "mul.wide.u32 %rd2, %r0, 256;",
            "add.u64 %rd2, %rd1, %rd2;",
        ]
        entries[name] = dict(code=code, expected=out, threads=threads)
        return code, out

    def store(code, reg, slot):
        code.append(f"st.global.u32 [%rd2+{slot*4}], {reg};")

    c, e = new("numeric")
    c += [
        "mul.wide.u32 %rd3, %r0, 4;",
        "add.u64 %rd3, %rd0, %rd3;",
        "ld.global.b32 %r3, [%rd3];",
        "cvt.rn.bf16.f32 %rs0, %r3;",
        "cvt.u32.u16 %r4, %rs0;",
    ]
    store(c, "%r4", 0)
    c += ["cvt.f32.bf16 %r5, %rs0;"]
    store(c, "%r5", 1)
    # A distinct FP32 operation produces the second packed source.
    c += [
        "mov.f32 %f0, 0f3f800000;",
        "mov.f32 %f1, 0fbf800000;",
        "add.f32 %f2, %f1, %f1;",
        "cvt.rn.bf16x2.f32 %r6, %f0, %f2;",
    ]
    store(c, "%r6", 2)
    c += ["cvt.rn.bf16x2.f32 %r6, %f2, %f0;"]
    store(c, "%r6", 3)
    c += [
        "mov.b32 %r7, 0f3f808000;",
        "mov.b32 %r8, 0f3f818000;",
        "cvt.rn.bf16x2.f32 %r6, %r7, %r8;",
    ]
    store(c, "%r6", 4)
    c += ["mov.b16 %rs1, 1;", "cvt.f32.bf16 %r6, %rs1;"]
    store(c, "%r6", 5)
    c += [
        "mov.f32 %f0, 0f3fc00000;",
        "cvt.rn.f16.f32 %rs1, %f0;",
        "cvt.f32.f16 %f1, %rs1;",
        "mov.b32 %r6, %f1;",
    ]
    store(c, "%r6", 6)
    c += ["cvt.rzi.s32.f32 %r6, %f1;"]
    store(c, "%r6", 7)
    for i, row in enumerate(e):
        v = rounded[i % len(patterns)]
        row[:8] = [
            v,
            v << 16,
            0x3F80C000,
            0xC0003F80,
            0x3F803F82,
            0x10000,
            0x3FC00000,
            1,
        ]
    c, e = new("register_copies")
    c += [
        "mul.wide.u32 %rd3, %r0, 4;",
        "add.u64 %rd3, %rd0, %rd3;",
        "ld.global.b32 %r3, [%rd3];",
        "cvt.rn.bf16.f32 %rs0, %r3;",
        "cvt.f32.bf16 %r20, %rs0;",
        "mov.b32 %f3, %r20;",
        "mov.b32 %r4, %f3;",
        "add.u32 %r21, %r0, 7;",
        "mov.b32 %r5, %r21;",
        "shl.b32 %r22, %r0, 2;",
        "mov.b32 %r6, %r22;",
        "mov.f32 %f0, 0f3fc00000;",
        "mov.f32 %f1, 0f40000000;",
        "fma.rn.f32 %f2, %f0, %f1, %f0;",
        "mov.b32 %r7, %f2;",
        "mov.u32 %r8, 11;",
        "add.u32 %r23, %r8, 5;",
        "mov.b32 %r8, %r23;",
        "mov.u32 %r9, 0;",
        "copy_loop:",
        "add.u32 %r24, %r9, 1;",
        "mov.b32 %r9, %r24;",
        "setp.lt.u32 %p0, %r9, 3;",
        "@%p0 bra copy_loop;",
    ]
    for slot, reg in enumerate(["%r4", "%r5", "%r6", "%r7", "%r8", "%r9"]):
        store(c, reg, slot)
    for i, row in enumerate(e):
        row[:6] = [rounded[i % len(patterns)] << 16, i + 7, i << 2,
                   0x40900000, 16, 3]

    c, e = new("register_copy_live")
    c += [
        "add.u32 %r20, %r0, 7;",
        "mov.b32 %r4, %r20;",
        "add.u32 %r5, %r20, 1;",
        ".shared .align 4 .b32 copy_shared;",
        "setp.eq.u32 %p0, %r0, 0;",
        "mov.u32 %r10, 17;",
        "@%p0 st.shared.b32 [copy_shared], %r10;",
        "bar.sync 0;",
        "mov.u32 %r21, copy_shared;",
        "mov.b32 %r6, %r21;",
        "ld.shared.u32 %r7, [%r21];",
        "add.u32 %r22, %r0, 3;",
        "mov.b32 %r8, %r22;",
        "st.global.v2.b32 [%rd2+16], {%r22, %r8};",
        "add.u32 %r23, %r0, 4;",
        "mov.b32 %r9, %r23;",
        "add.u32 %r23, %r0, 5;",
    ]
    for slot, reg in [(0, "%r4"), (1, "%r5"), (2, "%r7"),
                      (6, "%r9"), (7, "%r23")]:
        store(c, reg, slot)
    for i, row in enumerate(e):
        row[:8] = [i + 7, i + 8, 17, SENTINEL, i + 3, i + 3, i + 4, i + 5]

    c, e = new("register_copy_control")
    c += [
        "mov.u32 %r4, 99;",
        "setp.eq.u32 %p0, %r1, 0;",
        "add.u32 %r20, %r0, 7;",
        "@%p0 mov.b32 %r4, %r20;",
        "add.u32 %r21, %r0, 8;",
        "copy_boundary:",
        "mov.b32 %r5, %r21;",
        "mov.u32 %r22, 17;",
        "@%p0 add.u32 %r22, %r0, 9;",
        "mov.b32 %r6, %r22;",
    ]
    for slot, reg in enumerate(["%r4", "%r5", "%r6"]):
        store(c, reg, slot)
    for i, row in enumerate(e):
        row[:3] = [i + 7 if i % 32 == 0 else 99, i + 8,
                   i + 9 if i % 32 == 0 else 17]

    c, e = new("collectives")
    c += [
        "setp.lt.u32 %p0, %r0, 32;",
        "selp.u32 %r2, 100, 1000, %p0;",
        "add.u32 %r3, %r2, %r1;",
    ]
    for slot, inst in enumerate(
        [
            "shfl.sync.bfly.b32 %r4|%p1, %r3, 16, 31, -1;",
            "shfl.sync.up.b32 %r4|%p1, %r3, 2, 0, -1;",
            "shfl.sync.idx.b32 %r4|%p1, %r3, 7, 3, -1;",
            "shfl.sync.bfly.b32 %r4|%p1, %r3, 16, 31, -1;",
        ]
    ):
        c += [inst]
        store(c, "%r4", 2 * slot)
        c += ["selp.u32 %r5, 1, 0, %p1;"]
        store(c, "%r5", 2 * slot + 1)
    c += ["shfl.sync.bfly.b32 %r3, %r3, 16, 31, -1;"]
    store(c, "%r3", 8)
    c += [
        "and.b32 %r6, %r1, 1;",
        "setp.eq.u32 %p2, %r6, 0;",
        "vote.sync.ballot.b32 %r7, %p2, -1;",
    ]
    store(c, "%r7", 9)
    c += ["vote.sync.ballot.b32 %r8, !%p2, -1;"]
    store(c, "%r8", 10)
    c += ["@!%p2 bra after_dynamic;", "vote.sync.ballot.b32 %r8, %p2, %r7;"]
    store(c, "%r8", 11)
    c += [
        "mov.u32 %r14, 0;",
        "mov.u32 %r15, 31;",
        "shfl.sync.idx.b32 %r12|%p5, %r3, %r14, %r15, %r7;",
    ]
    store(c, "%r12", 24)
    c += ["selp.u32 %r12, 1, 0, %p5;"]
    store(c, "%r12", 25)
    c += ["after_dynamic:", "add.u32 %r9, %r1, 1;", "redux.sync.add.s32 %r10, %r9, -1;"]
    store(c, "%r10", 12)
    c += ["sub.u32 %r9, %r1, 16;", "redux.sync.add.s32 %r10, %r9, -1;"]
    store(c, "%r10", 13)
    c += ["elect.sync %r10|%p3, -1;"]
    store(c, "%r10", 14)
    c += ["selp.u32 %r11, 1, 0, %p3;"]
    store(c, "%r11", 15)
    c += ["elect.sync _|%p3, -1;", "selp.u32 %r11, 1, 0, %p3;"]
    store(c, "%r11", 16)
    c += [
        "setp.lt.u32 %p4, %r1, 16;",
        "@%p4 bra after_high;",
        "add.u32 %r3, %r2, %r1;",
        "shfl.sync.idx.b32 %r4|%p1, %r3, 3, 0x100f, 0xffff0000;",
    ]
    store(c, "%r4", 17)
    c += ["selp.u32 %r5, 1, 0, %p1;"]
    store(c, "%r5", 18)
    c += ["vote.sync.ballot.b32 %r4, %p2, 0xffff0000;"]
    store(c, "%r4", 19)
    c += ["add.u32 %r9, %r1, 1;", "redux.sync.add.s32 %r10, %r9, 0xffff0000;"]
    store(c, "%r10", 20)
    c += ["elect.sync %r10|%p3, 0xffff0000;"]
    store(c, "%r10", 21)
    c += ["selp.u32 %r11, 1, 0, %p3;"]
    store(c, "%r11", 22)
    c += ["elect.sync %r10|%p3, 0xffff0000;"]
    store(c, "%r10", 23)
    c += ["after_high:"]
    for i, row in enumerate(e):
        lane = i % 32
        v = 100 if i < 32 else 1000
        row[:11] = [
            v + (lane ^ 16),
            1,
            v + max(0, lane - 2) if lane >= 2 else v + lane,
            int(lane >= 2),
            v + lane,
            0,
            v + (lane ^ 16),
            1,
            v + (lane ^ 16),
            0x55555555,
            0xAAAAAAAA,
        ]
        if lane % 2 == 0:
            row[11] = 0x55555555
            row[24:26] = [v + 16, 1]
        row[12:17] = [528, 0xFFFFFFF0, 0, int(lane == 0), int(lane == 0)]
        if lane >= 16:
            row[17:24] = [v + 19, 1, 0x55550000, 392, 16, int(lane == 16), 16]
    c, e = new("bits_loads")
    c.insert(0, ".shared .align 8 .b64 address_values[64];")
    c += ["mov.u32 %r3, %lanemask_le;"]
    store(c, "%r3", 0)
    for j, v in enumerate([0, 1, 8, 0x80000000]):
        c += [f"mov.u32 %r4, {v};", "bfind.shiftamt.u32 %r5, %r4;"]
        store(c, "%r5", j * 2 + 1)
        c += ["bfind.u32 %r5, %r4;"]
        store(c, "%r5", j * 2 + 2)
    c += [
        "setp.ne.u32 %p0, %r1, 0;",
        "mul.wide.u32 %rd3, %r0, 4;",
        "add.u64 %rd3, %rd0, %rd3;",
        "mov.u32 %r6, 0xdeadbeef;",
        "@%p0 ld.global.L1::evict_first.b32 %r6, [%rd3];",
    ]
    store(c, "%r6", 9)
    c += ["mov.u32 %r6, 0xdeadbeef;", "@%p0 ld.global.L1::evict_last.b32 %r6, [%rd3];"]
    store(c, "%r6", 10)
    c += [
        "and.b32 %r7, %r1, 3;",
        "cvt.u64.u32 %rd5, %r7;",
        "mov.u64 %rd4, bytes;",
        "add.u64 %rd4, %rd4, %rd5;",
        "mov.u32 %r6, 0xdeadbeef;",
        "@%p0 ld.global.L1::evict_first.b8 %r6, [%rd4];",
    ]
    store(c, "%r6", 11)
    # A negative immediate leaves carry bits in the implementation's wide
    # storage. A b32 address still denotes only the low 32 bits, including
    # when the loaded/stored data itself is 64 bits wide.
    c += [
        "mov.u32 %r17, address_values;",
        "mul.lo.u32 %r18, %r0, 8;",
        "add.u32 %r17, %r17, %r18;",
        "cvt.u64.u32 %rd6, %r0;",
        "or.b64 %rd6, %rd6, 0x1234567800000000;",
        "add.s32 %r18, %r17, 8;",
        "add.s32 %r18, %r18, -8;",
        "st.shared.u64 [%r18], %rd6;",
        "ld.shared.u64 %rd7, [%r17];",
        "mov.b64 {%r19, %r20}, %rd7;",
    ]
    store(c, "%r19", 12)
    store(c, "%r20", 13)
    c += ["ld.shared.u64 %rd7, [%r18];", "mov.b64 {%r19, %r20}, %rd7;"]
    store(c, "%r19", 14)
    store(c, "%r20", 15)
    for i, row in enumerate(e):
        row[:9] = [(1 << (i % 32 + 1)) - 1, 0xFFFFFFFF, 0xFFFFFFFF, 31, 0, 28, 3, 0, 31]
        if i % 32:
            row[9:12] = [inputs[i], inputs[i], [0, 0x7F, 0x80, 0xFF][i % 4]]
        row[12:16] = [i, 0x12345678, i, 0x12345678]
    for name in ("shared_aligned_loop", "shared_misaligned", "shared_branch"):
        c, e = new(name)
        c.insert(0, ".shared .align 16 .b32 aligned_values[1024];")
        c += ["mov.u32 %r14, aligned_values;", "shl.b32 %r4, %r0, 6;",
              "add.u32 %r3, %r14, %r4;"]
        if name == "shared_misaligned":
            c += ["add.u32 %r3, %r3, 4;"]
        if name == "shared_branch":
            c += ["and.b32 %r4, %r0, 1;", "setp.eq.u32 %p0, %r4, 0;",
                  "@%p0 bra aligned_path;", "add.u32 %r3, %r3, 4;",
                  "aligned_path:"]
        c += ["mov.u32 %r6, 0;", "aligned_loop:",
              "shl.b32 %r7, %r6, 4;", "add.u32 %r9, %r3, %r7;",
              "add.u32 %r15, %r0, %r6;"]
        for j in range(4):
            c += [f"add.u32 %r8, %r15, {100+j};",
                  f"st.shared.u32 [%r9+{4*j}], %r8;"]
        for j in (3, 1, 0, 2):
            c += [f"ld.shared.u32 %r{10+j}, [%r9+{4*j}];"]
        for j in range(4):
            store(c, f"%r{10+j}", j)
        if name == "shared_aligned_loop":
            c += ["add.u32 %r6, %r6, 1;", "setp.lt.u32 %p0, %r6, 2;",
                  "@%p0 bra aligned_loop;"]
        for i, row in enumerate(e):
            row[:4] = [100+i+j+(name == "shared_aligned_loop") for j in range(4)]

    c, e = new("shared_vector")
    c.insert(0, ".shared .align 16 .b32 vector_values[512];")
    c += [
        "mov.u32 %r3, vector_values;",
        "mul.lo.u32 %r4, %r0, 32;",
        "add.u32 %r3, %r3, %r4;",
        "mul.lo.u32 %r4, %r0, 10;",
        "add.u32 %r4, %r4, 1000;",
    ]
    for j in range(8):
        c += [f"add.u32 %r5, %r4, {j};", f"st.shared.u32 [%r3+{j*4}], %r5;"]
    c += [
        # The v4 read proves alignment; the following scalar reads are out of
        # address order. All eight destinations must retain their own word.
        "ld.shared.v4.u32 {%r4,%r5,%r6,%r7}, [%r3+16];",
        "ld.shared.u32 %r10, [%r3+8];",
        "ld.shared.u32 %r8, [%r3];",
        "ld.shared.u32 %r11, [%r3+12];",
        "ld.shared.u32 %r9, [%r3+4];",
    ]
    for j in range(8):
        store(c, f"%r{8+j}" if j < 4 else f"%r{j}", j)
    c += [
        "ld.shared.v4.u32 {%r4,%r5,%r6,%r7}, [%r3+16];",
        "ld.shared.v2.u32 {%r10,%r11}, [%r3+8];",
        "ld.shared.u32 %r8, [%r3];",
        "ld.shared.u32 %r9, [%r3+4];",
    ]
    for j in range(4):
        store(c, f"%r{8+j}", 8+j)
    c += [
        # Repeated destinations and incomplete words cannot form a v4 read.
        "ld.shared.v4.u32 {%r4,%r5,%r6,%r7}, [%r3+16];",
        "ld.shared.u32 %r8, [%r3];",
        "ld.shared.u32 %r8, [%r3+4];",
        "ld.shared.u32 %r9, [%r3+8];",
        "ld.shared.u32 %r10, [%r3+12];",
    ]
    for j in range(3):
        store(c, f"%r{8+j}", 12+j)
    c += [
        "add.u32 %r3, %r3, 4;",
        # Four scalar reads alone do not prove 16-byte alignment.
        "ld.shared.u32 %r8, [%r3];",
        "ld.shared.u32 %r9, [%r3+4];",
        "ld.shared.u32 %r10, [%r3+8];",
        "ld.shared.u32 %r11, [%r3+12];",
    ]
    for j in range(4):
        store(c, f"%r{8+j}", 15+j)
    c += [
        "add.u32 %r3, %r3, -4;",
        "mov.u32 %r8, 777;",
        "setp.lt.u32 %p0, %r1, 16;",
        "ld.shared.v4.u32 {%r4,%r5,%r6,%r7}, [%r3+16];",
        "@%p0 ld.shared.u32 %r8, [%r3];",
        "ld.shared.u32 %r9, [%r3+4];",
    ]
    store(c, "%r8", 19)
    c += [
        # A load that overwrites its base changes subsequent addresses.
        "add.u32 %r8, %r3, 16;",
        "st.shared.u32 [%r3], %r8;",
        "ld.shared.v4.u32 {%r4,%r5,%r6,%r7}, [%r3+16];",
        "ld.shared.u32 %r3, [%r3];",
        "ld.shared.v2.u32 {%r8,%r9}, [%r3];",
    ]
    store(c, "%r8", 20)
    store(c, "%r9", 21)
    for i, row in enumerate(e):
        v = 1000 + i * 10
        row[:8] = [v+j for j in range(8)]
        row[8:12] = [v+j for j in range(4)]
        row[12:15] = [v+1, v+2, v+3]
        row[15:19] = [v+j for j in range(1, 5)]
        row[19:22] = [v if i % 32 < 16 else 777, v+4, v+5]

    for partial in [False, True]:
        name = "warp_partial" if partial else "warp_full"
        c, e = new(name)
        c += ["setp.ge.u32 %p0, %r0, 32;", "@%p0 bra other_warp;"]
        if partial:
            c += ["setp.lt.u32 %p1, %r1, 16;", "@%p1 bra done;"]
        c += [
            "mov.u32 %r4, shared_values;",
            "mul.lo.u32 %r5, %r1, 4;",
            "add.u32 %r4, %r4, %r5;",
            "add.u32 %r6, %r1, 100;",
            "st.shared.u32 [%r4], %r6;",
            "mov.u32 %r16, " + ("0xffff0000;" if partial else "-1;"),
            "bar.warp.sync " + ("%r16;" if partial else "-1;"),
            "xor.b32 %r5, %r5, 4;",
            "mov.u32 %r4, shared_values;",
            "add.u32 %r4, %r4, %r5;",
            "ld.shared.u32 %r7, [%r4];",
        ]
        store(c, "%r7", 0)
        c += [
            "bra done;",
            "other_warp:",
            "mov.u32 %r8, 100;",
            "delay:",
            "sub.u32 %r8, %r8, 1;",
            "setp.ne.u32 %p2, %r8, 0;",
            "@%p2 bra delay;",
            "mov.u32 %r9, 77;",
        ]
        store(c, "%r9", 1)
        c += ["done:"]
        for i, row in enumerate(e):
            if i >= 32:
                row[1] = 77
            elif not partial or i >= 16:
                row[0] = 100 + (i ^ 1)
    for count in [32, 64]:
        c, e = new("chain" + str(count), 32)
        c += ["mov.u32 %r3, %r1;"] + [
            f"shfl.sync.bfly.b32 %r3, %r3, 1, 31, -1;" for _ in range(count)
        ]
        store(c, "%r3", 0)
        for i, row in enumerate(e):
            row[0] = i
    c, e = new("throughput", 1024)
    c += [f"mov.u32 %r{j+3}, %r1;" for j in range(8)]
    c += [
        f"shfl.sync.bfly.b32 %r{j+3}, %r{j+3}, 1, 31, -1;"
        for _ in range(16)
        for j in range(8)
    ]
    for j in range(8):
        store(c, f"%r{j+3}", j)
    for i, row in enumerate(e):
        row[:8] = [i % 32] * 8
    c, e = new("legacy", 32)
    c += ["add.u32 %r3, %r1, 100;", "shfl.bfly.b32 %r3|%p0, %r3, 16, 31;"]
    store(c, "%r3", 0)
    c += ["selp.u32 %r4, 1, 0, %p0;"]
    store(c, "%r4", 1)
    for i, row in enumerate(e):
        row[:2] = [100 + (i ^ 16), 1]
    header = "// Stage 1 directed PTX\n.version 8.7\n.target sm_90a\n.address_size 64\n.global .align 4 .b8 bytes[4]={0,127,128,255};\n"
    for name, item in entries.items():
        code = "\n".join(item["code"])
        program = f""".visible .entry {name}(.param .u64 input, .param .u64 output, .param .u32 n) {{
.reg .b32 %r<32>; .reg .b64 %rd<8>; .reg .b16 %rs<4>;
.reg .f32 %f<4>; .reg .pred %p<8>; .shared .align 4 .b32 shared_values[32];
{code}
ret;
}}\n"""
        if name == "legacy":
            (run / "legacy.ptx").write_text(
                "// Legacy shuffle regression\n.version 5.0\n.target sm_60\n.address_size 64\n"
                + program
            )
        else:
            header += program
        (run / (name + ".expected")).write_bytes(
            struct.pack(
                "<" + "I" * (len(item["expected"]) * STRIDE),
                *(v for row in item["expected"] for v in row),
            )
        )
    (run / "micro.ptx").write_text(header)
    return entries
