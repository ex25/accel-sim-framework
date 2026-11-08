"""Stage 3 PTX; coordinates and arithmetic expectations are constructed on host."""


def generate(run):
    entries = []
    for name, offsets, barriers in [
        ("identity", [0, 4088, 4092], False),
        ("mapa", [0, 4096, 8188], False),
        ("addresses", [0, 98304, 215036], False),
        ("address_sample", [0, 98304, 215036], False),
        ("cluster_sync", [0, 4096, 8188], True),
        ("ordinary", [0, 24, 28], False),
    ]:
        body = ""
        for i, offset in enumerate(offsets):
            body += f"mov.u32 %r16, {offset};\n@%p0 st.shared.u32 [%r16], %r10;\n"
        body += "bar.sync 0;\n"
        for i, offset in enumerate(offsets):
            body += f"""mov.u64 %rd4, {offset};
cvta.shared.u64 %rd5, %rd4;
cvta.to.shared.u64 %rd6, %rd5;
isspacep.shared %p1, %rd5;
selp.u32 %r17, 1, 0, %p1;
ld.u32 %r18, [%rd5];
st.global.u32 [%rd2+{(9+i)*4}], %r18;
cvt.u32.u64 %r19, %rd6;
st.global.u32 [%rd2+{(12+i)*4}], %r19;
st.global.u32 [%rd2+{(25+i)*4}], %r17;
"""
        # Save one full generic pointer, then exercise local conversion and generic I/O.
        body += """st.global.u64 [%rd2+112], %rd5;
mov.u64 %rd7, localbuf;
cvta.local.u64 %rd8, %rd7;
cvta.to.local.u64 %rd9, %rd8;
st.u32 [%rd8], %r0;
ld.local.u32 %r18, [localbuf];
st.global.u32 [%rd2+32], %r18;
cvt.u32.u64 %r19, %rd9;
st.global.u32 [%rd2+60], %r19;
isspacep.local %p1, %rd8;
selp.u32 %r17, 1, 0, %p1;
st.global.u32 [%rd2+120], %r17;
isspacep.global %p1, %rd0;
selp.u32 %r17, 1, 0, %p1;
st.global.u32 [%rd2+124], %r17;
"""
        if name == "mapa":
            body += """mov.u32 %r16, 4096;
mapa.shared::cluster.u32 %r17, %r16, 0;
mapa.shared::cluster.u32 %r18, %r16, 1;
st.global.u32 [%rd2+64], %r17;
st.global.u32 [%rd2+68], %r18;
"""
        if barriers:
            for round in range(3):
                body += f"""setp.eq.u32 %p2, %r1, 0;
setp.eq.u32 %p3, %r5, 1;
and.pred %p2, %p2, %p3;
mov.u32 %r20, 0;
@%p2 mul.lo.u32 %r20, %r15, {round+1};
mov.u32 %r21, 7;
setp.eq.u32 %p4, %r20, 0;
@%p4 bra done_delay_{round};
delay_{round}:
mad.lo.u32 %r21, %r21, 1664525, 1013904223;
sub.u32 %r20, %r20, 1;
setp.ne.u32 %p4, %r20, 0;
@%p4 bra delay_{round};
done_delay_{round}:
barrier.cluster.arrive.relaxed.aligned;
"""
                for i in range(8):
                    body += "mad.lo.u32 %r21, %r21, 1664525, 1013904223;\n"
                body += f"""st.global.u32 [%rd2+{(16+round)*4}], %r21;
mov.u32 %r22, %clock;
st.global.u32 [%rd2+{(19+round)*4}], %r22;
barrier.cluster.wait.aligned;
mov.u32 %r23, %clock;
st.global.u32 [%rd2+{(22+round)*4}], %r23;
"""
        sample = "setp.eq.u32 %p0, %r0, 0;\n@!%p0 bra sample_exit;" if name == "address_sample" else ""
        entries.append(f"""
.visible .entry {name}(.param .u64 output, .param .u32 delay) {{
.reg .u32 %r<32>;
.reg .u64 %rd<16>;
.reg .pred %p<8>;
.local .align 16 .b8 localbuf[16];
ld.param.u64 %rd0, [output];
ld.param.u32 %r15, [delay];
mov.u32 %r0, %tid.x;
{sample}
mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %ctaid.y;
mov.u32 %r3, %nctaid.x;
mad.lo.u32 %r4, %r2, %r3, %r1;
mov.u32 %r5, %cluster_ctarank;
mov.u32 %r6, %cluster_ctaid.x;
mov.u32 %r7, %cluster_ctaid.y;
mov.u32 %r8, %cluster_ctaid.z;
mov.u32 %r9, %smid;
mad.lo.u32 %r10, %r1, 100, 1;
mad.lo.u32 %r10, %r5, 10, %r10;
mov.u32 %r11, %ntid.x;
mad.lo.u32 %r12, %r4, %r11, %r0;
mul.wide.u32 %rd1, %r12, 128;
add.u64 %rd2, %rd0, %rd1;
st.global.u32 [%rd2], %r1;
st.global.u32 [%rd2+4], %r2;
st.global.u32 [%rd2+8], %r5;
st.global.u32 [%rd2+12], %r6;
st.global.u32 [%rd2+16], %r7;
st.global.u32 [%rd2+20], %r8;
st.global.u32 [%rd2+24], %r9;
st.global.u32 [%rd2+28], %r10;
setp.eq.u32 %p0, %r0, 0;
{body}
sample_exit:
ret;
}}
""")
    (run / "micro.ptx").write_text(
        "// Stage 3 legal PTX microtests\n.version 8.7\n.target sm_90a\n.address_size 64\n"
        + "\n".join(entries)
    )
