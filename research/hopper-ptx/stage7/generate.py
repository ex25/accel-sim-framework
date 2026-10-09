#!/usr/bin/env python3
"""Produce PTX fixtures; CPU expectations live independently in the harness."""
import pathlib
import sys
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
header = '''.version 8.7
.target sm_90a
.address_size 64
.visible .entry cluster_kernel(.param .u64 out) {
.reg .b32 %r<32>;
.reg .b64 %rd<8>;
.reg .pred %p<4>;
.shared .align 16 .b8 tile[256];
mov.u32 %r0, %tid.x;
mov.u32 %r1, %cluster_ctarank;
mov.u32 %r2, %cluster_nctarank;
mov.u32 %r3, %ctaid.x;
mov.u32 %r4, %ctaid.y;
mov.u32 %r5, %ctaid.z;
mov.u32 %r6, %nctaid.x;
mov.u32 %r7, %nctaid.y;
mad.lo.u32 %r8, %r7, %r5, %r4;
mad.lo.u32 %r8, %r6, %r8, %r3;
mov.u32 %r9, %clusterid.x;
mov.u32 %r10, %clusterid.y;
mov.u32 %r11, %clusterid.z;
mov.u32 %r12, %nclusterid.x;
mov.u32 %r13, %nclusterid.y;
mov.u32 %r14, %nclusterid.z;
mov.u32 %r15, %cluster_ctaid.x;
mov.u32 %r16, %cluster_ctaid.y;
mov.u32 %r17, %cluster_ctaid.z;
mov.u32 %r18, %cluster_nctaid.x;
mov.u32 %r19, %cluster_nctaid.y;
mov.u32 %r20, %cluster_nctaid.z;
mov.u32 %r21, %is_explicit_cluster;
ld.param.u64 %rd0, [out];
mul.lo.u32 %r22, %r8, 32;
add.u32 %r22, %r22, %r0;
mul.wide.u32 %rd1, %r22, 80;
add.u64 %rd0, %rd0, %rd1;
'''
meta = [1,2,9,10,11,12,13,14,15,16,17,18,19,20,21]
output = ''.join(f'st.global.u32 [%rd0+{i*4}], %r{r};\n' for i,r in enumerate(meta))
body = '''mul.lo.u32 %r23, %r0, 4;
mov.u32 %r24, tile;
add.u32 %r24, %r24, %r23;
mad.lo.u32 %r25, %r8, 1000, %r0;
st.shared.u32 [%r24], %r25;
// Delayed last rank forces other CTAs to block at the first wait.
add.u32 %r26, %r1, 1;
setp.eq.u32 %p0, %r26, %r2;
mov.u32 %r27, 60;
@!%p0 bra READY;
DELAY:
sub.u32 %r27, %r27, 1;
setp.ne.u32 %p1, %r27, 0;
@%p1 bra DELAY;
READY:
barrier.cluster.arrive.release.aligned;
barrier.cluster.wait.acquire.aligned;
rem.u32 %r26, %r26, %r2;
mapa.shared::cluster.u32 %r28, %r24, %r26;
getctarank.shared::cluster.u32 %r29, %r28;
ld.shared::cluster.u32 %r30, [%r28];
cvt.u64.u32 %rd2, %r24;
mapa.shared::cluster.u64 %rd3, %rd2, %r26;
ld.shared::cluster.u64 %rd4, [%rd3];
getctarank.shared::cluster.u64 %r23, %rd3;
add.u32 %r28, %r28, 128;
st.shared::cluster.u32 [%r28], %r25;
fence.proxy.async.shared::cluster;
barrier.cluster.arrive.relaxed.aligned;
barrier.cluster.wait.aligned;
add.u32 %r24, %r24, 128;
ld.shared.u32 %r31, [%r24];
'''
# Avoid an unaligned u64 load for odd lanes: test the u64 address form with u32 data.
body = body.replace('ld.shared::cluster.u64 %rd4, [%rd3];', 'ld.shared::cluster.u32 %r27, [%rd3];')
end = output + '''st.global.u32 [%rd0+60], %r29;
st.global.u32 [%rd0+64], %r30;
st.global.u32 [%rd0+68], %r31;
st.global.u32 [%rd0+72], %r27;
st.global.u32 [%rd0+76], %r23;
barrier.cluster.arrive.aligned;
barrier.cluster.wait.aligned;
ret;
}\n'''
full = header + body + end
(out/'cluster.ptx').write_text(full)
(out/'metadata.ptx').write_text(header+output+'ret;\n}\n')
negative = {
 'bad_modifier': full.replace('mapa.shared::cluster.u32','mapa.shared::cta.u32'),
 'bad_width': full.replace('mapa.shared::cluster.u32 %r28, %r24','mapa.shared::cluster.u32 %rd3, %r24'),
 'bad_operands': full.replace('%r28, %r24, %r26;', '%r28, %r24;'),
 'bad_barrier': full.replace('barrier.cluster.arrive.release.aligned', 'barrier.cluster.arrive.acquire.aligned'),
 'predicated_barrier': full.replace('barrier.cluster.arrive.release.aligned;', '@%p0 barrier.cluster.arrive.release.aligned;'),
 'bad_rank': full.replace('mapa.shared::cluster.u32 %r28, %r24, %r26;', 'mov.u32 %r26, 8;\nmapa.shared::cluster.u32 %r28, %r24, %r26;'),
 'bad_address': full.replace('mapa.shared::cluster.u32 %r28, %r24, %r26;', 'mov.u32 %r24, 256;\nmapa.shared::cluster.u32 %r28, %r24, %r26;'),
 'bad_access': full.replace('ld.shared::cluster.u32 %r30, [%r28];', 'add.u32 %r28, %r28, 1;\nld.shared::cluster.u32 %r30, [%r28];'),
 'cta_access': full.replace('ld.shared::cluster.u32 %r30', 'ld.shared.u32 %r30'),
 'wait_without_arrive': full.replace('barrier.cluster.arrive.release.aligned;', ''),
 'duplicate_arrive': full.replace('barrier.cluster.wait.acquire.aligned;', 'barrier.cluster.arrive.aligned;'),
 'divergent_barrier': full.replace('READY:', 'READY:\nsetp.eq.u32 %p2, %r0, 0;\n@%p2 bra EXIT;').replace('ret;', 'EXIT:\nret;'),
 'missing_arrival': full.replace('READY:', 'READY:\n@%p0 bra EXIT;').replace('ret;', 'EXIT:\nret;'),
 'setmax_inc': header+'setmaxnreg.inc.sync.aligned.u32 232;\nret;\n}\n',
 'setmax_dec': header+'setmaxnreg.dec.sync.aligned.u32 40;\nret;\n}\n',
 'bad_sreg': full.replace('%cluster_ctarank;', '%cluster_ctarank.x;'),
 'setmax_count': header+'setmaxnreg.inc.sync.aligned.u32 23;\nret;\n}\n',
 'setmax_modifier': header+'setmaxnreg.inc.sync.u32 232;\nret;\n}\n',
}
for name, text in negative.items(): (out/f'{name}.ptx').write_text(text)
