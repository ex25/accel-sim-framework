"""Legal PTX barriers: separate producer warps and explicit round handshakes."""


def generate(run):
    entries = []
    for name in ('local', 'hint', 'lanes', 'zero_tx', 'remote',
                 'pending_tx', 'combined_tx'):
        common = '''
.reg .u32 %r<20>;
.reg .u64 %rd<4>;
.reg .pred %p<6>;
ld.param.u64 %rd0, [output];
ld.param.u32 %r9, [delay];
mov.u32 %r0, %tid.x;
mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %ctaid.y;
mad.lo.u32 %r3, %r2, 2, %r1;
mul.wide.u32 %rd1, %r3, 128;
add.u64 %rd1, %rd0, %rd1;
mov.u32 %r4, 1024;
setp.eq.u32 %p0, %r0, 0;
'''
        count = 64 if name == 'lanes' else (1 if name in (
            'remote', 'zero_tx', 'pending_tx', 'combined_tx') else 2)
        common += f'''mov.u32 %r11, {count};
mov.u32 %r12, 0;
@%p0 mbarrier.init.shared::cta.b64 [%r4], %r11;
@%p0 fence.mbarrier_init.release.cluster;
bar.sync 0;
'''
        if name == 'remote':
            body = '''barrier.cluster.arrive.relaxed.aligned;
barrier.cluster.wait.aligned;
@!%p0 bra done;
setp.ne.u32 %p1, %r2, 0;
@%p1 bra consumer;
add.u32 %r8, %r1, 1;
mul.lo.u32 %r8, %r8, %r9;
mov.u32 %r7, 7;
delay_remote:
mad.lo.u32 %r7, %r7, 1664525, 1013904223;
sub.u32 %r8, %r8, 1;
setp.ne.u32 %p2, %r8, 0;
@%p2 bra delay_remote;
mapa.shared::cluster.u32 %r5, %r4, 1;
mbarrier.arrive.shared::cluster.b64 _, [%r5];
mbarrier.try_wait.parity.shared::cta.b64 %p3, [%r4], %r12;
selp.u32 %r6, 1, 0, %p3;
st.global.u32 [%rd1], %r6;
bra done;
consumer:
mov.u32 %r10, 0;
loop_remote:
mbarrier.try_wait.parity.shared::cta.b64 %p3, [%r4], %r12, %r9;
add.u32 %r10, %r10, 1;
@!%p3 bra loop_remote;
mov.u32 %r6, 77;
st.global.u32 [%rd1], %r6;
st.global.u32 [%rd1+4], %r10;
done:
barrier.cluster.arrive.relaxed.aligned;
barrier.cluster.wait.aligned;
'''
        elif name in ('pending_tx', 'combined_tx'):
            body = '''@!%p0 bra done;
mov.u32 %r13, 64;
'''
            if name == 'pending_tx':
                body += '''mbarrier.expect_tx.shared::cta.b64 [%r4], %r13;
mbarrier.arrive.shared::cta.b64 _, [%r4];
'''
            else:
                body += '''mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r4], %r13;
'''
            body += '''setp.eq.u32 %p3, 0, 0;
mbarrier.try_wait.parity.shared::cta.b64 %p3, [%r4], %r12;
@%p3 bra done;
mov.u32 %r6, 99;
st.global.u32 [%rd1], %r6;
done:
'''
        elif name in ('lanes', 'zero_tx'):
            body = ('mbarrier.arrive.shared::cta.b64 _, [%r4];\n' if name == 'lanes' else '''@%p0 mbarrier.expect_tx.shared::cta.b64 [%r4], 0;
@%p0 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r4], 0;
''')
            body += '''bar.sync 0;
@!%p0 bra done;
setp.eq.u32 %p3, 0, 1;
mbarrier.try_wait.parity.shared::cta.b64 %p3, [%r4], %r12;
@!%p3 bra done;
mov.u32 %r6, 99;
st.global.u32 [%rd1], %r6;
done:
'''
        else:
            body = ''
            for round in range(3):
                hint = ', %r9' if name == 'hint' else ''
                body += f'''mov.u32 %r12, {round%2};
setp.eq.u32 %p1, %r0, 32;
@%p1 bra producer_{round};
@!%p0 bra join_{round};
mbarrier.arrive.shared::cta.b64 _, [%r4];
setp.eq.u32 %p3, 0, 0;
mbarrier.try_wait.parity.shared::cta.b64 %p3, [%r4], %r12{hint};
selp.u32 %r6, 1, 0, %p3;
st.global.u32 [%rd1+{round*16}], %r6;
mov.u32 %r10, 0;
loop_{round}:
mbarrier.try_wait.parity.shared::cta.b64 %p3, [%r4], %r12{hint};
add.u32 %r10, %r10, 1;
@!%p3 bra loop_{round};
mov.u32 %r6, {(round+1)*11};
st.global.u32 [%rd1+{round*16+4}], %r6;
st.global.u32 [%rd1+{round*16+8}], %r10;
bra join_{round};
producer_{round}:
mov.u32 %r8, %r9;
mov.u32 %r7, 7;
delay_{round}:
mad.lo.u32 %r7, %r7, 1664525, 1013904223;
sub.u32 %r8, %r8, 1;
setp.ne.u32 %p2, %r8, 0;
@%p2 bra delay_{round};
mbarrier.arrive.shared::cta.b64 _, [%r4];
join_{round}:
bar.sync 0;
'''
        entries.append(f'.visible .entry {name}(.param .u64 output, .param .u32 delay) {{\n'+common+body+'ret;\n}\n')
    (run/'micro.ptx').write_text('// Stage 4 legal PTX microtests\n.version 8.7\n.target sm_90a\n.address_size 64\n'+''.join(entries))
