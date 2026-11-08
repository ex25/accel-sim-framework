from pathlib import Path

def generate(run):
    text = '// stage 6 matrix execution\n.version 8.7\n.target sm_90a\n.address_size 64\n.extern .shared .align 128 .b8 sm[];\n'
    def entry(name, body, threads=128):
        if name == 'integration':
            # The peer CTA participates in cluster synchronization while rank 0
            # performs the first TMA/WGMMA integration, as required by stage 5.
            body = '''barrier.cluster.arrive.relaxed.aligned;
barrier.cluster.wait.aligned;
mov.u32 %r31, %cluster_ctarank;
setp.ne.u32 %p7, %r31, 0; @%p7 bra JOINT_PEER;
''' + body + '''JOINT_PEER:
barrier.cluster.arrive.relaxed.aligned;
barrier.cluster.wait.aligned;
'''
        return f'''.visible .entry test_{name}(.param .u64 input, .param .u64 output,
            .param .u64 map_a, .param .u64 map_b, .param .u64 map_c)
.maxntid {threads}, 1, 1 {{
.reg .b32 %r<32>; .reg .b64 %rd<16>; .reg .f32 %f<256>;
.reg .pred %p<8>;
ld.param.u64 %rd0, [input]; ld.param.u64 %rd1, [output];
ld.param.u64 %rd10, [map_a]; ld.param.u64 %rd11, [map_b];
ld.param.u64 %rd12, [map_c];
mov.u32 %r0, %tid.x;
{body}
ret;
}}\n'''
    def dump(regs, offset=0):
        s=f'mul.wide.u32 %rd5, %r0, 256; add.u64 %rd5, %rd1, %rd5;\n'
        for j,r in enumerate(regs):
            s+=f'st.global.f32 [%rd5+{offset+j*4}], %f{r};\n'
        return s
    regs = [3*j+2 for j in range(64)]
    ys = [3*j+1 for j in range(64)]
    def mma(rs=regs):
        return 'wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {'+', '.join('%f'+str(r) for r in rs)+'}, %rd2, %rd3, %p0, 1, 1, 0, 0;\n'
    prep='''mov.u32 %r1, %r0;
COPY:
mul.wide.u32 %rd4, %r1, 4; add.u64 %rd4, %rd0, %rd4;
ld.global.u32 %r2, [%rd4]; mul.lo.u32 %r3, %r1, 4;
st.shared.u32 [%r3], %r2;
add.u32 %r1, %r1, 128; setp.lt.u32 %p1, %r1, 10240; @%p1 bra COPY;
bar.sync 0; fence.proxy.async.shared::cta;
'''
    for name in ['single64','single128','offset64','offset128','offset_low64','offset_low128','initial','chain','groups','loop','integration']:
        width=64 if name.endswith('64') else 128
        base=(1296 if 'low' in name else 1280) if name.startswith('offset') else 0
        phase=(base//128)%(width//16)
        # Physical start is specified independently of the descriptor decoder.
        start=base+[0,16,32,48,64,80,96,112][phase]
        def desc(start):
            return (start//16) | (7<<16) | ((8*width//16)<<32) | (phase<<49) | ((2 if width==64 else 1)<<62)
        body=prep
        if name=='integration':
            # B fills [16384, 32768); its padding must not overwrite mbarrier.
            body='''mov.u32 %r23, 0; mov.u32 %r20, 0; mov.u32 %r21, 16384; mov.u32 %r22, 40000;
setp.eq.u32 %p1, %r0, 0; @!%p1 bra INIT_DONE;
mbarrier.init.shared::cta.b64 [%r22], 1;
fence.mbarrier_init.release.cluster;
mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r22], 20480;
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%r20], [%rd10, {%r23,%r23}], [%r22], 0;
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%r21], [%rd11, {%r23,%r23}], [%r22], 0;
INIT_DONE:
bar.sync 0;
POLL:
mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r22], 0;
@!%p1 bra POLL;
'''
            # legal 64B swizzle boxes: 32 BF16 values per row, first K=16 consumed
            width=64;phase=0
        body+=f'mov.u64 %rd2, {desc(start)}; mov.u64 %rd3, {desc(start+16384)};\n'
        if name=='integration':
            body += f'mov.u64 %rd3, {(1<<62)|(64<<32)|(7<<16)|(16384//16)};\n'
        body+='setp.eq.u32 %p0, %r0, 9999;\n'
        if name=='initial':
            body+='mul.wide.u32 %rd6, %r0, 256; add.u64 %rd6, %rd0, %rd6;\n'
            for j,r in enumerate(regs):
                body+=f'ld.global.f32 %f{r}, [%rd6+{40960+j*4}];\n'
            body+='setp.eq.u32 %p0, %r0, %r0;\n'
        body+='''shr.u32 %r8, %r0, 5; add.u32 %r8, %r8, 1; mov.u32 %r9, 0;
SKEW:
add.u32 %r9, %r9, 1; setp.lt.u32 %p1, %r9, %r8; @%p1 bra SKEW;
wgmma.fence.sync.aligned;
'''

        if name=='loop': body+='mov.u32 %r8, 0;\nITER:\n'
        body+=mma()
        if name=='chain':
            body+='setp.eq.u32 %p0, %r0, %r0;\n'+mma()
        body+='wgmma.commit_group.sync.aligned;\n'
        if name=='groups':
            body+=mma(ys)+'setp.eq.u32 %p0, %r0, %r0;\n'+mma(ys)
            body+='wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 1;\n'+dump(regs)
            body+='wgmma.wait_group.sync.aligned 0;\n'+dump(ys,32768)
        else:
            body+='wgmma.wait_group.sync.aligned 0;\n'
            if name=='loop':
                body+='add.u32 %r8, %r8, 1; setp.lt.u32 %p1, %r8, 2; @%p1 bra ITER;\n'
            body+=dump(regs)
        text+=entry(name,body)
    for name in ['ldmatrix','stmatrix']:
        body='''and.b32 %r1, %r0, 31; mul.lo.u32 %r2, %r1, 16;
'''
        if name=='ldmatrix':
            body+='''mul.wide.u32 %rd4, %r0, 16; add.u64 %rd4, %rd0, %rd4;
ld.global.v4.b32 {%r4,%r5,%r6,%r7}, [%rd4];
st.shared.v4.b32 [%r2], {%r4,%r5,%r6,%r7}; bar.sync 0;
ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r4,%r5,%r6,%r7}, [%r2];
'''
        else:
            body+='''mul.wide.u32 %rd4, %r0, 16; add.u64 %rd4, %rd0, %rd4;
ld.global.v4.b32 {%r4,%r5,%r6,%r7}, [%rd4];
stmatrix.sync.aligned.x4.m8n8.shared.b16 [%r2], {%r4,%r5,%r6,%r7};
bar.sync 0; ld.shared.v4.b32 {%r4,%r5,%r6,%r7}, [%r2];
'''
        body+='''mul.wide.u32 %rd4, %r0, 16; add.u64 %rd4, %rd1, %rd4;
st.global.v4.b32 [%rd4], {%r4,%r5,%r6,%r7};
'''
        text+=entry(name,body,32)
    # Two input buffers; WGMMA results are independently converted/stored via
    # stmatrix, then TMA read-wait protects output-buffer reuse.
    def load(stage):
        a=stage*20480
        b=a+4096
        barrier=60000+8*stage
        coord=stage*64
        return f'''mov.u32 %r20, {a}; mov.u32 %r21, {b};
mov.u32 %r22, {barrier}; mov.u32 %r23, 0; mov.u32 %r24, {coord}; mov.u32 %r25, {stage*128};
setp.eq.u32 %p1, %r0, 0; @!%p1 bra LOAD_DONE{stage};
mbarrier.init.shared::cta.b64 [%r22], 1;
fence.mbarrier_init.release.cluster;
mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r22], 20480;
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%r20], [%rd10, {{%r23,%r24}}], [%r22], 0;
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%r21], [%rd11, {{%r23,%r25}}], [%r22], 0;
LOAD_DONE{stage}:
bar.sync 0;
'''
    def wait_input(stage):
        return f'''mov.u32 %r22, {60000+8*stage};
WAIT_INPUT{stage}:
mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r22], 0;
@!%p1 bra WAIT_INPUT{stage};
'''
    def calculate(stage,rs):
        da=(2<<62)|(32<<32)|(7<<16)|(stage*20480//16)
        db=(1<<62)|(64<<32)|(7<<16)|((stage*20480+4096)//16)
        return f'mov.u64 %rd2, {da}; mov.u64 %rd3, {db}; setp.eq.u32 %p0, %r0, 9999;\n'+mma(rs)+'wgmma.commit_group.sync.aligned;\n'
    def output(stage,rs):
        body=''
        # q=0/1 are row pairs, q=2/3 are the next eight-column strip.
        for strip in range(8):
            for q in range(4):
                j=strip*8+2*q
                body+=f'cvt.rn.bf16x2.f32 %r{4+q}, %f{rs[j+1]}, %f{rs[j]};\n'
            body+='and.b32 %r1, %r0, 31; shr.u32 %r2, %r0, 5; mul.lo.u32 %r2, %r2, 16;\n'
            # Address-provider lane 8q+r supplies row base plus nstrip offset.
            body+='and.b32 %r3, %r1, 7; add.u32 %r2, %r2, %r3;\n'
            body+='and.b32 %r3, %r1, 8; add.u32 %r2, %r2, %r3; mul.lo.u32 %r2, %r2, 128;\n'
            body+=f'and.b32 %r3, %r1, 16; add.u32 %r2, %r2, %r3; add.u32 %r2, %r2, {40960+(strip//4)*8192+(strip%4)*32};\n'
            # Physical address permutation, not a layout check: verification
            # below compares each final global value with the independent CPU C.
            body+='shr.u32 %r3, %r2, 7; and.b32 %r3, %r3, 7; shl.b32 %r3, %r3, 4; xor.b32 %r2, %r2, %r3;\n'
            body+='stmatrix.sync.aligned.x4.m8n8.shared.b16 [%r2], {%r4,%r5,%r6,%r7};\n'
        body+='bar.sync 0; fence.proxy.async.shared::cta;\n'
        body+=f'''setp.eq.u32 %p1, %r0, 0; @!%p1 bra STORE_DONE{stage};
mov.u32 %r20, 40960; mov.u32 %r21, 49152; mov.u32 %r23, 0;
mov.u32 %r24, {stage*64};
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd12, {{%r23,%r24}}], [%r20];
mov.u32 %r23, 64;
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd12, {{%r23,%r24}}], [%r21];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group.read 0;
STORE_DONE{stage}:
bar.sync 0;
'''
        return body
    def reuse_input():
        return 'mul.lo.u32 %r1, %r0, 4; mov.u32 %r2, 0; st.shared.u32 [%r1], %r2; fence.proxy.async.shared::cta;\n'
    for name in ['pipeline','serial']:
        body='wgmma.fence.sync.aligned;\n'+load(0)
        if name=='pipeline': body+=load(1)
        body+=wait_input(0)+calculate(0,regs)
        if name=='pipeline':
            body+=wait_input(1)+calculate(1,ys)+'wgmma.wait_group.sync.aligned 1;\n'+reuse_input()+output(0,regs)
        else:
            body+='wgmma.wait_group.sync.aligned 0;\n'+reuse_input()+output(0,regs)+load(1)+wait_input(1)+calculate(1,ys)
        body+='wgmma.wait_group.sync.aligned 0;\n'+output(1,ys)
        text+=entry(name,body)
    # Keep 200 values live so ptxas reports the actual initial quota (168).
    body='''mov.f32 %f255, 0f42280000;
setp.lt.u32 %p1, %r0, 128; @%p1 bra PRODUCER;
setmaxnreg.inc.sync.aligned.u32 232; bra QUOTA_DONE;
PRODUCER:
mov.u32 %r2, 0;
DELAY:
add.u32 %r2, %r2, 1; setp.lt.u32 %p1, %r2, 200; @%p1 bra DELAY;
setmaxnreg.dec.sync.aligned.u32 40;
QUOTA_DONE:
mul.wide.u32 %rd4, %r0, 4; add.u64 %rd4, %rd1, %rd4;
st.global.f32 [%rd4], %f255;
'''
    body = body.replace('mov.f32 %f255,', 'mov.u64 %rd6, %rd0;\n'+''.join(
        f'ld.volatile.global.f32 %f{j}, [%rd6+{32768+j*4}];\n' for j in range(200))+'mov.f32 %f255,')
    body += 'setp.eq.u32 %p1, %r0, 0; @!%p1 bra HIGH_DONE;\n'
    body += ''.join(f'st.volatile.global.f32 [%rd1+{4096+j*4}], %f{j};\n' for j in range(200))
    body+='HIGH_DONE:\n'
    text+=entry('quota',body,384)
    import re
    frozen = (Path(__file__).resolve().parents[3]/'artifacts/ptx/df3e0c49e4650c1f.ptx').read_text()
    tuple_text = re.search(r'wgmma\.mma_async[^\n]*?\{([^}]+)\}', frozen).group(1)
    real_names = re.findall(r'%f\d+', tuple_text)
    source = re.search(r'\.visible \.entry test_single128\(.*?\n}\n', text, re.S).group(0)
    source = source.replace('test_single128', 'test_realtuple').replace('%f<256>', '%f<7501>')
    mapping = {f'%f{reg}':name for reg,name in zip(regs,real_names)}
    source = re.sub(r'%f\d+', lambda m:mapping.get(m[0],m[0]), source)
    text += source
    # Existing eight-register WMMA load/store path, affected by tuple parsing.
    text += entry('legacy8', '''
wmma.load.c.sync.aligned.row.m16n16k16.global.f32 {%f0,%f1,%f2,%f3,%f4,%f5,%f6,%f7}, [%rd0], 16;
wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [%rd1], {%f0,%f1,%f2,%f3,%f4,%f5,%f6,%f7}, 16;
''', 32)
    (run/'real-tuple.txt').write_text(tuple_text+'\n')
    (run/'micro.ptx').write_text(text)
