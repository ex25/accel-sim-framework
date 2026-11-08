"""PTX tests use real ABI objects, publication, TMA and mbarrier polling."""
import hashlib
import json
from pathlib import Path

PTX = Path(__file__).resolve().parents[3] / 'artifacts/ptx/df3e0c49e4650c1f.ptx'
SHARED = '_ZN7cutlass4smemE'


def expert_samples(run):
    source = PTX.read_text()
    lines = source.splitlines()
    provenance = {}

    def take(name, instructions):
        found = []
        for instruction in instructions:
            matches = [i + 1 for i, line in enumerate(lines)
                       if ' '.join(line.split()) == instruction]
            assert matches, instruction
            found.append(dict(line=matches[0], instruction=instruction))
        provenance[name] = found
        return '\n'.join(instructions)

    address64 = take('expert64', [
        f'mov.u32 %r1445, {SHARED};',
        'add.s32 %r1470, %r1445, 196608;',
        'add.s32 %r1475, %r1445, 213680;'])
    take('expert64_all_addresses', [
        'add.s32 %r1470, %r1445, 196608;',
        'add.s32 %r1483, %r1445, 200704;',
        'add.s32 %r1496, %r1445, 204800;',
        'add.s32 %r1509, %r1445, 208896;',
        'add.s32 %r1522, %r1445, 196608;',
        'add.s32 %r1535, %r1445, 200704;',
        'add.s32 %r1548, %r1445, 204800;',
        'add.s32 %r1561, %r1445, 208896;'])
    take('expert_output_all_addresses', [
        'add.s32 %r932, %r648, 196608;',
        'add.s32 %r988, %r648, 200704;',
        'add.s32 %r1046, %r648, 204800;',
        'add.s32 %r1104, %r648, 208896;',
        'add.s32 %r1162, %r648, 196608;',
        'add.s32 %r1220, %r648, 200704;',
        'add.s32 %r1278, %r648, 204800;',
        'add.s32 %r1336, %r648, 208896;'])
    address128 = take('expert128', [
        'mov.u32 %r321, %cluster_ctarank;',
        f'mov.u32 %r1650, {SHARED};',
        'shl.b32 %r218, %r321, 12;',
        'shl.b32 %r1665, %r2019, 3;',
        'add.s32 %r1667, %r1650, %r1665;',
        'add.s32 %r1674, %r1667, 213632;',
        'shl.b32 %r1692, %r2019, 14;',
        'add.s32 %r1693, %r1692, %r218;',
        'shl.b32 %r1694, %r1693, 1;',
        'add.s32 %r1673, %r1650, %r1694;',
        'add.s32 %r1677, %r1673, 16384;',
        'add.s32 %r1687, %r2016, 64;'])
    address128_64 = take('expert128_64', [
        f'mov.u32 %r1650, {SHARED};',
        'shl.b32 %r1665, %r2019, 3;',
        'add.s32 %r1667, %r1650, %r1665;',
        'add.s32 %r1674, %r1667, 213632;',
        'shl.b32 %r1695, %r2019, 15;',
        'add.s32 %r1696, %r1650, %r1695;',
        'add.s32 %r1681, %r1696, 98304;',
        'add.s32 %r1685, %r1696, 114688;',
        'add.s32 %r1687, %r2016, 64;'])
    load64 = take('load64', [
        'mov.u64 %rd507, 1152921504606846976;',
        'cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%r1470], [%rd87, {%r206, %r205}], [%r1475], %rd507;'])
    load128 = take('load128', [
        'mov.u16 %rs25, 3;',
        'mov.u64 %rd679, 1152921504606846976;',
        'cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%r1673], [%rd99, {%r2016, %r230}], [%r1674], %rs25, %rd679;',
        'cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%r1677], [%rd99, {%r1687, %r230}], [%r1674], %rs25, %rd679;'])
    load128_64 = take('load128_64', [
        'mov.u64 %rd679, 1152921504606846976;',
        'cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%r1681], [%rd660, {%r2016, %r229}], [%r1674], %rd679;',
        'cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%r1685], [%rd660, {%r1687, %r229}], [%r1674], %rd679;'])
    declaration = f'.extern .shared .align 16 .b8 {SHARED}[];'
    assert declaration in lines
    sites = [dict(line=i + 1, instruction=line.strip())
             for i, line in enumerate(lines) if 'cp.async.bulk.tensor.2d' in line]
    site_classes = {
        'shared_to_global': sites[:8],
        'expert64_global_to_shared': sites[8:16],
        'expert128_multicast': sites[16:18],
        'expert128_global_to_shared': sites[18:20],
    }
    assert len(sites) == 20
    assert all('global.shared::cta.bulk_group' in x['instruction']
               for x in site_classes['shared_to_global'])
    assert all('multicast::cluster' in x['instruction']
               for x in site_classes['expert128_multicast'])
    (run / 'expert-address-source.json').write_text(json.dumps(dict(
        source=str(PTX), sha256=hashlib.sha256(source.encode()).hexdigest(),
        shared_declaration=declaration, fragments=provenance,
        site_classes=site_classes,
        formulas=dict(input64='shared + [196608, 200704, 204800, 208896] (twice)',
                      multicast128='shared + 32768 * stage + 8192 * rank; second tile +16384',
                      input128='shared + 98304 + 32768 * stage; second tile +16384',
                      output_shared=['shared + 196608', 'shared + 200704',
                                     'shared + 204800', 'shared + 208896']),
        expert64_starts=[196608, 200704, 204800, 208896] * 2,
        stage_values=[0, 1, 2], rank_values=[0, 1], half_values=[0, 1],
        tma_start_phase='All input and output starts are 1KiB-aligned when the shared symbol is 0; both 64B/128B start phases are 0.'),
        indent=2) + '\n')
    return {
        'expert64': (address64, load64, '''
.reg .u32 %r1445, %r1470, %r1475, %r206, %r205;
.reg .u64 %rd507, %rd87;''', '''
mov.u32 %r7, %r1470;
mov.u32 %r6, %r1475;
mov.u32 %r24, %r1445;
mov.u32 %r17, %r1470;
mov.u32 %r18, %r1470;''', '''
mov.u64 %rd87, %rd1;
mov.u32 %r206, %r10;
mov.u32 %r205, %r11;'''),
        'expert128': (address128, load128, '''
.reg .u32 %r321, %r1650, %r218, %r1665, %r1667, %r1674;
.reg .u32 %r1692, %r1693, %r1694, %r1673, %r1677, %r1687;
.reg .u32 %r2019, %r2016, %r230;
.reg .u64 %rd679, %rd99;
.reg .b16 %rs25;''', '''
mov.u32 %r7, %r1673;
// Rank 1 issues multicast: both destinations receive rank 1's address.
setp.eq.u32 %p6, %r321, 0;
@%p6 add.u32 %r7, %r7, 8192;
mov.u32 %r6, %r1674;
mov.u32 %r24, %r1650;
mov.u32 %r17, %r1673;
mov.u32 %r18, %r7;''', '''
mov.u64 %rd99, %rd1;
mov.u32 %r2016, %r10;
mov.u32 %r230, %r11;'''),
        'expert128_64': (address128_64, load128_64, '''
.reg .u32 %r1650, %r1665, %r1667, %r1674, %r1695, %r1696;
.reg .u32 %r1681, %r1685, %r1687, %r2019, %r2016, %r229;
.reg .u64 %rd679, %rd660;''', '''
mov.u32 %r7, %r1681;
mov.u32 %r6, %r1674;
mov.u32 %r24, %r1650;
mov.u32 %r17, %r1681;
mov.u32 %r18, %r1681;''', '''
mov.u64 %rd660, %rd1;
mov.u32 %r2016, %r10;
mov.u32 %r229, %r11;''')
    }, declaration


def expert_entry(name, address, loads, declarations, aliases, operands, shared):
    transfer_bytes = 4096 if name == 'expert64' else 32768
    dump_bytes = transfer_bytes + 256
    output_bytes = dump_bytes + 64
    cluster = name != 'expert64'
    stage_setup = '' if name == 'expert64' else 'mov.u32 %r2019, %r25;\n'
    if name == 'expert128':
        stage_setup += 'mov.u32 %r2016, %r10;\nmov.u32 %r230, %r11;\n'
    elif name == 'expert128_64':
        stage_setup += 'mov.u32 %r2016, %r10;\nmov.u32 %r229, %r11;\n'
    predicate = '%p0'
    if name == 'expert128':
        predicate = '%p6'
    load_code = '\n'.join('@' + predicate + ' ' + line for line in loads.splitlines())
    initial_copy = '\n'.join(
        f'@%p0 ld.param.u32 %r5, [initial+{offset}];\n'
        f'@%p0 mov.u32 %r4, {offset};\n'
        '@%p0 st.shared.u32 [%r4], %r5;'
        for offset in range(0, 128, 4)
    )
    cluster_barrier = (
        'barrier.cluster.arrive.relaxed.aligned;\n'
        'barrier.cluster.wait.aligned;\n'
        if cluster else ''
    )
    multicast_predicate = (
        'setp.eq.u32 %p6, %r28, 1;\nand.pred %p6, %p6, %p0;\n'
        if name == 'expert128' else ''
    )
    return f'''.visible .entry {name}(
.param .u64 result, .param .align 64 .b8 initial[128],
.param .u64 published, .param .u64 base,
.param .u32 x, .param .u32 y, .param .u32 stage
) {{
.reg .u32 %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8;
.reg .u32 %r10, %r11, %r12, %r14, %r15, %r16, %r17, %r18, %r19;
.reg .u32 %r24, %r25, %r26, %r27, %r28;
.reg .u64 %rd0, %rd1, %rd2, %rd3, %rd4, %rd5;
.reg .pred %p0, %p1, %p2, %p3, %p4, %p5, %p6;
{declarations}
ld.param.u64 %rd0, [result];
ld.param.u64 %rd1, [published];
ld.param.u64 %rd2, [base];
ld.param.u32 %r10, [x];
ld.param.u32 %r11, [y];
ld.param.u32 %r25, [stage];
mov.u32 %r0, %tid.x;
mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %ctaid.y;
mad.lo.u32 %r3, %r2, 1, %r1;
mul.wide.u32 %rd3, %r3, {output_bytes};
add.u64 %rd0, %rd0, %rd3;
mul.wide.u32 %rd3, %r3, 128;
add.u64 %rd1, %rd1, %rd3;
setp.eq.u32 %p0, %r0, 0;
mov.u32 %r28, %cluster_ctarank;
mov.u32 %r24, {shared};
{initial_copy}
bar.sync 0;
@!%p0 bra descriptor_ready;
mov.u32 %r8, 0;
tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%r8], %rd2;
tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%r8], 0, 128;
tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%r8], 1, 160;
tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%r8], 2, 1;
tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%r8], 3, 1;
tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%r8], 4, 1;
mov.u64 %rd4, 320;
tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%r8], 0, %rd4;
descriptor_ready:
bar.sync 0;
mov.u32 %r8, 0;
tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%rd1], [%r8], 128;
fence.proxy.tensormap::generic.acquire.gpu [%rd1], 128;
{stage_setup}{address}
{aliases}{operands}
sub.u32 %r19, %r18, 128;
shl.b32 %r14, %r0, 2;
mov.u32 %r15, 2779096485;
fill_tile_guard:
add.u32 %r27, %r19, %r14;
st.shared.u32 [%r27], %r15;
add.u32 %r14, %r14, 128;
setp.lt.u32 %p1, %r14, {dump_bytes};
@%p1 bra fill_tile_guard;
bar.sync 0;
@%p0 mbarrier.init.shared::cta.b64 [%r6], 1;
@%p0 fence.mbarrier_init.release.cluster;
bar.sync 0;
{cluster_barrier}{multicast_predicate}
@%p0 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r6], {transfer_bytes};
{cluster_barrier}{load_code}
{cluster_barrier}mov.u32 %r26, 0;
wait_tma:
mbarrier.try_wait.parity.shared::cta.b64 %p2, [%r6], %r26;
@!%p2 bra wait_tma;
bar.sync 0;
shl.b32 %r14, %r0, 2;
dump_tile:
add.u32 %r27, %r19, %r14;
ld.shared.u32 %r15, [%r27];
cvt.u64.u32 %rd4, %r14;
add.u64 %rd4, %rd0, %rd4;
st.global.u32 [%rd4], %r15;
add.u32 %r14, %r14, 128;
setp.lt.u32 %p1, %r14, {dump_bytes};
@%p1 bra dump_tile;
@%p0 add.u64 %rd4, %rd0, {dump_bytes};
@%p0 st.global.u32 [%rd4], %r24;
@%p0 add.u64 %rd4, %rd4, 4;
@%p0 st.global.u32 [%rd4], %r17;
@%p0 add.u64 %rd4, %rd4, 4;
@%p0 st.global.u32 [%rd4], %r18;
@%p0 add.u64 %rd4, %rd4, 4;
@%p0 st.global.u32 [%rd4], %r6;
@%p0 add.u64 %rd4, %rd4, 4;
@%p0 st.global.u32 [%rd4], %r28;
@%p0 add.u64 %rd4, %rd4, 4;
@%p0 st.global.u32 [%rd4], %r25;
@%p0 add.u64 %rd4, %rd4, 4;
@%p0 st.global.u32 [%rd4], %r10;
@%p0 add.u64 %rd4, %rd4, 4;
@%p0 st.global.u32 [%rd4], %r11;
@%p0 add.u64 %rd4, %rd4, 4;
@%p0 st.global.u32 [%rd4], {transfer_bytes};
ret;
}}
'''


def generate(run):
    entries = []
    for name in ('input', 'output', 'multicast', 'rotate'):
        common = '''
.reg .u32 %r<32>;
.reg .u64 %rd<12>;
.reg .b16 %h, %mask;
.reg .pred %p<8>;
ld.param.u64 %rd0, [result];
ld.param.u64 %rd1, [published];
ld.param.u64 %rd2, [base];
ld.param.u64 %rd4, [destmap];
ld.param.u32 %r10, [x];
ld.param.u32 %r11, [y];
ld.param.u32 %r12, [bytes];
ld.param.u32 %r13, [mask];
cvt.u16.u32 %mask, %r13;
mov.u32 %r0, %tid.x;
mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %ctaid.y;
mad.lo.u32 %r3, %r2, 2, %r1;
mul.wide.u32 %rd3, %r3, 128;
add.u64 %rd1, %rd1, %rd3;
mul.wide.u32 %rd3, %r3, 32768;
add.u64 %rd0, %rd0, %rd3;
setp.eq.u32 %p0, %r0, 0;
@!%p0 bra param_done;
mov.u32 %r4, 0;
PARAM_COPY
param_done:
mov.u32 %r6, 1024;
mov.u32 %r7, 1280;
setp.eq.u32 %p0, %r0, 0;
@%p0 mbarrier.init.shared::cta.b64 [%r6], 1;
@%p0 fence.mbarrier_init.release.cluster;
bar.sync 0;
@!%p0 bra edited;
mov.u32 %r8, 0;
tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%r8], %rd2;
tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%r8], 0, 128;
tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%r8], 1, 160;
tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%r8], 2, 1;
tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%r8], 3, 1;
tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%r8], 4, 1;
mov.u64 %rd6, 320;
tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%r8], 0, %rd6;
mov.u64 %rd6, 160;
tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%r8], 1, %rd6;
tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%r8], 2, %rd6;
tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%r8], 3, %rd6;
edited:
bar.sync 0;
mov.u32 %r8, 0;
tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%rd1], [%r8], 128;
fence.proxy.tensormap::generic.acquire.gpu [%rd1], 128;
mov.u32 %r14, %r0;
fill:
mad.lo.u32 %r15, %r14, 2, %r7;
mov.b16 %h, 43981;
st.shared.b16 [%r15], %h;
add.u32 %r14, %r14, 32;
shr.u32 %r16, %r12, 1;
setp.lt.u32 %p1, %r14, %r16;
@%p1 bra fill;
bar.sync 0;
'''
        common = common.replace('PARAM_COPY', '\n'.join(
            f'ld.param.u32 %r5, [initial+{i}]; st.shared.u32 [%r4+{i}], %r5;'
            for i in range(0,128,4)))
        if name == 'rotate':
            common += '''
// Keep an independent B descriptor, then update the original after issue.
mul.lo.u32 %r4, %r0, 4;
ld.shared.u32 %r5, [%r4];
add.u32 %r4, %r4, 256;
st.shared.u32 [%r4], %r5;
bar.sync 0;
add.u64 %rd5, %rd1, 128;
mov.u32 %r8, 256;
tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%rd5], [%r8], 128;
fence.proxy.tensormap::generic.acquire.gpu [%rd5], 128;
mov.u64 %rd7, 0;
@%p0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%r7], [%rd1, {%r10, %r11}], [%r6], %rd7;
@%p0 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r6], %r12;
ld.param.u64 %rd11, [initial];
mov.u32 %r8, 0;
@%p0 tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%r8], %rd11;
bar.sync 0;
tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%rd1], [%r8], 128;
fence.proxy.tensormap::generic.acquire.gpu [%rd1], 128;
@!%p0 bra transfer_done;
mov.u32 %r20, 1056;
mov.u32 %r21, 9472;
mbarrier.init.shared::cta.b64 [%r20], 1;
fence.mbarrier_init.release.cluster;
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%r21], [%rd1, {%r10, %r11}], [%r20], %rd7;
mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r20], %r12;
mov.u32 %r22, 1088;
mov.u32 %r23, 17664;
mbarrier.init.shared::cta.b64 [%r22], 1;
fence.mbarrier_init.release.cluster;
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%r23], [%rd5, {%r10, %r11}], [%r22], %rd7;
mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r22], %r12;
wait_second:
mbarrier.try_wait.parity.shared::cta.b64 %p2, [%r20], 0;
@!%p2 bra wait_second;
wait_third:
mbarrier.try_wait.parity.shared::cta.b64 %p2, [%r22], 0;
@!%p2 bra wait_third;
'''
        elif name == 'multicast':
            common += '''barrier.cluster.arrive.relaxed.aligned;
barrier.cluster.wait.aligned;
@!%p0 bra transfer_done;
mov.u32 %r18, 1;
shl.b32 %r18, %r18, %r2;
and.b32 %r18, %r18, %r13;
setp.ne.u32 %p4, %r18, 0;
@%p4 mbarrier.expect_tx.shared::cta.b64 [%r6], %r12;
@%p4 mbarrier.arrive.shared::cta.b64 _, [%r6];
setp.eq.u32 %p5, %r2, 0;
mov.u64 %rd7, 0;
@%p5 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%r7], [%rd1, {%r10, %r11}], [%r6], %mask, %rd7;
@!%p4 bra transfer_done;
'''
        elif name == 'output':
            common += 'bra transfer_done;\n'
        else:
            common += '''@!%p0 bra transfer_done;
mov.u64 %rd7, 0;
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%r7], [%rd1, {%r10, %r11}], [%r6], %rd7;
mbarrier.expect_tx.shared::cta.b64 [%r6], %r12;
mbarrier.arrive.shared::cta.b64 _, [%r6];
'''
        body = '''poll:
mbarrier.try_wait.parity.shared::cta.b64 %p2, [%r6], 0;
@!%p2 bra poll;
transfer_done:
bar.sync 0;
'''
        if name == 'output':
            body += '''ld.param.u64 %rd10, [prepared];
mov.u32 %r14, %r0;
prepare:
mul.wide.u32 %rd8, %r14, 2;
add.u64 %rd8, %rd10, %rd8;
ld.global.b16 %h, [%rd8];
mad.lo.u32 %r15, %r14, 2, %r7;
st.shared.b16 [%r15], %h;
add.u32 %r14, %r14, 32;
setp.lt.u32 %p1, %r14, %r16;
@%p1 bra prepare;
bar.sync 0;
fence.proxy.async.shared::cta;
@%p0 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd4, {%r10, %r11}], [%r7];
@%p0 cp.async.bulk.commit_group;
@%p0 cp.async.bulk.wait_group.read 0;
bar.sync 0;
// Source may now be reused while global acknowledgements are still pending.
mov.u32 %r14, %r0;
overwrite:
mad.lo.u32 %r15, %r14, 2, %r7;
mov.b16 %h, 12345;
st.shared.b16 [%r15], %h;
add.u32 %r14, %r14, 32;
setp.lt.u32 %p1, %r14, %r16;
@%p1 bra overwrite;
bar.sync 0;
fence.proxy.async.shared::cta;
add.u64 %rd4, %rd4, 128;
@%p0 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd4, {%r10, %r11}], [%r7];
@%p0 cp.async.bulk.commit_group;
@%p0 cp.async.bulk.wait_group.read 1;
@%p0 cp.async.bulk.wait_group.read 0;
bar.sync 0;
'''
        if name == 'rotate':
            body += 'mov.u32 %r16, 10240;\n'
        body += '''mov.u32 %r14, %r0;
dump:
mad.lo.u32 %r15, %r14, 2, %r7;
ld.shared.b16 %h, [%r15];
mul.wide.u32 %rd8, %r14, 2;
add.u64 %rd8, %rd0, %rd8;
st.global.b16 [%rd8], %h;
add.u32 %r14, %r14, 32;
setp.lt.u32 %p1, %r14, %r16;
@%p1 bra dump;
'''
        if name == 'multicast':
            body += 'barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;\n'
        entries.append('.visible .entry '+name+'''(
.param .u64 result, .param .align 64 .b8 initial[128],
.param .u64 published, .param .u64 base, .param .u64 destmap,
.param .u32 x, .param .u32 y, .param .u32 bytes, .param .u32 mask,
.param .u64 prepared
) {\n'''+common+body+'ret;\n}\n')
    expert_entries, shared_declaration = expert_samples(run)
    for name, (address, loads, declarations, aliases, operands) in expert_entries.items():
        entries.append(expert_entry(name, address, loads, declarations, aliases,
                                    operands, '_ZN7cutlass4smemE'))
    (run/'micro.ptx').write_text(
        '// Stage 5\n.version 8.7\n.target sm_90a\n.address_size 64\n'
        + shared_declaration + '\n' + ''.join(entries))
