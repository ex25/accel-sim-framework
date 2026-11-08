"""Read-only event evidence; no injected arrivals or memory completion."""
import gdb
import json
from pathlib import Path

gdb.execute('set pagination off')
gdb.execute('set breakpoint pending on')
ROOT = Path(__file__).resolve().parents[5]


def emit(kind, **kw):
    print('OBS '+json.dumps(dict(kind=kind, **kw), sort_keys=True))


def dim(v):
    return [int(v[x]) for x in ('x','y','z')]


def cycle(core):
    gpu=core['m_gpu'].cast(gdb.lookup_type('gpgpu_sim').pointer())
    return int(gpu['gpu_sim_cycle'])+int(gpu['gpu_tot_sim_cycle'])


def caller(name):
    f=gdb.newest_frame()
    while f and name not in (f.name() or ''):
        f=f.older()
    return f


def identity(f):
    core=f.read_var('core').cast(gdb.lookup_type('shader_core_ctx').pointer())
    thread=f.read_var('thread')
    kernel=thread['m_kernel']
    return dict(uid=int(kernel['m_uid']), cta=dim(thread['m_ctaid']),
                sm=int(core['m_sid']), cycle=cycle(core))


def snapshot(b):
    return dict(pending=int(b['pending_thread_count']),tx=int(b['tx_count']),
                phase=int(b['phase']))


class After(gdb.FinishBreakpoint):
    def __init__(self, fn):
        super().__init__(gdb.newest_frame(),internal=True)
        self.fn=fn
    def stop(self):
        self.fn()
        return False


class At(gdb.Breakpoint):
    def __init__(self,spec,fn):
        super().__init__(spec,internal=True)
        self.fn=fn
    def stop(self):
        f=gdb.newest_frame()
        if not (f.name() or '').endswith('@plt'):
            self.fn(f)
        return False


def transition(f):
    b=f.read_var('this')
    before=snapshot(b)
    service=caller('mbarrier_complete')
    if service:
        meta=identity(service)
        op=int(service.read_var('inst')['m_syncs_op'])
    else:
        delivery=caller('kernel_info_t::service_cluster_arrivals')
        if not delivery: return
        meta=dict(uid=int(delivery.read_var('this')['m_uid']),
                  cycle=int(delivery.read_var('cycle')))
        op='remote-arrive'
    meta.update(cluster=dim(b['cluster_cta_identifier']['cluster_id']),
                rank=int(b['cluster_cta_identifier']['cluster_rank']),
                address=int(b['bar_addr']))
    After(lambda:emit('state', **meta, op=op, before=before, after=snapshot(b)))


def initialize(f):
    service=caller('mbarrier_complete')
    if not service: return
    b=f.read_var('this')
    meta=identity(service)
    def done():
        emit('init', **meta, rank=int(b['cluster_cta_identifier']['cluster_rank']),
             address=int(b['bar_addr']), **snapshot(b))
    After(done)


def send(f):
    core=caller('mbarrier_complete').read_var('core').cast(gdb.lookup_type('shader_core_ctx').pointer())
    emit('remote_send', uid=int(f.read_var('this')['m_uid']),
         source=dim(f.read_var('source')),address=int(f.read_var('address')),
         count=int(f.read_var('count')),cycle=int(f.read_var('cycle')),
         latency=int(f.read_var('latency')),sm=int(core['m_sid']))


def predicate(f):
    b=f.read_var('barrier')
    meta=identity(f)
    operand=f.read_var('operand');lane=int(f.read_var('lane'))
    inst=f.read_var('inst')
    emit('query',**meta,rank=int(b['cluster_cta_identifier']['cluster_rank']),
         address=int(b['bar_addr']), lane=lane, pc=int(inst['pc']),
         warp=int(inst['m_warp_id']),
         parity=int(operand['u']['wait']['phase'][lane]),
         predicate_encoding=int(f.read_var('result')['pred']),**snapshot(b))


def finish(f):
    inst=f.read_var('inst')
    if bool(inst['m_ptx_mbarrier']):
        emit('writeback', uid=int(inst['m_uid']), warp=int(inst['m_warp_id']),
             sm=int(f.read_var('this')['m_sid']),pc=int(inst['pc']),
             cycle=cycle(f.read_var('this')['m_core']))


At('mbarrier_t::mbarrier_t',initialize)
At('mbarrier_t::arrive_on',transition)
At('mbarrier_t::arrive_expect_on',transition)
At('mbarrier_t::expect_on',transition)
At('mbarrier_t::complete_on',transition)
At('kernel_info_t::cluster_mbarrier_arrive',send)
# This line runs after the result is computed and before set_operand_value.
src=ROOT/'gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc'
line=next(i for i,t in enumerate(src.read_text().splitlines(),1)
          if 'thread->set_operand_value(p->dst(), result, PRED_TYPE' in t)
At(f'{src}:{line}',predicate)
# Use the completion call site, after collector writeback succeeded.
src2=ROOT/'gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc'
line2=next(i for i,t in enumerate(src2.read_text().splitlines(),1)
           if 'if (inst.m_ptx_mbarrier) m_core->complete_ptx_mbarrier' in t)
At(f'{src2}:{line2}',finish)
gdb.execute('run')
