"""Read-only GDB observations for stage 0; run with -x and --args stage0 TEST.

Uses the debug build's private state, without inferior calls or state changes.
The checked source revision and line locations belong in the run report.
"""
import gdb
import json

gdb.execute('set pagination off')
gdb.execute('set breakpoint pending on')


def emit(kind, **data):
    print('OBS ' + json.dumps(dict(kind=kind, **data), sort_keys=True))


def vec(v):
    impl = v['_M_impl']
    return [(impl['_M_start'] + i).dereference()
            for i in range(int(impl['_M_finish'] - impl['_M_start']))]


def name(v):
    return v['_M_dataplus']['_M_p'].string()


def stack_size(stack):
    return int(stack['_M_impl']['_M_node']['_M_size'])


def stack_back(stack):
    node = stack['_M_impl']['_M_node']['_M_prev'].cast(
        gdb.lookup_type('std::_List_node<kernel_config>').pointer())
    return node['_M_storage'].address.cast(
        gdb.lookup_type('kernel_config').pointer()).dereference()


def dim(v):
    return [int(v[x]) for x in ('x', 'y', 'z')]


class After(gdb.FinishBreakpoint):
    def __init__(self, callback):
        super().__init__(gdb.newest_frame(), internal=True)
        self.callback = callback

    def stop(self):
        self.callback(self.return_value)
        return False


class At(gdb.Breakpoint):
    def __init__(self, spec, callback):
        super().__init__(spec, internal=True)
        self.callback = callback

    def stop(self):
        frame = gdb.newest_frame()
        if '@plt' not in (frame.name() or ''):
            self.callback(frame)
        return False


def launch(f):
    ctx = f.read_var('ctx')
    entry = f.read_var('entry')
    stack = ctx['api']['g_cuda_launch_stack']
    before = stack_size(stack)
    emit('launch', entry=name(entry['m_name']), depth=before,
         grid=dim(f.read_var('gridDim')), block=dim(f.read_var('blockDim')),
         shared=int(f.read_var('sharedMem')), stream=int(f.read_var('stream')),
         regs=int(entry['m_kernel_info']['regs']),
         base=int(entry['m_kernel_info']['smem']))
    After(lambda result: emit('launch_return', depth=stack_size(stack),
                             before=before, result=int(result)))


def pop(f):
    stack = f.read_var('stack')
    top = stack_back(stack)
    expected = dict(grid=dim(top['m_GridDim']), block=dim(top['m_BlockDim']),
                    shared=int(top['m_sharedMem']), stream=int(top['m_stream']))
    grid, block = f.read_var('gridDim'), f.read_var('blockDim')
    shared, stream = f.read_var('sharedMem'), f.read_var('stream')
    stream = stream.cast(gdb.lookup_type('CUstream_st').pointer().pointer())
    before = stack_size(stack)

    def done(result):
        actual = dict(grid=dim(grid.dereference()), block=dim(block.dereference()),
                      shared=int(shared.dereference()),
                      stream=int(stream.dereference()))
        emit('pop', expected=expected, actual=actual,
             before=before, after=stack_size(stack), result=int(result))
    After(done)


def finalize(f):
    entry = f.read_var('this')

    def done(_):
        configs = [[int(p['first']), int(p['second'])]
                   for p in vec(entry['m_param_configs'])]
        symbols = [dict(name=name(p['m_name']), address=int(p['m_address']),
                        alignment=int(p['m_type']['m_type_info']['m_alignment_spec']))
                   for p in vec(entry['m_args'])]
        emit('parameters', entry=name(entry['m_name']),
             configs=configs, symbols=symbols)
    After(done)


def resource(f, release=False):
    core, kernel = f.read_var('this'), f.read_var('k')
    before = int(core['m_occupied_shmem'])
    base = int(kernel['m_kernel_entry']['m_kernel_info']['smem'])
    dynamic = int(kernel['m_dynamic_shared_memory'])
    occupy = True if release else bool(f.read_var('occupy'))
    # Failed admission probes can occur every cycle; record only mutations.
    if not occupy:
        return

    def done(result):
        after = int(core['m_occupied_shmem'])
        emit('release' if release else 'occupy', before=before, after=after,
             base=base, dynamic=dynamic, ctas=int(core['m_occupied_ctas']),
             accepted=True if release else bool(result))
    After(done)


def max_cta(f):
    kernel, config = f.read_var('k'), f.read_var('this')
    if bool(kernel['cache_config_set']):
        return

    def done(result):
        entry = kernel['m_kernel_entry']
        cache = config['m_L1D_config']
        emit('max_cta', entry=name(entry['m_name']),
             base=int(entry['m_kernel_info']['smem']),
             dynamic=int(kernel['m_dynamic_shared_memory']),
             limit=int(entry['m_max_dynamic_shared_memory']),
             capacity=int(config['gpgpu_shmem_size']), result=int(result),
             l1_kib=int(cache['m_assoc'] * cache['m_nset'] * cache['m_line_sz']) // 1024)
    After(done)


def ptxas(f):
    emit('ptxas', command=f.read_var('buff').string())


def kernel_done(f):
    kernel, gpu = f.read_var('kernel'), f.read_var('this')
    emit('device_interval', entry=name(kernel['m_kernel_entry']['m_name']),
         uid=int(kernel['m_uid']), start=int(kernel['start_cycle']),
         end=int(gpu['gpu_tot_sim_cycle']) + int(gpu['gpu_sim_cycle']))


At('cuda_runtime_api.cc:2083', launch)
At('cuda_runtime_api.cc:3978', pop)
At('function_info::finalize', finalize)
At('gpu-sim.cc:2004', resource)
At('gpu-sim.cc:2040', lambda f: resource(f, True))
At('shader.cc:4479', max_cta)
At('ptx_loader.cc:377', ptxas)
At('gpgpu_sim::set_kernel_done', kernel_done)
gdb.execute('run')
