import gdb, json
from pathlib import Path
updates=[]
samples=[]
gdb.execute('set breakpoint pending on')
gdb.execute('set pagination off')
gdb.execute('set print thread-events off')
class Update(gdb.Breakpoint):
    def stop(self):
        f=gdb.newest_frame()
        gpu=f.read_var('gpu').dereference()
        updates.append(dict(total=int(gpu['gpu_tot_sim_cycle']),current=int(gpu['gpu_sim_cycle'])))
        return False
class Elapsed(gdb.Breakpoint):
    def stop(self):
        f=gdb.newest_frame()
        a=f.read_var('start').dereference();b=f.read_var('end').dereference()
        samples.append(dict(start=int(a['m_gpu_tot_sim_cycle']),end=int(b['m_gpu_tot_sim_cycle'])))
        Path('observed.json').write_text(json.dumps(dict(updates=updates,samples=samples),indent=2)+'\n')
        return False
Update('stream_manager.cc:218',internal=True)
Elapsed('cuda_runtime_api.cc:cudaEventElapsedTime',internal=True)
