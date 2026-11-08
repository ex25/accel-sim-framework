"""Read-only evidence for atomic cluster admission, shared lifetime and barriers."""
import gdb
import json
from pathlib import Path

gdb.execute("set pagination off")
gdb.execute("set breakpoint pending on")


def emit(kind, **values):
    print("OBS " + json.dumps(dict(kind=kind, **values), sort_keys=True))


def dim(v):
    return [int(v[x]) for x in ("x", "y", "z")]


def cycle(core):
    gpu = core["m_gpu"].cast(gdb.lookup_type("gpgpu_sim").pointer())
    return int(gpu["gpu_sim_cycle"]) + int(gpu["gpu_tot_sim_cycle"])


class After(gdb.FinishBreakpoint):
    def __init__(self, fn):
        super().__init__(gdb.newest_frame(), internal=True)
        self.fn = fn

    def stop(self):
        self.fn(self.return_value)
        return False


class At(gdb.Breakpoint):
    def __init__(self, spec, fn):
        super().__init__(spec, internal=True)
        self.fn = fn

    def stop(self):
        frame = gdb.newest_frame()
        if not (frame.name() or "").endswith("@plt"):
            self.fn(frame)
        return False


def create(f):
    kernel = f.read_var("this")
    uid = int(kernel["m_uid"])
    cta = dim(f.read_var("cta"))
    After(lambda result: emit("shared_create", uid=uid, cta=cta, shared=int(result)))


def retire(f):
    kernel = f.read_var("this")
    core = f.older().read_var("this")
    emit("shared_retire", uid=int(kernel["m_uid"]), cta=dim(f.read_var("cta")),
         sm=int(core["m_sid"]), active=int(core["m_n_active_cta"]), cycle=cycle(core))


def issue(f):
    core = f.read_var("this")
    kernel = f.read_var("kernel")
    uid = int(kernel["m_uid"])
    cta = dim(kernel["m_next_cta"])
    sm = int(core["m_sid"])
    entry = kernel["m_kernel_entry"].dereference()
    info = entry["m_kernel_info"]
    threads = dim(kernel["m_block_dim"])
    threads = threads[0] * threads[1] * threads[2]
    regs = int(info["regs"])
    shared = int(info["smem"]) + int(kernel["m_dynamic_shared_memory"]) + int(kernel["m_driver_shared_memory"])
    config = core["m_config"]
    concurrent = bool(config["gpgpu_concurrent_kernel_sm"])

    def done(_):
        active = int(core["m_n_active_cta"])
        emit("cta_issue", uid=uid, cta=cta, sm=sm, cycle=cycle(core), active=active,
             threads=threads, registers_per_thread=regs, sm_target=int(info["sm_target"]), shared_per_cta=shared,
             occupied_threads=int(core["m_occupied_n_threads"]) if concurrent else active*((threads+31)//32*32),
             occupied_registers=int(core["m_occupied_regs"]) if concurrent else active*((threads+31)//32*32)*((regs+3)&~3),
             occupied_shared=int(core["m_occupied_shmem"]) if concurrent else active*shared)
    After(done)


def arrive(f):
    kernel = f.read_var("this")
    uid, cluster = int(kernel["m_uid"]), dim(f.read_var("id"))
    epoch, count = int(f.read_var("epoch")), int(f.read_var("count"))
    now, latency = int(f.read_var("cycle")), int(f.read_var("latency"))
    emit("arrive_send", uid=uid, cluster=cluster, epoch=epoch, count=count,
         cycle=now, latency=latency)


def apply(f):
    kernel = f.read_var("this")
    event = f.read_var("event")
    item = f.read_var("item")
    emit("arrive_visible", uid=int(kernel["m_uid"]), cluster=dim(item["first"]),
         epoch=int(event["epoch"]), count=int(event["count"]),
         due=int(event["cycle"]), cycle=int(f.read_var("cycle")))


def release(f):
    kernel = f.read_var("this")
    # The caller's shd_warp has the current core and simulation clock.
    caller = f.older()
    warp = caller.read_var("this")
    core = warp["m_shader"]
    emit("wait_release", uid=int(kernel["m_uid"]), cluster=dim(f.read_var("id")),
         epoch=int(f.read_var("epoch")), count=int(f.read_var("count")),
         sm=int(core["m_sid"]), warp=int(warp["m_warp_id"]), cycle=cycle(core))


At("kernel_info_t::create_cluster_shared", create)
At("kernel_info_t::retire_cluster_cta", retire)
At("shader_core_ctx::issue_block2core", issue)
At("kernel_info_t::cluster_arrive", arrive)
At("kernel_info_t::cluster_wait_release", release)
src = Path(__file__).resolve().parents[5] / "gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc"
line = next(i for i, text in enumerate(src.read_text().splitlines(), 1)
            if "state.rounds[event.epoch].arrived += event.count;" in text)
At(f"{src}:{line}", apply)
gdb.execute("run")
