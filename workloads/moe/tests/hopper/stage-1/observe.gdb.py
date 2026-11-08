"""Read-only decode, collective, issue/completion and warp-memory-order evidence."""

import gdb
import json
import os
from pathlib import Path

gdb.execute("set pagination off")
gdb.execute("set breakpoint pending on")
ROOT = Path("/home/wangjz/in-switch/accel-sim")
SIM = ROOT / "gpu-simulator/gpgpu-sim"


def emit(kind, **data):
    print("OBS " + json.dumps(dict(kind=kind, **data), sort_keys=True))


def cycle(core):
    gpu = core["m_gpu"].cast(gdb.lookup_type("gpgpu_sim").pointer())
    return int(gpu["gpu_sim_cycle"]) + int(gpu["gpu_tot_sim_cycle"])


def mask(v):
    word = v["_M_w"]
    return int(word[0]) if word.type.code == gdb.TYPE_CODE_ARRAY else int(word)


def array(v):
    return [int(v[i]) for i in range(v.type.range()[1] + 1)]


def vec(v):
    x = v["_M_impl"]
    return [
        (x["_M_start"] + i).dereference()
        for i in range(int(x["_M_finish"] - x["_M_start"]))
    ]


class At(gdb.Breakpoint):
    def __init__(self, spec, callback):
        super().__init__(spec, internal=True)
        self.callback = callback

    def stop(self):
        f = gdb.newest_frame()
        if "@plt" not in (f.name() or ""):
            self.callback(f)
        return False


class After(gdb.FinishBreakpoint):
    def __init__(self, callback):
        super().__init__(gdb.newest_frame(), internal=True)
        self.callback = callback

    def stop(self):
        self.callback(self.return_value)
        return False


interesting = {}
seen = set()


def decode(f):
    p = f.read_var("this")
    opcode = int(p["m_opcode"])
    name = gdb.parse_and_eval("g_opcode_string")[opcode].string()
    if name not in [
        "shfl",
        "vote",
        "redux",
        "elect",
        "bar",
        "cvt",
        "bfind",
        "ld",
        "st",
    ]:
        return

    def done(_):
        pc = int(p["pc"])
        interesting[pc] = name
        emit(
            "decode",
            pc=pc,
            opcode=name,
            unit=int(p["op"]),
            latency=int(p["latency"]),
            initiation=int(p["initiation_interval"]),
            sources=array(p["in"]),
            destinations=array(p["out"]),
        )

    After(done)


def collective(f):
    w = f.read_var("w")
    p = f.read_var("p")
    active = mask(w["m_warp_active_mask"])
    lane = (active & -active).bit_length() - 1
    masks = f.read_var("masks")
    members = int(masks[lane])
    key = (int(w["pc"]), active, members)
    if key in seen:
        return
    seen.add(key)
    emit(
        "collective",
        pc=int(w["pc"]),
        warp=int(w["m_warp_id"]),
        lane=lane,
        mask=members,
        arrived=active,
        inputs=array(f.read_var("values")),
        destinations=array(p["out"]),
        cycle=cycle(f.read_var("core")),
    )


def issue(f):
    inst = f.read_var("next_inst")
    pc = int(inst["pc"])
    if pc in interesting:
        emit(
            "issue",
            pc=pc,
            opcode=interesting[pc],
            warp=int(f.read_var("warp_id")),
            cycle=cycle(f.read_var("this")),
        )


def complete(f):
    inst = f.read_var("inst")
    pc = int(inst["pc"])
    if pc in interesting:
        emit(
            "complete",
            pc=pc,
            opcode=interesting[pc],
            warp=int(inst["m_warp_id"]),
            cycle=cycle(f.read_var("this")),
        )


def membar(f):
    core = f.read_var("this")
    wid = int(f.read_var("warp_id"))
    warp = vec(core["m_warp"])[wid]
    if not bool(warp["m_membar"]) or not bool(warp["m_warp_mem_barrier"]):
        return
    state = (int(warp["m_inst_in_pipeline"]), int(warp["m_stores_outstanding"]))
    key = ("wait", wid, state)
    if key not in seen:
        seen.add(key)
        emit(
            "warp_barrier_wait",
            warp=wid,
            pipeline=state[0],
            stores=state[1],
            cycle=cycle(core),
        )

    def done(result):
        if not bool(result):
            emit("warp_barrier_release", warp=wid, cycle=cycle(core))

    After(done)


def kernel_done(f):
    kernel = f.read_var("kernel")
    gpu = f.read_var("this")
    emit(
        "device_interval",
        start=int(kernel["start_cycle"]),
        end=int(gpu["gpu_tot_sim_cycle"]) + int(gpu["gpu_sim_cycle"]),
    )


if os.environ.get("STAGE1_OBSERVE_KERNELS"):
    At("gpgpu_sim::set_kernel_done", kernel_done)
elif os.environ.get("STAGE1_OBSERVE_TIMING"):
    At("ptx_instruction::pre_decode", decode)
    count = [0]

    def accept(f):
        unit = f.read_var("this")
        inst = f.read_var("ready_reg").dereference().dereference()
        pc = int(inst["pc"])
        if interesting.get(pc) != "shfl":
            return
        emit(
            "unit_accept",
            pc=pc,
            unit=int(unit),
            warp=int(inst["m_warp_id"]),
            interval=int(inst["initiation_interval"]),
            cycle=cycle(unit["m_core"]),
        )
        count[0] += 1
        if count[0] == 64:
            accept_bp.enabled = False

    src = (SIM / "src/gpgpu-sim/shader.cc").read_text().splitlines()
    start = next(
        i for i, s in enumerate(src) if s.startswith("void pipelined_simd_unit::issue(")
    )
    line = next(
        i + 1
        for i in range(start, len(src))
        if "m_core->incexecstat((*ready_reg));" in src[i]
    )
    accept_bp = At(str(SIM / "src/gpgpu-sim/shader.cc") + ":" + str(line), accept)
else:
    At("ptx_instruction::pre_decode", decode)
    src = (SIM / "src/cuda-sim/instructions.cc").read_text().splitlines()
    line = next(
        i + 1 for i, s in enumerate(src) if "if (p->is_warp_barrier()) continue;" in s
    )
    At(str(SIM / "src/cuda-sim/instructions.cc") + ":" + str(line), collective)
    At("shader_core_ctx::issue_warp", issue)
    At("shader_core_ctx::warp_inst_complete", complete)
    At("shader_core_ctx::warp_waiting_at_mem_barrier", membar)
gdb.execute("run")
