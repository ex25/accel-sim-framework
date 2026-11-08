"""Read-only PTX copy issue, LSU return, shared delivery and consumer evidence."""

import gdb
import json
from pathlib import Path

gdb.execute("set pagination off")
gdb.execute("set breakpoint pending on")


def emit(kind, **v):
    print("OBS " + json.dumps(dict(kind=kind, **v), sort_keys=True))


def integer(v):
    return int(v)


def mask(v):
    w = v["_M_w"]
    return int(w[0]) if w.type.code == gdb.TYPE_CODE_ARRAY else int(w)


def cyc(core):
    gpu = core["m_gpu"].cast(gdb.lookup_type("gpgpu_sim").pointer())
    return int(gpu["gpu_sim_cycle"]) + int(gpu["gpu_tot_sim_cycle"])


def vec(v):
    p = v["_M_impl"]
    return [
        (p["_M_start"] + i).dereference()
        for i in range(int(p["_M_finish"] - p["_M_start"]))
    ]


def groups(core, wid):
    warp = core["m_warp"]["_M_impl"]["_M_start"][wid]
    return [
        dict(
            lane=i,
            committed=int(warp["m_ptx_cp_async_groups"][i]["committed"]),
            cutoff=int(warp["m_ptx_cp_async_groups"][i]["cutoff"]),
        )
        for i in range(32)
    ]


class After(gdb.FinishBreakpoint):
    def __init__(self, fn):
        super().__init__(gdb.newest_frame(), internal=True)
        self.fn = fn

    def stop(self):
        self.fn()
        return False


class At(gdb.Breakpoint):
    def __init__(self, spec, fn):
        super().__init__(spec, internal=True)
        self.fn = fn

    def stop(self):
        f = gdb.newest_frame()
        if "@plt" not in (f.name() or ""):
            self.fn(f)
        return False


def decode(f):
    p = f.read_var("this")

    def done():
        if int(p["m_ptx_cp_async_op"]) == 1:
            emit(
                "decode",
                pc=int(p["pc"]),
                sources=[int(p["in"][i]) for i in range(24)],
                outputs=[int(p["out"][i]) for i in range(8)],
                ar1=int(p["ar1"]),
                ar2=int(p["ar2"]),
                cache=int(p["cache_op"]),
                data_size=int(p["data_size"]),
            )

    After(done)


def issued(f):
    inst = f.read_var("inst")
    core = f.read_var("this")

    def done():
        kind = int(inst["m_ptx_cp_async_op"])
        wid = int(inst["m_warp_id"])
        if kind:
            data = dict(
                op=kind,
                uid=int(inst["m_uid"]),
                pc=int(inst["pc"]),
                warp=wid,
                mask=mask(inst["m_warp_active_mask"]),
                cycle=cyc(core),
                groups=groups(core, wid),
            )
            if kind == 1:
                data["requests"] = len(vec(inst["m_accessq"]))
                payload = inst["m_ptx_cp_async_copies"]["_M_ptr"]
                if int(payload):
                    data["copies"] = [
                        dict(
                            lane=int(c["lane"]),
                            dst=int(c["dst"]),
                            src=int(
                                vec(inst["m_per_scalar_thread"])[i]["memreqaddr"][
                                    "_M_impl"
                                ]["_M_start"][0]
                            ),
                            size=int(c["source_size"]),
                            shared=str(c["shared"]),
                        )
                        for i, c in enumerate(vec(payload.dereference()))
                        if mask(inst["m_warp_active_mask"]) & (1 << i)
                    ]
            else:
                data["n"] = int(inst["m_ptx_cp_async_wait"])
            emit("issue", **data)
        elif str(inst["space"]["m_type"]) == "shared_space":
            emit(
                "shared_consume" if str(inst["op"]) == "LOAD_OP" else "shared_store",
                uid=int(inst["m_uid"]),
                pc=int(inst["pc"]),
                warp=wid,
                cycle=cyc(core),
            )
        elif int(inst["op"]) == int(gdb.parse_and_eval("INTP_OP")):
            # Keep a few independent arithmetic events, not every iteration.
            if counts.get(wid, 0) < 12 or wid == 1 and counts.get(wid, 0) < 20:
                emit("arithmetic", warp=wid, pc=int(inst["pc"]), cycle=cyc(core))
            counts[wid] = counts.get(wid, 0) + 1

    After(done)


counts = {}


def delivery(f):
    inst = f.read_var("inst")
    core = f.read_var("this")
    if int(inst["m_ptx_cp_async_op"]) != 1:
        return
    uid = int(inst["m_uid"])
    wid = int(inst["m_warp_id"])
    copies = vec(inst["m_ptx_cp_async_copies"]["_M_ptr"].dereference())
    active = mask(inst["m_warp_active_mask"])
    emit("last_return", uid=uid, warp=wid, cycle=cyc(core))

    def done():
        emit(
            "shared_delivered",
            uid=uid,
            warp=wid,
            cycle=cyc(core),
            copies=[
                dict(lane=i, group=int(copies[i]["group"]), dst=int(copies[i]["dst"]))
                for i in range(32)
                if active & (1 << i)
            ],
            groups=groups(core, wid),
        )

    After(done)


wait_seen = {}


def wait_state(f):
    warp = f.read_var("this")
    wid = int(warp["m_warp_id"])
    core = warp["m_shader"]
    if int(warp["n_completed"]) == 32:
        return
    state = bool(f.read_var("waiting"))
    key = (str(warp), wid)
    if any(int(warp["m_ptx_cp_async_groups"][i]["cutoff"]) for i in range(32)):
        if wait_seen.get(key) != state:
            emit(
                "wait_state",
                warp=wid,
                blocked=state,
                cycle=cyc(core),
                groups=groups(core, wid),
            )
        wait_seen[key] = state


shader = Path(
    "/home/wangjz/in-switch/accel-sim/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc"
)
lines = shader.read_text().splitlines()
line = next(
    i + 2
    for i, s in enumerate(lines)
    if "for (const auto &groups : m_ptx_cp_async_groups) waiting |=" in s
)
At(str(shader) + ":" + str(line), wait_state)


def request_return(f):
    inst = f.read_var("inst")
    if int(inst["m_ptx_cp_async_op"]) == 1:
        emit(
            "request_return",
            uid=int(inst["m_uid"]),
            warp=int(inst["m_warp_id"]),
            cycle=cyc(f.read_var("this")["m_core"]),
        )


At("ldst_unit::writeback_complete", request_return)
At("ptx_instruction::pre_decode", decode)
At("exec_shader_core_ctx::func_exec_inst", issued)
At("shader_core_ctx::unset_depbar", delivery)
gdb.execute("run")
