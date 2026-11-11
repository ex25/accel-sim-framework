"""GDB observer for the simulator's existing CUDA-event device timestamps.

The runtime's cudaEventElapsedTime returns host wall time. Read the recorded
device cycles without changing runtime code or the frozen CUDA executable.
"""
import json
from pathlib import Path
import re
import time

import gdb

gdb.execute("set breakpoint pending on")
gdb.execute("set pagination off")
gdb.execute("set print thread-events off")
started = {}


class Record(gdb.Breakpoint):
    def stop(self):
        event = gdb.newest_frame().read_var("event").dereference()
        started[int(event["m_uid"])] = time.monotonic()
        return False


class Elapsed(gdb.Breakpoint):
    def stop(self):
        frame = gdb.newest_frame()
        first = frame.read_var("start").dereference()
        last = frame.read_var("end").dereference()
        begin, end = int(first["m_gpu_tot_sim_cycle"]), int(last["m_gpu_tot_sim_cycle"])
        config = Path("gpgpusim.config").read_text()
        mhz = float(re.search(r"^-gpgpu_clock_domains\s+([\d.]+)", config, re.M)[1])
        assert end >= begin and int(first["m_updates"]) == int(last["m_updates"]) == 1
        result = {
            "method": "Existing CUDA-event stream timestamps, observed through GDB; no kernel-duration summation.",
            "start_cycle": begin,
            "end_cycle": end,
            "elapsed_cycles": end - begin,
            "core_MHz": mhz,
            "device_elapsed_ms": (end - begin) / mhz / 1000,
            "host_forward_wall_seconds": time.monotonic() - started[int(first["m_uid"])],
            "scope": "Events bracket the frozen forward, excluding initialization and final output download. Includes modeled launch/inter-kernel waits. Explicit DMA duration and host submission gaps are not modeled.",
            "runtime_elapsed_time_note": "output.json cuda_event_ms is the legacy simulator host wall timer; use this file for modeled device time.",
        }
        Path("event-timing.json").write_text(json.dumps(result, indent=2) + "\n")
        gdb.write("SIM_EVENT_TIMING " + json.dumps(result) + "\n")
        return False


Record("cuda_runtime_api.cc:cudaEventRecordInternal", internal=True)
Elapsed("cuda_runtime_api.cc:cudaEventElapsedTime", internal=True)
