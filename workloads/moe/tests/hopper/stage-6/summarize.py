"""Check Stage 6 trace evidence and save compact, reproducible measurements."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


def events(log, kind):
    result = []
    for line in log.splitlines():
        if kind + " " in line:
            fields = dict(re.findall(r"(\w+)=([^ ]+)", line.split(kind, 1)[1]))
            result.append({k: int(v) if v.isdecimal() else v for k, v in fields.items()})
    return result


def summarize(run):
    meta = json.loads((run / "run.json").read_text())
    result = {"run": str(run), "library_sha256": meta["library_sha256"], "tests": {}}
    ptx = (run / "micro.ptx").read_text()
    for test in meta["tests"]:
        assert test["passed"], test
        name = test["test"]
        log = (run / (name + ".log")).read_text()
        info = {"cycles": list(map(int, re.findall(r"^gpu_sim_cycle = (\d+)", log, re.M)))}
        assert len(info["cycles"]) == 2, name
        stamps = defaultdict(list)
        for stage, value in re.findall(r"HOST_TIME (\w+) ([\d.]+)", log):
            stamps[stage].append(float(value))
        info["registration_seconds"] = stamps["registration_done"][0] - stamps["main"][0]
        for first, last, key in [("kernel_begin", "kernel_end", "simulation_seconds"),
                                 ("compare_begin", "compare_end", "compare_seconds")]:
            info[key] = [b - a for a, b in zip(stamps[first], stamps[last])]
        time = (run / (name + ".time")).read_text()
        info["max_rss_kib"] = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", time)[1])
        info["elapsed"] = re.search(r"Elapsed \(wall clock\) time.*: (\S+)", time)[1]
        body = re.search(r"\.visible \.entry test_" + name + r"\(.*?\n}\n", ptx, re.S)[0]
        tuples = [re.findall(r"%f\d+", text) for text in re.findall(r"wgmma\.mma_async[^\n]*?\{([^}]+)\}", body)]
        frames = []
        for frame in log.split("HOST_TIME kernel_begin ")[1:]:
            reads = events(frame, "PTX_GMMA read")
            arrivals = events(frame, "PTX_GMMA arrive")
            dispatch = events(frame, "PTX_GMMA dispatch")
            writes = events(frame, "PTX_GMMA writeback")
            ports = events(frame, "PTX_GMMA ports")
            names = events(frame, "PTX_GMMA tuple")
            assert len(dispatch) == len(writes) == 4 * len(reads), name
            for arrive in arrivals:
                uid, seq = arrive["uid"], arrive["seq"]
                actual = [v for v in names if v["uid"] == uid]
                expected = tuples[seq % len(tuples)]
                assert len(actual) == 64 and arrive["outputs"] == 64
                for j, (value, reg) in enumerate(zip(actual, expected)):
                    assert value["j"] == j and value["name"] == reg
                    assert value["reg"] == value["out"] and value["dst"] == int(reg[2:])
                service = [v for v in ports if v["uid"] == uid]
                wb = [v for v in writes if v["uid"] == uid]
                assert sum(v["serviced"] for v in service) == 64
                assert all(v["complete"] == 0 for v in service[:-1])
                assert service[-1]["complete"] == 1 and len(wb) == 1
                assert wb[0]["cycle"] == service[-1]["cycle"]
            operations = []
            for read in reads:
                seq = read["seq"]
                arrive = [v for v in arrivals if v["seq"] == seq]
                done = [v for v in writes if v["seq"] == seq]
                sent = [v for v in dispatch if v["seq"] == seq]
                assert len(arrive) == 4 and arrive[-1]["mask"] == 15
                assert max(v["cycle"] for v in arrive) <= read["cycle"]
                assert read["math"] == 262144 and done[-1]["pending"] == 0
                assert all(v["pending"] for v in done[:-1])
                operations.append(dict(seq=seq, read=read["cycle"], ready=read["ready"],
                                       dispatch=min(v["cycle"] for v in sent),
                                       complete=max(v["cycle"] for v in done)))
            out = dict(logical_operations=len(reads), warp_dispatches=len(dispatch), operations=operations,
                       port_batches=sorted({len([p for p in ports if p["uid"] == w["uid"]]) for w in writes}))
            if len(operations) > 1:
                out["second_dispatched_before_first_complete"] = operations[1]["dispatch"] < operations[0]["complete"]
            if name == "groups":
                early = [v for v in events(frame, "PTX_GMMA release") if v["kind"] == 6 and v["groups"] == 1]
                assert len(early) == 4
                assert operations[0]["complete"] <= min(v["cycle"] for v in early)
                assert max(v["cycle"] for v in early) < operations[2]["complete"]
                out["wait1_early_return"] = [v["cycle"] for v in early]
            delivered = events(frame, "PTX_TMA delivered")
            if name in ("integration", "pipeline", "serial"):
                assert len(delivered) == (2 if name == "integration" else 4)
                assert max(v["cycle"] for v in delivered[:2]) <= operations[0]["read"]
                if name != "integration":
                    assert max(v["cycle"] for v in delivered[2:]) <= operations[1]["read"]
                    out["next_input_delivered_during_first_gmma"] = any(
                        operations[0]["dispatch"] < v["cycle"] < operations[0]["complete"] for v in delivered[2:])
                    out["tma_output_read_done"] = [v["cycle"] for v in events(frame, "PTX_TMA read_done")]
                    assert len(out["tma_output_read_done"]) == 4
                out["tma_input_delivery"] = [v["cycle"] for v in delivered]
            frames.append(out)
        info["launches"] = frames
        if name == "quota":
            quota = events(log, "PTX_GMMA quota")
            assert "QUOTA initial_regs=168 total=64512" in log
            assert all(any(v["group"] == group and v["waiting"] == 1 and v["free"] == 0 for v in quota) for group in (1, 2))
            assert all(any(v["group"] == group and v["quota"] == target and v["waiting"] == 0 for v in quota) for group, target in ((0,40),(1,232),(2,232)))
            info["quota_events"] = quota
        result["tests"][name] = info
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps([summarize(run) for run in args.runs], indent=2) + "\n")
    print("PASS trace counts, tuple order, port completion, waits, TMA ordering and quota")
