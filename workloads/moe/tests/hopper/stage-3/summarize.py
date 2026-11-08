#!/usr/bin/env python3
"""Assert saved stage 3 host results and read-only event evidence."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import struct

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("root", type=Path)
a = p.parse_args()
root = a.root.resolve()
checks = 0


def require(value, message):
    global checks
    assert value, message
    checks += 1


def host_report(directory):
    meta = json.loads((directory / "run.json").read_text())
    results = []
    for case in meta["tests"]:
        require(case["passed"], f"{directory}: {case}")
        name = case["test"]
        text = (directory / (name + ".log")).read_text()
        require(f"PASS {name}" in text and "errors=0" in text, name)
        cycles = defaultdict(list)
        for kernel, count in re.findall(r"kernel_name = (\S+).*?gpu_sim_cycle = (\d+)", text, re.S):
            cycles[kernel].append(int(count))
        stamps = defaultdict(list)
        for phase, value in re.findall(r"HOST_TIME (\w+) ([\d.]+)", text):
            stamps[phase].append(float(value))
        times = (directory / (name + ".time")).read_text()
        results.append(dict(
            test=name, cycles=dict(cycles),
            registration_wall_s=stamps["registration_done"][0] - stamps["process_init_begin"][0],
            launch_sync_wall_s=[b-a for a,b in zip(stamps["kernel_begin"], stamps["kernel_done"])],
            compare_wall_s=[b-a for a,b in zip(stamps["compare_begin"], stamps["compare_end"])],
            max_rss_kib=int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", times)[1]),
            process_wall=re.search(r"Elapsed \(wall clock\) time .*?: (.*)", times)[1],
        ))
    return dict(library_sha256=meta["library_sha256"], config_sha256=meta["config_sha256"], tests=results)


def event_report(path):
    events = [json.JSONDecoder().raw_decode(line.split("OBS ",1)[1])[0]
              for line in path.read_text().splitlines() if "OBS {" in line]
    require(bool(events), str(path))
    groups = defaultdict(list)
    creates, retires = defaultdict(dict), defaultdict(dict)
    sends, visible, releases = defaultdict(list), defaultdict(list), defaultdict(list)
    multi = False
    high_sms = set()
    for e in events:
        if e["kind"] == "cta_issue":
            require(e["occupied_threads"] <= 2048, "threads over capacity")
            require(e["occupied_registers"] <= 65536, "registers over capacity")
            require(e["occupied_shared"] <= 233472, "shared over capacity")
            if e["shared_per_cta"] > 1024 + 32:
                groups[(e["uid"], e["cta"][0], e["cta"][1]//2, e["cta"][2])].append(e)
            else:
                multi |= e["active"] >= 2
            if e["shared_per_cta"] == 216064:
                high_sms.add(e["sm"])
                require(e["active"] == 1 and e["threads"] == 384, "large CTA resources")
        elif e["kind"] == "shared_create":
            creates[e["uid"]][tuple(e["cta"])] = e["shared"]
        elif e["kind"] == "shared_retire":
            retires[e["uid"]][tuple(e["cta"])] = e
        elif e["kind"] in ("arrive_send", "arrive_visible", "wait_release"):
            key = (e["uid"], tuple(e["cluster"]), e["epoch"])
            {"arrive_send":sends, "arrive_visible":visible, "wait_release":releases}[e["kind"]][key].append(e)
    for key, issue in groups.items():
        require(len(issue) == 2, f"half cluster admitted {key}")
        require(issue[0]["sm"] != issue[1]["sm"], "members share an SM")
        require(issue[0]["cycle"] == issue[1]["cycle"], "admission interleaved")
        require([e["cta"][1]%2 for e in issue] == [0,1], "CTA enumeration")
    for uid, objects in creates.items():
        require(objects.keys() == retires[uid].keys(), "shared lifetime missing retirement")
        # All members live at admission; no two concurrent members may alias.
        for cta, pointer in objects.items():
            if cta[1] % 2 == 0:
                peer = (cta[0], cta[1]+1, cta[2])
                require(pointer != objects[peer], "shared objects alias")
    phases = []
    for key, issued in sends.items():
        require(sum(e["count"] for e in issued) == 768, "arrival thread count")
        require(sum(e["count"] for e in visible[key]) == 768, "delivered thread count")
        require(sum(e["count"] for e in releases[key]) == 768, "released thread count")
        ordered = visible[key]
        require(all(e["cycle"] == e["due"] for e in ordered), "delivery time")
        require(all(b["cycle"] > a["cycle"] for a,b in zip(ordered,ordered[1:])), "collector throughput")
        for send, received in zip(issued, ordered):
            require(received["cycle"] >= send["cycle"] + send["latency"], "instant notification")
        done = max(e["cycle"] for e in ordered)
        first_release = min(e["cycle"] for e in releases[key])
        require(first_release >= done, "premature wait release")
        phases.append(dict(uid=key[0], cluster=list(key[1]), epoch=key[2],
                           first_send=min(e["cycle"] for e in issued),
                           last_visible=done, first_release=first_release,
                           last_release=max(e["cycle"] for e in releases[key]),
                           transport_latency=issued[0]["latency"]))
    return dict(cta_groups=len(groups), shared_kernels=len(creates), ordinary_multiresident=multi,
                large_shared_sms=sorted(high_sms), phases=phases)


report = dict(runs={}, events={}, regressions={})
for directory in sorted(root.iterdir()):
    if (directory / "run.json").is_file():
        if directory.name.startswith("regression-"):
            meta = json.loads((directory / "run.json").read_text())
            for test in meta["tests"]:
                require(test["passed"], f"regression {directory}: {test}")
            report["regressions"][directory.name] = meta["tests"]
        else:
            report["runs"][directory.name] = host_report(directory)
for path in sorted(root.glob("events-*.log")):
    report["events"][path.stem] = event_report(path)
require("debug-two-sm" in report["runs"], "missing debug results")
require("release-two-sm" in report["runs"], "missing release results")
for name in ("identity", "mapa", "addresses", "cluster_sync", "ordinary", "plain_ex"):
    debug = next(t for t in report["runs"]["debug-two-sm"]["tests"] if t["test"] == name)
    release = next(t for t in report["runs"]["release-two-sm"]["tests"] if t["test"] == name)
    kernel = "ordinary" if name == "plain_ex" else name
    require(debug["cycles"][kernel] == release["cycles"][kernel], f"debug/release device cycles {name}")
for name in ("events-sync", "events-delay128", "events-transport129"):
    phases = report["events"][name]["phases"]
    require(len(phases) == 12, "two kernels, two clusters, three epochs")
    for uid in {x["uid"] for x in phases}:
        slow = [x for x in phases if x["uid"] == uid and x["cluster"] == [0,0,0]]
        fast = [x for x in phases if x["uid"] == uid and x["cluster"] == [1,0,0]]
        require(fast[0]["first_release"] < slow[0]["last_visible"], "independent cluster blocked")
    require(all(x["transport_latency"] == (129 if name.endswith("129") else 29) for x in phases), "transport parameter")
base = report["events"]["events-sync"]["phases"]
late = report["events"]["events-delay128"]["phases"]
late_by_key = {(x["uid"], tuple(x["cluster"]), x["epoch"]): x for x in late}
for b in base:
    l = late_by_key[(b["uid"], tuple(b["cluster"]), b["epoch"])]
    if b["cluster"] == [0,0,0]:
        require(l["first_release"] > b["first_release"], "late CTA did not delay wait")
# An early warp performs arithmetic between arrive and wait while its peer is
# still pending: compare PTX clock output against the corresponding event time.
raw = (root / "debug-two-sm/cluster_sync.actual-0").read_bytes()
words = struct.unpack("<"+"I"*(len(raw)//4), raw)
first = next(x for x in base if x["cluster"] == [0,0,0] and x["epoch"] == 0)
require(words[19] < words[(2*384)*32+19], "arrive blocked independent work")
require(words[19] < first["last_visible"], "no work observed before completion")
require(set([0,79,80,131]).issubset(report["events"]["events-high-sm"]["large_shared_sms"]), "high SM run coverage")
require(report["events"]["events-ordinary"]["ordinary_multiresident"], "ordinary CTA residency")
report["assertions"] = checks
(root / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
print(f"PASS {checks} saved-result/event assertions; {root/'summary.json'}")
