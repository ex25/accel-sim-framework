"""One paid H100 invocation; analysis and compilation stay on the local host."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

import modal

PACKAGE = Path(os.environ.get("H100_NUMERICS_PACKAGE", "/work"))
image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.04-py3")
if modal.is_local():
    # Upload only runtime inputs and the already compiled executable.
    image = image.add_local_dir(
        PACKAGE, "/work", copy=True,
        ignore=lambda p: p.suffix in (".ptx", ".fatbin", ".cubin", ".cu", ".inc", ".S", ".o"),
    )
app = modal.App("hopper-numerical-evidence-once")


@app.function(image=image, gpu="H100!", timeout=120, max_containers=1,
              retries=0, single_use_containers=True)
def collect():
    work = Path("/work")
    result = Path("/tmp/evidence")
    result.mkdir()
    begin = time.monotonic()
    expected = json.loads((work / "sha256.json").read_text())
    checked = {}
    for p in work.iterdir():
        if p.name in expected:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest != expected[p.name]:
                raise ValueError("Upload differs: " + p.name)
            checked[p.name] = digest
    env = os.environ.copy()
    # The standalone executable statically links native CUDA 12.8 runtime.
    for key in ("PTX_SIM_MODE_FUNC", "PTX_SIM_USE_PTX_FILE", "PTX_SIM_KERNELFILE",
                "CUOBJDUMP_SIM_FILE", "LD_PRELOAD"):
        env.pop(key, None)
    cache = Path("/tmp/h100-numerics-jit-cache")
    cache.mkdir()
    env.update(CUDA_CACHE_PATH=str(cache), CUDA_CACHE_DISABLE="0",
               CUDA_CACHE_MAXSIZE=str(64 * 1024 * 1024))
    device = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader"], text=True)
    if "H100" not in device:
        raise RuntimeError("Requested H100, received " + device)
    with (result / "batch.log").open("w") as log:
        try:
            proc = subprocess.run([str(work / "batch"), str(result), "h100"],
                                  cwd=work, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=60)
            status = dict(returncode=proc.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            status = dict(returncode=None, timed_out=True)
    # Return any already-collected evidence even on failure; never auto-retry.
    status.update(device=device, execution_and_validation_seconds=time.monotonic()-begin,
                  uploaded_sha256=checked, gpu_request="H100!", retries=0,
                  note="Includes diagnostic synchronization and copies; not a performance measurement.")
    status["container_ptxas"] = subprocess.check_output(
        ["/usr/local/cuda/bin/ptxas", "--version"], text=True)
    with tarfile.open(result / "driver-jit-cache.tar.gz", "w:gz", compresslevel=1) as tar:
        tar.add(cache, arcname="cache")
    status["outputs_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in result.iterdir() if p.is_file()}
    (result / "run.json").write_text(json.dumps(status, indent=2)+"\n")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz", compresslevel=1) as tar:
        for p in sorted(result.iterdir()):
            tar.add(p, arcname=p.name)
    return archive.getvalue()


@app.local_entrypoint()
def main(result_dir: str):
    preflight = json.loads((PACKAGE.parent / "preflight.json").read_text())
    if not preflight["passed"]:
        raise RuntimeError("Local preflight has not passed")
    hashes = json.loads((PACKAGE / "sha256.json").read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Local bundle changed: " + name)
    if hashlib.sha256((PACKAGE / "sha256.json").read_bytes()).hexdigest() != preflight["manifest_sha256"]:
        raise RuntimeError("Bundle changed after preflight")
    destination = Path(result_dir)
    destination.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    payload = collect.remote()  # Exactly one call; no loops or retries.
    (destination / "evidence.tar.gz").write_bytes(payload)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        for member in tar.getmembers():
            if Path(member.name).name != member.name or not member.isfile():
                raise ValueError("Unexpected archive member")
            (destination / member.name).write_bytes(tar.extractfile(member).read())
    record = json.loads((destination / "run.json").read_text())
    print(json.dumps(dict(result_dir=str(destination), download_bytes=len(payload),
                          total_client_seconds=time.monotonic()-begin, **record), indent=2))
