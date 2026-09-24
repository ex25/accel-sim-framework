# 012 — P1 PTX synchronization event producer

## Scope

P1 adds an opt-in bridge from timing-visible PTX synchronization instructions
to the P0 JSONL event contract. It covers the PTX execution-driven path at the
warp issue/completion boundary:

| GPU-side observation | P0 event kind |
| --- | --- |
| `bar.sync`, `bar.arrive`, `bar.red` issued | `local_barrier_arrive` |
| CTA barrier condition releases waiting warps | `barrier_wakeup` |
| `membar.*` issued | `memory_release` |
| outstanding writes no longer block the warp | `memory_acquire` |

The stream is enabled only when `PTX_SIM_EVENT_FILE` is set. Each line is a
P0-compatible `CommunicationEvent` JSON object. `PTX_SIM_RANK` overrides the
single-process default rank (the shader id); this is the hook that later
multi-GPU launchers will use to assign ranks 0–7.

## Timing and limits

`ready_time_ps` currently carries the global GPGPU-Sim cycle count and is marked
with `metadata.time_domain=sim_cycle`. It is deliberately not converted to
picoseconds yet; clock-domain conversion belongs to the later multi-GPU/fabric
phase. The producer has no NCCL, BookSim, NVLink, NVSwitch, or NVLS side effect,
and disabling the environment variable leaves simulator behavior unchanged.

The event is emitted once per issued warp instruction. This avoids lane-level
duplicates while preserving the timing model's CTA barrier accounting. The
completion event is an observation of the existing GPGPU-Sim release path, not
a new synchronization implementation.

## Verification

**Verified**:

1. `cmake --build gpu-simulator/build/release -j8` completed and linked the
   rebuilt `shader.cc` into `libgpgpusim.a`, `accel-sim.out`, and `libcudart.so`.
2. After `cmake --install gpu-simulator/build/release`, an external PTX fixture
   that prepends `membar.cta; bar.sync 0;` to the existing `cais_add` kernel
   completed with `status=PASS`.
3. The emitted JSONL parsed through
   `tools/booksim_nvlink/communication_events.py` and contained all four event
   kinds listed above. The observed sample had 25 events for eight warps;
   exact count depends on the fixture.
4. Existing P0 unit tests and the transport contract tests remained green.

The end-to-end run should use an existing Accel-Sim run directory containing
the generated `gpgpusim.config` and `config_hopper_islip.icnt`; copying only the
source SM90 config to an empty directory is not a supported launcher setup and
can enter the pre-existing cuobjdump initialization failure.

## Next phase

P2 will consume this stream through a point-to-point endpoint adapter. It must
add explicit communication events and completion callbacks without moving
network timing into the P1 GPU-local producer.
