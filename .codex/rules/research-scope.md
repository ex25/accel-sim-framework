# Scope guard and phase gates

The first round is reconnaissance and evidence capture: audit repositories and
history, read `.claude`, capture the A800 environment, build existing code,
run a small real trace if possible, map one instruction, audit Hopper and
multi-GPU gaps, study NCCL/NVLink/NVSwitch/NVLS/BookSim, and document an
integration design. It must not begin with a large simulator-core rewrite,
full NVSwitch, full NVLS, a new NCCL implementation, or a copied 8-rank trace
presented as multi-GPU execution.

The confirmed research direction is PTX-first. After the A800 SASS baseline,
the next design gate is a small SM90 PTX execution-driven baseline followed by a
minimal custom instruction or synthetic instruction/event prototype. SASS is
used to validate existing hardware behaviour; it is not the primary way to
generate inputs for a GPU or instruction that does not exist in hardware.

Do not advance to implementation until the current phase has an exit record:
baseline SHA/command, source evidence, known limitations, focused test plan,
and a next-step decision. The minimal implementation order is:

1. PTX baseline and source call path;
2. one custom/synthetic instruction with focused timing and dependency tests;
3. communication-event schema and logical GPU contexts;
4. BookSim transport facade and clock/event coupling;
5. Ring collective;
6. NVSwitch topology and semantic state;
7. NVLS multicast/reduction/completion;
8. SASS/H100 validation and hardware calibration.

Real H100/NVSwitch calibration is a later gate, not a reason to manufacture
hardware claims from an A800 or from a synthetic PTX workload.
