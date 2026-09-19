# Scope guard and phase gates

The first round is reconnaissance and evidence capture: audit repositories and
history, read `.claude`, capture the A800 environment, build existing code,
run a small real trace if possible, map one instruction, audit Hopper and
multi-GPU gaps, study NCCL/NVLink/NVSwitch/NVLS/BookSim, and document an
integration design. It must not begin with a large simulator-core rewrite,
full NVSwitch, full NVLS, a new NCCL implementation, or a copied 8-rank trace
presented as multi-GPU execution.

Do not advance to implementation until the current phase has an exit record:
baseline SHA/command, source evidence, known limitations, focused test plan,
and a next-step decision. The minimal implementation order is transport smoke
test, logical endpoints, clock/event coupling, Ring collective, NVSwitch
topology, then NVLS semantics. Real H100/NVSwitch calibration is a later gate,
not a reason to manufacture hardware claims from an A800.
