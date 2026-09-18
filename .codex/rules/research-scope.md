# Scope guard for the first round

The first round may audit repositories and history, capture the environment,
build the existing code, run a small A800 trace if the environment permits,
map one trace instruction, and design/document the future integration. It must
not begin with a large simulator-core rewrite, full NVSwitch, full NVLS, a new
NCCL implementation, or a copied 8-rank trace presented as multi-GPU execution.
