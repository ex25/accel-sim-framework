# Stage 2 text PTX harness

This directory contains a small execution driven entry point. `stage2_kernel.ptx`
is read as text by GPGPU-Sim's existing parser, kernel parameters are packed with
the same `gpgpu_ptx_sim_arg` layout used by the runtime, and the functional
executor writes a global result buffer. The harness deliberately uses ordinary
PTX arithmetic, global memory, shared memory, and `bar.sync`; it does not emulate
or fabricate Hopper instruction results.

Run it from this directory with `./run.sh`. The script copies the selected H100
configuration into a temporary run directory, builds the direct C++ entry point,
and records stdout/stderr in `artifacts/run.log`. `parser_errors.sh` runs three
invalid PTX fixtures and checks that each diagnostic contains the fixture path
and source line.
