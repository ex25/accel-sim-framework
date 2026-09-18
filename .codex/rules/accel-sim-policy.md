# Accel-Sim policy

Understand the existing trace-driven and PTX execution-driven paths before
changing timing logic. Keep the nested GPGPU-Sim repository boundary explicit.
Trace-format changes must update producer, parser, compatibility/version notes,
fixtures, and a source-to-consumer call-path document. Do not treat a tested
SM90 single-GPU config as multi-GPU/NVSwitch support.
