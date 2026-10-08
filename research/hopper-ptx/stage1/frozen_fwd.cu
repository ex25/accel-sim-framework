// Instantiate the upstream FA3 launcher for exactly one fixed-length workload.
// The kernel implementation and launch parameter conversion remain upstream.
#include "flash_fwd_launch_template.h"

template void run_flash_fwd<
    90, 128, 128, 1, cutlass::half_t, cutlass::half_t,
    false, false, false, false, false, false, false, false, false, false>(
    Flash_fwd_params &, cudaStream_t);
