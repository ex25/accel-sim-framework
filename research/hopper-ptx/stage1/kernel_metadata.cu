#include <iostream>
#include "flash_fwd_launch_template.h"

// Mirror the aliases selected by run_flash_fwd<90,128,128,1,half_t,half_t,...>.
using ShapeMNK = cute::Shape<cute::Int<128>, cute::Int<176>, cute::Int<128>>;
using ShapeMNKPV = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<176>>;
using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
using Mainloop = flash::CollectiveMainloopFwdSm90<
    2, Cluster, ShapeMNK, 128, cutlass::half_t, float, cutlass::arch::Sm90,
    false, false, false, false, false, false, false, true, true,
    false, false, false>;
using Epilogue = flash::CollectiveEpilogueFwd<
    ShapeMNKPV, Cluster, cutlass::half_t, cutlass::arch::Sm90,
    Mainloop::NumMmaThreads, false, false, false, false>;
using Kernel = flash::FlashAttnFwdSm90<
    Mainloop, Epilogue, flash::StaticPersistentTileScheduler<false>>;
static_assert(std::get<0>(tile_size_fwd_sm90(128,128,false,false)) == 128);
static_assert(std::get<1>(tile_size_fwd_sm90(128,128,false,false)) == 176);
static_assert(std::get<2>(tile_size_fwd_sm90(128,128,false,false)));
static_assert(std::get<3>(tile_size_fwd_sm90(128,128,false,false)));

int main() {
    std::cout << "{\"block\":[" << Kernel::MaxThreadsPerBlock
              << ",1,1],\"dynamic_shared_bytes\":" << Kernel::SharedStorageSize
              << ",\"parameter_bytes\":" << sizeof(Kernel::Params)
              << ",\"parameter_alignment\":" << alignof(Kernel::Params)
              << ",\"mma_threads\":" << Mainloop::NumMmaThreads
              << ",\"producer_threads\":" << Mainloop::NumProducerThreads
              << "}\n";
}
