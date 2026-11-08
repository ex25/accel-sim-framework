// Explicit event ordering for the real warp state; no simulated DRAM here.
#include "libcuda/gpgpu_context.h"
#include "src/cuda-sim/ptx_ir.h"
#include <cassert>
#include <cstdio>

static void completion(unsigned ack0, unsigned ack1) {
  shd_warp_t warp(nullptr, 32);
  auto &bulk = warp.m_ptx_bulk_groups[0];
  auto g0 = bulk.add();
  bulk.commit();
  auto g1 = bulk.add();
  bulk.commit();
  warp.m_ptx_tensor_pending = 2;
  warp.inc_store_req(true);
  warp.inc_store_req(true);
  assert(warp.ordinary_stores_done() && !warp.stores_done());
  // Ordinary cp.async and a different lane cannot commit the bulk queue.
  warp.m_ptx_cp_async_groups[0].commit();
  warp.m_ptx_bulk_groups[1].commit();
  assert(bulk.committed == 2);
  for (unsigned cycle = 0; cycle <= ack1; ++cycle) {
    if (cycle == 3)
      bulk.complete(g0);
    if (cycle == 6)
      bulk.complete(g1);
    if (cycle == ack0 || cycle == ack1) {
      --warp.m_ptx_tensor_pending;
      --warp.m_ptx_tensor_stores_outstanding;
      warp.dec_store_req();
    }
    bulk.wait(1);
    assert(bulk.waiting() == (cycle < 3));
    bulk.wait(0);
    assert(bulk.waiting() == (cycle < 6));
    assert(warp.hardware_done() == (cycle == ack1));
  }
  bulk.commit();        // An empty group still advances the sequence.
  auto g3 = bulk.add(); // Uncommitted work is excluded by wait0.
  bulk.wait(0);
  assert(!bulk.waiting());
  bulk.commit();
  bulk.wait(0);
  assert(bulk.waiting());
  bulk.complete(g3);
  assert(!bulk.waiting());
  printf("COMPONENT read0=3 read1=6 ack0=%u ack1=%u PASS\n", ack0, ack1);
}
int main() {
  completion(20, 40);
  completion(200, 400); // Only the acknowledgements move; reads stay 3/6.
  shd_warp_t trace(nullptr, 32);
  trace.add_outstanding_tma_store(9, 2);
  trace.commit_tma_group();
  trace.dec_tma_stores_outstanding(9);
  trace.dec_tma_stores_outstanding(9);
  assert(trace.hardware_done());
  puts("COMPONENT SASS group adapter PASS (no trace execution)");
}
