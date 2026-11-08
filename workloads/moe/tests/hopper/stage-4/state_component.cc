// Artificial completion through the shared state API, not a DRAM/TMA test.
#include "libcuda/gpgpu_context.h"
#include "src/cuda-sim/ptx_ir.h"
#include <cassert>
#include <cstdio>

static mbarrier_t make(unsigned count) {
  return mbarrier_t(ClusterCTAIdentifier(dim3(0), 0), dim3(0), 1024, count);
}
static void state(const char *op, const mbarrier_t &b, unsigned pending, int tx,
                  unsigned phase) {
  printf("STATE %s pending=%u tx=%d phase=%u\n", op,
         b.get_pending_thread_count(), b.get_tx_count(), b.get_phase());
  assert(b.get_pending_thread_count() == pending);
  assert(b.get_tx_count() == tx);
  assert(b.get_phase() == phase);
  assert(b.waiting_parity(phase & 1));
  assert(!b.waiting_parity((phase & 1) ^ 1));
}
int main() {
  {
    auto b = make(2);
    state("init2", b, 2, 0, 0);
    b.expect_on(64);
    state("expect64", b, 2, 64, 0);
    b.arrive_on(1);
    state("arrive1", b, 1, 64, 0);
    b.arrive_on(1);
    state("arrive2", b, 0, 64, 0);
    b.complete_on(32);
    state("complete32", b, 0, 32, 0);
    b.complete_on(32);
    state("complete32", b, 2, 0, 1);
  }
  {
    auto b = make(2);
    b.expect_on(64);
    state("expect64", b, 2, 64, 0);
    b.complete_on(64);
    state("complete-first", b, 2, 0, 0);
    b.arrive_on(1);
    state("arrive1", b, 1, 0, 0);
    b.arrive_on(1);
    state("arrive2", b, 2, 0, 1);
  }
  {
    auto b = make(1);
    b.complete_on(64);
    state("early-complete", b, 1, -64, 0);
    b.arrive_on(1);
    state("arrive", b, 0, -64, 0);
    b.expect_on(64);
    state("late-expect", b, 1, 0, 1);
  }
  {
    auto b = make(1);
    b.arrive_expect_on(1, 64);
    state("combined", b, 0, 64, 0);
    b.complete_on(64);
    state("complete-combined", b, 1, 0, 1);
  }
  {
    auto b = make(2);
    for (unsigned round = 0; round < 3; ++round) {
      b.arrive_expect_on(1, 64);
      state("round-expect", b, 1, 64, round);
      b.complete_on(64);
      state("round-complete", b, 1, 0, round);
      b.arrive_on(1);
      state("round-arrive", b, 2, 0, round + 1);
      // SASS adapter extracts bit31; PTX consumes parity directly.
      unsigned sass_phase = (round & 1) << 31;
      assert(!b.waiting_parity(sass_phase >> 31));
      assert(!b.waiting_parity(round & 1));
    }
  }
  // Same completion API and offset lookup as SASS local/multicast service.
  ClusterMbarriersLookupTable table;
  for (unsigned rank = 0; rank < 2; ++rank) {
    unsigned address = (rank << 24) | 1024;
    table.insert_mbarrier(std::make_unique<mbarrier_t>(
        ClusterCTAIdentifier(dim3(0), rank), dim3(0, rank, 0), address, 1));
    table.lookup_clustermbar(address)->arrive_expect_on(1, 64);
  }
  table.lookup_clustermbar(1024)->complete_on(64);
  state("local-complete", *table.lookup_clustermbar(1024), 1, 0, 1);
  state("peer-before-multicast", *table.lookup_clustermbar(0x01000400), 0, 64,
        0);
  auto peers = table.lookup_clustermbars_by_offset(1024);
  for (auto it = peers.first; it != peers.second; ++it)
    if (it->second->get_cluster_cta_identifier().cluster_rank == 1)
      it->second->complete_on(64);
  state("peer-multicast-complete", *table.lookup_clustermbar(0x01000400), 1, 0,
        1);
  puts("PASS fixed states, signed tx, atomic arrival, rounds, SASS "
       "parity/table components");
}
