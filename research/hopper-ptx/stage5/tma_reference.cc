#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "cuda-sim/tma.h"
#include "cuda-sim/ptx_sim.h"
#include "libcuda/gpgpu_context.h"

static void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
int main() {
  ptx_tma_descriptor d;
  d.rank = 1; d.element_bytes = 4; d.base = 0x1000;
  d.dimensions = {6, 0, 0, 0, 0};
  d.byte_strides = {4, 0, 0, 0, 0};
  d.box = {3, 0, 0, 0, 0};
  auto one = ptx_tma_expand(d, 1, {1, 0, 0, 0, 0});
  check(one.bytes == 12 && one.oob_bytes == 0 &&
        one.addresses == std::vector<uint64_t>({0x1004, 0x1008, 0x100c}), "1D reference");
  d.rank = 2;
  d.dimensions = {3, 2, 0, 0, 0};
  d.byte_strides = {4, 16, 0, 0, 0}; // padded rows
  d.box = {2, 2, 0, 0, 0};
  auto two = ptx_tma_expand(d, 2, {1, 0, 0, 0, 0});
  check(two.oob_bytes == 0 &&
        two.addresses == std::vector<uint64_t>({0x1004, 0x1008, 0x1014, 0x1018}), "2D reference");
  auto transfer = ptx_tma_expand(d, 2, {2, 0, 0, 0, 0});
  check(transfer.bytes == 16 && transfer.oob_bytes == 8 &&
        transfer.addresses[0] == 0x1008 && transfer.addresses[2] == 0x1018 &&
        transfer.in_bounds == std::vector<bool>({true, false, true, false}), "OOB reference");
  auto negative = ptx_tma_expand(d, 2, {-1, 0, 0, 0, 0});
  check(negative.in_bounds == std::vector<bool>({false, true, false, true}), "negative coordinates");
  auto all_oob = ptx_tma_expand(d, 2, {9, 9, 0, 0, 0});
  check(all_oob.bytes == all_oob.oob_bytes, "all OOB");

  gpgpu_context context;
  ptx_cta_info cta(0, &context);
  std::array<std::array<unsigned char, 16>, 3> shared;
  for (auto &tile : shared) tile.fill(0xcc);
  unsigned reads = 0, writes = 0, completions = 0;
  std::vector<ptx_tma_target> targets;
  for (unsigned rank = 0; rank < 3; ++rank) {
    cta.mbarrier_init(8 * rank, 1);
    cta.mbarrier_expect_tx(8 * rank, 16);
    cta.mbarrier_arrive(8 * rank, 1, 0);
    targets.push_back({rank,
      [&, rank](size_t offset, const void *data, size_t bytes) {
        check(!cta.mbarrier_test_wait(8 * rank, 0, true), "completion before write");
        std::memcpy(shared[rank].data() + offset, data, bytes);
        ++writes;
      },
      [&, rank](size_t bytes) {
        check(writes == 2 && shared[rank][0] == 8 && shared[rank][4] == 0 &&
              shared[rank][8] == 24 && shared[rank][12] == 0, "shared write visibility");
        cta.mbarrier_complete_tx(8 * rank, bytes);
        ++completions;
      }});
  }
  cta.tma_submit([&]() {
    ptx_tma_complete(transfer, 5, targets,
      [&](uint64_t address, void *data, size_t bytes) {
        ++reads;
        for (size_t i = 0; i < bytes; ++i)
          static_cast<unsigned char *>(data)[i] = address - 0x1000 + i;
      });
  });
  check(reads == 0 && writes == 0 && completions == 0 &&
        !cta.mbarrier_test_wait(0, 0, true), "issue must not complete");
  cta.tma_progress();
  check(reads == 2 && writes == 2 && completions == 2 &&
        cta.mbarrier_test_wait(0, 0, true) && cta.mbarrier_test_wait(16, 0, true) &&
        !cta.mbarrier_test_wait(8, 0, true) && shared[1][0] == 0xcc, "multicast targets");
  bool rejected = false;
  try { ptx_tma_complete(transfer, 8, targets, [](uint64_t, void *, size_t) {}); }
  catch (const std::runtime_error &) { rejected = true; }
  check(rejected && completions == 2 && writes == 2, "unknown target must fail before write");
  rejected = false;
  try { ptx_tma_expand(d, 1, {}); }
  catch (const std::runtime_error &) { rejected = true; }
  check(rejected, "descriptor rank mismatch");
  std::printf("REFERENCE_RESULT status=PASS one_d=1 two_d=1 padded=1 negative_oob=1 all_oob=1 multicast_mask=5\n");
  std::printf("EVENT_RESULT issue=1 global_reads=%u shared_writes=%u completion=%u wait_before=0 wait_after=1\n", reads, writes, completions);
}
