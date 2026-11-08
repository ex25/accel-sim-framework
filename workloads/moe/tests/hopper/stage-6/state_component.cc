// Controlled completions exercise the same group and quota state as real PTX.
#include "libcuda/gpgpu_context.h"
#include "src/cuda-sim/ptx_ir.h"
#include <cassert>
#include <cstdio>
int main() {
  Scoreboard scoreboard(0, 1, nullptr);
  shader_core_config config(GPGPU_Context());
  config.warp_size = 32;
  warp_inst_t matrix(&config);
  matrix.incount = 0;
  matrix.pred = matrix.ar1 = matrix.ar2 = 0;
  matrix.m_ptx_matrix = PTX_WGMMA;
  matrix.outcount = 64;
  for (unsigned j = 0; j < 64; ++j)
    matrix.out[j] = j + 1;
  // warp_inst_t is an actual issued warp, not a hand-wired set of reservations.
  active_mask_t mask;
  mask.set();
  matrix.issue(mask, 0, 0, 0, 0, 0);
  scoreboard.reserveRegisters(&matrix);
  scoreboard.reserveRegisters(&matrix);
  inst_t consumer;
  consumer.pred = consumer.ar1 = consumer.ar2 = 0;
  consumer.incount = 1;
  consumer.in[0] = 64;
  consumer.outcount = 0;
  assert(!scoreboard.checkCollision(0, &matrix));
  assert(scoreboard.checkCollision(0, &consumer));
  scoreboard.releaseRegisters(&matrix);
  assert(scoreboard.pendingWrites(0) &&
         scoreboard.checkCollision(0, &consumer));
  scoreboard.releaseRegisters(&matrix);
  assert(!scoreboard.pendingWrites(0) &&
         !scoreboard.checkCollision(0, &consumer));
  puts("COMPONENT scoreboard 64 targets two writers final dependency PASS");
  ptx_matrix_warpgroup group;
  auto first = std::make_shared<ptx_matrix_op>();
  auto second = std::make_shared<ptx_matrix_op>();
  group.groups.push_back({{first}});
  group.groups.push_back({{second}});
  shd_warp_t warp(nullptr, 32);
  auto copy = warp.m_ptx_cp_async_groups[0].add();
  warp.m_ptx_cp_async_groups[0].commit();
  auto bulk = warp.m_ptx_bulk_groups[0].add();
  warp.m_ptx_bulk_groups[0].commit();
  for (unsigned cycle = 0; cycle < 12; ++cycle) {
    if (cycle == 3)
      first->pending = 8; // Three warps written; fourth still pending.
    if (cycle == 4)
      first->pending = 0;
    if (cycle == 9)
      second->pending = 0;
    assert(group.waiting(1) == (cycle < 4));
    assert(group.waiting(0) == (cycle < 9));
    // Matrix completion must not alter either copy queue.
    warp.m_ptx_cp_async_groups[0].wait(0);
    warp.m_ptx_bulk_groups[0].wait(0);
    assert(warp.m_ptx_cp_async_groups[0].waiting());
    assert(warp.m_ptx_bulk_groups[0].waiting());
  }
  warp.m_ptx_cp_async_groups[0].complete(copy);
  warp.m_ptx_bulk_groups[0].complete(bulk);
  assert(!warp.m_ptx_cp_async_groups[0].waiting());
  assert(!warp.m_ptx_bulk_groups[0].waiting());
  puts("COMPONENT wait1=4 wait0=9 last_warp=4 copy_queues_unchanged PASS");
  ptx_matrix_cta cta, neighbour;
  cta.initial = neighbour.initial = 168;
  for (unsigned i = 0; i < 3; ++i) {
    auto g = std::make_shared<ptx_matrix_warpgroup>();
    g->quota = 168;
    cta.groups[i] = g;
    if (i) {
      g->quota_request = std::make_shared<ptx_matrix_sync>();
      g->quota_request->argument = 232;
    }
  }
  cta.allocate();
  assert(cta.free == 0);
  for (unsigned i = 1; i < 3; ++i)
    assert(cta.groups[i]->quota_request);
  cta.free = 16384;
  cta.groups[0]->quota = 40;
  cta.allocate();
  assert(cta.free == 0 && neighbour.free == 0);
  assert(cta.groups[0]->quota * 128 == 5120);
  for (unsigned i = 1; i < 3; ++i) {
    assert(cta.groups[i]->quota * 128 == 29696);
    assert(!cta.groups[i]->quota_request);
  }
  assert(128 * (cta.groups[0]->quota + cta.groups[1]->quota +
                cta.groups[2]->quota) ==
         64512);
  puts("COMPONENT quota initial=168 pool=16384 final=40/232/232 total=64512 "
       "PASS");
}
