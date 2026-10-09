#include <cassert>
#include <cstdio>
#include "abstract_hardware_model.h"
int main() {
  mbarrier_t barrier(0x100, 32);
  barrier.expect_on(4);
  barrier.arrive_on(32);
  const unsigned token = 0;
  assert(barrier.get_pending_thread_count() == 0 && barrier.get_tx_count() == 4);
  assert(!barrier.test_wait(token, false));
  barrier.complete_on(4);
  assert(barrier.get_phase() == 1 && barrier.get_pending_thread_count() == 32);
  assert(barrier.test_wait(token, false));
  assert(barrier.test_wait(0, true));
  barrier.drop_on(32);
  barrier.arrive_on(32);
  assert(barrier.get_expected_arrival_thread_count() == 0 &&
         barrier.get_pending_thread_count() == 0);
  std::printf("STATE_RESULT status=PASS phase=%u pending=%u tx=%u expected=%u\n",
              barrier.get_phase(), barrier.get_pending_thread_count(),
              barrier.get_tx_count(), barrier.get_expected_arrival_thread_count());
  return 0;
}
