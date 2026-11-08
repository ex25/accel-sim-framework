#include "cuda-sim/cp_async.h"
#include <cstdio>
int main() {
  for (unsigned n : {0, 2, 4, 6}) {
    cp_async_groups g;
    unsigned count = n + 2;
    for (unsigned i = 0; i < count; ++i) {
      assert(g.add() == i);
      g.commit();
    }
    g.wait(n);
    assert(g.waiting());
    for (unsigned i = 2; i < count; ++i)
      g.complete(i);
    assert(g.waiting());
    g.complete(0);
    assert(g.waiting());
    g.complete(1);
    assert(!g.waiting());
    cp_async_groups h;
    for (unsigned i = 0; i < count; ++i) {
      h.add();
      h.commit();
    }
    h.wait(n);
    h.complete(0);
    h.complete(1);
    assert(!h.waiting()); // newest N may remain incomplete
    auto open = h.add();
    h.wait(0);
    for (unsigned i = 2; i < count; ++i)
      h.complete(i);
    assert(!h.waiting()); // uncommitted copy is outside wait0
    h.commit();
    h.wait(0);
    assert(h.waiting());
    h.complete(open);
    assert(!h.waiting());
    h.commit(); // empty group still advances the sequence
    assert(h.committed == count + 2);
    printf("PASS component wait%u reverse completion / recent groups / open / "
           "empty\n",
           n);
  }
}
