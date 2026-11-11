#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

#include "../stage-1/host.h"
struct Launch {
  const char *name;
  const unsigned char *fatbin;
  std::vector<unsigned> shape;
  std::vector<std::vector<unsigned char>> args;
  std::vector<std::pair<unsigned, unsigned>> pointers;
};
#include "real_launches.inc"
static char markers[110];
static unsigned qwen_blocks = 8;
static bool single_cta = false;
static void run(unsigned module, std::initializer_list<const void *> ptrs) {
  auto spec = specs.at(module);
  if (module == 71)
    spec.shape[0] = qwen_blocks;
  if (single_cta)
    spec.shape[0] = 1;
  Wrapper wrapper{0x466243b1, 1, spec.fatbin, nullptr};
  void **h = __cudaRegisterFatBinary(&wrapper);
  __cudaRegisterFunction(h, &markers[module], const_cast<char *>(spec.name),
                         spec.name, -1, nullptr, nullptr, nullptr, nullptr,
                         nullptr);
  __cudaRegisterFatBinaryEnd(h);
  auto it = ptrs.begin();
  for (auto where : spec.pointers) {
    const void *p = *it++;
    std::memcpy(spec.args[where.first].data() + where.second, &p, 8);
  }
  std::vector<void *> args;
  for (auto &a : spec.args)
    args.push_back(a.data());
  auto &s = spec.shape;
  printf("REAL_MODULE %u grid=%u,%u,%u block=%u,%u,%u shared=%u\n", module,
         s[0], s[1], s[2], s[3], s[4], s[5], s[9]);
  stamp("kernel_begin");
  CHECK(cudaLaunchKernel(&markers[module], dim3(s[0], s[1], s[2]),
                         dim3(s[3], s[4], s[5]), args.data(), s[9]));
  CHECK(cudaDeviceSynchronize());
  stamp("kernel_done");
}
template <class T> static T *upload(const std::vector<T> &values) {
  T *p;
  CHECK(cudaMalloc(&p, values.size() * sizeof(T)));
  CHECK(cudaMemcpy(p, values.data(), values.size() * sizeof(T),
                   cudaMemcpyHostToDevice));
  return p;
}
template <class T>
static std::vector<T> download(T *p, size_t n, const char *name) {
  std::vector<T> v(n);
  CHECK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  std::ofstream f(std::string(name) + ".actual", std::ios::binary);
  f.write((char *)v.data(), n * sizeof(T));
  return v;
}
template <class T>
static unsigned compare(const std::vector<T> &actual,
                        const std::vector<T> &expected, const char *name) {
  unsigned errors = 0;
  stamp("compare_begin");
  for (size_t i = 0; i < actual.size(); ++i)
    if (actual[i] != expected[i]) {
      if (errors < 10)
        printf("MISMATCH %s index=%zu actual=%g expected=%g\n", name, i,
               double(actual[i]), double(expected[i]));
      ++errors;
    }
  stamp("compare_end");
  printf("%s %s elements=%zu errors=%u\n", errors ? "FAIL" : "PASS", name,
         actual.size(), errors);
  return errors;
}
static float expand(unsigned short v) {
  unsigned bits = unsigned(v) << 16;
  float f;
  memcpy(&f, &bits, 4);
  return f;
}
// CPU reference uses an explicit nearest-neighbor decision, independent of
// the simulator's integer rounding implementation.
static unsigned short round_bf16(float f) {
  unsigned bits;
  memcpy(&bits, &f, 4);
  unsigned hi = bits >> 16, lo = bits & 65535;
  if (lo > 32768 || (lo == 32768 && (hi & 1)))
    ++hi;
  return hi;
}
int main(int argc, char **argv) {
  stamp("main");
  bool qwen = !strcmp(argv[1], "qwen");
  bool phi = !strcmp(argv[1], "phi");
  qwen_blocks = argc > 2 ? std::stoul(argv[2]) : 8;
  unsigned qwen_iterations = argc > 3 ? std::stoul(argv[3]) : 16;
  single_cta = argc > 4 && std::stoul(argv[4]);
  bool single_iteration = argc > 5 && std::stoul(argv[5]);
  unsigned h = qwen ? 2048 : 4096, experts = qwen ? 64 : phi ? 16 : 8;
  unsigned reduction = qwen ? qwen_iterations * 128 : h;
  if (single_iteration)
    reduction = qwen ? 128 : 32;
  std::vector<float> x(128 * h), w(experts * h), want(128 * experts);
  for (unsigned t = 0; t < 128; ++t)
    for (unsigned k = 0; k < h; ++k)
      x[t * h + k] = int((3 * t + k) % 7) - 3;
  for (unsigned e = 0; e < experts; ++e)
    for (unsigned k = 0; k < h; ++k)
      w[e * h + k] = int((5 * e + 2 * k) % 11) - 5;
  for (unsigned t = 0; t < 128; ++t)
    for (unsigned e = 0; e < experts; ++e) {
      int sum = 0;
      for (unsigned k = 0; k < reduction; ++k)
        sum += int(x[t * h + k]) * int(w[e * h + k]);
      want[t * experts + e] =
          ((qwen && (qwen_blocks == 1 || single_cta) &&
            (t >= 32 || e >= 32)) || (!qwen && single_cta && t >= 32))
              ? -99999.f : sum;
    }
  std::vector<unsigned short> bf(x.size());
  for (size_t i = 0; i < x.size(); ++i)
    bf[i] = round_bf16(x[i]);
  // Module 71 reads FP32 weights first, BF16 tokens second. Module 2 reads
  // FP32 tokens first, FP32 weights second; the output of both is token-major.
  const void *xp = qwen ? (const void *)upload(bf) : (const void *)upload(x);
  auto wp = upload(w);
  auto yp = upload(std::vector<float>(want.size(), -99999.f));
  stamp("inputs_ready");
  if (qwen)
    run(71, {wp, xp, yp});
  else
    run(phi ? 41 : 2, {xp, wp, yp});
  auto actual = download(yp, want.size(), argv[1]);
  unsigned errors = compare(actual, want, "real gating");
  float max_error = 0;
  for (size_t i = 0; i < want.size(); ++i)
    max_error = std::max(max_error, std::abs(actual[i] - want[i]));
  printf("GATING max_abs_error=%g rows=128 cols=%u\n", max_error, experts);
  std::ofstream f(std::string(argv[1]) + ".expected", std::ios::binary);
  f.write((char *)want.data(), want.size() * 4);
  return errors ? 2 : 0;
}
