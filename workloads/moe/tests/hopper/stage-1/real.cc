#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

#include "host.h"
struct Launch {
  const char *name;
  const unsigned char *fatbin;
  std::vector<unsigned> shape;
  std::vector<std::vector<unsigned char>> args;
  std::vector<std::pair<unsigned, unsigned>> pointers;
};
#include "real_launches.inc"
static char markers[110];
static void run(unsigned module, std::initializer_list<const void *> ptrs) {
  auto spec = specs.at(module);
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
  for (auto &a : spec.args) args.push_back(a.data());
  auto &s = spec.shape;
  printf("REAL_MODULE %u grid=%u,%u,%u block=%u,%u,%u shared=%u\n", module,
         s[0], s[1], s[2], s[3], s[4], s[5], s[9]);
  stamp("kernel_begin");
  CHECK(cudaLaunchKernel(&markers[module], dim3(s[0], s[1], s[2]),
                         dim3(s[3], s[4], s[5]), args.data(), s[9]));
  CHECK(cudaDeviceSynchronize());
  stamp("kernel_done");
}
template <class T>
static T *upload(const std::vector<T> &values) {
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
  if (lo > 32768 || (lo == 32768 && (hi & 1))) ++hi;
  return hi;
}
int main(int argc, char **argv) {
  stamp("main");
  unsigned errors = 0;
  if (!strcmp(argv[1], "convert")) {
    const unsigned short patterns[] = {0, 0x3f80, 0xc000, 0x3f81};
    unsigned count;
    memcpy(&count, specs.at(70).args[2].data(), 4);
    std::vector<unsigned short> input(count);
    std::vector<unsigned> expected(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = patterns[i % 4];
      expected[i] = unsigned(input[i]) << 16;
    }
    auto in = upload(input);
    auto out = upload(std::vector<unsigned>(input.size(), 0xdeadbeef));
    run(70, {in, out});
    errors +=
        compare(download(out, input.size(), "convert"), expected, "convert");
  } else if (!strcmp(argv[1], "route")) {
    // Each token rotates a distinct permutation of 0..63, so value/index
    // pairing and the selected set both have an exact independent reference.
    std::vector<float> scores(128 * 64), want_values(128 * 4);
    std::vector<long long> want_indices(128 * 4);
    for (unsigned t = 0; t < 128; ++t) {
      for (unsigned e = 0; e < 64; ++e)
        scores[t * 64 + e] = float((e + 5 * t) % 64);
      for (unsigned j = 0; j < 4; ++j) {
        want_values[t * 4 + j] = 63 - j;
        want_indices[t * 4 + j] = (63 - j + 64 - (5 * t) % 64) % 64;
      }
    }
    auto in = upload(scores);
    auto values = upload(std::vector<float>(128 * 4, -1));
    auto indices = upload(std::vector<long long>(128 * 4, -1));
    run(3, {in, values, indices});
    auto a = download(values, 512, "topk-values");
    auto b = download(indices, 512, "topk-indices");
    auto sorted_a = a;
    auto sorted_b = b;
    for (unsigned t = 0; t < 128; ++t) {
      std::vector<std::pair<float, long long>> pairs;
      for (unsigned j = 0; j < 4; ++j)
        pairs.emplace_back(a[t * 4 + j], b[t * 4 + j]);
      std::sort(pairs.begin(), pairs.end(),
                std::greater<std::pair<float, long long>>());
      for (unsigned j = 0; j < 4; ++j) {
        sorted_a[t * 4 + j] = pairs[j].first;
        sorted_b[t * 4 + j] = pairs[j].second;
      }
    }
    errors += compare(sorted_a, want_values, "topk-value-set");
    errors += compare(sorted_b, want_indices, "topk-pair-set");
    run(4, {values, indices});
    errors += compare(download(values, 512, "sort-values"), want_values,
                      "sort-values");
    errors += compare(download(indices, 512, "sort-indices"), want_indices,
                      "sort-indices");
  } else if (!strcmp(argv[1], "mapping")) {
    std::vector<unsigned char> mask(128 * 64, 0);
    std::vector<long long> counts(64, 0);
    std::vector<int> prefix(128 * 129, -9), map(128 * 129, -9);
    for (unsigned t = 0; t < 128; ++t)
      for (unsigned j = 0; j < 4; ++j)
        mask[t * 64 + (t * t + 3 * t + 7 * j) % 64] = 1;
    for (unsigned e = 0; e < 64; ++e)
      for (unsigned t = 0; t < 128; ++t) {
        if (mask[t * 64 + e]) {
          ++counts[e];
          prefix[t * 129 + e] = counts[e];
          map[t * 129 + e] = counts[e] - 1;
        } else {
          prefix[t * 129 + e] = 0;
          map[t * 129 + e] = -1;
        }
      }
    auto routing = upload(mask);
    auto dc = upload(std::vector<long long>(64, -1));
    run(77, {routing, dc});
    errors += compare(download(dc, 64, "counts"), counts, "counts");
    auto dm = upload(std::vector<int>(128 * 129, -9));
    auto workspace = upload(std::vector<int>(64, -1));
    run(10, {routing, dm, workspace});
    errors += compare(download(dm, 128 * 129, "prefix"), prefix, "prefix");
    std::vector<int> count32(counts.begin(), counts.end());
    errors +=
        compare(download(workspace, 64, "tile-counts"), count32, "tile-counts");
    std::vector<long long> offsets(64);
    std::partial_sum(counts.begin(), counts.end() - 1, offsets.begin() + 1);
    auto global_map = map;
    for (unsigned t = 0; t < 128; ++t)
      for (unsigned e = 0; e < 64; ++e)
        if (map[t * 129 + e] >= 0) global_map[t * 129 + e] += offsets[e];
    run(38, {dm, workspace});
    errors += compare(download(dm, 128 * 129, "global-rows"), global_map,
                      "global-rows");
    run(78, {dm});
    auto packed = download(dm, 128 * 129, "packed-map");
    // A row stores local row ids [0..63], expert ids [64..127], and count at
    // 128. Sorting may permute equal row ids; validate pairs without imposing
    // a CPU tie order that the device does not promise.
    for (unsigned t = 0; t < 128; ++t) {
      std::vector<int> seen;
      if (packed[t * 129 + 128] != 4) ++errors;
      for (unsigned j = 0; j < 4; ++j) {
        int e = packed[t * 129 + 64 + j], row = packed[t * 129 + j];
        if (e < 0 || e >= 64 || !mask[t * 64 + e] ||
            row != global_map[t * 129 + e])
          ++errors;
        seen.push_back(e);
      }
      std::sort(seen.begin(), seen.end());
      if (std::unique(seen.begin(), seen.end()) != seen.end()) ++errors;
    }
    std::vector<unsigned short> input(128 * 2048), expected(512 * 2048);
    for (unsigned t = 0; t < 128; ++t)
      for (unsigned c = 0; c < 2048; ++c)
        // Each pair encodes the token and the full column coordinate. Both
        // words are finite BF16 bit patterns; rows never alias at token 32.
        input[t * 2048 + c] = c % 2 ? 0x6000 + c : 0x2000 + t * 128 + (c & 127);
    for (unsigned e = 0; e < 64; ++e)
      for (unsigned t = 0; t < 128; ++t)
        if (mask[t * 64 + e]) {
          size_t row = offsets[e] + map[t * 129 + e];
          std::copy_n(input.begin() + t * 2048, 2048,
                      expected.begin() + row * 2048);
        }
    auto data = upload(input);
    auto reordered =
        upload(std::vector<unsigned short>(expected.size(), 0xdead));
    run(79, {data, reordered, dm});
    errors += compare(download(reordered, expected.size(), "reorder"), expected,
                      "reorder");
    std::vector<long long> order(64);
    for (unsigned e = 0; e < 64; ++e) order[e] = 63 - e;
    auto dev_order = upload(order);
    auto inverse = upload(std::vector<int>(512, -1));
    run(80, {dc, dev_order, inverse});
    auto inv = download(inverse, 512, "inverse");
    std::vector<int> want_inv(512);
    long long destination = 0;
    for (unsigned e : order) {
      for (int row = 0; row < counts[e]; ++row)
        want_inv[offsets[e] + row] = destination + row;
      destination += counts[e];
    }
    errors += compare(inv, want_inv, "inverse");
    auto permuted =
        upload(std::vector<unsigned short>(expected.size(), 0xdead));
    run(81, {reordered, permuted, inverse});
    std::vector<unsigned short> want_permuted(expected.size());
    for (unsigned row = 0; row < 512; ++row)
      std::copy_n(expected.begin() + row * 2048, 2048,
                  want_permuted.begin() + want_inv[row] * 2048);
    errors += compare(download(permuted, expected.size(), "permuted"),
                      want_permuted, "permuted");
    printf("%s packed-pairs elements=512\n", errors ? "FAIL" : "PASS");
  } else if (!strcmp(argv[1], "swiglu")) {
    const float x[] = {0, 1, -1, 0.5f, -0.5f, 2, -2, 0.25f};
    const float y[] = {1, -1, 0.5f, 2, -2, 0.25f, -0.5f, 1.5f};
    std::vector<unsigned short> input(512 * 2816), expected(512 * 1408);
    for (unsigned row = 0; row < 512; ++row)
      for (unsigned c = 0; c < 1408; ++c) {
        float a = x[(row + c) % 8], b = y[(2 * row + c) % 8];
        input[row * 2816 + c] = round_bf16(a);
        input[row * 2816 + 1408 + c] = round_bf16(b);
        expected[row * 1408 + c] = round_bf16(a / (1 + std::exp(-a)) * b);
      }
    auto in = upload(input);
    auto out = upload(std::vector<unsigned short>(expected.size(), 0xdead));
    run(82, {in, out});
    auto actual = download(out, expected.size(), "swiglu");
    double max_error = 0;
    stamp("compare_begin");
    for (size_t i = 0; i < actual.size(); ++i) {
      double a = expand(actual[i]), b = expand(expected[i]),
             d = std::abs(a - b);
      max_error = std::max(max_error, d);
      if (!std::isfinite(a) || d > 1e-5 + 0.02 * std::abs(b)) {
        if (errors < 8) printf("MISMATCH swiglu %zu %g %g\n", i, a, b);
        ++errors;
      }
    }
    stamp("compare_end");
    printf("%s swiglu elements=%zu errors=%u max_abs_error=%g\n",
           errors ? "FAIL" : "PASS", actual.size(), errors, max_error);
  } else
    return 3;
  printf("%s real %s errors=%u\n", errors ? "FAIL" : "PASS", argv[1], errors);
  return errors ? 2 : 0;
}
