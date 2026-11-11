#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>

#include "../stage-1/host.h"

struct Launch {
  const char *name;
  const unsigned char *fatbin;
  std::vector<unsigned> shape;
  std::vector<std::vector<unsigned char>> args;
  std::vector<std::pair<unsigned, unsigned>> pointers;
};
#include "routing_launches.inc"

static char markers[110];

static void run(unsigned module, std::initializer_list<const void *> ptrs) {
  auto spec = specs.at(module);
  Wrapper wrapper{0x466243b1, 1, spec.fatbin, nullptr};
  void **handle = __cudaRegisterFatBinary(&wrapper);
  __cudaRegisterFunction(handle, &markers[module],
                         const_cast<char *>(spec.name), spec.name, -1, nullptr,
                         nullptr, nullptr, nullptr, nullptr);
  __cudaRegisterFatBinaryEnd(handle);

  auto pointer = ptrs.begin();
  for (auto [argument, offset] : spec.pointers) {
    const void *value = *pointer++;
    std::memcpy(spec.args[argument].data() + offset, &value, sizeof(value));
  }
  std::vector<void *> arguments;
  for (auto &argument : spec.args)
    arguments.push_back(argument.data());
  const auto &shape = spec.shape;
  CHECK(cudaLaunchKernel(&markers[module], dim3(shape[0], shape[1], shape[2]),
                         dim3(shape[3], shape[4], shape[5]), arguments.data(),
                         shape[9]));
  CHECK(cudaDeviceSynchronize());
}

template <class T> static T *upload(const std::vector<T> &values) {
  T *pointer;
  CHECK(cudaMalloc(&pointer, values.size() * sizeof(T)));
  CHECK(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T),
                   cudaMemcpyHostToDevice));
  return pointer;
}

template <class T>
static std::vector<T> download(T *pointer, size_t count, const char *name) {
  std::vector<T> values(count);
  CHECK(cudaMemcpy(values.data(), pointer, count * sizeof(T),
                   cudaMemcpyDeviceToHost));
  std::ofstream file(std::string(name) + ".actual", std::ios::binary);
  file.write(reinterpret_cast<const char *>(values.data()),
             values.size() * sizeof(T));
  return values;
}

static uint32_t mix_bits(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

static uint16_t round_bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t high = bits >> 16;
  uint32_t low = bits & 0xffff;
  if (low > 0x8000 || (low == 0x8000 && (high & 1)))
    ++high;
  return static_cast<uint16_t>(high);
}

static float expand_bf16(uint16_t value) {
  uint32_t bits = uint32_t(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

static uint16_t recipe_value(uint64_t index, uint32_t seed, float scale) {
  uint32_t key = uint32_t(index) ^ mix_bits(uint32_t(index >> 32)) ^ seed;
  float value = (int(mix_bits(key) % 17) - 8) * scale;
  return round_bf16(value);
}

static unsigned compare_exact(const std::vector<float> &actual,
                              const std::vector<float> &expected,
                              const char *name) {
  unsigned errors = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      if (errors < 8)
        printf("MISMATCH %s index=%zu actual=%.9g expected=%.9g\n", name, i,
               actual[i], expected[i]);
      ++errors;
    }
  }
  printf("%s %s elements=%zu errors=%u\n", errors ? "FAIL" : "PASS", name,
         actual.size(), errors);
  return errors;
}

int main(int argc, char **argv) {
  bool cpu_scores = argc > 1 && !std::strcmp(argv[1], "cpu-scores");
  constexpr unsigned tokens = 128;
  constexpr unsigned hidden = 2048;
  constexpr unsigned experts = 64;
  std::vector<uint16_t> input(tokens * hidden), router(experts * hidden);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = recipe_value(i, 2026, 1.0f / 16.0f);
  for (size_t i = 0; i < router.size(); ++i)
    router[i] = recipe_value(i, 2027, 1.0f / 128.0f);

  auto input_device = upload(input);
  auto router_device = upload(router);
  auto router_fp32_device = upload(std::vector<float>(router.size(), -1.0f));
  if (!cpu_scores)
    run(70, {router_device, router_fp32_device});

  std::vector<float> router_expected(router.size());
  for (size_t i = 0; i < router.size(); ++i)
    router_expected[i] = expand_bf16(router[i]);
  unsigned errors =
      cpu_scores ? 0
                 : compare_exact(download(router_fp32_device, router.size(),
                                          "router-convert"),
                                 router_expected, "router-convert");

  auto scores_device = upload(std::vector<float>(tokens * experts, -12345.0f));
  if (!cpu_scores)
    run(71, {router_fp32_device, input_device, scores_device});

  std::vector<float> scores_expected(tokens * experts, 0.0f);
  for (unsigned token = 0; token < tokens; ++token) {
    for (unsigned expert = 0; expert < experts; ++expert) {
      double sum = 0.0;
      for (unsigned k = 0; k < hidden; ++k)
        sum += double(expand_bf16(input[token * hidden + k])) *
               expand_bf16(router[expert * hidden + k]);
      scores_expected[token * experts + expert] = static_cast<float>(sum);
    }
  }
  if (cpu_scores)
    CHECK(cudaMemcpy(scores_device, scores_expected.data(),
                     scores_expected.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
  errors += compare_exact(
      download(scores_device, scores_expected.size(), "router-scores"),
      scores_expected, "router-scores");

  auto values_device = upload(std::vector<float>(tokens * 4, -1.0f));
  auto indices_device = upload(std::vector<long long>(tokens * 4, -1));
  run(3, {scores_device, values_device, indices_device});
  auto values = download(values_device, tokens * 4, "topk-values");
  auto indices = download(indices_device, tokens * 4, "topk-indices");
  unsigned topk_errors = 0;
  unsigned cutoff_ties = 0;
  std::vector<float> expected_values(tokens * 4);
  for (unsigned token = 0; token < tokens; ++token) {
    std::vector<std::pair<float, unsigned>> ranked;
    for (unsigned expert = 0; expert < experts; ++expert)
      ranked.emplace_back(scores_expected[token * experts + expert], expert);
    std::sort(ranked.begin(), ranked.end(), std::greater<>());
    expected_values[token * 4 + 0] = ranked[0].first;
    expected_values[token * 4 + 1] = ranked[1].first;
    expected_values[token * 4 + 2] = ranked[2].first;
    expected_values[token * 4 + 3] = ranked[3].first;
    if (ranked[3].first == ranked[4].first)
      ++cutoff_ties;

    std::vector<float> selected(values.begin() + token * 4,
                                values.begin() + token * 4 + 4);
    std::sort(selected.begin(), selected.end(), std::greater<>());
    std::vector<unsigned> seen;
    for (unsigned j = 0; j < 4; ++j) {
      long long expert = indices[token * 4 + j];
      if (expert < 0 || expert >= experts ||
          scores_expected[token * experts + expert] != values[token * 4 + j])
        ++topk_errors;
      if (expert >= 0 && expert < experts)
        seen.push_back(expert);
    }
    std::sort(seen.begin(), seen.end());
    if (std::adjacent_find(seen.begin(), seen.end()) != seen.end())
      ++topk_errors;
    for (unsigned j = 0; j < 4; ++j)
      if (selected[j] != expected_values[token * 4 + j])
        ++topk_errors;
  }
  printf("%s topk-values elements=%u errors=%u cutoff_tie_tokens=%u\n",
         topk_errors ? "FAIL" : "PASS", tokens * 4, topk_errors, cutoff_ties);
  errors += topk_errors;

  run(4, {values_device, indices_device});
  values = download(values_device, tokens * 4, "sorted-topk-values");
  indices = download(indices_device, tokens * 4, "sorted-topk-indices");
  unsigned sort_errors = 0;
  for (unsigned token = 0; token < tokens; ++token) {
    std::vector<unsigned> seen;
    for (unsigned j = 0; j < 4; ++j) {
      long long expert = indices[token * 4 + j];
      if (expert < 0 || expert >= experts ||
          scores_expected[token * experts + expert] != values[token * 4 + j])
        ++sort_errors;
      if (expert >= 0 && expert < experts)
        seen.push_back(expert);
      if (values[token * 4 + j] != expected_values[token * 4 + j])
        ++sort_errors;
      if (j && values[token * 4 + j] > values[token * 4 + j - 1])
        ++sort_errors;
    }
    std::sort(seen.begin(), seen.end());
    if (std::adjacent_find(seen.begin(), seen.end()) != seen.end())
      ++sort_errors;
  }
  printf("%s topk-sort elements=%u errors=%u\n", sort_errors ? "FAIL" : "PASS",
         tokens * 4, sort_errors);
  errors += sort_errors;

  auto exponent_device = upload(std::vector<float>(tokens * 4, -1.0f));
  run(72, {values_device, exponent_device});
  auto exponent = download(exponent_device, tokens * 4, "softmax-exponent");
  unsigned exponent_errors = 0;
  for (unsigned token = 0; token < tokens; ++token) {
    for (unsigned j = 0; j < 4; ++j) {
      float expected = std::exp(values[token * 4 + j] - values[token * 4]);
      if (std::abs(exponent[token * 4 + j] - expected) > 1e-6f)
        ++exponent_errors;
    }
  }
  printf("%s softmax-exponent elements=%u errors=%u\n",
         exponent_errors ? "FAIL" : "PASS", tokens * 4, exponent_errors);
  errors += exponent_errors;

  auto weights_device = upload(std::vector<float>(tokens * experts, -1.0f));
  auto mask32_device = upload(std::vector<unsigned>(tokens * experts, 999));
  run(73, {weights_device});
  run(74, {mask32_device});
  run(75, {indices_device, exponent_device, weights_device, mask32_device});
  auto weights = download(weights_device, tokens * experts, "routing-weights");
  auto mask32 = download(mask32_device, tokens * experts, "routing-mask32");
  std::vector<float> weights_expected(tokens * experts, 0.0f);
  std::vector<unsigned> mask_expected(tokens * experts, 0);
  for (unsigned token = 0; token < tokens; ++token) {
    float sum = 0.0f;
    for (unsigned j = 0; j < 4; ++j)
      sum += exponent[token * 4 + j];
    for (unsigned j = 0; j < 4; ++j) {
      unsigned expert = indices[token * 4 + j];
      weights_expected[token * experts + expert] =
          exponent[token * 4 + j] / sum;
      mask_expected[token * experts + expert] = 1;
    }
  }
  unsigned weights_errors = 0;
  unsigned mask_errors = 0;
  for (size_t i = 0; i < weights.size(); ++i) {
    if (std::abs(weights[i] - weights_expected[i]) > 1e-7f)
      ++weights_errors;
    if (mask32[i] != mask_expected[i])
      ++mask_errors;
  }
  printf("%s routing-weights elements=%zu errors=%u\n",
         weights_errors ? "FAIL" : "PASS", weights.size(), weights_errors);
  printf("%s routing-mask32 elements=%zu errors=%u\n",
         mask_errors ? "FAIL" : "PASS", mask32.size(), mask_errors);
  errors += weights_errors + mask_errors;

  auto mask8_device = upload(std::vector<unsigned char>(tokens * experts, 99));
  auto counts_device = upload(std::vector<long long>(experts, -1));
  run(76, {mask32_device, mask8_device});
  run(77, {mask8_device, counts_device});
  auto mask8 = download(mask8_device, tokens * experts, "routing-mask8");
  auto counts = download(counts_device, experts, "expert-counts");
  unsigned counts_errors = 0;
  unsigned mask8_errors = 0;
  for (unsigned expert = 0; expert < experts; ++expert) {
    long long expected_count = 0;
    for (unsigned token = 0; token < tokens; ++token) {
      expected_count += mask_expected[token * experts + expert];
      if (mask8[token * experts + expert] !=
          mask_expected[token * experts + expert])
        ++mask8_errors;
    }
    if (counts[expert] != expected_count)
      ++counts_errors;
  }
  printf("%s routing-mask8 elements=%zu errors=%u\n",
         mask8_errors ? "FAIL" : "PASS", mask8.size(), mask8_errors);
  printf("%s expert-counts elements=%zu errors=%u\n",
         counts_errors ? "FAIL" : "PASS", counts.size(), counts_errors);
  errors += mask8_errors + counts_errors;
  return errors ? 1 : 0;
}
