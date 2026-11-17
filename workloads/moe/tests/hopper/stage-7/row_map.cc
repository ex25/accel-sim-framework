#include "../stage-1/host.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

extern "C" const unsigned char row_fatbin[];
static char marker;
int main(int argc, char **argv) {
  const unsigned blocks = std::stoul(argv[1]);
  unsigned stride = 129;
  const int guard = 0x5a5a5a5a;
  std::vector<int> input(blocks * stride + 64, guard), expected = input;
  for (unsigned b = 0; b < blocks; ++b) {
    std::vector<std::pair<int, int>> rows;
    const unsigned counts[] = {0, 1, 7, 32, 64};
    unsigned count = counts[b % 5];
    for (unsigned t = 0; t < 64; ++t) {
      const unsigned shuffled = (t * 29 + b * 7) % 64;
      input[b * stride + t] = shuffled < count ? int(1000 + shuffled * 13) : -1;
      if (shuffled < count)
        rows.emplace_back(input[b * stride + t], t);
    }
    std::sort(rows.begin(), rows.end(), std::greater<std::pair<int, int>>());
    std::copy_n(input.begin() + b * stride, 64, expected.begin() + b * stride);
    for (unsigned i = 0; i < rows.size(); ++i) {
      expected[b * stride + i] = rows[i].first;
      expected[b * stride + 64 + i] = rows[i].second;
    }
    expected[b * stride + 128] = rows.size();
  }
  int *data;
  CHECK(cudaMalloc(&data, input.size() * sizeof(int)));
  CHECK(cudaMemcpy(data, input.data(), input.size() * sizeof(int),
                   cudaMemcpyHostToDevice));
  Wrapper wrapper{0x466243b1, 1, row_fatbin, nullptr};
  void **handle = __cudaRegisterFatBinary(&wrapper);
  const char *entry = "moe_kernel_cab47fdb58edf01c";
  __cudaRegisterFunction(handle, &marker, const_cast<char *>(entry), entry, -1,
                         nullptr, nullptr, nullptr, nullptr, nullptr);
  __cudaRegisterFatBinaryEnd(handle);
  void *args[] = {&data, &stride};
  CHECK(cudaLaunchKernel(&marker, dim3(blocks), dim3(128), args, 256));
  CHECK(cudaDeviceSynchronize());
  std::vector<int> output(input.size());
  CHECK(cudaMemcpy(output.data(), data, output.size() * sizeof(int),
                   cudaMemcpyDeviceToHost));
  unsigned errors = 0;
  for (unsigned i = 0; i < output.size(); ++i) {
    if (output[i] != expected[i]) {
      if (errors < 8)
        std::printf("MISMATCH index=%u expected=%d actual=%d\n", i, expected[i],
                    output[i]);
      ++errors;
    }
  }
  std::printf(
      "ROW blocks=%u threads=128 stride=%u shared=256 elements=%zu errors=%u\n",
      blocks, stride, output.size(), errors);
  CHECK(cudaFree(data));
  return errors ? 1 : 0;
}
