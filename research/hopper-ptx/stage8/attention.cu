// Portable FP16-input/FP32-accumulate attention. No Tensor Core/FA2/FA3 claim.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
__global__ void attention(const half *q, const half *k, const half *v, half *out, int seq) {
  __shared__ float query[128], weights[256], denom;
  const int lane = threadIdx.x, row = blockIdx.x, head = blockIdx.y;
  if (lane < 128) query[lane] = __half2float(q[(row * 8 + head) * 128 + lane]);
  __syncthreads();
  if (lane < seq) {
    float score = 0;
    for (int d = 0; d < 128; ++d)
      score = fmaf(query[d], __half2float(k[(lane * 8 + head) * 128 + d]), score);
    weights[lane] = score * (1.f / sqrtf(128.f));
  }
  __syncthreads();
  if (lane == 0) {
    float maxval = weights[0], sum = 0;
    for (int j = 1; j < seq; ++j) maxval = fmaxf(maxval, weights[j]);
    for (int j = 0; j < seq; ++j) { weights[j] = expf(weights[j] - maxval); sum += weights[j]; }
    denom = sum;
  }
  __syncthreads();
  if (lane < 128) {
    float value = 0;
    for (int j = 0; j < seq; ++j)
      value = fmaf(weights[j], __half2float(v[(j * 8 + head) * 128 + lane]), value);
    out[(row * 8 + head) * 128 + lane] = __float2half_rn(value / denom);
  }
}
static std::vector<half> load(const std::string &path, size_t count) {
  std::vector<half> data(count);
  std::ifstream file(path, std::ios::binary);
  if (!file.read(reinterpret_cast<char *>(data.data()), count * sizeof(half)) || file.peek() != EOF)
    throw std::runtime_error("input length mismatch: " + path);
  return data;
}
int main(int argc, char **argv) {
  if (argc != 5 && argc != 7) { fprintf(stderr, "usage: attention CASE_DIR SEQ REPETITIONS OUTPUT [ROWS HEADS]\n"); return 2; }
  try {
    const int seq = std::stoi(argv[2]), reps = std::stoi(argv[3]);
    const int rows = argc == 7 ? std::stoi(argv[5]) : seq;
    const int heads = argc == 7 ? std::stoi(argv[6]) : 8;
    if ((seq != 64 && seq != 128 && seq != 256) || reps < 1) throw std::runtime_error("invalid shape/repetitions");
    if (rows < 1 || rows > seq || heads < 1 || heads > 8) throw std::runtime_error("invalid output extent");
    const size_t count = seq * 8 * 128, bytes = count * sizeof(half);
    const std::string dir = argv[1];
    auto q = load(dir + "/q.f16le", count), k = load(dir + "/k.f16le", count), v = load(dir + "/v.f16le", count);
    std::vector<half> out(count);
    cudaDeviceProp prop; check(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 8 || prop.minor != 0 || std::string(prop.name).find("A800") == std::string::npos)
      throw std::runtime_error("this A800 validation requires NVIDIA A800 sm_80");
    half *dq, *dk, *dv, *dout;
    check(cudaMalloc(&dq, bytes)); check(cudaMalloc(&dk, bytes));
    check(cudaMalloc(&dv, bytes)); check(cudaMalloc(&dout, bytes));
    const int threads = std::max(128, seq); const dim3 grid(rows, heads);
    cudaFuncAttributes attr; check(cudaFuncGetAttributes(&attr, attention));
    int active_blocks; check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, attention, threads, 0));
    cudaEvent_t start, end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
    check(cudaMemcpy(dq, q.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dk, k.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dv, v.data(), bytes, cudaMemcpyHostToDevice));
    // A single invocation is suitable for ncu/NVBit; normal timing uses warmup.
    if (reps > 1) for (int i = 0; i < 10; ++i) attention<<<grid, threads>>>(dq, dk, dv, dout, seq);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    std::vector<float> times;
    for (int i = 0; i < reps; ++i) {
      check(cudaEventRecord(start)); attention<<<grid, threads>>>(dq, dk, dv, dout, seq);
      check(cudaGetLastError()); check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
      float ms; check(cudaEventElapsedTime(&ms, start, end)); times.push_back(ms * 1000.f);
    }
    // Separate host interval: 3 H2D copies + one kernel + one D2H copy.
    const auto host_start = std::chrono::steady_clock::now();
    check(cudaMemcpy(dq, q.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dk, k.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dv, v.data(), bytes, cudaMemcpyHostToDevice));
    attention<<<grid, threads>>>(dq, dk, dv, dout, seq); check(cudaGetLastError());
    check(cudaMemcpy(out.data(), dout, bytes, cudaMemcpyDeviceToHost));
    const double total_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - host_start).count();
    std::ofstream file(argv[4], std::ios::binary); file.write(reinterpret_cast<char *>(out.data()), bytes);
    if (!file) throw std::runtime_error("output write failed");
    std::sort(times.begin(), times.end());
    // Logical global bytes are algorithmic loads/stores, not physical DRAM.
    const unsigned long long logical_bytes = 4ULL * heads * rows * 128 * (seq + 1);
    printf("A800_RESULT {\"status\":\"PASS\",\"device\":\"%s\",\"cc\":\"%d.%d\",\"sm_count\":%d,\"seq\":%d,\"rows\":%d,\"heads\":%d,\"threads\":%d,\"repetitions\":%d,\"kernel_us_min\":%.6f,\"kernel_us_median\":%.6f,\"kernel_us_max\":%.6f,\"transfer_kernel_host_us\":%.6f,\"registers\":%d,\"static_shared_bytes\":%zu,\"theoretical_occupancy\":%.6f,\"logical_global_bytes\":%llu,\"io_bytes\":%zu,\"gemm_flops\":%llu}\n",
      prop.name,prop.major,prop.minor,prop.multiProcessorCount,seq,rows,heads,threads,reps,times.front(),times[times.size()/2],times.back(),total_us,
      attr.numRegs,attr.sharedSizeBytes,double(active_blocks*threads)/prop.maxThreadsPerMultiProcessor,logical_bytes,4*bytes,4ULL*heads*rows*seq*128);
    check(cudaFree(dq)); check(cudaFree(dk)); check(cudaFree(dv)); check(cudaFree(dout));
    check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
    return 0;
  } catch (const std::exception &error) { fprintf(stderr,"A800_FAILURE %s\n",error.what()); return 1; }
}
