#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call) do { auto e = (call); if (e != cudaSuccess) { \
 fprintf(stderr, "%s failed: %d\n", #call, int(e)); exit(1); } } while (0)
extern "C" void **__cudaRegisterFatBinary(void *);
extern "C" void __cudaRegisterFatBinaryEnd(void **);
extern "C" void __cudaRegisterFunction(void **, const char *, char *, const char *,
                                      int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
extern "C" const unsigned char module_a[], module_b[];
struct Wrapper { unsigned magic, version; const void *data; void *filename; };
static char markers[6];
static void stamp(const char *phase) {
  printf("HOST_TIME %s %.6f\n", phase,
         std::chrono::duration<double>(std::chrono::steady_clock::now()
                                     .time_since_epoch()).count());
  fflush(stdout);
}
__attribute__((constructor(101))) static void startup_stamp() {
  stamp("process_init_begin");
}
__global__ void launch_test(unsigned *output, unsigned tag, unsigned iterations) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned v = id;
  for (unsigned i = 0; i < iterations; ++i) v = v * 3 + 1;
  output[id * 5] = gridDim.x;
  output[id * 5 + 1] = blockDim.x;
  output[id * 5 + 2] = tag;
  output[id * 5 + 3] = id;
  output[id * 5 + 4] = v;
}
__global__ void stream_test(unsigned *values) {
  unsigned id = threadIdx.x;
  values[id] = values[id] * 3 + 7;
}
static void launch(const char *name, const void *f, void **args, unsigned grid=1,
                   unsigned block=32, size_t shared=0, cudaStream_t stream=0) {
  stamp(name);
  CHECK(cudaLaunchKernel(f, dim3(grid), dim3(block), args, shared, stream));
  CHECK(cudaStreamSynchronize(stream));
  stamp("kernel_done");
}
int main(int argc, char **argv) {
  stamp("start");
  const char *test = argc > 1 ? argv[1] : "all";
  if (!strcmp(test,"launch") || !strcmp(test,"work")) {
    unsigned grids[] = {2,3,1,4}, blocks[] = {32,64,32,32};
    unsigned iterations = !strcmp(test,"work") ? 100 : 0;
    for (unsigned k=0; k<4; ++k) {
      unsigned n = grids[k] * blocks[k], tag = 11*(k+1);
      unsigned *d; CHECK(cudaMalloc(&d,n*5*4));
      void *args[] = {&d,&tag,&iterations};
      stamp("launch_begin");
      if (k%2) launch_test<<<grids[k],blocks[k]>>>(d,tag,iterations);
      else CHECK(cudaLaunchKernel((const void *)launch_test,grids[k],blocks[k],args));
      CHECK(cudaDeviceSynchronize()); stamp("launch_done");
      std::vector<unsigned> result(n*5);
      CHECK(cudaMemcpy(result.data(),d,result.size()*4,cudaMemcpyDeviceToHost));
      stamp("compare_begin");
      for(unsigned i=0;i<n;++i) {
        unsigned v=i; for(unsigned j=0;j<iterations;++j) v=v*3+1;
        unsigned expected[]={grids[k],blocks[k],tag,i,v};
        if(memcmp(result.data()+i*5,expected,20)) return 2;
      }
      printf("PASS launch %u threads=%u tag=%u iterations=%u\n",k,n,tag,iterations);
      stamp("compare_end");
      CHECK(cudaFree(d));
    }
    cudaStream_t stream; CHECK(cudaStreamCreate(&stream));
    unsigned *d; CHECK(cudaMalloc(&d,32*4));
    std::vector<unsigned> result(32);
    for(unsigned i=0;i<32;++i) result[i]=55+i;
    CHECK(cudaMemcpyAsync(d,result.data(),32*4,cudaMemcpyHostToDevice,stream));
    // Nonzero shared and stream also exercise all four compiler Pop outputs.
    stream_test<<<1,32,128,stream>>>(d);
    CHECK(cudaMemcpyAsync(result.data(),d,32*4,cudaMemcpyDeviceToHost,stream));
    CHECK(cudaStreamSynchronize(stream));
    stamp("compare_begin");
    for(unsigned i=0;i<32;++i) if(result[i]!=(55+i)*3+7) return 3;
    puts("PASS nondefault stream copy/kernel/copy");
    stamp("compare_end");
    CHECK(cudaStreamDestroy(stream)); CHECK(cudaFree(d));
    return 0;
  }
  Wrapper wrappers[]={{0x466243b1,1,module_a,nullptr},{0x466243b1,1,module_b,nullptr}};
  const char *names[]={"write_a","mixed","large","shared_test","failure","write_b"};
  for(unsigned m=0;m<2;++m) {
    void **h=__cudaRegisterFatBinary(&wrappers[m]);
    for(unsigned i=m?5:0;i<(m?6:5);++i)
      __cudaRegisterFunction(h,&markers[i],const_cast<char *>(names[i]),names[i],
                             -1,nullptr,nullptr,nullptr,nullptr,nullptr);
    __cudaRegisterFatBinaryEnd(h);
  }
  stamp("registration_done");
  unsigned char *d; CHECK(cudaMalloc(&d,4096));
  if (!strcmp(test,"modules")) {
    unsigned result[3]; unsigned indices[]={0,5,0};
    for(unsigned i=0;i<3;++i) {
      auto p=d+4*i; void *args[]={&p}; launch(names[indices[i]],&markers[indices[i]],args);
    }
    CHECK(cudaMemcpy(result,d,12,cudaMemcpyDeviceToHost));
    unsigned expected[]={11,22,11};
    if(memcmp(result,expected,12)) return 4;
    printf("PASS modules [%u,%u,%u]\n",result[0],result[1],result[2]);
  } else if (!strcmp(test,"parameters")) {
    unsigned head=0x13579bdf; unsigned long long tail=0x0123456789abcdefULL;
    unsigned char payload[1344]; for(unsigned i=0;i<1344;++i) payload[i]=(7*i+3)%256;
    void *args[]={&head,payload,&tail,&d}; launch("mixed",&markers[1],args);
    unsigned char result[1344]; CHECK(cudaMemcpy(result,d,144,cudaMemcpyDeviceToHost));
    if(memcmp(result,&head,4)||memcmp(result+4,payload,128)||memcmp(result+136,&tail,8)) return 5;
    puts("PASS mixed head,128 bytes,tail");
    void *large_args[]={payload,&d}; launch("large",&markers[2],large_args);
    CHECK(cudaMemcpy(result,d,1344,cudaMemcpyDeviceToHost));
    if(memcmp(result,payload,1344)) return 6;
    for(unsigned i: {0,63,64,127,128,1128,1136,1240,1343}) printf("BYTE %u=%u\n",i,result[i]);
    puts("PASS all 1344 bytes");
  } else if (!strcmp(test,"shared")) {
    CHECK(cudaFuncSetAttribute(&markers[3],cudaFuncAttributeMaxDynamicSharedMemorySize,65536));
    for(unsigned bytes: {32768,65536,32768}) {
      void *args[]={&d,&bytes}; launch("shared_test",&markers[3],args,32,32,bytes);
      unsigned result[96]; CHECK(cudaMemcpy(result,d,sizeof(result),cudaMemcpyDeviceToHost));
      for(unsigned i=0;i<32;++i) if(result[3*i]!=101||result[3*i+1]!=202||result[3*i+2]!=77) return 7;
      printf("PASS shared D=%u all 32 CTAs [101,202,77]\n",bytes);
    }
  } else {
    unsigned mode=!strcmp(test,"assert")?1:!strcmp(test,"trap")?2:0;
    void *args[]={&d,&mode}; launch("failure",&markers[4],args);
    unsigned result; CHECK(cudaMemcpy(&result,d,4,cudaMemcpyDeviceToHost));
    if(mode || result!=123) return 8;
    puts("PASS normal noreturn declaration marker=123");
  }
  CHECK(cudaFree(d)); stamp("comparison_done");
}
