#!/usr/bin/env python3
"""Generate IMAD, logic, SHFL, FMA and shared-read resource probes.

The native compiler fuses some pure XORs; inspect SASS/dynamic counters before
using that pattern for a one-instruction throughput comparison. Use core cycles,
not legacy simulator clock64 values, which currently include a factor of four.
"""
from pathlib import Path
import sys
P=Path(sys.argv[1]).resolve()
P.mkdir(parents=True, exist_ok=True)
ops={
 'imad':'asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(r[J]) : "r"(seed|1),"r"(seed));',
 'xor':'asm volatile("xor.b32 %0,%0,%1;" : "+r"(r[J]) : "r"(r[N]));',
 'shfl':'asm volatile("shfl.sync.bfly.b32 %0,%0,1,31,-1;" : "+r"(r[J]));',
 'fma':'asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(f[J]) : "f"(scale),"f"(bias));',
 'lds':'asm volatile("ld.volatile.shared.u32 %0,[%1];" : "=r"(s[J]) : "r"(addr));',
}
patterns=['imad','xor','shfl','fma','lds','imad_xor','imad_shfl','xor_shfl','imad_fma','shfl_lds','imad_lds']
text='''#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstring>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\\n",#x,cudaGetErrorString(e));return 1;} }while(0)
'''
for name in patterns:
 body=''
 for i in range(64):
  op=name.split('_')[i%len(name.split('_'))];j=(i//len(name.split('_')))%8
  body+=ops[op].replace('J',str(j)).replace('N',str((j+1)%8))+'\n'
 text+=f'''__global__ void pipe_{name}(uint32_t *output, unsigned long long *times, uint32_t seed, float scale, float bias, unsigned loops){{
__shared__ uint32_t shared[1024];
unsigned t=threadIdx.x; shared[t]=seed+t; __syncthreads();
unsigned addr=__cvta_generic_to_shared(shared+t);
uint32_t r[8],s[8];float f[8];
#pragma unroll
for(unsigned j=0;j<8;++j){{r[j]=seed+t+j;s[j]=0;f[j]=1.0f+j;}}
unsigned long long begin=clock64();
for(unsigned k=0;k<loops;++k){{
{body}
}}
unsigned long long end=clock64();
#pragma unroll
for(unsigned j=0;j<8;++j){{output[t*24+j]=r[j];output[t*24+8+j]=s[j];output[t*24+16+j]=__float_as_uint(f[j]);}}
times[t]=end-begin;
}}
'''
text+='''int main(int argc,char **argv){
unsigned loops=argc>1?std::stoul(argv[1]):128; unsigned threads=512; unsigned repeats=argc>2?std::stoul(argv[2]):4;
uint32_t *out;unsigned long long *times;
CK(cudaMalloc(&out,threads*24*4));CK(cudaMalloc(&times,threads*8));
uint32_t seed=7;float scale=1.0f,bias=0.0009765625f;
std::vector<uint32_t> result(threads*24);std::vector<unsigned long long> cycles(threads);
const char *names[]={'''+','.join('"'+x+'"' for x in patterns)+'''};
'''
for n,name in enumerate(patterns):
 text+=f'''for(unsigned rep=0;rep<repeats;++rep){{
pipe_{name}<<<1,threads>>>(out,times,seed,scale,bias,loops);CK(cudaDeviceSynchronize());
CK(cudaMemcpy(result.data(),out,result.size()*4,cudaMemcpyDeviceToHost));
CK(cudaMemcpy(cycles.data(),times,cycles.size()*8,cudaMemcpyDeviceToHost));
unsigned failures=0;
for(unsigned t=0;t<threads;++t){{
 uint32_t r[2][8],s[8]={{}};float f[8];for(unsigned j=0;j<8;++j){{r[0][j]=seed+t+j;r[1][j]=seed+(t^1)+j;f[j]=1.0f+j;}}
 for(unsigned k=0;k<loops;++k){{
'''
 for i in range(64):
  op=name.split('_')[i%len(name.split('_'))];j=(i//len(name.split('_')))%8
  # shfl bfly exchanges paired lanes. Every lane has the same recurrence except
  # initial lane; track parity to compute the paired input independently.
  if op=='imad': ref=f'for(unsigned lane=0;lane<2;++lane)r[lane][{j}]=r[lane][{j}]*(seed|1)+seed;'
  elif op=='xor':ref=f'for(unsigned lane=0;lane<2;++lane)r[lane][{j}]=r[lane][{j}]^r[lane][{(j+1)%8}];'
  elif op=='lds':ref=f's[{j}]=seed+t;'
  elif op=='fma':ref=f'f[{j}]=std::fma(f[{j}],scale,bias);'
  else:ref=f'{{uint32_t tmp=r[0][{j}];r[0][{j}]=r[1][{j}];r[1][{j}]=tmp;}}'
  text+=ref+'\n'
 text+=''' }
'''
 text+='''for(unsigned j=0;j<8;++j){
float actual;memcpy(&actual,&result[t*24+16+j],4);
failures+=result[t*24+j]!=r[0][j] || result[t*24+8+j]!=s[j] || !std::isfinite(actual) || actual!=f[j];
}}
unsigned long long low=cycles[0],high=cycles[0];for(auto c:cycles){if(c<low)low=c;if(c>high)high=c;}
printf("PIPE name=%s rep=%u loops=%u threads=%u min_cycles=%llu max_cycles=%llu failures=%u\\n",names['''+str(n)+'''],rep,loops,threads,low,high,failures);if(failures)return 1;
}
'''
text+='CK(cudaFree(out));CK(cudaFree(times));return 0;}\n'
# string's stoul comes from the standard string declaration.
text=text.replace('#include <cstring>', '#include <cstring>\n#include <string>')
(P/'probe.cu').write_text(text)
