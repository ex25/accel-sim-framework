#!/usr/bin/env python3
"""Generate small, explicit PTX fixtures; no frozen workload is rewritten."""
from pathlib import Path
import sys
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
a = ['// Stage 0 fixture\n.version 8.7\n.target sm_90\n.address_size 64\n', '''
.global .u32 global_a = 5;
.const .u32 constant_a = 6;
.visible .entry write_a(.param .u64 output) {
.reg .u64 p;
.reg .u32 a,b,c;
.shared .u32 fixed_a[8];
ld.param.u64 p,[output];
ld.global.u32 a,[global_a];
ld.const.u32 b,[constant_a];
add.u32 c,a,b;
st.shared.u32 [fixed_a],c;
ld.shared.u32 c,[fixed_a];
st.global.u32 [p],c;
ret;
}
.visible .entry mixed(.param .u32 head, .param .align 64 .b8 payload[128],
                     .param .u64 tail, .param .u64 output) {
.reg .u64 p,t;
.reg .u32 r;
ld.param.u64 p,[output];
ld.param.u32 r,[head]; st.global.u32 [p],r;
ld.param.u64 t,[tail]; st.global.u64 [p+136],t;
''']
for i in range(0, 128, 4):
 a.append(f'ld.param.u32 r,[payload+{i}]; st.global.u32 [p+{i+4}],r;\n')
a.append('ret;\n}\n.visible .entry large(.param .align 64 .b8 payload[1344], .param .u64 output) {\n.reg .u64 p; .reg .u32 r;\nld.param.u64 p,[output];\n')
for i in range(0, 1344, 4):
 a.append(f'ld.param.u32 r,[payload+{i}]; st.global.u32 [p+{i}],r;\n')
a.append('''ret;
}
.extern .shared .align 16 .b8 dynamic_shared[];
.visible .entry shared_test(.param .u64 output, .param .u32 bytes) {
.reg .u64 p,offset;
.reg .u32 t,b,end,s,addr,r;
.reg .pred pred;
.shared .u32 fixed_shared[8];
ld.param.u64 p,[output];
ld.param.u32 end,[bytes];
sub.u32 end,end,4;
mov.u32 s,dynamic_shared;
add.u32 addr,s,end;
mov.u32 t,%tid.x;
setp.eq.u32 pred,t,0;
@!pred bra read;
st.shared.u32 [fixed_shared],77;
st.shared.u32 [s],101;
st.shared.u32 [addr],202;
read:
bar.sync 0;
setp.eq.u32 pred,t,1;
@!pred bra done;
mov.u32 b,%ctaid.x;
mul.wide.u32 offset,b,12;
add.u64 p,p,offset;
ld.shared.u32 r,[s]; st.global.u32 [p],r;
ld.shared.u32 r,[addr]; st.global.u32 [p+4],r;
ld.shared.u32 r,[fixed_shared]; st.global.u32 [p+8],r;
done:
ret;
}
.extern .func __assertfail(
.param .b64 message, .param .b64 file, .param .b32 line,
.param .b64 function, .param .b64 charsize
) .noreturn;
.visible .entry failure(.param .u64 output, .param .u32 mode) {
.reg .u64 p;
.reg .u32 m;
.reg .pred q;
ld.param.u64 p,[output]; ld.param.u32 m,[mode];
setp.eq.u32 q,m,0; @q bra success;
setp.eq.u32 q,m,2; @q bra trap_path;
{
.param .b64 a; .param .b64 b; .param .b32 c; .param .b64 d; .param .b64 e;
st.param.b64 [a],0; st.param.b64 [b],0; st.param.b32 [c],42;
st.param.b64 [d],0; st.param.b64 [e],1;
call.uni __assertfail,(a,b,c,d,e);
}
trap_path: trap;
success: st.global.u32 [p],123;
ret;
}
''')
(out/'a.ptx').write_text(''.join(a))
(out/'b.ptx').write_text('''// Stage 0 fixture
.version 8.7
.target sm_90a
.address_size 64
.global .u32 global_b = 9;
.const .u32 constant_b = 13;
.visible .entry write_b(.param .u64 output) {
.reg .u64 p;
.reg .u32 a,b,c;
.shared .u32 fixed_b[16];
ld.param.u64 p,[output];
ld.global.u32 a,[global_b];
ld.const.u32 b,[constant_b];
add.u32 c,a,b;
st.shared.u32 [fixed_b],c;
ld.shared.u32 c,[fixed_b];
st.global.u32 [p],c;
ret;
}
''')
