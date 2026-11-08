# 阶段 0 实现方案：让正确的代码和参数进入模拟器

本文件是待执行方案，不是完成报告。当前没有本阶段代码改动或运行验收。配套：[阶段 0 验收](../validation/stage-0.md)。

## 完成后应当能做什么

本阶段完成后，一个独立 CUDA 测试程序应能加载两份小型 PTX 代码，交替启动其中的函数，把参数和每次申请的共享内存正确传给模拟器，得到可检查的输出和设备周期。

这里的“内核”是主机启动、由 GPU 线程执行的函数；entry 是这段函数在 PTX 文件中的入口；“启动”就是指定执行哪个函数、用多少线程、传哪些参数。`libcudart.so` 是 CUDA 运行时库，本任务要加载模拟器提供的版本，才能进入模拟执行。

```text
CUDA 主机程序
  → 登记 PTX 模块与函数入口
  → 提交本次线程配置、参数、动态 shared 字节数
  → 创建本次 kernel 执行对象和参数内存
  → 调度 CTA，执行 PTX 并累计周期
  → 把结果复制回主机比较
```

CTA 指线程块；shared 是同一个线程块内的线程共同使用的存储。本阶段接通上面这条普通启动路径，不实现 TMA、cluster 或 WGMMA。冻结完整程序会先编码 tensor map、注册全部 110 模块，因此本阶段通过仍不等于它已经可以执行前向。

## 工作范围和具体交付

| 工作 | 完成后必须满足的具体结果 | 验收 |
| --- | --- | --- |
| 保留 `sm_90a` 目标名 | 含该目标的测试 PTX 交给 ptxas 时仍使用 `--gpu-name=sm_90a`，入口和资源信息对应原模块 | [代码目标](../validation/stage-0.md#target) |
| 接受 64B 参数对齐 | 实际 1344B 参数完整传入；混合参数的起始偏移与 PTX 声明一致 | [参数](../validation/stage-0.md#parameters) |
| 修正两种启动入口 | 直接 API 与 `<<<>>>` 都使用本次 grid/block/shared/stream；调用结束不遗留配置 | [启动](../validation/stage-0.md#launch) |
| 传递动态 shared | 同一函数两次申请不同字节数，执行对象、驻留计算和资源归还均使用各自的值 | [shared](../validation/stage-0.md#shared) |
| 接受外部失败函数声明 | 含 `.noreturn __assertfail` 声明的正常程序能运行；真正执行失败分支时明确报错终止 | [声明与失败出口](../validation/stage-0.md#declarations) |
| 核查多模块归属 | 登记 A、B 后调用 A→B→A，分别得到 A、B、A 的数据和入口信息 | [多模块](../validation/stage-0.md#modules) |
| 提供最小测试入口 | 测试 agent 能只装载被测模块、准备合法输入、保存输出/周期/墙钟记录 | [运行入口](../validation/stage-0.md#runner) |

前五项有下面列出的静态代码证据。多模块是否出错尚未运行确认；不能把“需要测试”写成“已经丢失模块”。

<a id="target"></a>

## 1. 保留代码的完整架构目标名

### 这是什么，为什么要改

PTX 中的 `.target sm_90a` 告诉工具链，这段代码使用哪种 GPU 架构的指令能力。`sm_90a` 中的 `a` 是目标名的一部分，不是可以删除的注释。模拟器还会调用 CUDA 的 `ptxas` 工具取得寄存器数、shared 用量等资源信息；这一步也必须使用正确目标。

当前提取路径从文件名读出 `90a`，再用 `atoi` 变成数字 90；后续 `gpgpu_ptx_info_load_from_filename` 又用 `sm_%u` 拼成 `sm_90`。这是已核实的后缀丢失路径。它可能使含专用指令的代码无法按原目标处理；本轮未运行完整 workload 复现报错。

### 具体改法

1. 保留 `version_filename` 现有的数字能力排序，用 90 判断是否落在当前最大能力配置内；另外按提取后的文件名保存完整目标字符串，例如 `sm_90a`。两者分别服务于“排序”和“交给工具链”，不能再由数字反推完整名字。
2. 在 `cuobjdumpParseBinary` 遍历文件、调用资源提取时，将该文件对应的完整目标传给 `gpgpu_ptx_info_load_from_filename`。同步修改函数声明和本路径调用点。
3. 资源提取命令使用该字符串生成 `--gpu-name=sm_90a`。不要把 `.target` 本身改写为 `sm_90`，也不要为了让 ptxas 成功删除模块中的指令。
4. PTX parser 的 `.target` 规则目前接受标识符，不需要另加一个仅匹配 `sm_90a` 的词法分支。旧 `sm_90` 测试仍应使用自己的完整目标。

**实现边界：**先修冻结程序实际使用的提取路径；未经过的旧 CUDA/PTXPlus 路径不做兼容性重构。目标字符串可沿已有元数据保存，不新增通用架构描述框架。

**SASS 复用：**这一项解决 PTX 工具链输入，SASS 性能模型不能代替。后续从 ptxas 得到的资源信息仍交给已有资源计算使用。

<a id="parameters"></a>

## 2. 参数对齐：让主机写入的位置与内核读取的位置一致

### 先区分参数大小、对齐和字段偏移

内核参数在模拟器中存放在一段按字节寻址的参数内存里。**大小**是参数占多少字节；**对齐**规定这个参数从哪里开始：64B 对齐表示起点地址必须是 64 的整数倍。它既不表示参数只有 64B，也不要求参数内部的每个字段都占 64B。

例如，下列四个参数按声明顺序排列：

```ptx
.param .u32 head,
.param .align 64 .b8 payload[128],
.param .u64 tail,
.param .u64 output
```

| 参数 | 大小 | 起点要求 | 参数区中的起始偏移 |
| --- | ---: | --- | ---: |
| head | 4B | 4B 对齐 | 0 |
| payload | 128B | 64B 对齐 | 64 |
| tail | 8B | 8B 对齐 | 192 |
| output | 8B | 8B 对齐 | 200 |

`head` 占偏移 0～3；下一项不能紧接着从 4 开始，要留出间隙，让 `payload` 从 64 开始。假如主机参数拷贝按 4 放置，而内核按 64 读取，读到的就不是传入的那部分数据。这样的错误会表现为矩阵地址、尺寸或其它参数不对。

真实专家入口不是这个示例，而是单个 `.param .align 64 .b8 ...[1344]`：**1344B 是对象大小，64B 是对象起点对齐。** PTX 随后按对象内部的固定偏移读取字段，例如 `+1128`、`+1136`。本阶段不解释其中 tensor map 的内容，只保证这些原始字节完整传入。

### 现有检查在做什么

`ptx_recognizer::add_function_arg` 有一个断言，只接受 1、2、4、8、16B 对齐。它的作用是拒绝作者当时未覆盖的参数声明，避免带着未知布局继续执行。64B 是当前程序合法使用的值，但不在该列表中，所以当前代码会在解析这里拒绝它。

这里不能把“检查限制”误说成“整个布局算法不支持64B”：

- `function_info::add_config_param` 已经按传入对齐值向上计算起始偏移，并保存 `(大小, 偏移)`。
- `cudaLaunchKernelInternal` 用 `get_param_config` 取得这对值，交给 `cudaSetupArgumentInternal`。
- `function_info::finalize` 已从参数声明读取显式对齐，再把参数内容复制到参数内存，并核对偏移。

因此首选是让已有算法接受当前合法声明，再验证两端一致，不重写一套参数布局系统。

### 具体改法与保持不变的约定

1. 更新 `add_function_arg` 现有支持范围，使当前使用的64B对齐通过；保留原来小对齐行为。参数数组的大小仍由声明计算：`.b8[1344]` 是1344字节，不是1字节或64字节。
2. 保留现有偏移算法：`offset = ceil(previous_end / alignment) * alignment`，再令 `previous_end = offset + size`。检查上述示例在 `m_param_configs` 中得到 `(4,0)、(128,64)、(8,192)、(8,200)`。
3. 主机 `args[i]` 指向第 i 个参数值本身；`cudaSetupArgumentInternal(args[i], size, offset)` 中的 offset 是**目的参数区偏移**，不能再加到 `args[i]` 上。以一个指针参数为例，应复制保存该指针值的8个字节，而不是复制它指向的矩阵内容。
4. `finalize` 继续复用分页参数内存写入；检查最终参数符号地址与 `get_param_config` 的偏移相同，复制长度等于声明长度。仅在测试发现两端不一致时修改对应计算，不能用取较小长度、补零或忽略断言掩盖传参错误。
5. 保留参数对象内部的原始字段布局，不按执行 agent 自己写的 C++ struct 重新排列。混合参数测试检查“参数之间的间隙”，1344B测试检查“单个对象完整传递”，两者分别证明不同问题。

**范围限制：**`get_args_aligned_size` 中另有按参数大小对齐的代码，但当前看到的调用用于设备端启动子内核；冻结主机启动路径不依赖它。`param_to_shared` 在当前非PTXPlus配置下直接返回。本阶段不为这两条未使用路径重构ABI，也不扩全部奇异尺寸、非法对齐测试。

**SASS 复用：**参数值需要 PTX 路径自己生成，SASS trace 不能提供替代答案。参数就位后，普通 `ld.param`、寄存器和load/store执行路径继续复用。

<a id="launch"></a>

## 3. 启动配置：使直接 API 和编译器入口都传对本次参数

### 问题是什么

一次启动需要函数、grid（线程块数量）、block（每块线程数）、动态shared字节数和stream（提交顺序所属队列）。`kernel_config` 是当前保存这些值和参数列表的对象；“配置栈”就是暂存尚未提交启动配置的容器。

冻结启动器直接调用 `cudaLaunchKernel(...)`。但当前 `cudaLaunchKernelInternal` 只在 CUDA<10 分支里建立配置；后面的 `cudaSetupArgumentInternal` 和 `cudaLaunchInternal` 都要求栈非空。对当前工具链，直接调用不能依赖编译器事先替它压入配置。

另一条路径 `kernel<<<...>>>(...)` 会经过编译器生成的主机包装代码。当前 `__cudaPushCallConfiguration` 会压栈，而 `__cudaPopCallConfiguration` 仅返回成功，没有回填参数或出栈。因此不能只无条件再压一份配置就结束修正。

### 采用的调用约定

```text
直接 API：
  cudaLaunchKernel(显式配置)
    → 建立本次配置 → 放入实参 → 创建kernel并入队 → 移除本次配置

当前编译器入口：
  __cudaPushCallConfiguration(<<<>>>配置)
    → __cudaPopCallConfiguration：取出值并移除这份临时配置
    → cudaLaunchKernel(取出的显式配置)
    → 与直接API相同的提交路径
```

实现时保存本机冒烟程序的编译器生成包装代码，核对其实际调用符合这一路径。若本机生成代码不同，只调整这个ABI衔接位置并记录证据，不能以“栈空则猜默认配置”的回退同时维持两种互相冲突的约定。

### 具体改法

1. 为 `kernel_config` 提供读取动态shared字节数的访问方法；grid、block、stream已有访问方法。
2. `__cudaPopCallConfiguration` 从栈顶取本次值，按它的参数约定写回 grid/block/shared/stream，再移除这一份临时配置。不能返回成功却留下调用方输出未初始化。
3. 当前 `cudaLaunchKernelInternal` 从自己的显式实参调用已有 `cudaConfigureCallInternal` 建立一次配置，再按 `get_param_config` 逐项绑定参数，最后调用 `cudaLaunchInternal`。内部调用使用同一个 context，返回实际调用结果，不把下层失败改写成成功。
4. 保留 `cudaLaunchInternal` 创建kernel、加入stream队列及移除本次配置的现有职责。已有旧式 configure→setup→launch 入口继续按其原约定使用，不能在共用提交函数里再次压栈。
5. 验收时每次启动记录：调用入口、函数、grid/block/shared/stream及提交前后栈深度。正确条件是结束时恢复调用前深度，且输出反映本次配置；这属于测试观察，不是在生产路径增加自动清栈机制。

**SASS 复用：**复用现有stream队列、kernel执行对象和普通CTA调度；SASS trace通常已给出启动信息，不能代替CUDA主机ABI修正。

<a id="shared"></a>

## 4. 动态 shared：区分允许申请多少与本次实际申请多少

### 用具体例子说明

`cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)` 的意思是允许函数 f 使用最多65536字节动态shared。随后同一个 f 启动两次，一次传32768，一次传65536：两个执行对象的实际需求必须分别是32768和65536，不能都按许可上限分配。

动态shared的“动态”指大小在启动时给出。静态shared则由PTX中的固定大小声明决定。资源模型用每个线程块占用多少shared来限制同一SM能同时放几个线程块；漏掉动态部分会允许过多线程块一起执行，从而把时间估得偏短。

### 当前缺口在什么位置

`kernel_config` 已保存 `m_sharedMem`，但 `cudaLaunchInternal` 调用 `gpgpu_cuda_ptx_sim_init_grid` 时没有传它，后者创建的 `kernel_info_t` 也没有本次动态字节信息。`cudaFuncSetAttribute` 目前忽略设置并返回成功。

占用代码读取的是 `ptx_sim_kernel_info(entry)->smem`。这个已有字段由 `ptxinfo_smem(declared, system)` 写成两者之和，**不应简单称为PTX静态shared**。它与每次launch的动态部分是不同来源。

### 具体改法

1. 在函数元数据中保存当前需要的动态shared许可上限，以函数入口身份定位。这里只实现 `cudaFuncAttributeMaxDynamicSharedMemorySize` 的当前消费者语义，不顺带实现所有函数属性。
2. 从 `kernel_config` 读出本次动态字节数，增加 `gpgpu_cuda_ptx_sim_init_grid` 对应参数，并保存到本次 `kernel_info_t`。该值属于一次launch，不能写回共用的 `function_info::m_kernel_info.smem`，否则同一函数的第二次启动会改变第一次的需求。
3. 新增一个集中取得“本次每CTA shared占用”的入口，返回**已有ptxas资源基数 + 本次动态字节数**。在 `shader_core_config::max_cta`、L1/shared划分、`occupy_shader_resource_1block` 的容量判断/增加以及 `release_shader_resource_1block` 的归还处使用同一口径。不能只改最大驻留数而漏掉占用或归还。
4. 在当前主机启动边界落实已设置的许可上限和实际设备容量约束；测试使用合法配置。不要为了让大shared程序启动而打开“忽略资源限制”，也不需要逐次访存重复检查同一launch保证。
5. 继续复用按页保存的shared功能存储。`memory_space_impl<16*1024>` 的16KiB是分页大小，不是CTA总容量，不能据此把它替换成一份固定215040B数组。普通shared地址及已有静态/extern符号分配先沿用，用合法首尾写读确认；只有测试证明符号重叠才改对应布局。
6. 核对ptxas的 declared/system 记录，避免把已计入的占用再加一次。H100专家记录的1024B驱动保留量不是PTX静态数组；完整216064B占用核对和超过96KiB/132SM地址支持归阶段3。本阶段先用32KiB/64KiB动态请求验证本次字节的传递和资源公式。

**SASS 复用与兼容：**沿用现有shared容量、寄存器/线程共同限制及自适应L1模型。SASS构造的kernel对象保持新增动态字段默认0，原trace资源基数照旧使用，避免重复计量；修改的共用资源函数需做针对性回归。

<a id="declarations"></a>

## 5. 外部失败函数声明：正常路径可加载，失败路径仍失败

`.extern .func __assertfail(...) .noreturn;` 是一条函数声明：告诉解析器有一个名为 `__assertfail` 的失败处理入口，调用后不会返回。**文件里存在声明不代表本次运行触发了失败**；正常路由内核也可能包含这条声明。

当前词法/语法中没有 `.noreturn` 的对应处理。`call_impl` 对少数内置函数有入口，但没有 `__assertfail` 专门分支；`trap_impl` 已走 `inst_not_implemented` 打印错误并 `abort()`。这些分别是语法缺口、失败调用接入缺口和可复用的现有终止行为。

具体实现：

1. 在 `ptx.l` 增加 `.noreturn` token，在 `ptx.y` 的函数声明后缀位置接受它，覆盖实际形式“参数右括号 → .noreturn → 分号”。保留函数参数、extern身份和声明信息；不能把整个声明从PTX文本里删掉。
2. 不把该修饰当作内核入口或设备指令，不要求为它添加一般性的控制流优化。现有extern函数跳过函数体汇编的处理继续复用。
3. 在 `call_impl` 的内置函数分派处识别确切的 `__assertfail`，在进入普通函数调用栈之前报告失败位置并非零终止。最少给出PTX源位置/入口和线程身份即可；不为本轮实现完整CUDA错误恢复或所有断言消息格式化。
4. `trap` 已经会报错终止，本阶段先验证即可，不为了“实现”二字重写已有终止逻辑。`__assertfail` 和trap都不能吞掉错误后返回成功结果。

**SASS 复用：**无对应性能功能需要搬用。失败分支不做周期精调；正常程序仍走原算术/访存/分支模型。

<a id="modules"></a>

## 6. 多模块与可复用测试入口

模块是一份已编译GPU代码，当前每份PTX有一个entry；不是一个MoE专家。模拟器先把主机函数标记绑定到entry，之后启动时用这个标记查函数。当前已有 `fatbinary handle → 符号表` 和 `hostFun → function_info` 两张映射，应优先复用。

多份文件共用 `default` 名称和全局符号表，并不能证明旧函数被覆盖：共用表可能同时包含A和B的唯一入口。因此本项的目标是取得运行结论，不是预先重写模块系统。

具体步骤：

1. 构造两个独立fatbinary（CUDA代码包），各含唯一entry `write_a`、`write_b`，各自有不同名称、不同值的global/constant标记。global是设备全局数据，constant是设备只读数据区，标记用于确认取到了哪个模块的数据。沿 `__cudaRegisterFatBinary` → `__cudaRegisterFunction` → `__cudaRegisterFatBinaryEnd` 登记代码包及各自的主机函数标记，再交替调用A→B→A。
2. 观察 `CUctx_st::register_function/get_kernel` 找到的函数、参数配置和ptxas资源信息。输出分别为11→22→11，并保留模块各自资源记录。不能只记录注册次数就通过。
3. 测试通过则保留现有共享表。若失败，从首次错误的 handle/hostFun/entry/资源映射定位：只修错误的键或绑定，不在每次launch重新解析全部PTX，也不建立“失败后尝试最后一个模块”的回退。
4. 用同一小型host测试入口登记指定fatbinary、按真实参数清单构造 `void *args[]`、调用原API并回收输出。以后测试真实模块时保持它的入口、固定尺寸和合法参数；不改冻结启动器跳过110模块或tensor map步骤。

**SASS 复用：**模块注册和PTX符号解析不能由SASS替代；函数找到后复用已有stream、CTA调度和统计。未使用的跨模块同名符号链接规则暂缓，当前测试使用唯一名称。

## 实施顺序、记录和停止边界

先确认debug可构建和测试加载库；修正目标、声明与参数；接通直接/编译器启动和动态shared；最后运行两模块组合。测试入口可以随这些步骤逐步完善，不需要提前造通用测试框架。

运行记录同时区分设备周期与本机墙钟：前者用于预测GPU耗时，后者用于估计后续验收需要多久。首个小程序开始记录装载/初始化、模拟执行和峰值RSS，但不能用它推断专家矩阵模拟速度。

本阶段不做cluster、大shared generic地址扩展、TMA/WGMMA、设备端启动子内核、完整DMA引擎、所有CUDA版本或非法输入穷举。必要测试通过即进入阶段1；未使用路径的静态线索只记录，不借本阶段扩大修复。

## 源码定位与执行 agent 交付要求

以下是本轮静态核查的入口；无需读这些文件才能理解上文，执行改动时再沿函数定位。

| 源码 | 本阶段相关位置 |
| --- | --- |
| [cuda_runtime_api.cc](../../gpu-simulator/gpgpu-sim/libcuda/cuda_runtime_api.cc) | `extract_ptx_files_using_cuobjdump_internal`、`cuobjdumpParseBinary`；configure/setup/launch、Push/Pop、FuncSetAttribute、init_grid |
| [ptx_loader.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_loader.cc) | `gpgpu_ptx_info_load_from_filename` 的ptxas命令 |
| [ptx_parser.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_parser.cc)、[ptx_ir.h](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx_ir.h) | `add_identifier/add_function_arg`、`add_config_param/get_param_config` |
| [cuda-sim.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc) | `add_param_data/finalize`、`ptxinfo_smem`、`ptx_sim_init_thread` |
| [cuda_api_object.h](../../gpu-simulator/gpgpu-sim/libcuda/cuda_api_object.h)、[abstract_hardware_model.h](../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h) | `kernel_config`、主机标记映射、`kernel_info_t` |
| [shader.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc)、[gpu-sim.cc](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc) | `max_cta`、`occupy_shader_resource_1block/release_shader_resource_1block` |
| [ptx.l](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.l)、[ptx.y](../../gpu-simulator/gpgpu-sim/src/cuda-sim/ptx.y)、[instructions.cc](../../gpu-simulator/gpgpu-sim/src/cuda-sim/instructions.cc) | `.noreturn`声明后缀、`call_impl`、`trap_impl` |
| [专家原始PTX](../../workloads/moe/artifacts/ptx/df3e0c49e4650c1f.ptx)、[断言声明实例](../../workloads/moe/artifacts/ptx/0f77e1c592ff33ce.ptx) | 1344B/align64参数，以及真实extern/noreturn声明形式 |

实际改动、指令/接口、复用及自测填写到 [阶段 0 实际实现记录](../execute/implementation/stage-0/README.md)；独立运行结论填写到 [实际验收记录](../execute/validation/stage-0/README.md)。本文件保留实现方案，不重复维护实际状态。
