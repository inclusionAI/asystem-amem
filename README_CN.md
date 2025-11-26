# AMem NCCL-plugin：NCCL显存透明卸载和恢复

## 简介
AMem NCCL-plugin是一个NCCL功能扩展库，由蚂蚁集团Asystem团队开发，通过分层架构设计和轻量级hook**首次实现对NCCL显存的透明卸载和恢复**，适用于大规模强化学习训练场景，可节省显存10~20GB/卡。已经用于Ring-1T强化训练中。

+ **NCCL分布式显存**：识别并解决cross-rank的**显存交叉引用问题**，实现正确的显存透明释放与恢复。
    - 为了高效通信，rank间P2P显存地址互相reference导致难以卸载，而P2P显存是NCCL显存头号开销。
+ **兼容性**：支持NCCL（基于2.27开发）所有并行方式和主流硬件；已集成验证SGLang和Megatron。
+ **效率高**：显存卸载可做到基本常数，典型耗时<1sec。

以下是个简单对比：

| | NCCL (2.27) | **Slime** | **verl, AReal, openRLHF** | AMem |
| --- | --- | --- | --- | --- |
| Live offload | <font style="color:#DF2A3F;">N</font> | <font style="color:#DF2A3F;">N</font>*<br/>*采用重启进程来卸载显存，<font style="color:#DF2A3F;">代价是重建通信开销大，最坏可能每次数分钟</font> | <font style="color:#DF2A3F;">N</font><br/><font style="color:#DF2A3F;"> (不支持卸载，浪费显存)</font> | Y |


![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1761125537655-6489480b-8e2c-4883-88de-062919720fbc.png)

图1：AMem NCCL-plugin功能对比。（slime介绍: Xiaolin Zhu. Slime V0.1.0. [https://zhuanlan.zhihu.com/p/1945237948166547268](https://zhuanlan.zhihu.com/p/1945237948166547268))

## 效果展示
AMem NCCL-plugin可以将NCCL的显存几乎全部卸载（并按需恢复）。取决于集群规模（特别是alltoall）、并行策略（通常会3D~5D并行）以及CUDA/NCCL版本等，大规模任务的NCCL显存开销可能10GB~20GB /GPU

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1760150373401-663981e7-5f67-4375-9d09-abbd173da074.png)        ![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1760150400495-2eabb3bc-4c0a-4605-9a36-21d0436eae49.png)

图2：AMem NCCL-plugin 可将NCCL分配的显存（如上图）几乎全部卸载

+ cuda context显存不卸载（典型~800MB），这部分显存会和训/推进程共用。
+ 首次卸载较慢（因为分配CPU pinned buffer），后续通常 <1sec。
    - 默认卸载全部的NCCL显式分配显存。可配置P2P buffer显存只释放不做offload从而显著降低传输耗时，即只offload其他显存数据（通常是NCCL的元数据，联建信息等），耗时基本为常数。

## 背景问题
**强化学习**：典型强化学习训练时，如果采用训推共卡部署，一个任务完成后需要将GPU资源快速、干净释放给后续任务，以提高资源效率。GPU算力是无状态的，因而可用完即放；而显存是有状态的，需要进行显式的清理改造（以及redo），有一定的工作量，例如执行必要的offload保存内容后再释放，后续按需恢复，甚至对NCCL显存存在较大的技术挑战，这里设计一些显存分配、引用和状态恢复等。

+ 如果显存不释放，例如图1所示NCCL显存可能就占据10GB ~ 20GB，会显著影响训推的batch size。而强化学习总体是吞吐密集，batch size比较重要。

****

**显存管理**：CUDA显存管理有多种APIs，为了满足进程存活而释放显存资源，需要采用Virtual Memory Management APIs (VMM or cuMem)，这组API提供了两层地址管理和动态映射能力，具体详见图3总结。当前PyTorch、NCCL等都有参数可选激活VMM显存分配方式。

+ 如果销毁训推进程，也可以实现干净释放显存。而代价是各种初始化、大规模NCCL通信建联，其时间开销往往较大（通常分钟级，当然存在优化空间如Meta等最近的工作）

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1756363908841-9d94a4e0-9505-415d-82f7-d108d0a48d5a.png)

图3：NVIDIA VMM显存管理API和典型操作步骤

**显存卸载：**强化学习场景中显存卸载典型包括如下。社区对多数已有初步支持，但仍存在不足甚至盲区，其中NCCL显存是其中比较吃力的一个卡点，详见下节。

+ 训练：权重、优化器状态、激活等，NCCL显存、cuda graph等。
+ 推理：权重、KV cache、激活等，NCCL显存、cuda graph等。

## 技术挑战
相比于PyTorch/python里的显存卸载，NCCL透明显存卸载主要面临以下三个挑战：

1. NCCL是C/C++实现，独立于PyTorch显存池之外，现有的各种python方案不支持。
2. 分布式P2P**显存交叉引用**：尤其是，区别于rank自身数据（例如已切分后的权重、激活、KV等），NCCL为集合通信而生，典型多卡环境下引入了复杂的cross-rank P2P引用。进程如果只free自己的显存并不会释放资源给驱动，且多个回合后，老的不去，不断新分，NCCL显存占用反而越来越大。本质上这里有个独特的分布式显存交叉引用问题。
    1. 相应的，恢复时必须严丝合缝，如数还原，否则易引发crash或hang等问题。
3. 动态建联、3D/4D混合并行等导致复杂逻辑：NCCL修改难度大，corner case多。例如2024年NVIDIA针对NVSwitch高速集合通信进一步推出了symmetric memory，其显存管理逻辑更为复杂（如图4）

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1756366491390-f7e6a696-9a9b-4966-9846-f48b8519e316.png)

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1756369092529-36512037-954b-43e2-8153-8ee3a93b3f51.png)

图4：NVIDIA symmetric memory相关API

## 方案设计
AMem NCCL-Plugin基于VMM API，设计了两层解耦方案，实现了对NCCL显存透明卸载和恢复。

+ **NCCL Hook**: 修改了NCCL极少量代码。集中在几处显存相关（分配、释放、map）操作以获取元数据信息。
    - NCCL的核心逻辑不动，升级打patch比较方便。
+ **AMem Plugin**: 核心逻辑封装在一个单独的lib，独立于NCCL源码。
    - **元数据管理**：例如显存地址信息、引用信息、当前状态等。
    - **分布式引用识别**和卸载：实现跨进程和跨rank的动态溯源。
    - **分布式resume：**根据元数据执行redo，包括跨进程、跨rank重新导出和映射。
    - **进程组：**通过内部一套Unix Domain Socket（UDS）实现fd跨进程传递；对训推进程进行逻辑分组以正确识别引用并避免误操作。代码实现借鉴了团队之前开源的工作[GLake](https://github.com/antgroup/glake)。
        * 共卡部署时，训推进程同时并存，很容易分配得到相同的地址（仅在进程空间内有效）。

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1760162230824-512f8c55-c062-42d7-832a-0f4d610be057.png?x-oss-process=image%2Fformat%2Cwebp)

图5：AMem NCCL-plugin总体架构图

### 元数据：交叉引用溯源
图6展示一个进程的NCCL显存（P2P buffer）地址（handle0）通过VMM API导出给其他多个进程。如果每个进程只释放自身地址而未等待peer释放，显存资源并不会归还给系统。

AMem NCCL-plugin会动态跟踪记录某个handle被哪些peer所引用，这个后续功能的关键基础，确保了**“释放时一个不漏，恢复时一个不少”**。例如，返回之前引用均完成释放；恢复时，基于元数据记录来精确执行redo。详见图8的流程。



![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1756369240340-22501a29-62d8-45a5-9fd6-a3172ff1b938.png)

图6：NVIDIA P2P显存交叉引用和处理（注：多卡对等，示例为简化展示）

### 元数据：状态管理
AMem NCCL-plugin对进程状态和每个NCCL显存分配地址（dptr）维护、更新内部状态，如图7所示。

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1756364076528-5228dd66-7e8d-4c88-93a2-f502eb019bf2.png)

图7：进程和显存状态和转移示意

### 分布式卸载与恢复流程
通过内置的UDS通信，AMem NCCL-pluin重点实现了跨进程P2P reference溯源、元数据更新和正确的redo，具体流程如图8所示。

+ 多卡（rank）本质是对等关系，图中仅以rank0的视角示例说明核心流程。

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1760160197422-3cd52022-e09c-40ee-b547-b51bfbf48c86.png?x-oss-process=image%2Fformat%2Cwebp)

图8：AMem NCCL-plugin分布式NCCL显存卸载与恢复流程

# 2 安装编译
AMem NCCL-plugin的交付物主要是3个文件：扩展版的** nccl.h、libnccl.so.2和libamem_nccl.so。**

在保持NCCL现有功能的基础上，我们扩展了多个API以支持显存透明卸载、恢复和显存用量统计功能。

```c
##### NCCL.h新增了以下5个API

## 每个进程显式调用：ncclPause返回则本卡的显存释放完毕、本卡引用其他卡的显存计数减1
## 注意:
## 1. Pause和Resume是同步调用。Pause之后不能再对NCCL调用。否则可能crash、hang、invalid mem等
## 2. Pause和Resume必须成对使用，按序调用。用户负责。否则调用可能无效，或异常
## 3. 多卡之间的状态一致性由调用者负责。例如必须等待多卡都完成了Resume，才能继续使用。
ncclResult_t ncclPause(ncclComm_t * comm = NULL);
ncclResult_t ncclResume(ncclComm_t* comm = NULL);

# 统计NCCL的显存分配总量、哪些func调用了显存分配。
ncclResult_t ncclMemStats();

# 如GPU上有多进程显式为同属一组的进程设立ID。AMem用此区分防止错误对显存溯源。例如
# GPU0 1...7上的训练进程，每个进程显式调用 设置为100
# GPU0 1...7上的推理进程，每个进程显式调用 设置为200
# 设置必须要在第一次的NCCL显存分配之前，否则不生效。
ncclResult_t ncclSetGroupID(int id);
ncclResult_t ncclGetGroupID(int* id);
```

**要求**：

1. NVIDIA GPU >= sm_80；功能已在sm_80, sm_90, sm_100测试
2. 推荐 CUDA >=12.2

注：首次编译耗时~10min

```bash
# 推荐使用docker nvcr.io/nvidia/pytorch:25.08-py3
cd amem/ 

git submodule init
git submodule update
./build.sh

**NCCL显存统计：统计功能和pause/resume独立**

+ 调用ncclMemStats()

```bash
AMEM groupID:170 pid:197780 caller_1 allocBytes:3024093184
AMEM groupID:170 pid:197780 caller_3 allocBytes:201326592
AMEM groupID:170 pid:197780 caller_7 allocBytes:2818572288
AMEM groupID:170 pid:197780 total allocBytes:6043992064 (5764 MB)
```


**重要参数：**

+ NCCL_ENABLE_CUMEM=1  #必须打开NCCL CUMEM
+ AMEM_ENABLE=1 # 激活NCCL Mem卸载与恢复。框架层需按需调用API
+ AMEM_GROUPID=xxx  #为训练和推理进程组设置不同的groupID
    - 注：当和RL框架集成时，以上环境变量需要传递给Ray或训推框架

**可选配置：**

+ AMEM_NCCL_OFFLOAD_FREE_TAG=7 #P2P buffer直接释放不做offload CPU
+ GMM_LOG: 默认3（INFO）。数字越大 log越多，最大为5.

# 3.快速体验
基于nccl-tests快速测试典型并行下（如allreduce，allgather, alltoall等）动态显存卸载和恢复功能。它不依赖与任何框架，编译后测试通常耗时~10min。 

+ 为了测试AMem nccl-plugin 需要进行少量修改：主要是调用ncclPause()/ncclResume()来激活功能。
+ 完整未修改版本：[https://github.com/NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests)  

```bash
# Run quick tests about nccl mem offloading/resume
export MPI_HOME=your/openmpi/home
bash ./run.sh
```

```bash
AMEM [INFO] amem_nccl.cpp:amem_addAllocInfo:102 groupID:170 pid:197780 allocSz:67108864 curDev:2 peer:-1 (detected:-1) dptr:a02000000 type:local localHandle:3621d20 rmtHandle:0 caller:3
AMEM [INFO] amem_nccl.cpp:amem_addAllocInfo:102 groupID:170 pid:197805 allocSz:67108864 curDev:7 peer:-1 (detected:-1) dptr:a02000000 type:local localHandle:4a73d60 rmtHandle:0 caller:3
AMEM [INFO] amem_nccl.cpp:amem_addAllocInfo:102 groupID:170 pid:197783 allocSz:67108864 curDev:5 peer:-1 (detected:-1) dptr:a02000000 type:local localHandle:3c05b30 rmtHandle:0 caller:3
AMEM [INFO] amem_nccl.cpp:amem_addAllocInfo:102 groupID:170 pid:197779 allocSz:67108864 curDev:1 peer:-1 (detected:-1) dptr:a02000000 type:local localHandle:43b71d0 rmtHandle:0 caller:3
AMEM [INFO] amem_nccl.cpp:amem_addAllocInfo:102 groupID:170 pid:197781 allocSz:67108864 curDev:3 peer:-1 (detected:-1) dptr:a02000000 type:local localHandle:3c58cc0 rmtHandle:0 caller:3
AMEM [INFO] amem_nccl.cpp:amem_addAllocInfo:102 groupID:170 pid:197793 allocSz:67108864 curDev:6 peer:-1 (detected:-1) dptr:a02000000 type:local localHandle:32d3880 rmtHandle:0 caller:3
AMEM [INFO] amem_nccl.cpp:amem_addAllocInfo:102 groupID:170 pid:197782 allocSz:67108864 curDev:4 peer:-1 (detected:-1) dptr:a02000000 type:local localHandle:32c9100 rmtHandle:0 caller:3
AMEM [INFO] amem_nccl.cpp:amem_addAllocInfo:102 groupID:170 pid:197778 allocSz:67108864 curDev:0 peer:-1 (detected:-1) dptr:a02000000 type:local localHandle:379f840 rmtHandle:0 caller:3
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197781 GPU:3 memUsed: 4042 MB before paused
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197783 GPU:5 memUsed: 4042 MB before paused
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197805 GPU:7 memUsed: 4042 MB before paused
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197778 GPU:0 memUsed: 4364 MB before paused
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197782 GPU:4 memUsed: 4042 MB before paused
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197780 GPU:2 memUsed: 4042 MB before paused
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197793 GPU:6 memUsed: 4042 MB before paused
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197779 GPU:1 memUsed: 4042 MB before paused

AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197778 GPU:0 pauseCnt:1 mem:1294 MB after paused, released:3076 MB, duration:11476 ms offload:549 releaseLocal:448 releasePeer:448
AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197780 GPU:2 pauseCnt:1 mem:972 MB after paused, released:3076 MB, duration:11522 ms offload:549 releaseLocal:448 releasePeer:448
AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197782 GPU:4 pauseCnt:1 mem:972 MB after paused, released:3076 MB, duration:11524 ms offload:549 releaseLocal:448 releasePeer:448
AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197793 GPU:6 pauseCnt:1 mem:972 MB after paused, released:3076 MB, duration:11534 ms offload:549 releaseLocal:448 releasePeer:448
AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197805 GPU:7 pauseCnt:1 mem:972 MB after paused, released:3076 MB, duration:11559 ms offload:549 releaseLocal:448 releasePeer:448
AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197779 GPU:1 pauseCnt:1 mem:972 MB after paused, released:3076 MB, duration:11591 ms offload:549 releaseLocal:448 releasePeer:448
AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197781 GPU:3 pauseCnt:1 mem:972 MB after paused, released:3076 MB, duration:11663 ms offload:549 releaseLocal:448 releasePeer:448
AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197783 GPU:5 pauseCnt:1 mem:972 MB after paused, released:3076 MB, duration:11764 ms offload:549 releaseLocal:448 releasePeer:448
AMEM [INFO] amem_nccl.cpp:amem_memResume:497 groupID:170 pid:197780 GPU:2 resumeCnt:1 mem:4048 MB after resumed, duration:16697 ms offloadLeft:0 releaseLocalLeft:0 releasePeerLeft:0 exportSz:2688

    67108864       2097152     float    none      -1   3551.8   18.89   16.53      0    676.9   99.15   86.75    N/A
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197779 GPU:1 memUsed: 4048 MB before paused
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197805 GPU:7 memUsed: 4048 MB before paused
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197783 GPU:5 memUsed: 4048 MB before paused
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197781 GPU:3 memUsed: 4048 MB before paused
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197782 GPU:4 memUsed: 4048 MB before paused
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197780 GPU:2 memUsed: 4048 MB before paused
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197778 GPU:0 memUsed: 4370 MB before paused
AMEM [INFO] amem_nccl.cpp:amem_memPause:335 groupID:170 pid:197793 GPU:6 memUsed: 4048 MB before paused
AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197782 GPU:4 pauseCnt:3 mem:972 MB after paused, released:3076 MB, duration:820 ms 
AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197793 GPU:6 pauseCnt:3 mem:972 MB after paused, released:3076 MB, duration:848 ms 
AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197805 GPU:7 pauseCnt:3 mem:972 MB after paused, released:3076 MB, duration:882 ms 
AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197780 GPU:2 pauseCnt:3 mem:972 MB after paused, released:3076 MB, duration:904 ms
AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197779 GPU:1 pauseCnt:3 mem:972 MB after paused, released:3076 MB, duration:971 ms
AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197778 GPU:0 pauseCnt:3 mem:1294 MB after paused, released:3076 MB, duration:971 ms
AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197781 GPU:3 pauseCnt:3 mem:972 MB after paused, released:3076 MB, duration:971 ms
AMEM [INFO] amem_nccl.cpp:amem_memPause:395 groupID:170 pid:197783 GPU:5 pauseCnt:3 mem:972 MB after paused, released:3076 MB, duration:987 ms

AMEM [INFO] amem_nccl.cpp:amem_memResume:497 groupID:170 pid:197780 GPU:2 resumeCnt:3 mem:4048 MB after resumed, duration:14810 ms 
AMEM [INFO] amem_nccl.cpp:amem_memResume:497 groupID:170 pid:197805 GPU:7 resumeCnt:3 mem:4048 MB after resumed, duration:15452 ms
AMEM [INFO] amem_nccl.cpp:amem_memResume:497 groupID:170 pid:197781 GPU:3 resumeCnt:3 mem:4048 MB after resumed, duration:15428 ms
AMEM [INFO] amem_nccl.cpp:amem_memResume:497 groupID:170 pid:197779 GPU:1 resumeCnt:3 mem:4048 MB after resumed, duration:15533 ms 
AMEM [INFO] amem_nccl.cpp:amem_memResume:497 groupID:170 pid:197782 GPU:4 resumeCnt:3 mem:4048 MB after resumed, duration:15865 ms offloadLeft:0 releaseLocalLeft:0 releasePeerLeft:0 exportSz:2688
AMEM [INFO] amem_nccl.cpp:amem_memResume:497 groupID:170 pid:197783 GPU:5 resumeCnt:3 mem:4048 MB after resumed, duration:15887 ms offloadLeft:0 releaseLocalLeft:0 releasePeerLeft:0 exportSz:2688
AMEM [INFO] amem_nccl.cpp:amem_memResume:497 groupID:170 pid:197778 GPU:0 resumeCnt:3 mem:4370 MB after resumed, duration:16270 ms offloadLeft:0 releaseLocalLeft:0 releasePeerLeft:0 exportSz:2688
AMEM [INFO] amem_nccl.cpp:amem_memResume:497 groupID:170 pid:197793 GPU:6 resumeCnt:3 mem:4048 MB after resumed, duration:16429 ms offloadLeft:0 releaseLocalLeft:0 releasePeerLeft:0 exportSz:2688

AMEM groupID:170 pid:197780 caller_1 allocBytes:3024093184
AMEM groupID:170 pid:197780 caller_3 allocBytes:201326592
AMEM groupID:170 pid:197780 caller_7 allocBytes:2818572288
AMEM groupID:170 pid:197780 total allocBytes:6043992064 (5764 MB)

# Out of bounds values : 0 OK
# Avg bus bandwidth    : 51.6429 
```

# 4.框架集成
AMem NCCL-plugin不影响正常NCCL功能使用，而扩充了新的接口，用户可按需调用：

+ ncclPause()：释放该进程中NCCL所分配的显存，同步方式执行。
+ ncclResume()：恢复之前pause所释放的所有显存，同步方式执行。
+ ncclSetGroupID()：为当前程设置进程组。
+ ncclMemStats()：统计当前进程NCCL所使用的显存量和分类。

补充说明：

+ ncclPause和ncclResume接口均是幂等的，即可以被多次调用而不会产生额外的影响。
+ 框架层负责必要的跨进程同步，确保所有rank均执行卸载、恢复完成。
+ 支持进程所创建的多个通信组（例如3D/4D并行）。
+ 如果应用并发只有一个任务在运行，例如只做推理或只做训练，不需要额外设置groupID。

### pynccl
很多上层应用如SGlang、vLLM等，均支持pynccl调用方式，即将NCCL包装成一个python的接口，加载NCCL动态库，打开API的函数句柄，然后通过python调用。以下示例sglang等调用pynccl方式。

### SGLang
仅需修改pynccl以及pynccl_wrapper类。如下是修改pynccl_wrapper来加载上述三个对应函数句柄的代码（注意，这里ncclComm的参数可以直接设置为NULL）：

```python
# ncclResult_t ncclPause(ncclComm_t comm);
Function("ncclPause", ncclResult_t, [ncclComm_t]),
# ncclResult_t ncclResume(ncclComm_t comm);
Function("ncclResume", ncclResult_t, [ncclComm_t]),
Function("ncclSetGroupID", ncclResult_t, [ctypes.c_int]),
```

当需要释放NCCL显存时，参考以下代码示例：

```python
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_group

tp_group = get_tensor_model_parallel_group().pynccl_comm
if tp_group.nccl.enable_amem_nccl:
    tp_group.nccl_pause()
```

当需要恢复NCCL显存时，参考以下代码示例：

```yaml
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_group

tp_group = get_tensor_model_parallel_group().pynccl_comm
if tp_group.nccl.enable_amem_nccl:
    tp_group.nccl_resume()
```

### Megatron
由于Megatron中并没有引入pynccl，所以推荐以下集成和使用方式：

1. 在镜像中安装集成了AMemd sglang版本
2. 在Megatron实例初始化时，初始化一个pynccl的对象。
3. 在需要显存释放和恢复的地方，按照以上SGLang中的例子显式调用对应的函数。

### RL
RL框架包含了训练框架和推理框架。取决于部署形态，有以下两种集成方式：

+ 如果训推分离部署，集成方式参考上述SGLang和Megatron集成。
+ 如果是共卡方案，需要额外传递并设置GroupID以区分训推进程组。考虑到单个GPU上有两类进程（训和推），所以在RL框架中，当训练进程组初始化时，调用ncclSetGroupID设置一个groupID；当推理进程组初始化时，类似设置一个不同的groupID。其他如释放和恢复，参考以上使用说明。

## Roadmap
- [ ] Support NCCL 2.28
- [ ] More tests on symm mem

Welcome contributions and integrations.


## 参考
+ Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model, [https://arxiv.org/abs/2510.18855](https://arxiv.org/abs/2510.18855)
+ GLake: [https://github.com/antgroup/glake](https://github.com/antgroup/glake) or ASPLOS24  [https://dl.acm.org/doi/abs/10.1145/3620665.3640423](https://dl.acm.org/doi/abs/10.1145/3620665.3640423) 
+ Zhiyi Hu, Siyuan Shen, Tommaso Bonato, Sylvain Jeaugey, Cedell Alexander, Eric Spada, James Dinan, Jeff Hammond, Torsten Hoefler.Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms, arXiv preprint arXiv:[2507.04786](https://arxiv.org/abs/2507.04786)
+ NVIDIA. NCCL 2.27. [https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/.](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/.)Accessed: 2025-10-10
+ Xiaolin Zhu. Slime V0.1.0. [https://zhuanlan.zhihu.com/p/1945237948166547268](https://zhuanlan.zhihu.com/p/1945237948166547268). Accessed: 2025-10-10
