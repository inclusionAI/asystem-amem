# AMem NCCL-plugin：透明 NCCL 显存卸载和恢复
## TL；DR 技术概述

NCCL 是 NVIDIA Collective Communications Library（英伟达集合通信库）的缩写，它是多 GPU 和多节点分布式深度学习的核心通信库，提供了包括 AllReduce、Broadcast 等多种高效集体通信操作。

AMem NCCL-Plugin 是蚂蚁 ASystem 团队自研的 **NCCL 功能扩展库**，主要提供了 ncclPause() 和 ncclResume() 两个显存管理 API，旨在解决 RL 流程中，**通信库 NCCL 分配的显存无法被高效卸载的这一难题**。通过轻量级插件方式，在保留 NCCL 通信连接的情况下，实现对训推引擎 NCCL 显存的透明卸载（offload）与恢复（reload）<sup>注1</sup>，这些优势特点已在 **Ring-1T 万亿参数推理模型**的强化学习训练中得到了验证。



AMem NCCL-Plugin 的优势体现在如下两个方面：

+ **显存节约**：通过识别并解决 NCCL 通信库中 cross-rank 的显存交叉引用问题，实现正确的显存透明释放与恢复。在训推切换时，可在保持通信组连接的情况下，单卡（Hopper 架构卡）释放出 10GB+ 显存；
+ **极致高效**：因为保留了通信组的连接，训推转换仅卸载和恢复 NCCL 的元数据，无需重建通信连接（典型耗时为数秒钟），从而实现典型耗时 <1 秒的极致优化。



与社区已知方案的在 Hopper 架构卡上的能力对比：

| **组件** | **方案** | **内存节省情况** | **每step卸载恢复耗时** |
| --- | --- | --- | --- |
| Slime<sup>注2</sup> | 通过销毁和重建训练引擎通信组清理 NCCL 显存 | 推理不节省：残留 2GB<br/>训练节省 10GB+ | 数秒 |
| OpenRLHF | 不支持卸载 NCCL 显存 | 推理不节省：残留 2GB<br/>训练不节省：残留 10GB+ | 0s |
| AMem | 通过 Plugin 卸载 NCCL 显存 | 推理节省 2GB<br/>训练节省 10GB+ | <1s |


_图1：AMem NCCL-plugin 功能对比_

_注 1: 显存释放：把显存交还操作系统；显存卸载：把显存中的信息放入CPU pinned buffer, 然后释放显存；显存恢复：把显存重新分配回来，把暂存 CPU pinned buffer 的信息拷贝回显存中。_

_注 2: slime 介绍, Slime V0.1.0._[_https://zhuanlan.zhihu.com/p/1945237948166547268_](https://zhuanlan.zhihu.com/p/1945237948166547268)

代码地址：[https://github.com/inclusionAI/asystem-amem](https://github.com/inclusionAI/asystem-amem)

## 背景问题
**强化学习共卡部署的难点**：典型强化学习系统，如果采用训推共卡部署，一个任务完成后需要将GPU资源快速、干净释放给后续任务，以提高资源效率。而 GPU **算力是无状态的，用完即放 —— 而显存是有状态的**，对它进行管理有一定的工作量。例如：需暂存关键内容到主机内存后再释放显存；后续恢复显存时需要把关键信息拷回。这对显存管理带来了较大的技术挑战，涉及**显存分配、跨进程引用、状态恢复等复杂问题**。

**显存管理的难点**：CUDA 显存管理有多种 APIs，为了满足进程存活而释放显存资源，需要采用 Virtual Memory Management APIs (VMM or cuMem)，这组 API 提供了两层地址管理和动态映射能力，具体详见图 2 总结。当前 PyTorch、NCCL 等都有参数可选激活 VMM 显存分配方式。

![](./docs/images/vmm_api_ops.png)

_图2：NVIDIA VMM显存管理API和典型操作_

在进行显存管理的时候，需要追溯各种显存分配的来源，用户态的显存分配都可以被精细管理。强化学习场景中显存需要卸载的典型存储内容包括如下：

+ 训练：权重、优化器状态、激活等，NCCL显存、cuda graph 等；
+ 推理：权重、KV cache、激活等，NCCL显存、cuda graph 等；

社区对多数显存已有初步管理支持，但对某些显存的管理仍存在不足， NCCL 显存就是其中比较突出的一个难点。

**NCCL 显存卸载的难点**：NCCL 通信库所占显存，没有对外暴露管理接口，造成管理不便。常见的管理方案有：

+ 不释放 NCCL 显存，例如图 1 所示 NCCL 显存可能就占据 10GB ~ 20GB，会显著影响训推的 batch size；而强化学习总体是吞吐密集，batch size 比较重要。本方案可以节省 RL 每个 step 的建联开销；
+ 如果销毁训推进程或者通信组，也可以实现干净释放显存。而代价是各种初始化、大规模 NCCL 通信建联，其时间开销往往较大；

上述两种方案都有不足之处，第一个是用空间换时间，第二个是时间换空间。**有没有两者兼得的方案**，是我们研究的重点。

## 技术挑战
相比于 PyTorch/python 里的显存卸载，NCCL 透明显存卸载主要面临以下**三个主要挑战**：

1. NCCL 是 C/C++ 实现，独立于 PyTorch 显存池之外，现有的各种 python 方案不支持；
2. 分布式 P2P **显存交叉引用**：尤其是，区别于 rank 自身数据（例如已切分后的权重、激活、KV 等），NCCL为集合通信而生，典型多卡环境下引入了复杂的 cross-rank P2P 引用。进程如果只 free 自己的显存并不会释放资源给驱动，且多个回合后，老的不去，不断新分，NCCL 显存占用反而越来越大。本质上这里有个独特的分布式显存交叉引用问题。同时，恢复时必须严丝合缝，如数还原，否则易引发 crash 或 hang 等问题；
3. 动态建联、3D/4D 混合并行等导致复杂逻辑：NCCL 修改难度大，测速验证 corner case 多。例如 2024 年NVIDIA 针对 NVSwitch 高速集合通信进一步推出了 symmetric memory，其显存管理逻辑更为复杂（见下图图 3）

![](./docs/images/sym_mem.png)

![](./docs/images/nv_switch.png)

_图 3：NVIDIA symmetric memory 相关 API_

## 方案设计
AMem NCCL-Plugin 基于 CUDA 的 VMM API，设计了**简洁的两层解耦方案**，实现了对 NCCL 显存**透明卸载和恢复的三重功能保障**。

+ **接口耦合层：NCCL Hook**：只修改了 NCCL 极少量代码，修改集中在几处显存相关（分配、释放、map）操作，保证了 NCCL 的核心逻辑不动，这样能做到：
    - 升级打 patch 比较方便；
    - 只调用几个简单的 AMem 管理显存元数据的 API，修改简单；
+ **功能实现解耦层：AMem Plugin**：核心逻辑封装在一个单独的 lib，独立于 NCCL 源码。其主要功能如下：
    - **元数据管理**：例如管理显存地址信息、引用信息、当前状态等；
    - **分布式引用识别和卸载**：实现跨进程和跨 rank 的动态溯源；
    - **分布式 resume：**根据元数据执行 redo，包括跨进程、跨 rank 重新导出和映射；
    - **进程组通信：**通过内部一套 Unix Domain Socket（UDS）实现 fd 跨进程传递；对训推进程进行逻辑分组以正确识别引用并避免误操作，这部分代码实现借鉴了团队之前开源的工作 [GLake](https://github.com/antgroup/glake)。



![](./docs/images/overall_arch.png)

_图4：AMem NCCL-plugin总体架构图_

### 功能保障一：溯源保障：交叉引用元数据
图 5 展示一个进程的 NCCL 显存（P2P buffer）地址（handle0）通过 VMM API 导出给其他多个进程。如果每个进程只释放自身地址而未等待 peer 释放，显存资源并不会归还给系统。

AMem NCCL-plugin 会动态跟踪记录“某个 handle 被哪些 peer 所引用”这一关键元信息，从而确保 **“释放时一个不漏，恢复时一个不少”** —— 返回之前引用均完成释放，恢复时，基于元数据记录来精确执行 redo。相关细节，详见下文图 7 的流程。

而针对共卡部署的场景，训练、推理进程在同一张 GPU 上并存，很容易分配得到相同的地址（仅在进程空间内有效），这会带来潜在的元数据问题。因此，我们增设一个 Group 的概念，用以区分不同进程的分配情况，保证元数据被正确记录。



![](./docs/images/p2p_mem_ref.png)

_图 5：NVIDIA P2P 显存交叉引用和处理（注：多卡对等，示例为简化展示）_

### 功能保障二：状态管理保障
AMem NCCL-plugin 对进程状态和每个 NCCL 显存分配地址（dptr）维护、更新内部状态，如图 6 所示，保证状态管理的完备和实时。

![](./docs/images/process_status.png)

_图6：进程和显存状态和转移示意_

### 功能保障三：流程保障：分布式卸载与恢复
通过内置的 UDS 通信，AMem NCCL-plugin 重点实现了跨进程 P2P reference 溯源、元数据更新和正确的 redo，流程上保证了分布式情况下依然可以有效实现卸载与恢复，具体流程如图 7 所示。

需要注意的是，多卡（rank）本质是对等关系，图中仅以 rank0 的视角示例，来说明核心流程。

![](./docs/images/workflow.png)

_图7：AMem NCCL-plugin分布式NCCL显存卸载与恢复流程_

## 总结 & 效果展示
AMem NCCL-plugin 可以将 NCCL 的显存几乎全部卸载，并按需恢复<sup>注3</sup>，同时保留 NCCL 通信组。能卸载的显存取决于集群规模、所用集合通信<sup>注4</sup>的通信组数量（特别是 AlltoAll）、并行策略（通常会 3D~5D 并行）以及CUDA/NCCL 版本等，大规模任务的 NCCL 显存开销可能会达到 10GB~20GB /GPU。目前无需重建通信组，使得 AMem 的 **NCCL 显存恢复耗时典型值不到 1s**<sup>注5</sup>。

![](./docs/images/result1.webp)        ![](./docs/images/result2.webp)

_图 8：AMem NCCL-plugin 可将 NCCL 分配的显存几乎全部卸载（左右为不同卡型）_

_注 3: cuda context 显存不卸载（典型~800MB），这部分显存会和训/推进程共用。_

_注 4: 常用的集合通信（分布式策略的一种实现方式）源语包括：（框架层）Broadcast, Scatter, Gather, Reduce；（训推）AllGather （所有的卡都执行 Gather），AllReduce, ReduceScatter, AlltoAll 等。_

_注 5: 首次卸载较慢（因为需要分配 CPU pinned buffer），后续通常 <1 sec。此处的 CPU pinned buffer，用来卸载 NCCL 的元数据、建联信息等，而用户分配的显存可以全部释放。_

## Getting Started 安装编译
### 代码
AMem NCCL-plugin 的产物主要是3个文件：扩展版的** nccl.h、libnccl.so.2 和 libamem_nccl.so。**

在保持 NCCL 现有功能的基础上，我们扩展了多个API以支持显存透明卸载、恢复和显存用量统计功能。

```c
///// NCCL.h新增了以下5个API

// 每个进程显式调用：ncclPause 返回则本卡的显存释放完毕、本卡引用其他卡的显存计数减1
// 注意:
// 1. Pause 和 Resume 是同步调用。Pause 之后不能再对NCCL调用。否则可能 crash、hang、invalid mem等
// 2. Pause 和 Resume 必须成对使用，按序调用。用户负责。否则调用可能无效，或异常
// 3. 多卡之间的状态一致性由调用者负责。例如必须等待多卡都完成了Resume，才能继续使用。
ncclResult_t ncclPause(ncclComm_t * comm = NULL);
ncclResult_t ncclResume(ncclComm_t* comm = NULL);

// 统计NCCL的显存分配总量、哪些func调用了显存分配。
ncclResult_t ncclMemStats();

// 如GPU上有多进程显式为同属一组的进程设立ID。AMem用此区分防止错误对显存溯源。例如
// GPU0 1...7上的训练进程，每个进程显式调用 设置为100
// GPU0 1...7上的推理进程，每个进程显式调用 设置为200
// 设置必须要在第一次的 NCCL 显存分配之前，否则不生效。
ncclResult_t ncclSetGroupID(int id);
ncclResult_t ncclGetGroupID(int* id);
```

**要求**：

1. NVIDIA GPU >= sm80
2. 推荐 CUDA >=12.2

注：首次编译耗时~10min；具体步骤见仓库README

构建步骤：

```yaml
# Recommend docker nvcr.io/nvidia/pytorch:25.08-py3
cd asystem-amem/ 

git submodule init
git submodule update
./build.sh
```

**NCCL显存统计：统计功能和pause/resume独立**

+ 调用 ncclMemStats()

```bash
AMEM groupID:170 pid:197780 caller_1 allocBytes:3024093184
AMEM groupID:170 pid:197780 caller_3 allocBytes:201326592
AMEM groupID:170 pid:197780 caller_7 allocBytes:2818572288
AMEM groupID:170 pid:197780 total allocBytes:6043992064 (5764 MB)
```



**重要参数：**

+ NCCL_CUMEM_ENABLE=1  #必须打开NCCL CUMEM
+ AMEM_ENABLE=1 # 激活 NCCL Mem 卸载与恢复。框架层需按需调用 API
+ AMEM_GROUPID=xxx  #为训练和推理进程组设置不同的 groupID
    - 注：当和 RL 框架集成时，以上环境变量需要传递给 Ray 或训推框架

**可选配置：**

+ AMEM_NCCL_OFFLOAD_FREE_TAG=7 #P2P buffer直接释放不做 offload CPU
+ GMM_LOG: 默认 3（INFO）。数字越大 log 越多，最大为 5

### 单元测试
基于 nccl-tests 快速测试典型并行下（如allreduce，allgather, alltoall等）动态显存卸载和恢复功能。它不依赖与任何框架，编译后测试通常耗时~10min。 

+ 为了测试 AMem nccl-plugin 需要进行少量修改：主要是调用 ncclPause()/ncclResume() 来激活功能。
+ 完整未修改版本：[https://github.com/NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests)  

```bash
# Run quick tests about nccl mem offloading/resume
export MPI_HOME=your/openmpi/home
bash ./run.sh
```

测试运行示例：

![](./docs/images/run_result.webp)

### 框架集成
AMem NCCL-plugin 不影响正常 NCCL 功能使用，而扩充了新的接口，用户可按需调用：

+ ncclPause()：释放该进程中NCCL所分配的显存，同步方式执行。
+ ncclResume()：恢复之前pause所释放的所有显存，同步方式执行。
+ ncclSetGroupID()：为当前程设置进程组。
+ ncclMemStats()：统计当前进程NCCL所使用的显存量和分类。

补充说明：

+ ncclPause 和 ncclResume 接口均是幂等的，即可以被多次调用而不会产生额外的影响。
+ 框架层负责必要的跨进程同步，确保所有rank均执行卸载、恢复完成。
+ 支持进程所创建的多个通信组（例如 3D/4D 并行）。
+ 如果应用并发只有一个任务在运行，例如只做推理或只做训练，不需要额外设置groupID。

#### pynccl
很多上层应用如 SGlang、vLLM 等，均支持 pynccl 调用方式，即将 NCCL 包装成一个 python 的接口，加载NCCL 动态库，打开 API 的函数句柄，然后通过 python 调用。以下示例 sglang 等调用 pynccl 方式。

#### SGLang
仅需修改 pynccl 以及 pynccl_wrapper 类。如下是修改 pynccl_wrapper 加载上述三个对应函数句柄的代码（注意：这里 ncclComm 的参数可以直接设置为 NULL）：

```python
# ncclResult_t ncclPause(ncclComm_t comm);
Function("ncclPause", ncclResult_t, [ncclComm_t]),
# ncclResult_t ncclResume(ncclComm_t comm);
Function("ncclResume", ncclResult_t, [ncclComm_t]),
Function("ncclSetGroupID", ncclResult_t, [ctypes.c_int]),
```

当需要释放 NCCL 显存时，参考以下代码示例：

```python
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_group

tp_group = get_tensor_model_parallel_group().pynccl_comm
if tp_group.nccl.enable_amem_nccl:
    tp_group.nccl_pause()
```

当需要恢复 NCCL 显存时，参考以下代码示例：

```python
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_group

tp_group = get_tensor_model_parallel_group().pynccl_comm
if tp_group.nccl.enable_amem_nccl:
    tp_group.nccl_resume()
```

#### Megatron
由于 Megatron 中并没有引入 pynccl，所以推荐以下集成和使用方式：

1. 在 Megatron 代码中类似地引入上述的 pynccl 类；
2. 在 Megatron 实例初始化时，初始化一个 pynccl 的对象；
3. 在需要显存释放和恢复的地方，按照以上 SGLang 中的例子显式调用对应的函数。

#### RL
RL框架包含了训练框架和推理框架。取决于部署形态，有以下两种集成方式：

1. 如果训推分离部署，集成方式参考上述 SGLang 和 Megatron 集成。
2. 如果是共卡方案，需要额外传递并设置 GroupID 以区分训推进程组。考虑到单个 GPU 上有两类进程（训和推），所以在 RL 框架中，当训练进程组初始化时，调用 ncclSetGroupID 设置一个 groupID；当推理进程组初始化时，类似设置一个不同的 groupID。如其他需要释放和恢复，参考以上使用说明。

## 后续规划
显存管理和优化是一个需要长期投入的过程。对于 NCCL 这类承载着历史兼容性包袱的库而言，更需要以精益求精的态度，通过持续的技术迭代来逐步完善。与此同时，社区的集思广益和丰富的应用场景验证，能够进一步推动这一优化的深入。我们后续的规划包括：

短期规划：

1. 支持 NCCL 2.28 版本；
2. 与 NCCL 社区沟通，探讨后续演进；
3. 针对 symmetric memory 做更有针对性的测试案例。

中长期规划：

1. 将 AMem 的实践应用在新型硬件；
2. 针对 agentic 的具体场景做进一步的优化；
3. 探索通信和显存的深入一体化管理和加速。

## 参考
+ Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model, [https://arxiv.org/abs/2510.18855](https://arxiv.org/abs/2510.18855)
+ GLake: [https://github.com/antgroup/glake](https://github.com/antgroup/glake) or ASPLOS24  [https://dl.acm.org/doi/abs/10.1145/3620665.3640423](https://dl.acm.org/doi/abs/10.1145/3620665.3640423) 
+ Zhiyi Hu, Siyuan Shen, Tommaso Bonato, Sylvain Jeaugey, Cedell Alexander, Eric Spada, James Dinan, Jeff Hammond, Torsten Hoefler.Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms, arXiv preprint arXiv:[2507.04786](https://arxiv.org/abs/2507.04786)
+ NVIDIA. NCCL 2.27. [https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/.](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/.)Accessed: 2025-10-10
+ Xiaolin Zhu. Slime V0.1.0. [https://zhuanlan.zhihu.com/p/1945237948166547268](https://zhuanlan.zhihu.com/p/1945237948166547268). Accessed: 2025-10-10
