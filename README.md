# AMem NCCL-plugin: NCCL memory transparent release and restore

## Overview
AMem NCCL-plugin is an NCCL extension library developed by Ant Group's Asystem team. Through a two-layer architecture and lightweight hooks, it achieves transparent release and restoration of NCCL GPU memory for the first time.  With large-scale reinforcement learning training scenarios, it could save up to 10~20GB of memory per card. It's already deplopyed with [Ring-1T](https://arxiv.org/abs/2510.18855) RL training.

Key features:
+ **NCCL distributed memory live release and resume**: resolves cross-rank memory cross-reference, achieving transparent memory release and restoration.
+ **Compatibility**: supports all NCCL parallelisms and mainstream GPUs (based on NCCL 2.27), integrated and verified with SGLang and Megatron. 
+ **Efficiency**: memory release can be done at a near-constant time, usually <1 second.


Below is a brief comparison:

| | NCCL | Slime | verl, AReal, openRLHF | AMem |
| --- | --- | --- | --- | --- |
| Live offload | <font style="color:#DF2A3F;">N</font> | <font style="color:#DF2A3F;">N</font>*<br/>restart process at the cost of communication rebuild overhead, potentially a few minutes. | <font style="color:#DF2A3F;">N</font> | Y<br/> |



![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1760150373401-663981e7-5f67-4375-9d09-abbd173da074.png)        ![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1760150400495-2eabb3bc-4c0a-4605-9a36-21d0436eae49.png)

Figure 1: AMem NCCL-plugin could  completely release NCCL allocated memory

+ CUDA context memory is out of release scope (typically ~800MB), which is shared with the training/rollout process.
+ The initial offloading may be a bit slow (because of the allocation of the CPU pinned buffer), subsequent offload is fast usually <1 second.
+ By default, all NCCL explicitly allocated memory is offloaded to CPU. P2P buffer memory is configurable to be released without offloading, significantly reducing the time.

### Challenges

Compared to memory offloading in Python, NCCL's transparent memory release faces additional challenges:
1. NCCL is implemented in C/C++, independent of the PyTorch, and is not supported by existing Python-based solutions.
2. **Distributed P2P memory cross-referencing**: In particular, unlike sharding data like weights etc., NCCL is designed for collective communication, introducing complex cross-rank, dynamic P2P reference in multi-GPU environments. Essentially, this involves a unique distributed memory cross-referencing problem, prone to memory leak.
3. **Complex logic **due to dynamic NCCL init, 3D/4D hybrid parallelism etc: may hit crashes or hangs.

### Design

AMem NCCL-Plugin, built upon CUDA VMM API, employs a two-layer decoupling scheme to achieve transparent GPU memory release and restoration.
● **Upper layer "NCCL Hook"**: NCCL codes are slightly modified just to extract memory metadata information (allocation, release, export etc).The core logic of NCCL remains unchanged, making upgrade or patch easy.
● **Core layer "AMem Plugin"**: The core logics are implemented in a separate library, including:
○ **Metadata management**: such as memory address, reference information, current status, etc.
○ **Distributed reference and release**: enables dynamic tracing across processes and ranks.
○ **Distributed resume**: redo based on metadata, e.g. cross-rank re-export and mapping.
○ **Process group (PG)**: cross-process metadata is exchanged through an internal Unix Domain Socket (UDS) system; Different tasks like training and rollout processes are logically seperated with a unique groupID to correctly identify references and avoid erroneous operations.

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1760162230824-512f8c55-c062-42d7-832a-0f4d610be057.png?x-oss-process=image%2Fformat%2Cwebp)

Figure 2：AMem NCCL-plugin architecture

Built on internal UDS communication, the overall NCCL memory release and resume routines are shown in Figure 3, e.g., cross-process P2P reference tracing, metadata updates, and redo .
+ Multi-rank essentially is a peer-to-peer relationship; the figure shown here only illustrates the core process from the perspective of rank 0.

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1760160197422-3cd52022-e09c-40ee-b547-b51bfbf48c86.png?x-oss-process=image%2Fformat%2Cwebp)

Figure 3: AMem NCCL-plugin distributed memory release and recovery routine

## Installation

AMem NCCL-plugin has three outputs: extended nccl.h, libnccl.so.2 (with our hook) and libamem_nccl.so
+ It keeps all NCCL existing functionalities while extends with the following new APIs: transparent GPU memory release and resume, internal GPU memory stats (e.g., memory usage for P2P)

```c
/*
 * NCCL Pause()/Resume() APIs by AMem NCCL-plugin that release (then re-alloc) phy GPU memory while keeping dptr unchanged
 * Pause(): offload (P2P buffer is optional) then release phy addr including references from peer, virtual dptr keep intact
 * Resume(): basically redo, e.g., alloc new phy addr, remap to existing dptr, preload data, notifer peer etc, thus everything is restored
 * Notes:
 * - both APIs are blocked until all release/offload/preload run to complete
 * - APIs shall be invoked in-order and in-pair e.g. Pause() then Resume()
 * - for inter-GPU state consistency, it shall be handled by caller frameworks like SGLang, Megatron-LM etc
 * - input argument comm currently is unused, just set as NULL
 */
ncclResult_t ncclPause(ncclComm_t * comm = NULL);
ncclResult_t ncclResume(ncclComm_t* comm = NULL);
ncclResult_t ncclMemStats();
ncclResult_t ncclSetGroupID(int id);
ncclResult_t ncclGetGroupID(int* id);

# Logic process groups to correctly track reference and avoid mis operation
# e.g., each training process shall set and share ID, different with rollout
# 100 for actor process on GPU0, 1...7
# 200 for rollout process on GPU0, 1...7
ncclResult_t ncclSetGroupID(int id);
ncclResult_t ncclGetGroupID(int* id);
```

Requirements:

1. NVIDIA GPU >= sm_80.  Functions are tested on 80/90/100
2. Recommend CUDA >=12.2
+ essentially it leverages NVIDIA[ Virtual Memory Management APIs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html) (introduced on CUDA 10.2, or so-called cuMem) for transparent GPU memory release then remap.

Build steps: 

```bash
# Recommend docker nvcr.io/nvidia/pytorch:25.08-py3
cd amem/ 

git submodule init
git submodule update
./build.sh

```

## Try it
Below shows tests using [nccl-test](https://github.com/NVIDIA/nccl-tests)（allreduce，allgather, alltoall, etc). 

```bash
# Run quick tests about nccl mem offloading/resume
export MPI_HOME=your/openmpi/home
bash ./run.sh
```

** Typical running logs**

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

****

**Import configs：**

+ NCCL_ENABLE_CUMEM=1  #cuMem must be enabled, it's env from NCCL
+ AMEM_ENABLE=1 # >=1 to eanble NCCL Mem offloading
+ AMEM_GROUPID=xxx  # Unique ID for actor or rollout 

**Optional configs:**

+ AMEM_NCCL_OFFLOAD_FREE_TAG=7 # Skip P2P buffer offload for fast speed
+ GMM_LOG:  default 3（INFO). Set at most as 5 for more log info

## Integration with frameworks
AMem NCCL-plugin expands NCCL interfaces with new features:  
-	`ncclPause()`: Releases the GPU memory allocated by NCCL in the current process, executed synchronously.  
-	`ncclResume()`: Restores all GPU memory released by the previous `pause`, executed synchronously.  
-	 `ncclSetGroupID()`: Sets the process logical groupID for the current process.  
-	 `ncclMemStats()`: Statistics on the amount of GPU memory used and the category of NCCL in the current process.  


Additional Notes:  
-	Both `ncclPause` and `ncclResume` are idempotent, meaning they may be called multiple times without side-effect.  
-	The framework shall handle necessary cross-rank synchronization, such as wait until all ranks complete ncclPause() and ncclResume().  
-	Supports multiple communication groups created by a process (e.g., for 3D/4D parallelisms).  
-	If only one task, e.g, rollout or training only, no additional ncclSetGroupID() is required.



### pynccl
Many high-level applications like SGlang and vLLM, support the pynccl interfaces which wrap NCCL lib into a Python interface, opens the API function handles, then calls it through Python. The following example demonstrates how SGlang calls pynccl.

Simple modifications of pynccl and pynccl_wrapper classes are required. Below we demo the code for modifying pynccl_wrapper to load the new NCCL APIs (note that the parameter ncclComm can be set to NULL):

```bash
# ncclResult_t ncclPause(ncclComm_t comm);
Function("ncclPause", ncclResult_t, [ncclComm_t]),
# ncclResult_t ncclResume(ncclComm_t comm);
Function("ncclResume", ncclResult_t, [ncclComm_t]),
Function("ncclSetGroupID", ncclResult_t, [ctypes.c_int]),
```

When need to release NCCL memory, refer to the following code example:

```bash
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_group

tp_group = get_tensor_model_parallel_group().pynccl_comm
if tp_group.nccl.enable_amem_nccl:
    tp_group.nccl_pause()
```

When need to restore NCCL memory, refer to the following code example:

```bash
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_group

tp_group = get_tensor_model_parallel_group().pynccl_comm
if tp_group.nccl.enable_amem_nccl:
    tp_group.nccl_resume()
```

### Megatron
Since Megatron does not use pynccl, the following integration is recommended:

1. Install the AMem enabled SGlang version integrated in the image.
2. Initialize a pynccl object when initializing the Megatron instance.
3. Explicitly call the new APIs as needed.



### RL
RL framework involves a training framework and rollout framework. Depending on the deployments, there are two integration methods:

+ If training and rollout are deployed separately, refer to the SGLang and Megatron integration methods described above.
+ For co-located deployment, groupID needs to be set to distinguish between training and inference process groups. When the training process group is initialized, `ncclSetGroupID` is called to set a groupID; when the rollout process group is initialized, a different groupID shall be set. After that, just refer to above examples.

## Code structure
Current implementation is based on NCCL 2.27.5 (June 2025).

+ **nccl_master**: modified nccl source code with AMem hook, AMem plugin core logics
    - amem_nccl_plugin/: core logics, internal cross-rank/cross-process unix domain socket 
        * amem_nccl.h, amem_nccl.cpp: metadata magt, pause, resume, stats
        * gmm*: internal unix domain socket, logical process group support. Most of those codes are refactored based on our another open-source project [GLake](https://github.com/antgroup/glake)
+ **nccl-tests**: NVIDIA nccl-tests with AMem ncclPause(), ncclResume() example
    - src/common.cu: run with ncclPause()/Resume()/MemStats()
+ **nccl_patch**: code change diff vs. original nccl-2.27.5 and nccl-tests

## Roadmap
- [ ] Support NCCL 2.28
- [ ] More tests on symm mem

and welcome contributions and integrations 

## Reference
+ AntGroup Ling team. Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model, [https://arxiv.org/abs/2510.18855](https://arxiv.org/abs/2510.18855)
+ GLake: [https://github.com/antgroup/glake](https://github.com/antgroup/glake) or GMLake ASPLOS24 [https://dl.acm.org/doi/abs/10.1145/3620665.3640423](https://dl.acm.org/doi/abs/10.1145/3620665.3640423) 
+ Zhiyi Hu, Siyuan Shen, Tommaso Bonato, Sylvain Jeaugey, Cedell Alexander, Eric Spada, James Dinan, Jeff Hammond, Torsten Hoefler.Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms, arXiv preprint arXiv:[2507.04786](https://arxiv.org/abs/2507.04786)
+ NVIDIA. NCCL 2.27. [https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/.](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/.) Accessed: 2025-10-10
+ Xiaolin Zhu. Slime V0.1.0. [https://zhuanlan.zhihu.com/p/1945237948166547268](https://zhuanlan.zhihu.com/p/1945237948166547268). Accessed: 2025-10-10
