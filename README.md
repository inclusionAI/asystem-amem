
# <font style="color:rgb(13, 18, 57);">AMem NCCL-Plugin: Transparent NCCL GPU Memory Offloading and Restoration</font>


<font style="color:rgb(0, 0, 0);">In recent years, Reinforcement Learning (RL) has become the core technology driving the expansion of the large model frontier. From ChatGPT's RLHF, to the post-training systems of DeepSeek, Claude, and Llama, all rely on reinforcement learning to make models align better with human preferences and possess stronger reasoning capabilities.</font>

<font style="color:rgb(0, 0, 0);">In October this year, </font>**<font style="color:rgb(0, 0, 0);">Ant Ling</font>**<font style="color:rgb(0, 0, 0);"> officially open-sourced two industry-leading trillion-parameter flagship models: the non-reasoning model </font>**<font style="color:rgb(0, 0, 0);">Ling-1T</font>**<font style="color:rgb(0, 0, 0);"> and the reasoning model </font>**<font style="color:rgb(0, 0, 0);">Ring-1T</font>**<font style="color:rgb(0, 0, 0);">. The training of a reasoning model on the massive scale of Ring-1T imposed extremely high demands on system engineering. This represents not just an algorithmic breakthrough, but a system engineering challenge that combines extreme technical depth and meticulous craftsmanship.</font>

**<font style="color:rgb(0, 0, 0);">It's time for our weekly update!</font>**<font style="color:rgb(0, 0, 0);"> This marks the second installment in our carefully planned technical analysis series: </font>**<font style="color:rgb(0, 0, 0);">"ASystem System Open-Source,"</font>**<font style="color:rgb(0, 0, 0);"> which explores core technologies.</font>

**<font style="color:rgb(0, 0, 0);">Today, we will unveil the second critical component in the ASystem stack:</font>**<font style="color:rgb(0, 0, 0);"> </font><font style="color:rgb(0, 0, 0);">AMem NCCL-Plugin. We will reveal how we resolve GPU memory bottlenecks and the time-consuming challenge of communication connection in RL training to achieve high-performance computing.</font>

<font style="color:rgb(0, 0, 0);">Please continue to follow our official account. Over the coming weeks, we will continue to release technical analyses of several more system-level key components—each installment is well worth the wait!</font>





## <font style="color:rgb(37, 39, 42);">TL；DR </font><font style="color:rgb(13, 18, 57);">Technical Overview</font>
<font style="color:rgb(13, 18, 57);">This week, we continue sharing another key component of the ASystem series: </font>**<font style="color:rgb(13, 18, 57);">AMem NCCL-Plugin</font>**<font style="color:rgb(13, 18, 57);">.</font>

**<font style="color:rgb(13, 18, 57);">NCCL</font>**<font style="color:rgb(13, 18, 57);"> </font><font style="color:rgb(13, 18, 57);">stands for NVIDIA Collective Communications Library. It is the core communication library for multi-GPU and multi-node distributed deep learning, providing highly efficient collective communication operations such as AllReduce and Broadcast.</font>

**<font style="color:rgb(13, 18, 57);">AMem NCCL-Plugin</font>**<font style="color:rgb(13, 18, 57);"> is a self-developed NCCL extension library by Ant Group’s ASystem team. It introduces two memory management APIs—</font>`<font style="color:rgb(13, 18, 57);">ncclPause()</font>`<font style="color:rgb(13, 18, 57);"> and </font>`<font style="color:rgb(13, 18, 57);">ncclResume()</font>`<font style="color:rgb(13, 18, 57);">—</font><font style="color:rgb(13, 18, 57);">to address a critical challenge in reinforcement learning (RL) workflows: the inability to efficiently offload GPU memory allocated by the NCCL communication library. Through a lightweight plugin approach, AMem enables transparent offloading and restoration of NCCL memory used by training/inference engines while preserving existing NCCL communication connections</font><sup>1</sup><font style="color:rgb(13, 18, 57);">. These advantages have already been validated in RL training for </font>**<font style="color:rgb(13, 18, 57);">Ring-1T, a trillion-parameter model</font>**<font style="color:rgb(13, 18, 57);">.</font>

<font style="color:rgb(13, 18, 57);">The benefits of AMem NCCL-Plugin are demonstrated in two key aspects:</font>

+ **<font style="color:rgb(13, 18, 57);">Memory Savings</font>**<font style="color:rgb(13, 18, 57);">: By identifying and resolving cross-rank GPU memory cross-references within the NCCL communication library, AMem correctly implements transparent memory release and restoration. During transitions between training and inference, it can free over 10 GB of GPU memory per card (Hopper architecture) while maintaining communication group connectivity.</font>
+ **<font style="color:rgb(13, 18, 57);">Extreme Efficiency</font>**<font style="color:rgb(13, 18, 57);">: Since communication group connections are preserved, switching between training and inference only requires offloading and restoring NCCL metadata—no need to rebuild communication connections (which typically takes minutes). This reduces typical transition latency from minutes to </font>**<font style="color:rgb(13, 18, 57);">under 1 second</font>**<font style="color:rgb(13, 18, 57);">.</font>

<font style="color:rgb(13, 18, 57);">Comparison with Community Solutions on Hopper Architecture GPUs:</font>

| System | Solution | Memory Saved | Per-step Offload/Reload Time |
| --- | --- | --- | --- |
| **Slime**<sup>****</sup> | <font style="color:rgb(13, 18, 57);">Clean NCCL GPU memory by destroying and recreating the training engine's communication group</font> | Inference: No saving (2 GB left)<br/>Training: Saves 10 GB+ | Several minutes |
| **Verl, OpenRLHF** | <font style="color:rgb(13, 18, 57);">Does not support offloading NCCL GPU memory</font> | Inference: No saving (2 GB left)<br/>Training: No saving (10 GB+ left) | 0s |
| **AMem** | <font style="color:rgb(13, 18, 57);">Offload and restore NCCL GPU memory via Plugin</font> | Inference: Saves 2 GB<br/>Training: Saves 10 GB+ | <1s |


_<font style="color:rgb(13, 18, 57);">Figure 1: Functional comparison of AMem NCCL-Plugin.</font>_

_**<font style="color:rgb(139, 139, 139);">Note 1</font>**__<font style="color:rgb(139, 139, 139);">:</font>_

+ _**<font style="color:rgb(139, 139, 139);">Memory Release</font>**__<font style="color:rgb(139, 139, 139);">: Returning GPU memory back to the OS.</font>_
+ _**<font style="color:rgb(139, 139, 139);">Memory Offload</font>**__<font style="color:rgb(139, 139, 139);">: Moving data from GPU memory into CPU pinned buffers, then releasing GPU memory.</font>_
+ _**<font style="color:rgb(139, 139, 139);">Memory Restore</font>**__<font style="color:rgb(139, 139, 139);">: Reallocating GPU memory and copying data back from CPU pinned buffers.</font>_

## Background Challenges
**<font style="color:rgb(13, 18, 57);">Co-location Deployment in Reinforcement Learning</font>**<font style="color:rgb(13, 18, 57);">:  
</font><font style="color:rgb(13, 18, 57);">In typical RL systems using co-located training and inference on the same GPU, after completing one task, GPU resources must be quickly and cleanly released for subsequent tasks to improve resource efficiency. While GPU compute units are stateless and can be released immediately after use, GPU memory is stateful—requiring careful management. For example:</font>

+ <font style="color:rgb(13, 18, 57);">Critical data must first be saved to host memory before freeing GPU memory.</font>
+ <font style="color:rgb(13, 18, 57);">When restoring, this data must be copied back accurately.</font>

<font style="color:rgb(13, 18, 57);">This poses significant technical challenges involving memory allocation, cross-process references, and state restoration.</font>

**<font style="color:rgb(13, 18, 57);">GPU Memory Management Complexity</font>**<font style="color:rgb(13, 18, 57);">:  
</font><font style="color:rgb(13, 18, 57);">CUDA provides multiple memory management APIs. To release GPU memory while keeping processes alive, Virtual Memory Management APIs (VMM or cuMem) must be used. These APIs offer two-layer address management and dynamic mapping capabilities (see Figure 2). Modern frameworks like PyTorch and NCCL already support optional VMM-based memory allocation.</font>

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1756363908841-9d94a4e0-9505-415d-82f7-d108d0a48d5a.png)

_<font style="color:rgb(13, 18, 57);">Figure 2: NVIDIA VMM Memory Management APIs and Typical Operations</font>_

<font style="color:rgb(13, 18, 57);">During memory management, all memory allocations must be traced. User-space allocations can generally be managed precisely. In RL scenarios, typical memory content requiring offloading includes:</font>

+ **<font style="color:rgb(13, 18, 57);">Training</font>**<font style="color:rgb(13, 18, 57);">: Weights, optimizer states, activations, NCCL memory, CUDA graphs, etc.</font>
+ **<font style="color:rgb(13, 18, 57);">Inference</font>**<font style="color:rgb(13, 18, 57);">: Weights, KV cache, activations, NCCL memory, CUDA graphs, etc.</font>

<font style="color:rgb(13, 18, 57);">While the community has made initial progress managing most memory types,</font><font style="color:rgb(13, 18, 57);"> </font>**<font style="color:rgb(13, 18, 57);">NCCL memory remains a notable gap</font>**<font style="color:rgb(13, 18, 57);">.</font>

**<font style="color:rgb(13, 18, 57);">Challenges in Offloading NCCL Memory</font>**<font style="color:rgb(13, 18, 57);">:  
</font><font style="color:rgb(13, 18, 57);">NCCL does not expose external interfaces for managing its allocated GPU memory, making it difficult to control. Common approaches include:</font>

1. **<font style="color:rgb(13, 18, 57);">Not releasing NCCL memory</font>**<font style="color:rgb(13, 18, 57);">: As shown in Figure 1, NCCL memory may occupy 10–20 GB, significantly limiting batch size—critical for throughput-intensive RL workloads. This approach avoids connection setup overhead per RL step.</font>
2. **<font style="color:rgb(13, 18, 57);">Destroying and recreating training/inference processes or communication groups</font>**<font style="color:rgb(13, 18, 57);">: This cleanly releases memory but incurs high initialization costs (typically minutes), though recent optimizations (e.g., from Meta) show potential.</font>

<font style="color:rgb(13, 18, 57);">Both approaches involve trade-offs: the first sacrifices memory for speed; the second trades time for memory. Our research focuses on achieving </font>**<font style="color:rgb(13, 18, 57);">both</font>**<font style="color:rgb(13, 18, 57);">.</font>

## Technical Challenges
<font style="color:rgb(13, 18, 57);">Compared to memory offloading in PyTorch/Python, transparent NCCL memory offloading faces three main challenges:</font>

1. **<font style="color:rgb(13, 18, 57);">NCCL is implemented in C/C++</font>**<font style="color:rgb(13, 18, 57);">, operating outside PyTorch’s memory pool—existing Python-based solutions don’t apply.</font>
2. **<font style="color:rgb(13, 18, 57);">Distributed P2P Memory Cross-References</font>**<font style="color:rgb(13, 18, 57);">: Unlike per-rank data (e.g., sharded weights, activations, KV cache), NCCL creates complex cross-rank P2P references for collective communication. Simply freeing local memory doesn’t release resources to the driver. Over multiple rounds, unreleased old buffers accumulate, causing NCCL memory usage to grow. This unique</font><font style="color:rgb(13, 18, 57);"> </font>**<font style="color:rgb(13, 18, 57);">distributed memory cross-reference problem</font>**<font style="color:rgb(13, 18, 57);"> </font><font style="color:rgb(13, 18, 57);">requires precise restoration—any mismatch risks crashes or hangs.</font>
3. **<font style="color:rgb(13, 18, 57);">Complex Logic from Dynamic Connections & Hybrid Parallelism</font>**<font style="color:rgb(13, 18, 57);">: NCCL is hard to modify, and corner cases are numerous during validation. For example, NVIDIA’s 2024 </font>**<font style="color:rgb(13, 18, 57);">symmetric memory</font>**<font style="color:rgb(13, 18, 57);"> (for NVSwitch-based high-speed collectives) introduces even more complex memory management logic (see Figure 3).</font>

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1756366491390-f7e6a696-9a9b-4966-9846-f48b8519e316.png)

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1756369092529-36512037-954b-43e2-8153-8ee3a93b3f51.png)

_<font style="color:rgb(13, 18, 57);">Figure 3: NVIDIA Symmetric Memory–Related APIs</font>_

## Solution Design
<font style="color:rgb(13, 18, 57);">AMem NCCL-Plugin leverages CUDA’s VMM APIs and employs a clean two-layer decoupled design to ensure</font><font style="color:rgb(13, 18, 57);"> </font>**<font style="color:rgb(13, 18, 57);">threefold guarantees</font>**<font style="color:rgb(13, 18, 57);"> </font><font style="color:rgb(13, 18, 57);">for transparent NCCL memory offloading and restoration.</font>

+ **<font style="color:rgb(13, 18, 57);">Interface Coupling Layer</font>**<font style="color:rgb(13, 18, 57);">—</font>**<font style="color:rgb(13, 18, 57);">NCCL Hook: </font>**<font style="color:rgb(13, 18, 57);">Minimal NCCL code modifications—only a few memory-related operations (allocation, deallocation, mapping) are altered. </font><font style="color:rgb(13, 18, 57);">Preserves NCCL’s core logic, enabling:</font>
    - <font style="color:rgb(13, 18, 57);">Easy patching during NCCL upgrades.</font>
    - <font style="color:rgb(13, 18, 57);">Simple integration via a few AMem metadata management APIs.</font>
+ **<font style="color:rgb(13, 18, 57);">Functional Decoupling Layer</font>****<font style="color:rgb(13, 18, 57);">—</font>****<font style="color:rgb(13, 18, 57);">AMem Plugin</font>**<font style="color:rgb(13, 18, 57);">: </font><font style="color:rgb(13, 18, 57);">Encapsulated in a standalone library (</font>`<font style="color:rgb(13, 18, 57);">libamem_nccl.so</font>`<font style="color:rgb(13, 18, 57);">), independent of NCCL source code. Key functions include:</font>
    - **<font style="color:rgb(13, 18, 57);">Metadata Management</font>**<font style="color:rgb(13, 18, 57);">: Tracks memory addresses, reference counts, and current states.</font>
    - **<font style="color:rgb(13, 18, 57);">Distributed Reference Identification & Offload</font>**<font style="color:rgb(13, 18, 57);">: Dynamically traces cross-process and cross-rank references.</font>
    - **<font style="color:rgb(13, 18, 57);">Distributed Resume</font>**<font style="color:rgb(13, 18, 57);">: Executes precise redo operations based on metadata, including cross-process/rank re-exporting and remapping.</font>
    - **<font style="color:rgb(13, 18, 57);">Process Group Communication</font>**<font style="color:rgb(13, 18, 57);">: Uses Unix Domain Sockets (UDS) to pass file descriptors across processes. Logical grouping of training/inference processes ensures correct reference tracking and prevents misoperations—inspired by our open-source project </font>[**GLake**](https://github.com/antgroup/glake)<font style="color:rgb(13, 18, 57);">.</font>



![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1760162230824-512f8c55-c062-42d7-832a-0f4d610be057.png?x-oss-process=image%2Fformat%2Cwebp)

_<font style="color:rgb(13, 18, 57);">Figure 4: Overall Architecture of AMem NCCL-Plugin</font>_

### Guarantee 1: Traceability via Cross-Reference Metadata
<font style="color:rgb(13, 18, 57);">Figure 5 illustrates how a process exports its NCCL P2P buffer (handle0) to multiple peers via VMM APIs. If each process frees its local address without waiting for peers, memory isn’t returned to the system.</font>

<font style="color:rgb(13, 18, 57);">AMem dynamically tracks</font><font style="color:rgb(13, 18, 57);"> </font>**<font style="color:rgb(13, 18, 57);">“which peers reference a given handle”</font>**<font style="color:rgb(13, 18, 57);">, ensuring:</font>

+ **<font style="color:rgb(13, 18, 57);">No missed releases</font>**<font style="color:rgb(13, 18, 57);"> </font><font style="color:rgb(13, 18, 57);">during offload.</font>
+ **<font style="color:rgb(13, 18, 57);">Exact restoration</font>**<font style="color:rgb(13, 18, 57);"> </font><font style="color:rgb(13, 18, 57);">during reload.</font>

<font style="color:rgb(13, 18, 57);">For co-located deployment (training + inference on the same GPU), identical virtual addresses may appear in different processes, risking metadata conflicts. To resolve this, AMem introduces a </font>**<font style="color:rgb(13, 18, 57);">Group concept</font>**<font style="color:rgb(13, 18, 57);"> to distinguish allocations across process groups.</font>



![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1756369240340-22501a29-62d8-45a5-9fd6-a3172ff1b938.png)

_<font style="color:rgb(13, 18, 57);">Figure 5: NVIDIA P2P Memory Cross-Reference and Handling (simplified multi-GPU example)</font>_

### Guarantee 2: State Management
<font style="color:rgb(13, 18, 57);">AMem maintains and updates internal states for each process and NCCL memory allocation (</font>`<font style="color:rgb(13, 18, 57);">dptr</font>`<font style="color:rgb(13, 18, 57);">), ensuring completeness and real-time accuracy (Figure 6).</font>

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1756364076528-5228dd66-7e8d-4c88-93a2-f502eb019bf2.png)

_<font style="color:rgb(13, 18, 57);">Figure 6: Process and Memory State Transitions</font>_

### Guarantee 3: Workflow Guarantee – Distributed Offload & Restore
<font style="color:rgb(13, 18, 57);">Using built-in UDS communication, AMem ensures correct cross-process P2P reference tracing, metadata updates, and redo execution—even in distributed settings (Figure 7). Note: Multi-rank systems are peer-to-peer; the diagram only shows rank0’s perspective for clarity.</font>

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1760160197422-3cd52022-e09c-40ee-b547-b51bfbf48c86.png?x-oss-process=image%2Fformat%2Cwebp)

_<font style="color:rgb(13, 18, 57);">Figure 7: Distributed NCCL Memory Offload & Restore Workflow</font>_

### <font style="color:rgb(13, 18, 57);">Summary & Results</font>
<font style="color:rgb(13, 18, 57);">AMem NCCL-Plugin can </font>**<font style="color:rgb(13, 18, 57);">nearly fully offload NCCL-allocated GPU memory</font>**<font style="color:rgb(13, 18, 57);"> and restore it on demand</font><sup>2</sup><font style="color:rgb(13, 18, 57);">, </font>**<font style="color:rgb(13, 18, 57);">without rebuilding NCCL communication groups</font>**<font style="color:rgb(13, 18, 57);">. The amount of offloadable memory depends on:</font>

+ <font style="color:rgb(13, 18, 57);">Cluster scale</font>
+ <font style="color:rgb(13, 18, 57);">Number of collective communication groups</font><sup>3</sup><font style="color:rgb(13, 18, 57);"> (especially AlltoAll)</font>
+ <font style="color:rgb(13, 18, 57);">Parallel strategy (typically 3D–5D)</font>
+ <font style="color:rgb(13, 18, 57);">CUDA/NCCL version</font>

<font style="color:rgb(13, 18, 57);">In large-scale tasks, NCCL memory overhead can reach </font>**<font style="color:rgb(13, 18, 57);">10–20 GB per GPU</font>**<font style="color:rgb(13, 18, 57);">. With AMem, restoration latency is typically </font>**<font style="color:rgb(13, 18, 57);">under 1 second</font>**<sup>**4**</sup><font style="color:rgb(13, 18, 57);">.</font>

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1760150373401-663981e7-5f67-4375-9d09-abbd173da074.png)        ![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/189864/1760150400495-2eabb3bc-4c0a-4605-9a36-21d0436eae49.png)

_<font style="color:rgb(13, 18, 57);">Figure 8: AMem NCCL-Plugin nearly fully offloads NCCL memory (left/right: different GPU types)</font>_

_**<font style="color:rgb(139, 139, 139);">Note 2</font>**__<font style="color:rgb(139, 139, 139);">: CUDA context memory (~800 MB) is </font>__**<font style="color:rgb(139, 139, 139);">not offloaded</font>**__<font style="color:rgb(139, 139, 139);">, as it’s shared between training/inference processes.  
</font>__**<font style="color:rgb(139, 139, 139);">Note 3</font>**__<font style="color:rgb(139, 139, 139);">: Common collective communication primitives include: Broadcast, Scatter, Gather, Reduce, AllGather, AllReduce, ReduceScatter, AlltoAll, etc.</font>_

_**<font style="color:rgb(139, 139, 139);">Note 4</font>**__<font style="color:rgb(139, 139, 139);">: First offload is slower (due to CPU pinned buffer allocation); subsequent operations take <1 sec. CPU pinned buffers store NCCL metadata and connection info; user-allocated GPU memory is fully released.</font>_

## Getting Started: Installation & Compilation
### <font style="color:rgb(13, 18, 57);">Code Artifacts</font>
<font style="color:rgb(13, 18, 57);">AMem NCCL-Plugin produces three files:</font>

+ <font style="color:rgb(13, 18, 57);">Extended</font><font style="color:rgb(13, 18, 57);"> </font>`<font style="color:rgb(13, 18, 57);">nccl.h</font>`
+ `<font style="color:rgb(13, 18, 57);">libnccl.so.2</font>`
+ `<font style="color:rgb(13, 18, 57);">libamem_nccl.so</font>`

<font style="color:rgb(13, 18, 57);">It extends NCCL with new APIs for transparent memory offload, restore, and usage statistics—</font>**<font style="color:rgb(13, 18, 57);">without altering existing functionality</font>**<font style="color:rgb(13, 18, 57);">.</font>

```c
///// The following 5 new APIs have been added to nccl.h

// Each process must explicitly call ncclPause(). Upon return, 
// the GPU memory on this device has been fully released, 
// and the reference count from this device to memory on other devices is decremented by 1.
//
// Notes:
// 1. ncclPause() and ncclResume() are synchronous calls. 
//    After calling ncclPause(), no further NCCL operations should be invoked; 
//    otherwise, crashes, hangs, or invalid memory accesses may occur.
// 2. ncclPause() and ncclResume() must be used in matched pairs and called in order. 
//    It is the user's responsibility to ensure this; otherwise, the calls may be ineffective or cause errors.
// 3. The caller is responsible for maintaining state consistency across multiple GPUs. 
//    For example, all GPUs must complete ncclResume() before NCCL operations can safely resume.
ncclResult_t ncclPause(ncclComm_t* comm = NULL);
ncclResult_t ncclResume(ncclComm_t* comm = NULL);

// Reports total NCCL GPU memory allocation and which functions triggered the allocations.
ncclResult_t ncclMemStats();

// When multiple processes coexist on the same GPU, they can explicitly assign a group ID 
// to indicate they belong to the same logical group. AMem uses this ID to correctly trace 
// memory references and avoid cross-group interference. For example:
//   - Training processes on GPUs 0–7 each explicitly call this API with group ID 100.
//   - Inference processes on GPUs 0–7 each explicitly call this API with group ID 200.
// This group ID must be set BEFORE the first NCCL memory allocation; otherwise, it will have no effect.
ncclResult_t ncclSetGroupID(int id);
ncclResult_t ncclGetGroupID(int* id);
```

#### <font style="color:rgb(13, 18, 57);">Requirements</font>
+ <font style="color:rgb(13, 18, 57);">NVIDIA GPU with compute capability ≥ sm80</font>
+ <font style="color:rgb(13, 18, 57);">Recommended: CUDA ≥ 12.2</font>

<font style="color:#000000;">First compilation takes ~10 minutes; see README for details.</font>

#### <font style="color:rgb(13, 18, 57);">Build Steps</font>
```yaml
# Recommend docker nvcr.io/nvidia/pytorch:25.08-py3
cd asystem-amem/ 

git submodule init
git submodule update
./build.sh
```

**<font style="color:rgb(13, 18, 57);">NCCL Memory Statistics</font>**<font style="color:rgb(13, 18, 57);"> (independent of pause/resume): call </font>`<font style="color:rgb(13, 18, 57);">ncclMemStats()</font>`

```bash
AMEM groupID:170 pid:197780 caller_1 allocBytes:3024093184
AMEM groupID:170 pid:197780 caller_3 allocBytes:201326592
AMEM groupID:170 pid:197780 caller_7 allocBytes:2818572288
AMEM groupID:170 pid:197780 total allocBytes:6043992064 (5764 MB)
```

#### <font style="color:rgb(13, 18, 57);">Key Environment Variables</font>
```bash
NCCL_ENABLE_CUMEM=1    # Required: enable NCCL CUMEM
AMEM_ENABLE=1          # Enable NCCL memory offload/restore
AMEM_GROUPID=xxx       # Assign distinct group IDs for training/inference processes
```

<font style="color:#000000;">When integrating with RL frameworks, pass these variables to Ray or the training/inference framework.</font>

#### <font style="color:rgb(13, 18, 57);">Optional Environment Variables</font>
```bash
AMEM_NCCL_OFFLOAD_FREE_TAG=7  # Directly free P2P buffers without CPU offload
GMM_LOG=3                     # Log level (default: 3/INFO; max: 5)
```

### Unit Testing
<font style="color:rgb(13, 18, 57);">Based on</font><font style="color:rgb(13, 18, 57);"> </font>`<font style="color:rgb(13, 18, 57);">nccl-tests</font>`<font style="color:rgb(13, 18, 57);">, validate dynamic memory offload/restore under typical parallel patterns (AllReduce, AllGather, AlltoAll, etc.).</font>

+ <font style="color:rgb(13, 18, 57);">Framework-independent</font>
+ <font style="color:rgb(13, 18, 57);">Takes ~10 minutes post-compilation</font>
+ <font style="color:rgb(13, 18, 57);">Requires minor modifications: insert calls to</font><font style="color:rgb(13, 18, 57);"> </font>`<font style="color:rgb(13, 18, 57);">ncclPause()</font>`<font style="color:rgb(13, 18, 57);">/</font>`<font style="color:rgb(13, 18, 57);">ncclResume()</font>`

<font style="color:rgb(13, 18, 57);">Original tests: </font>[<font style="color:rgb(94, 92, 230);">https://github.com/NVIDIA/nccl-tests</font>](https://github.com/NVIDIA/nccl-tests)

```bash
# Run quick tests about nccl mem offloading/resume
export MPI_HOME=your/openmpi/home
bash ./run.sh
```

<font style="color:rgb(13, 18, 57);">Test run example</font>：

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/115222/1764066210423-18377997-1e56-40eb-8158-7a712c400fd0.png)

### Framework Integration
<font style="color:rgb(13, 18, 57);">AMem NCCL-Plugin</font><font style="color:rgb(13, 18, 57);"> </font>**<font style="color:rgb(13, 18, 57);">does not affect normal NCCL usage</font>**<font style="color:rgb(13, 18, 57);"> </font><font style="color:rgb(13, 18, 57);">but adds new APIs:</font>

+ `<font style="color:rgb(13, 18, 57);">ncclPause()</font>`<font style="color:rgb(13, 18, 57);">: Synchronously releases NCCL-allocated GPU memory in the current process.</font>
+ `<font style="color:rgb(13, 18, 57);">ncclResume()</font>`<font style="color:rgb(13, 18, 57);">: Synchronously restores all memory previously released by</font><font style="color:rgb(13, 18, 57);"> </font>`<font style="color:rgb(13, 18, 57);">ncclPause()</font>`<font style="color:rgb(13, 18, 57);">.</font>
+ `<font style="color:rgb(13, 18, 57);">ncclSetGroupID()</font>`<font style="color:rgb(13, 18, 57);">: Sets a process group ID for the current process.</font>
+ `<font style="color:rgb(13, 18, 57);">ncclMemStats()</font>`<font style="color:rgb(13, 18, 57);">: Reports NCCL memory usage and breakdown.</font>

<font style="color:rgb(13, 18, 57);">Additional Notes:</font>

+ `<font style="color:rgb(13, 18, 57);">ncclPause</font>`<font style="color:rgb(13, 18, 57);">/</font>`<font style="color:rgb(13, 18, 57);">ncclResume</font>`<font style="color:rgb(13, 18, 57);"> </font><font style="color:rgb(13, 18, 57);">are</font><font style="color:rgb(13, 18, 57);"> </font>**<font style="color:rgb(13, 18, 57);">idempotent</font>**<font style="color:rgb(13, 18, 57);"> </font><font style="color:rgb(13, 18, 57);">(safe for repeated calls).</font>
+ <font style="color:rgb(13, 18, 57);">The framework must ensure</font><font style="color:rgb(13, 18, 57);"> </font>**<font style="color:rgb(13, 18, 57);">cross-process synchronization</font>**<font style="color:rgb(13, 18, 57);"> </font><font style="color:rgb(13, 18, 57);">so all ranks complete offload/restore.</font>
+ <font style="color:rgb(13, 18, 57);">Supports</font><font style="color:rgb(13, 18, 57);"> </font>**<font style="color:rgb(13, 18, 57);">multiple communication groups</font>**<font style="color:rgb(13, 18, 57);"> </font><font style="color:rgb(13, 18, 57);">per process (e.g., 3D/4D parallelism).</font>
+ <font style="color:rgb(13, 18, 57);">If only one task runs at a time (e.g., inference-only or training-only), </font>`<font style="color:rgb(13, 18, 57);">groupID</font>`<font style="color:rgb(13, 18, 57);"> is unnecessary.</font>

#### PyNCCL Integration
<font style="color:rgb(13, 18, 57);">Many upper-layer applications (e.g., SGLang, vLLM) use </font>**<font style="color:rgb(13, 18, 57);">PyNCCL</font>**<font style="color:rgb(13, 18, 57);">—a Python wrapper that loads NCCL’s dynamic library and exposes APIs via function handles.</font>

#### SGLang Example
<font style="color:rgb(13, 18, 57);">Modify </font>`<font style="color:rgb(13, 18, 57);">pynccl</font>`<font style="color:rgb(13, 18, 57);"> and </font>`<font style="color:rgb(13, 18, 57);">pynccl_wrapper</font>`<font style="color:rgb(13, 18, 57);"> to load the three new function handles. ( </font>`<font style="color:rgb(13, 18, 57);">ncclComm</font>`<font style="color:rgb(13, 18, 57);"> parameter can be set to NULL. )</font>

```python
# ncclResult_t ncclPause(ncclComm_t comm);
Function("ncclPause", ncclResult_t, [ncclComm_t]),
# ncclResult_t ncclResume(ncclComm_t comm);
Function("ncclResume", ncclResult_t, [ncclComm_t]),
Function("ncclSetGroupID", ncclResult_t, [ctypes.c_int]),
```

**<font style="color:rgb(13, 18, 57);">To offload NCCL memory:</font>**

```python
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_group

tp_group = get_tensor_model_parallel_group().pynccl_comm
if tp_group.nccl.enable_amem_nccl:
    tp_group.nccl_pause()
```

**<font style="color:rgb(13, 18, 57);">To restore NCCL memory:</font>**

```python
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_group

tp_group = get_tensor_model_parallel_group().pynccl_comm
if tp_group.nccl.enable_amem_nccl:
    tp_group.nccl_resume()
```

#### <font style="color:rgb(13, 18, 57);">Megatron Integration</font>
<font style="color:rgb(13, 18, 57);">Since Megatron doesn’t use PyNCCL:</font>

1. <font style="color:rgb(13, 18, 57);">Introduce a PyNCCL-like class in Megatron code.</font>
2. <font style="color:rgb(13, 18, 57);">Initialize a PyNCCL object during Megatron instance setup.</font>
3. <font style="color:rgb(13, 18, 57);">Explicitly call offload/restore functions as in the SGLang example.</font>

#### RL Framework Integration
<font style="color:rgb(13, 18, 57);">RL frameworks combine training and inference components. Integration depends on deployment mode:</font>

+ **<font style="color:rgb(13, 18, 57);">Separate Training/Inference</font>**<font style="color:rgb(13, 18, 57);">: Follow SGLang/Megatron integration.</font>
+ **<font style="color:rgb(13, 18, 57);">Co-located Deployment</font>**<font style="color:rgb(13, 18, 57);">: Set distinct</font><font style="color:rgb(13, 18, 57);"> </font>`<font style="color:rgb(13, 18, 57);">groupID</font>`<font style="color:rgb(13, 18, 57);">s for training and inference process groups. During initialization:</font>
    - <font style="color:rgb(13, 18, 57);">Training process group: call</font><font style="color:rgb(13, 18, 57);"> </font>`<font style="color:rgb(13, 18, 57);">ncclSetGroupID(group_id_train)</font>`
    - <font style="color:rgb(13, 18, 57);">Inference process group: call </font>`<font style="color:rgb(13, 18, 57);">ncclSetGroupID(group_id_infer)</font>`
+ <font style="color:rgb(13, 18, 57);">Other usage follows previous guidelines.</font>

## <font style="color:rgb(37, 39, 42);">Future Roadmap</font>
<font style="color:rgb(13, 18, 57);">Memory management and optimization require sustained investment. For legacy-compatible libraries like NCCL, continuous iteration and meticulous engineering are essential. Community collaboration and diverse real-world validations will further drive improvements.</font>

#### <font style="color:rgb(13, 18, 57);">Short-Term Plans:</font>
+ <font style="color:rgb(13, 18, 57);">Support NCCL 2.28</font>
+ <font style="color:rgb(13, 18, 57);">Engage with NCCL community on future evolution</font>
+ <font style="color:rgb(13, 18, 57);">Develop targeted test cases for symmetric memory</font>

#### <font style="color:rgb(13, 18, 57);">Mid-to-Long-Term Plans:</font>
+ <font style="color:rgb(13, 18, 57);">Apply AMem practices to next-gen hardware</font>
+ <font style="color:rgb(13, 18, 57);">Optimize for agentic AI scenarios</font>
+ <font style="color:rgb(13, 18, 57);">Explore deep integration of communication and memory management for acceleration</font>

## References
+ Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model, [https://arxiv.org/abs/2510.18855](https://arxiv.org/abs/2510.18855)
+ GLake: [https://github.com/antgroup/glake](https://github.com/antgroup/glake) or ASPLOS24  [https://dl.acm.org/doi/abs/10.1145/3620665.3640423](https://dl.acm.org/doi/abs/10.1145/3620665.3640423) 
+ Zhiyi Hu, Siyuan Shen, Tommaso Bonato, Sylvain Jeaugey, Cedell Alexander, Eric Spada, James Dinan, Jeff Hammond, Torsten Hoefler.Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms, arXiv preprint arXiv:[2507.04786](https://arxiv.org/abs/2507.04786)
+ NVIDIA. NCCL 2.27. [https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/.](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/.)Accessed: 2025-10-10









<font style="color:rgb(13, 18, 57);">We warmly welcome every developer interested in reinforcement learning and large language models to try </font>**<font style="color:rgb(13, 18, 57);">AMem NCCL-Plugin</font>**<font style="color:rgb(13, 18, 57);"> and share your valuable feedback and suggestion together, let’s drive continuous innovation in RL systems!</font>

<font style="color:rgb(13, 18, 57);">📦</font><font style="color:rgb(13, 18, 57);"> </font>**<font style="color:rgb(13, 18, 57);">GitHub Repository</font>**<font style="color:rgb(13, 18, 57);">:</font><font style="color:rgb(13, 18, 57);"> </font>[<font style="color:rgb(94, 92, 230);">https://github.com/inclusionAI/asystem-amem</font>](https://github.com/inclusionAI/asystem-amem)<font style="color:rgb(13, 18, 57);">  
</font><font style="color:rgb(13, 18, 57);">⭐</font><font style="color:rgb(13, 18, 57);"> Please feel free to</font><font style="color:rgb(13, 18, 57);"> </font>**<font style="color:rgb(13, 18, 57);">Star</font>**<font style="color:rgb(13, 18, 57);"> </font><font style="color:rgb(13, 18, 57);">and</font><font style="color:rgb(13, 18, 57);"> </font>**<font style="color:rgb(13, 18, 57);">Fork</font>**<font style="color:rgb(13, 18, 57);"> </font><font style="color:rgb(13, 18, 57);">the repo, and we’d love to see your</font><font style="color:rgb(13, 18, 57);"> </font>**<font style="color:rgb(13, 18, 57);">PRs</font>**<font style="color:rgb(13, 18, 57);">!</font>

<font style="color:rgb(13, 18, 57);"></font>

<font style="color:rgb(13, 18, 57);"></font>

<font style="color:rgb(13, 18, 57);">Stay tuned for the latest releases from Ant Group’s </font>**<font style="color:rgb(13, 18, 57);">Bailing Models</font>**<font style="color:rgb(13, 18, 57);">:  
</font><font style="color:rgb(13, 18, 57);">🤗</font><font style="color:rgb(13, 18, 57);"> </font>**<font style="color:rgb(13, 18, 57);">Hugging Face</font>**<font style="color:rgb(13, 18, 57);">: </font>[<font style="color:rgb(94, 92, 230);">https://huggingface.co/inclusionAI</font>](https://huggingface.co/inclusionAI)<font style="color:rgb(13, 18, 57);">  
</font><font style="color:rgb(13, 18, 57);">🤖</font><font style="color:rgb(13, 18, 57);"> </font>**<font style="color:rgb(13, 18, 57);">ModelScope Community</font>**<font style="color:rgb(13, 18, 57);">: </font>[<font style="color:rgb(94, 92, 230);">https://www.modelscope.cn/organization/inclusionAI</font>](https://www.modelscope.cn/organization/inclusionAI)

<font style="color:rgb(94, 92, 230);"></font>

<font style="color:rgb(94, 92, 230);"></font>

<font style="color:rgb(13, 18, 57);">The</font><font style="color:rgb(13, 18, 57);"> </font>**<font style="color:rgb(13, 18, 57);">Ant ASystem team</font>**<font style="color:rgb(13, 18, 57);"> </font><font style="color:rgb(13, 18, 57);">is also actively hiring top talent from the industry. If you’re passionate about reinforcement learning, training/inference engines, and pushing the boundaries of cutting-edge systems in a rapidly evolving world, we’d love for you to join us!</font>

<font style="color:rgb(13, 18, 57);">Interested candidates can apply via:  
</font>**<font style="color:rgb(13, 18, 57);">Ant Group – Training & Inference System R&D Expert – Hangzhou / Beijing / Shanghai</font>**<font style="color:rgb(13, 18, 57);">  
</font>[<font style="color:rgb(94, 92, 230);">https://talent.antgroup.com/off-campus-position?positionId=25052904956438&tid=0b442eeb17633881544247991e1cc0</font>](https://talent.antgroup.com/off-campus-position?positionId=25052904956438&tid=0b442eeb17633881544247991e1cc0)

<font style="color:rgb(13, 18, 57);">Or send your resume directly to: </font>[**<font style="color:rgb(94, 92, 230);">ASystem@service.alipay.com</font>**](mailto:ASystem@service.alipay.com)

