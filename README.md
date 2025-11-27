
# AMem NCCL-Plugin: Transparent NCCL GPU Memory Offloading and Restoration





## TL；DR Technical Overview

**NCCL** stands for NVIDIA Collective Communications Library. It is the core communication library for multi-GPU and multi-node distributed deep learning, providing highly efficient collective communication operations such as AllReduce and Broadcast.

**AMem NCCL-Plugin** is a self-developed NCCL extension library by Ant Group’s ASystem team. It introduces two memory management APIs—`ncclPause()` and `ncclResume()`—to address a critical challenge in reinforcement learning (RL) workflows: the inability to efficiently offload GPU memory allocated by the NCCL communication library. Through a lightweight plugin approach, AMem enables transparent offloading and restoration of NCCL memory used by training/inference engines while preserving existing NCCL communication connections<sup>1</sup>. These advantages have already been validated in RL training for **Ring-1T, a trillion-parameter model**.

The benefits of AMem NCCL-Plugin are demonstrated in two key aspects:

+ **Memory Savings**: By identifying and resolving cross-rank GPU memory cross-references within the NCCL communication library, AMem correctly implements transparent memory release and restoration. During transitions between training and inference, it can free over 10 GB of GPU memory per card (Hopper architecture) while maintaining communication group connectivity.
+ **Extreme Efficiency**: Since communication group connections are preserved, switching between training and inference only requires offloading and restoring NCCL metadata—no need to rebuild communication connections (which typically takes seconds). This reduces typical transition latency to **under 1 second**.

Comparison with Community Solutions on Hopper Architecture GPUs:

| System | Solution | Memory Saved | Per-step Offload/Reload Time |
| --- | --- | --- | --- |
| **Slime** | Clean NCCL GPU memory by destroying and recreating the training engine's communication group | Inference: No saving (2 GB left)<br/>Training: Saves 10 GB+ | Several seconds |
| **OpenRLHF** | Does not support offloading NCCL GPU memory | Inference: No saving (2 GB left)<br/>Training: No saving (10 GB+ left) | 0s |
| **AMem** | Offload and restore NCCL GPU memory via Plugin | Inference: Saves 2 GB<br/>Training: Saves 10 GB+ | <1s |


_Figure 1: Functional comparison of AMem NCCL-Plugin._

_**Note 1**:_

+ _**Memory Release**: Returning GPU memory back to the OS._
+ _**Memory Offload**: Moving data from GPU memory into CPU pinned buffers, then releasing GPU memory._
+ _**Memory Restore**: Reallocating GPU memory and copying data back from CPU pinned buffers._

## Background Challenges
**Co-location Deployment in Reinforcement Learning**:  
In typical RL systems using co-located training and inference on the same GPU, after completing one task, GPU resources must be quickly and cleanly released for subsequent tasks to improve resource efficiency. While GPU compute units are stateless and can be released immediately after use, GPU memory is stateful—requiring careful management. For example:

+ Critical data must first be saved to host memory before freeing GPU memory.
+ When restoring, this data must be copied back accurately.

This poses significant technical challenges involving memory allocation, cross-process references, and state restoration.

**GPU Memory Management Complexity**:  
CUDA provides multiple memory management APIs. To release GPU memory while keeping processes alive, Virtual Memory Management APIs (VMM or cuMem) must be used. These APIs offer two-layer address management and dynamic mapping capabilities (see Figure 2). Modern frameworks like PyTorch and NCCL already support optional VMM-based memory allocation.

![](./docs/images/vmm_api_ops.png)

_Figure 2: NVIDIA VMM Memory Management APIs and Typical Operations_

During memory management, all memory allocations must be traced. User-space allocations can generally be managed precisely. In RL scenarios, typical memory content requiring offloading includes:

+ **Training**: Weights, optimizer states, activations, NCCL memory, CUDA graphs, etc.
+ **Inference**: Weights, KV cache, activations, NCCL memory, CUDA graphs, etc.

While the community has made initial progress managing most memory types, **NCCL memory remains a notable gap**.

**Challenges in Offloading NCCL Memory**:  
NCCL does not expose external interfaces for managing its allocated GPU memory, making it difficult to control. Common approaches include:

1. **Not releasing NCCL memory**: As shown in Figure 1, NCCL memory may occupy 10–20 GB, significantly limiting batch size—critical for throughput-intensive RL workloads. This approach avoids connection setup overhead per RL step.
2. **Destroying and recreating training/inference processes or communication groups**: This cleanly releases memory but incurs high initialization costs.

Both approaches involve trade-offs: the first sacrifices memory for speed; the second trades time for memory. Our research focuses on achieving **both**.

## Technical Challenges
Compared to memory offloading in PyTorch/Python, transparent NCCL memory offloading faces three main challenges:

1. **NCCL is implemented in C/C++**, operating outside PyTorch’s memory pool—existing Python-based solutions don’t apply.
2. **Distributed P2P Memory Cross-References**: Unlike per-rank data (e.g., sharded weights, activations, KV cache), NCCL creates complex cross-rank P2P references for collective communication. Simply freeing local memory doesn’t release resources to the driver. Over multiple rounds, unreleased old buffers accumulate, causing NCCL memory usage to grow. This unique **distributed memory cross-reference problem** requires precise restoration—any mismatch risks crashes or hangs.
3. **Complex Logic from Dynamic Connections & Hybrid Parallelism**: NCCL is hard to modify, and corner cases are numerous during validation. For example, NVIDIA’s 2024 **symmetric memory** (for NVSwitch-based high-speed collectives) introduces even more complex memory management logic (see Figure 3).

![](./docs/images/sym_mem.png)

![](./docs/images/nv_switch.png)

_Figure 3: NVIDIA Symmetric Memory–Related APIs_

## Solution Design
AMem NCCL-Plugin leverages CUDA’s VMM APIs and employs a clean two-layer decoupled design to ensure **threefold guarantees** for transparent NCCL memory offloading and restoration.

+ **Interface Coupling Layer—NCCL Hook**: Minimal NCCL code modifications—only a few memory-related operations (allocation, deallocation, mapping) are altered. Preserves NCCL’s core logic, enabling:
    - Easy patching during NCCL upgrades.
    - Simple integration via a few AMem metadata management APIs.
+ **Functional Decoupling Layer—AMem Plugin**: Encapsulated in a standalone library (`libamem_nccl.so`), independent of NCCL source code. Key functions include:
    - **Metadata Management**: Tracks memory addresses, reference counts, and current states.
    - **Distributed Reference Identification & Offload**: Dynamically traces cross-process and cross-rank references.
    - **Distributed Resume**: Executes precise redo operations based on metadata, including cross-process/rank re-exporting and remapping.
    - **Process Group Communication**: Uses Unix Domain Sockets (UDS) to pass file descriptors across processes. Logical grouping of training/inference processes ensures correct reference tracking and prevents misoperations—inspired by our open-source project [**GLake**](https://github.com/antgroup/glake).



![](./docs/images/overall_arch.png)

_Figure 4: Overall Architecture of AMem NCCL-Plugin_

### Guarantee 1: Traceability via Cross-Reference Metadata
Figure 5 illustrates how a process exports its NCCL P2P buffer (handle0) to multiple peers via VMM APIs. If each process frees its local address without waiting for peers, memory isn’t returned to the system.

AMem dynamically tracks **“which peers reference a given handle”**, ensuring:

+ **No missed releases** during offload.
+ **Exact restoration** during reload.

For co-located deployment (training + inference on the same GPU), identical virtual addresses may appear in different processes, risking metadata conflicts. To resolve this, AMem introduces a **Group concept** to distinguish allocations across process groups.



![](./docs/images/p2p_mem_ref.png)

_Figure 5: NVIDIA P2P Memory Cross-Reference and Handling (simplified multi-GPU example)_

### Guarantee 2: State Management
AMem maintains and updates internal states for each process and NCCL memory allocation (`dptr`), ensuring completeness and real-time accuracy (Figure 6).

![](./docs/images/process_status.png)

_Figure 6: Process and Memory State Transitions_

### Guarantee 3: Workflow Guarantee – Distributed Offload & Restore
Using built-in UDS communication, AMem ensures correct cross-process P2P reference tracing, metadata updates, and redo execution—even in distributed settings (Figure 7). Note: Multi-rank systems are peer-to-peer; the diagram only shows rank0’s perspective for clarity.

![](./docs/images/workflow.png)

_Figure 7: Distributed NCCL Memory Offload & Restore Workflow_

### Summary & Results
AMem NCCL-Plugin can **nearly fully offload NCCL-allocated GPU memory** and restore it on demand<sup>2</sup>, **without rebuilding NCCL communication groups**. The amount of offloadable memory depends on:

+ Cluster scale
+ Number of collective communication groups<sup>3</sup> (especially AlltoAll)
+ Parallel strategy (typically 3D–5D)
+ CUDA/NCCL version

In large-scale tasks, NCCL memory overhead can reach **10–20 GB per GPU**. With AMem, restoration latency is typically **under 1 second**<sup>**4**</sup>.

![](./docs/images/result1.webp)        ![](./docs/images/result2.webp)

_Figure 8: AMem NCCL-Plugin nearly fully offloads NCCL memory (left/right: different GPU types)_

_**Note 2**: CUDA context memory (~800 MB) is __**not offloaded**__, as it’s shared between training/inference processes._
_**Note 3**: Common collective communication primitives include: Broadcast, Scatter, Gather, Reduce, AllGather, AllReduce, ReduceScatter, AlltoAll, etc._

_**Note 4**: First offload is slower (due to CPU pinned buffer allocation); subsequent operations take <1 sec. CPU pinned buffers store NCCL metadata and connection info; user-allocated GPU memory is fully released._

## Getting Started: Installation & Compilation
### Code Artifacts
AMem NCCL-Plugin produces three files:

+ Extended `nccl.h`
+ `libnccl.so.2`
+ `libamem_nccl.so`

It extends NCCL with new APIs for transparent memory offload, restore, and usage statistics—**without altering existing functionality**.

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

#### Requirements
+ NVIDIA GPU with compute capability ≥ sm80
+ Recommended: CUDA ≥ 12.2

<font style="color:#000000;">First compilation takes ~10 minutes; see README for details.

#### Build Steps
```yaml
# Recommend docker nvcr.io/nvidia/pytorch:25.08-py3
cd asystem-amem/ 

git submodule init
git submodule update
./build.sh
```

**NCCL Memory Statistics** (independent of pause/resume): call `ncclMemStats()`

```bash
AMEM groupID:170 pid:197780 caller_1 allocBytes:3024093184
AMEM groupID:170 pid:197780 caller_3 allocBytes:201326592
AMEM groupID:170 pid:197780 caller_7 allocBytes:2818572288
AMEM groupID:170 pid:197780 total allocBytes:6043992064 (5764 MB)
```

#### Key Environment Variables
```bash
NCCL_CUMEM_ENABLE=1    # Required: enable NCCL CUMEM
AMEM_ENABLE=1          # Enable NCCL memory offload/restore
AMEM_GROUPID=xxx       # Assign distinct group IDs for training/inference processes
```

<font style="color:#000000;">When integrating with RL frameworks, pass these variables to Ray or the training/inference framework.

#### Optional Environment Variables
```bash
AMEM_NCCL_OFFLOAD_FREE_TAG=7  # Directly free P2P buffers without CPU offload
GMM_LOG=3                     # Log level (default: 3/INFO; max: 5)
```

### Unit Testing
Based on `nccl-tests`, validate dynamic memory offload/restore under typical parallel patterns (AllReduce, AllGather, AlltoAll, etc.).

+ Framework-independent
+ Takes ~10 minutes post-compilation
+ Requires minor modifications: insert calls to `ncclPause()`/`ncclResume()`

Original tests: [<font style="color:rgb(94, 92, 230);">https://github.com/NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests)

```bash
# Run quick tests about nccl mem offloading/resume
export MPI_HOME=your/openmpi/home
bash ./run.sh
```

Test run example：

![](./docs/images/run_result.webp)

### Framework Integration
AMem NCCL-Plugin **does not affect normal NCCL usage** but adds new APIs:

+ `ncclPause()`: Synchronously releases NCCL-allocated GPU memory in the current process.
+ `ncclResume()`: Synchronously restores all memory previously released by `ncclPause()`.
+ `ncclSetGroupID()`: Sets a process group ID for the current process.
+ `ncclMemStats()`: Reports NCCL memory usage and breakdown.

Additional Notes:

+ `ncclPause`/`ncclResume` are **idempotent** (safe for repeated calls).
+ The framework must ensure **cross-process synchronization** so all ranks complete offload/restore.
+ Supports **multiple communication groups** per process (e.g., 3D/4D parallelism).
+ If only one task runs at a time (e.g., inference-only or training-only), `groupID` is unnecessary.

#### PyNCCL Integration
Many upper-layer applications (e.g., SGLang, vLLM) use **PyNCCL**—a Python wrapper that loads NCCL’s dynamic library and exposes APIs via function handles.

#### SGLang Example
Modify `pynccl` and `pynccl_wrapper` to load the three new function handles. ( `ncclComm` parameter can be set to NULL. )

```python
# ncclResult_t ncclPause(ncclComm_t comm);
Function("ncclPause", ncclResult_t, [ncclComm_t]),
# ncclResult_t ncclResume(ncclComm_t comm);
Function("ncclResume", ncclResult_t, [ncclComm_t]),
Function("ncclSetGroupID", ncclResult_t, [ctypes.c_int]),
```

**To offload NCCL memory:**

```python
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_group

tp_group = get_tensor_model_parallel_group().pynccl_comm
if tp_group.nccl.enable_amem_nccl:
    tp_group.nccl_pause()
```

**To restore NCCL memory:**

```python
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_group

tp_group = get_tensor_model_parallel_group().pynccl_comm
if tp_group.nccl.enable_amem_nccl:
    tp_group.nccl_resume()
```

#### Megatron Integration
Since Megatron doesn’t use PyNCCL:

1. Introduce a PyNCCL-like class in Megatron code.
2. Initialize a PyNCCL object during Megatron instance setup.
3. Explicitly call offload/restore functions as in the SGLang example.

#### RL Framework Integration
RL frameworks combine training and inference components. Integration depends on deployment mode:

+ **Separate Training/Inference**: Follow SGLang/Megatron integration.
+ **Co-located Deployment**: Set distinct `groupID`s for training and inference process groups. During initialization:
    - Training process group: call `ncclSetGroupID(group_id_train)`
    - Inference process group: call `ncclSetGroupID(group_id_infer)`
+ Other usage follows previous guidelines.

## Future Roadmap
Memory management and optimization require sustained investment. For legacy-compatible libraries like NCCL, continuous iteration and meticulous engineering are essential. Community collaboration and diverse real-world validations will further drive improvements.

#### Short-Term Plans:
+ Support NCCL 2.28
+ Engage with NCCL community on future evolution
+ Develop targeted test cases for symmetric memory

#### Mid-to-Long-Term Plans:
+ Apply AMem practices to next-gen hardware
+ Optimize for agentic AI scenarios
+ Explore deep integration of communication and memory management for acceleration

## References
+ Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model, [https://arxiv.org/abs/2510.18855](https://arxiv.org/abs/2510.18855)
+ GLake: [https://github.com/antgroup/glake](https://github.com/antgroup/glake) or ASPLOS24  [https://dl.acm.org/doi/abs/10.1145/3620665.3640423](https://dl.acm.org/doi/abs/10.1145/3620665.3640423) 
+ Zhiyi Hu, Siyuan Shen, Tommaso Bonato, Sylvain Jeaugey, Cedell Alexander, Eric Spada, James Dinan, Jeff Hammond, Torsten Hoefler.Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms, arXiv preprint arXiv:[2507.04786](https://arxiv.org/abs/2507.04786)
+ NVIDIA. NCCL 2.27. [https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/.](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/.)Accessed: 2025-10-10

