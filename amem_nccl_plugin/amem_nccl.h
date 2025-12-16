// Licensed to the Asystem-Amem developers under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef AMEM_NCCL_H_
#define AMEM_NCCL_H_

#include <sys/types.h>
#include <unistd.h>

#include <unordered_map>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "gmm_cuda_common.h"

#define AMEM_MAX_DEVS (8)
#define AMEM_MAX_CALLER (16)

// NOTE: set as FABRIC in sm_100a
#define AMEM_DEFAULT_HANDLE_TYPE CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

// State for each memory alloc
typedef enum {
  AMEM_STATE_INVALID = 0,

  AMEM_STATE_ALLOC,
  AMEM_STATE_OFFLOADING,
  AMEM_STATE_HOLE,
  AMEM_STATE_PRELOADING,
  AMEM_STATE_REMAPPED,

  AMEM_STATE_MAX,
} amem_allocState;

// Type of each alloc
typedef enum {
  AMEM_TYPE_CUMEM_LOCAL        = 0,
  AMEM_TYPE_CUMEM_LOCAL_POSIX  = 0,

  AMEM_TYPE_CUMEM_LOCAL_FABRIC = 1,
  AMEM_TYPE_CUMEM_LOCAL_SYMM   = 2,

  AMEM_TYPE_CUMEM_PEER_POSIX,
  AMEM_TYPE_CUMEM_PEER_FABRIC,
  AMEM_TYPE_CUMEM_PEER_SYMM,

  AMEM_TYPE_CUMEM_MAX,
} amem_allocType;

// Caller type for memory allocation tracking (mapped from NCCL internal usage)
typedef enum {
  AMEM_CALLER_NCCL_DEFAULT = 0,        // NCCL default/generic allocation
  AMEM_CALLER_NCCL_P2P = 1,            // NCCL P2P allocation
  AMEM_CALLER_NCCL_FABRIC = 2,         // NCCL fabric handle allocation
  AMEM_CALLER_NCCL_POSIX = 3,          // NCCL posix handle allocation
  AMEM_CALLER_NCCL_SYMM = 4,           // NCCL symmetric memory allocation
  AMEM_CALLER_NCCL_NVLS = 5,           // NCCL NVLS memory allocation
  AMEM_CALLER_NCCL_RESERVED = 6,       // Reserved
  AMEM_CALLER_NCCL_P2P_PEER = 7,       // NCCL P2P peer memory
  AMEM_CALLER_NCCL_RESERVED2 = 8,      // Reserved
  AMEM_CALLER_NCCL_PROXY = 9,          // NCCL proxy registration
  AMEM_CALLER_NCCL_SYMM_PEER = 10,     // NCCL symmetric peer memory
  AMEM_CALLER_MAX = AMEM_MAX_CALLER
} amem_caller_type;

static inline int amem_fromPeer(amem_allocType type) { 
  return (type <= AMEM_TYPE_CUMEM_LOCAL_SYMM) ? 0 : 1;
}

// Metadata for each alloc
typedef struct amem_allocMdata_t {
  int dev;
  pid_t pid;
  int refcount;

  int sharedFD; // local handle exported
  CUmemGenericAllocationHandle handle;
  CUmemGenericAllocationHandle mcHandle; // for symmem in NVLS etc
  CUevent evt;

  size_t allocSz;
  size_t tags;

  amem_allocType  type;
  amem_allocState state;

  // if dptr is mapped from peer, then peers[] is peerDev's dptr
  CUdeviceptr     peers[AMEM_MAX_DEVS];
  bool            hasPeer; //true when it's imported by peer
  bool            oosFlag; 

  void*           cpuAddr;
  void*           userInfo; 

  amem_allocMdata_t(int in_dev, size_t in_allocSz, 
		  amem_allocType in_type, 
		  amem_allocState in_state, 
		  CUmemGenericAllocationHandle in_handle = 0ULL,
		  void *in_ptr = NULL, 
		  size_t in_tags = 0,
		  bool in_oosFlag=false,
		  CUmemGenericAllocationHandle in_mcHandle = 0ULL) {
    dev     = in_dev;
    pid     = getpid();
    allocSz = in_allocSz;
    state   = in_state;
    type    = in_type;
    userInfo= in_ptr;
    tags    = in_tags;
    refcount= 0;
    sharedFD= 0;
    handle = in_handle;
    mcHandle=in_mcHandle;
    cpuAddr = NULL;
    hasPeer = false;
    oosFlag = in_oosFlag;
    // Init peer's dptr as 0
    for (int i = 0; i< AMEM_MAX_DEVS; ++i) peers[i] = 0;
    CHECK_DRV(cuEventCreate(&evt, CU_EVENT_DISABLE_TIMING));
  }

} amem_allocMdata;

// Release the phy handle, the last reference release would free the handle
static inline CUresult amem_cuMemReleaseHandle(void *ptr, int refcount) {
  CUresult curet = CUDA_SUCCESS;
  if (ptr == NULL) return curet;

  size_t size = 0;
  CUmemGenericAllocationHandle  handle;
  CHECK_DRV(cuMemRetainAllocationHandle(&handle, ptr));
  CHECK_DRV(cuMemRelease(handle));

  CHECK_DRV(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  CHECK_DRV(cuMemUnmap((CUdeviceptr)ptr, size));
  CHECK_DRV(cuMemRelease(handle));

  if (refcount > 0) {
    while (refcount--) {
      CHECK_DRV(cuMemRelease(handle));
    }
  }

  return curet;
}

// Alloc phy handle, export it when needed
static CUresult amem_cuAllocAndExport(CUdeviceptr dptr, int curDev, size_t size, CUmemGenericAllocationHandle *newHandle, int &sharedFD, bool hasPeer)
{
  CUresult curet = CUDA_SUCCESS;

  CUmemGenericAllocationHandle handle;
  CUmemAllocationProp mprop = {};
  mprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  mprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  mprop.location.id = curDev;
  mprop.requestedHandleTypes = AMEM_DEFAULT_HANDLE_TYPE;
  // mprop.allocFlags.gpuDirectRDMACapable = 1;

  curet = CHECK_DRV(cuMemCreate(&handle, size, &mprop, 0));
  curet = CHECK_DRV(cuMemMap(dptr, size, 0, handle, 0));
  CUmemAccessDesc access;
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = curDev;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  curet = CHECK_DRV(cuMemSetAccess(dptr, size, &access, 1));

  *newHandle = handle;

  // export the handle
  if (hasPeer) {
    curet = CHECK_DRV(cuMemExportToShareableHandle((void *)&sharedFD, handle, mprop.requestedHandleTypes, 0));
    // note, the handle may export multiple times, with different FDs
  }

  return curet;
}

// Map the handle with dptr
static inline CUresult amem_mapAndAccess(int curDev, CUdeviceptr ptr, CUmemGenericAllocationHandle handle, size_t size) {
  CUresult curet = CUDA_SUCCESS;
  if (ptr == 0) return curet;

  curet = CHECK_DRV(cuMemMap(ptr, size, 0, handle, 0));
  if (curet == 0) {
    CUmemAccessDesc access;
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = curDev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    curet = CHECK_DRV(cuMemSetAccess(ptr, size, &access, 1));
    //printf("AMEM pid:%d func:%s %d newAcces sret:%d\n", getpid(),__FUNCTION__,  __LINE__, curet);fflush(stdout);
  } else {
    LOGGER(ERROR, "pid:%d func:%s %d newMap ret:%d", getpid(),__FUNCTION__,  __LINE__, curet);
  }
  
  return curet;
}

// Re-import the handle from FD, map and setAccess 
static inline CUresult amem_importAndMap(int curDev, CUdeviceptr dptr, size_t size, int sharedFD)
{
  CUresult curet = CUDA_SUCCESS;
  CUmemGenericAllocationHandle handle;

  // Import to get the handle
  curet = CHECK_DRV(cuMemImportFromShareableHandle(&handle, (void *)(uintptr_t)sharedFD, AMEM_DEFAULT_HANDLE_TYPE));
  if (curet == CUDA_SUCCESS) {
    curet = amem_mapAndAccess(curDev, dptr, handle, size); 
  }
  close(sharedFD);

  if (curet != CUDA_SUCCESS) {
    LOGGER(ERROR, "pid:%d func:%s %d curDev:%d newHandle:%llx inFD:%d ret:%d", getpid(),__FUNCTION__, 
             __LINE__, curDev, handle, sharedFD, curet);
  }

  return curet;
}

// Alloc host pinned memory, invoked for the first time of offloading
static inline CUresult amem_allocHost(void **ptr, size_t size)
{
  CUresult curet = CUDA_SUCCESS;

  size_t granularity = 0;
  CUdevice currentDev;
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle;
  int cudaDev;
  int cpuNumaNodeId = -1;

  (cudaGetDevice(&cudaDev));
  CHECK_DRV(cuDeviceGet(&currentDev, cudaDev));
  CHECK_DRV(cuDeviceGetAttribute(&cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));
  if (cpuNumaNodeId < 0) cpuNumaNodeId = 0;

  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = AMEM_DEFAULT_HANDLE_TYPE;
  prop.location.id = cpuNumaNodeId;
  if ((cuMemCreate(&handle, size, &prop, 0) != CUDA_SUCCESS)) {
    prop.location.id = 0;
    // NB, suppose the GPU is on NUMA 1, but cuMemCreate always failed. don't know why.
    if ((cuMemCreate(&handle, size, &prop, 0) != CUDA_SUCCESS)) {
      LOGGER(ERROR, "pid:%d GPU:%d numa:%d failed, force set to numa 0 failed", getpid(), cudaDev, cpuNumaNodeId);
      ASSERT(0, "HostAlloc failed on NUMA 0");
    } else {
      LOGGER(DEBUG, "pid:%d GPU:%d numa:%d failed, have to force set to numa 0", getpid(), cudaDev, cpuNumaNodeId);
    }
  }

  CHECK_DRV(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
  CHECK_DRV(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CHECK_DRV(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  /* Now allow RW access to the newly mapped memory from the CPU */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  accessDesc.location.id = cpuNumaNodeId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CHECK_DRV(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  return curet;
}

// Free host pinned memory
static inline CUresult amem_freeHost(void* ptr) 
{
  CUresult curet = CUDA_SUCCESS;
  if (ptr == nullptr) return curet;

  CUmemGenericAllocationHandle handle;
  size_t size = 0;
  CHECK_DRV(cuMemRetainAllocationHandle(&handle, ptr));
  CHECK_DRV(cuMemRelease(handle));
  CHECK_DRV(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  CHECK_DRV(cuMemUnmap((CUdeviceptr)ptr, size));
  CHECK_DRV(cuMemRelease(handle));
  CHECK_DRV(cuMemAddressFree((CUdeviceptr)ptr, size));

  return curet;
}

// Offload the content to cpu
static inline CUresult amem_launchOffload(void *cpuAddr, CUdeviceptr dptr, size_t size, CUstream stream, CUevent evt, bool async)
{
  CUresult curet = CUDA_SUCCESS;
  curet = CHECK_DRV(cuMemcpyDtoHAsync(cpuAddr, dptr, size, stream));
  curet = CHECK_DRV(cuEventRecord(evt, stream));
  if (!async) {
    curet = CHECK_DRV(cuEventSynchronize(evt));
  }
  return curet;
}

// Preload the content from cpu
static inline CUresult amem_lauchPreload(CUdeviceptr dptr, void *cpuAddr, size_t size, CUstream stream, CUevent evt, bool async)
{
  CUresult curet = CUDA_SUCCESS;
  curet = CHECK_DRV(cuMemcpyHtoDAsync(dptr, cpuAddr, size, stream));
  curet = CHECK_DRV(cuEventRecord(evt, stream));

  if (!async) {
    curet = CHECK_DRV(cuEventSynchronize(evt));
  }

  return curet;
}

// Add peer dptr info which mapped on the same srcHandle. e.g., when srcHandle is mapped to peer
int amem_registerPeerInfo(CUmemGenericAllocationHandle srcHandle, CUdeviceptr peerPtr, int peerDev);

// Register a peer info for srcPtr
int amem_addPeerInfo(CUdeviceptr srcPtr, CUdeviceptr peerPtr, int peerDev);

// Re-import from the sharedFD to get new handle, then map to dptr e.g., when src GPU handle is re-allocated/exported
int amem_updatePeerInfo(CUdeviceptr dptr, int sharedFD);

//////////// for external usage
// Add metadata info, whenever GPU mem get allocated/mapped either local all or map remote
// 1) local alloc: peerDev set to -1, peerHandle as 0
// 2) map peer   : peerDev set, peerHandle set
// userInfo: optional, a pointer for the caller ctx, such as ncclComm
// caller_type: indicates which component calls this function
int amem_addAllocInfo(CUdeviceptr localDptr, size_t allocSz, int type, int localDev, 
		      CUmemGenericAllocationHandle localHandle, int peerDev = -1, uint64_t peerHandle = 0ULL, 
		      void *userInfo = NULL, amem_caller_type caller_type = AMEM_CALLER_NCCL_DEFAULT);
// Clean up metadata info before dptr free
int amem_delAllocInfo(CUdeviceptr dptr, CUmemGenericAllocationHandle  handle, int caller = -1);

// Pause: do necessary offload, then release physical GPU memory (local and mapped from peer) but keep dptr unchanged
int amem_memPause(pid_t pid, uint64_t tag = 0);
// Resume: alloc new physical GPU mem, do necessary preload, notify peer to map the new allocated mem
// Note: pause and resume shall be invoked in-pair. both are blocked until release/offload/preload run to complete
int amem_memResume(pid_t pid, uint64_t tag = 0);

void amem_dumpAllocStats(bool verbose = true);

// Check pause status, show warning if necessary
bool amem_checkPaused(bool warn = true);

// Add additional refcount
int amem_addRefcount(void *dptr, int refcount);

// User may explicitly set the groupID *before* the first NCCL mem allocation
// after that, setting groupID is rejected
int amem_setGroupID(int id);
void amem_getGroupID(int* id);

#endif
