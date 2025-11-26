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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "amem_nccl.h"
#include "gmm_client.h"
#include "gmm_client_cfg.h"
#include "gmm_common.h"
#include "gmm_util.h"

static gmm_client_cfg *g_cfg;
static int amem_plugin_disable;
static int amem_offload_free_tag;
static gmm_client_ctx *gmm_ctx_p;
static std::mutex gmm_lock;
static std::atomic<bool> init_topo;

void *gmm_libP = NULL;
int amem_groupID = 0;

// by default, offload all NCCL allocated GPU memory. 
// however P2P buffer (currently it's tag as 7) could be offload-free w/o breaking functions
#define AMEM_OFFLOAD_FREE_TAG (-1)

// by default, disable AMEM pause/resume
#define AMEM_DISABLE_DEFAULT (1)

static void msleep(unsigned int time_msec) {
  const long c_1e6 = 1e6;
  struct timespec tv = (struct timespec){
      .tv_sec = time_msec / 1000,
      .tv_nsec = (time_msec % 1000) * c_1e6,
  };
  nanosleep(&tv, NULL);
}

// Init a client ctx
static void InitClientCtx(int curDev) {
  if (!init_topo) {
    init_topo = true;

    std::lock_guard<std::mutex> lock_(gmm_lock);
    gmm_client_cfg_init(gmm_libP, g_cfg);
    if (!gmm_ctx_p) {  // defer init until first mem alloc
      gmm_ctx_p = new gmm_client_ctx(g_cfg);

      CHECK_DRV(cuStreamCreate(&gmm_ctx_p->offloadStream[curDev], CU_STREAM_NON_BLOCKING));
      CHECK_DRV(cuStreamCreate(&gmm_ctx_p->preloadStream[curDev], CU_STREAM_NON_BLOCKING));
      gmm_ctx_p->releaseShadowCnt = gmm_ctx_p->releaseLocalCnt = gmm_ctx_p->offloadCnt = gmm_ctx_p->smokeLog = 0;
      gmm_ctx_p->pauseCnt = gmm_ctx_p->resumeCnt = 0;
      for (int i = 0; i < AMEM_MAX_CALLER; ++i) {
        gmm_ctx_p->allocBytes[i] = 0;
      }
      gmm_ctx_p->delBytes = 0;
      gmm_ctx_p->paused = false;
    }
  }

}

void amem_dumpAllocStats(bool verbose)
{
  if (amem_plugin_disable > 0) {
    return;
  }

  size_t total = 0;
  for (int i = 0; i < AMEM_MAX_CALLER; ++i) {
    if (gmm_ctx_p->allocBytes[i] != 0 && verbose) {
      LOGGER(INFO, "groupID:%d pid:%d caller_%d allocBytes:%ld", amem_groupID, getpid(), i, gmm_ctx_p->allocBytes[i]);
    }
    total += gmm_ctx_p->allocBytes[i];
  }
  total += gmm_ctx_p->delBytes; 
  LOGGER(INFO, "groupID:%d pid:%d total GPU alloc:%ld (%ld MB), "
               "note including those mapped from peer, and defensive skiped %ld MB", amem_groupID, getpid(), 
               total, total >> 20, gmm_ctx_p->delBytes >> 20);
}

// Add metadata info when memory get allocated/mapped either local all or map remote
int amem_addAllocInfo(CUdeviceptr localDptr, size_t allocSz, int type, int localDev, 
		      CUmemGenericAllocationHandle localHandle,
                      int peerDev, uint64_t peerHandle, void *userInfo, amem_caller_type caller_type)
{
  if (amem_plugin_disable > 0) {
    return 0;
  }

  int ret = 0;
  int curDev = localDev;
  int peerFlag = amem_fromPeer((amem_allocType)type);
  int srcDevDetected = -1;

  // A special case: GPU1 dptr's handle is fromPeer, but may have two sources:
  //     from GPUx.nccl (hasPeer is set in mdata). will be paused/resume.
  //     from GPUy.non-nccl (so not tracked by our mdata). out of scope
  bool oosFlag = (peerFlag && peerDev == -1) ? true : false;
  InitClientCtx(curDev);
  
  {
    std::lock_guard<std::mutex> lock_(gmm_ctx_p->pause_mtx);
    if (gmm_ctx_p->paused) {
      LOGGER(ERROR, "groupID:%d pid:%d NCCL is already paused! "
                    "DO NOT issue any NCCL op including memory alloc!", amem_groupID, getpid());
      return 1;
    }

    std::unordered_map<CUdeviceptr, amem_allocMdata>* tablePtr = &gmm_ctx_p->allocTable[localDev];
    auto it = tablePtr->find(localDptr);
    // Add metadata entry for the first time localDptr get created
    if (it == tablePtr->end()) {
      tablePtr->emplace(std::pair<CUdeviceptr, amem_allocMdata>(localDptr,
             amem_allocMdata(curDev, allocSz, (amem_allocType)type, AMEM_STATE_ALLOC, localHandle, userInfo, (size_t)caller_type, oosFlag, peerHandle)));
      if ((int)caller_type < AMEM_MAX_CALLER) {
        gmm_ctx_p->allocBytes[(int)caller_type] += allocSz;
      }

      if (peerFlag) {
        // detect the src cudaDevID from localHandle by querying prop!!
        CUmemAllocationProp prop = {};
        CUresult curet = (cuMemGetAllocationPropertiesFromHandle(&prop, localHandle));
        srcDevDetected = prop.location.id;

        // additional warning if detected src cudaDev is diff than input
        if (peerDev >= 0 && (peerDev != srcDevDetected)) {
          LOGGER(WARN, "groupID:%d pid:%d allocSz:%ld curDev:%d peer:%d detected:%d are different! "
                        "remote dptr:%llx localHandle:%llx rmtHandle:%lx caller:%d",
                       amem_groupID, getpid(), allocSz, curDev, peerDev, srcDevDetected,
                       localDptr, localHandle, peerHandle, (int)caller_type);
        }
      }

      // debug purpose logs
      if (gmm_ctx_p->smokeLog == 0) {
        // always print only for the first successful add
        LOGGER(INFO, "groupID:%d pid:%d allocSz:%ld curDev:%d peer:%d (detected:%d) "
                     "dptr:%llx type:%s localHandle:%llx rmtHandle:%lx caller:%d", amem_groupID,
                      getpid(), allocSz, curDev, peerDev, srcDevDetected, localDptr,
                      (peerFlag == 0) ?"local":"remote", localHandle, peerHandle, (int)caller_type);
      } else {
        // then downgrade as DEBUG level
        LOGGER(DEBUG, "groupID:%d pid:%d allocSz:%ld curDev:%d peer:%d (detected:%d) "
                      "dptr:%llx type:%s localHandle:%llx rmtHandle:%lx caller:%d", amem_groupID,
                      getpid(), allocSz, curDev, peerDev, srcDevDetected, localDptr,
                      (peerFlag == 0) ?"local":"remote", localHandle, peerHandle, (int)caller_type);
      }
    } else {
      // metdata already exist!?
      LOGGER(WARN, "groupID:%d pid:%d allocSz:%ld curDev:%d peer:%d (detected:%d) "
                   "dptr:%llx type:%s localHandle:%llx rmtHandle:%lx caller:%d already exists!", amem_groupID,
                   getpid(), allocSz, curDev, peerDev, srcDevDetected, localDptr,
                   (peerFlag == 0) ?"local":"remote", localHandle, peerHandle, (int)caller_type);
      it->second.type     = (amem_allocType) type;
      it->second.state    = AMEM_STATE_ALLOC;
      it->second.allocSz  = allocSz;
      it->second.handle   = localHandle;
      it->second.userInfo = userInfo;
      it->second.tags     = (size_t)caller_type;
    }
  }

  // Then notify peer (src dev) to registerPeerInfo
  if (peerFlag && peerDev >= 0) {
    // Case1: peerDev is input by NCCL hook
    gmm_shmInfo_t *msgPtr = new gmm_shmInfo_t(GMM_IPC_MEM_NV_DEV_SHARE, curDev, (void *)localDptr, localHandle, allocSz, -1);
    // example: current is GPU1, map a physical handle from GPU0
    //   curDev is  GPU1, handle is 'localHandle', mapped to 'localDptr'
    //   peerDev is GPU0, handle is 'peerHandle'
    // NOTE: src dev has no such mem info, remove its medatadata, so it's out of pause/resume scope
    ret = gmm_ctx_p->register_peer(msgPtr, peerHandle, localDptr, curDev, peerDev);
    delete msgPtr;

    if (ret != 0) {
      LOGGER(WARN, "groupID:%d pid:%d allocSz:%ld curDev:%d dptr:%llx "
                   "localHandle:%llx rmtHandle:%lx register peer:%d failed, remove metadata", amem_groupID,
                   getpid(), allocSz, curDev, localDptr, localHandle, peerHandle, peerDev);
      if (gmm_ctx_p->smokeLog == 0) { gtrace(); }

      std::unordered_map<CUdeviceptr, amem_allocMdata>* tablePtr = &gmm_ctx_p->allocTable[localDev];
      auto it = tablePtr->find(localDptr);
      if (it != tablePtr->end()) {
        tablePtr->erase(localDptr);
      }
    }

  } else if (peerFlag && peerDev == -1) {
    // Case2: input peerDev is -1 now we use the detected cudaDev
    if (srcDevDetected >= 0) {
      gmm_shmInfo_t *msgPtr = new gmm_shmInfo_t(GMM_IPC_MEM_NV_DEV_SHARE, curDev, (void *)localDptr, localHandle, allocSz, -1);
      ret = gmm_ctx_p->register_peer(msgPtr, peerHandle, localDptr, curDev, srcDevDetected);
      delete msgPtr;

      if (ret != 0) {
        LOGGER(WARN, "groupID:%d pid:%d allocSz:%ld curDev:%d dptr:%llx "
                     "localHandle:%llx rmtHandle:%lx register peer:%d failed, remove metadata", amem_groupID,
                     getpid(), allocSz, curDev, localDptr, localHandle, peerHandle, srcDevDetected);
        std::unordered_map<CUdeviceptr, amem_allocMdata>* tablePtr = &gmm_ctx_p->allocTable[localDev];
        auto it = tablePtr->find(localDptr);
        if (it != tablePtr->end()) {
          tablePtr->erase(localDptr);
        }
      }

    } else {
      // Case3: error! as the src dev may unmap -> remap new handle after resume which cause inconsistent
      LOGGER(ERROR, "groupID:%d pid:%d allocSz:%ld curDev:%d dptr:%llx localHandle:%llx rmtHandle:%lx "
                    "peer:%d info is invalid!! may cause inconsistent or hang", amem_groupID,
                    getpid(), allocSz, curDev, localDptr, localHandle, peerHandle, peerDev);
      std::unordered_map<CUdeviceptr, amem_allocMdata>* tablePtr = &gmm_ctx_p->allocTable[localDev];
      auto it = tablePtr->find(localDptr);
      if (it != tablePtr->end()) {
        tablePtr->erase(localDptr);
      }
      ret = 1;
    }

  } else if(!peerFlag) {
    // Case4: local accessed memory, add new entry
    gmm_ctx_p->handleTable[curDev].emplace(std::pair<CUmemGenericAllocationHandle, CUdeviceptr>(localHandle, localDptr));
  }

  gmm_ctx_p->smokeLog++;
  return ret;
}

// Update dptr with new handle, e.g., when resumed
inline int amem_updateAllocInfo(CUdeviceptr dptr, int localDev, int type, CUmemGenericAllocationHandle newHandle, int fd)
{
  int ret = 0;

  std::unordered_map<CUdeviceptr, amem_allocMdata>* tablePtr = &gmm_ctx_p->allocTable[localDev];
  auto it = tablePtr->find(dptr);
  
  if (it != tablePtr->end()) {
    it->second.handle   = newHandle;
    it->second.state = AMEM_STATE_ALLOC;
  } else {
    LOGGER(ERROR, "groupID:%d pid:%d failed to find info for dptr:%llx newHandle:%llx", amem_groupID, getpid(),
                  dptr, newHandle);
    ret = 1;
  }

  return ret;
}

// Add peer dptr which mapped on the same srcHandle. e.g., when srcHandle is mapped to peer
// it's from my curDev point of view: my handle 'srcHandle' is mapped to peer 'peerDev' at peer addr 'peerPtr'
// now lookup my handleTable by key 'srcHandle' to get my ptr 'dptr', 
// then at dptr's metadata peer[], to update the table at index 'peerDev' with value 'peerPtr'
int amem_registerPeerInfo(CUmemGenericAllocationHandle srcHandle, CUdeviceptr peerPtr, int peerDev)
{
  int ret = 1;
  int curDev = 0;
  cudaGetDevice(&curDev);

  //std::lock_guard<std::mutex> lock_(gmm_ctx_p->pause_mtx);

  std::unordered_map<CUmemGenericAllocationHandle, CUdeviceptr>* tablePtr = &gmm_ctx_p->handleTable[curDev];
  auto it = tablePtr->find(srcHandle);
 
  // find by handle to get dptr 
  if (it != tablePtr->end()) {
    std::unordered_map<CUdeviceptr, amem_allocMdata>* ptrTable = &gmm_ctx_p->allocTable[curDev];
    auto it_ = ptrTable->find(it->second);
    if (it_ != ptrTable->end()) {
      it_->second.peers[peerDev] = peerPtr;
      it_->second.hasPeer = true;
      ret = 0;
      LOGGER(DEBUG, "groupID:%d pid:%d localDptr:%llx localHandle:%llx register peer:%d peerPtr:%llx done", amem_groupID, getpid(),
                 it->second, srcHandle, peerDev, peerPtr);
    } else {
      LOGGER(ERROR, "groupID:%d pid:%d failed to find info for handle:%llx dptr:%llx to register peer:%d peerPtr:%llx", amem_groupID, getpid(),
                  srcHandle, it->second, peerDev, peerPtr);
    }

  } else {
    LOGGER(ERROR, "groupID:%d pid:%d failed to find info for handle:%llx to register peer:%d peerPtr:%llx", amem_groupID, getpid(),
                  srcHandle, peerDev, peerPtr);
  }

  return ret;
}

// Add additional refcount
int amem_addRefcount(void *dptr, int refcount)
{
  if (amem_plugin_disable > 0) {
    return 0;
  }

  int ret = 1;
  int curDev = 0;
  cudaGetDevice(&curDev);

  // std::lock_guard<std::mutex> lock_(gmm_ctx_p->pause_mtx);
  std::unordered_map<CUdeviceptr, amem_allocMdata>* ptrTable = &gmm_ctx_p->allocTable[curDev];
  auto it_ = ptrTable->find((CUdeviceptr)dptr);
  if (it_ != ptrTable->end()) {
    it_->second.refcount += refcount;
    ret = 0;
    LOGGER(DEBUG, "groupID:%d pid:%d dptr:%p add refcount:%d done", amem_groupID, getpid(),
               dptr, refcount);
  } else {
    LOGGER(ERROR, "groupID:%d pid:%d failed to find metadata for dptr:%p when adding refcount:%d", amem_groupID, getpid(),
               dptr, refcount);
  }

  return ret;
}

// Register a peer info for srcPtr
int amem_addPeerInfo(CUdeviceptr srcPtr, CUdeviceptr peerPtr, int peerDev)
{
  int ret = 1;
  int curDev = 0;
  cudaGetDevice(&curDev);

  //std::lock_guard<std::mutex> lock_(gmm_ctx_p->pause_mtx);

  std::unordered_map<CUdeviceptr, amem_allocMdata>* ptrTable = &gmm_ctx_p->allocTable[curDev];
  auto it_ = ptrTable->find(srcPtr);
  if (it_ != ptrTable->end()) {
    it_->second.peers[peerDev] = peerPtr;
    it_->second.hasPeer = true;
    ret = 0;
    LOGGER(DEBUG, "groupID:%d pid:%d localDptr:%llx register a peer:%d peerPtr:%llx done", amem_groupID, getpid(),
               srcPtr, peerDev, peerPtr);
  } else {
    LOGGER(ERROR, "groupID:%d pid:%d failed to find for localDptr:%llx to register peer:%d peerPtr:%llx", amem_groupID, getpid(),
                  srcPtr, peerDev, peerPtr);
  }

  return ret;
}

// Re-import from the sharedFD to get new handle, then map to 'dptr'
// invoked when src GPU handle is newly allocated/exported during resume
int amem_updatePeerInfo(CUdeviceptr dptr, int sharedFD)
{
  int ret = 0;
  int curDev = 0;
  cudaGetDevice(&curDev);

  //std::lock_guard<std::mutex> lock_(gmm_ctx_p->pause_mtx);

  std::unordered_map<CUdeviceptr, amem_allocMdata>* tablePtr = &gmm_ctx_p->allocTable[curDev];
  auto it = tablePtr->find(dptr);
  
  if (it != tablePtr->end()) {
    ret = amem_importAndMap(curDev, it->first, it->second.allocSz, sharedFD);
    if (ret == 0) {
      it->second.state = AMEM_STATE_ALLOC; 
      gmm_ctx_p->releaseShadowCnt--;
      LOGGER(DEBUG, "groupID:%d pid:%d curDev:%d update ptr:%llx fd:%d done", amem_groupID, getpid(), curDev, dptr, sharedFD);
    }
  } else {
    LOGGER(ERROR, "groupID:%d pid:%d failed to find info for dptr:%llx to remap fd:%x", amem_groupID, getpid(), dptr, sharedFD);
    ret = 1;
  }

  return ret;
}

// Clean up metadata info before dptr free
// it's caller's responsiblity to unmap peer etc
// TODO: update as INVALID state rather than removing it
int amem_delAllocInfo(CUdeviceptr dptr, CUmemGenericAllocationHandle  handle, int caller) 
{
  CUresult curet = CUDA_SUCCESS;
  if (amem_plugin_disable > 0) {
    return curet;
  }

  if (dptr == 0) return curet;
  
  if (handle == 0) {
    curet = CHECK_DRV(cuMemRetainAllocationHandle(&handle, (void *)dptr));
    curet = CHECK_DRV(cuMemRelease(handle));
  }

  int curDev = 0;
  cudaGetDevice(&curDev);

  std::lock_guard<std::mutex> lock_(gmm_ctx_p->pause_mtx);

  std::unordered_map<CUdeviceptr, amem_allocMdata>* tablePtr = &gmm_ctx_p->allocTable[curDev];
  auto it = tablePtr->find(dptr);
  if (it != tablePtr->end()) {
    if (it->second.tags < AMEM_MAX_CALLER) {
      gmm_ctx_p->allocBytes[caller] -= it->second.allocSz; 
    } else {
      gmm_ctx_p->delBytes += it->second.allocSz; 
    }
    tablePtr->erase(dptr);
    LOGGER(DEBUG, "groupID:%d pid:%d curDev:%d erase dptr:%llx done", amem_groupID, getpid(), curDev, dptr);
  }

  gmm_ctx_p->handleTable[curDev].erase(handle);

  return (int)curet;
}


// Check pause status, show warning if necessary
bool amem_checkPaused(bool warn)
{ 
  if (amem_plugin_disable > 0) {
    return false;
  }

  std::lock_guard<std::mutex> lock_(gmm_ctx_p->pause_mtx);
  if (warn && gmm_ctx_p->paused) {
    LOGGER(ERROR, "groupID:%d pid:%d NCCL of current process is already paused! DO NOT issue any NCCL op!", amem_groupID, getpid()); 
    gtrace();
  }
  return gmm_ctx_p->paused;
}

// Handler for mem pause
int amem_memPause(pid_t pid, uint64_t tag)
{
  if (amem_plugin_disable > 0) {
    return 0;
  }

  int ret = 0;
  CUresult curet = CUDA_SUCCESS;
  size_t releaseSz = 0;
  int curDev = 0;
  amem_offload_free_tag = getenv("AMEM_NCCL_OFFLOAD_FREE_TAG") ? atoi(getenv("AMEM_NCCL_OFFLOAD_FREE_TAG")) : AMEM_OFFLOAD_FREE_TAG;

  cudaGetDevice(&curDev); 
  { 
    std::lock_guard<std::mutex> lock_(gmm_ctx_p->pause_mtx);
    if (gmm_ctx_p->paused) {
      return ret;
    } else {
      gmm_ctx_p->paused = true;
    }
  }

  auto t0  = std::chrono::steady_clock::now();
  if (!gmm_ctx_p->allocTable[curDev].empty()) {
    auto it = gmm_ctx_p->allocTable[curDev].begin();
    int thisDev = it->second.dev;
    CHECK_CUDA(cudaDeviceSynchronize());

    size_t free_sz, total;
    cudaMemGetInfo(&free_sz, &total);
    LOGGER(INFO, "groupID:%d pid:%d GPU:%d memUsed: %ld MB before paused", amem_groupID, getpid(), curDev, (total-free_sz)>>20); 

    // 1. Issue offload to cpu in background stream, for local memory only
    CUmemGenericAllocationHandle oldHandle;
    for (auto& [key, value]: gmm_ctx_p->allocTable[curDev]) {
      if ((value.state == AMEM_STATE_ALLOC) && !amem_fromPeer(value.type) && value.tags != AMEM_OFFLOAD_FREE_TAG) {
        value.state = AMEM_STATE_OFFLOADING;
        if(value.cpuAddr == NULL) { // note, allocHost may increase a bit GPU mem (~2MB)
          amem_allocHost(&value.cpuAddr, value.allocSz);
	}
        amem_lauchOffload(value.cpuAddr, key, value.allocSz, gmm_ctx_p->offloadStream[curDev], value.evt, false);
        gmm_ctx_p->offloadCnt++;
      }
    }

    // 2. Release handle if it's from peer, so that peer could continue handle release
    // TODO: can we query the handle is from peer or local ???? instead of my mdata
    for (auto& [key, value]: gmm_ctx_p->allocTable[curDev]) {
      //if ((value.state == AMEM_STATE_ALLOC) && amem_fromPeer(value.type) && !value.oosFlag) {
      if ((value.state == AMEM_STATE_ALLOC) && amem_fromPeer(value.type)) {
        curet = amem_cuMemReleaseHandle((void*)key, 0);
        // no need to offload as data is owned by peer
        value.state = AMEM_STATE_HOLE;
        gmm_ctx_p->releaseShadowCnt++;
      }
    }

    // 3. Release local mem handles 
    for (auto& [key, value]: gmm_ctx_p->allocTable[curDev]) {
      // sync offloading if issued, so ensure data is backed up before final release
      if (!amem_fromPeer(value.type)) {
        if (value.state == AMEM_STATE_OFFLOADING) {
          CHECK_DRV(cuEventSynchronize(value.evt));
        }

        if (value.type == AMEM_TYPE_CUMEM_LOCAL_SYMM) {
          CHECK_DRV(cuMulticastUnbind(value.mcHandle, value.dev, 0, value.allocSz));
        }
        curet = CHECK_DRV(cuMemRetainAllocationHandle(&oldHandle, (void *)key));
        curet = CHECK_DRV(cuMemRelease(oldHandle));

        if (value.refcount > 0) {
          LOGGER(DEBUG, "groupID:%d pid:%d dptr:%llx refcount:%d", amem_groupID, getpid(), key, value.refcount);
	}

        curet = amem_cuMemReleaseHandle((void*)key, value.refcount);
        for (int i = 0; i < AMEM_MAX_DEVS; ++i) {
          if (value.peers[i] != 0) 
            gmm_ctx_p->releaseLocalCnt++;
        } // notify peer update
	
        value.state = AMEM_STATE_HOLE;
        value.handle = 0;
        releaseSz += value.allocSz;
        // remove handle entry (only local)
        gmm_ctx_p->handleTable[curDev].erase(oldHandle);
      } // release local done
    } 

    cuStreamSynchronize(gmm_ctx_p->offloadStream[curDev]);
    gmm_ctx_p->pauseCnt++;
     
    auto t1  = std::chrono::steady_clock::now();
    auto duration =std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
    cudaMemGetInfo(&free_sz, &total);
    LOGGER(INFO, "groupID:%d pid:%d GPU:%d pauseCnt:%ld memUsed:%ld MB after paused, released:%ld MB, duration:%ld ms" 
		  " offload:%d releaseLocal:%d releasePeer:%d", amem_groupID, getpid(), 
		    thisDev, gmm_ctx_p->pauseCnt, (total-free_sz)>>20, releaseSz >> 20, duration/1000,
		  gmm_ctx_p->offloadCnt, gmm_ctx_p->releaseLocalCnt, gmm_ctx_p->releaseShadowCnt); 
    //amem_dumpAllocStats(true);
  }
    
  return ret;
}

// Handler for mem resume
int amem_memResume(pid_t pid, uint64_t tag)
{
  if (amem_plugin_disable > 0) {
    return 0;
  }

  int ret = 0;
  CUresult curet = CUDA_SUCCESS;
  int curDev = 0;
  amem_offload_free_tag = getenv("AMEM_NCCL_OFFLOAD_FREE_TAG") ? atoi(getenv("AMEM_NCCL_OFFLOAD_FREE_TAG")) : AMEM_OFFLOAD_FREE_TAG;

  {
    std::lock_guard<std::mutex> lock_(gmm_ctx_p->pause_mtx);
    if (!gmm_ctx_p->paused) {
      return ret;
    }
  }

  cudaGetDevice(&curDev);
  size_t exportSz = 0;
  size_t free_sz, total;
  cudaMemGetInfo(&free_sz, &total);
  LOGGER(INFO, "groupID:%d pid:%d GPU:%d memUsed:%ld MB before resumed", amem_groupID, getpid(), curDev, (total-free_sz)>>20);

  auto t0  = std::chrono::steady_clock::now();
  // re-alloc local handle and lauch preload async
  for (auto& [key, value]: gmm_ctx_p->allocTable[curDev]) {
    if (value.state == AMEM_STATE_HOLE && !amem_fromPeer(value.type)) {
      int sharedFD = 0;
      CUmemGenericAllocationHandle newHandle;

      // 1. Alloc new handle, export it optionally
      // Once handle is exported (no matter imported or not), reference is added
      // Query handle prop to check its actual physical source by cuMemGetAllocationPropertiesFromHandle
      // the fd must be closed once all peer registered, otherwise memory can't be freed
      amem_cuAllocAndExport(key, curDev, value.allocSz, &newHandle, sharedFD, value.hasPeer);
      if (value.type == AMEM_TYPE_CUMEM_LOCAL_SYMM) {
        CHECK_DRV(cuMulticastBindMem(value.mcHandle, 0, newHandle, 0, value.allocSz, 0));
      }
      if (value.hasPeer) exportSz += value.allocSz;

      // 2. Launch preload in background
      if (value.tags != AMEM_OFFLOAD_FREE_TAG) {
        value.state = AMEM_STATE_PRELOADING;
        amem_lauchPreload(key, value.cpuAddr, value.allocSz, gmm_ctx_p->preloadStream[curDev], value.evt, false);
        gmm_ctx_p->offloadCnt--;
      }

      // 3. Early update state, though preload may be running in background
      amem_updateAllocInfo(key, curDev, value.type, newHandle, sharedFD);
      value.state = AMEM_STATE_ALLOC;
      // Add new handle entry
      gmm_ctx_p->handleTable[curDev].emplace(std::pair<CUmemGenericAllocationHandle, CUdeviceptr>(newHandle, key));

      // 4. Notify peer about new handle's FD
      if(value.hasPeer) {
	for (int i = 0; i < AMEM_MAX_DEVS; ++i) {
          gmm_shmInfo_t *msgPtr = new gmm_shmInfo_t(GMM_IPC_MEM_NV_DEV_SHARE, curDev, (void *)key, newHandle, value.allocSz, sharedFD);
          // only notify the peer which peer[dev] has mapped src handle
          if (value.peers[i] != 0) {
            ret = gmm_ctx_p->update_peer(msgPtr, (CUdeviceptr)value.peers[i], sharedFD, i); 
            if (ret != 0) {
              LOGGER(ERROR, "groupID:%d pid:%d localDptr:%llx newHandle:%llx fd:%d notify peer %d to remap dptr:%llx ret:%d", amem_groupID, 
		            getpid(), key, newHandle, sharedFD, i, value.peers[i], ret);
            } else {
              gmm_ctx_p->releaseLocalCnt--;
            }
          }
          delete msgPtr;
	}
	close(sharedFD);
      }
    } // loop each hole local dptr
  } // loop each local dtpr

  // synch preload stream to ensure all date is back brfore return
  CHECK_DRV(cuStreamSynchronize(gmm_ctx_p->preloadStream[curDev]));

  // if run to here, meas offload and remap local memory complete
  // howerver the process can't control other workers' progress, may be faster or slower,
  // thus spin until those memory is done before we return 
  while (gmm_ctx_p->releaseShadowCnt > 0) { 
    LOGGER(INFO, "groupID:%d pid:%d GPU:%d peerLeft:%d, retry until 0", amem_groupID, getpid(), curDev, gmm_ctx_p->releaseShadowCnt);
    msleep(500);
  }

  gmm_ctx_p->paused = false;
  gmm_ctx_p->resumeCnt++;


  auto t1  = std::chrono::steady_clock::now();
  auto duration =std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
  cudaMemGetInfo(&free_sz, &total);
  LOGGER(INFO, "groupID:%d pid:%d GPU:%d resumeCnt:%ld memUsed:%ld MB after resumed, duration:%ld ms "
		  "offloadLeft:%d releaseLocalLeft:%d releasePeerLeft:%d exportSz:%ld", amem_groupID, getpid(), 
		  curDev, gmm_ctx_p->resumeCnt, (total-free_sz)>>20, duration/1000, 
		  gmm_ctx_p->offloadCnt, gmm_ctx_p->releaseLocalCnt, gmm_ctx_p->releaseShadowCnt, exportSz>>20); 
  //amem_dumpAllocStats(true);
  return ret; 
}

int amem_setGroupID(int id)
{
  int ret = 0;
  if (amem_plugin_disable > 0) {
    return ret;
  }

  if (!gmm_ctx_p) {
    LOGGER(INFO, "groupID:%d (old) pid:%d, set as new groupID:%d done", amem_groupID, getpid(), id);
    amem_groupID = id;
  } else {
    LOGGER(WARN, "groupID:%d pid:%d, trying to set to new groupID:%d, reject as to late", amem_groupID, getpid(), id);
    ret = 1;
  }
  return ret;
}

void amem_getGroupID(int *id)
{
  if (amem_plugin_disable > 0) {
    return;
  }

  *id = amem_groupID;
}

// Constructor when lib loaded
static void __attribute__((constructor)) x_init(void) {
  // clean up any previous left over socket file
  int result = system("rm -rf /dev/shm/gmm_config; rm -rf /tmp/gmm*");
  gmm_set_log_level();

  amem_plugin_disable = getenv("AMEM_ENABLE") ? !atoi(getenv("AMEM_ENABLE")) : AMEM_DISABLE_DEFAULT;
  amem_offload_free_tag = getenv("AMEM_NCCL_OFFLOAD_FREE_TAG") ? atoi(getenv("AMEM_NCCL_OFFLOAD_FREE_TAG")) : AMEM_OFFLOAD_FREE_TAG;
  amem_groupID = getenv("AMEM_GROUPID") ? atoi(getenv("AMEM_GROUPID")) : 0;

  int cuMemFlag = getenv("NCCL_CUMEM_ENABLE") ? atoi(getenv("NCCL_CUMEM_ENABLE")) : 1;
  if (cuMemFlag == 0) {
    amem_plugin_disable = 1;
  }

  LOGGER(INFO, "groupID:%d pid:%d build:%s %s NCCL plugin loaded. pause func:%s offload_free_tag:%d cuMemEnabled:%d", 
		  amem_groupID, getpid(), __DATE__, __TIME__, amem_plugin_disable?"Off":"On", 
		  amem_offload_free_tag, cuMemFlag);
}

// Destructor when lib unloaded
static void __attribute__((destructor)) x_fini(void) {
  gmm_client_cfg_destroy(gmm_libP);
  LOGGER(INFO, "groupID:%d pid:%d exit", amem_groupID, getpid());
}
