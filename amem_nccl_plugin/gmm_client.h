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

#pragma once

#include <cuda.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <atomic>
#include <set>
#include <unordered_map>

#include "gmm_client_cfg.h"
#include "gmm_common.h"
//#include "gmm_cuda_mem.h"
#include "gmm_host_mem.h"
#include "gmm_host_shm.h"
#include "amem_nccl.h"

struct gmm_req_t;
struct gmm_config_t;

struct gmm_lock_t {
  pthread_mutex_t mutex;
  pthread_mutexattr_t mutex_attr;
};

struct mp_status {
  std::atomic<bool> pending_mp;
};

struct shm_base_cmp {
  // desc order
  bool operator()(const void *left, const void *right) {
    return (left > right);
  }
};

enum mem_advise_t {
  MEM_ADVISE_NONE = 0,
  MEM_ADVISE_READONLY = 1, /* for data dedup */
};

struct shmDev_base_cmp {
  // desc order
  bool operator()(CUdeviceptr left, CUdeviceptr right) {
    return (left > right);
  }
};

// Global Mem Magt client: support global alloc, data moving (DM) with
// multi-path, global tiering one per client process
class gmm_client_ctx {
  uint64_t uuid;     // assigned by admin
  uint16_t slot_id;  // assigned by admin
  int dev_cnt;       // tot gpu num visible
  gmm_client_cfg *client_cfg;

  // GDR can offer much lower H2D latency when 1) size <=64KB, 2) host mem is
  // pinned
  int gdr_supported;

  size_t vmm_granularity;

  gmm_req_t *op_buf;  // to the shm area
  gmm_lock_t gmm_locks[MAX_DEV_NR];

  int admin_connect;                // cache the connect to gmm scheduler socket
  int worker_connects[MAX_DEV_NR];  // cache the connect to gmm worker socket

  CUevent mp_evt[MAX_DEV_NR][MAX_IPC_EVT_NR];

  gmm_config_t *config_shm;
  int config_fd;
  int sched_fd;
  pid_t pid;
  fifo_queue<gmm_req_t *> req_pool;

  CUmemAllocationProp props[MAX_DEV_NR];
  CUmemAccessDesc accessDescs[MAX_DEV_NR];

  struct flock f_lock;

  // CUstream mp_stream[MAX_DEV_NR][3]; // H2D/D2H/D2D streams


  std::mutex stream_map_lock;
  std::unordered_map<CUstream, bool> stream_map;
  std::unordered_map<CUevent, CUstream> evt_map;

  // TODO: merge and magt by single common set/map?
  // for host(pinned) mem allocation
  // std::set<void *, shm_base_cmp> hostMem_set; // set: desc ordered by dev
  // baseAddr
  std::map<void *, host_mem *> hostMem_map;  // map, key by dev baseAddr

  // for dev mem allocation
  // std::set<CUdeviceptr, shmDev_base_cmp> devMem_set; // set: desc ordered by
  // dev baseAddr
  //std::map<CUdeviceptr, CUDA_devMem *> devMem_map;  // map, key by dev baseAddr

  // for pinned host shm
  std::atomic<uint32_t> shm_idx;
  // TODO: if assuming unified addressing, could use same key for both host and
  // dev memory
  // std::set<void *, shm_base_cmp> shm_baseAddr_set; // set: desc ordered by
  // host baseAddr
  std::map<void *, gmm_shmInfo_t *> shm_table;  // map key by addr

  // for each IPC-share dev mem, multi-gpu, large allocation
  // std::set<CUdeviceptr, shmDev_base_cmp> shmDev_baseAddr_set; // base addr
  // set: desc ordered by dev baseAddr std::unordered_map<CUdeviceptr,
  // gmm_shmInfo_t*> shmDev_table; // map key by dev baseAddr
  std::map<CUdeviceptr, gmm_shmInfo_t *>
      shmDev_table;  // map key by dev baseAddr

  // gmm_vstore_mgr store_mgr;

 public:
   // metadata for NCCL mem plugin offloading
   std::unordered_map<CUdeviceptr, amem_allocMdata>              allocTable[AMEM_MAX_DEVS]; 
   std::unordered_map<CUmemGenericAllocationHandle, CUdeviceptr> handleTable[AMEM_MAX_DEVS]; 

   CUstream offloadStream[AMEM_MAX_DEVS];
   CUstream preloadStream[AMEM_MAX_DEVS];

   // stats for most recent pause iteration; reset as 0 after successful resume
   std::mutex pause_mtx; 
   int releaseLocalCnt;
   int releaseShadowCnt;
   int offloadCnt;
   int smokeLog;
   size_t allocBytes[AMEM_MAX_CALLER]; // stats. by caller
   size_t delBytes; // if not known by caller
   bool paused;  
   size_t pauseCnt;
   size_t resumeCnt;
  
  int get_devCnt() const { return dev_cnt; }
  bool get_gdrSupport() const { return (gdr_supported >= 1) ? true : false; }
  uint16_t get_slotID() const { return slot_id; }
  uint64_t get_UUID() const { return uuid; }

  // run a set of pre functions
  int exec_preAlloc();
  int exec_alloc();
  int exec_postAlloc();

  int exec_preFree();
  int exec_free();
  int exec_postFree();

  gmm_client_ctx(gmm_client_cfg *&cfg);
  ~gmm_client_ctx();

  int fetch(char *tgt_addr, char *src_addr, size_t bytes, CUstream &stream);
  int evict(char *src_addr, size_t bytes, CUstream &stream);

  // DM via multi-path sync interface, no req handle needed
  int htod_async(char *tgt_addr, char *host_addr, size_t bytes,
                 CUstream &stream);

  inline int htod(char *tgt_addr, char *host_addr, size_t bytes) {
    CUstream stream = nullptr;
    return htod_async(tgt_addr, host_addr, bytes, stream);
  }

  inline int dtoh(char *host_addr, char *src_addr, size_t bytes) {
    CUstream stream = nullptr;
    return dtoh_async(host_addr, src_addr, bytes, stream);
  }

  inline int dtod(char *tgt_addr, char *src_addr, size_t bytes) {
    CUstream stream = nullptr;
    return dtod_async(tgt_addr, src_addr, bytes, stream);
  }

  // scatter-gather info
  inline int insert_sg_info(gmm_id gid, int dev_id, char *host_addr,
                            size_t bytes, gmm_ipc_worker_req *workers,
                            int worker_cnt) {
    return 1;
  }

  // dst_addr: the address copy to.
  // host_addr: search key.
  inline int find_sg_info(char *dst_addr, char *host_addr,
                          gmm_ipc_worker_req *workers, int &worker_cnt) {
    return 1;
  }

  inline void delete_sg_info(char *host_addr) { }

  int dtoh_async(char *host_addr, char *src_addr, size_t bytes, CUstream &stream);
  int dtod_async(char *tgt_addr, char *src_addr, size_t bytes, CUstream &stream);

  // new API
  // e.g. RO thus able to dedup
  int malloc_hint(char *dptr, size_t bytes, const char *host_ptr, mem_advise_t advise);

  // merge vector IOs at host with SIMD, then copy to dst, buf is optional
  // if buf provided, use that buf, else internally alloc and release imme
  // (sync) or lazy (async)
  int devMemCopyVectorHtoD(char *tgt, size_t bytes, void *vectors, int count, char *buf);
  int devMemCopyVectorHtoD_async(char *tgt, size_t bytes, void *vectors, int count, char *buf);

  // query req status
  int query(int cur_dev, const gmm_req_t *&req);

  // blocking current thread until req_in done
  int synchronize(int cur_dev, gmm_req_t *req);

  // let stream wait on data moving for req
  int streamWait(int cur_dev, const cudaStream_t &stream, gmm_req_t *req);

  int reclaim_stream(const CUstream &stream, int dev_id);

  bool has_pending_mp(const CUstream &stream) {
    std::lock_guard<std::mutex> lock_(stream_map_lock);
    auto it = stream_map.find(stream);
    return (it != stream_map.end()) ? stream_map[stream] : false;
  }

  int reclaim_evt(CUevent &user_evt) {
    int ret = 0;

    auto it = evt_map.find(user_evt);
    if (it != evt_map.end()) {
      CUstream user_stream = evt_map[user_evt];
      if (has_pending_mp(user_stream)) {
        reclaim_stream(user_stream, -1);
        return ret;
      }
    }

    return ret;
  }

  void mark_evt(CUevent &evt, CUstream &stream) {
    if (has_pending_mp(stream)) evt_map.insert(std::make_pair(evt, stream));
  }
  pid_t get_pid() { return pid; }

  bool mp_ok(size_t bytes) { return false; }

  // check after allocation
  /*
  bool gdr_ok(CUDA_devMem *&ent, host_mem *&host_ent, size_t current_io_bytes) {
    return false;
  }
  */

  bool gdr_ok(size_t bytes) { return false; }

  //bool init_and_set_gdr(gdr_t &gH) { return false; }

  uint32_t get_shmIdx() { return ++shm_idx; }

  void add_shmEntry(gmm_shmInfo_t *&shm, bool dev_type = false) {
    if (dev_type == false) {
      // shm_baseAddr_set.insert(shm->get_addr());
      // printf("--[%s] ptr:[%p %p]\n", __func__, shm->get_addr(),
      // (char*)shm->get_addr() + shm->get_size());
      shm_table[(char *)shm->get_addr() + shm->get_size()] = shm;
    } else {
      // printf("--[%s Dev] ptr:[%p %p]\n", __func__, shm->get_addr(),
      // (char*)shm->get_addr() + shm->get_size());
      // shmDev_baseAddr_set.insert((CUdeviceptr)shm->get_addr());
      shmDev_table[(CUdeviceptr)shm->get_addr() + shm->get_size()] = shm;
    }
  }
  /*
  void add_devMemEntry(CUDA_devMem *&ent) {
    devMem_map[(CUdeviceptr)ent->get_addr() + ent->get_orig_size()] = ent;
  }
  */

  void add_hostMemEntry(host_mem *&ent) {
    hostMem_map[(char *)ent->get_addr() + ent->get_orig_size()] = ent;
  }

  // first lookup baseAddr if found, then lookup shmTable via baseAddr to
  // further check range
  int get_shmInfo(void *ptr, gmm_shmInfo_t *&shmInfo, size_t *offset) {
    auto it = shm_table.upper_bound(ptr);
    if (it != shm_table.end()) {
      if (ptr >= (it->second->get_addr())) {
        *offset = ((char *)ptr) - (char *)(it->second->get_addr());
        shmInfo = it->second;
        return 0;
      }
    }
    // gtrace();
    LOGGER(WARN, "Faield to find dev shmInfo for dptr:%p", ptr);
    // printf("--Fail %s dptr:%p\n", __func__, ptr);
#if 0
    auto it = shm_baseAddr_set.lower_bound(ptr);

    if (it != shm_baseAddr_set.end()) {
      auto ent = shm_table.find(*it);
      if (ent != shm_table.end() && ptr >= ent->second->get_addr() && 
          ptr < ((char *)(ent->second->get_addr()) + ent->second->get_size())) {
        *offset = ((char *)ptr) - (char *)(ent->second->get_addr());
	shmInfo = ent->second;
	printf("--[%s] Found shm_set:%zu ptr:%p\n", __func__, shm_baseAddr_set.size(), ptr);
        return 0;
      }
    } else {
      LOGGER(WARN, "Faield to find shmInfo for ptr:%p", ptr);
      gtrace();
    }
#endif
    return 1;
  }

  int get_shmInfo_dev(CUdeviceptr dptr, gmm_shmInfo_t *&shmInfo,
                      size_t *offset) {
    auto it = shmDev_table.upper_bound(dptr);

    if (it != shmDev_table.end()) {
      if (dptr >= ((CUdeviceptr)it->second->get_addr())) {
        *offset = ((char *)dptr) - (char *)(it->second->get_addr());
        shmInfo = it->second;
        return 0;
      }
    }
    // gtrace();
    LOGGER(WARN, "Faield to find dev shmInfo for dptr:%llx", dptr);
    return 1;
  }
  /*
  CUDA_devMem *find_devMemEntry(CUdeviceptr dptr) {
    auto it = devMem_map.upper_bound(dptr);
    if (it != devMem_map.end()) {
      if (dptr >= ((CUdeviceptr)it->second->get_addr())) {
        return it->second;
      }
    }
    // printf("--%s return nullptr\n", __func__);
    return nullptr;
  }
  */

  host_mem *find_hostMemEntry(void *ptr) {
    auto it = hostMem_map.upper_bound(ptr);
    if (it != hostMem_map.end()) {
      if (ptr >= (it->second->get_addr())) {
        return it->second;
      }
    }
    return nullptr;
  }

  gmm_shmInfo_t *find_shmEntry(void *&base_ptr) {
    auto ent = shm_table.find(base_ptr);
    if (ent != shm_table.end()) {
      return ent->second;
    } else {
      return nullptr;
    }
  }

  gmm_shmInfo_t *find_shmDevEntry(CUdeviceptr dptr) {
    auto ent = shmDev_table.find(dptr);
    if (ent != shmDev_table.end()) {
      return ent->second;
    } else {
      return nullptr;
    }
  }

  void del_shmDevEntry(CUdeviceptr dptr) {
    auto ent = shmDev_table.find(dptr);
    if (ent != shmDev_table.end()) {
      delete ent->second;
      shmDev_table.erase(dptr);
    }
  }
  /*
  void del_devMemEntry(CUdeviceptr dptr) {
    auto ent = devMem_map.find(dptr);
    if (ent != devMem_map.end()) {
      delete ent->second;
      devMem_map.erase(dptr);
    }
  }
  */

  void del_hostMemEntry(void *&ptr) {
    auto ent = hostMem_map.find(ptr);
    if (ent != hostMem_map.end()) {
      delete ent->second;
      hostMem_map.erase(ptr);
    }
  }

  // delete from hashtable and baseAddr set
  void del_shmEntry(void *&ptr) {
    auto ent = shm_table.find(ptr);
    if (ent != shm_table.end()) {
      delete ent->second;
      shm_table.erase(ptr);
    }
  }

  int register_shm(int dev_id, gmm_shmInfo_t *&shm, bool dev_shm);
  int deregister_shm(gmm_shmInfo_t *&shm, bool dev_shm);

  // Notify peer GPU about my addr 'dptr' is mapped from srcHandle at peer_dev
  int register_peer(gmm_shmInfo_t *&shm, CUmemGenericAllocationHandle srcHandle, CUdeviceptr dptr, int cur_dev, int peer_dev);
  // Notify peer GPU about new shareable FD when current GPU handle get re-allocated and exported during resume
  int update_peer(gmm_shmInfo_t *&shm, CUdeviceptr dptr, int sharedFD, int peer_dev);

  int release_peer_handle(gmm_shmInfo_t *&shm, CUdeviceptr dptr, int peer_dev);

  // entry for pinned host mem alloc
  int hostMem_alloc(CUdevice cur_dev, void **&pp, size_t bytesize, unsigned int Flags);
  // entry for pinned host mem free
  int hostMem_free(void *&p);

  // entry for dev mem allocation
  int devMem_alloc(CUdevice cur_dev, CUdeviceptr *&dptr, size_t bytesize);

  // entry for dev mem free
  int devMem_free(CUdevice cur_dev, CUdeviceptr dptr);

  int exec_devMem_preAlloc(CUdevice cur_dev, size_t bytesize);
  /*
  CUresult exec_devMem_alloc(CUdevice cur_dev, size_t bytesize,
                             CUDA_devMem *&ent);
  int exec_devMem_postAlloc(CUdevice cur_dev, CUDA_devMem *&ent);
  int devMem_postAlloc_export(CUdevice cur_dev, CUDA_devMem *&ent,
                              int &shared_fd);

  int exec_devMem_preFree(CUdevice cur_dev, CUdeviceptr dptr,
                          CUDA_devMem *&ent);
  CUresult exec_devMem_free(CUdevice cur_dev, CUdeviceptr dptr,
                            CUDA_devMem *&ent);
  int exec_devMem_postFree(CUdevice cur_dev, CUdeviceptr dptr,
                           CUDA_devMem *&ent);
  */

 private:
  void gmm_close();
  bool is_ready();

  bool is_valid(int pid) {
    // User pid must not be 0 or 1.
    return pid > 1;
  }

  bool is_sameProcess() {
    bool same_process = true;

    for (int i = 0; i < MAX_DEV_NR; ++i) {
      int w_pid = config_shm->worker_creator[i];
      if (is_valid(w_pid) && pid != w_pid) {
        same_process = false;
      }
    }
    return same_process;
  }

  int get_evt_at(int launcher_dev, int dev_idx, uint32_t evt_idx,
                 CUevent &pEvt);
  void reset_pending_mp(const CUstream &stream);
  int reclaim_evt(int dev_id, gmm_ipc_admin_req &mp);

  // connect to worker thr for the first time,then cache the connect
  void connect_if_not(int cur_dev, int tgt_dev);

  // TODO: perform multi-path data moving

  int set_req(gmm_req_t *req, int src_dev, int tgt_dev, size_t size, gmm_ipc_op op);
};
