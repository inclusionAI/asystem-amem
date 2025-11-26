#include <sys/socket.h>
#include <sys/un.h>

#include <chrono>

#include "gmm_api_stats.h"
#include "gmm_client.h"

#if defined(MODULE_STATUS)
#undef MODULE_STATUS
#define MODULE_STATUS CUresult
#else
#define MODULE_STATUS CUresult
#endif

// notice: use __CF("") to call CUDA directly, avoiding invoke hook again
extern void *gmm_libP;
extern gmm_client_cfg *g_cfg;
extern int amem_groupID;

gmm_client_ctx::gmm_client_ctx(gmm_client_cfg *&client_cfg_) {
  pid = getpid();
  slot_id = -1;
  uuid = 0;
  int cur_dev = 0;
  size_t free_sz, total;
  gmm_state status = GMM_STATE_INIT;

  CHECK_CUDA(cudaGetDevice(&cur_dev));
  cudaMemGetInfo(&free_sz, &total);

  LOGGER(INFO, "groupID:%d pid:%d to init client, now memUsed:%ld MB", amem_groupID, pid, (total-free_sz)>>20);
  dev_cnt = client_cfg_->dev_cnt;

  // char worker_log_file[128];
  // snprintf(worker_log_file, 127, "/tmp/gmm-client-%d-%d.log", pid, cur_dev);
  // gmm_log_file = fopen(worker_log_file, "w");

  //gmm_set_log_level();
  client_cfg = client_cfg_;

  // try to enable GDR (if suported) for the first time

  // gmm_enable_p2p(cur_dev, dev_cnt); // create ctx on other dev! and not
  // necessary for VMM, but required for IPC evt!! (why?)
  // CHECK_CUDA(cudaSetDevice(cur_dev));

  for (int tgt_dev = 0; tgt_dev < dev_cnt; tgt_dev++) {
    worker_connects[tgt_dev] = -1;
  }

  // 1. try to start admin thread if not ready
  // but always one single admin in a process-group (such as a container) no
  // matter DP/DDP
  if (1) {
    status = gmm_launch_admin_thr();
    if (status == GMM_STATE_ADMIN_READY || status == GMM_STATE_ADMIN_EXIST) {
      LOGGER(DEBUG, "pid:%d cur_dev:%d GMM admin %s", pid, cur_dev,
             (status == GMM_STATE_ADMIN_READY) ? "created" : "already exists");
      sleep(1);
    } else {
      LOGGER(ERROR, "GMM admin launch return error:%d", status);
    }
    CHECK_CUDA(cudaSetDevice(cur_dev));
  }

  char gmm_file[1024]; 
  memset(gmm_file, 0, sizeof(gmm_file));
  snprintf(gmm_file, sizeof(gmm_file) - 1, "%s_%x", GMM_CONFIG_SHM, amem_groupID);

  for(int i = 0; i < 30; ++i) {
    config_fd = shm_open(gmm_file, O_RDWR, 0666);
    if (config_fd < 0) sleep(1);
  } 

  if (config_fd < 0) {
     LOGGER(ERROR, "pid:%d cur_dev:%d error open %s error:%s\n", pid, cur_dev,
           gmm_file, strerror(errno));
     ASSERT(0, "Failed on shm_open");
  }
  config_shm =
      (gmm_config_t *)mmap(NULL, sizeof(gmm_config_t), PROT_READ | PROT_WRITE,
                           MAP_SHARED, config_fd, 0);
  if (config_shm == MAP_FAILED) {
    perror("-- mmap:");
    LOGGER(ERROR, "pid:%d cur_dev:%d failed to mmap %s error:%s", pid,
           cur_dev, gmm_file, strerror(errno));
    close(config_fd);
  }
  // ensure admin is ready and config_shm is set correctly
  while (config_shm->state != GMM_STATE_ADMIN_READY) {
    sched_yield();
    LOGGER(DEBUG, "pid:%d cur_dev:%d waiting admin state ready...", pid, cur_dev);
  }

  // 2. start worker threads if not default global mode
  if (config_shm->worker_mode > GMM_MODE_DEFAULT) {
    // start worker thread for cur_dev
    LOGGER(DEBUG, "pid:%d cur_dev:%d mode:%d, launching my worker ...", pid,
           cur_dev, config_shm->worker_mode);
    status = gmm_launch_worker_thr(cur_dev, cur_dev, true);
    if (status == GMM_STATE_ADMIN_READY) {
      LOGGER(DEBUG, "Launching worker done ...");
    }

    CHECK_CUDA(cudaSetDevice(cur_dev));

    if (config_shm->worker_mode == GMM_MODE_DP_BIND) {
      for (int i = 0; i < dev_cnt; ++i) {
        if (i == cur_dev) continue;
        LOGGER(DEBUG,
               "pid:%d cur_dev:%d DP mode, launching other workers %d ...", pid,
               cur_dev, i);
        status = gmm_launch_worker_thr(cur_dev, i, true);
      }
    }
    CHECK_CUDA(cudaSetDevice(cur_dev));
    LOGGER(INFO, "groupID:%d pid:%d launch my worker done", amem_groupID, pid);
  }

  // 3. connect to admin and worker thr
  {
    struct sockaddr_un addr;
    char gmm_ipc_socket_path[MAX_SHM_PATH_LEN];
    admin_connect = 0;
    memset(gmm_ipc_socket_path, 0, sizeof(gmm_ipc_socket_path));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
    snprintf(gmm_ipc_socket_path, sizeof(gmm_ipc_socket_path) - 1, "%s/%s_%x",
             config_shm->ipc_dir, gmm_admin_socket, amem_groupID);
#pragma GCC diagnostic pop
    admin_connect = socket(AF_UNIX, SOCK_STREAM, 0);
    if (admin_connect < 0) {
      LOGGER(ERROR, "pid:%d cur_dev:%d failed to create socket error:%s", pid,
             cur_dev, strerror(errno));
    }

    strncpy(addr.sun_path, gmm_ipc_socket_path, sizeof(addr.sun_path) - 1);
    addr.sun_family = AF_UNIX;

    if (connect(admin_connect, (struct sockaddr *)&addr,
                sizeof(struct sockaddr_un)) != 0) {
      LOGGER(ERROR, "pid:%d cur_dev:%d failed to connect to admin %s, error:%s",
             pid, cur_dev, gmm_ipc_socket_path, strerror(errno));
      ASSERT(0, "Failed on connect");
    }

    gmm_ipc_admin_req req;
    gmm_ipc_admin_rsp rsp;
    req.data.newClient_req.pid = pid;
    req.data.newClient_req.dev_cnt = dev_cnt;
    req.op = GMM_OP_NEW_CLIENT;

    if (gmm_send(admin_connect, (void *)&req, sizeof(req)) > 0 &&
        gmm_recv(admin_connect, (void *)&rsp, sizeof(rsp)) > 0) {
      LOGGER(INFO, "clinet slot_id:%d", rsp.data.new_client.slot_id);
      slot_id = rsp.data.new_client.slot_id;
      uuid = rsp.data.new_client.uuid;
    } else {
      LOGGER(ERROR, "groupID:%d pid:%d error to connect admin thread", amem_groupID, pid);
      ASSERT(0, "Failed on connect to admin");
    }
  }

  if (config_shm->worker_mode == GMM_MODE_DDP_BIND) {
    while (config_shm->ready_cnt < dev_cnt) {
      LOGGER(INFO, "groupID:%d pid:%d readyWorkerCnt < dev_cnt:%d retry...", amem_groupID, pid, dev_cnt);
      sleep(1);
    }
  }

  for (int tgt_dev = 0; tgt_dev < dev_cnt; ++tgt_dev) {
    connect_if_not(cur_dev, tgt_dev);
  }
  
  LOGGER(INFO, "groupID:%d pid:%d connect to peer workers done", amem_groupID, pid);

  for (int i = 0; i < MAX_REQ_NR; i++) {
    // req_pool.push(new gmm_req_t(i));
    // TODO: now only one client that take over all req, to support multi-client
    // that share req
    req_pool.push(&config_shm->req_pool[i]);
  }

  memset(gmm_file, 0, sizeof(gmm_file));
  snprintf(gmm_file, sizeof(gmm_file) - 1, "%s_%x", GMM_LOCK_FILE, amem_groupID);
  if ((sched_fd = open(gmm_file, O_RDWR)) < 0) {
    LOGGER(ERROR, "failed to open %s error:%s", gmm_file, strerror(errno));
  }

  CHECK_CUDA(cudaSetDevice(cur_dev));
  shm_idx = 1;
  
  cudaMemGetInfo(&free_sz, &total);
  LOGGER(INFO, "groupID:%d pid:%d client init ok memUsed:%ld MB", amem_groupID, pid, (total-free_sz)>>20);
}

// detr
gmm_client_ctx::~gmm_client_ctx() {
  int cur_dev = 0;
  CHECK_CUDA(cudaGetDevice(&cur_dev));

  for (int d = 0; d < dev_cnt; ++d) {
    if (worker_connects[d] > 0) {
      LOGGER(VERBOSE, "Closing connect to %d id:%d", d, worker_connects[d]);
      close(worker_connects[d]);
    }
  }

  CHECK_CUDA(cudaSetDevice(cur_dev));
  LOGGER(INFO, "GMM client pid:%d exit\n", pid);

  if (gmm_log_file) {
    fclose(gmm_log_file);
  }
}

bool gmm_client_ctx::is_ready() {
  if (sched_fd > 0 && config_shm) {
    int ready_cnt = 0;
    f_lock.l_type = F_RDLCK;
    f_lock.l_whence = SEEK_SET;
    f_lock.l_start = 0;
    f_lock.l_len = 0;

    fcntl(sched_fd, F_SETLKW, &f_lock);
    ready_cnt = config_shm->ready_cnt;
    f_lock.l_type = F_UNLCK;
    fcntl(sched_fd, F_SETLKW, &f_lock);
    return (ready_cnt == config_shm->tot_dev);
  } else {
    return false;
  }
}

void gmm_client_ctx::gmm_close() {
  for (int i = 0; i < dev_cnt; i++) {
    if (worker_connects[i] > 0) {
      close(worker_connects[i]);
    }
  }
  for (int i = 0; i < MAX_REQ_NR; i++) {
    // TODO: now only one client that take over all req, to support multi-client
    // that share req
    std::shared_ptr<gmm_req_t *> obj = req_pool.pop();
    delete obj.get();
  }
  if (sched_fd > 0) close(sched_fd);
}

// client to get the usable evt
inline int gmm_client_ctx::get_evt_at(int launcher_dev_unused, int worker_dev,
                                      uint32_t evt_idx, CUevent &pEvt) {
  if (launcher_dev_unused >= MAX_DEV_NR || worker_dev >= MAX_DEV_NR ||
      evt_idx >= MAX_IPC_EVT_NR) {
    LOGGER(ERROR, "Invalid launcher_dev:%d or worker_dev:%d or evt:%d",
           launcher_dev_unused, worker_dev, evt_idx);
    return -1;
  }

  if (pid != config_shm->gmm_evt[worker_dev].creator_pid) {
    pEvt = mp_evt[worker_dev][evt_idx];
  } else {
    pEvt = config_shm->gmm_evt[worker_dev].evt[evt_idx];
  }

  return 0;
}

// connect to worker thr for the first time,then cache the connect
void gmm_client_ctx::connect_if_not(int cur_dev, int tgt_dev) {
  if (worker_connects[tgt_dev] <= 0) {
    char gmm_ipc_socket_path[MAX_SHM_PATH_LEN];
    struct sockaddr_un addr;
    pid_t worker_pid = config_shm->worker_creator[tgt_dev];

    // race: spin until peer worker get ready
    int retry_cnt = 0;
    while (worker_pid == 0 && retry_cnt < 30) {
      //sched_yield();
      worker_pid = config_shm->worker_creator[tgt_dev];
      LOGGER(INFO, "groupID:%d pid:%d peer %d is not ready, retry %d...", amem_groupID, pid, tgt_dev, retry_cnt);
      retry_cnt++;
      sleep(1);
    }

    if (worker_pid == 0) {
      LOGGER(ERROR, "groupID:%d pid:%d error to connect to peer %d!!", amem_groupID, pid, tgt_dev);
      ASSERT(0, "Failed to connect to peer");
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
    snprintf(gmm_ipc_socket_path, sizeof(gmm_ipc_socket_path) - 1,
             "%s/gmm_worker_%d_%d.sock", config_shm->ipc_dir, worker_pid,
             tgt_dev);
#pragma GCC diagnostic pop

    worker_connects[tgt_dev] = socket(AF_UNIX, SOCK_STREAM, 0);
    if (worker_connects[tgt_dev] < 0) {
      LOGGER(ERROR, "pid:%d failed on create socket for %s, error:%s", pid,
             gmm_ipc_socket_path, strerror(errno));
    }

    strncpy(addr.sun_path, gmm_ipc_socket_path, sizeof(addr.sun_path) - 1);
    addr.sun_family = AF_UNIX;

    int ret = -1;
    for (int i = 0; (ret != 0  && i < 10); ++i) {
      if ((ret = connect(worker_connects[tgt_dev], (struct sockaddr *)&addr,
                sizeof(struct sockaddr_un))) != 0) {
       LOGGER(ERROR, "pid:%d failed on connect to %s, error:%s", pid,
             gmm_ipc_socket_path, strerror(errno));
      }
    }
    if (ret != 0) {
      LOGGER(ERROR, "groupID:%d pid:%d error on connecting to %s after retry, error:%s", amem_groupID, pid,
             gmm_ipc_socket_path, strerror(errno));
      ASSERT(0, "Failed on peer connect");
    }

    // IPC evt only for cross-process, otherwise error code=201
    // cudaErrorDeviceUninitialized
    if (0 && pid != config_shm->gmm_evt[tgt_dev].creator_pid) {
      for (int i = 0; i < MAX_IPC_EVT_NR; i++) {
        CHECK_DRV(__CF("cuIpcOpenEventHandle")( &mp_evt[tgt_dev][i], config_shm->gmm_evt[tgt_dev].evt_handle[i]));
      }
    }  // else: they're from the same process, use config_shm-
  }
}

// TODO: to eliminate overhead to query H's baseAddr when
// cuPointerGetAttribute(&base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
// (CUdeviceptr)host_shm_offset), a possible option:
// pre-malloc big enough virtual mem, e.g. 80GB, very cheap as long as don't
// modify the mem record its baseAddr, notify peer to map when cuMemHostAlloc,
// alloc from above virtual mem, register thus pinned, notify peer to pin when
// H2D, check it's hostAlloc and offset can be quickly - baseAddr note: if ptr
// is malloc w/o register, cuPointerGetAttribute would return
// CUDA_ERROR_INVALID_VALUE(1)
int gmm_client_ctx::register_shm(int cur_dev, gmm_shmInfo_t *&shm,
                                 bool dev_mem) {
  int ret = 0;
  int task_num = 0;

  if (is_sameProcess()) {
    LOGGER(DEBUG, "same process, skip register");
    return ret;
  }

  gmm_ipc_worker_req dm_task[MAX_PATH_NR];
  gmm_ipc_worker_rsp worker_rsp[MAX_PATH_NR];

  gmm_ipc_admin_req req;
  gmm_ipc_admin_rsp rsp;

  // 1.query admin to get proper neighbor
  req.op = GMM_OP_PATH_QUERY_SHM;
  req.data.register_shm_req.pid = pid;
  req.data.register_shm_req.dev_id = cur_dev;

  if (gmm_send(admin_connect, (void *)&req, sizeof(req)) > 0 &&
      gmm_recv(admin_connect, (void *)&rsp, sizeof(rsp)) > 0) {
    task_num = rsp.data.neighbors.path_nr;
    for (int i = 0; i < task_num; ++i) {
      shm->shm_peer[i].dev_id = rsp.data.neighbors.path[i];

      dm_task[i].gmm_op =
          dev_mem ? GMM_OP_REGISTER_SHM_DEV : GMM_OP_REGISTER_SHM_HOST;
      dm_task[i].pid = pid;
      dm_task[i].src_dev = cur_dev;
      dm_task[i].worker_dev = rsp.data.neighbors.path[i];
      dm_task[i].byte = shm->get_size();
      dm_task[i].src_addr = (char *)shm->get_addr();

      if (dev_mem) {
        dm_task[i].shared_fd = shm->get_shmFd();
      } else {
        dm_task[i].shm_idx = shm->get_idx();
      }
    }
    shm->shm_num = task_num;

  } else {
    LOGGER(WARN,
           "pid:%d cur_dev:%d Register shm and query admin return error, "
           "fallback to normal. TODO<---",
           pid, cur_dev);
    return 1;
  }

  gmm_ipc_worker_req *task = &dm_task[0];
  for (int i = 0; i < task_num; ++i) {
    int gpu_id = task[i].worker_dev;
    // 3. submit to workers
    if (dev_mem == false) {  // host
      LOGGER(INFO, "pid:%d cur_dev:%d register %s shm to worker:%d op:%d pid:%d "
             "shm_idx:%d", pid, cur_dev, dev_mem ? "dev" : "host", gpu_id, task[i].gmm_op,
             task[i].pid, task[i].shm_idx);
      if (gmm_send(worker_connects[gpu_id], (void *)&task[i], sizeof(task[0])) >
              0 &&
          gmm_recv(worker_connects[gpu_id], (void *)&worker_rsp[i],
                   sizeof(worker_rsp[0])) > 0) {
        shm->shm_peer[i].shmInfo_addr = worker_rsp[i].data.shmInfo_addr;
        LOGGER(INFO, "pid:%d cur_dev:%d register host shm to worker:%d done",
               pid, cur_dev, gpu_id);
      } else {
        LOGGER(WARN,
               "pid:%d cur_dev:%d register host shm to worker:%d return error, "
               "continue",
               pid, cur_dev, gpu_id);
      }
    } else {
      if (gmm_send(worker_connects[gpu_id], (void *)&task[i], sizeof(task[0]),
                   true, task[i].shared_fd) > 0 &&
          gmm_recv(worker_connects[gpu_id], (void *)&worker_rsp[i],
                   sizeof(worker_rsp[0])) > 0) {
        shm->shm_peer[i].shmInfo_addr = worker_rsp[i].data.shmInfo_addr;
        LOGGER(INFO, "pid:%d cur_dev:%d register dev shm to worker:%d done",
               pid, cur_dev, gpu_id);
      } else {
        LOGGER(WARN, "Register dev shm to worker:%d return error, continue",
               gpu_id);
      }
    }
  }

  return ret;
}

int gmm_client_ctx::deregister_shm(gmm_shmInfo_t *&shm, bool dev_mem) {
  int ret = 0;

  if (is_sameProcess()) {
    return ret;
  }

  gmm_ipc_worker_req dm_task[MAX_PATH_NR];
  for (int i = 0; i < shm->shm_num; ++i) {
    dm_task[i].worker_dev = shm->get_peerShm(i).dev_id;

    dm_task[i].gmm_op =
        dev_mem ? GMM_OP_DEREGISTER_SHM_DEV : GMM_OP_DEREGISTER_SHM_HOST;
    dm_task[i].shmInfo_addr_src = shm->get_peerShm(i).shmInfo_addr;
    dm_task[i].pid = pid;
  }

  gmm_ipc_worker_req *task = &dm_task[0];
  for (int i = 0; i < shm->shm_num; ++i) {
    int gpu_id = task[i].worker_dev;
    // 3. submit to worker threads
    if (gmm_send(worker_connects[gpu_id], (void *)&task[i], sizeof(task[0])) >
        0) {
      // if (gmm_recv(worker_connects[gpu_id], (void *)&ret, sizeof(ret)) > 0)
    }
  }

  return ret;
}

// Notify peer GPU about my addr 'dptr' is mapped from srcHandle at peer_dev
int gmm_client_ctx::register_peer(gmm_shmInfo_t *&shm, CUmemGenericAllocationHandle srcHandle, CUdeviceptr dptr, int cur_dev, int peer_dev) {
  int ret = 1;

  if (is_sameProcess()) {
    return ret;
  }

  gmm_ipc_worker_req req;
  req.gmm_op           = GMM_OP_REGISTER_PEER_INFO;
  req.worker_dev       = cur_dev; // when the msg sent to 'peer_dev' and handled by 'peer_dev', my 'cur_dev' becomes its peer
  req.shmInfo_addr_src = srcHandle;
  req.shmInfo_addr_tgt = dptr;
  req.pid = pid;

  // Submit to peer_dev
  if (gmm_send(worker_connects[peer_dev], (void *)&req, sizeof(req)) > 0) {
    if (gmm_recv(worker_connects[peer_dev], (void *)&ret, sizeof(ret)) > 0) { 
    }
  }
  if (ret != 0) {
    LOGGER(WARN, "groupID:%d pid:%d peer_dev:%d conn:%d dptr:%llx send|recv error", amem_groupID, pid, peer_dev, worker_connects[peer_dev], dptr);
  }

  return ret;
}

// Notify peer GPU 'peer_dev' that my newly handle is ready at 'newFD', pls re-map the addr at 'peerDptr'
// when current GPU handle get newly allocated and exported during resume
int gmm_client_ctx::update_peer(gmm_shmInfo_t *&shm, CUdeviceptr peerDptr, int newFD, int peer_dev) {
  int ret = 1;

  if (is_sameProcess()) {
    return 0;
  }

  gmm_ipc_worker_req req;
  req.gmm_op           = GMM_OP_UPDATE_PEER_INFO;
  req.shmInfo_addr_tgt = peerDptr;
  req.shared_fd        = newFD;
  req.pid = pid;

  // Submit to peer's worker thread
  if (gmm_send(worker_connects[peer_dev], (void *)&req, sizeof(req), true, newFD) > 0) {
    if (gmm_recv(worker_connects[peer_dev], (void *)&ret, sizeof(ret)) > 0) {
    }
  }
  if (ret != 0) {
    LOGGER(WARN, "groupID:%d pid:%d peer_dev:%d peer_dptr:%llx newFD:%d send|recv error", amem_groupID, pid, peer_dev, peerDptr, newFD);
  }

  return ret;
}

// Notify peer GPU 'peer_dev' on peerDptr to unmap then release handle
int gmm_client_ctx::release_peer_handle(gmm_shmInfo_t *&shm, CUdeviceptr peerDptr, int peer_dev) {
  int ret = 1;

  if (is_sameProcess()) {
    return 0;
  }

  gmm_ipc_worker_req req;
  req.gmm_op           = GMM_OP_RELEASE_PEER_HANDLE;
  req.shmInfo_addr_tgt = peerDptr;
  req.pid = pid;

  // Submit to peer's worker thread
  if (gmm_send(worker_connects[peer_dev], (void *)&req, sizeof(req)) > 0) {
    if (gmm_recv(worker_connects[peer_dev], (void *)&ret, sizeof(ret)) > 0) {
    }
  }

  if (ret != 0) {
    LOGGER(WARN, "gropID:%d pid:%d peer_dev:%d peer dptr:%llx unamp failed", amem_groupID, pid, peer_dev, peerDptr);
  }
  return ret;
}

int gmm_client_ctx::htod_async(char *tgt_addr, char *host_addr, size_t bytes,
                               CUstream &stream) {
/*
  CUDA_devMem *dev_ent = find_devMemEntry((CUdeviceptr)tgt_addr);
  if (dev_ent == nullptr) {
    printf("--%s find_devMemEntry fail, ptr:%p\n", __func__, tgt_addr);
  }
  host_mem *host_ent = find_hostMemEntry(host_addr);
  if (dev_ent && host_ent && mp_ok(bytes)) {
  }
*/

  return 1;
}

int gmm_client_ctx::dtoh_async(char *host_addr, char *src_addr, size_t bytes,
                               CUstream &stream) {
  // TODO: amem DtoHAsync
  return 0;
}

// entry for pinned host mem alloc
int gmm_client_ctx::hostMem_alloc(CUdevice cur_dev, void **&pp, size_t bytesize,
                                  unsigned int Flags) {
  int ret = 0;
  CUresult rst = CUDA_SUCCESS;
  host_mem_type type = HOST_MEM_TYPE_INVALID;

  rst = __CF("cuMemHostAlloc")(
      pp, bytesize, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
  if (rst == CUDA_SUCCESS) {
    type = HOST_MEM_TYPE_PINNED;
  } else {
    return (int)rst;
  }

out:
  host_mem *ent = new host_mem(CPU_DEV, *pp, bytesize, bytesize, type);
  if (ent) {
    add_hostMemEntry(ent);
  }
  return ret;
}

// entry for pinned host mem free
int gmm_client_ctx::hostMem_free(void *&p) {
  // 1. notify peer workers to de-register
  // 2. deregister, unpin, unmap/free
  // 3. remove record
  CUresult rst = CUDA_SUCCESS;
  gmm_shmInfo_t *shm = find_shmEntry(p);
  if (shm) {
    int ret = deregister_shm(shm, false);  // async?
    rst = __CF("cuMemHostUnregister")(shm->get_addr());
    gmm_shmClose(shm);
    del_shmEntry(p);
    del_hostMemEntry(p);
    shm = nullptr;
    return 0;
  } else {
    rst = __CF("cuMemFreeHost")(p);

    host_mem *ent = find_hostMemEntry(p);
    if (ent) {
      del_hostMemEntry(p);
    }
    return 0;
  }
}

// perform any pre-check before allocation such as quota-check
// ret 0: success
//    >0: failure
int gmm_client_ctx::exec_devMem_preAlloc(CUdevice cur_dev, size_t bytesize) {
  int ret = 0;
  return ret;
}

// exec dev mem allocation
// ret 0: success
//    >0: failure
/*
CUresult gmm_client_ctx::exec_devMem_alloc(CUdevice cur_dev, size_t bytesize,
                                           CUDA_devMem *&ent) {
  cuda_mem_type type = GMM_MEM_TYPE_DEFAULT;
  CUmemGenericAllocationHandle vmm_handle;
  CUdeviceptr dptr;
  size_t alloc_size;

  CUresult rst = gmm_cuda_vmm_alloc(bytesize, cur_dev, dptr, vmm_handle,
                                    &alloc_size, false);
  if (rst == CUDA_SUCCESS) {
    client_cfg->inc_alloc_size(cur_dev, alloc_size);
  } else {
    // We return error code to client when GPU memory is used up, which has the
    // same behavior as cudaMalloc. Maybe we can use the following methods
    // (e.g., CUDA unified memory) to handle this case in the future.
    printf("WARN: %s() alloc GPU memory fail. size=%zu, dev=%d\n", __func__,
           bytesize, cur_dev);
    return rst;
  }

  if (rst == CUDA_SUCCESS) {
    ent =
        new CUDA_devMem(cur_dev, dptr, vmm_handle, bytesize, alloc_size, type);
    add_devMemEntry(ent);
    LOGI("GPU Mem alloc size:%ld dptr:0x%llx type:%d\n", bytesize, dptr, type);
  }

  return rst;
}

// for large alloc
int gmm_client_ctx::devMem_postAlloc_export(CUdevice cur_dev, CUDA_devMem *&ent,
                                            int &shared_fd) {
  CUresult rst = CUDA_SUCCESS;
  CUmemAccessDesc accessDesc;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = cur_dev;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  // TODO:may only accessable to NVLink connected dev
  for (int i = 0; i < get_devCnt(); ++i) {
    prop.location.id = i;
    accessDesc.location = prop.location;
    rst = __CF("cuMemSetAccess")(ent->get_addr(), ent->get_alloc_size(),
                                 &accessDesc, 1);
  }

  rst = __CF("cuMemExportToShareableHandle")(
      (void *)&shared_fd, ent->get_vmm_handle(),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  return (int)rst;
  // TODO: close the fd
}

// exec post action after dev mem allocation succeed
// ret 0: success
//    >0: failure
int gmm_client_ctx::exec_devMem_postAlloc(CUdevice cur_dev, CUDA_devMem *&ent) {
  // every successful alloc has an CUDA_devMem entry for magt purpose
  // depends on alloc type, size, optimizaiton flag, some of them may have
  // addition post alloc small alloc (<=64KB) : map by GDR large alloc (>= 8MB)
  // : register for IPC if multi-gpu exist medium alloc         : do-nothing
  // optimization for small obj
  int ret = 0;
  size_t alloc_size = ent->get_alloc_size();
  size_t orig_size = ent->get_orig_size();

  if (false && mp_ok(ent->get_orig_size()) && get_devCnt() > 1) {
    int shared_fd;
    ret = devMem_postAlloc_export(cur_dev, ent, shared_fd);
    gmm_shmInfo_t *shm =
        new gmm_shmInfo_t(GMM_IPC_MEM_NV_DEV, cur_dev, (void *)ent->get_addr(),
                          ent->get_vmm_handle(), alloc_size, shared_fd);
    if (register_shm(cur_dev, shm, true) == 0) {
      add_shmEntry(shm, true);
      ent->set_type(GMM_MEM_TYPE_IPC);
    } else {
      delete shm;
      ret = 1;
    }
  }

  // exec other post action
  return ret;
}
*/

// entry for dev mem allocation
int gmm_client_ctx::devMem_alloc(CUdevice cur_dev, CUdeviceptr *&dptr,
                                 size_t bytesize) {
  int ret = 0;
  CUresult rst = CUDA_SUCCESS;
/*
  CUDA_devMem *ent = nullptr;
  if ((0 == (ret = exec_devMem_preAlloc(cur_dev, bytesize))) &&
      (CUDA_SUCCESS == (rst = exec_devMem_alloc(cur_dev, bytesize, ent)))) {
    *dptr = ent->get_addr();
    ret = 0;

    // allocation is done, exec any post action
    exec_devMem_postAlloc(cur_dev, ent);
    return ret;
  }
*/

  return (rst == CUDA_SUCCESS) ? ret : (int)rst;
}

// pre action before free the dev mem
/*
int gmm_client_ctx::exec_devMem_preFree(CUdevice cur_dev, CUdeviceptr dptr,
                                        CUDA_devMem *&ent) {
  // 1. notify peer workers to de-register
  // 2. deregister, unpin, unmap/free
  // 3. remove record
  gmm_shmInfo_t *shm = find_shmDevEntry(dptr);
  if (shm && deregister_shm(shm, true)) {
    // TODO: any drain
    del_shmDevEntry(dptr);
  }

  return 0;
}


CUresult gmm_client_ctx::exec_devMem_free(CUdevice cur_dev, CUdeviceptr dptr,
                                          CUDA_devMem *&ent) {
  if (ent == nullptr) {
    // TODO:why ent is null?
    return CUDA_SUCCESS;
  }

  CUresult rst = CUDA_SUCCESS;
  rst = __CF("cuMemUnmap")(dptr, ent->get_alloc_size());
  rst = __CF("cuMemAddressFree")(dptr, ent->get_alloc_size());
  rst = __CF("cuMemRelease")(ent->get_vmm_handle());

  if (ent->get_type() >= GMM_MEM_TYPE_ALLOC) {
    client_cfg->dec_alloc_size(cur_dev, ent->get_alloc_size());
  }
  return rst;
}

int gmm_client_ctx::exec_devMem_postFree(CUdevice cur_dev, CUdeviceptr dptr,
                                         CUDA_devMem *&ent) {
  del_shmDevEntry(dptr);
  del_devMemEntry(dptr);
  return 0;
}
*/

// entry for dev mem free
int gmm_client_ctx::devMem_free(CUdevice cur_dev, CUdeviceptr dptr) {
  int ret = 0;
  CUresult rst = CUDA_SUCCESS;
/*
  CUDA_devMem *ent = nullptr;

  // TODO: locking
  if ((0 == (ret = exec_devMem_preFree(cur_dev, dptr, ent))) &&
      (CUDA_SUCCESS == (rst = exec_devMem_free(cur_dev, dptr, ent)))) {
    exec_devMem_postFree(cur_dev, dptr, ent);
    return ret;
  }
*/
  return (rst == CUDA_SUCCESS) ? ret : (int)rst;
}

/*int get_dev_bus_id()
 {
   char pciBusId[20];
   CHECK_DRV(cuDeviceGetPCIBusId(pciBusId, 20, src_dev));
}
*/
