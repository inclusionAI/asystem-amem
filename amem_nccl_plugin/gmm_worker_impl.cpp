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

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <chrono>
#include <memory>
#include <unordered_map>

#include "gmm_host_shm.h"
#include "gmm_singleton.h"
#include "gmm_worker.h"
#include "amem_nccl.h"
#include "gmm_client_cfg.h"

#define MODULE_STATUS CUresult
extern void *gmm_libP;
extern gmm_client_cfg *g_cfg;
extern int amem_groupID;

extern std::chrono::time_point<std::chrono::steady_clock> g_req_start_t;

// map from vmm_handle to fd_shared
struct fd_shared_t {
  int client_id;  // connect id
  int shared_fd;  // shareable fd
  pid_t pid;      // src client pid
  size_t bytes;   // alloc bytes
};

static std::unordered_map<CUmemGenericAllocationHandle, fd_shared_t> gmm_handle_map;

static void gmm_mem_cleanup(int worker, int connectID, size_t &borrow_sz) 
{
  for (auto iter = gmm_handle_map.begin(); iter != gmm_handle_map.end();) {
    if (iter->second.client_id == connectID) {
      int fd = iter->second.shared_fd;
      close(fd);
      CHECK_DRV(cuMemRelease(iter->first));
      if (borrow_sz >= iter->second.bytes) borrow_sz -= iter->second.bytes;

      LOGGER(INFO,
             "worker:%d clean up pid:%d fd:%d bytes:%ld cur_borrow:%ld MB",
             worker, iter->second.pid, fd, iter->second.bytes, borrow_sz >> 20);

      auto cur = iter;
      ++iter;
      gmm_handle_map.erase(cur);
    } else {
      ++iter;
    }
  }
}

int gmm_worker::init() {
  int ret = 0;

  ipc_dir = getenv("GMM_IPC_DIR") ? getenv("GMM_IPC_DIR") : GMM_DEFAULT_IPC_DIR;
  snprintf(gmm_socket_path, sizeof(gmm_socket_path) - 1, "%s/%s_%x", ipc_dir,
           gmm_admin_socket, amem_groupID);
  admin_connect = socket(AF_UNIX, SOCK_STREAM, 0);

  if (admin_connect < 0) {
    LOGGER(ERROR, "pid:%d worker:%d failed to create socket error:%s\n", pid,
           cur_dev, strerror(errno));
    ret = 1;
  }

  snprintf(log_file, 127, "/tmp/gmm-worker%d.log", cur_dev);
  gmm_log_file = fopen(log_file, "w");

  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, gmm_socket_path, sizeof(addr.sun_path) - 1);

  // connect to admin so ensure admin is ready
  while (((ret = connect(admin_connect, (struct sockaddr *)&addr,
                         sizeof(struct sockaddr_un))) != 0) &&
         (--max_retry >= 0)) {
    LOGGER(ERROR, "pid:%d worker:%d failed to connect to %s, error:%s", pid,
           cur_dev, gmm_socket_path, strerror(errno));
    sleep(1);
  }
  if (ret != 0) {
    LOGGER(ERROR, "pid:%d worker:%d failed to connect to admin", pid, cur_dev);
    ret = 2;
  }

  char gmm_file[1024];
  memset(gmm_file, 0, sizeof(gmm_file));
  snprintf(gmm_file, sizeof(gmm_file) - 1, "%s_%x", GMM_LOCK_FILE, amem_groupID); 
  if ((lock_fd = open(gmm_file, O_RDWR)) < 0) {
    LOGGER(ERROR, "pid:%d worker:%d failed to open %s error:%s", pid, cur_dev,
           gmm_file, strerror(errno));
    ret = 3;
  }

  memset(gmm_file, 0, sizeof(gmm_file));
  snprintf(gmm_file, sizeof(gmm_file) - 1, "%s_%x", GMM_CONFIG_SHM, amem_groupID);

  if ((config_fd = shm_open(gmm_file, O_RDWR, 0666)) < 0) {
    LOGGER(ERROR, "pid:%d worker:%d error open %s error:%s\n", pid, cur_dev,
           gmm_file, strerror(errno));
    ret = 4;
  }

  if ((config = (gmm_config_t *)mmap(NULL, sizeof(gmm_config_t),
                                     PROT_READ | PROT_WRITE, MAP_SHARED,
                                     config_fd, 0)) == MAP_FAILED) {
    LOGGER(ERROR, "pid:%d worker:%d mmap %s failed, error:%s", pid, cur_dev,
           gmm_file, strerror(errno));
    ret = 5;
  }

  return ret;
}

int gmm_worker::init_dm_res() {
  int ret = 0;

  dev_buf_sz = config->dev_buf_sz;
  cpu_buf_sz = config->cpu_buf_sz;
  cpu_buf = (char *)malloc(cpu_buf_sz);

  if (create_ctx) {
    CHECK_DRV(cuInit(0));
    CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx, cur_dev));
    CHECK_DRV(cuCtxSetCurrent(ctx));
    LOGGER(DEBUG, "pid:%d worker:%d create ctx done", pid, cur_dev);
  } else {
    LOGGER(DEBUG, "pid:%d worker:%d create ctx skip!!!", pid, cur_dev);
  }

  /*
  LOGGER(INFO, "pid:%d worker:%d to malloc, libP:%p", pid, cur_dev, gmm_libP);
  __CF("cuMemAlloc_v2")(((CUdeviceptr*)&gpu_buf, buf_size));
  LOGGER(INFO, "pid:%d worker:%d malloc done", pid, cur_dev);
  __CF("cuMemAlloc_v2")(((CUdeviceptr*)&gpu_buf2, buf_size));
  LOGGER(INFO, "pid:%d worker:%d malloc done2", pid, cur_dev);
  */
  for (int i = 0; i < MAX_PATH_NR + 1; ++i) {
    // TODO: filter and only create stream per NVLink interconnected
    CHECK_DRV(cuStreamCreate(&dm_stream[i], CU_STREAM_NON_BLOCKING));

    for (int j = 0; j < MAX_DM_EVT_NR; ++j) {
      CHECK_DRV(cuEventCreate(&dm_evt[j], CU_EVENT_DISABLE_TIMING));
    }
    free_dm_evt_pool.fill(MAX_DM_EVT_NR);
  }

  memset(mp_evt, 0, sizeof(mp_evt));
  for (int i = 0; i < MAX_IPC_EVT_NR; ++i) {
    if (config->worker_mode !=
        GMM_MODE_DP_BIND) {  // interprocess for global or ddp mode
      CHECK_DRV(cuEventCreate(&config->gmm_evt[cur_dev].evt[i],
                              CU_EVENT_DISABLE_TIMING | CU_EVENT_INTERPROCESS));
      CHECK_DRV(cuIpcGetEventHandle(
          (CUipcEventHandle *)&config->gmm_evt[cur_dev].evt_handle[i],
          config->gmm_evt[cur_dev].evt[i]));
      LOGGER(DEBUG, "worker:%d IPC evt:%p", cur_dev,
             config->gmm_evt[cur_dev].evt[i]);
    } else {
      CHECK_DRV(cuEventCreate(&config->gmm_evt[cur_dev].evt[i],
                              CU_EVENT_DISABLE_TIMING));
      LOGGER(DEBUG, "worker:%d same-process evt:%p", cur_dev,
             config->gmm_evt[cur_dev].evt[i]);
    }
  }
  config->gmm_evt[cur_dev].creator_pid = pid;

  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  accessDesc.location.id = cur_dev;
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

  CHECK_DRV(cuMemGetInfo(&dev_free_sz, &dev_tot_sz));

  if (init_dev_memPool_sz > dev_free_sz) init_dev_memPool_sz = dev_free_sz;
  if (max_dev_memPool_sz > dev_tot_sz) max_dev_memPool_sz = dev_tot_sz;

  return ret;
}

int gmm_worker::register_and_serve() {
  int ret = 0;
  gmm_ipc_admin_req req;
  req.data.newDev_req.pid = pid;
  req.data.newDev_req.dev_id = cur_dev;
  // CHECK_DRV(cuDeviceGetPCIBusId(req.data.newDev_req.dev_bus, 20, cur_dev));
  req.op = GMM_OP_NEW_WORKER;

  // reigster to admin
  if (gmm_send(admin_connect, (void *)&req, sizeof(req)) > 0 &&
      gmm_recv(admin_connect, (void *)&ret, sizeof(ret)) > 0 && ret == 0) {
  } else {
    LOGGER(ERROR, "pid:%d worker:%d failed to register to GMM admin", pid,
           cur_dev);
    return 1;
  }

  // start service
  if ((socket_fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
    LOGGER(ERROR, "pid:%d worker:%d failed to create socket error:%s", pid,
           cur_dev, strerror(errno));
    return 2;
  }

  snprintf(gmm_socket_path, sizeof(gmm_socket_path) - 1,
           "%s/gmm_worker_%d_%d.sock", ipc_dir, pid, cur_dev);
  strncpy(addr.sun_path, gmm_socket_path, sizeof(addr.sun_path));
  unlink(gmm_socket_path);
  if (bind(socket_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    LOGGER(ERROR, "pid:%d worker:%d failed to bind error:%s", pid, cur_dev,
           strerror(errno));
    return 3;
  }

  if (listen(socket_fd, MAX_CLIENT_NUM) < 0) {
    LOGGER(ERROR, "pid:%d worker:%d failed to listen error:%s", pid, cur_dev,
           strerror(errno));
    return 4;
  }

  struct flock f_lock;
  f_lock.l_type = F_WRLCK;
  f_lock.l_whence = SEEK_SET;
  f_lock.l_start = 0;
  f_lock.l_len = 0;

  fcntl(lock_fd, F_SETLKW, &f_lock);
  config->ready_cnt++;
  f_lock.l_type = F_UNLCK;
  fcntl(lock_fd, F_SETLKW, &f_lock);
  close(lock_fd);

  // finally set the pid, so that other peer would query the pid and get connected
  config->worker_creator[cur_dev] = pid;

  LOGGER(INFO,
         "pid:%d tid:%d worker:%d started, GPU mem freeMB:%ld totMB:%ld log:%s "
         "path:%s ctx:%d",
         pid, gettid(), cur_dev, dev_free_sz >> 20, dev_tot_sz >> 20, log_file,
         gmm_socket_path, create_ctx);

  // gmm_set_log_level();

  return ret;
}

int gmm_worker::register_cuda_shm_handler(gmm_ipc_worker_req &req,
                                          gmm_shmInfo_t *&shm_out,
                                          shared_fd fd) {
  int ret = 0;

  CUdeviceptr dptr;
  CHECK_DRV(cuMemAddressReserve(&dptr, req.byte, 0ULL, 0U, 0));
  CUmemGenericAllocationHandle handle;
  CHECK_DRV(
      cuMemImportFromShareableHandle(&handle, (void *)(uintptr_t)fd,
                                     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  CHECK_DRV(cuMemMap((CUdeviceptr)dptr, req.byte, 0ULL, handle, 0ULL));

  accessDesc.location.id = cur_dev;
  CHECK_DRV(cuMemSetAccess((CUdeviceptr)dptr, req.byte, &accessDesc, 1));
  gmm_shmInfo_t *shm = new gmm_shmInfo_t(GMM_IPC_MEM_NV_DEV_SHARE, req.src_dev,
                                         (void *)dptr, handle, req.byte, fd);
  LOGGER(INFO, "worker:%d pid:%d req.pid:%d register dev shm send rsp", cur_dev,
         pid, req.pid);
  shm_out = shm;

  return ret;
}

static void *gmm_worker_proc(void *args) {
  worker_args *arg = (worker_args *)args;
  bool create_ctx = arg->create_ctx;
  int cur_dev = arg->cur_dev;
  int launcher_dev = arg->launcher_dev;
  pid_t ppid = arg->launcher_pid;
  pid_t pid = getpid();
  int ret = 0;

  thread_local auto start_t = std::chrono::steady_clock::now();
  gmm_worker worker(arg);
  ret = worker.init();
  ret = worker.init_dm_res();
  ret = worker.register_and_serve();

  int socket_fd = worker.get_socket();
  fd_set active_fd_set, read_fd_set;
  FD_ZERO(&active_fd_set);
  FD_SET(socket_fd, &active_fd_set);
  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 10;

  std::mutex mtx;
  gmm_config_t *config = worker.get_config();
  size_t dev_buf_sz = worker.get_gpuBuf_size();
  CUstream in_stream = worker.get_inStream();
  CUstream out_stream = worker.get_outStream();

  arg->ready = GMM_STATE_WORKER_READY;
  while (true) {
    read_fd_set = active_fd_set;

    int ret = select(FD_SETSIZE, &read_fd_set, NULL, NULL, &timeout);
    if (ret >= 0) {
      for (int i = 0; i < FD_SETSIZE; i++) {
        if (FD_ISSET(i, &read_fd_set)) {
          if (i == socket_fd) {  // new connect
            int new_socket = accept(socket_fd, NULL, NULL);
            if (new_socket < 0) {
              continue;
            }
            LOGGER(DEBUG, "pid:%d worker:%d new connect id:%d", getpid(), cur_dev, new_socket);
            FD_SET(new_socket, &active_fd_set);
          } else {
            // active then handling the req
            gmm_ipc_worker_req req = {};
            shared_fd fd;
            mtx.lock();

            // auto t0  = std::chrono::steady_clock::now();
            // int ret = recv(i, &req, sizeof(gmm_ipc_worker_req), 0);
            int ret = gmm_recv(i, &req, sizeof(req), 1, &fd);
            // auto t1  = std::chrono::steady_clock::now();
            // auto duration =
            // std::chrono::duration_cast<std::chrono::microseconds>(t1
            // -t0).count();
            // std::cout << "-- recv dur:" << duration << " us\n";
            if (ret > 0) {  // data comes

              CUevent pre_evt = nullptr, done_evt = nullptr;
              //worker.prepare_evt(req, pre_evt, done_evt);

              switch (req.gmm_op) {
                case GMM_OP_REGISTER_SHM_DEV: {
                  gmm_shmInfo_t *shm = nullptr;
                  ret = worker.register_cuda_shm_handler(req, shm, fd);

                  gmm_ipc_worker_rsp rsp(ret, (uint64_t)shm);
                  gmm_send(i, &rsp, sizeof(rsp));
                  break;
                }

                case GMM_OP_DEREGISTER_SHM_DEV: {
                  gmm_shmInfo_t *shm = (gmm_shmInfo_t *)req.shmInfo_addr_src;
                  if (shm) {
                    CHECK_DRV(cuMemUnmap((CUdeviceptr)shm->get_addr(),
                                         shm->get_size()));
                    CHECK_DRV(cuMemAddressFree((CUdeviceptr)shm->get_addr(),
                                               shm->get_size()));
                    CHECK_DRV(cuMemRelease(shm->get_handle()));
                    close(shm->get_shmFd());
                  }
                  // no rsp
                  break;
                }

                case GMM_OP_REGISTER_PEER_INFO: {
                  CUmemGenericAllocationHandle src_handle = (CUmemGenericAllocationHandle)req.shmInfo_addr_src;
		  int peer_dev            = req.worker_dev;
                  CUdeviceptr peer_dptr  = (CUdeviceptr)req.shmInfo_addr_tgt;
                  ret = amem_registerPeerInfo(src_handle, peer_dptr, peer_dev);
                  gmm_send(i, &ret, sizeof(ret));
                  break;
                }

                case GMM_OP_UPDATE_PEER_INFO: {
                  CUdeviceptr peer_dptr  = (CUdeviceptr)req.shmInfo_addr_tgt;
		  int new_fd             = fd; // must use special recvmsg to get fd
                  ret = amem_updatePeerInfo(peer_dptr, new_fd);
                  gmm_send(i, &ret, sizeof(ret));
                  break;
                }

                case GMM_OP_RELEASE_PEER_HANDLE: {
                  CUdeviceptr peer_dptr  = (CUdeviceptr)req.shmInfo_addr_tgt;
                  ret = amem_cuMemReleaseHandle((void *)peer_dptr, 0);
                  gmm_send(i, &ret, sizeof(ret));
                  break;
                }

                case GMM_OP_STOP: {
                  LOGGER(ERROR, "To support");
                  break;
                }

                default: {
                  LOGGER(ERROR, "worker:%d invalid op:%d", cur_dev, req.gmm_op);
                  break;
                }
              }
            } else if (ret == 0 && errno != EINTR) {
              LOGGER(INFO, "admin: client:%d read 0", i);
              // gmm_mem_cleanup(dev, i, borrow_sz);
              close(i);
              FD_CLR(i, &active_fd_set);
            } else {  // exception
              LOGGER(INFO, "admin: client:%d read ret -1", i);
              // gmm_mem_cleanup(dev, i, borrow_sz);
              close(i);
              FD_CLR(i, &active_fd_set);
            }
            mtx.unlock();
          }
        }  // if FD_ISSET
      }    // For FD_SETSIZE
    }      // select ret>0

  }  // while

worker_error:
  arg->ready = GMM_STATE_WORKER_ERROR;
  return nullptr;
}

// start workers if doesn't exist
gmm_state gmm_launch_worker_thr(int cur_dev, int worker_dev, bool create_ctx) {
  worker_args arg = {.launcher_pid = getpid(),
                     .launcher_dev = cur_dev,
                     .cur_dev = worker_dev,
                     .ready = GMM_STATE_INIT,
                     .create_ctx = create_ctx};
  pthread_t thr;
  int ret = pthread_create(&thr, NULL, gmm_worker_proc, &arg);
  if (ret != 0) {
    LOGGER(ERROR, "failed to create worker thread error:%s", strerror(errno));
    ASSERT(0, "Failed on worker thread create");
  }

  std::string name{"Worker_"};
  name += std::to_string(worker_dev);
  pthread_setname_np(thr, name.c_str());

  while (arg.ready == GMM_STATE_INIT) {
    sched_yield();
  }

  return arg.ready;
}
