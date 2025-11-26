#pragma once
#include <sys/mman.h>
#include <sys/wait.h>

#include <deque>
#include <memory>
#include <unordered_map>

#include "gmm_common.h"
#include "gmm_cuda_common.h"
#include "gmm_host_shm.h"
#include "gmm_queue.h"
//#include "gmm_mempool.h"

// worker thread per GPU
// TODO: register to admin?, when admin exit, drain worker before exit
struct worker_args {
  pid_t launcher_pid;  // launcher pid
  int launcher_dev;    // launcher process's cur_dev
  int cur_dev;         // worker dev
  gmm_state ready;
  size_t init_dev_memPool_sz;
  size_t max_dev_memPool_sz;
  bool create_ctx;
};

// to manage GMM worker
class gmm_worker {
  int cur_dev;
  int launcher_dev;
  pid_t pid;
  pid_t ppid;
  bool create_ctx;

  size_t init_dev_memPool_sz;
  size_t max_dev_memPool_sz;
  size_t dev_buf_sz;
  size_t cpu_buf_sz;
  size_t dev_free_sz;
  size_t dev_tot_sz;
  size_t borrow_sz;
  int test_sz;
  int config_fd;
  int lock_fd;
  int max_retry;

  char gmm_socket_path[MAX_SHM_PATH_LEN];
  char log_file[128];

  char *gpu_buf;
  char *gpu_buf2;
  char *cpu_buf;
  const char *ipc_dir;
  gmm_config_t *config;

  struct sockaddr_un addr;
  int socket_fd;
  int admin_connect;

  CUdevice cu_dev;
  CUcontext ctx;
  // evt for pre/post evt, support IPC
  CUevent mp_evt[MAX_DEV_NR][MAX_IPC_EVT_NR];
  // 0: in_stream(H2D), 1: out_stream(D2H), one stream(for bother DMA and
  // compute) per peer (exclude self)
  CUstream dm_stream[MAX_PATH_NR + 1];
  // ctrl evts shared for all dm_streams
  CUevent dm_evt[MAX_DM_EVT_NR];  // evt for DMA pipeline
  gmm_evt_queue free_dm_evt_pool;

  gmm_shm_table shmHost_table;
  gmm_shm_table shmDev_table;
  CUmemAccessDesc accessDesc;

  //gmm_memPool *dev_memPool;  // pre-alloc dev mem pool
  //gmm_vstore_mgr store_mgr;  // cached obj

 public:
  
  gmm_worker(worker_args *arg) {
    pid = getpid();
    ppid = arg->launcher_pid;
    create_ctx = arg->create_ctx;
    cur_dev = arg->cur_dev;
    launcher_dev = arg->launcher_dev;

    init_dev_memPool_sz = arg->init_dev_memPool_sz;
    max_dev_memPool_sz = arg->max_dev_memPool_sz;

    config_fd = 0;
    lock_fd = 0;
    socket_fd = 0;
    max_retry = 10;

    dev_buf_sz = cpu_buf_sz = 0;
    dev_free_sz = dev_tot_sz = borrow_sz = 0;
    test_sz = 2UL << 20;

    gpu_buf = gpu_buf2 = cpu_buf = nullptr;
    config = nullptr;
  }

  ~gmm_worker() {
    if (admin_connect > 0) {
      close(admin_connect);
    }

    // TODO
    if (cpu_buf) {
      free(cpu_buf);
    }
    if (gpu_buf) {
    }
    if (gpu_buf2) {
    }


    if (ctx) {
      // TODO delete other GPU resource
      CHECK_DRV(cuCtxDestroy(ctx))
    }
    // TODO
    if (config) {
      munmap(config, sizeof(gmm_config_t));
    }

    if (config_fd > 0) {
      close(config_fd);
    }

    if (lock_fd > 0) {
      close(lock_fd);
    }

    if (gmm_is_file_exist(gmm_socket_path)) {
      remove(gmm_socket_path);
    }

    if (gmm_log_file) {
      fclose(gmm_log_file);
    }
  }
  pid_t get_pid() { return pid;} 
  int get_socket() { return socket_fd; }
  gmm_config_t *get_config() { return config; }
  size_t get_gpuBuf_size() { return dev_buf_sz; }

  CUstream get_inStream() { return dm_stream[0]; }
  CUstream get_outStream() { return dm_stream[1]; }
  CUstream get_stream(int dev_idx) { return dm_stream[dev_idx + 2]; }
  char *get_gpuBuf() { return gpu_buf; }
  char *get_gpuBuf2() { return gpu_buf2; }

  int init();
  int init_dm_res();
  int register_and_serve();

  int get_evt_at(int dev_idx, uint32_t evt_idx, CUevent &pEvt) {
    if (dev_idx >= MAX_DEV_NR || evt_idx >= MAX_IPC_EVT_NR) {
      LOGGER(ERROR, "Invalid dev:%d or evt:%d", dev_idx, evt_idx);
      return -1;
    }

    if (pid != config->gmm_evt[dev_idx].creator_pid) {
      if (mp_evt[dev_idx][evt_idx] == 0) {
        CHECK_DRV(
            cuIpcOpenEventHandle(&mp_evt[dev_idx][evt_idx],
                                 config->gmm_evt[dev_idx].evt_handle[evt_idx]));
      }
      pEvt = mp_evt[dev_idx][evt_idx];
      LOGGER(DEBUG, "IPC evt at src dev:%d evt_idx:%d evt:%p IPC evt", dev_idx,
             evt_idx, pEvt);
    } else {
      pEvt = config->gmm_evt[dev_idx].evt[evt_idx];
      LOGGER(DEBUG, "evt at src dev:%d evt_idx:%d evt:%p same-process", dev_idx,
             evt_idx, pEvt);
    }

    return 0;
  }

  int prepare_evt(gmm_ipc_worker_req &req, CUevent &pre_evt,
                  CUevent &done_evt) {
    int ret = 0;

    if (req.gmm_op >= GMM_OP_DM_START_MARK &&
        req.gmm_op <= GMM_OP_DM_END_MARK) {
      get_evt_at(cur_dev, req.worker_evt_idx,
                 done_evt);  // must to sync to complete
      if (req.async) {
        get_evt_at(req.src_dev, req.pre_evt_idx, pre_evt);
      }
    }

    return ret;
  }

  int dm_handler(gmm_ipc_worker_req &req, CUevent &pre_evt, CUevent &post_evt);
  int register_cuda_shm_handler(gmm_ipc_worker_req &req,
                                gmm_shmInfo_t *&shm_out, shared_fd fd);
  int register_cpu_shm_handler(gmm_ipc_worker_req &req,
                               gmm_shmInfo_t *&shm_out);

};

gmm_state gmm_launch_worker_thr(int cur_dev, int worker_dev, bool create_ctx);
