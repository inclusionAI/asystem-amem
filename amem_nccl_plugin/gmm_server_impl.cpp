#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>

#include <unordered_map>

#include "gmm_host_shm.h"
#include "gmm_server.h"
#include "gmm_singleton.h"
#include "gmm_client_cfg.h"

#define MODULE_STATUS CUresult
extern void *gmm_libP;
extern gmm_client_cfg *g_cfg;
extern int amem_groupID;

// static thread_local FILE *gmm_log_file = nullptr;

// both server and client may call CUDA driver API
// client shall be wrapped and invoked with gmm_client_ctx,
//

// map from vmm_handle to fd_shared
struct fd_shared_t {
  int client_id;  // connect id
  int shared_fd;  // shareable fd
  pid_t pid;      // src client pid
  size_t bytes;   // alloc bytes
};
static std::unordered_map<CUmemGenericAllocationHandle, fd_shared_t>
    gmm_handle_map;

static void gmm_mem_cleanup(int worker, int connectID, size_t &borrow_sz) {
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

// admin thread for control, ideally one per node
void *gmm_admin_proc(void *args) {
  gmm_state *status = (gmm_state *)args;
  pid_t pid = getpid();
  std::mutex mtx;
  size_t borrow_sz = 0;
  const char *ipc_dir =
      getenv("GMM_IPC_DIR") ? getenv("GMM_IPC_DIR") : GMM_DEFAULT_IPC_DIR;
  char gmm_socket_path[1024];
  char gmm_lock_file[1024];
  struct sockaddr_un addr;
  struct timeval timeout;
  int socket_fd = -1;
  int lock_fd = -1;

  // make sure single admin process
  memset(gmm_socket_path, 0, sizeof(gmm_socket_path));
  memset(gmm_lock_file, 0, sizeof(gmm_lock_file));
  addr.sun_family = AF_UNIX;
  snprintf(gmm_socket_path, sizeof(gmm_socket_path) - 1, "%s/%s_%x", ipc_dir,
           gmm_admin_socket, amem_groupID);
  strncpy(addr.sun_path, gmm_socket_path, sizeof(addr.sun_path));

  SingletonProcess singleton(gmm_socket_path);
  if (!singleton()) {
    LOGGER(INFO, "pid:%d failed to bind %s error:%s", pid, gmm_socket_path,
           strerror(errno));
    *status = GMM_STATE_ADMIN_EXIST;
    return nullptr;
  }
  // clean up any previous left over tmp file
  //int result = system("rm -rf /tmp/gmm_*");

  socket_fd = singleton.GetSocket();

  snprintf(gmm_lock_file, sizeof(gmm_lock_file) - 1, "%s_%x", GMM_LOCK_FILE, amem_groupID);
  remove(gmm_lock_file);

  char worker_log_file[128];
  snprintf(worker_log_file, 127, "/tmp/gmm-admin.log");
  gmm_log_file = fopen(worker_log_file, "w");

  CHECK_DRV(cuInit(0));
  gmm_config_t *config = nullptr;
  gmm_mgr_t *mgr = new gmm_mgr_t(config);

  // re-create lock file to sync btw admin and worker
  lock_fd = open(gmm_lock_file, O_CREAT | O_RDWR, 0666);
  if (lock_fd < 0) {
    LOGGER(ERROR, "pid:%d failed to create lock file:%s error:%s", pid,
           gmm_lock_file, strerror(errno));
    goto admin_error;
  } else {
    close(lock_fd);
  }

  fd_set active_fd_set, read_fd_set;
  FD_ZERO(&active_fd_set);
  FD_SET(socket_fd, &active_fd_set);
  timeout.tv_sec = 0;
  timeout.tv_usec = 50;
  if (listen(socket_fd, MAX_CLIENT_NUM) < 0) {
    LOGGER(ERROR, "pid:%d GMM admin start failed to listen error:%s", pid,
           strerror(errno));
    goto admin_error;
  }

  mgr->config_shm->set_state(GMM_STATE_ADMIN_READY);
  *status = GMM_STATE_ADMIN_READY;
  //gmm_set_log_level();
  LOGGER(INFO, "groupID:%d pid:%d lockfile:%s adminfile:%s admin ok!", amem_groupID, getpid(), gmm_lock_file, gmm_socket_path);

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
            //LOGGER(DEBUG, "gmm admin: new connect id:%d", new_socket);
            FD_SET(new_socket, &active_fd_set);
          } else {
            // active then handling the req
            gmm_ipc_admin_req req = {};

            mtx.lock();
            // ret = recv(i, &req, sizeof(req), 0);
            ret = gmm_recv(i, &req, sizeof(req));
            if (ret > 0) {  // data comes
              switch (req.op) {
                case GMM_OP_NEW_WORKER: {
                  ret = mgr->newDev_handler(i, req);
                  gmm_send(i, &ret, sizeof(ret));
                  break;
                }

                case GMM_OP_NEW_CLIENT: {
                  gmm_ipc_admin_rsp rsp;
                  ret = mgr->newClient_handler(i, req, rsp);
                  gmm_send(i, &rsp, sizeof(ret));
                  break;
                }

                default: {
                  LOGGER(ERROR, "op:%d invalid", req.op);
                  break;
                }
              }
            } else if (ret == 0 && errno == EAGAIN) {
              LOGGER(INFO, "admin:%d client: read 0, EAGAIN ", i);
              continue;
            } else if (ret == 0 && errno != EINTR) {
              LOGGER(
                  INFO,
                  "admin:%d errno:%d %s client: read 0, do necessary clean up",
                  i, errno, strerror(errno));
              // gmm_mem_cleanup(dev, i, borrow_sz);
              close(i);
              FD_CLR(i, &active_fd_set);
            } else {  // exception
              LOGGER(INFO,
                     "admin:%d client: read ret -1, do necessary clean up", i);
              // gmm_mem_cleanup(dev, i, borrow_sz);
              close(i);
              FD_CLR(i, &active_fd_set);
            }
            mtx.unlock();
          }
        }
      }
    }
  }  // while

admin_error:
  if (gmm_is_file_exist(gmm_socket_path)) {
    remove(gmm_socket_path);
  }

  if (lock_fd) {
    close(lock_fd);
  }

  if (gmm_is_file_exist(gmm_lock_file)) {
    remove(gmm_lock_file);
  }

  if (mgr) {
    delete mgr;
  }

  if (gmm_log_file) {
    fclose(gmm_log_file);
  }
  *status = GMM_STATE_ADMIN_ERROR;
  return nullptr;
}

// start scheduler thread (one per job/node) if doesn't exist
gmm_state gmm_launch_admin_thr() {
  gmm_state ready = GMM_STATE_INIT;

  pthread_t thr;
  int ret = pthread_create(&thr, NULL, gmm_admin_proc, &ready);
  if (ret != 0) {
    LOGGER(INFO, "failed to create scheduler thread error:%s", strerror(errno));
  }
  std::string name{"Glake_Admin"};
  pthread_setname_np(thr, name.c_str());

  while (ready == GMM_STATE_INIT) {
    sched_yield();
  }
  return ready;
}

gmm_mgr_t::gmm_mgr_t(gmm_config_t *&config)
    : cpu_buffer(nullptr), config_shm(nullptr), config_fd(0) {
  int cur_dev = 0;
  pid_t pid = getpid();
  counter = 0;
  char gmm_config_file[1024];

  snprintf(gmm_config_file, sizeof(gmm_config_file) - 1, "%s_%x", GMM_CONFIG_SHM, amem_groupID);

  // create shm config buffer btw scheduler and workers
  config_fd = shm_open(gmm_config_file, O_CREAT | O_RDWR, 0666);
  if (config_fd < 0) {
    LOGGER(ERROR, "error open %s errno:%d error:%s", gmm_config_file, errno,
           strerror(errno));
    ASSERT(0, "Failed on shm_open");
  }

  int ret = ftruncate(config_fd, sizeof(gmm_config_t));

  config_shm =
      (gmm_config_t *)mmap(NULL, sizeof(gmm_config_t), PROT_READ | PROT_WRITE,
                           MAP_SHARED, config_fd, 0);
  if (config_shm == MAP_FAILED) {
    LOGGER(ERROR, "mmap failed, error:%s", strerror(errno));
    close(config_fd);
  }

  config_shm->init();

  const char *ipc_dir =
      getenv("GMM_IPC_DIR") ? getenv("GMM_IPC_DIR") : GMM_DEFAULT_IPC_DIR;
  snprintf(config_shm->ipc_dir, sizeof(config->ipc_dir) - 1, "%s", ipc_dir);

  config_shm->cpu_buf_sz =
      (getenv("GMM_CPU_BUF") ? atol(getenv("GMM_CPU_BUF")) : 128UL) << 20L;
  config_shm->dev_buf_sz =
      (getenv("GMM_GPU_BUF") ? atol(getenv("GMM_GPU_BUF")) : 128UL) << 20L;
  config_shm->remove_exist =
      (getenv("GMM_REMOVE_SOCKET") ? atoi(getenv("GMM_REMOVE_SOCKET")) : 0)
          ? true
          : false;
  config_shm->sync =
      (getenv("GMM_SYNC") ? atoi(getenv("GMM_SYNC")) : 0) ? true : false;
  config_shm->creator_pid = pid;
  config_shm->min_mp_size =
      (getenv("GMM_MP_MIN") ? atoi(getenv("GMM_MP_MIN")) : MIX_MP_ENABLE_MB)
      << 20UL;
  config_shm->max_slot = getenv("GMM_MAX_SLOT") ? atoi(getenv("GMM_MAX_SLOT"))
                                                : GMM_SERVER_MAX_SLOT;

  //config_shm->worker_mode = GMM_MODE_GLOBAL;
  config_shm->worker_mode = GMM_MODE_DDP_BIND;
  if ((getenv("GMM_DP") ? atoi(getenv("GMM_DP")) : 0)) {
    LOGGER(DEBUG, "=======Setting as DP mode");
    config_shm->worker_mode = GMM_MODE_DP_BIND;
  } else if ((getenv("GMM_DDP") ? atoi(getenv("GMM_DDP")) : 0)) {
    config_shm->worker_mode = GMM_MODE_DDP_BIND;
  }

  CHECK_DRV(cuDeviceGetCount(&devCnt));
  ASSERT(devCnt > 0 && devCnt <= MAX_DEV_NR, "Failed on dev_cnt");
  config_shm->tot_dev = devCnt;

  //config_shm->get_bus_link(devCnt);
  // config_shm->print_bus_link(devCnt);
  config = config_shm;

  //config_shm->setup_prefer_mp_HD();
  //config_shm->setup_prefer_mp_SG();
  //config_shm->setup_prefer_mp_DD();

  for (uint16_t i = 0; i < config_shm->max_slot; ++i) {
    slot_queue.push(i);
  }

  LOGGER(INFO,
         "dev_cnt:%d CUDA:%d.%d GMM_IPC_DIR:%s GMM_CPU_BUF:%ld GMM_GPU_BUF:%ld "
         "MB MODE:%d MIN_MP:%ld\n",
         config_shm->tot_dev, (config_shm->cuda_ver / 1000),
         (config_shm->cuda_ver % 1000) / 10, config_shm->ipc_dir,
         config_shm->cpu_buf_sz >> 20, config_shm->dev_buf_sz >> 20,
         config_shm->worker_mode, config_shm->min_mp_size);
}

int gmm_mgr_t::newDev_handler(int socket, gmm_ipc_admin_req &req) {
  int ret = 0;
  int dev_id = req.data.newDev_req.dev_id;
  pid_t pid = req.data.newDev_req.pid;

  free_evt_list[dev_id].fill(MAX_IPC_EVT_NR);
  return ret;
}

int gmm_mgr_t::newClient_handler(int socket_fd, gmm_ipc_admin_req &req,
                                 gmm_ipc_admin_rsp &rsp) {
  int ret = 0;

  get_slot(&rsp.data.new_client.slot_id);
  rsp.data.new_client.uuid = ++counter;
  //printf("new client socket_fd:%d slot_id:%d uuid:%ld\n", socket_fd,
   //      rsp.data.new_client.slot_id, rsp.data.new_client.uuid);

  client_tmp_resource *tmp =
      new client_tmp_resource(rsp.data.new_client.slot_id);
  client_to_cleanup[socket_fd] = tmp;
  return ret;
}

int gmm_mgr_t::delClient_handler(int socket_fd) {
  int ret = 0;

  client_tmp_resource *tmp = client_to_cleanup[socket_fd];
  //printf("client socket_fd:%d exit slot_id:%d\n", socket_fd, tmp->get_slotID());
  put_slot(tmp->get_slotID());
  client_to_cleanup.erase(socket_fd);
  delete tmp;

  return ret;
}

gmm_mgr_t::~gmm_mgr_t() {
  if (cpu_buffer) {
    free(cpu_buffer);
    cpu_buffer = nullptr;
  }

  // TODO: drain scheduler req and scheduler thread
  munmap(config_shm, sizeof(gmm_config_t));
  close(config_fd);

  char gmm_config_file[1024];
  snprintf(gmm_config_file, sizeof(gmm_config_file) - 1, "%s_%d", GMM_CONFIG_SHM, amem_groupID);
  shm_unlink(gmm_config_file);

  // TODO: remove evt
  LOGGER(INFO, "destroy gmm_mgr");
}
