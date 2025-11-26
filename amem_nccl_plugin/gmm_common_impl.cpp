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

#include <sys/socket.h>
#include <sys/un.h>

#include <vector>
//#include <numa.h>

#include "gmm_common.h"
#include "gmm_queue.h"

static std::atomic<uint64_t> req_cnt;

static int get_link_weight(int rank) {
  int weight = 0;
  switch (rank) {
    case SELF_LINK_PERF_RANK: {
      weight = MIN_LINK_WEIGHT * 4;
      break;
    }
    case 0: {
      weight = MIN_LINK_WEIGHT * 4;
      break;
    }
    case 1: {
      weight = MIN_LINK_WEIGHT * 2;
      break;
    }
    case 2: {
      weight = MIN_LINK_WEIGHT;
      break;
    }
    default: {
      weight = MIN_LINK_WEIGHT;
      break;
    }
  }
  return weight;
}

inline size_t roundup(size_t value, unsigned int multiple) {
  return ((value - 1u) & ~(multiple - 1u)) + multiple;
}

void gmm_config_t::init() {
  for (int i = 0; i < MAX_REQ_NR; ++i) {
    req_pool[i].req_idx = i;
  }

  for (int i = 0; i < MAX_DEV_NR; ++i) {
    worker_creator[i] = 0;
    dev_bw_perf_rank[i] = new std::multimap<int, int, CmpKeyDesc>;
  }

  state = 0;
  ready_cnt = 0;
}

gmm_config_t::gmm_config_t() { init(); }

void gmm_req_t::set_param(char *tgt_a, int tgt_d, char *src_a, int src_d,
                          size_t bytes, int pri_in, gmm_ipc_op op) {
  tgt_dev = tgt_d;
  src_dev = src_d;
  src_addr = src_a;
  tgt_addr = tgt_a;
  tot_size = bytes;
  priority = pri_in;
  gmm_op = op;
  req_id = ++req_cnt;
  src_pid = getpid();
}

// block until req is done
int gmm_req_t::synchronize() {
  int ret = 0;
  // TODO: sync each event in sub task
  printf("%s\n", __FUNCTION__);
  return ret;
}

int gmm_req_t::query() {
  int ret = 0;
  // TODO: query each evt status in sub task
  printf("%s\n", __FUNCTION__);
  return ret;
}

void gmm_config_t::get_bus_link(int cnt) {
  // TODO:V100*8
  dev_cnt_per_PCIE_group = 2;

  for (int i = 0; i < cnt; ++i) {
    CHECK_DRV(cuDeviceGetPCIBusId(dev_bus[i], 20, i));
    for (int j = 0; j < cnt; ++j) {
      int perfRank = 0;
      int p2p = 0;
      if (j == i) {
        //dev_bw_perf_rank[i]->emplace(SELF_LINK_PERF_RANK, j);
        link_perf[i][j] = get_link_weight(SELF_LINK_PERF_RANK);
        p2p_support[i][j] = 1;
        continue;
      }

      CHECK_CUDA(
          cudaDeviceGetP2PAttribute(&p2p, cudaDevP2PAttrAccessSupported, i, j));
      CHECK_CUDA(cudaDeviceGetP2PAttribute(
          &perfRank, cudaDevP2PAttrPerformanceRank, i, j));
      link_perf[i][j] = get_link_weight(perfRank);
      p2p_support[i][j] = p2p;
      //dev_bw_perf_rank[i]->emplace(perfRank, j);
    }
  }
}

// check whether two dev are from same PCIe group(root-complex)
// for node config: V100 * 8
// TODO: parse lspci -vt output
inline bool gmm_config_t::same_PCIe_group(int src_dev, int tgt_dev) {
  return (src_dev / dev_cnt_per_PCIE_group) ==
         (tgt_dev / dev_cnt_per_PCIE_group);
}

// check any slower dev in same PCIe group of tgt_dev
// note: it shall also support P2P and high enough BW
bool gmm_config_t::has_slow_dev(int src_dev, int tgt_dev) {
  int group_idx = tgt_dev / dev_cnt_per_PCIE_group;
  int next_group = group_idx + 1;

  for (int i = group_idx * dev_cnt_per_PCIE_group;
       i < next_group * dev_cnt_per_PCIE_group; ++i) {
    // note: return true if has a dev with slower BW than tgt_dev but high
    // enough BW (NVLink)
    if (p2p_support[src_dev][i] && (link_perf[src_dev][i] > MIN_LINK_WEIGHT) &&
        (link_perf[src_dev][i] < link_perf[src_dev][tgt_dev])) {
      return true;
    }
  }

  return false;
}

// setup mp based on dev inter-connect topologies and bandwidth
// pre: perf_rank and link_perf and p2p table already setup
void gmm_config_t::setup_prefer_mp_HD() {
  memset(HD_MP, 0, sizeof(HD_MP));
  for (int i = 0; i < tot_dev; ++i) {
    // adding itself
    HD_MP[i].mp[0] = i;
    HD_MP[i].mp_nr++;

    for (auto iter = dev_bw_perf_rank[i]->begin();
         iter != dev_bw_perf_rank[i]->end(); ++iter) {
      int dev = iter->second;
      int rank = iter->first;

      // for host<->device MP: choose high inter-connect BW, P2P, and from diff
      // PCIe root-complex note: only 1 hop considered currently note: src dev
      // iteself is included
      if (rank <= NVLINK_PERF_SINGLE && p2p_support[i][dev] &&
          !same_PCIe_group(i, dev)) {
        // TODO: pick just one dev (prefer slower interconnect) from given PCIe
        // group
        if (has_slow_dev(i, dev)) {
          continue;
        }

        HD_MP[i].mp[HD_MP[i].mp_nr] = dev;
        HD_MP[i].mp_nr++;
      }
    }
  }

  // print result
  printf("-------HD MP ------\n");
  for (int i = 0; i < tot_dev; ++i) {
    printf("GPU %d: ", i);
    for (int j = 0; j < HD_MP[i].mp_nr; ++j) {
      printf("%3d", HD_MP[i].mp[j]);
    }
    printf("\n");
  }
}

void gmm_config_t::setup_prefer_mp_SG() {
  for (int i = 0; i < tot_dev; ++i) {
    SG_MP[i].mp_nr = 0;
    for (auto iter = dev_bw_perf_rank[i]->begin();
         iter != dev_bw_perf_rank[i]->end(); ++iter) {
      int dev = iter->second;
      int rank = iter->first;

      // for SG: choose diff dev, high inter-connect BW, and P2P
      if (i != dev && rank <= NVLINK_PERF_SINGLE && p2p_support[i][dev]) {
        SG_MP[i].mp[SG_MP[i].mp_nr] = dev;
        SG_MP[i].mp_nr++;
      }
    }
  }

  printf("-----Scatter-Gather MP -----\n");
  for (int i = 0; i < tot_dev; ++i) {
    printf("GPU %d: ", i);
    for (int j = 0; j < SG_MP[i].mp_nr; ++j) {
      printf("%3d", SG_MP[i].mp[j]);
    }
    printf("\n");
  }
}

// for DD:
// 1) just use P2P (mark as -1) if high inter-connect BW
// 2) or pick up a intermediate dev that connects src and tgt via high BW
// connection(rather than via CPU) e.g. DD_MP:
//         0  1  2  3  4  5  6  7
// GPU 0:  -1 -1 -1 -1  2  6 -1  1
// -1 means NVLink supported thus no need to goto MP via forwarding
// X(>0) means fwd via this dev=X, i.e., 0->4 via GPU2
void gmm_config_t::setup_prefer_mp_DD() {
  memset(DD_MP, DD_PATH_INVALID, sizeof(DD_MP));

  for (int i = 0; i < tot_dev; ++i) {
    for (int j = 0; j < tot_dev; ++j) {
      if (i == j) {
        DD_MP[i][j] = DD_PATH_P2P;
        continue;
      }

      if (link_perf[i][j] > MIN_LINK_WEIGHT && p2p_support[i][j]) {
        DD_MP[i][j] = DD_PATH_P2P;
        continue;

      } else if (DD_MP[j][i] > DD_PATH_P2P) {
        DD_MP[i][j] = DD_MP[j][i];
        continue;

      } else {
        bool find = false;
        int last_choice = DD_PATH_INVALID;

        for (auto iter_i = dev_bw_perf_rank[i]->begin();
             iter_i != dev_bw_perf_rank[i]->end(); ++iter_i) {
          int dev = iter_i->second;
          int rank = iter_i->first;

          if (i != dev && rank <= NVLINK_PERF_SINGLE && p2p_support[i][dev]) {
            for (auto iter_j = dev_bw_perf_rank[j]->begin();
                 iter_j != dev_bw_perf_rank[j]->end(); ++iter_j) {
              if (dev == iter_j->second && dev != j && p2p_support[j][dev] &&
                  (iter_j->first <= NVLINK_PERF_SINGLE)) {
                // note: for V100*8 don't waist the BW if they're not balanced
                if (link_perf[i][dev] != link_perf[j][dev]) {
                  last_choice = dev;
                  continue;
                } else {
                  DD_MP[i][j] = dev;
                  find = true;
                  break;
                }
              }
            }
          }

          if (find) {
            break;
          } else {
            DD_MP[i][j] = last_choice;
          }
        }  // for given i->j
      }
    }
  }

  printf("-------DD MP -------\n");
  printf("        ");
  for (int i = 0; i < tot_dev; ++i) {
    printf("%2d ", i);
  }
  printf("\n");

  for (int i = 0; i < tot_dev; ++i) {
    printf("GPU %d: ", i);
    for (int j = 0; j < tot_dev; ++j) {
      printf("%3d", DD_MP[i][j]);
    }
    printf("\n");
  }
}

void gmm_config_t::print_bus_link(int cnt) {
  printf(
      "============= GPU inter-dev bandwidth weight info: the higher, the "
      "faster ==========\n");
  printf("      ");
  for (int i = 0; i < cnt; i++) {
    printf("%3d ", i);
  }
  printf("\n");

  for (int i = 0; i < cnt; i++) {
    printf("GPU%d: ", i);
    for (int j = 0; j < cnt; j++) {
      if (i != j)
        printf("%3d ", link_perf[i][j]);
      else
        printf("   - ");
    }
    printf("\n");
  }

  printf(
      "============= GPU inter-dev link bandwidth ranking: dev:perf "
      "==========\n");
  for (int i = 0; i < cnt; i++) {
    printf("GPU%d-%ld:", i, dev_bw_perf_rank[i]->size());
    for (auto iter = dev_bw_perf_rank[i]->begin();
         iter != dev_bw_perf_rank[i]->end(); ++iter) {
      printf("%d(%d) ", iter->second, iter->first);
    }
    printf("\n");
  }
}

// send out msg at buffer with bytes, optional ctrl_msg and input fd
// ret:>0, ok the send bytes, others: error
ssize_t gmm_send(int connectFD, void *buffer, size_t bytes, bool ctrl_msg,
                 shared_fd fd) {
  ssize_t ret = 0;

  struct iovec iov[1];
  struct cmsghdr *cmsg = nullptr;
  char cmsg_buf[CMSG_SPACE(sizeof(fd))];
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  iov[0].iov_base = buffer;
  iov[0].iov_len = bytes;

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_flags = 0;

  if (ctrl_msg) {
    memset(&control_un, '\0', sizeof(control_un));
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);
    cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_len = CMSG_LEN(sizeof(fd));

    memmove(CMSG_DATA(cmsg), &fd, sizeof(fd));
    gmm_ipc_worker_req *req = (gmm_ipc_worker_req *)buffer;
    // LOGGER(DEBUG, "pid:%d to sendmsg fd:%d sendbyte:%ld buffer.byte:%ld",
    // req->pid, req->shared_fd, ret, bytes);
  }

  if ((ret = sendmsg(connectFD, &msg, 0)) < 0) {
    LOGGER(ERROR, "pid:%d sendmsg failed on connect:%d error:%s", getpid(),
           connectFD, strerror(errno));
  }

  return ret;
}

// recv msg to buffer with bytes, optional ctrl_flag and fd
// ret:>0, ok the recv bytes, others: error
ssize_t gmm_recv(int connectFD, void *buffer, size_t bytes, int ctrl_flag,
                 shared_fd *fd) {
  ssize_t ret = 0;

  struct iovec iov[1];
  struct cmsghdr *cmsg = nullptr;
  char cmsg_buf[CMSG_SPACE(sizeof(shared_fd))];
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  // Union to guarantee alignment requirements for control array
  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof(control_un.control);

  iov[0].iov_base = buffer;
  iov[0].iov_len = bytes;
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  if ((ret = recvmsg(connectFD, &msg, 0)) <= 0) {
    LOGGER(ERROR, "pid:%d recvmsg failed on connect:%d error:%s", getpid(),
           connectFD, strerror(errno));
    return ret;
  }

  if (ctrl_flag) {
    gmm_ipc_worker_req *req = (gmm_ipc_worker_req *)buffer;
    LOGGER(DEBUG, "pid:%d recvmsg ctrl fd:%d pid:%d byte:%ld recv-byte:%ld",
           getpid(), *fd, req->pid, req->byte, ret);

    if (req->gmm_op == GMM_OP_UPDATE_PEER_INFO || req->gmm_op == GMM_OP_REGISTER_SHM_DEV) {
      cmsg = CMSG_FIRSTHDR(&msg);
      if (cmsg != NULL && (cmsg->cmsg_len == CMSG_LEN(sizeof(shared_fd))) &&
          (cmsg->cmsg_level == SOL_SOCKET) && (cmsg->cmsg_type == SCM_RIGHTS)) {
        memmove(fd, CMSG_DATA(cmsg), sizeof(shared_fd));
      } else {
        LOGGER(ERROR,
               "pid:%d recvmsg done but unexpected control header setting",
               getpid());
        return -1;
      }
    }
  }

  return ret;
}
