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
// for host mem management

#include <sys/types.h>
#include <unistd.h>

#include <cstring>

enum host_mem_type {
  HOST_MEM_TYPE_INVALID = 0,
  HOST_MEM_TYPE_PAGEABLE = 1,
  HOST_MEM_TYPE_PAGEABLE_SHM = 2,

  HOST_MEM_TYPE_PINNED = 3,
  HOST_MEM_TYPE_PINNED_SHM = 4,

};

// basic info for CPU/host pinned mem
struct host_mem {
  int dev_id;
  host_mem_type type;

  void *ptr;
  size_t orig_size;   // orig alloc req size
  size_t alloc_size;  // actual alloc size due to aligement

 public:
  int get_devID() const { return dev_id; }
  void *get_addr() const { return ptr; }
  size_t get_alloc_size() const { return alloc_size; }
  size_t get_orig_size() const { return orig_size; }
  host_mem_type get_type() const { return type; }

  void set_type(host_mem_type t) { type = t; }

 public:
  host_mem(int dev_, void *&ptr_, size_t size_, size_t alloc_size_,
           host_mem_type type_) {
    dev_id = dev_, ptr = ptr_;
    orig_size = size_;
    alloc_size = alloc_size_;
    type = type_;
    // printf("--new host mem addr:%p, size:%zuKB\n", ptr, size_>>10);
  }

  ~host_mem() {}
};
