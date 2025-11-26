CUR_DIR=$(shell pwd)
LOCAL_NCCL_DIR=$CUR_DIR/../third_party/nccl
CUDA_DIR=/usr/local/cuda

# may need to -I<current 2.27 NCCL.h directory>
FLAGS = -I${LOCAL_NCCL_DIR}/include \
   -I${CUDA_DIR}/include \
   -L${LOCAL_NCCL_DIR}/lib \
   -L${CUDA_DIR}/lib64 \
   -D_GNU_SOURCE \
   -g -O2 -DNDEBUG \
   -std=c++17 \
   -shared -lrt -ldl \
   -lpthread \

AMEMFLAGS = \
   -Xcompiler "-fPIC" \
   -Xlinker "-rpath=${GDR_LIB}" \
   -Xlinker "-soname,libamem_nccl.so.1" \
   -lcuda -lcudart \
   -gencode arch=compute_75,code=sm_75 \
   -gencode arch=compute_80,code=sm_80 \
   -gencode arch=compute_89,code=sm_89 \
   -gencode arch=compute_90,code=sm_90 \
   -gencode arch=compute_100a,code=sm_100a \

amem: libamem_nccl.so.1
libamem_nccl.so.1:amem_nccl.o gmm_common_impl.o gmm_client_cfg.o gmm_client_impl.o gmm_server_impl.o gmm_worker_impl.o gmm_host_shm_impl.o gmm_api_stats.o
	mkdir -p ../lib
	nvcc -o ../lib/$@ $^ ${FLAGS} ${AMEMFLAGS}
	ln -sf ../lib/libamem_nccl.so.1 ../lib/libamem_nccl.so

%.o: %.cpp
	g++ -c -g -o $@ $^ -fPIC ${FLAGS}
format:
	clang-format-11 -i *.cu *.cpp *.h --style=google

clean:
	rm -f *.o ../bin/libamem_nccl.so.1
