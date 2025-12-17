### AMem NCCL Plugin test case (without RDMA)
set -x

######################## nccl-tests with AMem plugin
LOCAL_NCCL_TEST_PATH=$(pwd)/third_party/nccl-tests
LOCAL_NCCL_PATH=$(pwd)/third_party/nccl
export PATH=/usr/local/cuda/bin:$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LOCAL_NCCL_PATH/build/lib:$LD_LIBRARY_PATH
export GPU_NUM=8
export LIVEPAUSE=1 

# Explicitly enable the plugin for offloading
export NCCL_CUMEM_ENABLE=1
export AMEM_ENABLE=1
export AMEM_GROUPID=170 
# the larger setting, the more logs. 3: INFO; 4: DEBUG; 5: VERBOSE
export GMM_LOG=3 
export AMEM_NCCL_OFFLOAD_FREE_TAG=7

NCCL_TEST_PATH=$(pwd)/third_party/nccl-tests

TESTS="all_reduce_perf all_gather_perf alltoall_perf broadcast_perf"
for test in ${TESTS}; do
  # or test it w/o mpirun
  #./build/${test} -b 1M -e 8M -f 2 -g 8 -R 2
  mpirun --allow-run-as-root -bind-to none \
    -npernode ${GPU_NUM} \
    -x LD_LIBRARY_PATH \
    -x PATH \
    $LOCAL_NCCL_TEST_PATH/build/${test} -b 4M -e 8M -f 2 -g 1 -n 10

    if [ $? -ne 0 ]; then
       echo "===========================>>>>>>> testing failed:" ${test}
       exit 1
    else
       echo "===========================>>>>>>> testing done:" ${test}
    fi
done
exit 0

