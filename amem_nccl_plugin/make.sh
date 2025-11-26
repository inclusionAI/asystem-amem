# first, build out libamem_nccl.so by this script, which is needed by new version of libnccl.so
# next, build libnccl.so
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
make amem
