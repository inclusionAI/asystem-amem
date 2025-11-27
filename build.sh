#!/bin/bash

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

AMEM_ROOT_DIR=$(pwd)
THIRD_NCCL=$AMEM_ROOT_DIR/third_party/nccl;
THIRD_NCCL_TEST=$AMEM_ROOT_DIR/third_party/nccl-tests;

apply_patches() {
    # apply patches
    PATCH_FILES=(
        "nccl_patch/nccl_2.27.5-1.diff"
        "nccl_patch/nccl-tests.diff"
    )

    SUBMODULE_PATHS=(
        "third_party/nccl"
        "third_party/nccl-tests"
    )

    for i in "${!SUBMODULE_PATHS[@]}"; do
        SUBMODULE_PATH="${SUBMODULE_PATHS[$i]}"
        PATCH_FILE="${PATCH_FILES[$i]}"

        FULL_SUBMODULE_PATH="${AMEM_ROOT_DIR}/$SUBMODULE_PATH"
        FULL_PATCH_PATH="${AMEM_ROOT_DIR}/$PATCH_FILE"

        echo "-----------------------------------------------------"
        echo "Processing submodule: $SUBMODULE_PATH"
        echo "Using patch file:   $PATCH_FILE"

        # Check if the submodule directory actually exists
        if [ ! -d "$FULL_SUBMODULE_PATH" ]; then
            echo "Error: Submodule path not found at $FULL_SUBMODULE_PATH"
            echo "Skipping this entry."
            continue
        fi

        # Check if the patch file actually exists
        if [ ! -f "$FULL_PATCH_PATH" ]; then
            echo "Error: Patch file not found at $FULL_PATCH_PATH"
            echo "Skipping this entry."
            continue
        fi

        # Navigate into the submodule directory in a subshell
        (
            cd "$FULL_SUBMODULE_PATH" || exit 1
            echo "Checking if patch is already applied..."
            git apply --check "$FULL_PATCH_PATH"
            if [ $? -eq 0 ]; then
                echo "Patch is not applied. Applying now..."
                git apply "$FULL_PATCH_PATH"
                if [ $? -eq 0 ]; then
                    echo "SUCCESS: Applied $PATCH_FILE."
                else
                    echo "FAILED: 'git apply' command failed for $PATCH_FILE."
                fi
            else
                echo "INFO: Patch $PATCH_FILE is already applied or cannot be applied."
            fi
        )
    done

    echo "-----------------------------------------------------"
    echo "Patch process finished."
}

build_project() {
    # build nccl
    cd $THIRD_NCCL;
    make -j96 src.build \
        NVCC_GENCODE="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100a,code=sm_100a"
    cd $AMEM_ROOT_DIR;

    # build plugin and cp lib
    cd $AMEM_ROOT_DIR/amem_nccl_plugin;
    make amem;
    cd $AMEM_ROOT_DIR;

    mkdir -p $THIRD_NCCL/build/lib
    cp $AMEM_ROOT_DIR/lib/libamem_nccl.so.1 $THIRD_NCCL/build/lib

    # apply patches
    apply_patches

    # rebuild nccl with patch
    cd $THIRD_NCCL;
    make -j96 src.build \
        NVCC_GENCODE="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100a,code=sm_100a"
    cd $AMEM_ROOT_DIR;

    # build nccl-tests
    cd $THIRD_NCCL_TEST;
    make -j96 MPI=1 \
        MPI_HOME=${MPI_HOME:-/opt/hpcx/ompi} \
        NCCL_HOME=$THIRD_NCCL/build \
        NVCC_GENCODE="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100a,code=sm_100a"
    cd $AMEM_ROOT_DIR;
}

clean_project() {
    cd $AMEM_ROOT_DIR;
    rm -rf ./bin ./amem_nccl_plugin/*.o $THIRD_NCCL/build $THIRD_NCCL_TEST/build
}

if [ "$1" == "clean" ]; then
    echo "Cleaning build artifacts..."
    clean_project
else
    build_project
fi
