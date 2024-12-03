#!/bin/bash
# build.sh

#### Build
mkdir -p build
cd build

cmake \
  -DPFM_INCLUDE_DIR=/usr/include \
  -DPFM_LIBRARY=/opt/nvidia/nsight-compute/2024.3.2/host/linux-desktop-glibc_2_11_3-x64/libpfm.so.4 \
  -DPROFILING_ENABLED=ON \
  -DCMAKE_BUILD_TYPE=Release \
  ..

make -j 64
cd ..
# ./build/matmul