#!/bin/bash
# Rebuild build-kokkos-cuda at current HEAD (G5 gate, item 4.3).
# Log: validation_logs/2026-11-27_4.3-CUDA/build-cuda-r1.log
cd "$(dirname "$0")/../.."
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export PATH=/usr/local/cuda-12.9/bin:$PATH
export SU2_KOKKOS_HOST_COMPILER=$GXX
export SU2_KOKKOS_CUDA_ARCH=sm_70
export NVCC_WRAPPER_DEFAULT_COMPILER=$GXX
mkdir -p validation_logs/2026-11-27_4.3-CUDA
ninja -C build-kokkos-cuda -j10 SU2_CFD/src/SU2_CFD > validation_logs/2026-11-27_4.3-CUDA/build-cuda-r1.log 2>&1
echo "RC=$?" >> validation_logs/2026-11-27_4.3-CUDA/build-cuda-r1.log
