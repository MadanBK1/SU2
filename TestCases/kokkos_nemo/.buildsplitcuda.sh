#!/usr/bin/env bash
# ResSplit dev-queue step 3: rebuild CUDA (V100 sm_70) after the Residual.cpp TU split; record time.
# Log -> validation_logs/2026-11-27_ResSplit/build-split-cuda-r1.log ; run from repo root.
set -u
cd "$(dirname "$0")/../.." || exit 2
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export SU2_KOKKOS_HOST_COMPILER=$GXX SU2_KOKKOS_CUDA_ARCH=sm_70 NVCC_WRAPPER_DEFAULT_COMPILER=$GXX
# meson regen needs the vendored meson (dev commit changed meson.build)
export PYTHONPATH="$PWD/externals/meson"
D=validation_logs/2026-11-27_ResSplit
mkdir -p "$D"
echo "start $(date -Is)"
time ninja -C build-kokkos-cuda -j10 SU2_CFD/src/SU2_CFD > "$D/build-split-cuda-r1.log" 2>&1
rc=$?
echo "rc=$rc end $(date -Is)"
tail -5 "$D/build-split-cuda-r1.log"
grep -cE " error" "$D/build-split-cuda-r1.log" || true
