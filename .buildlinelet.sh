#!/bin/bash
# review-27 step 1 (4.6-Linelet): rebuild strict + serial with device LINELET preconditioner.
cd "$(dirname "$0")"
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export PYTHONPATH="$PWD/externals/meson"
ninja -C build-kokkos-strict -j4 SU2_CFD/src/SU2_CFD > ../kk/logs/build-linelet-r1.log 2>&1
echo "strict rc=$?"
ninja -C build-kokkos-serial -j4 SU2_CFD/src/SU2_CFD > ../kk/logs/build-linelet-serial-r1.log 2>&1
echo "serial rc=$?"
tail -3 ../kk/logs/build-linelet-r1.log
tail -3 ../kk/logs/build-linelet-serial-r1.log
