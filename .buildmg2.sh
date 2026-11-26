#!/bin/bash
# review-18 step 1 (4.6-MG2): rebuild strict + serial with device MG transfer operators.
cd "$(dirname "$0")"
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
ninja -C build-kokkos-strict -j4 SU2_CFD/src/SU2_CFD > ../kk/logs/build-mg2-r1.log 2>&1
echo "strict rc=$?"
ninja -C build-kokkos-serial -j4 SU2_CFD/src/SU2_CFD > ../kk/logs/build-mg2-serial-r1.log 2>&1
echo "serial rc=$?"
tail -3 ../kk/logs/build-mg2-r1.log
tail -3 ../kk/logs/build-mg2-serial-r1.log
