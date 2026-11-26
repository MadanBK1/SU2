#!/bin/bash
# round 73 (review 16, 4.6-MG): rebuild strict + serial
cd "$(dirname "$0")"
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
ninja -C build-kokkos-strict -j4 SU2_CFD/src/SU2_CFD > ../kk/logs/build-strict-r73.log 2>&1
echo "strict rc=$?"
ninja -C build-kokkos-serial -j4 SU2_CFD/src/SU2_CFD > ../kk/logs/build-serial-r73.log 2>&1
echo "serial rc=$?"
tail -2 ../kk/logs/build-strict-r73.log
tail -2 ../kk/logs/build-serial-r73.log
