#!/bin/bash
# review-21 step 1 (6b.a-DynGrid): rebuild strict + serial with dynamic-grid code.
cd "$(dirname "$0")"
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
# vendored meson must be importable for ninja's "regenerate build.ninja" rule
export PYTHONPATH="$PWD/externals/meson"
ninja -C build-kokkos-strict -j4 SU2_CFD/src/SU2_CFD > ../kk/logs/build-dyn-r1.log 2>&1
echo "strict rc=$?"
ninja -C build-kokkos-serial -j4 SU2_CFD/src/SU2_CFD > ../kk/logs/build-dyn-serial-r1.log 2>&1
echo "serial rc=$?"
tail -3 ../kk/logs/build-dyn-r1.log
tail -3 ../kk/logs/build-dyn-serial-r1.log
