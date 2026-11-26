#!/bin/bash
# review-23 step 5 (6b.b-TwallIn): strict build WITH python wrapper (pysu2/PYDRIVER support).
# Same options as build-kokkos-strict + -Denable-pywrapper=true. Log: ../kk/logs/build-tw-py-r1.log.
cd "$(dirname "$0")"
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export PYTHONPATH="$PWD/externals/meson"
LOG=../kk/logs/build-tw-py-r1.log
python3 meson.py --no-codi --no-medi --no-opdi --no-mpp --no-coolprop --no-mel --no-fado --no-mlpcpp --no-eigen setup build-kokkos-strict-py \
  -Denable-kokkos=true -Denable-autodiff=false -Denable-directdiff=false \
  -Denable-tecio=false -Denable-cgns=false -Dbuildtype=release \
  -Dkokkos-backend=serial -Dkokkos-arch=none -Dwith-mpi=disabled \
  -Denable-fast-math=false -Dcpp_args=-ffp-contract=off -Denable-pywrapper=true > $LOG 2>&1
echo "setup rc=$?"
ninja -C build-kokkos-strict-py -j8 SU2_CFD/src/SU2_CFD >> $LOG 2>&1
echo "build rc=$?"
tail -3 $LOG
