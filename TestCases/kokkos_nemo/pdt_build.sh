#!/usr/bin/env bash
# 4.6-PeriodicDT: build strict + serial SU2_CFD, logs -> validation_logs/2026-11-27_4.6-PeriodicDT/
set -u
cd "$(dirname "$0")/../.." || exit 2
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
D=validation_logs/2026-11-27_4.6-PeriodicDT
mkdir -p "$D"
echo "== strict build =="
ninja -C build-kokkos-strict -j10 SU2_CFD/src/SU2_CFD > "$D/build-pdt-r1.log" 2>&1
rc1=$?
echo "strict rc=$rc1"
tail -5 "$D/build-pdt-r1.log"
echo "== serial build =="
ninja -C build-kokkos-serial -j10 SU2_CFD/src/SU2_CFD > "$D/build-pdt-serial-r1.log" 2>&1
rc2=$?
echo "serial rc=$rc2"
tail -5 "$D/build-pdt-serial-r1.log"
exit $(( rc1 + rc2 ))
