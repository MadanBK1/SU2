#!/usr/bin/env bash
# 4.6-Periodic step 1: rebuild strict + serial after dev code 6dcd2bf (device periodic rows).
# Logs -> validation_logs/2026-11-27_4.6-Periodic/ ; run from repo root.
set -u
cd "$(dirname "$0")/../.." || exit 2
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
# meson is vendored at externals/meson (no mesonbuild module in /usr/bin/python3).
export PYTHONPATH="$PWD/externals/meson"
D=validation_logs/2026-11-27_4.6-Periodic
mkdir -p "$D"

echo "== strict build =="
ninja -C build-kokkos-strict -j10 SU2_CFD/src/SU2_CFD UnitTests/nemo_gas_kokkos_validation \
  > "$D/build-per-r1.log" 2>&1
rc_s=$?
echo "== serial build =="
ninja -C build-kokkos-serial -j10 SU2_CFD/src/SU2_CFD UnitTests/nemo_gas_kokkos_validation \
  > "$D/build-per-serial-r1.log" 2>&1
rc_t=$?
echo "rc strict=$rc_s serial=$rc_t"
echo "--- strict tail ---"; tail -5 "$D/build-per-r1.log"
echo "--- serial tail ---"; tail -5 "$D/build-per-serial-r1.log"
