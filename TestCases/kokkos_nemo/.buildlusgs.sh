#!/usr/bin/env bash
# 4.8-LUSGSHyb step 1: rebuild strict + serial SU2_CFD (dev code DeviceLUSGSForwardRow/BackwardRow + SegmentLevels)
set -u
cd "$(dirname "$0")/../.."
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
LOGS=$HOME/software/AgenticTest/kk/logs
mkdir -p "$LOGS"
CXX=$GXX CC=$GCC ninja -C build-kokkos-strict -j4 SU2_CFD/src/SU2_CFD UnitTests/nemo_gas_kokkos_validation > "$LOGS/build-lusgs-r1.log" 2>&1
echo "strict rc=$?"
CXX=$GXX CC=$GCC ninja -C build-kokkos-serial -j4 SU2_CFD/src/SU2_CFD UnitTests/nemo_gas_kokkos_validation > "$LOGS/build-lusgs-serial-r1.log" 2>&1
echo "serial rc=$?"
tail -3 "$LOGS/build-lusgs-r1.log"; tail -3 "$LOGS/build-lusgs-serial-r1.log"
grep -m2 -E "LU_SGS|error" "$LOGS/build-lusgs-r1.log" | head -5 || true
