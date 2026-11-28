#!/usr/bin/env bash
# One-shot: limoff control (PERIODIC_LIM=0) with STAGE_VERIFY=1 -> StageCompare evidence (review 40 step 2).
set -u
cd "$(dirname "$0")/../.." || exit 2
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
mkdir -p "$RUN_DIR"
OUT="$PWD/validation_logs/2026-11-27_4.6-PeriodicLim/limoff-verify-r3.log"
STRICT="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD"
# PERIODIC_LIM=0 + STAGE_VERIFY=1: forces limiter host AND periodic comms host (full-host limoff path),
# EXPECT key invper_lim_limoff via RUN_TAG.
env BIN="$STRICT" REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 SU2_KOKKOS_NEMO_PERIODIC_LIM=0 SU2_KOKKOS_STAGE_VERIFY=1 RUN_TAG=_limoff \
  timeout 600 bash TestCases/kokkos_nemo/check.sh invper_lim > "$OUT" 2>&1
rc=$?
echo "rc=$rc"
tail -25 "$OUT"
