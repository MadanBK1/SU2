#!/bin/bash
# One-shot noise-floor driver: production serial BIN vs strict, K=10.
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
LOG=$1; shift
CASE=${1:-supin_visc}
SU2_ROOT=$SU2 MESH_DIR=$KK/meshes RUN_DIR=$KK/runs \
BIN=$SU2/build-kokkos-serial/SU2_CFD/src/SU2_CFD \
BIN_STRICT=$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD \
CASE=$CASE timeout 3000 bash $SU2/TestCases/kokkos_nemo/check_noisefloor.sh > "$LOG" 2>&1
rc=$?
tail -12 "$LOG"
exit $rc
