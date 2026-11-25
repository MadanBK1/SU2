#!/bin/bash
# One-shot G1 driver: strict binary, bitwise tolerances.
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
LOG=$1; shift
CASES="$*"
BIN=$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD \
SU2_ROOT=$SU2 MESH_DIR=$KK/meshes RUN_DIR=$KK/runs \
REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 \
bash $SU2/TestCases/kokkos_nemo/check.sh $CASES > $LOG 2>&1
rc=$?
tail -8 "$LOG"
exit $rc
