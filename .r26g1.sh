#!/bin/bash
# R26 G1 driver: strict binary, bitwise tolerances. In-repo run/log dirs (sandbox keeps writes in the repo).
# Usage: bash .r26g1.sh <logfile> <case ...>
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
MESH=/home/mpr7366/software/AgenticTest/kk/meshes
mkdir -p "$SU2/runs-r26"
LOG=$1; shift
CASES="$*"
BIN=$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD \
SU2_ROOT=$SU2 MESH_DIR=$MESH RUN_DIR=$SU2/runs-r26 \
REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 \
bash $SU2/TestCases/kokkos_nemo/check.sh $CASES > $LOG 2>&1
rc=$?
tail -8 "$LOG"
exit $rc
