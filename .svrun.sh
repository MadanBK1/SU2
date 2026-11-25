#!/bin/bash
# STAGE_VERIFY A/B driver: strict binary, 3 iters, invwedge_roe.
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
LOG=$1
BIN=$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD \
SU2_ROOT=$SU2 MESH_DIR=$KK/meshes RUN_DIR=$KK/runs \
REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 ITER=3 RUN_TAG=_sv \
SU2_KOKKOS_STAGE_VERIFY=1 \
bash $SU2/TestCases/kokkos_nemo/check.sh ${2:-invwedge_roe} > $LOG 2>&1
rc=$?
tail -8 "$LOG"
exit $rc
