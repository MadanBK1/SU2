#!/bin/bash
# LU_SGS A/B verify driver (review-15 queue step 6): strict binary, SU2_KOKKOS_PRECOND_VERIFY=1.
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
LOG=$1; shift
BIN=$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD \
SU2_ROOT=$SU2 MESH_DIR=$KK/meshes RUN_DIR=$KK/runs \
REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_lusgs_sv \
SU2_KOKKOS_PRECOND_VERIFY=1 \
bash $SU2/TestCases/kokkos_nemo/check.sh invwedge_lusgs > $LOG 2>&1
rc=$?
tail -8 "$LOG"
exit $rc
