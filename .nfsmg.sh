#!/bin/bash
# One-shot noise-floor driver (review-16 queue step 8): production + strict binaries, invwedge_mg.
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
LOG=$1
SU2_ROOT=$SU2 MESH_DIR=$KK/meshes RUN_DIR=$KK/runs CASE=invwedge_mg \
  BIN=$SU2/build-kokkos-serial/SU2_CFD/src/SU2_CFD \
  BIN_STRICT=$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD \
  bash $SU2/TestCases/kokkos_nemo/check_noisefloor.sh > $LOG 2>&1
rc=$?
tail -15 "$LOG"
exit $rc
