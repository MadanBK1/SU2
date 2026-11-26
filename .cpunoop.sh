#!/bin/bash
# review-23 step 4 (6b.b-TwallIn): CPU no-op — plain super_cat (no MARKER_PYTHON_CUSTOM), strict binary.
# Compare chk_super_cat_noop_NO/restart_flow.csv vs chk_super_cat_pycustom_NO/restart_flow.csv (bitwise).
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
BIN=$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD
timeout 1800 env SU2_ROOT=$SU2 MESH_DIR=$KK/meshes RUN_DIR=$KK/runs \
  BIN=$BIN REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_noop \
  bash $SU2/TestCases/kokkos_nemo/check.sh super_cat > ../kk/logs/cpu-noop-tw-r1.log 2>&1
rc=$?
tail -8 ../kk/logs/cpu-noop-tw-r1.log
exit $rc
