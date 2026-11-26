#!/bin/bash
# review-23 step 2 (6b.b-TwallIn): G1 identity, super_cat, MARKER_PYTHON_CUSTOM, no PYDRIVER.
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
LOG=${1:-$KK/logs/g1-tw-ident-r1.log}
BIN=$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD \
SU2_ROOT=$SU2 MESH_DIR=$KK/meshes RUN_DIR=$KK/runs \
REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_pycustom \
EXTRA_CFG="MARKER_PYTHON_CUSTOM= ( Wall )" \
bash $SU2/TestCases/kokkos_nemo/check.sh super_cat > "$LOG" 2>&1
rc=$?
tail -8 "$LOG"
exit $rc
