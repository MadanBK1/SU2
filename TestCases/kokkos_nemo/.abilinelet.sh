#!/usr/bin/env bash
# 4.6-Linelet step 4: A/B device-vs-host LINELET apply with SU2_KOKKOS_PRECOND_VERIFY=1
set -u
cd "$(dirname "$0")/../.."
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
LOGS="$HOME/software/AgenticTest/kk/logs"
mkdir -p "$LOGS"
BIN="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD" \
SU2_KOKKOS_PRECOND_VERIFY=1 REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_linelet \
EXTRA_CFG="LINEAR_SOLVER_PREC= LINELET" \
  bash TestCases/kokkos_nemo/check.sh super_cat > "$LOGS/ab-linelet-r1.log" 2>&1
echo "rc=$?"
grep -n "KOKKOS_NEMO_VERIFY" "$HOME/software/AgenticTest/kk/runs/chk_super_cat_linelet_YES/log.txt" | head -6
tail -8 "$LOGS/ab-linelet-r1.log"
