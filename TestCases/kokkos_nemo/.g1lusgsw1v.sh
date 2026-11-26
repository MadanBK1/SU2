#!/usr/bin/env bash
# 4.8-LUSGSHyb step 6: viscous super_cat LU_SGS with SU2_KOKKOS_LUSGS_WIDE=1 (strict, bitwise)
set -u
cd "$(dirname "$0")/../.."
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
LOGS="$HOME/software/AgenticTest/kk/logs"
mkdir -p "$LOGS"
BIN="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD" \
SU2_KOKKOS_LUSGS_WIDE=1 EXTRA_CFG="LINEAR_SOLVER_PREC= LU_SGS" \
REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_lusgsw1v \
  bash TestCases/kokkos_nemo/check.sh super_cat > "$LOGS/g1-lusgs-visc-w1-r1.log" 2>&1
echo "rc=$?"
tail -25 "$LOGS/g1-lusgs-visc-w1-r1.log"
echo "--- startup/perf ---"
grep -E "LU_SGS device apply|device LU_SGS applies|resident state|preconditioner" "$RUN_DIR/chk_super_cat_lusgsw1v_YES/log.txt" | head -8
