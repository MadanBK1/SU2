#!/usr/bin/env bash
# 4.6-Limiters step 3: G1 WALL_DISTANCE slope limiter, viscous super_cat (strict, bitwise)
set -u
cd "$(dirname "$0")/../.."
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
LOGS="$HOME/software/AgenticTest/kk/logs"
mkdir -p "$LOGS"
BIN="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD" \
REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_walld \
EXTRA_CFG="MUSCL_FLOW= YES;SLOPE_LIMITER_FLOW= WALL_DISTANCE" \
  timeout 1500 bash TestCases/kokkos_nemo/check.sh super_cat > "$LOGS/g1-lim-walld-r1.log" 2>&1
echo "rc=$?"
tail -25 "$LOGS/g1-lim-walld-r1.log"
