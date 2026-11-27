#!/usr/bin/env bash
# 4.6-Limiters step 2: G1 SHARP_EDGES slope limiter (strict, bitwise)
set -u
cd "$(dirname "$0")/../.."
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
LOGS="$HOME/software/AgenticTest/kk/logs"
mkdir -p "$LOGS"
BIN="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD" \
REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_sharp \
EXTRA_CFG="SLOPE_LIMITER_FLOW= SHARP_EDGES" \
  timeout 1500 bash TestCases/kokkos_nemo/check.sh invwedge_ausm > "$LOGS/g1-lim-sharp-r1.log" 2>&1
echo "rc=$?"
tail -25 "$LOGS/g1-lim-sharp-r1.log"
