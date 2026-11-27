#!/usr/bin/env bash
# 4.6-Linelet step 6: noise floor super_cat LINELET (serial production vs strict CPU baseline), K=10
set -u
cd "$(dirname "$0")/../.."
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
LOGS="$HOME/software/AgenticTest/kk/logs"
mkdir -p "$LOGS"
BIN="$PWD/build-kokkos-serial/SU2_CFD/src/SU2_CFD" \
BIN_STRICT="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD" \
CASE=super_cat K=10 EXTRA_CFG="LINEAR_SOLVER_PREC= LINELET" \
  bash TestCases/kokkos_nemo/check_noisefloor.sh > "$LOGS/nf-linelet-r1.log" 2>&1
echo "rc=$?"
tail -15 "$LOGS/nf-linelet-r1.log"
