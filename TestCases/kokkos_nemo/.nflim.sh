#!/usr/bin/env bash
# 4.6-Limiters step 6: noise floor invwedge_ausm SHARP_EDGES (serial production vs strict CPU baseline), K=10
set -u
cd "$(dirname "$0")/../.."
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
LOGS="$HOME/software/AgenticTest/kk/logs"
mkdir -p "$LOGS"
BIN="$PWD/build-kokkos-serial/SU2_CFD/src/SU2_CFD" \
BIN_STRICT="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD" \
CASE=invwedge_ausm K=10 EXTRA_CFG="SLOPE_LIMITER_FLOW= SHARP_EDGES" \
  bash TestCases/kokkos_nemo/check_noisefloor.sh > "$LOGS/nf-lim-r1.log" 2>&1
echo "rc=$?"
tail -15 "$LOGS/nf-lim-r1.log"
