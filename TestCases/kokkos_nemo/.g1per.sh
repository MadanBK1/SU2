#!/usr/bin/env bash
# 4.6-Periodic step 3: G1 strict invper_ausm (EXPECT: periodic device line + device-resident ON)
set -u
cd "$(dirname "$0")/../.."
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
LOGS="$HOME/software/AgenticTest/kk/logs"
mkdir -p "$LOGS"
BIN="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD" \
REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 \
  timeout 1500 bash TestCases/kokkos_nemo/check.sh invper_ausm > "$LOGS/g1-per-r1.log" 2>&1
echo "rc=$?"
tail -25 "$LOGS/g1-per-r1.log"
