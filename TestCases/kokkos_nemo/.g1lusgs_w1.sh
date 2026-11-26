#!/usr/bin/env bash
# 4.8-LUSGSHyb step 3: all-wide (WIDE=1) forces per-level launches on every level (strict, bitwise)
set -u
cd "$(dirname "$0")/../.."
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
LOGS="$HOME/software/AgenticTest/kk/logs"
mkdir -p "$LOGS"
BIN="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD" \
SU2_KOKKOS_LUSGS_WIDE=1 REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_lusgsw1 \
  bash TestCases/kokkos_nemo/check.sh invwedge_lusgs > "$LOGS/g1-lusgs-w1-r1.log" 2>&1
echo "rc=$?"
tail -25 "$LOGS/g1-lusgs-w1-r1.log"
echo "--- startup/perf ---"
grep -E "LU_SGS device apply|device LU_SGS applies|resident state" "$RUN_DIR/chk_invwedge_lusgs_lusgsw1_YES/log.txt" | head -6
