#!/usr/bin/env bash
# 4.8-LUSGSHyb step 7: regressions (default list + invwedge_mg invwedge_dyn supin_visc, strict bitwise)
set -u
cd "$(dirname "$0")/../.."
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
LOGS="$HOME/software/AgenticTest/kk/logs"
mkdir -p "$LOGS"
BIN="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD" \
REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 \
  bash TestCases/kokkos_nemo/check.sh invwedge_ausm thermalbath super_cat axi_visccone invwedge_mg invwedge_dyn supin_visc \
  > "$LOGS/g1-lusgs-regress-r1.log" 2>&1
echo "rc=$?"
grep -E "exit=|PASS|FAIL|OVERALL" "$LOGS/g1-lusgs-regress-r1.log"
