#!/usr/bin/env bash
# 4.6-Limiters step 4: A/B — SHARP_EDGES invwedge_ausm with SU2_KOKKOS_STAGE_VERIFY=1 (strict, bitwise)
set -u
cd "$(dirname "$0")/../.."
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
LOGS="$HOME/software/AgenticTest/kk/logs"
mkdir -p "$LOGS"
BIN="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD" \
SU2_KOKKOS_STAGE_VERIFY=1 REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_sharppv \
EXTRA_CFG="SLOPE_LIMITER_FLOW= SHARP_EDGES" \
  timeout 1500 bash TestCases/kokkos_nemo/check.sh invwedge_ausm > "$LOGS/ab-lim-sharp-r1.log" 2>&1
echo "rc=$?"
tail -25 "$LOGS/ab-lim-sharp-r1.log"
echo "--- verify lines ---"
grep -E "KOKKOS_NEMO_VERIFY|StageCompare|Limiter_Primitive|Solution_Max|Solution_Min" "$RUN_DIR/chk_invwedge_ausm_sharppv_YES/log.txt" 2>/dev/null | head -20
