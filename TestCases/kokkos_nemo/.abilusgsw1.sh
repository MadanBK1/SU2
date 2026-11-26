#!/usr/bin/env bash
# 4.8-LUSGSHyb step 5: A/B — WIDE=1 with SU2_KOKKOS_PRECOND_VERIFY=1 (LU_SGS apply device vs host)
set -u
cd "$(dirname "$0")/../.."
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
LOGS="$HOME/software/AgenticTest/kk/logs"
mkdir -p "$LOGS"
BIN="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD" \
SU2_KOKKOS_LUSGS_WIDE=1 SU2_KOKKOS_PRECOND_VERIFY=1 REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_lusgsw1v \
  bash TestCases/kokkos_nemo/check.sh invwedge_lusgs > "$LOGS/ab-lusgs-w1-r1.log" 2>&1
echo "rc=$?"
tail -25 "$LOGS/ab-lusgs-w1-r1.log"
echo "--- verify/perf ---"
grep -E "KOKKOS_NEMO_VERIFY|device LU_SGS applies" "$RUN_DIR/chk_invwedge_lusgs_lusgsw1v_YES/log.txt" | head -8
