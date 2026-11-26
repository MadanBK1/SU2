#!/usr/bin/env bash
# 4.8-ILUTeam step 3: A/B device-vs-host ILU(0) apply with team-per-row build
set -u
cd "$(dirname "$0")/../.."
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
LOGS="$HOME/software/AgenticTest/kk/logs"
mkdir -p "$LOGS"
BIN="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD" \
SU2_KOKKOS_PRECOND_VERIFY=1 REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_iluteamv \
  bash TestCases/kokkos_nemo/check.sh super_cat > "$LOGS/ab-ilu-team-r1.log" 2>&1
echo "rc=$?"
grep -n "KOKKOS_NEMO_VERIFY" "$HOME/software/AgenticTest/kk/runs/chk_super_cat_iluteamv_YES/log.txt" | head -5
tail -6 "$LOGS/ab-ilu-team-r1.log"
