#!/usr/bin/env bash
# 4.8-ILUTeam step 4: old row-per-thread build still works (strict, bitwise)
set -u
cd "$(dirname "$0")/../.."
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
LOGS="$HOME/software/AgenticTest/kk/logs"
mkdir -p "$LOGS"
BIN="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD" \
SU2_KOKKOS_ILU_BUILD=row REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_ilurow \
  bash TestCases/kokkos_nemo/check.sh super_cat > "$LOGS/g1-ilu-row-r1.log" 2>&1
echo "rc=$?"
grep -h "ILU(0) device build\|preconditioner builds" "$HOME/software/AgenticTest/kk/runs/chk_super_cat_ilurow_YES/log.txt" | head -4
tail -6 "$LOGS/g1-ilu-row-r1.log"
