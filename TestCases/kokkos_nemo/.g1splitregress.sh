#!/usr/bin/env bash
# ResSplit dev-queue step 2: G1 strict regressions (default list + invwedge_lusgs invwedge_mg invwedge_dyn supin_visc)
# Log -> validation_logs/2026-11-27_ResSplit/g1-split-regress-r1.log ; archives chk_<case>_{NO,YES}/log.txt pairs.
set -u
cd "$(dirname "$0")/../.."
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
export RUN_TAG=_split
D="$PWD/validation_logs/2026-11-27_ResSplit"
mkdir -p "$D"
CASES="invwedge_ausm thermalbath super_cat axi_visccone invwedge_lusgs invwedge_mg invwedge_dyn supin_visc"
BIN="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD" \
REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 \
  bash TestCases/kokkos_nemo/check.sh $CASES \
  > "$D/g1-split-regress-r1.log" 2>&1
rc=$?
echo "rc=$rc"
for c in $CASES; do
  cp -f "$RUN_DIR/chk_${c}_split_NO/log.txt"  "$D/g1-split-regress-r1__${c}_NO.log"  2>/dev/null
  cp -f "$RUN_DIR/chk_${c}_split_YES/log.txt" "$D/g1-split-regress-r1__${c}_YES.log" 2>/dev/null
done
grep -E "PASS|FAIL|OVERALL" "$D/g1-split-regress-r1.log"
