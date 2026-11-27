#!/usr/bin/env bash
# Generic G1 runner: .g1run.sh <logfile> <case> [case...]
# Env in: BIN (default strict), REF_MODE (HOST), RES_TOL/FIELD_TOL (0), RUN_TAG, EXTRA_CFG.
set -u
cd "$(dirname "$0")/../.."
export SU2_ROOT="$PWD"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
LOG="$1"; shift
[ -n "${BIN:-}" ] || BIN="$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD"
REF_MODE=${REF_MODE:-HOST} RES_TOL=${RES_TOL:-0} FIELD_TOL=${FIELD_TOL:-0} \
  timeout 1500 bash TestCases/kokkos_nemo/check.sh "$@" > "$LOG" 2>&1
echo "rc=$?"
grep -E "PASS|FAIL|OVERALL" "$LOG"
tail -6 "$LOG"
