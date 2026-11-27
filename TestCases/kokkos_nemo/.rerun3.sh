#!/usr/bin/env bash
# Re-run the three failed queue steps with the fixed check scripts; new r2 log files (never overwrite r1).
set -u
cd "$(dirname "$0")/../.." || exit 2
SU2="$PWD"
export SU2_ROOT="$SU2"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
OUT="$SU2/validation_logs/2026-11-27_4.6-Periodic"
STRICT="$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD"
SERIAL="$SU2/build-kokkos-serial/SU2_CFD/src/SU2_CFD"
BIN="$STRICT" REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 SU2_KOKKOS_STAGE_VERIFY=1 RUN_TAG=_perab \
  bash TestCases/kokkos_nemo/check.sh invper_ausm > "$OUT/ab-per-r2.log" 2>&1
echo "ab-per-r2 rc=$?"
BIN="$STRICT" REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 SU2_KOKKOS_STAGE_VERIFY=1 RUN_TAG=_permusclab \
  EXTRA_CFG="MUSCL_FLOW= YES;SLOPE_LIMITER_FLOW= VAN_ALBADA_EDGE" \
  bash TestCases/kokkos_nemo/check.sh invper_ausm > "$OUT/ab-per-muscl-r2.log" 2>&1
echo "ab-per-muscl-r2 rc=$?"
# venk triage: stage-verify A/B on the venk config -> per-stage non-bitwise counts localize the divergence
BIN="$STRICT" REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 SU2_KOKKOS_STAGE_VERIFY=1 RUN_TAG=_pervenk \
  EXTRA_CFG="MUSCL_FLOW= YES;SLOPE_LIMITER_FLOW= VENKATAKRISHNAN" \
  bash TestCases/kokkos_nemo/check.sh invper_ausm > "$OUT/g1-per-venk-sv-r1.log" 2>&1
echo "venk-sv-r1 rc=$?"
BIN="$SERIAL" BIN_STRICT="$STRICT" CASE=invper_ausm K=10 \
  bash TestCases/kokkos_nemo/check_noisefloor.sh > "$OUT/nf-per-r2.log" 2>&1
echo "nf-per-r2 rc=$?"
