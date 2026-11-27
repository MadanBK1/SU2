#!/usr/bin/env bash
# Re-run the 4.6-Periodic noise floor step (nf-per-r3, new log file; never overwrite).
set -u
cd "$(dirname "$0")/../.." || exit 2
SU2="$PWD"
export SU2_ROOT="$SU2"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
OUT="$SU2/validation_logs/2026-11-27_4.6-Periodic"
STRICT="$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD"
SERIAL="$SU2/build-kokkos-serial/SU2_CFD/src/SU2_CFD"
BIN="$SERIAL" BIN_STRICT="$STRICT" CASE=invper_ausm K=10 \
  bash TestCases/kokkos_nemo/check_noisefloor.sh > "$OUT/nf-per-r3.log" 2>&1
echo "nf-per-r3 rc=$?"
for it in 11 100; do
  for n in prodNO strictNO prodYES; do
    d="$RUN_DIR/nf_invper_ausm_${it}_${n}"
    if [ -d "$d" ]; then
      echo "== $d =="; ls "$d" | head -5
      tail -4 "$d/log.txt" 2>/dev/null
    else
      echo "MISSING $d"
    fi
  done
done
