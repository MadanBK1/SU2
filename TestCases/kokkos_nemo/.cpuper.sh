#!/usr/bin/env bash
# 4.6-Periodic step 2: sanity CPU-only run of invper_ausm (ENABLE_KOKKOS=NO, 21 iters).
# Verifies the periodic preprocessing finds matching vertices (translation sign).
set -u
cd "$(dirname "$0")/../.." || exit 2
SU2="$PWD"
RUN="$HOME/software/AgenticTest/kk/runs"
LOGS="$HOME/software/AgenticTest/kk/logs"
mkdir -p "$LOGS" "$RUN"
d=$RUN/cpu_per_r1; rm -rf "$d"; mkdir -p "$d"; cd "$d"
cp "$SU2/TestCases/kokkos_nemo/meshes/invper_ausm.cfg" run.cfg
ln -sf "$SU2/TestCases/kokkos_nemo/meshes/"*.su2 .
sed -i '/^ENABLE_KOKKOS/d' run.cfg; printf "\nENABLE_KOKKOS= NO\n" >> run.cfg
t0=$(date +%s)
timeout 900 "$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD" run.cfg > log.txt 2>&1
rc=$?
echo "rc=$rc wall=$(( $(date +%s) - t0 ))s"
echo "--- residual rows: $(grep -cE '^\|[[:space:]]+[0-9]+\|' log.txt) ---"
tail -8 log.txt
grep -n "PERIODIC\|periodic" log.txt | head -5
