#!/bin/bash
# CPU sanity (review-12 step 2): NO (host) side of smol_maxwell only, strict binary.
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
BIN=$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD
LOG=${1:-/home/mpr7366/software/AgenticTest/kk/logs/cpu-smol-r55.log}
d=$KK/runs/cpusmol_r55; rm -rf "$d"; mkdir -p "$d"; cd "$d"
cp $SU2/TestCases/nonequilibrium/visc_wedge/smol_maxwell.cfg run.cfg
ln -sf $KK/meshes/*.su2 .
printf "\nENABLE_KOKKOS= NO\n" >> run.cfg
t0=$(date +%s)
"$BIN" run.cfg > log.txt 2>&1
rc=$?
w=$(( $(date +%s) - t0 ))
{ echo "smol_maxwell NO exit=$rc wall=${w}s"; tail -14 log.txt; } > "$LOG" 2>&1 || true
tail -14 "$LOG"
exit $rc
