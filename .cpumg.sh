#!/bin/bash
# round 73 (review 16, step 2): CPU baseline — ENABLE_KOKKOS=NO run of invwedge_mg.cfg
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
BIN=$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD
d=$KK/runs/cpu_mg_r73; rm -rf "$d"; mkdir -p "$d"; cd "$d"
cp $SU2/TestCases/nonequilibrium/invwedge/invwedge_mg.cfg run.cfg
ln -sf $KK/meshes/*.su2 .
sed -i '/^ENABLE_KOKKOS/d' run.cfg; printf "\nENABLE_KOKKOS= NO\n" >> run.cfg
"$BIN" run.cfg > log.txt 2>&1
echo "exit=$?"
tail -12 log.txt
