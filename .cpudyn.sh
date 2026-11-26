#!/bin/bash
# review-21 step 2 (6b.a-DynGrid): CPU baseline — ENABLE_KOKKOS=NO run of invwedge_dyn.cfg
# must run all 4 time iterations without error.
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
BIN=$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD
d=$KK/runs/cpu_dyn_r1; rm -rf "$d"; mkdir -p "$d"; cd "$d"
cp $SU2/TestCases/nonequilibrium/invwedge/invwedge_dyn.cfg run.cfg
ln -sf $KK/meshes/*.su2 .
sed -i '/^ENABLE_KOKKOS/d' run.cfg; printf "\nENABLE_KOKKOS= NO\n" >> run.cfg
"$BIN" run.cfg > log.txt 2>&1
echo "exit=$?"
tail -12 log.txt
