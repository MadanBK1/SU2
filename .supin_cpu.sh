#!/bin/bash
# CPU sanity (step 2): pure-CPU (ENABLE_KOKKOS=NO) supin_visc, once.
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
LOG=$1
D=$KK/runs/supin_cpu_sanity
rm -rf "$D"; mkdir -p "$D"; cd "$D"
cp $SU2/TestCases/nonequilibrium/visc_wedge/supin_visc.cfg run.cfg
cp $KK/meshes/viscwedge.su2 .
sed -i '/^ENABLE_KOKKOS/d' run.cfg
printf "\nENABLE_KOKKOS= NO\n" >> run.cfg
t0=$(date +%s)
$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD run.cfg > log.txt 2>&1
rc=$?
t1=$(date +%s)
echo "cpu_supin_visc exit=$rc wall=$((t1-t0))s"
echo "--- tail ---"; tail -25 log.txt
echo "--- scan ---"; grep -niE "nan|not operational|abort|error|segmentation|invalid" log.txt | head
echo "--- device-resident line (should be absent/NO) ---"; grep -n "device-resident" log.txt | head
exit $rc
