#!/bin/bash
# CPU-only sanity driver: strict binary, ENABLE_KOKKOS=NO side only.
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
LOG=$1; shift
CASES="$*"
declare -A CFG=( [supin_visc]=$SU2/TestCases/nonequilibrium/visc_wedge/supin_visc.cfg
                 [smol_maxwell]=$SU2/TestCases/nonequilibrium/visc_wedge/smol_maxwell.cfg
                 [super_cat]=$SU2/TestCases/nonequilibrium/visc_wedge/super_cat.cfg
                 [invwedge_dyn]=$SU2/TestCases/nonequilibrium/invwedge/invwedge_dyn.cfg )
c=$1
d=$KK/runs/cpunosh_${c}; rm -rf "$d"; mkdir -p "$d"; cd "$d"
cp "${CFG[$c]}" run.cfg; ln -sf $KK/meshes/*.su2 .
sed -i '/^ENABLE_KOKKOS/d' run.cfg; printf "\nENABLE_KOKKOS= NO\n" >> run.cfg
SU2_ROOT=$SU2 MESH_DIR=$KK/meshes RUN_DIR=$d \
  timeout 900 $SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD run.cfg > $LOG 2>&1
rc=$?
echo "cpnosh $c exit=$rc"
tail -8 "$LOG"
exit $rc
