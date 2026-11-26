#!/bin/bash
# Manually time the thermalbath case (review-16 item 9: wall=0.01s check).
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
D=/home/mpr7366/software/AgenticTest/kk/runs/chk_thermalbath_NO
cd "$D"
t0=$(date +%s.%N)
"$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD" run.cfg > "$D/tb_manual.log" 2>&1
rc=$?
t1=$(date +%s.%N)
echo "exit=$rc wall=$(echo "$t1 - $t0" | bc)s"
grep -c "Exit Success\|Exit Error" "$D/tb_manual.log"
tail -2 "$D/tb_manual.log"
