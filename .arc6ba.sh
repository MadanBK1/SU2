#!/bin/bash
# Archive review-21 (6b.a-DynGrid) gate logs into validation_logs/.
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
D=$SU2/validation_logs/2026-11-26_6b.a-DynGrid
mkdir -p $D
cp $KK/runs/cpu_dyn_r1/log.txt $D/cpu-dyn-r1.log
for c in invwedge_dyn invwedge_dyn_dynoff invwedge_dyn_dyn1st invwedge_dyn_dyntr super_cat_dynvisc invwedge_dyn_dynpoison; do
  [ -f $KK/runs/chk_${c}_NO/log.txt ] && cp $KK/runs/chk_${c}_NO/log.txt $D/${c}_NO.log
  [ -f $KK/runs/chk_${c}_YES/log.txt ] && cp $KK/runs/chk_${c}_YES/log.txt $D/${c}_YES.log
done
for c in invwedge_ausm thermalbath super_cat axi_visccone; do
  [ -f $KK/runs/chk_${c}_NO/log.txt ] && cp $KK/runs/chk_${c}_NO/log.txt $D/regress1__${c}_NO.log
  [ -f $KK/runs/chk_${c}_YES/log.txt ] && cp $KK/runs/chk_${c}_YES/log.txt $D/regress1__${c}_YES.log
done
for c in invwedge_mg invwedge_lusgs; do
  [ -f $KK/runs/chk_${c}_NO/log.txt ] && cp $KK/runs/chk_${c}_NO/log.txt $D/regress2__${c}_NO.log
  [ -f $KK/runs/chk_${c}_YES/log.txt ] && cp $KK/runs/chk_${c}_YES/log.txt $D/regress2__${c}_YES.log
done
for f in build-dyn-r1.log build-dyn-serial-r1.log cpu-dyn-r1.log g1-dyn-r1.log g1-dynoff-r1.log g1-dyn1st-r1.log g1-dyntr-r1.log g1-dynvisc-r1.log poison-dyn-r1.log g1-dynregress-r1.log g1-dynregress2-r1.log; do
  [ -f $KK/logs/$f ] && cp $KK/logs/$f $D/$f
done
ls -la $D | sed -n '4,100p'
echo ===
du -sh $D
find $D -size +1M
exit 0
