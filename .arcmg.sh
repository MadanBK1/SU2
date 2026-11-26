#!/bin/bash
# Archive review-16 (4.6-MG step 1) gate logs into validation_logs/.
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
D=$SU2/validation_logs/2026-11-26_4.6-MG
mkdir -p $D
cp $KK/runs/cpu_mg_r73/log.txt $D/cpu-mg-r1.log
for c in invwedge_mg invwedge_mg_mgoff invwedge_mg_mgpost invwedge_mg_mgw invwedge_mg_mgcs super_cat_mg invwedge_mg_mgpoison; do
  cp $KK/runs/chk_${c}_NO/log.txt $D/${c}_NO.log
  cp $KK/runs/chk_${c}_YES/log.txt $D/${c}_YES.log
done
for c in invwedge_ausm thermalbath super_cat axi_visccone; do
  cp $KK/runs/chk_${c}_NO/log.txt $D/regress1__${c}_NO.log
  cp $KK/runs/chk_${c}_YES/log.txt $D/regress1__${c}_YES.log
done
for c in invwedge_lusgs invwedge_roe; do
  cp $KK/runs/chk_${c}_NO/log.txt $D/regress2__${c}_NO.log
  cp $KK/runs/chk_${c}_YES/log.txt $D/regress2__${c}_YES.log
done
for f in g1-mg-r1.log g1-mgoff-r1.log g1-mgpost-r1.log g1-mgw-r1.log g1-mgcs-r1.log g1-mgvisc-r1.log poison-mg-r1.log nf-mg-r1.log g1-regress-r1.log g1-regress2-r1.log; do
  cp $KK/logs/$f $D/$f
done
ls -la $D | sed -n '4,100p'
echo ===
du -sh $D
find $D -size +1M
exit 0
