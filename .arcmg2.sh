#!/bin/bash
# Archive review-20 (4.6-MG2) gate logs into validation_logs/.
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
D=$SU2/validation_logs/2026-11-26_4.6-MG2
mkdir -p $D
for c in invwedge_mg invwedge_mg_mg2off invwedge_mg_mg2post invwedge_mg_mg2w invwedge_mg_mg2cs super_cat_mg2visc invwedge_mg_mg2poison; do
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
for f in g1-mg2-r1.log g1-mg2off-r1.log g1-mg2post-r1.log g1-mg2w-r1.log g1-mg2cs-r1.log g1-mg2visc-r1.log poison-mg2-r1.log nf-mg2-r1.log g1-mg2regress-r1.log g1-mg2regress2-r1.log; do
  cp $KK/logs/$f $D/$f
done
ls -la $D | sed -n '4,100p'
echo ===
du -sh $D
find $D -size +1M
exit 0
