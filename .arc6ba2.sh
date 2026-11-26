#!/bin/bash
# Archive review-22 (6b.a-DynGrid r2) per-case logs into validation_logs/.
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
D=$SU2/validation_logs/2026-11-26_6b.a-DynGrid
mkdir -p $D
# r2 gates (per-case logs still current in kk/runs)
cp $KK/runs/chk_invwedge_dyn_NO/log.txt        $D/g1-dyn-r2__invwedge_dyn_NO.log
cp $KK/runs/chk_invwedge_dyn_YES/log.txt        $D/g1-dyn-r2__invwedge_dyn_YES.log
cp $KK/runs/chk_super_cat_dynvisc_NO/log.txt    $D/g1-dynvisc-r2__super_cat_NO.log
cp $KK/runs/chk_super_cat_dynvisc_YES/log.txt   $D/g1-dynvisc-r2__super_cat_YES.log
cp $KK/runs/chk_super_cat_dynpoisonvisc_NO/log.txt $D/poison-dynvisc-r2__super_cat_dynpoisonvisc_NO.log
cp $KK/runs/chk_super_cat_dynpoisonvisc_YES/log.txt $D/poison-dynvisc-r2__super_cat_dynpoisonvisc_YES.log
for c in invwedge_ausm thermalbath super_cat axi_visccone; do
  cp $KK/runs/chk_${c}_NO/log.txt  $D/g1-dynregress-r2__${c}_NO.log
  cp $KK/runs/chk_${c}_YES/log.txt $D/g1-dynregress-r2__${c}_YES.log
done
for c in invwedge_mg invwedge_lusgs; do
  cp $KK/runs/chk_${c}_NO/log.txt  $D/g1-dynregress2-r2__${c}_NO.log
  cp $KK/runs/chk_${c}_YES/log.txt $D/g1-dynregress2-r2__${c}_YES.log
done
# r1 per-case logs still current in kk/runs (not re-run in r2)
for t in dynoff dyn1st dyntr; do
  cp $KK/runs/chk_invwedge_dyn_${t}_NO/log.txt  $D/g1-dyn${t}-r1__invwedge_dyn_${t}_NO.log
  cp $KK/runs/chk_invwedge_dyn_${t}_YES/log.txt $D/g1-dyn${t}-r1__invwedge_dyn_${t}_YES.log
done
cp $KK/runs/chk_invwedge_dyn_dynpoison_NO/log.txt  $D/poison-dyn-r1__invwedge_dyn_dynpoison_NO.log
cp $KK/runs/chk_invwedge_dyn_dynpoison_YES/log.txt $D/poison-dyn-r1__invwedge_dyn_dynpoison_YES.log
cp $KK/runs/chk_super_cat_dynvisc_sv_NO/log.txt  $D/sv-dynvisc-r1__super_cat_dynvisc_sv_NO.log
cp $KK/runs/chk_super_cat_dynvisc_sv_YES/log.txt $D/sv-dynvisc-r1__super_cat_dynvisc_sv_YES.log
# r1 driver logs not yet archived
for f in sv-dyn-r1.log sv-dyn-inv-r1.log; do
  [ -f $KK/logs/$f ] && cp $KK/logs/$f $D/$f
done
# noise floor r1
[ -f $KK/logs/nf-dyn-r1.log ] && cp $KK/logs/nf-dyn-r1.log $D/nf-dyn-r1.log
echo "=== archive done ==="
ls $D | wc -l
du -sh $D
find $D -size +1M
exit 0
