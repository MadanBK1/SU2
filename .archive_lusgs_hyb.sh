#!/usr/bin/env bash
# 4.8-LUSGSHyb step 9: archive steps 3-8 logs + chk/nf log.txt pairs to validation_logs/
set -eu
cd "$(dirname "$0")"
KK="$HOME/software/AgenticTest/kk"
D=validation_logs/2026-11-26_4.8-LUSGSHyb
cp -f "$KK/logs/g1-lusgs-w1-r1.log"       "$D/g1-lusgs-w1-r1.log"
cp -f "$KK/runs/chk_invwedge_lusgs_lusgsw1_NO/log.txt"   "$D/g1-lusgs-w1-r1__invwedge_lusgs_NO.log"
cp -f "$KK/runs/chk_invwedge_lusgs_lusgsw1_YES/log.txt"  "$D/g1-lusgs-w1-r1__invwedge_lusgs_YES.log"
cp -f "$KK/logs/g1-lusgs-single-r1.log"   "$D/g1-lusgs-single-r1.log"
cp -f "$KK/runs/chk_invwedge_lusgs_lusgssingle_NO/log.txt" "$D/g1-lusgs-single-r1__invwedge_lusgs_NO.log"
cp -f "$KK/runs/chk_invwedge_lusgs_lusgssingle_YES/log.txt" "$D/g1-lusgs-single-r1__invwedge_lusgs_YES.log"
cp -f "$KK/logs/ab-lusgs-w1-r1.log"       "$D/ab-lusgs-w1-r1.log"
cp -f "$KK/runs/chk_invwedge_lusgs_lusgsw1v_NO/log.txt"  "$D/ab-lusgs-w1-r1__invwedge_lusgs_NO.log"
cp -f "$KK/runs/chk_invwedge_lusgs_lusgsw1v_YES/log.txt" "$D/ab-lusgs-w1-r1__invwedge_lusgs_YES.log"
cp -f "$KK/logs/g1-lusgs-visc-w1-r1.log"  "$D/g1-lusgs-visc-w1-r1.log"
cp -f "$KK/runs/chk_super_cat_lusgsw1v_NO/log.txt" "$D/g1-lusgs-visc-w1-r1__super_cat_NO.log"
cp -f "$KK/runs/chk_super_cat_lusgsw1v_YES/log.txt" "$D/g1-lusgs-visc-w1-r1__super_cat_YES.log"
cp -f "$KK/logs/g1-lusgs-regress-r1.log"  "$D/g1-lusgs-regress-r1.log"
cp -f "$KK/runs/chk_invwedge_ausm_NO/log.txt"   "$D/g1-lusgs-regress-r1__invwedge_ausm_NO.log"
cp -f "$KK/runs/chk_invwedge_ausm_YES/log.txt"  "$D/g1-lusgs-regress-r1__invwedge_ausm_YES.log"
cp -f "$KK/runs/chk_thermalbath_NO/log.txt"     "$D/g1-lusgs-regress-r1__thermalbath_NO.log"
cp -f "$KK/runs/chk_thermalbath_YES/log.txt"    "$D/g1-lusgs-regress-r1__thermalbath_YES.log"
cp -f "$KK/runs/chk_super_cat_NO/log.txt"       "$D/g1-lusgs-regress-r1__super_cat_NO.log"
cp -f "$KK/runs/chk_super_cat_YES/log.txt"      "$D/g1-lusgs-regress-r1__super_cat_YES.log"
cp -f "$KK/runs/chk_axi_visccone_NO/log.txt"    "$D/g1-lusgs-regress-r1__axi_visccone_NO.log"
cp -f "$KK/runs/chk_axi_visccone_YES/log.txt"   "$D/g1-lusgs-regress-r1__axi_visccone_YES.log"
cp -f "$KK/runs/chk_invwedge_mg_NO/log.txt"     "$D/g1-lusgs-regress-r1__invwedge_mg_NO.log"
cp -f "$KK/runs/chk_invwedge_mg_YES/log.txt"    "$D/g1-lusgs-regress-r1__invwedge_mg_YES.log"
cp -f "$KK/runs/chk_invwedge_dyn_NO/log.txt"    "$D/g1-lusgs-regress-r1__invwedge_dyn_NO.log"
cp -f "$KK/runs/chk_invwedge_dyn_YES/log.txt"   "$D/g1-lusgs-regress-r1__invwedge_dyn_YES.log"
cp -f "$KK/runs/chk_supin_visc_NO/log.txt"      "$D/g1-lusgs-regress-r1__supin_visc_NO.log"
cp -f "$KK/runs/chk_supin_visc_YES/log.txt"     "$D/g1-lusgs-regress-r1__supin_visc_YES.log"
cp -f "$KK/logs/nf-lusgs-r1.log"        "$D/nf-lusgs-r1.log"
for s in nf_invwedge_lusgs_11_prodNO nf_invwedge_lusgs_11_prodYES nf_invwedge_lusgs_11_strictNO \
         nf_invwedge_lusgs_100_prodNO nf_invwedge_lusgs_100_prodYES nf_invwedge_lusgs_100_strictNO; do
  out="${s#nf_}"; out="${out/invwedge_lusgs_/}"
  cp -f "$KK/runs/$s/log.txt" "$D/nf-lusgs-r1__invwedge_lusgs_$out.log"
done
echo "files in archive:"; ls "$D" | wc -l
du -b "$D"/* | awk '$1>1048576 {print "OVER1MB:", $2, $1}' || true
