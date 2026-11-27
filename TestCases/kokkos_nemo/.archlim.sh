#!/usr/bin/env bash
# Archive 4.6-Limiters step 5 (regress) + step 6 (noise floor) logs into validation_logs
set -eu
cd "$(dirname "$0")/../.."
R=/home/mpr7366/software/AgenticTest/kk/runs
L=/home/mpr7366/software/AgenticTest/kk/logs
D=validation_logs/2026-11-27_4.6-Limiters
for c in invwedge_ausm thermalbath super_cat axi_visccone invwedge_lusgs invwedge_mg invwedge_dyn supin_visc; do
  for s in NO YES; do cp "$R/chk_${c}_${s}/log.txt" "$D/g1-lim-regress-r1__${c}_${s}.log"; done
done
for it in 11 100; do
  for s in prodNO prodYES strictNO; do cp "$R/nf_invwedge_ausm_${it}_${s}/log.txt" "$D/nf-lim-r1__invwedge_ausm_${it}_${s}.log"; done
done
cp "$L/g1-lim-regress-r1.log" "$D/g1-lim-regress-r1.log"
cp "$L/nf-lim-r1.log" "$D/nf-lim-r1.log"
ls "$D" | wc -l
