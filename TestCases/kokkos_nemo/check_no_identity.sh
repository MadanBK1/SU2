#!/usr/bin/env bash
# ENABLE_KOKKOS=NO bit-identity: OLD vs NEW binary, restart files and screen history must be identical.
SU2=${SU2_ROOT:-/home/claude/su2}
MESH=${MESH_DIR:-/home/claude/meshes}; RUN=${RUN_DIR:-/home/claude/runs}
OLD=${OLD:-$SU2/../bin_pre/SU2_CFD_serial_64b77ed}; NEW=${NEW:-$SU2/build-kokkos-serial/SU2_CFD/src/SU2_CFD}
TC=$SU2/TestCases/nonequilibrium
declare -A CFG=( [invwedge_ausm]=$TC/invwedge/invwedge_ausm.cfg [thermalbath]=$TC/thermalbath/finitechemistry/thermalbath.cfg
                 [super_cat]=$TC/visc_wedge/super_cat.cfg [ausmplusm]=$TC/invwedge/invwedge_am.cfg )
st=0
for c in invwedge_ausm thermalbath super_cat ausmplusm; do
  for w in OLD NEW; do
    d=$RUN/noid_${c}_$w; rm -rf $d; mkdir -p $d; cd $d; ln -sf $MESH/*.su2 .
    cp ${CFG[$c]} run.cfg; sed -i '/^ENABLE_KOKKOS/d' run.cfg; printf "\nENABLE_KOKKOS= NO\n" >> run.cfg
    b=$OLD; [ $w = NEW ] && b=$NEW; $b run.cfg > log.txt 2>&1 || st=1
    grep -E '^\|\s+[0-9]+\|' log.txt > hist.txt
  done
  a=$RUN/noid_${c}_OLD; n=$RUN/noid_${c}_NEW
  r1=same; nf=0
  for f in $a/restart_flow*.csv; do nf=$((nf+1)); cmp -s $f $n/$(basename $f) || r1=DIFF; done
  [ $nf -eq 0 ] && r1=MISSING
  r2=$(cmp -s $a/hist.txt $n/hist.txt && echo same || echo DIFF)
  echo "$c: restart($nf files) $r1, history $r2 ($(wc -l < $a/hist.txt) rows)"; [ "$r1$r2" = samesame ] || st=1
done
echo "NO IDENTITY: $([ $st = 0 ] && echo PASS || echo FAIL)"; exit $st
