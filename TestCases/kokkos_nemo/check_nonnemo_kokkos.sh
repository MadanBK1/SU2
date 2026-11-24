#!/usr/bin/env bash
# Non-NEMO Kokkos linear-solver smoke test: ENABLE_KOKKOS=YES on the QuickStart inviscid NACA0012 (CEulerSolver, the
# mesh is in the repository), OLD vs NEW binary, restart file and screen history must be bit-identical.
# Variants: FGMRES + ILU (cfg default) and BCGSTAB + JACOBI. Env: OLD, NEW, ITERS (default 20).
SU2=${SU2_ROOT:-/home/claude/su2}
RUN=${RUN_DIR:-/home/claude/runs}
OLD=${OLD:-/home/claude/bin_pre/SU2_CFD_serial_e1aafef}; NEW=${NEW:-$SU2/build-kokkos-serial/SU2_CFD/src/SU2_CFD}
ITERS=${ITERS:-20}; Q=$SU2/QuickStart
declare -A MOD=( [fgmres_ilu]="" [bcgstab_jacobi]="LINEAR_SOLVER= BCGSTAB|LINEAR_SOLVER_PREC= JACOBI" )
st=0
for v in fgmres_ilu bcgstab_jacobi; do
  for w in OLD NEW; do
    d=$RUN/nonnemo_${v}_$w; rm -rf $d; mkdir -p $d; cd $d; ln -sf $Q/mesh_NACA0012_inv.su2 .
    cp $Q/inv_NACA0012.cfg run.cfg; printf "\n" >> run.cfg
    IFS='|' read -ra kv <<< "${MOD[$v]}"
    for x in "${kv[@]}"; do [ -z "$x" ] && continue; k=${x%%=*}; sed -i "/^$k *=/d" run.cfg; echo "$x" >> run.cfg; done
    sed -i '/^ENABLE_KOKKOS/d; /^OUTPUT_FILES/d; /^ITER=/d; /^SCREEN_OUTPUT/d' run.cfg
    printf "ENABLE_KOKKOS= YES\nOUTPUT_FILES= (RESTART_ASCII)\nITER= %s\nSCREEN_OUTPUT= (INNER_ITER, RMS_RES, LIFT, DRAG)\n" $ITERS >> run.cfg
    b=$OLD; [ $w = NEW ] && b=$NEW; $b run.cfg > log.txt 2>&1 || { st=1; tail -3 log.txt; }
    grep -E '^\|\s+[0-9]+\|' log.txt > hist.txt
  done
  a=$RUN/nonnemo_${v}_OLD; n=$RUN/nonnemo_${v}_NEW
  r1=$(cmp -s $a/restart_flow.csv $n/restart_flow.csv && echo same || echo DIFF)
  r2=$(cmp -s $a/hist.txt $n/hist.txt && echo same || echo DIFF)
  echo "$v: restart $r1, history $r2 ($(wc -l < $a/hist.txt) rows, last: $(tail -1 $n/hist.txt | tr -s ' '))"
  [ "$r1$r2" = samesame ] || st=1
done
echo "NON-NEMO KOKKOS IDENTITY: $([ $st = 0 ] && echo PASS || echo FAIL)"; exit $st
