#!/usr/bin/env bash
# Step 6 A/B (strict build): device-resident mode (device Prepare/Complete/explicit update, device precond build,
# linear-solver handoff, lazy host sync) vs SU2_KOKKOS_NEMO_RESIDENT=0 (same device stages, host-authoritative state,
# host update, host precond build). Restart file and screen history must be bit-identical. The resident run also has
# SU2_KOKKOS_PRECOND_VERIFY=1 (device ILU/Jacobi build vs the host build on the same values, bitwise) and, with
# POISON=1, SU2_KOKKOS_NEMO_POISON=1 (stale host copies are NaN: a missed host sync changes the results).
# Usage: ab_resident.sh [name ...]
BIN=${BIN:-/home/claude/su2/build-kokkos-strict/SU2_CFD/src/SU2_CFD}
POISON=${POISON:-1}
TC=/home/claude/su2/TestCases/nonequilibrium
declare -A BASE MOD
W=$TC/invwedge/invwedge_ausm.cfg; TB=$TC/thermalbath/finitechemistry/thermalbath.cfg
BASE[ausm]=$W;            MOD[ausm]=""
BASE[jacobi]=$W;          MOD[jacobi]="LINEAR_SOLVER_PREC= JACOBI"
BASE[fgmres]=$W;          MOD[fgmres]="LINEAR_SOLVER= FGMRES|LINEAR_SOLVER_ITER= 8"
BASE[ilu1]=$W;            MOD[ilu1]="LINEAR_SOLVER_ILU_FILL_IN= 1"
BASE[vanalbada]=$W;       MOD[vanalbada]="SLOPE_LIMITER_FLOW= VAN_ALBADA_EDGE"
BASE[nolim]=$W;           MOD[nolim]="SLOPE_LIMITER_FLOW= NONE"
BASE[bj_gg]=$W;           MOD[bj_gg]="SLOPE_LIMITER_FLOW= BARTH_JESPERSEN|NUM_METHOD_GRAD= GREEN_GAUSS"
BASE[wang]=$W;            MOD[wang]="SLOPE_LIMITER_FLOW= VENKATAKRISHNAN_WANG|MUSCL_KAPPA_FLOW= 0.3"
BASE[explicit]=$W;        MOD[explicit]="TIME_DISCRE_FLOW= EULER_EXPLICIT|CFL_NUMBER= 0.5"
BASE[first]=$W;           MOD[first]="MUSCL_FLOW= NO"
BASE[air7]=$W;            MOD[air7]="GAS_MODEL= AIR-7|GAS_COMPOSITION= (1e-9, 0.77, 0.229999999, 0.0, 0.0, 0.0, 0.0)"
BASE[maxupd]=$W;          MOD[maxupd]="MAX_UPDATE_FLOW= 0.05"
BASE[output]=$W;          MOD[output]="OUTPUT_WRT_FREQ= 3|OUTPUT_FILES= (RESTART_ASCII, PARAVIEW_ASCII)"
BASE[supersonic]=$W;      MOD[supersonic]="MARKER_FAR= ( Farfield )|MARKER_OUTLET= DELETE|MARKER_SUPERSONIC_INLET= ( Inlet, 288.15, 101325.0, 1700.0, 0.0, 0.0 )|INLET_GAS_COMPOSITION= (0.77, 0.23, 0.0, 0.0, 0.0)|MARKER_SUPERSONIC_OUTLET= ( Exit )"
BASE[subout]=$W;          MOD[subout]="MACH_NUMBER= 0.6|MARKER_OUTLET= ( Exit, 90000.0 )"
BASE[tb]=$TB;             MOD[tb]=""
BASE[tb_impl]=$TB;        MOD[tb_impl]="TIME_DISCRE_FLOW= EULER_IMPLICIT"
BASE[tb_air5]=$TB;        MOD[tb_air5]="GAS_MODEL= AIR-5|GAS_COMPOSITION= (0.6, 0.2, 0.1, 0.05, 0.05)|TIME_DISCRE_FLOW= EULER_IMPLICIT"
BASE[tb_steady]=$TB;      MOD[tb_steady]="TIME_DOMAIN= NO|TIME_MARCHING= DELETE|ITER= 11|TIME_DISCRE_FLOW= EULER_IMPLICIT"
BASE[ausmplusm]=$TC/invwedge/invwedge_am.cfg; MOD[ausmplusm]=""
BASE[supercat]=$TC/visc_wedge/super_cat.cfg;  MOD[supercat]="ITER= 11"
names=("$@"); [ ${#names[@]} -eq 0 ] && names=(ausm jacobi fgmres ilu1 vanalbada nolim bj_gg wang explicit first air7 maxupd output supersonic subout tb tb_impl tb_air5 tb_steady ausmplusm supercat)
st=0
for n in "${names[@]}"; do
  for mode in ON OFF; do
    d=/home/claude/runs/abres/${n}_$mode; rm -rf $d; mkdir -p $d; cd $d; ln -sf /home/claude/meshes/*.su2 .
    cp "${BASE[$n]}" run.cfg; printf "\n" >> run.cfg
    sed -i '/^ENABLE_KOKKOS/d; /^OUTPUT_FILES/d; /^SCREEN_OUTPUT/d' run.cfg
    printf "ENABLE_KOKKOS= YES\nOUTPUT_FILES= (RESTART_ASCII)\n" >> run.cfg
    IFS='|' read -ra kv <<< "${MOD[$n]}"
    for x in "${kv[@]}"; do [ -z "$x" ] && continue; k=${x%%=*}; sed -i "/^$k *=/d" run.cfg; [ "${x#*= }" = DELETE ] || echo "$x" >> run.cfg; done
    grep -qE "^(TIME_)?ITER" run.cfg || echo "ITER= 11" >> run.cfg
    if [ $mode = ON ]; then
      SU2_KOKKOS_PRECOND_VERIFY=1 SU2_KOKKOS_NEMO_POISON=$POISON $BIN run.cfg > log.txt 2>&1; rc=$?
    else
      SU2_KOKKOS_NEMO_RESIDENT=0 $BIN run.cfg > log.txt 2>&1; rc=$?
    fi
    [ $rc -ne 0 ] && { echo "== $n $mode exit=$rc"; tail -3 log.txt; st=1; }
    grep -E '^\|\s+[0-9]+\|' log.txt > hist.txt
  done
  a=/home/claude/runs/abres/${n}_ON; b=/home/claude/runs/abres/${n}_OFF
  res=$(grep -m1 "device-resident state" $a/log.txt | sed 's/.*Step 6): //' | cut -c1-90)
  r1=same; nf=0
  for f in $b/restart_flow*.csv $b/*.vtu; do [ -e "$f" ] || continue; nf=$((nf+1)); cmp -s $f $a/$(basename $f) || r1=DIFF; done
  [ $nf -eq 0 ] && r1=MISSING
  r2=$(cmp -s $a/hist.txt $b/hist.txt && echo same || echo DIFF)
  pv=$(grep "preconditioner device build vs host" $a/log.txt | sed 's/.*: //')
  echo "== $n: resident $res | files($nf) $r1, history $r2 ($(wc -l < $a/hist.txt) rows) | precond A/B: ${pv:-none}"
  grep "steady iteration" $a/log.txt | sed 's/\[KOKKOS_NEMO_LEDGER\] MG 0 /   ON  ledger /'
  grep "steady iteration" $b/log.txt | sed 's/\[KOKKOS_NEMO_LEDGER\] MG 0 /   OFF ledger /'
  [ "$r1$r2" = samesame ] || st=1
  case "$pv" in *non-bitwise=0) ;; "") ;; *) st=1;; esac
done
echo "AB RESIDENT OVERALL: $([ $st = 0 ] && echo PASS || echo FAIL)"; exit $st
