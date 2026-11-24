#!/usr/bin/env bash
# Step 6 A/B (strict build): device-resident mode (device Prepare/Complete/explicit update, device precond build,
# linear-solver handoff, lazy host sync) vs SU2_KOKKOS_NEMO_RESIDENT=0 (same device stages, host-authoritative state,
# host update, host precond build). Restart file and screen history must be bit-identical. The resident run also has
# SU2_KOKKOS_PRECOND_VERIFY=1 (device ILU/Jacobi build vs the host build on the same values, bitwise) and, with
# POISON=1, SU2_KOKKOS_NEMO_POISON=1 (stale host copies are NaN: a missed host sync changes the results).
# Usage: ab_resident.sh [name ...]
# Step 10: NP=<n> runs both modes under `mpirun --allow-run-as-root --oversubscribe -np <n>` (MPI build, e.g.
#   BIN=.../build-kokkos-strict-mpi/...), run directories /home/claude/runs/abres_np<n>/.
BIN=${BIN:-/home/claude/su2/build-kokkos-strict/SU2_CFD/src/SU2_CFD}
NP=${NP:-}
MPI_OPTS=${MPI_OPTS---mca osc ^rdma}  # see check.sh
MPIRUN=""; [ -n "$NP" ] && MPIRUN="mpirun --allow-run-as-root --oversubscribe $MPI_OPTS -np $NP"
R=/home/claude/runs/abres${NP:+_np$NP}
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
# Step 8: NEMO Navier-Stokes (forces / heat flux in the screen history exercise the BOUNDARY host sync)
SC=$TC/visc_wedge/super_cat.cfg
SO="SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY_0, RMS_DENSITY_1, RMS_ENERGY, RMS_ENERGY_VE, LIFT, DRAG, TOTAL_HEATFLUX)"
BASE[sc_cat]=$SC;  MOD[sc_cat]="CATALYTIC_WALL= (Wall)|ITER= 11|$SO"
BASE[sc_partcat]=$SC;  MOD[sc_partcat]="CATALYTIC_WALL= (Wall)|SUPERCATALYTIC_WALL= NO|CATALYTIC_EFFICIENCY= 0.2|TIME_DISCRE_FLOW= EULER_EXPLICIT|CFL_NUMBER= 0.5|ITER= 11|$SO"
BASE[sc_gy]=$SC;  MOD[sc_gy]="TRANSPORT_COEFF_MODEL= GUPTA-YOS|ITER= 11|$SO"
BASE[sc_suth]=$SC;  MOD[sc_suth]="TRANSPORT_COEFF_MODEL= SUTHERLAND|ITER= 11|$SO"
BASE[sc_gg]=$SC;  MOD[sc_gg]="NUM_METHOD_GRAD= GREEN_GAUSS|ITER= 11|$SO"
BASE[sc_heatflux]=$SC;  MOD[sc_heatflux]="MARKER_ISOTHERMAL= DELETE|MARKER_HEATFLUX= ( Wall, 0.0 )|ITER= 11|$SO"
BASE[sc_explicit]=$SC;  MOD[sc_explicit]="TIME_DISCRE_FLOW= EULER_EXPLICIT|CFL_NUMBER= 0.5|ITER= 11|$SO"
BASE[sc_n2]=$SC;  MOD[sc_n2]="GAS_MODEL= N2|GAS_COMPOSITION= (0.99, 0.01)|SUPERCATALYTIC_WALL_COMPOSITION= (0.99, 0.01)|CATALYTIC_WALL= (Wall)|ITER= 11|$SO"
BASE[sc_jacobi]=$SC;  MOD[sc_jacobi]="LINEAR_SOLVER_PREC= JACOBI|ITER= 11|$SO"
BASE[sc_forces]=$SC;  MOD[sc_forces]="ITER= 11|$SO"
BASE[sc_output]=$SC;  MOD[sc_output]="ITER= 7|OUTPUT_WRT_FREQ= 3|OUTPUT_FILES= (RESTART_ASCII, PARAVIEW_ASCII)|$SO"
BASE[sc_muscl]=$SC;  MOD[sc_muscl]="MUSCL_FLOW= YES|SLOPE_LIMITER_FLOW= VENKATAKRISHNAN|ITER= 11|$SO"
# Batch A: AUSM+M sensor, CFL adaptation, axisymmetry, AUSM+M supersonic BCs, RK, dual time
AM=$TC/invwedge/invwedge_am.cfg; AXI=$TC/visc_wedge/axi_visccone.cfg
CFLP="CFL_ADAPT= YES|CFL_ADAPT_PARAM= ( 0.5, 1.5, 1.0, 20.0, 0.9, 2 )|SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY_0, RMS_DENSITY_1, RMS_ENERGY, AVG_CFL, MIN_CFL, MAX_CFL)"
BASE[am_impl]=$AM;        MOD[am_impl]="TIME_DISCRE_FLOW= EULER_IMPLICIT|CFL_NUMBER= 3"
BASE[am_first]=$AM;       MOD[am_first]="MUSCL_FLOW= NO"
BASE[cfl_ausm]=$W;        MOD[cfl_ausm]="$CFLP|ITER= 21"
BASE[cfl_am]=$AM;         MOD[cfl_am]="TIME_DISCRE_FLOW= EULER_IMPLICIT|CFL_NUMBER= 3|$CFLP|ITER= 21"
BASE[cfl_sc]=$TC/visc_wedge/super_cat.cfg; MOD[cfl_sc]="$CFLP|ITER= 21"
BASE[axi]=$AXI;           MOD[axi]=""
BASE[axi_gg]=$AXI;        MOD[axi_gg]="NUM_METHOD_GRAD= GREEN_GAUSS"
BASE[axi_inv]=$W;         MOD[axi_inv]="AXISYMMETRIC= YES"
BASE[axi_expl]=$AXI;      MOD[axi_expl]="TIME_DISCRE_FLOW= EULER_EXPLICIT|CFL_NUMBER= 0.5"
BASE[supout_am]=$AM;      MOD[supout_am]="MARKER_OUTLET= DELETE|MARKER_SUPERSONIC_OUTLET= ( Exit )"
BASE[supin_am]=$AM;       MOD[supin_am]="MARKER_FAR= ( Inlet )|MARKER_SUPERSONIC_INLET= ( Farfield, 288.15, 101325.0, 1700.0, 0.0, 0.0 )|INLET_GAS_COMPOSITION= (0.77, 0.23, 0.0, 0.0, 0.0)"
BASE[supinout_am]=$AM;    MOD[supinout_am]="TIME_DISCRE_FLOW= EULER_IMPLICIT|CFL_NUMBER= 3|MARKER_FAR= ( Inlet )|MARKER_OUTLET= DELETE|MARKER_SUPERSONIC_OUTLET= ( Exit )|MARKER_SUPERSONIC_INLET= ( Farfield, 288.15, 101325.0, 1700.0, 0.0, 0.0 )|INLET_GAS_COMPOSITION= (0.77, 0.23, 0.0, 0.0, 0.0)"
BASE[tb_rk]=$TB;          MOD[tb_rk]="TIME_DISCRE_FLOW= RUNGE-KUTTA_EXPLICIT"
BASE[tb_rk4]=$TB;         MOD[tb_rk4]="TIME_DISCRE_FLOW= CLASSICAL_RK4_EXPLICIT"
BASE[rk_ausm]=$W;         MOD[rk_ausm]="TIME_DISCRE_FLOW= RUNGE-KUTTA_EXPLICIT|CFL_NUMBER= 0.5"
BASE[rk4_am]=$AM;         MOD[rk4_am]="TIME_DISCRE_FLOW= CLASSICAL_RK4_EXPLICIT"
DT="TIME_DOMAIN= YES|TIME_STEP= 2e-7|TIME_ITER= 3|INNER_ITER= 4|ITER= DELETE|OUTPUT_WRT_FREQ= 1"
BASE[dt2]=$W;             MOD[dt2]="TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER|$DT"
BASE[dt1]=$W;             MOD[dt1]="TIME_MARCHING= DUAL_TIME_STEPPING-1ST_ORDER|$DT"
BASE[dt_expl]=$W;         MOD[dt_expl]="TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER|TIME_DISCRE_FLOW= EULER_EXPLICIT|CFL_NUMBER= 0.5|$DT"
BASE[dt_unstcfl]=$W;      MOD[dt_unstcfl]="TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER|UNST_CFL_NUMBER= 50|$DT"
BASE[dt_sc]=$TC/visc_wedge/super_cat.cfg; MOD[dt_sc]="TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER|$DT"
names=("$@"); [ ${#names[@]} -eq 0 ] && names=(ausm jacobi fgmres ilu1 vanalbada nolim bj_gg wang explicit first air7 maxupd output supersonic subout tb tb_impl tb_air5 tb_steady ausmplusm supercat sc_forces sc_output sc_muscl sc_cat sc_partcat sc_gy sc_suth sc_gg sc_heatflux sc_explicit sc_n2 sc_jacobi am_impl am_first cfl_ausm cfl_am cfl_sc axi axi_gg axi_inv axi_expl supout_am supin_am supinout_am tb_rk tb_rk4 rk_ausm rk4_am dt2 dt1 dt_expl dt_unstcfl dt_sc)
st=0
for n in "${names[@]}"; do
  for mode in ON OFF; do
    d=$R/${n}_$mode; rm -rf $d; mkdir -p $d; cd $d; ln -sf /home/claude/meshes/*.su2 .
    cp "${BASE[$n]}" run.cfg; printf "\n" >> run.cfg
    sed -i '/^ENABLE_KOKKOS/d; /^OUTPUT_FILES/d; /^SCREEN_OUTPUT/d' run.cfg
    printf "ENABLE_KOKKOS= YES\nOUTPUT_FILES= (RESTART_ASCII)\n" >> run.cfg
    IFS='|' read -ra kv <<< "${MOD[$n]}"
    for x in "${kv[@]}"; do [ -z "$x" ] && continue; k=${x%%=*}; sed -i "/^$k *=/d" run.cfg; [ "${x#*= }" = DELETE ] || echo "$x" >> run.cfg; done
    grep -qE "^(TIME_)?ITER" run.cfg || echo "ITER= 11" >> run.cfg
    if [ $mode = ON ]; then
      SU2_KOKKOS_PRECOND_VERIFY=1 SU2_KOKKOS_NEMO_POISON=$POISON $MPIRUN $BIN run.cfg > log.txt 2>&1; rc=$?
    else
      SU2_KOKKOS_NEMO_RESIDENT=0 $MPIRUN $BIN run.cfg > log.txt 2>&1; rc=$?
    fi
    [ $rc -ne 0 ] && { echo "== $n $mode exit=$rc"; tail -3 log.txt; st=1; }
    grep -E '^\|\s+[0-9]+\|' log.txt > hist.txt
  done
  a=$R/${n}_ON; b=$R/${n}_OFF
  res=$(grep -m1 "device-resident state" $a/log.txt | sed 's/.*Step 6): //' | cut -c1-90)
  r1=same; nf=0
  for f in $b/restart_flow*.csv $b/*.vtu $b/*.vtk; do [ -e "$f" ] || continue; nf=$((nf+1)); cmp -s $f $a/$(basename $f) || r1=DIFF; done
  [ $nf -eq 0 ] && r1=MISSING
  r2=$(cmp -s $a/hist.txt $b/hist.txt && echo same || echo DIFF)
  pv=$(grep "preconditioner device build vs host" $a/log.txt | sed 's/.*: //')
  echo "== $n: resident $res | files($nf) $r1, history $r2 ($(wc -l < $a/hist.txt) rows) | precond A/B: ${pv:-none}"
  grep "steady iteration" $a/log.txt | grep -v " rank " | sed 's/\[KOKKOS_NEMO_LEDGER\] MG 0 /   ON  ledger /'
  grep "steady iteration" $b/log.txt | grep -v " rank " | sed 's/\[KOKKOS_NEMO_LEDGER\] MG 0 /   OFF ledger /'
  [ -n "$NP" ] && grep " rank .*steady iteration" $a/log.txt | sed 's/.*MG 0 /   ON  /'
  [ "$r1$r2" = samesame ] || st=1
  case "$pv" in *non-bitwise=0) ;; "") ;; *) st=1;; esac
done
echo "AB RESIDENT OVERALL: $([ $st = 0 ] && echo PASS || echo FAIL)"; exit $st
