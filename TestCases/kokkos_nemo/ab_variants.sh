#!/usr/bin/env bash
# Strict-build per-stage A/B (SU2_KOKKOS_STAGE_VERIFY=1) over config variants. Usage: ab_variants.sh [name ...]
SU2=${SU2_ROOT:-/home/claude/su2}
M=${MESH_DIR:-/home/claude/meshes}
RUN=${RUN_DIR:-/home/claude/runs}
BIN=${BIN:-$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD}
# Step 10: NP=<n> runs under `mpirun --allow-run-as-root --oversubscribe -np <n>` (MPI build), dirs runs/ab_np<n>/;
#   the VERIFY counts are those of rank 0.
NP=${NP:-}
MPI_OPTS=${MPI_OPTS---mca osc ^rdma}  # see check.sh
MPIRUN=""; [ -n "$NP" ] && MPIRUN="mpirun --allow-run-as-root --oversubscribe $MPI_OPTS -np $NP"
TC=$SU2/TestCases/nonequilibrium
declare -A BASE MOD
BASE[ausm]=$TC/invwedge/invwedge_ausm.cfg;                 MOD[ausm]=""
BASE[ausmplusm]=$TC/invwedge/invwedge_am.cfg;              MOD[ausmplusm]=""
BASE[vanalbada]=$TC/invwedge/invwedge_ausm.cfg;            MOD[vanalbada]="SLOPE_LIMITER_FLOW= VAN_ALBADA_EDGE"
BASE[nolim]=$TC/invwedge/invwedge_ausm.cfg;                MOD[nolim]="SLOPE_LIMITER_FLOW= NONE"
BASE[bj_gg]=$TC/invwedge/invwedge_ausm.cfg;                MOD[bj_gg]="SLOPE_LIMITER_FLOW= BARTH_JESPERSEN|NUM_METHOD_GRAD= GREEN_GAUSS"
BASE[wang]=$TC/invwedge/invwedge_ausm.cfg;                 MOD[wang]="SLOPE_LIMITER_FLOW= VENKATAKRISHNAN_WANG|MUSCL_KAPPA_FLOW= 0.3"
BASE[explicit]=$TC/invwedge/invwedge_ausm.cfg;             MOD[explicit]="TIME_DISCRE_FLOW= EULER_EXPLICIT|CFL_NUMBER= 0.5"
BASE[first]=$TC/invwedge/invwedge_ausm.cfg;                MOD[first]="MUSCL_FLOW= NO"
BASE[air7]=$TC/invwedge/invwedge_ausm.cfg;                 MOD[air7]="GAS_MODEL= AIR-7|GAS_COMPOSITION= (1e-9, 0.77, 0.229999999, 0.0, 0.0, 0.0, 0.0)"
BASE[ns_muscl]=$TC/visc_wedge/super_cat.cfg;               MOD[ns_muscl]="MUSCL_FLOW= YES|SLOPE_LIMITER_FLOW= VENKATAKRISHNAN|ITER= 11"
BASE[supercat]=$TC/visc_wedge/super_cat.cfg;               MOD[supercat]="ITER= 11"
BASE[air7ion]=$TC/invwedge/invwedge_ausm.cfg;              MOD[air7ion]="GAS_MODEL= AIR-7|IONIZATION= YES|GAS_COMPOSITION= (1e-9, 0.77, 0.229999999, 0.0, 0.0, 0.0, 0.0)"
BASE[amimpl]=$TC/invwedge/invwedge_am.cfg;                MOD[amimpl]="TIME_DISCRE_FLOW= EULER_IMPLICIT|CFL_NUMBER= 3"
BASE[up2]=$TC/invwedge/invwedge_ausmplusup2.cfg;           MOD[up2]=""
# Step 7: source terms (chemistry + V-T) and Euler BCs
TB=$TC/thermalbath/finitechemistry/thermalbath.cfg
BASE[tb]=$TB;                                              MOD[tb]=""
BASE[tb_impl]=$TB;                                         MOD[tb_impl]="TIME_DISCRE_FLOW= EULER_IMPLICIT"
BASE[tb_frozen]=$TB;                                       MOD[tb_frozen]="FROZEN_MIXTURE= YES|TIME_DISCRE_FLOW= EULER_IMPLICIT"
BASE[tb_vtlim]=$TB;                                        MOD[tb_vtlim]="VT_RESIDUAL_LIMITING= YES|TIME_DISCRE_FLOW= EULER_IMPLICIT"
BASE[tb_air5]=$TB;                                         MOD[tb_air5]="GAS_MODEL= AIR-5|GAS_COMPOSITION= (0.6, 0.2, 0.1, 0.05, 0.05)|TIME_DISCRE_FLOW= EULER_IMPLICIT"
BASE[supersonic]=$TC/invwedge/invwedge_ausm.cfg;           MOD[supersonic]="MARKER_FAR= ( Farfield )|MARKER_OUTLET= DELETE|MARKER_SUPERSONIC_INLET= ( Inlet, 288.15, 101325.0, 1700.0, 0.0, 0.0 )|INLET_GAS_COMPOSITION= (0.77, 0.23, 0.0, 0.0, 0.0)|MARKER_SUPERSONIC_OUTLET= ( Exit )"
BASE[subout]=$TC/invwedge/invwedge_ausm.cfg;               MOD[subout]="MACH_NUMBER= 0.6|MARKER_OUTLET= ( Exit, 90000.0 )"
BASE[subout_am]=$TC/invwedge/invwedge_am.cfg;              MOD[subout_am]="MACH_NUMBER= 0.6|MARKER_OUTLET= ( Exit, 90000.0 )|TIME_DISCRE_FLOW= EULER_IMPLICIT|CFL_NUMBER= 3"
# Step 8: NEMO Navier-Stokes (device transport, viscous residual, NS walls, viscous far field)
SC=$TC/visc_wedge/super_cat.cfg
BASE[sc_cat]=$SC;  MOD[sc_cat]="CATALYTIC_WALL= (Wall)|ITER= 11"
BASE[sc_partcat]=$SC;  MOD[sc_partcat]="CATALYTIC_WALL= (Wall)|SUPERCATALYTIC_WALL= NO|CATALYTIC_EFFICIENCY= 0.2|TIME_DISCRE_FLOW= EULER_EXPLICIT|CFL_NUMBER= 0.5|ITER= 11"
BASE[sc_gy]=$SC;  MOD[sc_gy]="TRANSPORT_COEFF_MODEL= GUPTA-YOS|ITER= 11"
BASE[sc_suth]=$SC;  MOD[sc_suth]="TRANSPORT_COEFF_MODEL= SUTHERLAND|ITER= 11"
BASE[sc_gg]=$SC;  MOD[sc_gg]="NUM_METHOD_GRAD= GREEN_GAUSS|ITER= 11"
BASE[sc_heatflux]=$SC;  MOD[sc_heatflux]="MARKER_ISOTHERMAL= DELETE|MARKER_HEATFLUX= ( Wall, 0.0 )|ITER= 11"
BASE[sc_explicit]=$SC;  MOD[sc_explicit]="TIME_DISCRE_FLOW= EULER_EXPLICIT|CFL_NUMBER= 0.5|ITER= 11"
BASE[sc_n2]=$SC;  MOD[sc_n2]="GAS_MODEL= N2|GAS_COMPOSITION= (0.99, 0.01)|SUPERCATALYTIC_WALL_COMPOSITION= (0.99, 0.01)|CATALYTIC_WALL= (Wall)|ITER= 11"
BASE[sc_jacobi]=$SC;  MOD[sc_jacobi]="LINEAR_SOLVER_PREC= JACOBI|ITER= 11"
BASE[partial_cat]=$TC/visc_wedge/partial_cat.cfg;  MOD[partial_cat]=""
BASE[axi_visccone]=$TC/visc_wedge/axi_visccone.cfg;  MOD[axi_visccone]=""
W=$TC/invwedge/invwedge_ausm.cfg
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
names=("$@"); [ ${#names[@]} -eq 0 ] && names=(ausm ausmplusm amimpl vanalbada nolim bj_gg wang explicit first air7 air7ion ns_muscl supercat up2 tb tb_impl tb_frozen tb_vtlim tb_air5 supersonic subout subout_am sc_cat sc_partcat sc_gy sc_suth sc_gg sc_heatflux sc_explicit sc_n2 sc_jacobi partial_cat axi_visccone am_impl am_first cfl_ausm cfl_am cfl_sc axi axi_gg axi_inv axi_expl supout_am supin_am supinout_am tb_rk tb_rk4 rk_ausm rk4_am dt2 dt1 dt_expl dt_unstcfl dt_sc)
st=0
for n in "${names[@]}"; do
  d=/home/claude/runs/ab${NP:+_np$NP}/$n; rm -rf $d; mkdir -p $d; cd $d; ln -sf /home/claude/meshes/*.su2 .
  cp "${BASE[$n]}" run.cfg; printf "\n" >> run.cfg
  IFS='|' read -ra kv <<< "${MOD[$n]}"
  for x in "${kv[@]}"; do [ -z "$x" ] && continue; k=${x%%=*}; sed -i "/^$k *=/d" run.cfg; [ "${x#*= }" = DELETE ] || echo "$x" >> run.cfg; done
  sed -i '/^ENABLE_KOKKOS/d; /^OUTPUT_FILES/d' run.cfg; printf "ENABLE_KOKKOS= YES\nOUTPUT_FILES= (RESTART_ASCII)\n" >> run.cfg
  grep -qE "^(TIME_)?ITER" run.cfg || echo "ITER= 11" >> run.cfg
  SU2_KOKKOS_STAGE_VERIFY=1 $MPIRUN $BIN run.cfg > log.txt 2>&1; rc=$?
  echo "== $n exit=$rc $(grep 'stage upwind' log.txt | sed 's/.*stage upwind//')"
  grep -E "stage (source|bc:)" log.txt | sed 's/\[KOKKOS_NEMO\] stage /   /' | sed 's/ (eager download after the stage)//'
  grep -c "Warning!! Instances of NaN" log.txt | sed 's/^/   NaN-warning lines: /'
  grep -m1 "subsonic outlet" log.txt | sed 's/\[KOKKOS_NEMO_PERF\] MG 0 /   /'
  grep "KOKKOS_NEMO_VERIFY\] MG 0 [A-Za-z]* compares" log.txt | sed 's/\[KOKKOS_NEMO_VERIFY\] MG 0 /   /'
  grep -m1 "reconstructed states" log.txt | sed 's/^/   /'
  grep -E "stage (viscous|NS transport)|device-resident" log.txt | sed 's/\[KOKKOS_NEMO\] /   /' | cut -c1-110
  [ $rc -ne 0 ] && { tail -3 log.txt; st=1; }
  grep "KOKKOS_NEMO_VERIFY\] MG 0 [A-Za-z]* compares" log.txt | grep -qv "non-bitwise=0 " && st=1
done
echo "AB OVERALL: $([ $st = 0 ] && echo PASS || echo FAIL)"; exit $st
