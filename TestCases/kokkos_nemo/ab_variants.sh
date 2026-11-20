#!/usr/bin/env bash
# Strict-build per-stage A/B (SU2_KOKKOS_STAGE_VERIFY=1) over config variants. Usage: ab_variants.sh [name ...]
BIN=${BIN:-/home/claude/su2/build-kokkos-strict/SU2_CFD/src/SU2_CFD}
TC=/home/claude/su2/TestCases/nonequilibrium
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
names=("$@"); [ ${#names[@]} -eq 0 ] && names=(ausm ausmplusm amimpl vanalbada nolim bj_gg wang explicit first air7 air7ion ns_muscl supercat up2 tb tb_impl tb_frozen tb_vtlim tb_air5 supersonic subout subout_am sc_cat sc_partcat sc_gy sc_suth sc_gg sc_heatflux sc_explicit sc_n2 sc_jacobi partial_cat axi_visccone)
st=0
for n in "${names[@]}"; do
  d=/home/claude/runs/ab/$n; rm -rf $d; mkdir -p $d; cd $d; ln -sf /home/claude/meshes/*.su2 .
  cp "${BASE[$n]}" run.cfg; printf "\n" >> run.cfg
  IFS='|' read -ra kv <<< "${MOD[$n]}"
  for x in "${kv[@]}"; do [ -z "$x" ] && continue; k=${x%%=*}; sed -i "/^$k *=/d" run.cfg; [ "${x#*= }" = DELETE ] || echo "$x" >> run.cfg; done
  sed -i '/^ENABLE_KOKKOS/d; /^OUTPUT_FILES/d' run.cfg; printf "ENABLE_KOKKOS= YES\nOUTPUT_FILES= (RESTART_ASCII)\n" >> run.cfg
  grep -qE "^(TIME_)?ITER" run.cfg || echo "ITER= 11" >> run.cfg
  SU2_KOKKOS_STAGE_VERIFY=1 $BIN run.cfg > log.txt 2>&1; rc=$?
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
