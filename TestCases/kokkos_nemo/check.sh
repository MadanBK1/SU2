#!/usr/bin/env bash
# Regression check: run NEMO cases with ENABLE_KOKKOS=NO and YES on the Kokkos
# Serial build, compare final screen residuals and restart fields.
# Usage: check.sh [case ...]   (cases: invwedge_ausm thermalbath super_cat partial_cat axi_visccone invwedge_am invwedge_lax invwedge_roe;
#   default: invwedge_ausm thermalbath super_cat axi_visccone (batch A: axisymmetric source on the device))
# Exit 0 only if every case: both runs exit 0, YES residual row == NO residual row
# to within RES_TOL (abs, log10 units), and max field |diff|/max|field| <= FIELD_TOL.
# REF_MODE=NO (default): reference run is ENABLE_KOKKOS=NO (differs from YES also by the Kokkos linear solver,
#   whose reductions are ordered differently -> ~1e-10 fields on invwedge even in a strict IEEE build).
# REF_MODE=HOST: reference run is ENABLE_KOKKOS=YES with every NEMO stage forced to the host
#   (SU2_KOKKOS_NEMO_HOST=all; same Kokkos linear solver), isolating the device NEMO stages. Strict-build
#   exactness gate: REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 (bitwise restart fields).
# PHYS_SCALE=1: species partial densities are scaled by max mixture density, momenta by max sqrt(rho*rhoE)
#   (zero-velocity / trace-species columns are otherwise pure round-off relative to themselves).
# RES_NOISE_ORDERS=N: residual columns more than N orders below the row maximum (in NO and YES) are round-off
#   (e.g. rms[RhoU] of the 0-D thermal bath) and are not compared.
# Step 10 (MPI builds):
# NP=<n>: run both binaries under `mpirun --allow-run-as-root --oversubscribe $MPI_OPTS -np <n>` (default: no mpirun;
#   MPI_OPTS default "--mca osc ^rdma").
# REF_MODE=YES: the reference run is ENABLE_KOKKOS=YES device as well (use with REF_BIN / REF_NP: compare binaries).
# REF_BIN=<path>, REF_NP=<n>: binary / rank count of the reference ("NO") run (default: BIN, NP; REF_NP=0: no mpirun).
# RUN_TAG=<s>: suffix of the run directories (default: _np<NP> if NP is set, else empty).
# EXTRA_CFG="KEY= VALUE;KEY2= VALUE2": options appended to both configs (e.g. KOKKOS_GPU_AWARE_MPI= YES).
# Extra environment (e.g. SU2_KOKKOS_NEMO_POISON=1) is inherited by both runs.
set -u
SU2=${SU2_ROOT:-/home/claude/su2}
BIN=${BIN:-$SU2/build-kokkos-serial/SU2_CFD/src/SU2_CFD}
RES_TOL=${RES_TOL:-2e-6}
FIELD_TOL=${FIELD_TOL:-1e-9}
REF_MODE=${REF_MODE:-NO}
PHYS_SCALE=${PHYS_SCALE:-0}            # 1: field diffs scaled physically (see below) instead of per-column max|field|
RES_NOISE_ORDERS=${RES_NOISE_ORDERS:-} # e.g. 10: ignore residual columns > N orders below the row maximum (round-off)
NP=${NP:-}
REF_BIN=${REF_BIN:-$BIN}
REF_NP=${REF_NP:-$NP}
RUN_TAG=${RUN_TAG-${NP:+_np$NP}}
MPI_OPTS=${MPI_OPTS---mca osc ^rdma}  # OpenMPI 4.1 in this container: the rdma one-sided component fails MPI_Win_create at -np 1
mpiprefix() { [ -n "$1" ] && [ "$1" != 0 ] && echo "mpirun --allow-run-as-root --oversubscribe $MPI_OPTS -np $1"; }
TC=$SU2/TestCases/nonequilibrium
M=${MESH_DIR:-/home/claude/meshes}
RUN=${RUN_DIR:-/home/claude/runs}
declare -A CFG=( [invwedge_ausm]=$TC/invwedge/invwedge_ausm.cfg
                 [thermalbath]=$TC/thermalbath/finitechemistry/thermalbath.cfg
                 [super_cat]=$TC/visc_wedge/super_cat.cfg
                 [partial_cat]=$TC/visc_wedge/partial_cat.cfg
                 [axi_visccone]=$TC/visc_wedge/axi_visccone.cfg
                 [invwedge_am]=$TC/invwedge/invwedge_am.cfg
                 [invwedge_lax]=$TC/invwedge/invwedge_lax.cfg
                 [invwedge_roe]=$TC/invwedge/invwedge_roe.cfg
                 [invwedge_msw]=$TC/invwedge/invwedge_msw.cfg
                 [invwedge_ausmplusup2]=$TC/invwedge/invwedge_ausmplusup2.cfg
                 [smol_maxwell]=$TC/visc_wedge/smol_maxwell.cfg
                 [supin_visc]=$TC/visc_wedge/supin_visc.cfg
                 [invwedge_lusgs]=$TC/invwedge/invwedge_lusgs.cfg
                 [invwedge_mg]=$TC/invwedge/invwedge_mg.cfg
                 [invwedge_dyn]=$TC/invwedge/invwedge_dyn.cfg )
declare -A REF=( [invwedge_ausm]="-1.073693 -1.598456 -18.298997 -18.626405 -18.572419 2.241766 1.868557 5.286079 0.843747" )
# Stage line the YES run must print (guards against a silent host fallback passing G1 trivially).
# Several required lines are separated by '|'.
declare -A EXPECT=( [invwedge_ausm]="[KOKKOS] ILU(0) device apply:"  # 4.8-ILUApplyHyb: hybrid (default) or single-team (SU2_KOKKOS_ILU_APPLY=single)
                    [invwedge_ausm_part]="[KOKKOS] ILU partitioned (LINEAR_SOLVER_PREC_THREADS=|[KOKKOS] ILU(0) device apply:"  # 4.8-ILUPart: RUN_TAG=_part EXTRA_CFG="LINEAR_SOLVER_PREC_THREADS= 200"
                    [invwedge_lax]="stage centered residual (LAX-FRIEDRICH): device|stage bc:FAR_FIELD (Inlet): device|stage bc:OUTLET_FLOW (Exit): device|stage bc:EULER_WALL (Wall): device"
                    [invwedge_roe]="stage upwind (ROE + MUSCL, residual + Jacobian): device|stage bc:FAR_FIELD (Inlet): device|stage bc:OUTLET_FLOW (Exit): device|stage bc:EULER_WALL (Wall): device"
                    [invwedge_msw]="stage upwind (MSW + MUSCL, residual + Jacobian): device|stage bc:FAR_FIELD (Inlet): device|stage bc:OUTLET_FLOW (Exit): device|stage bc:EULER_WALL (Wall): device"
                    [invwedge_ausmplusup2]="stage upwind (AUSM+-up2, residual): device|stage bc:FAR_FIELD (Inlet): device|stage bc:OUTLET_FLOW (Exit): device|stage bc:EULER_WALL (Wall): device"
                    [smol_maxwell]="stage bc:SMOLUCHOWSKI_MAXWELL (Wall): device"
                    [supin_visc]="stage bc:SUPERSONIC_INLET (Inlet): device|stage bc:FAR_FIELD (Farfield) [+ viscous]: device|stage bc:ISOTHERMAL (Wall) [non-catalytic]: device"
                    [invwedge_lusgs]="stage linear preconditioner (LU_SGS apply): device|device-resident state (Step 6): ON"
                    [invwedge_mg]="device-resident state (Step 6): ON|MG level 1 device-resident state: ON|MG level 2 device-resident state: ON|MG device transfers="
                    [invwedge_dyn]="stage dynamic grid (grid-velocity dt/eigenvalue terms + GCL, metric re-upload per time step): device|stage time step (spectral radii + local dt + min/max reduce): device|stage dual-time source (2nd order): device|device-resident state (Step 6): ON"
                    [invwedge_lusgs_lusgshyb]="stage linear preconditioner (LU_SGS apply): device|LU_SGS device apply: hybrid|device-resident state (Step 6): ON"
                    [invwedge_lusgs_lusgsw1]="stage linear preconditioner (LU_SGS apply): device|LU_SGS device apply: hybrid (wide >= 1 rows|device-resident state (Step 6): ON"
                    [invwedge_lusgs_lusgssingle]="stage linear preconditioner (LU_SGS apply): device|LU_SGS device apply: single-team|device-resident state (Step 6): ON"
                    [super_cat_iluteam]="ILU(0) device build: team-per-row|device-resident state (Step 6): ON"
                    [super_cat_ilurow]="ILU(0) device build: row-per-thread|device-resident state (Step 6): ON"
                    [super_cat_pycustom]="stage bc:ISOTHERMAL (Wall) [non-catalytic] [py-custom Twall]: device|device-resident state (Step 6): ON"
                    [invwedge_dyn_dyn1st]="stage dynamic grid (grid-velocity dt/eigenvalue terms + GCL, metric re-upload per time step): device|stage time step (spectral radii + local dt + min/max reduce): device|stage dual-time source (1st order): device|device-resident state (Step 6): ON" )
cases=("$@"); [ ${#cases[@]} -eq 0 ] && cases=(invwedge_ausm thermalbath super_cat axi_visccone)
status=0
for c in "${cases[@]}"; do
  for mode in NO YES; do
    d=$RUN/chk_${c}${RUN_TAG}_$mode; rm -rf "$d"; mkdir -p "$d"; cd "$d"
    cp "${CFG[$c]}" run.cfg; ln -sf $M/*.su2 .
    kmode=$mode; hostenv=""; bin=$BIN; np=$NP
    if [ "$mode" = NO ]; then bin=$REF_BIN; np=$REF_NP; fi
    if [ "$mode" = NO ] && [ "$REF_MODE" = HOST ]; then kmode=YES; hostenv=all; fi
    if [ "$mode" = NO ] && [ "$REF_MODE" = YES ]; then kmode=YES; fi
    sed -i '/^ENABLE_KOKKOS/d' run.cfg; printf "\nENABLE_KOKKOS= %s\n" "$kmode" >> run.cfg  # leading newline: some cfgs lack a trailing one
    if [ -n "${EXTRA_CFG:-}" ]; then echo "$EXTRA_CFG" | tr ';' '\n' | while read -r l; do
      [ -n "$l" ] && { sed -i "/^${l%%=*}[ =]/d" run.cfg; [ -n "${l#*=}" ] && echo "$l" >> run.cfg; }; done; fi  # "KEY=" (no value) deletes the line (e.g. ITER= when going unsteady)
    t0=$(date +%s.%N); runner=("$bin"); [ -n "${PYDRIVER:-}" ] && runner=(python3 "$PYDRIVER")  # 6b.b: pysu2 driver script (both runs)
    SU2_KOKKOS_NEMO_HOST=$hostenv $(mpiprefix "$np") "${runner[@]}" run.cfg > log.txt 2>&1; rc=$?
    echo "$c $mode${hostenv:+ (REF_MODE=HOST: YES, all NEMO stages host)}${np:+ np=$np} exit=$rc wall=$(printf '%.2f' "$(echo "$(date +%s.%N)-$t0" | bc)")s"
    [ $rc -ne 0 ] && { status=1; tail -5 log.txt; }
    expkey=$c$RUN_TAG; [ -z "${EXPECT[$expkey]:-}" ] && expkey=$c   # per-RUN_TAG EXPECT key first, else plain case key (backward compatible)
    if [ "$mode" = YES ] && [ -n "${EXPECT[$expkey]:-}" ]; then
      IFS='|' read -ra exp_lines <<< "${EXPECT[$expkey]}"
      for e in "${exp_lines[@]}"; do
        grep -qF "$e" log.txt || { echo "  expected stage line missing: $e"; status=1; miss=1; }
      done
      # Show what the run actually reported (the host reason), so a fallback is diagnosable from the gate log.
      [ -n "${miss:-}" ] && { grep -E '^\[KOKKOS_NEMO\] stage (upwind|centered|bc:|linear)|device-resident' log.txt | head -n 12 | sed 's/^/    got: /'; unset miss; }
    fi
  done
  python3 - "$c$RUN_TAG" "$RES_TOL" "$FIELD_TOL" "${REF[$c]:-}" "$PHYS_SCALE" "${RES_NOISE_ORDERS:-inf}" <<'EOF' || status=1
import os, sys, re, glob, numpy as np
RUN = os.environ.get('RUN_DIR', '/home/claude/runs')
c, rtol, ftol, ref = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
phys, noise = sys.argv[5] == '1', float(sys.argv[6])
def lastrow(mode):
    rows = [l for l in open(f'{RUN}/chk_{c}_{mode}/log.txt') if re.match(r'^\|\s+\d+\|', l)]
    return np.array([float(x) for x in rows[-1].strip().strip('|').split('|')[1:]]) if rows else None
a, b = lastrow('NO'), lastrow('YES')
ok = a is not None and b is not None and a.shape == b.shape
if ok:
    hl = [l for l in open(f'{RUN}/chk_{c}_NO/log.txt') if re.match(r'^\|\s*(Time_Iter|Inner_Iter|Outer_Iter|Iter)\|', l) and 'rms[' in l]
    names = [x.strip() for x in hl[-1].strip().strip('|').split('|')][1:] if hl else []
    isres = np.array([n.startswith('rms[') for n in names]) if len(names) == len(a) else np.ones(len(a), bool)
    mx = lambda r: r[isres].max()
    keep = ~(isres & (a < mx(a) - noise) & (b < mx(b) - noise))
    dr = np.abs(a - b)[keep].max(); ok = dr <= rtol
    if not keep.all(): print(f'  residual columns ignored as round-off (> {noise:g} orders below max): {[names[k] if len(names) == len(a) else k for k in np.where(~keep)[0]]}')
    print(f'  residual row NO : {a}\n  residual row YES: {b}\n  max |dres| = {dr:.2e} (tol {rtol})')
    if ref:
        r = np.array([float(x) for x in ref.split()]); print(f'  NO vs reference max |d| = {np.abs(a[:len(r)]-r).max():.2e}')
try:
    pick = lambda m: sorted(glob.glob(f'{RUN}/chk_{c}_{m}/restart_flow*.csv'))[-1]
    fa = np.genfromtxt(pick('NO'), delimiter=',', skip_header=1)[:, 3:]
    fb = np.genfromtxt(pick('YES'), delimiter=',', skip_header=1)[:, 3:]
    scale = np.abs(fa).max(0)
    if phys:
        hdr = open(pick('NO')).readline().replace('"', '').strip().split(',')[3:]
        sp = [k for k, h in enumerate(hdr) if h.startswith('Density_')]
        mo = [k for k, h in enumerate(hdr) if h.startswith('Momentum_')]
        en = hdr.index('Energy') if 'Energy' in hdr else None
        rho = fa[:, sp].sum(1)
        if sp: scale[sp] = rho.max()
        if mo and en is not None: scale[mo] = np.sqrt(rho * np.abs(fa[:, en])).max()
    df = (np.abs(fa - fb).max(0) / np.maximum(scale, 1e-300)).max()
    print(f'  max field |diff|/{"phys scale" if phys else "max|field|"} = {df:.2e} (tol {ftol})'); ok = ok and df <= ftol
except Exception as e:
    print('  field compare failed:', e); ok = False
print(f'  {c}: {"PASS" if ok else "FAIL"}'); sys.exit(0 if ok else 1)
EOF
done
echo "OVERALL: $([ $status -eq 0 ] && echo PASS || echo FAIL)"; exit $status
