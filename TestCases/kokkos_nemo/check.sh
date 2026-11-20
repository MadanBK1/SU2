#!/usr/bin/env bash
# Regression check: run NEMO cases with ENABLE_KOKKOS=NO and YES on the Kokkos
# Serial build, compare final screen residuals and restart fields.
# Usage: check.sh [case ...]   (cases: invwedge_ausm thermalbath super_cat partial_cat axi_visccone;
#   default: the first three)
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
set -u
SU2=/home/claude/su2
BIN=${BIN:-$SU2/build-kokkos-serial/SU2_CFD/src/SU2_CFD}
RES_TOL=${RES_TOL:-2e-6}
FIELD_TOL=${FIELD_TOL:-1e-9}
REF_MODE=${REF_MODE:-NO}
PHYS_SCALE=${PHYS_SCALE:-0}            # 1: field diffs scaled physically (see below) instead of per-column max|field|
RES_NOISE_ORDERS=${RES_NOISE_ORDERS:-} # e.g. 10: ignore residual columns > N orders below the row maximum (round-off)
TC=$SU2/TestCases/nonequilibrium
M=/home/claude/meshes
declare -A CFG=( [invwedge_ausm]=$TC/invwedge/invwedge_ausm.cfg
                 [thermalbath]=$TC/thermalbath/finitechemistry/thermalbath.cfg
                 [super_cat]=$TC/visc_wedge/super_cat.cfg
                 [partial_cat]=$TC/visc_wedge/partial_cat.cfg
                 [axi_visccone]=$TC/visc_wedge/axi_visccone.cfg )
declare -A REF=( [invwedge_ausm]="-1.073693 -1.598456 -18.298997 -18.626405 -18.572419 2.241766 1.868557 5.286079 0.843747" )
cases=("$@"); [ ${#cases[@]} -eq 0 ] && cases=(invwedge_ausm thermalbath super_cat)
status=0
for c in "${cases[@]}"; do
  for mode in NO YES; do
    d=/home/claude/runs/chk_${c}_$mode; rm -rf "$d"; mkdir -p "$d"; cd "$d"
    cp "${CFG[$c]}" run.cfg; ln -sf $M/*.su2 .
    kmode=$mode; hostenv=""
    if [ "$mode" = NO ] && [ "$REF_MODE" = HOST ]; then kmode=YES; hostenv=all; fi
    sed -i '/^ENABLE_KOKKOS/d' run.cfg; printf "\nENABLE_KOKKOS= %s\n" "$kmode" >> run.cfg  # leading newline: some cfgs lack a trailing one
    t0=$(date +%s.%N); SU2_KOKKOS_NEMO_HOST=$hostenv "$BIN" run.cfg > log.txt 2>&1; rc=$?
    echo "$c $mode${hostenv:+ (REF_MODE=HOST: YES, all NEMO stages host)} exit=$rc wall=$(printf '%.2f' "$(echo "$(date +%s.%N)-$t0" | bc)")s"
    [ $rc -ne 0 ] && { status=1; tail -5 log.txt; }
  done
  python3 - "$c" "$RES_TOL" "$FIELD_TOL" "${REF[$c]:-}" "$PHYS_SCALE" "${RES_NOISE_ORDERS:-inf}" <<'EOF' || status=1
import sys, re, glob, numpy as np
c, rtol, ftol, ref = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
phys, noise = sys.argv[5] == '1', float(sys.argv[6])
def lastrow(mode):
    rows = [l for l in open(f'/home/claude/runs/chk_{c}_{mode}/log.txt') if re.match(r'^\|\s+\d+\|', l)]
    return np.array([float(x) for x in rows[-1].strip().strip('|').split('|')[1:]]) if rows else None
a, b = lastrow('NO'), lastrow('YES')
ok = a is not None and b is not None and a.shape == b.shape
if ok:
    hl = [l for l in open(f'/home/claude/runs/chk_{c}_NO/log.txt') if re.match(r'^\|\s*(Time_Iter|Inner_Iter|Outer_Iter|Iter)\|', l) and 'rms[' in l]
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
    pick = lambda m: sorted(glob.glob(f'/home/claude/runs/chk_{c}_{m}/restart_flow*.csv'))[-1]
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
