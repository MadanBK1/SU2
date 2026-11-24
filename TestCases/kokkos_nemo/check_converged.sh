#!/usr/bin/env bash
# Production (fast-math) gate: converged-solution comparison of ENABLE_KOKKOS=NO vs YES.
#
# Why: under -ffast-math the device kernels and the CPU templates are contracted/re-associated differently,
# so they agree only to a few ULP. NEMO's MUSCL non-physical-reconstruction switch (CheckNonPhys + sticky
# Non_Physical) and the limiter turn such ULP noise into O(1e-4) transient field differences after ~10
# iterations (a discrete switch flips for a few edges). That is not a physics difference; the meaningful
# check is that both runs converge to the same steady solution with the same convergence history.
# Bitwise exactness is gated separately on the strict IEEE build (check.sh REF_MODE=HOST, FIELD_TOL=0).
#
# Usage: check_converged.sh [case ...]   (default: invwedge_ausm)
# Env:   BIN (default production build), ITER (default 2000), PAR (default 1: run NO and YES concurrently),
#        FIELD_TOL (default 1e-6, max over fields of max|diff| / scale, scale = max|field| per column except the
#        species partial densities, which use max of the mixture density (trace species of ~1e-20 kg/m^3 are noise),
#        and momenta, which use max sqrt(rho*rhoE)), HIST_TOL (default 0.25, max |d log10 res| over the history, all columns,
#        after the first HIST_SKIP=20 iterations), FINAL_TOL (default 0.05, |d log10 res| of the final row).
set -u
SU2=${SU2_ROOT:-/home/claude/su2}
BIN=${BIN:-$SU2/build-kokkos-serial/SU2_CFD/src/SU2_CFD}
ITER=${ITER:-2000}
PAR=${PAR:-1}
FIELD_TOL=${FIELD_TOL:-1e-6}
HIST_TOL=${HIST_TOL:-0.25}
HIST_SKIP=${HIST_SKIP:-20}
FINAL_TOL=${FINAL_TOL:-0.05}
TC=$SU2/TestCases/nonequilibrium
M=${MESH_DIR:-/home/claude/meshes}
RUN=${RUN_DIR:-/home/claude/runs}
declare -A CFG=( [invwedge_ausm]=$TC/invwedge/invwedge_ausm.cfg
                 [super_cat]=$TC/visc_wedge/super_cat.cfg )
cases=("$@"); [ ${#cases[@]} -eq 0 ] && cases=(invwedge_ausm)
status=0
for c in "${cases[@]}"; do
  run_one() {
    local mode=$1 d=$RUN/conv_${c}_$1
    rm -rf "$d"; mkdir -p "$d"; cd "$d"
    cp "${CFG[$c]}" run.cfg; ln -sf $M/*.su2 .
    sed -i '/^ENABLE_KOKKOS/d; s/^ITER=.*/ITER= '"$ITER"'/; s/^OUTPUT_WRT_FREQ=.*/OUTPUT_WRT_FREQ= 100000/' run.cfg
    sed -i '/^CONV_RESIDUAL_MINVAL/d; /^CONV_CAUCHY_EPS/d' run.cfg  # run the full ITER count
    printf "\nCONV_RESIDUAL_MINVAL= -50\nCONV_CAUCHY_EPS= 1E-30\nENABLE_KOKKOS= %s\n" "$mode" >> run.cfg
    local t0=$(date +%s.%N); "$BIN" run.cfg > log.txt 2>&1; local rc=$?
    echo "$c $mode ITER=$ITER exit=$rc wall=$(printf '%.1f' "$(echo "$(date +%s.%N)-$t0" | bc)")s"
    return $rc
  }
  if [ "$PAR" = 1 ]; then
    run_one NO & p1=$!; run_one YES & p2=$!
    wait $p1 || status=1; wait $p2 || status=1
  else
    run_one NO || status=1; run_one YES || status=1
  fi
  python3 - "$c" "$FIELD_TOL" "$HIST_TOL" "$HIST_SKIP" "$FINAL_TOL" <<'EOF' || status=1
import os, sys, re, glob, numpy as np
RUN = os.environ.get('RUN_DIR', '/home/claude/runs')
c, ftol, htol, hskip, fintol = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
def hist(mode):
    rows = {}
    for l in open(f'{RUN}/conv_{c}_{mode}/log.txt'):
        if re.match(r'^\|\s+\d+\|', l):
            v = l.strip().strip('|').split('|'); rows[int(v[0])] = [float(x) for x in v[1:]]
    return rows
a, b = hist('NO'), hist('YES')
it = sorted(set(a) & set(b)); ok = len(it) > 0
if ok:
    A = np.array([a[i] for i in it]); B = np.array([b[i] for i in it])
    sel = [k for k, i in enumerate(it) if i >= hskip]
    dh = np.abs(A[sel] - B[sel]).max() if sel else 0.0
    df = np.abs(A[-1] - B[-1]).max()
    drop = A[-1, 0] - A[0, 0]
    print(f'  iterations compared: {len(it)} (last {it[-1]}), residual[0] drop NO: {drop:.2f} orders')
    print(f'  final row NO : {A[-1]}\n  final row YES: {B[-1]}')
    print(f'  history max |d| (iter >= {hskip}) = {dh:.3e} (tol {htol}); final row max |d| = {df:.3e} (tol {fintol})')
    ok = dh <= htol and df <= fintol
try:
    pick = lambda m: sorted(glob.glob(f'{RUN}/conv_{c}_{m}/restart_flow*.csv'))[-1]
    fa = np.genfromtxt(pick('NO'), delimiter=',', skip_header=1)[:, 3:]
    fb = np.genfromtxt(pick('YES'), delimiter=',', skip_header=1)[:, 3:]
    hdr = open(pick('NO')).readline().replace('"', '').strip().split(',')[3:]
    scale = np.abs(fa).max(0)
    sp = [k for k, h in enumerate(hdr) if h.startswith('Density_')]
    mo = [k for k, h in enumerate(hdr) if h.startswith('Momentum_')]
    rho = fa[:, sp].sum(1)
    if sp: scale[sp] = rho.max()
    if mo and 'Energy' in hdr: scale[mo] = np.sqrt(rho * np.abs(fa[:, hdr.index('Energy')])).max()
    per = np.abs(fa - fb).max(0) / np.maximum(scale, 1e-300)
    print('  per-field max|diff|/scale: ' + ' '.join(f'{h}={v:.2e}' for h, v in zip(hdr, per)))
    print(f'  final field max |diff|/scale = {per.max():.3e} (tol {ftol}), worst {hdr[int(per.argmax())]}')
    ok = ok and per.max() <= ftol
except Exception as e:
    print('  field compare failed:', e); ok = False
print(f'  {c}: {"PASS" if ok else "FAIL"}'); sys.exit(0 if ok else 1)
EOF
done
echo "CONVERGED OVERALL: $([ $status -eq 0 ] && echo PASS || echo FAIL)"; exit $status
