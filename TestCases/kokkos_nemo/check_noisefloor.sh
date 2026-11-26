#!/usr/bin/env bash
# Production noise-floor gate for MUSCL cases whose non-physical switch amplifies ULP noise (invwedge_ausm):
# the deviation of ENABLE_KOKKOS=YES from NO (production -ffast-math binary) must not exceed K x the deviation
# between the CPU-only (NO) runs of the strict IEEE and the production binaries (same source, different FP
# contraction), at ITER iterations. Also compares the residual history (max |d log10|) the same way.
# Env: BIN, BIN_STRICT, ITERS (default "11 100"), K (default 10), CASE (invwedge_ausm | super_cat; Step 8).
# super_cat: the history also holds LIFT / DRAG / TOTAL_HEATFLUX (printed with 4-6 significant digits), so history
# differences are measured relative to max(|value|, 1) per entry instead of absolutely.
set -u
SU2=${SU2_ROOT:-/home/claude/su2}
M=${MESH_DIR:-/home/claude/meshes}
RUN=${RUN_DIR:-/home/claude/runs}
BIN=${BIN:-$SU2/build-kokkos-serial/SU2_CFD/src/SU2_CFD}
BIN_STRICT=${BIN_STRICT:-$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD}
ITERS=${ITERS:-"11 100"}; K=${K:-10}
CASE=${CASE:-invwedge_ausm}
case $CASE in
  invwedge_ausm) CFG=$SU2/TestCases/nonequilibrium/invwedge/invwedge_ausm.cfg; HREL=0 ;;
  invwedge_msw) CFG=$SU2/TestCases/nonequilibrium/invwedge/invwedge_msw.cfg; HREL=0 ;;
  invwedge_ausmplusup2) CFG=$SU2/TestCases/nonequilibrium/invwedge/invwedge_ausmplusup2.cfg; HREL=0 ;;
  super_cat) CFG=$SU2/TestCases/nonequilibrium/visc_wedge/super_cat.cfg; HREL=1 ;;
  smol_maxwell) CFG=$SU2/TestCases/nonequilibrium/visc_wedge/smol_maxwell.cfg; HREL=1 ;;
  supin_visc) CFG=$SU2/TestCases/nonequilibrium/visc_wedge/supin_visc.cfg; HREL=1 ;;
  invwedge_lusgs) CFG=$SU2/TestCases/nonequilibrium/invwedge/invwedge_lusgs.cfg; HREL=0 ;;
  *) echo "unknown CASE $CASE"; exit 2 ;;
esac
st=0
for it in $ITERS; do
  run() { # name bin mode
    local d=$RUN/nf_${CASE}_${it}_$1; rm -rf $d; mkdir -p $d; cd $d; ln -sf $M/*.su2 .
    cp $CFG run.cfg; printf "\n" >> run.cfg; sed -i "/^ENABLE_KOKKOS/d; s/^ITER=.*/ITER= $it/; s/^OUTPUT_FILES=.*/OUTPUT_FILES= (RESTART_ASCII)/" run.cfg
    printf "\nENABLE_KOKKOS= %s\n" $3 >> run.cfg; $2 run.cfg > log.txt 2>&1
  }
  run prodNO $BIN NO & a=$!; run strictNO $BIN_STRICT NO & b=$!; wait $a $b
  run prodYES $BIN YES
  python3 - $it $K $CASE $HREL <<'PY' || st=1
import os, sys, re, numpy as np
RUN = os.environ.get('RUN_DIR', '/home/claude/runs')
it, K, case, hrel = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4] == '1'
d = lambda n: f'{RUN}/nf_{case}_{it}_{n}'
def fld(n):
    f = np.genfromtxt(d(n) + '/restart_flow.csv', delimiter=',', skip_header=1)[:, 3:]
    return f
def hist(n):
    return np.array([[float(x) for x in l.strip().strip('|').split('|')[1:]] for l in open(d(n) + '/log.txt') if re.match(r'^\|\s+\d+\|', l)])
P, S, Y = fld('prodNO'), fld('strictNO'), fld('prodYES')
rho = P[:, :5].sum(1).max(); scale = np.abs(P).max(0); scale[:5] = rho
scale[5:7] = np.sqrt(P[:, :5].sum(1) * np.abs(P[:, 7])).max()
dev = (np.abs(Y - P).max(0) / scale).max(); cpu = (np.abs(S - P).max(0) / scale).max()
hp, hs, hy = hist('prodNO'), hist('strictNO'), hist('prodYES')
hsc = np.maximum(np.abs(hp), 1.0) if hrel else 1.0
hdev, hcpu = (np.abs(hy - hp) / hsc).max(), (np.abs(hs - hp) / hsc).max()
ok = dev <= K * max(cpu, 1e-300) and hdev <= K * max(hcpu, 1e-6)
print(f'{case} ITER {it}: field |YES-NO| = {dev:.3e}, CPU strict-vs-prod |NO-NO| = {cpu:.3e} (ratio {dev/max(cpu,1e-300):.2f}, K {K:g});'
      f' history {hdev:.2e} vs {hcpu:.2e}: {"PASS" if ok else "FAIL"}')
sys.exit(0 if ok else 1)
PY
done
echo "NOISEFLOOR OVERALL: $([ $st = 0 ] && echo PASS || echo FAIL)"; exit $st
