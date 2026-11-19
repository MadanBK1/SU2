#!/usr/bin/env bash
# Production noise-floor gate for MUSCL cases whose non-physical switch amplifies ULP noise (invwedge_ausm):
# the deviation of ENABLE_KOKKOS=YES from NO (production -ffast-math binary) must not exceed K x the deviation
# between the CPU-only (NO) runs of the strict IEEE and the production binaries (same source, different FP
# contraction), at ITER iterations. Also compares the residual history (max |d log10|) the same way.
# Env: BIN, BIN_STRICT, ITERS (default "11 100"), K (default 10).
set -u
BIN=${BIN:-/home/claude/su2/build-kokkos-serial/SU2_CFD/src/SU2_CFD}
BIN_STRICT=${BIN_STRICT:-/home/claude/su2/build-kokkos-strict/SU2_CFD/src/SU2_CFD}
ITERS=${ITERS:-"11 100"}; K=${K:-10}
CFG=/home/claude/su2/TestCases/nonequilibrium/invwedge/invwedge_ausm.cfg
st=0
for it in $ITERS; do
  run() { # name bin mode
    local d=/home/claude/runs/nf_${it}_$1; rm -rf $d; mkdir -p $d; cd $d; ln -sf /home/claude/meshes/*.su2 .
    cp $CFG run.cfg; sed -i "/^ENABLE_KOKKOS/d; s/^ITER=.*/ITER= $it/; s/^OUTPUT_FILES=.*/OUTPUT_FILES= (RESTART_ASCII)/" run.cfg
    printf "\nENABLE_KOKKOS= %s\n" $3 >> run.cfg; $2 run.cfg > log.txt 2>&1
  }
  run prodNO $BIN NO & a=$!; run strictNO $BIN_STRICT NO & b=$!; wait $a $b
  run prodYES $BIN YES
  python3 - $it $K <<'PY' || st=1
import sys, re, numpy as np
it, K = sys.argv[1], float(sys.argv[2])
d = lambda n: f'/home/claude/runs/nf_{it}_{n}'
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
hdev, hcpu = np.abs(hy - hp).max(), np.abs(hs - hp).max()
ok = dev <= K * max(cpu, 1e-300) and hdev <= K * max(hcpu, 1e-6)
print(f'ITER {it}: field |YES-NO| = {dev:.3e}, CPU strict-vs-prod |NO-NO| = {cpu:.3e} (ratio {dev/max(cpu,1e-300):.2f}, K {K:g});'
      f' history {hdev:.2e} vs {hcpu:.2e}: {"PASS" if ok else "FAIL"}')
sys.exit(0 if ok else 1)
PY
done
echo "NOISEFLOOR OVERALL: $([ $st = 0 ] && echo PASS || echo FAIL)"; exit $st
