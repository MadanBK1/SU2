#!/usr/bin/env bash
# Round-200: review-41 queue launcher — build strict+serial at HEAD (step 1) with
# per-build logs, then run limfix_queue.sh (steps 2-6, resumable via .done markers).
set -u
cd "$(dirname "$0")/../.." || exit 2
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
OUT=validation_logs/2026-11-28_4.6-PeriodicLim-fix
mkdir -p "$OUT"
{
  echo "code $(git rev-parse --short HEAD) $(date -Is)"
  echo "=== BUILD strict $(date -Is) ==="
} | tee "$OUT/BUILD_HEAD.txt"
ninja -C build-kokkos-strict -j10 SU2_CFD/src/SU2_CFD > "$OUT/build-limfix-r1.log" 2>&1
echo "strict rc=$? $(date -Is)" | tee -a "$OUT/BUILD_HEAD.txt"
echo "=== BUILD serial $(date -Is) ===" | tee -a "$OUT/BUILD_HEAD.txt"
ninja -C build-kokkos-serial -j10 SU2_CFD/src/SU2_CFD > "$OUT/build-limfix-serial-r1.log" 2>&1
echo "serial rc=$? $(date -Is)" | tee -a "$OUT/BUILD_HEAD.txt"
if grep -q "strict rc=0" "$OUT/BUILD_HEAD.txt" && grep -q "serial rc=0" "$OUT/BUILD_HEAD.txt"; then
  echo "=== QUEUE $(date -Is) ===" | tee -a "$OUT/BUILD_HEAD.txt"
  bash TestCases/kokkos_nemo/limfix_queue.sh >> "$OUT/BUILD_HEAD.txt" 2>&1
  echo "queue rc=$? $(date -Is)" | tee -a "$OUT/BUILD_HEAD.txt"
else
  echo "BUILD FAILED - queue NOT started (see build-limfix*.log)" | tee -a "$OUT/BUILD_HEAD.txt"
fi
# Review 44: chain the zc2a-bndsync queue (same HEAD; its own logs + QUEUE_STATUS in validation_logs/2026-11-28_zc2a-bndsync/).
if grep -q "QUEUE FINISHED" "$OUT/QUEUE_STATUS.txt" 2>/dev/null; then
  bash TestCases/kokkos_nemo/bnd_queue.sh
fi
