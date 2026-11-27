#!/usr/bin/env bash
# 4.6-PeriodicAUSMM validation queue (review 39 steps B1-B7), DETACHED and RESUMABLE.
# Writes every log straight into validation_logs/2026-11-27_4.6-PeriodicAUSMM/ and one line per finished
# step into QUEUE_STATUS.txt there. A step whose .done marker exists is skipped, so re-launching resumes.
# Launch:  bash TestCases/kokkos_nemo/mm_queue.sh   (harness background task)
# Check:   cat validation_logs/2026-11-27_4.6-PeriodicAUSMM/QUEUE_STATUS.txt ; pgrep -f mm_queue.sh
set -u
cd "$(dirname "$0")/../.." || exit 2
SU2="$PWD"
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export SU2_ROOT="$SU2"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
OUT="$SU2/validation_logs/2026-11-27_4.6-PeriodicAUSMM"
ST="$OUT/QUEUE_STATUS.txt"
STRICT="$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD"
SERIAL="$SU2/build-kokkos-serial/SU2_CFD/src/SU2_CFD"
mkdir -p "$OUT" "$RUN_DIR"
LOCK="$OUT/.queue.lock"
exec 9> "$LOCK"; flock -n 9 || { echo "mm_queue already running"; exit 0; }
sha=$(git rev-parse --short HEAD)

step() {
  local name=$1 to=$2; shift 2
  [ -e "$OUT/.$name.done" ] && return 0
  echo "$(date -Is) START $name (code $sha)" >> "$ST"
  local t0=$(date +%s)
  timeout "$to" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  local summ
  summ=$(grep -E "OVERALL|expected stage line missing|NOISEFLOOR" "$OUT/$name.log" | tail -3 | tr '\n' ' ')
  echo "$(date -Is) END   $name rc=$rc wall=$(( $(date +%s) - t0 ))s :: ${summ:-<no OVERALL line; see log>}" >> "$ST"
  touch "$OUT/.$name.done"
}
g1() {  # g1 <name> [ENV=VAL ...] -- <cases...>
  local name=$1; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  step "$name" 1800 env BIN="$STRICT" REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 "${envs[@]}" \
    bash TestCases/kokkos_nemo/check.sh "$@"
}

# --- step 1: build strict + serial (incremental no-op unless code changed) ---
build_b() {
  local d=$SU2; ninja -C "$d/build-kokkos-strict" -j10 SU2_CFD/src/SU2_CFD; local r1=$?
  [ $r1 -eq 0 ] && echo "OVERALL: BUILD PASS" || echo "OVERALL: BUILD FAIL"
}
step build-mm-r1 1800 bash -c "$(declare -f build_b); SU2='$SU2' build_b"
step build-mm-serial-r1 1800 bash -c 'ninja -C build-kokkos-serial -j10 SU2_CFD/src/SU2_CFD; echo "OVERALL: ninja rc=$?"'

# --- step 2: CPU sanity (ENABLE_KOKKOS=NO, AUSM+M pressure-diffusion sensor + periodic) ---
cpu_sanity() {
  local d=$RUN_DIR/cpu_mm_r1; rm -rf "$d"; mkdir -p "$d"; cd "$d" || return 2
  cp "$SU2/TestCases/kokkos_nemo/meshes/invper_ausmm.cfg" run.cfg
  ln -sf "$SU2/TestCases/kokkos_nemo/meshes/"*.su2 .
  sed -i '/^ENABLE_KOKKOS/d' run.cfg; printf "\nENABLE_KOKKOS= NO\n" >> run.cfg
  "$STRICT" run.cfg > log.txt 2>&1; local rc=$?
  cp log.txt "$OUT/cpu-mm-r1-su2.log"
  local rows; rows=$(grep -cE '^\|[[:space:]]+[0-9]+\|' log.txt)
  echo "rc=$rc residual_rows=$rows"; tail -8 log.txt; grep -in "periodic\|sensor\|AUSM" log.txt | head -5
  [ $rc -eq 0 ] && [ "$rows" -ge 16 ] && echo "OVERALL: CPU SANITY PASS" || echo "OVERALL: CPU SANITY FAIL"
  return $rc
}
step cpu-mm-r1 900 bash -c "$(declare -f cpu_sanity); SU2='$SU2' RUN_DIR='$RUN_DIR' STRICT='$STRICT' OUT='$OUT' cpu_sanity"
if ! grep -q "CPU SANITY PASS" "$OUT/cpu-mm-r1.log"; then
  echo "$(date -Is) STOP: CPU sanity failed - inspect cpu-mm-r1-su2.log, else HEAVY; rm $OUT/.cpu-mm-r1.done and relaunch" >> "$ST"
  exit 1
fi

# --- step 3: G1 strict ---
g1 g1-mm-r1 -- invper_ausmm
# --- step 4: EULER_EXPLICIT variant (EXPECT falls back to plain key) ---
g1 g1-mm-exp-r1 RUN_TAG=_exp "EXTRA_CFG=TIME_DISCRE_FLOW= EULER_EXPLICIT" -- invper_ausmm
# --- step 5: control (AUSM+M gate restored -> host) ---
g1 g1-mm-off-r1 RUN_TAG=_mmoff SU2_KOKKOS_NEMO_PERIODIC_AUSMM=0 -- invper_ausmm
# --- step 6: A/B + poison ---
g1 ab-mm-r1       RUN_TAG=_mmab    SU2_KOKKOS_STAGE_VERIFY=1 -- invper_ausmm
g1 poison-mm-r1   RUN_TAG=_mmpois  SU2_KOKKOS_NEMO_POISON=1 -- invper_ausmm
# --- step 7: noise floor (K=10) ---
step nf-mm-r1 3600 env BIN="$SERIAL" BIN_STRICT="$STRICT" CASE=invper_ausmm K=10 bash TestCases/kokkos_nemo/check_noisefloor.sh
echo "$(date -Is) QUEUE FINISHED" >> "$ST"
