#!/usr/bin/env bash
# 4.6-PeriodicLim validation queue (review 39 steps C1-C5) + D (regressions), DETACHED and RESUMABLE.
# Writes every log straight into validation_logs/2026-11-27_4.6-PeriodicLim/ and one line per finished
# step into QUEUE_STATUS.txt there. A step whose .done marker exists is skipped, so re-launching resumes.
# Launch:  bash TestCases/kokkos_nemo/lim_queue.sh   (harness background task)
# Check:   cat validation_logs/2026-11-27_4.6-PeriodicLim/QUEUE_STATUS.txt ; pgrep -f lim_queue.sh
set -u
cd "$(dirname "$0")/../.." || exit 2
SU2="$PWD"
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export SU2_ROOT="$SU2"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
OUT="$SU2/validation_logs/2026-11-27_4.6-PeriodicLim"
ST="$OUT/QUEUE_STATUS.txt"
STRICT="$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD"
SERIAL="$SU2/build-kokkos-serial/SU2_CFD/src/SU2_CFD"
mkdir -p "$OUT" "$RUN_DIR"
LOCK="$OUT/.queue.lock"
exec 9> "$LOCK"; flock -n 9 || { echo "lim_queue already running"; exit 0; }
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

# --- step 1: build strict + serial (incremental no-op; same build as queue B) ---
build_b() {
  local d=$SU2; ninja -C "$d/build-kokkos-strict" -j10 SU2_CFD/src/SU2_CFD; local r1=$?
  [ $r1 -eq 0 ] && echo "OVERALL: BUILD PASS" || echo "OVERALL: BUILD FAIL"
}
step build-lim-r1 1800 bash -c "$(declare -f build_b); SU2='$SU2' build_b"
step build-lim-serial-r1 1800 bash -c 'ninja -C build-kokkos-serial -j10 SU2_CFD/src/SU2_CFD; echo "OVERALL: ninja rc=$?"'

# --- step 2: CPU sanity (ENABLE_KOKKOS=NO, MUSCL+VENKATAKRISHNAN limiter + periodic) ---
cpu_sanity() {
  local d=$RUN_DIR/cpu_lim_r1; rm -rf "$d"; mkdir -p "$d"; cd "$d" || return 2
  cp "$SU2/TestCases/kokkos_nemo/meshes/invper_lim.cfg" run.cfg
  ln -sf "$SU2/TestCases/kokkos_nemo/meshes/"*.su2 .
  sed -i '/^ENABLE_KOKKOS/d' run.cfg; printf "\nENABLE_KOKKOS= NO\n" >> run.cfg
  "$STRICT" run.cfg > log.txt 2>&1; local rc=$?
  cp log.txt "$OUT/cpu-lim-r1-su2.log"
  local rows; rows=$(grep -cE '^\|[[:space:]]+[0-9]+\|' log.txt)
  echo "rc=$rc residual_rows=$rows"; tail -8 log.txt; grep -in "periodic\|limiter\|MUSCL" log.txt | head -5
  [ $rc -eq 0 ] && [ "$rows" -ge 16 ] && echo "OVERALL: CPU SANITY PASS" || echo "OVERALL: CPU SANITY FAIL"
  return $rc
}
step cpu-lim-r1 900 bash -c "$(declare -f cpu_sanity); SU2='$SU2' RUN_DIR='$RUN_DIR' STRICT='$STRICT' OUT='$OUT' cpu_sanity"
if ! grep -q "CPU SANITY PASS" "$OUT/cpu-lim-r1.log"; then
  echo "$(date -Is) STOP: CPU sanity failed - inspect cpu-lim-r1-su2.log, else HEAVY; rm $OUT/.cpu-lim-r1.done and relaunch" >> "$ST"
  exit 1
fi

# --- step 3: G1 strict invper_lim (MUSCL + VENKATAKRISHNAN, limiter rows staged) ---
g1 g1-lim-r1 -- invper_lim
# --- step 4: variants ---
g1 g1-lim-bj-r1      RUN_TAG=_bj    "EXTRA_CFG=SLOPE_LIMITER_FLOW= BARTH_JESPERSEN" -- invper_lim
g1 g1-lim-wang-r1    RUN_TAG=_wang  "EXTRA_CFG=SLOPE_LIMITER_FLOW= VENKATAKRISHNAN_WANG" -- invper_lim
g1 g1-lim-off-r1     RUN_TAG=_limoff SU2_KOKKOS_NEMO_PERIODIC_LIM=0 -- invper_lim
g1 g1-lim-pervenk-r1 RUN_TAG=_pervenk "EXTRA_CFG=MUSCL_FLOW= YES;SLOPE_LIMITER_FLOW= VENKATAKRISHNAN" -- invper_ausm
# --- step 5: A/B + poison ---
g1 ab-lim-r1       RUN_TAG=_limab   SU2_KOKKOS_STAGE_VERIFY=1 -- invper_lim
g1 poison-lim-r1   RUN_TAG=_limpois SU2_KOKKOS_NEMO_POISON=1 -- invper_lim
# --- step 6: noise floor (K=10) ---
step nf-lim-r1 3600 env BIN="$SERIAL" BIN_STRICT="$STRICT" CASE=invper_lim K=10 bash TestCases/kokkos_nemo/check_noisefloor.sh

# --- step 7: queue D regressions (strict bitwise): default 8 + all 5 periodic cases ---
step g1-regress-r1 3600 env BIN="$STRICT" REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 \
  bash TestCases/kokkos_nemo/check.sh invwedge_ausm thermalbath super_cat axi_visccone invwedge_lusgs \
    invwedge_mg invwedge_dyn supin_visc invper_ausm viscper_ausm invper_dt invper_ausmm invper_lim
echo "$(date -Is) QUEUE FINISHED" >> "$ST"
