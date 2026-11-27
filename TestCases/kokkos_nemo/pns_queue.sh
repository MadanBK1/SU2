#!/usr/bin/env bash
# 4.6-PeriodicNS validation queue (review 37 steps 2-8), DETACHED and RESUMABLE.
# Writes every log straight into validation_logs/2026-11-27_4.6-PeriodicNS/ and one line per finished step into
# QUEUE_STATUS.txt there. A step whose .done marker exists is skipped, so re-launching resumes where it stopped.
# Launch (returns at once; survives the end of the agent session):
#   bash TestCases/kokkos_nemo/pns_queue.sh   (harness background task; NOT setsid - detached launch dies in sandbox)
# Check:  cat validation_logs/2026-11-27_4.6-PeriodicNS/QUEUE_STATUS.txt ; pgrep -f pns_queue.sh
set -u
cd "$(dirname "$0")/../.." || exit 2
SU2="$PWD"
export SU2_ROOT="$SU2"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
OUT="$SU2/validation_logs/2026-11-27_4.6-PeriodicNS"
ST="$OUT/QUEUE_STATUS.txt"
STRICT="$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD"
SERIAL="$SU2/build-kokkos-serial/SU2_CFD/src/SU2_CFD"
mkdir -p "$OUT" "$RUN_DIR"
LOCK="$OUT/.queue.lock"
exec 9> "$LOCK"; flock -n 9 || { echo "pns_queue already running"; exit 0; }
sha=$(git rev-parse --short HEAD)

# step <name> <timeout_s> <cmd...> : runs cmd > $OUT/<name>.log, records rc + summary.
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

# --- step 2: plain CPU sanity (ENABLE_KOKKOS=NO) ---
cpu_sanity() {
  local d=$RUN_DIR/cpu_pns_r1; rm -rf "$d"; mkdir -p "$d"; cd "$d" || return 2
  cp "$SU2/TestCases/kokkos_nemo/meshes/viscper_ausm.cfg" run.cfg
  ln -sf "$SU2/TestCases/kokkos_nemo/meshes/"*.su2 .
  sed -i '/^ENABLE_KOKKOS/d' run.cfg; printf "\nENABLE_KOKKOS= NO\n" >> run.cfg
  "$STRICT" run.cfg > log.txt 2>&1; local rc=$?
  cp log.txt "$OUT/cpu-pns-r1-su2.log"
  local rows; rows=$(grep -cE '^\|[[:space:]]+[0-9]+\|' log.txt)
  echo "rc=$rc residual_rows=$rows"; tail -8 log.txt; grep -in "periodic" log.txt | head -5
  [ $rc -eq 0 ] && [ "$rows" -ge 21 ] && echo "OVERALL: CPU SANITY PASS" || echo "OVERALL: CPU SANITY FAIL"
  return $rc
}
step cpu-pns-r1 900 bash -c "$(declare -f cpu_sanity); SU2='$SU2' RUN_DIR='$RUN_DIR' STRICT='$STRICT' OUT='$OUT' cpu_sanity"
if ! grep -q "CPU SANITY PASS" "$OUT/cpu-pns-r1.log"; then
  echo "$(date -Is) STOP: CPU sanity failed - try CFL_NUMBER= 1 in viscper_ausm.cfg (r2), else HEAVY; rm $OUT/.cpu-pns-r1.done and relaunch" >> "$ST"
  exit 1
fi

# --- step 3: G1 strict ---
g1 g1-pns-r1 -- viscper_ausm
# --- step 4: variants (review 37: RUN_TAG _exp / _muscl / _nsoff) ---
g1 g1-pns-exp-r1   RUN_TAG=_exp   "EXTRA_CFG=TIME_DISCRE_FLOW= EULER_EXPLICIT" -- viscper_ausm
g1 g1-pns-muscl-r1 RUN_TAG=_muscl "EXTRA_CFG=MUSCL_FLOW= YES;SLOPE_LIMITER_FLOW= VAN_ALBADA_EDGE" -- viscper_ausm
g1 g1-pns-nsoff-r1 RUN_TAG=_nsoff SU2_KOKKOS_NEMO_PERIODIC_NS=0 -- viscper_ausm
# --- step 5: A/B + poison ---
g1 ab-pns-r1       RUN_TAG=_pnsab    SU2_KOKKOS_STAGE_VERIFY=1 -- viscper_ausm
g1 poison-pns-r1   RUN_TAG=_pnspois  SU2_KOKKOS_NEMO_POISON=1 -- viscper_ausm
# --- step 6: regressions (default 8 + invper_ausm, strict bitwise) ---
g1 g1-pns-regress-r1 -- invwedge_ausm thermalbath super_cat axi_visccone invwedge_lusgs invwedge_mg invwedge_dyn supin_visc invper_ausm
# --- step 7: noise floor (K=10) ---
step nf-pns-r1 3600 env BIN="$SERIAL" BIN_STRICT="$STRICT" CASE=viscper_ausm K=10 bash TestCases/kokkos_nemo/check_noisefloor.sh
echo "$(date -Is) QUEUE FINISHED" >> "$ST"
