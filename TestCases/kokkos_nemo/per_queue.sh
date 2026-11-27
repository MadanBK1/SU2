#!/usr/bin/env bash
# 4.6-Periodic validation queue (steps 2-7 of review 31), DETACHED and RESUMABLE.
# Writes every log straight into validation_logs/2026-11-27_4.6-Periodic/ and one line per finished step into
# QUEUE_STATUS.txt there. A step whose .done marker exists is skipped, so re-launching resumes where it stopped.
# Launch (returns at once; survives the end of the agent session):
#   setsid nohup bash TestCases/kokkos_nemo/per_queue.sh > /dev/null 2>&1 < /dev/null &
# Check:  cat validation_logs/2026-11-27_4.6-Periodic/QUEUE_STATUS.txt ; pgrep -f per_queue.sh
set -u
cd "$(dirname "$0")/../.." || exit 2
SU2="$PWD"
export SU2_ROOT="$SU2"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
OUT="$SU2/validation_logs/2026-11-27_4.6-Periodic"
ST="$OUT/QUEUE_STATUS.txt"
STRICT="$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD"
SERIAL="$SU2/build-kokkos-serial/SU2_CFD/src/SU2_CFD"
mkdir -p "$OUT" "$RUN_DIR"
LOCK="$OUT/.queue.lock"
exec 9> "$LOCK"; flock -n 9 || { echo "per_queue already running"; exit 0; }
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
  local d=$RUN_DIR/cpu_per_r1; rm -rf "$d"; mkdir -p "$d"; cd "$d" || return 2
  cp "$SU2/TestCases/kokkos_nemo/meshes/invper_ausm.cfg" run.cfg
  ln -sf "$SU2/TestCases/kokkos_nemo/meshes/"*.su2 .
  sed -i '/^ENABLE_KOKKOS/d' run.cfg; printf "\nENABLE_KOKKOS= NO\n" >> run.cfg
  "$STRICT" run.cfg > log.txt 2>&1; local rc=$?
  cp log.txt "$OUT/cpu-per-r1-su2.log"
  local rows; rows=$(grep -cE '^\|[[:space:]]+[0-9]+\|' log.txt)
  echo "rc=$rc residual_rows=$rows"; tail -8 log.txt; grep -in "periodic" log.txt | head -5
  [ $rc -eq 0 ] && [ "$rows" -ge 21 ] && echo "OVERALL: CPU SANITY PASS" || echo "OVERALL: CPU SANITY FAIL"
  return $rc
}
step cpu-per-r1 900 bash -c "$(declare -f cpu_sanity); SU2='$SU2' RUN_DIR='$RUN_DIR' STRICT='$STRICT' OUT='$OUT' cpu_sanity"
if ! grep -q "CPU SANITY PASS" "$OUT/cpu-per-r1.log"; then
  echo "$(date -Is) STOP: CPU sanity failed - fix cfg/mesh first (translation sign 2.0 -> -2.0?), then rm $OUT/.cpu-per-r1.done and relaunch" >> "$ST"
  exit 1
fi

# --- step 3: G1 strict ---
g1 g1-per-r1 -- invper_ausm
# --- step 4: variants ---
g1 g1-per-exp-r1   RUN_TAG=_perexp   "EXTRA_CFG=TIME_DISCRE_FLOW= EULER_EXPLICIT;CFL_NUMBER= 0.5" -- invper_ausm
g1 g1-per-muscl-r1 RUN_TAG=_permuscl "EXTRA_CFG=MUSCL_FLOW= YES;SLOPE_LIMITER_FLOW= VAN_ALBADA_EDGE" -- invper_ausm
g1 g1-per-lax-r1   RUN_TAG=_perlax   "EXTRA_CFG=CONV_NUM_METHOD_FLOW= LAX-FRIEDRICH" -- invper_ausm
g1 g1-per-off-r1   RUN_TAG=_peroff   SU2_KOKKOS_NEMO_RESIDENT=0 -- invper_ausm
g1 g1-per-venk-r1  RUN_TAG=_pervenk  "EXTRA_CFG=MUSCL_FLOW= YES;SLOPE_LIMITER_FLOW= VENKATAKRISHNAN" -- invper_ausm
# --- step 5: A/B + poison ---
g1 ab-per-r1       RUN_TAG=_perab    SU2_KOKKOS_STAGE_VERIFY=1 -- invper_ausm
g1 ab-per-muscl-r1 RUN_TAG=_permusclab SU2_KOKKOS_STAGE_VERIFY=1 "EXTRA_CFG=MUSCL_FLOW= YES;SLOPE_LIMITER_FLOW= VAN_ALBADA_EDGE" -- invper_ausm
g1 poison-per-r1   RUN_TAG=_perpois  SU2_KOKKOS_NEMO_POISON=1 -- invper_ausm
# --- step 6: regressions ---
g1 g1-per-regress-r1 -- invwedge_ausm thermalbath super_cat axi_visccone invwedge_lusgs invwedge_mg invwedge_dyn supin_visc
# --- step 7: noise floor ---
step nf-per-r1 3600 env BIN="$SERIAL" BIN_STRICT="$STRICT" CASE=invper_ausm K=10 bash TestCases/kokkos_nemo/check_noisefloor.sh
echo "$(date -Is) QUEUE FINISHED" >> "$ST"
