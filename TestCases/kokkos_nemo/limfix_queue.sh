#!/usr/bin/env bash
# 4.6-PeriodicLim limValid-fix validation queue (review 41, steps 2-6; build = round-199 bg build),
# DETACHED and RESUMABLE. Writes every log into validation_logs/2026-11-28_4.6-PeriodicLim-fix/
# and one line per finished step into QUEUE_STATUS.txt there. A step whose .done marker exists is
# skipped, so re-launching resumes. Launch: bash TestCases/kokkos_nemo/limfix_queue.sh
set -u
cd "$(dirname "$0")/../.." || exit 2
SU2="$PWD"
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export SU2_ROOT="$SU2"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
OUT="$SU2/validation_logs/2026-11-28_4.6-PeriodicLim-fix"
ST="$OUT/QUEUE_STATUS.txt"
STRICT="$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD"
mkdir -p "$OUT" "$RUN_DIR"
LOCK="$OUT/.queue.lock"
exec 9> "$LOCK"; flock -n 9 || { echo "limfix_queue already running"; exit 0; }
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

# --- step 2: THE previously-failing limoff control (limiter host + device periodic comms) ---
g1 g1-limfix-off-r1 RUN_TAG=_limoff SU2_KOKKOS_NEMO_PERIODIC_LIM=0 -- invper_lim
# --- step 3: old periodic ausm case + MUSCL/Venkat limiter + limoff ---
g1 g1-limfix-pervenkoff-r1 RUN_TAG=_limoff SU2_KOKKOS_NEMO_PERIODIC_LIM=0 \
  "EXTRA_CFG=MUSCL_FLOW= YES;SLOPE_LIMITER_FLOW= VENKATAKRISHNAN" -- invper_ausm
# --- step 4: host-limiter control, NON-periodic (device upwind + host limiter) ---
g1 g1-limfix-hostlim-r1 RUN_TAG=_hostlim SU2_KOKKOS_NEMO_HOST=lim -- invwedge_roe
# --- step 5: device path + A/B stage-verify ---
g1 g1-limfix-r1 -- invper_lim
g1 ab-limfix-r1 RUN_TAG=_limab SU2_KOKKOS_STAGE_VERIFY=1 -- invper_lim
# --- step 6: regress 13 (same list as lim_queue g1-regress-r1) ---
step g1-limfix-regress-r1 3600 env BIN="$STRICT" REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 \
  bash TestCases/kokkos_nemo/check.sh invwedge_ausm thermalbath super_cat axi_visccone invwedge_lusgs \
    invwedge_mg invwedge_dyn supin_visc invper_ausm viscper_ausm invper_dt invper_ausmm invper_lim
echo "$(date -Is) QUEUE FINISHED" >> "$ST"
