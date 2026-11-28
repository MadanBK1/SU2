#!/usr/bin/env bash
# zc2a-bndsync validation queue (review 42 B1-B5; written by the reviewer in review 44),
# DETACHED and RESUMABLE, same pattern as limfix_queue.sh. Logs -> validation_logs/2026-11-28_zc2a-bndsync/,
# one line per finished step in QUEUE_STATUS.txt. A step whose .done marker exists is skipped.
# Launch: setsid nohup bash TestCases/kokkos_nemo/bnd_queue.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.." || exit 2
SU2="$PWD"
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export SU2_ROOT="$SU2"
export MESH_DIR="$HOME/software/AgenticTest/kk/meshes"
export RUN_DIR="$HOME/software/AgenticTest/kk/runs"
OUT="$SU2/validation_logs/2026-11-28_zc2a-bndsync"
ST="$OUT/QUEUE_STATUS.txt"
STRICT="$SU2/build-kokkos-strict/SU2_CFD/src/SU2_CFD"
mkdir -p "$OUT" "$RUN_DIR"
exec 9> "$OUT/.queue.lock"; flock -n 9 || { echo "bnd_queue already running"; exit 0; }
sha=$(git rev-parse --short HEAD)

step() {
  local name=$1 to=$2; shift 2
  [ -e "$OUT/.$name.done" ] && return 0
  echo "$(date -Is) START $name (code $sha)" >> "$ST"
  local t0=$(date +%s)
  timeout "$to" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  local summ
  summ=$(grep -E "OVERALL|expected stage line missing|BUILD rc|HIST" "$OUT/$name.log" | tail -3 | tr '\n' ' ')
  echo "$(date -Is) END   $name rc=$rc wall=$(( $(date +%s) - t0 ))s :: ${summ:-<no summary line; see log>}" >> "$ST"
  touch "$OUT/.$name.done"
}
g1() {  # g1 <name> [ENV=VAL ...] -- <cases...>
  local name=$1; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  step "$name" 1800 env BIN="$STRICT" REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 "${envs[@]}" \
    bash TestCases/kokkos_nemo/check.sh "$@"
}
bld() { ninja -C "$1" -j10 SU2_CFD/src/SU2_CFD; echo "BUILD rc=$? ($1, code $sha)"; }
hist() {  # B3: history of the HOST reference vs the device run, every column except time columns
  for c in axi_visccone invwedge_lax; do
    for tag in _bnd _bndall; do
      a="$RUN_DIR/chk_${c}${tag}_NO"; b="$RUN_DIR/chk_${c}${tag}_YES"
      fa=$(ls "$a"/history*.csv 2>/dev/null | head -1); fb=$(ls "$b"/history*.csv 2>/dev/null | head -1)
      [ -n "$fa" ] && [ -n "$fb" ] || { echo "HIST $c$tag: history file missing ($a / $b)"; continue; }
      python3 -I - "$fa" "$fb" "$c$tag" <<'EOF'
import csv, sys
def load(p):
    with open(p) as f:
        r = [[x.strip().strip('"') for x in row] for row in csv.reader(f) if row]
    return r[0], r[1:]
ha, ra = load(sys.argv[1]); hb, rb = load(sys.argv[2])
keep = [i for i, h in enumerate(ha) if 'time' not in h.lower()]
bad = [ha[i] for i in keep if i >= len(hb) or hb[i] != ha[i] or len(ra) != len(rb) or any(x[i] != y[i] for x, y in zip(ra, rb))]
print("HIST %s: %s, %d columns x %d rows compared%s" % (sys.argv[3], "IDENTICAL" if not bad else "DIFFER", len(keep), len(ra),
      "" if not bad else " -> differing: " + ",".join(bad[:20])))
EOF
    done
  done
}
ledger() {  # B4: steady-iteration ledger of the device runs
  for c in axi_visccone invwedge_lax; do for tag in _bnd _bndall; do
    echo "== $c$tag"; grep -E "KOKKOS_NEMO_LEDGER.*(steady iteration|oundary)|marker points" "$RUN_DIR/chk_${c}${tag}_YES/log.txt" | head -20
  done; done
}

step build-bnd-r1        3600 bld build-kokkos-strict
step build-bnd-serial-r1 3600 bld build-kokkos-serial
if grep -q "BUILD rc=0" "$OUT/build-bnd-r1.log" && grep -q "BUILD rc=0" "$OUT/build-bnd-serial-r1.log"; then
  g1 g1-bnd-r1    RUN_TAG=_bnd -- axi_visccone invwedge_lax
  g1 g1-bndall-r1 RUN_TAG=_bndall SU2_KOKKOS_NEMO_BND_ALL=1 -- axi_visccone invwedge_lax
  step g1-bnd-hist-r1 300 hist
  step ledger-bnd-r1  300 ledger
  step g1-bnd-regress-r1 3600 env BIN="$STRICT" REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 \
    bash TestCases/kokkos_nemo/check.sh invwedge_ausm thermalbath super_cat axi_visccone invwedge_lusgs \
      invwedge_mg invwedge_dyn supin_visc invper_ausm viscper_ausm invper_dt invper_ausmm invper_lim
  echo "$(date -Is) QUEUE FINISHED" >> "$ST"
else
  echo "$(date -Is) BUILD FAILED - queue stopped (see build-bnd*.log); fix, then delete .build-bnd*.done and relaunch" >> "$ST"
fi
