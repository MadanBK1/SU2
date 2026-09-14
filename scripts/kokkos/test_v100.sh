#!/usr/bin/env bash
set -euo pipefail

# Correctness and parallel validation for the current SU2 Kokkos branch.
# This script validates:
#   1) 1 MPI rank, CPU path
#   2) 1 MPI rank, 1 V100 Kokkos path
#   3) 4 MPI ranks, CPU path
#   4) 4 MPI ranks, 4 V100 Kokkos path
#
# IMPORTANT: this does NOT claim direct GPU-aware MPI yet. The current port
# still uses the existing SU2 communication path around the Kokkos SpMV.
# Direct device-buffer MPI should only be marked validated after the new
# GPU-aware halo-exchange implementation is added and tested separately.
#
# Usage:
#   export SU2_RUN=$HOME/SU2_RUN_8_5_KOKKOS_V100/bin
#   export CASE_ROOT=$HOME/Tutorials/SU2_Tutorials/compressible_flow/Laminar_Flat_Plate
#   bash scripts/kokkos/test_v100.sh

SU2_RUN="${SU2_RUN:-$HOME/SU2_RUN_8_5_KOKKOS_V100/bin}"
CASE_ROOT="${CASE_ROOT:-$HOME/Tutorials/SU2_Tutorials/compressible_flow/Laminar_Flat_Plate}"
NP="${NP:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
OUT_ROOT="${OUT_ROOT:-$CASE_ROOT/comparison-v100-full}"

SU2_CFD="$SU2_RUN/SU2_CFD"
CFG="$CASE_ROOT/lam_flatplate.cfg"
MESH="$CASE_ROOT/mesh_flatplate_65x65.su2"

[[ -x "$SU2_CFD" ]] || { echo "ERROR: SU2_CFD not found: $SU2_CFD" >&2; exit 2; }
[[ -f "$CFG" ]] || { echo "ERROR: case config not found: $CFG" >&2; exit 2; }
[[ -f "$MESH" ]] || { echo "ERROR: mesh not found: $MESH" >&2; exit 2; }

mkdir -p "$OUT_ROOT"/{cpu1,gpu1,cpu4,gpu4}

prepare_case() {
  local dir="$1"
  local kokkos="$2"
  cp "$CFG" "$dir/lam_flatplate.cfg"
  cp "$MESH" "$dir/mesh_flatplate_65x65.su2"
  if grep -q '^[[:space:]]*ENABLE_KOKKOS' "$dir/lam_flatplate.cfg"; then
    sed -i "s/^[[:space:]]*ENABLE_KOKKOS.*/ENABLE_KOKKOS= $kokkos/" "$dir/lam_flatplate.cfg"
  else
    echo "ENABLE_KOKKOS= $kokkos" >> "$dir/lam_flatplate.cfg"
  fi
}

run_case() {
  local name="$1"
  local dir="$2"
  shift 2
  echo
  echo "========== $name =========="
  cd "$dir"
  rm -f history.csv restart_flow.dat flow.vtu surface_flow.vtu *.log *-timing.txt || true
  /usr/bin/time -f "Elapsed=%e\nCPU=%P\nMemory=%M KB" -o "$name-timing.txt" "$@" > "$name.log" 2>&1
  local rc=$?
  echo "$name status: $rc"
  grep -E 'All convergence criteria satisfied|Exit Success' "$name.log" || true
  [[ $rc -eq 0 ]] || { tail -n 80 "$name.log"; exit "$rc"; }
  grep -q 'Exit Success (SU2_CFD)' "$name.log" || { echo "ERROR: $name did not report Exit Success" >&2; tail -n 80 "$name.log"; exit 3; }
  tail -n 1 history.csv
  cat "$name-timing.txt"
}

prepare_case "$OUT_ROOT/cpu1" NO
prepare_case "$OUT_ROOT/gpu1" YES
prepare_case "$OUT_ROOT/cpu4" NO
prepare_case "$OUT_ROOT/gpu4" YES

unset CUDA_VISIBLE_DEVICES KOKKOS_PRINT_CONFIGURATION KOKKOS_MAP_DEVICE_ID_BY || true
run_case cpu1 "$OUT_ROOT/cpu1" \
  mpirun --bind-to none -np 1 "$SU2_CFD" lam_flatplate.cfg

export CUDA_VISIBLE_DEVICES=0
export KOKKOS_MAP_DEVICE_ID_BY=mpi_rank
export KOKKOS_PRINT_CONFIGURATION=1
run_case gpu1 "$OUT_ROOT/gpu1" \
  mpirun --bind-to none -np 1 \
  -x CUDA_VISIBLE_DEVICES -x KOKKOS_MAP_DEVICE_ID_BY -x KOKKOS_PRINT_CONFIGURATION \
  "$SU2_CFD" lam_flatplate.cfg

echo "Single-GPU selection:"
grep 'Kokkos::Cuda' "$OUT_ROOT/gpu1/gpu1.log" || true

unset CUDA_VISIBLE_DEVICES KOKKOS_PRINT_CONFIGURATION KOKKOS_MAP_DEVICE_ID_BY || true
run_case cpu4 "$OUT_ROOT/cpu4" \
  mpirun --map-by core --bind-to core -np "$NP" "$SU2_CFD" lam_flatplate.cfg

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export KOKKOS_MAP_DEVICE_ID_BY=mpi_rank
export KOKKOS_PRINT_CONFIGURATION=1
run_case gpu4 "$OUT_ROOT/gpu4" \
  mpirun --map-by slot --bind-to none -np "$NP" \
  -x CUDA_VISIBLE_DEVICES -x KOKKOS_MAP_DEVICE_ID_BY -x KOKKOS_PRINT_CONFIGURATION \
  "$SU2_CFD" lam_flatplate.cfg

echo "Multi-GPU selections:"
grep 'Kokkos::Cuda' "$OUT_ROOT/gpu4/gpu4.log" || true

python3 - "$OUT_ROOT" <<'PY'
import csv, pathlib, sys
root = pathlib.Path(sys.argv[1])

def last_row(case):
    p = root/case/'history.csv'
    with p.open(newline='') as f:
        rows = list(csv.reader(f))
    data = [r for r in rows if r and any(x.strip() for x in r)]
    return [x.strip() for x in data[-1]]

for a,b in [('cpu1','gpu1'),('cpu4','gpu4')]:
    ra, rb = last_row(a), last_row(b)
    print(f'\n{a} final: {ra}')
    print(f'{b} final: {rb}')
    try:
        va = [float(x) for x in ra]
        vb = [float(x) for x in rb]
        if len(va) == len(vb):
            diffs = [abs(x-y) for x,y in zip(va,vb)]
            print(f'{a} vs {b} max absolute field difference: {max(diffs):.6e}')
    except ValueError:
        pass
PY

echo
echo "========== VALIDATION SUMMARY =========="
for c in cpu1 gpu1 cpu4 gpu4; do
  if grep -q 'Exit Success (SU2_CFD)' "$OUT_ROOT/$c/$c.log"; then
    printf '%-8s PASS\n' "$c"
  else
    printf '%-8s FAIL\n' "$c"
  fi
done

echo
echo "Validated here: CPU MPI, single V100 Kokkos, and one-MPI-rank-per-GPU multi-V100 execution."
echo "Not validated by this script: direct CUDA-device MPI buffers or multi-node GPU-aware MPI."
