#!/usr/bin/env bash
set -euo pipefail

# Validate SU2's direct Kokkos device-buffer MPI path on one 4x V100 node.
# This is intentionally separate from test_v100.sh: a successful multi-GPU run
# does not by itself prove that MPI accepted CUDA device pointers.
#
# Required:
#   export SU2_RUN=$HOME/SU2_RUN_8_5_KOKKOS_V100/bin
#   export CASE_ROOT=$HOME/Tutorials/SU2_Tutorials/compressible_flow/Laminar_Flat_Plate
#   bash scripts/kokkos/test_gpu_aware_mpi_v100.sh

: "${SU2_RUN:?Set SU2_RUN to the installed SU2 bin directory}"
: "${CASE_ROOT:?Set CASE_ROOT to the Laminar_Flat_Plate tutorial directory}"

SU2_CFD="$SU2_RUN/SU2_CFD"
SRC_CFG="$CASE_ROOT/lam_flatplate.cfg"
SRC_MESH="$CASE_ROOT/mesh_flatplate_65x65.su2"
OUT_ROOT="${OUT_ROOT:-$CASE_ROOT/comparison-v100-gpu-aware}"
NP="${NP:-4}"

[[ -x "$SU2_CFD" ]] || { echo "ERROR: $SU2_CFD is not executable" >&2; exit 2; }
[[ -f "$SRC_CFG" ]] || { echo "ERROR: missing $SRC_CFG" >&2; exit 2; }
[[ -f "$SRC_MESH" ]] || { echo "ERROR: missing $SRC_MESH" >&2; exit 2; }
command -v mpirun >/dev/null || { echo "ERROR: mpirun not found" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found" >&2; exit 2; }

mkdir -p "$OUT_ROOT/host-staged" "$OUT_ROOT/gpu-aware"

set_cfg_bool() {
  local file="$1" key="$2" value="$3"
  if grep -q "^[[:space:]]*${key}[[:space:]]*=" "$file"; then
    sed -i "s/^[[:space:]]*${key}[[:space:]]*=.*/${key}= ${value}/" "$file"
  else
    printf '\n%s= %s\n' "$key" "$value" >> "$file"
  fi
}

prepare_case() {
  local dir="$1" gpu_aware="$2"
  cp "$SRC_CFG" "$dir/lam_flatplate.cfg"
  cp "$SRC_MESH" "$dir/mesh_flatplate_65x65.su2"
  set_cfg_bool "$dir/lam_flatplate.cfg" ENABLE_KOKKOS YES
  set_cfg_bool "$dir/lam_flatplate.cfg" KOKKOS_GPU_AWARE_MPI "$gpu_aware"
}

prepare_case "$OUT_ROOT/host-staged" NO
prepare_case "$OUT_ROOT/gpu-aware" YES

echo "========== MPI / GPU environment =========="
mpirun --version | head -n 2 || true
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
if command -v ompi_info >/dev/null 2>&1; then
  echo
  echo "OpenMPI CUDA-awareness metadata (informational):"
  ompi_info --parsable -l 9 --all 2>/dev/null | \
    grep -Ei 'mpi_built_with_cuda_support|cuda_support|smcuda|accelerator' | head -n 40 || true
fi

echo
printf 'Config host-staged: '
grep -E '^(ENABLE_KOKKOS|KOKKOS_GPU_AWARE_MPI)[[:space:]]*=' "$OUT_ROOT/host-staged/lam_flatplate.cfg" | tr '\n' ' '; echo
printf 'Config gpu-aware:   '
grep -E '^(ENABLE_KOKKOS|KOKKOS_GPU_AWARE_MPI)[[:space:]]*=' "$OUT_ROOT/gpu-aware/lam_flatplate.cfg" | tr '\n' ' '; echo

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export KOKKOS_MAP_DEVICE_ID_BY=mpi_rank
export KOKKOS_PRINT_CONFIGURATION=1

run_case() {
  local name="$1" dir="$2"
  echo
  echo "========== $name =========="
  cd "$dir"
  rm -f history.csv restart_flow.dat flow*.vtu surface_flow*.vtp

  set +e
  /usr/bin/time -f "Elapsed=%e\nCPU=%P\nMemory=%M KB" \
    -o timing.txt \
    mpirun --map-by slot --bind-to none -np "$NP" \
      -x CUDA_VISIBLE_DEVICES \
      -x KOKKOS_MAP_DEVICE_ID_BY \
      -x KOKKOS_PRINT_CONFIGURATION \
      "$SU2_CFD" lam_flatplate.cfg \
      > run.log 2>&1
  local status=$?
  set -e

  echo "$status" > status.txt
  echo "$name status: $status"
  grep -E 'All convergence|Exit Success|Kokkos::Cuda.*Selected' run.log || true
  [[ -f history.csv ]] && tail -n 1 history.csv || true
  [[ -f timing.txt ]] && cat timing.txt || true

  return "$status"
}

host_status=0
gpu_status=0
run_case host-staged "$OUT_ROOT/host-staged" || host_status=$?
run_case gpu-aware "$OUT_ROOT/gpu-aware" || gpu_status=$?

echo
echo "========== RESULT COMPARISON =========="
python3 - "$OUT_ROOT/host-staged/history.csv" "$OUT_ROOT/gpu-aware/history.csv" <<'PY'
import csv, math, pathlib, sys

def last_row(path):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    rows=[]
    with p.open(newline='') as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith('#'):
                continue
            try:
                vals=[float(x.strip()) for x in row]
            except ValueError:
                continue
            rows.append(vals)
    return rows[-1] if rows else None

a=last_row(sys.argv[1]); b=last_row(sys.argv[2])
print('host-staged final:', a)
print('gpu-aware final:  ', b)
if a is not None and b is not None and len(a)==len(b):
    diffs=[abs(x-y) for x,y in zip(a,b)]
    print(f'max absolute field difference: {max(diffs):.6e}')
else:
    print('comparison unavailable')
PY

echo
echo "========== GPU-AWARE MPI VALIDATION SUMMARY =========="necho "host-staged status: $host_status"
echo "gpu-aware   status: $gpu_status"

if [[ $host_status -eq 0 && $gpu_status -eq 0 ]] && \
   grep -q 'Exit Success (SU2_CFD)' "$OUT_ROOT/gpu-aware/run.log"; then
  echo "GPU-aware MPI device-buffer run: PASS"
  echo "The KOKKOS_GPU_AWARE_MPI=YES branch completed with four MPI ranks / four V100 GPUs."
  exit 0
fi

echo "GPU-aware MPI device-buffer run: FAIL"
echo "Inspect: $OUT_ROOT/gpu-aware/run.log"
echo "A failure here can indicate either an SU2 GPU-aware path bug or an OpenMPI build/runtime without CUDA-device-pointer support."
exit 1
