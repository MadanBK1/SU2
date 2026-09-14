#!/usr/bin/env bash
set -euo pipefail

# Reproducible SU2 8.5 + Kokkos/CUDA build for the x86_64 Tesla V100S node
# validated on dev-amd20-v100 with GCC 12.3, OpenMPI 4.1.5 and CUDA 12.9.1.
#
# Usage:
#   cd ~/SU2_8_5_Kokkos_V100
#   bash scripts/kokkos/build_v100.sh
#
# Optional overrides:
#   SU2_PREFIX=$HOME/SU2_RUN_8_5_KOKKOS_V100
#   BUILD_DIR=build-kokkos-v100
#   JOBS=16
#   CLEAN_BUILD=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SU2_PREFIX="${SU2_PREFIX:-$HOME/SU2_RUN_8_5_KOKKOS_V100}"
BUILD_DIR="${BUILD_DIR:-build-kokkos-v100}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
CLEAN_BUILD="${CLEAN_BUILD:-0}"

if command -v module >/dev/null 2>&1; then
  module purge
  unset OMPI_CC OMPI_CXX OMPI_FC MPICH_CC MPICH_CXX MPICH_FC || true
  unset NVCC_WRAPPER_DEFAULT_COMPILER CC CXX CFLAGS CXXFLAGS CPPFLAGS LD_PRELOAD || true

  module load GCC/12.3.0
  module load OpenMPI/4.1.5-GCC-12.3.0
  module load CMake/3.26.3-GCCcore-12.3.0
  module load CUDA/12.9.1
fi

export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export GNU_GCC="$(command -v gcc)"
export GNU_GXX="$(command -v g++)"
export CC="$GNU_GCC"
export SU2_KOKKOS_HOST_COMPILER="$GNU_GXX"
export NVCC_WRAPPER_DEFAULT_COMPILER="$GNU_GXX"
export SU2_KOKKOS_CUDA_ARCH="${SU2_KOKKOS_CUDA_ARCH:-sm_70}"
export CXX="$REPO_ROOT/meson_scripts/kokkos_compiler_dispatcher"
unset SU2_KOKKOS_GCC_TOOLCHAIN CXXFLAGS CPPFLAGS || true
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
chmod u+x "$CXX"

echo "=== Toolchain ==="
uname -m
hostname
gcc --version | head -n 1
g++ --version | head -n 1
nvcc --version | tail -n 1
mpirun --version | head -n 1
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv

git submodule update --init --recursive

if [[ "$CLEAN_BUILD" == "1" ]]; then
  rm -rf "$BUILD_DIR"
fi

if [[ ! -f "$BUILD_DIR/build.ninja" ]]; then
  python3 meson.py setup "$BUILD_DIR" \
    -Denable-kokkos=true \
    -Dkokkos-backend=cuda \
    -Dkokkos-arch=VOLTA70 \
    -Denable-cuda=false \
    -Dwith-mpi=enabled \
    -Dwith-omp=false \
    -Denable-autodiff=false \
    -Denable-directdiff=false \
    -Denable-mlpcpp=true \
    --prefix="$SU2_PREFIX"
else
  echo "Reusing existing Meson build directory: $BUILD_DIR"
fi

./ninja -C "$BUILD_DIR" -j "$JOBS" 2>&1 | tee "$BUILD_DIR/build.log"
build_status=${PIPESTATUS[0]}
if [[ $build_status -ne 0 ]]; then
  echo "ERROR: build failed with status $build_status" >&2
  grep -nEi 'FAILED:|catastrophic error:|fatal error:|[^[:alpha:]]error:' "$BUILD_DIR/build.log" | head -n 100 || true
  exit "$build_status"
fi

./ninja -C "$BUILD_DIR" install 2>&1 | tee "$BUILD_DIR/install.log"
install_status=${PIPESTATUS[0]}
if [[ $install_status -ne 0 ]]; then
  echo "ERROR: install failed with status $install_status" >&2
  exit "$install_status"
fi

export SU2_HOME="$REPO_ROOT"
export SU2_RUN="$SU2_PREFIX/bin"
export PATH="$SU2_RUN:$PATH"
export PYTHONPATH="$SU2_RUN:${PYTHONPATH:-}"

echo "=== Installed executable ==="
command -v SU2_CFD
ldd "$SU2_RUN/SU2_CFD" | grep -E 'libmpi|libopen' || true

echo "=== Build complete ==="
echo "SU2_HOME=$SU2_HOME"
echo "SU2_RUN=$SU2_RUN"
echo "To use in a new shell:"
echo "  export SU2_RUN=\"$SU2_RUN\""
echo '  export PATH="$SU2_RUN:$PATH"'
echo '  export PYTHONPATH="$SU2_RUN:${PYTHONPATH:-}"'
