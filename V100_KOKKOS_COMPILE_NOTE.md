# SU2 8.5 with Kokkos 5.2.2 on NVIDIA V100

This note records the build and validation procedure used on an x86_64 node
with four NVIDIA Tesla V100S PCIe 32 GB GPUs. It covers the
`kokkos-linear-solver` branch of this repository, CUDA 12.9.1, GCC 12.3.0,
OpenMPI 4.1.5, and the Kokkos `VOLTA70` target.

## Configuration summary

| Component | Validated value |
|---|---|
| SU2 | 8.5.0 "Harrier" |
| Kokkos | 5.2.2, vendored in `subprojects/Kokkos` |
| CPU architecture | x86_64 |
| GPU | NVIDIA Tesla V100S PCIe 32 GB |
| Number of GPUs tested | 4 |
| GPU architecture | Volta 70 (`sm_70`) |
| CUDA toolkit | 12.9.1 (`nvcc` 12.9.86) |
| GCC host compiler | 12.3.0 |
| CMake | 3.26.3 |
| MPI | OpenMPI 4.1.5 built with GCC 12.3.0 |

The current port accelerates SU2's block-CSR matrix-vector product. FGMRES,
the ILU preconditioner, MPI halo exchange, and most other SU2 operations remain
host-resident. Matrix and vector data are copied between host and device for
each matrix-vector product, so this first port is primarily a correctness and
integration implementation.

## 1. Clone the branch

For a fresh checkout:

```bash
cd "$HOME"

git clone \
  --branch kokkos-linear-solver \
  --recurse-submodules \
  https://github.com/MadanBK1/SU2.git \
  SU2_8_5_Kokkos_V100

cd "$HOME/SU2_8_5_Kokkos_V100"
git submodule update --init --recursive
chmod u+x meson_scripts/kokkos_compiler_dispatcher
```

For an existing checkout:

```bash
cd "$HOME/SU2_8_5_Kokkos_V100"
git switch kokkos-linear-solver
git pull --ff-only origin kokkos-linear-solver
git submodule update --init --recursive
chmod u+x meson_scripts/kokkos_compiler_dispatcher
```

## 2. Load the x86_64 V100 environment

Use a clean compiler and MPI environment. Avoid mixing system GCC, NVHPC,
Cray MPI, or a different OpenMPI installation with this build.

```bash
module purge

unset OMPI_CC OMPI_CXX OMPI_FC
unset MPICH_CC MPICH_CXX MPICH_FC
unset NVCC_WRAPPER_DEFAULT_COMPILER
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LD_PRELOAD

module load GCC/12.3.0
module load OpenMPI/4.1.5-GCC-12.3.0
module load CMake/3.26.3-GCCcore-12.3.0
module load CUDA/12.9.1

# Optional: activate the Python environment containing Meson.
# source "$HOME/venv/bin/activate"

export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export GNU_GCC="$(command -v gcc)"
export GNU_GXX="$(command -v g++)"

export CC="$GNU_GCC"
export SU2_KOKKOS_HOST_COMPILER="$GNU_GXX"
export NVCC_WRAPPER_DEFAULT_COMPILER="$GNU_GXX"
export SU2_KOKKOS_CUDA_ARCH=sm_70
export CXX="$PWD/meson_scripts/kokkos_compiler_dispatcher"

unset SU2_KOKKOS_GCC_TOOLCHAIN
```

Verify that all tools belong to the intended environment:

```bash
which gcc g++ mpicc mpicxx nvcc cmake python3
gcc --version
g++ --version
mpirun --version
nvcc --version
cmake --version
python3 --version
"$CXX" --version
```

The validated compiler versions were GCC 12.3.0 and CUDA 12.9.86. The
dispatcher sends ordinary SU2 C++ sources to GCC and CUDA-dependent Kokkos
translation units through Kokkos's `nvcc_wrapper`.

## 3. Configure

The legacy SU2 CUDA backend and the Kokkos CUDA backend are mutually
exclusive. Use `-Denable-cuda=false` for this build.

```bash
cd "$HOME/SU2_8_5_Kokkos_V100"

export SU2_PREFIX="$HOME/SU2_RUN_8_5_KOKKOS_V100"

rm -rf build-kokkos-v100

python3 meson.py setup build-kokkos-v100 \
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
```

The configuration summary should report:

```text
Kokkos:         true
Kokkos Backend: cuda
Kokkos Arch:    VOLTA70
AD (reverse):   false
AD (forward):   false
```

CUDA 12.9 warns that offline compilation for architectures older than 7.5
will be removed in a future release. The V100 requires `sm_70`, so the warning
is expected and was nonfatal for this validated build.

## 4. Build and install

```bash
./ninja -C build-kokkos-v100 \
  -j "${SLURM_CPUS_PER_TASK:-8}" \
  2>&1 | tee build-kokkos-v100.log

build_status=${PIPESTATUS[0]}
echo "build status: $build_status"

if [ "$build_status" -eq 0 ]; then
  ./ninja -C build-kokkos-v100 install
fi
```

The validated build and installation both returned status zero. The install
placed SU2 executables in `$SU2_PREFIX/bin` and static Kokkos libraries in the
installation library directory.

## 5. Runtime environment

```bash
export SU2_HOME="$HOME/SU2_8_5_Kokkos_V100"
export SU2_RUN="$HOME/SU2_RUN_8_5_KOKKOS_V100/bin"
export PATH="$SU2_RUN:$PATH"
export PYTHONPATH="$SU2_RUN:${PYTHONPATH:-}"
```

Verify the installed executable and MPI linkage:

```bash
which SU2_CFD
mpirun --version
ldd "$SU2_RUN/SU2_CFD" | grep -E 'libmpi|libopen'
```

The validated executable linked to OpenMPI 4.1.5 from the same GCC 12.3.0
software stack used during configuration.

## 6. Enable Kokkos in an SU2 case

Add the following option to the case configuration:

```text
ENABLE_KOKKOS= YES
```

The validated linear-solver settings were:

```text
LINEAR_SOLVER= FGMRES
ENABLE_KOKKOS= YES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 10
```

Set `ENABLE_KOKKOS= NO` to exercise the CPU path in the same executable. Use
separate directories for CPU and Kokkos runs so their history, restart, and
visualization files do not overwrite one another.

## 7. Single-rank validation

The `Laminar_Flat_Plate` tutorial with the 65 by 65 mesh was tested with one
MPI rank on the CPU path and one MPI rank on one V100.

For the GPU run:

```bash
export KOKKOS_PRINT_CONFIGURATION=1
export KOKKOS_MAP_DEVICE_ID_BY=mpi_rank

/usr/bin/time \
  -f "Elapsed=%e\nCPU=%P\nMemory=%M KB" \
  -o kokkos-timing.txt \
  mpirun --bind-to none -np 1 \
  -x CUDA_VISIBLE_DEVICES=0 \
  -x KOKKOS_PRINT_CONFIGURATION \
  -x KOKKOS_MAP_DEVICE_ID_BY \
  "$SU2_RUN/SU2_CFD" lam_flatplate.cfg \
  > kokkos.log 2>&1
```

Both paths converged successfully at iteration 1599.

| Metric | CPU path | Kokkos/V100 path |
|---|---:|---:|
| Final `rms[Rho]` | -12.04695440 | -12.04696655 |
| Iteration | 1599 | 1599 |
| Wall time | 28.08 s | 34.89 s |
| Maximum memory | 416692 KiB | 417000 KiB |

The difference in final `rms[Rho]` was approximately `1.22e-5`. The V100 run
was approximately 24.3% slower for this very small mesh because CPU-device
copies and kernel-launch overhead dominated the limited SpMV work. This test
establishes numerical consistency, not representative GPU performance.

The Kokkos log confirmed:

```text
Kokkos Version: 5.2.2
Default Device: Cuda
KOKKOS_ENABLE_CUDA: yes
Kokkos::Cuda[ 0 ] Tesla V100S-PCIE-32GB : Selected
```

## 8. Four-rank CPU and four-GPU validation

Four CPU MPI ranks were launched with:

```bash
mpirun --map-by core --bind-to core -np 4 \
  "$SU2_RUN/SU2_CFD" lam_flatplate.cfg
```

Four MPI ranks were mapped to four V100 GPUs with:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export KOKKOS_MAP_DEVICE_ID_BY=mpi_rank
export KOKKOS_PRINT_CONFIGURATION=1

mpirun --map-by slot --bind-to none -np 4 \
  -x CUDA_VISIBLE_DEVICES \
  -x KOKKOS_MAP_DEVICE_ID_BY \
  -x KOKKOS_PRINT_CONFIGURATION \
  "$SU2_RUN/SU2_CFD" lam_flatplate.cfg
```

Both four-rank cases converged successfully.

| Metric | 4 CPU MPI ranks | 4 MPI ranks / 4 V100 GPUs |
|---|---:|---:|
| Final `rms[Rho]` | -12.00023601 | -12.00023581 |
| Reported iteration | 6239 | 6239 |
| Wall time | 33.84 s | 50.31 s |
| Aggregate CPU utilization | 383% | 390% |
| Maximum memory | 291780 KiB | 292576 KiB |

The final `rms[Rho]` values differed by approximately `2.0e-7`. Kokkos
reported devices 0, 1, 2, and 3 as selected exactly once, confirming one
distinct V100 per local MPI rank. Rank output was interleaved, so the selected
devices did not appear in numerical order in the combined log.

The four-GPU run was approximately 48.7% slower than the four-rank CPU run for
this small mesh. Each rank owned only a small subdomain, making transfer and
launch overhead larger than the accelerated computation. A substantially
larger mesh and repeated runs are required for a meaningful performance or
scaling assessment.

## 9. Capability status

| Capability | Status |
|---|---|
| x86_64 compilation | Validated |
| CUDA 12.9 compilation for `sm_70` | Validated |
| Kokkos 5.2.2 CUDA runtime | Validated |
| Single CPU rank | Validated |
| Single V100 GPU | Validated |
| Single-node CPU MPI | Validated with 4 ranks |
| Single-node multi-GPU | Validated with 4 ranks and 4 GPUs |
| Multi-node MPI | Not yet validated on this system |
| Direct GPU-aware MPI buffers | Not implemented in the current port |

The current MPI halo exchange uses host-resident SU2 buffers. Although the MPI
library may itself support CUDA-aware communication, this port does not yet
pass Kokkos device pointers directly to MPI. Multi-GPU execution therefore
means one MPI rank per GPU with host-side halo communication.

## 10. Troubleshooting notes

- `cc1plus: error: to generate dependencies you must specify either -M or
  -MM`: update the branch and ensure the current compiler dispatcher is used.
  It translates Meson's `-MQ` dependency target option into the form accepted
  by `nvcc_wrapper`.
- `KOKKOS_ENABLE_CUDA defined but ... __CUDACC__`: ensure `CXX` points to
  `meson_scripts/kokkos_compiler_dispatcher`, not directly to `g++`.
- An unknown `-Wext-lambda-captures-this`, `-fast`, or `-ffast-math` option
  indicates that flags are reaching the wrong compiler. Update the branch and
  configure from a new build directory.
- Do not set `CUDA_VISIBLE_DEVICES=0` for a four-GPU run. That makes every rank
  see only GPU 0. Use `0,1,2,3` or leave all allocated GPUs visible.
- The CUDA warning about pre-7.5 offline compilation is expected for V100
  (`sm_70`) with CUDA 12.9.
- Compare CPU and Kokkos results at the same MPI rank count. Changing the
  number of domain partitions can change Krylov/preconditioner convergence.
- Use a larger mesh before drawing performance conclusions.
