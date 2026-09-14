# SU2 8.5 with Kokkos 5.2.2 on NVIDIA GH200

This note records the build and runtime procedure validated on an NCSA
DeltaAI GH200 node (`aarch64`) using NVIDIA HPC SDK 25.5, CUDA 12.9,
GCC 14.2, HPC-X OpenMPI, and the `kokkos-linear-solver` branch of this
repository.

## Configuration summary

| Component | Validated value |
|---|---|
| SU2 | 8.5.0 "Harrier" |
| Kokkos | 5.2.2, vendored in `subprojects/Kokkos` |
| GPU | NVIDIA GH200 |
| GPU architecture | Hopper 90 (`sm_90`) |
| CUDA toolkit | 12.9.41 |
| NVIDIA HPC SDK | 25.5 |
| GCC host toolchain | 14.2.0 |
| CMake | 3.28.3 |
| Python | 3.11.9 |
| MPI used by this build | HPC-X OpenMPI 4.1.7rc1 |

The current Kokkos port accelerates the block-CSR matrix-vector product.
The Krylov solver, ILU preconditioner, MPI halo exchange, and most other SU2
operations remain host-resident. Matrix and vector data are currently copied
for each matrix-vector product, so this is a correctness-oriented first port.

## 1. Clone the branch

For a fresh checkout:

```bash
cd "$HOME"

git clone \
  --branch kokkos-linear-solver \
  --recurse-submodules \
  https://github.com/MadanBK1/SU2.git \
  SU2_8_5_Kokkos

cd "$HOME/SU2_8_5_Kokkos"
git submodule update --init --recursive
chmod u+x meson_scripts/kokkos_compiler_dispatcher
```

For an existing checkout:

```bash
cd "$HOME/SU2_8_5_Kokkos"
git switch kokkos-linear-solver
git pull --ff-only origin kokkos-linear-solver
git submodule update --init --recursive
chmod u+x meson_scripts/kokkos_compiler_dispatcher
```

## 2. Load the GH200 environment

Run these commands on a GH200 compute node:

```bash
module purge
module load PrgEnv-gnu/8.6.0
module load gcc-native/14
module load cudatoolkit/25.5_12.9

# Optional: activate the Python environment containing Meson.
source "$HOME/venv/bin/activate"

export NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5
export CUDA_HOME="${NVHPC_ROOT}/cuda/12.9"
export NVHPC_CUDA_HOME="$CUDA_HOME"
export NVCOMPILER_CUDA_HOME="$CUDA_HOME"
export NVCOMPILER_COMM_LIBS_HOME="${NVHPC_ROOT}/comm_libs/12.9"

export PATH="${NVHPC_ROOT}/compilers/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export GNU_GCC=/opt/cray/pe/gcc-native/14/bin/gcc
export GNU_GXX=/opt/cray/pe/gcc-native/14/bin/g++

export SU2_KOKKOS_HOST_COMPILER="${NVHPC_ROOT}/compilers/bin/nvc++"
export SU2_KOKKOS_GCC_TOOLCHAIN="$GNU_GCC"
export SU2_KOKKOS_CUDA_ARCH=sm_90
export NVCC_WRAPPER_DEFAULT_COMPILER="$GNU_GXX"

export CC="${NVHPC_ROOT}/compilers/bin/nvc"
export CXX="$PWD/meson_scripts/kokkos_compiler_dispatcher"

unset CXXFLAGS
unset CPPFLAGS
```

Check the selected tools:

```bash
"$CC" --version
"$CXX" --version
"$GNU_GXX" --version
nvcc --version
cmake --version
python3 --version
```

The dispatcher uses `nvc++` for ordinary SU2 sources and Kokkos's bundled
`nvcc_wrapper` for CUDA-dependent translation units. GCC 14 supplies the
C++20 standard-library headers used by the CUDA compilation.

## 3. Configure

The legacy SU2 CUDA backend and the Kokkos backend are mutually exclusive.
Therefore, use `-Denable-cuda=false` when enabling Kokkos CUDA.

```bash
cd "$HOME/SU2_8_5_Kokkos"

export SU2_PREFIX="$HOME/SU2_RUN_8_5_KOKKOS"

rm -rf build-kokkos-cuda

python3 meson.py setup build-kokkos-cuda \
  -Denable-kokkos=true \
  -Dkokkos-backend=cuda \
  -Dkokkos-arch=HOPPER90 \
  -Denable-cuda=false \
  -Dwith-mpi=enabled \
  -Dwith-omp=false \
  -Denable-autodiff=false \
  -Denable-directdiff=false \
  -Denable-mlpcpp=true \
  -Dcpp_args="--gcc-toolchain=${GNU_GCC}" \
  --prefix="$SU2_PREFIX"
```

The configuration summary should report:

```text
Kokkos:         true
Kokkos Backend: cuda
Kokkos Arch:    HOPPER90
AD (reverse):   false
AD (forward):   false
```

## 4. Build and install

```bash
./ninja -C build-kokkos-cuda \
  -j "${SLURM_CPUS_PER_TASK:-8}" \
  2>&1 | tee build-kokkos-cuda.log

build_status=${PIPESTATUS[0]}
echo "build status: $build_status"

if [ "$build_status" -eq 0 ]; then
  ./ninja -C build-kokkos-cuda install
fi
```

The installation places the SU2 executables in `$SU2_PREFIX/bin` and the
static Kokkos libraries in `$SU2_PREFIX/lib64`.

## 5. Runtime environment

```bash
export SU2_HOME="$HOME/SU2_8_5_Kokkos"
export SU2_RUN="$HOME/SU2_RUN_8_5_KOKKOS/bin"
export PATH="$SU2_RUN:$PATH"
export PYTHONPATH="$SU2_RUN:${PYTHONPATH:-}"
```

This validated build links against the HPC-X OpenMPI packaged with NVIDIA
HPC SDK 25.5. Use the matching runtime and correct its relocatable OpenMPI
prefix:

```bash
export HPCX_OMPI="${NVHPC_ROOT}/comm_libs/12.9/hpcx/hpcx-2.22.1/ompi"
export OPAL_PREFIX="$HPCX_OMPI"
export PATH="$HPCX_OMPI/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$HPCX_OMPI/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

unset MPICH_GPU_SUPPORT_ENABLED
```

Verify the installation and MPI linkage:

```bash
which SU2_CFD
which mpirun
mpirun --version
ldd "$SU2_RUN/SU2_CFD" | grep -E 'libmpi|libopen'
```

For this exact HPC-X build, launch with `mpirun --bind-to none`. Direct
`srun` uses the cluster's Cray MPI environment and is not compatible with
this HPC-X-linked executable.

## 6. Enable Kokkos in an SU2 case

Add this option to the SU2 configuration file:

```text
ENABLE_KOKKOS= YES
```

Example linear-solver section:

```text
LINEAR_SOLVER= FGMRES
ENABLE_KOKKOS= YES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 10
```

Run one MPI rank on one GH200 GPU:

```bash
"$HPCX_OMPI/bin/mpirun" \
  --bind-to none \
  -np 1 \
  "$SU2_RUN/SU2_CFD" case.cfg \
  2>&1 | tee su2-kokkos.log
```

Monitor GPU activity from another shell:

```bash
watch -n 0.5 nvidia-smi
```

For a host-path comparison using the same executable, change only:

```text
ENABLE_KOKKOS= NO
```

Use separate run directories so the two cases do not overwrite each other's
history, restart, and visualization files.

## 7. Validated flat-plate result

The `Laminar_Flat_Plate` tutorial was run with one MPI rank. Both variants
converged successfully at iteration 1599.

| Metric | Host path | Kokkos/CUDA path |
|---|---:|---:|
| Final `rms[Rho]` | -12.04692789 | -12.04693809 |
| Iteration | 1599 | 1599 |
| Wall time | 20.68 s | 19.62 s |
| Maximum memory | 110592 KiB | 110592 KiB |

For this small 4,225-point mesh, the measured wall-time ratio was approximately
1.054, or 5.1% lower elapsed time with Kokkos. This is a correctness result,
not a representative GPU-performance benchmark: initialization, transfers,
kernel-launch latency, MPI, and file I/O are significant at this mesh size.

The CPU and Kokkos convergence histories agree to normal floating-point
roundoff, and both restart and ParaView output completed successfully.

## 8. Troubleshooting notes

- `cannot open source file "concepts"`: ensure GCC 14 is loaded and preserve
  the `--gcc-toolchain` configuration argument.
- `CMake was unable to find ... Ninja`: use this branch's current Meson files;
  they pass the bundled `./ninja` path into the Kokkos CMake subproject.
- `hwloc_set_cpubind returned "Error"`: add `--bind-to none` to the HPC-X
  `mpirun` command.
- An OpenMPI help-file path referring to `comm_libs/12.8`: set `OPAL_PREFIX`
  to the HPC-X OpenMPI path shown above.
- Do not set `MPICH_GPU_SUPPORT_ENABLED` for this HPC-X OpenMPI build; that
  variable applies to Cray MPICH.
- Warnings about unsigned comparisons or Eigen relaxed `constexpr` behavior
  are nonfatal for the validated configuration.
- A larger mesh and repeated runs are required before drawing performance or
  scaling conclusions.

