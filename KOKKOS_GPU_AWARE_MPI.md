# Kokkos GPU-aware MPI for SU2 linear algebra

This branch contains an opt-in device-buffer MPI path for the Kokkos linear
solver. It is backend-neutral: Kokkos allocates, packs, and unpacks the halo
buffers, while the configured MPI implementation communicates the resulting
device pointers.

The same source supports:

- NVIDIA GPUs through the Kokkos CUDA backend and a CUDA-aware MPI library.
- Intel GPUs through the Kokkos SYCL backend and an MPI library with
  Level-Zero/SYCL USM-buffer support.

## Runtime options

```text
ENABLE_KOKKOS= YES
KOKKOS_GPU_AWARE_MPI= YES
```

`KOKKOS_GPU_AWARE_MPI` defaults to `NO`. Keep it disabled when the MPI library
cannot accept device pointers. The original host-staged halo exchange remains
the fallback path.

## Communication sequence

For every Kokkos block-CSR matrix-vector product, SU2 performs the following
operations:

1. Compute owned rows into the device-resident output vector.
2. Post nonblocking receives into persistent Kokkos device buffers.
3. Pack owned boundary entries into a persistent device send buffer.
4. Fence the Kokkos execution space before passing device pointers to MPI.
5. Issue `MPI_Isend` calls and complete all receives.
6. Unpack received values into device-resident ghost entries.
7. Complete all sends and copy the finished vector to the host.

Step 7 is still required because FGMRES, ILU, and most vector operations remain
host-resident. This change removes host staging from the halo messages, but it
does not yet make the entire linear solver device-resident.

## NVIDIA checks

### OpenMPI

Before enabling the option, inspect the MPI build:

```bash
which mpirun mpicxx ompi_info
mpirun --version
ompi_info | grep -i "MPI extensions"
ompi_info --parsable --all | grep -i 'mpi_built_with_cuda_support:value'
ldd "$SU2_RUN/SU2_CFD" | grep -E 'libmpi|libopen'
```

The OpenMPI installation must report CUDA support, and the runtime libraries
must match the MPI used to build SU2. OpenMPI provides the
`MPIX_Query_cuda_support()` extension in builds that include its CUDA
extension.

### HPE Cray MPICH

For a Cray MPICH CUDA build, the executable must be linked with the platform's
CUDA GPU transport layer. A typical runtime setup is:

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
```

Use the site-specific compiler wrapper and GPU transport library documented by
the target system. Do not combine a Cray MPICH launcher with an executable
linked against OpenMPI.

## Intel GPU checks

Build SU2 with the Kokkos SYCL backend and Intel PVC target:

```bash
python3 meson.py setup build-kokkos-pvc \
  -Denable-kokkos=true \
  -Dkokkos-backend=sycl \
  -Dkokkos-arch=INTEL_PVC \
  -Denable-cuda=false \
  -Dwith-mpi=enabled \
  -Dwith-omp=false \
  -Denable-autodiff=false \
  -Denable-directdiff=false \
  --prefix="$SU2_PREFIX"
```

Use an Intel MPI installation that supports GPU buffers through Level Zero.
The common Intel MPI runtime switch is:

```bash
export I_MPI_OFFLOAD=1
```

GPU/tile visibility and rank mapping are site-specific. On managed systems
such as Aurora, preserve the scheduler-provided affinity variables instead of
overriding them globally.

## Correctness test

Use identical MPI rank counts and separate run directories.

Host-staged Kokkos halo exchange:

```text
ENABLE_KOKKOS= YES
KOKKOS_GPU_AWARE_MPI= NO
```

Direct device-buffer halo exchange:

```text
ENABLE_KOKKOS= YES
KOKKOS_GPU_AWARE_MPI= YES
```

Example NVIDIA launch with one rank per visible GPU:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export KOKKOS_MAP_DEVICE_ID_BY=mpi_rank

mpirun --map-by slot --bind-to none -np 4 \
  -x CUDA_VISIBLE_DEVICES \
  -x KOKKOS_MAP_DEVICE_ID_BY \
  "$SU2_RUN/SU2_CFD" case.cfg \
  2>&1 | tee gpu-aware-mpi.log
```

Validate all of the following:

- The process exits successfully on every rank.
- Each local MPI rank selects a distinct GPU.
- The host-staged and device-buffer cases converge at the same iteration or
  exhibit only expected floating-point differences.
- Final residuals and physical output fields agree within the chosen
  tolerance.
- A multi-node run completes with the same result as a single-node run at the
  same decomposition.

## Current limitations

- The feature currently applies only to the halo exchange performed by the
  Kokkos block-CSR matrix-vector product.
- MPI collectives in Krylov dot products remain host-resident.
- ILU and FGMRES vector operations remain host-resident.
- MPI capability detection is not portable across vendors, so device-buffer
  communication requires explicit user opt-in.
- NVIDIA CUDA and Intel SYCL builds must be validated independently because
  MPI GPU support is a property of the installed MPI library, not Kokkos.
