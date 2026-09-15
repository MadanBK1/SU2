# Kokkos Linear Solver Port: NVIDIA V100 Validation Summary

## Scope

This document summarizes the Kokkos linear-algebra port validated on NVIDIA Tesla V100 GPUs for SU2 8.5 development work in this branch.

The objective of this work is to keep the expensive Krylov linear-algebra operations resident on the accelerator, minimize host/device synchronization, support multi-GPU MPI execution, and preserve the numerical behavior of the existing SU2 linear solvers.

Validated branch:

- `kokkos-v100-team64-validated`
- current V100 tuning commit: `c2d8abdcbdc844e6ae782cb53e882a1da4dd58cb`

## Ported Kokkos functionality

### Device-resident Krylov solvers

The following Krylov solver paths have Kokkos device-resident implementations:

- Conjugate Gradient (CG)
- BiCGSTAB
- FGMRES

For the device-resident solver paths, the large CFD vectors remain on the GPU through the Krylov iterations. Operations moved to Kokkos include:

- vector copy and fill
- scaling
- AXPY
- AXPBY
- linear combinations used by the Krylov iterations
- local dot products
- norms
- matrix-vector products
- Krylov basis-vector updates
- residual updates

FGMRES retains only the small Hessenberg matrix, Givens rotations, and reduced triangular solve on the CPU. These operations are only O(m^2) in the Krylov subspace dimension and do not justify GPU transfer or kernel overhead.

### Kokkos sparse matrix-vector product

The sparse matrix-vector product is executed on the GPU using device-resident matrix and vector data.

The matrix is transferred/synchronized to the accelerator and reused by the iterative solver rather than copied for every Krylov operation.

### Device ILU preconditioner application

The existing SU2 ILU factorization semantics are preserved while the repeated ILU application has been moved to the GPU.

The Kokkos implementation includes:

- persistent device ILU factor storage
- device row, column, and diagonal metadata
- level-scheduling metadata on the device
- persistent-team forward triangular sweep
- persistent-team backward triangular sweep
- synchronization only where required by the dependency graph or MPI communication

For the V100 validation case, the level-scheduled ILU graph contained relatively narrow levels. The persistent-team implementation substantially reduced launch overhead compared with launching an independent kernel for every dependency level.

### V100 ILU team-size tuning

The persistent ILU forward/backward kernels were tested with different Kokkos team sizes:

| Team size | BiCGSTAB elapsed time |
|---:|---:|
| `Kokkos::AUTO` | 81.07 s |
| 32 | 81.27 s |
| **64** | **80.35 s** |
| 128 | 81.00 s |

All four configurations converged in exactly 2887 outer iterations and produced the same reported final state.

The V100-validated implementation therefore uses a team size of 64 for the persistent ILU kernels.

This value is a V100 performance tuning result and should not be assumed to be optimal for NVIDIA Hopper, AMD, or Intel GPUs without separate benchmarking.

### Multi-GPU halo exchange

Two MPI halo-exchange modes are implemented for the Kokkos matrix-vector path.

#### Compact host-staged MPI

The validated V100 production path stages only the compact halo buffers through host memory rather than transferring the entire vector between the host and device.

The sequence is:

1. gather send entries on the GPU;
2. copy only the compact send buffer to the host;
3. perform MPI communication using host buffers;
4. copy only the received halo buffer to the GPU;
5. scatter the received values into the device vector.

This removed the previous full-vector D2H/H2D synchronization around host MPI communication.

#### GPU-aware MPI

A direct device-buffer MPI path is also implemented. It can exchange device halo buffers directly when the MPI stack supports CUDA-aware communication.

For the V100/OpenMPI environment used in this validation, the direct GPU-aware path was not the fastest configuration. The compact host-staged path is therefore the validated performance path for these results.

## Numerical validation

The validation workload was the SU2 laminar flat-plate case executed with two NVIDIA Tesla V100 GPUs.

### BiCGSTAB + ILU

Best validated configuration:

- Kokkos device-resident BiCGSTAB
- device sparse matrix-vector product
- persistent level-scheduled Kokkos ILU
- compact halo-only host staging
- persistent ILU team size = 64
- 2 MPI ranks / 2 V100 GPUs

Result:

- solver status: success
- outer iteration count: 2887
- elapsed time: **80.35 s**
- final reported state:

```text
-12.09157327
 -9.652765975
 -9.544814386
 -6.615349841
```

The result remained numerically unchanged during the tested ILU team-size variations.

### FGMRES + ILU

Best validated configuration:

- Kokkos device-resident FGMRES
- device sparse matrix-vector product
- persistent level-scheduled Kokkos ILU
- compact halo-only host staging
- batched MPI reduction for FGMRES projection coefficients
- persistent ILU team size = 64
- 2 MPI ranks / 2 V100 GPUs

Result:

- solver status: success
- outer iteration count: 6337
- elapsed time: **139.38 s**
- final reported state:

```text
-12.00228384
-10.18044348
-10.13424318
 -7.169942676
```

The validated FGMRES history file is byte-for-byte identical to the previous validated device-FGMRES trajectory.

SHA-256:

```text
420d5c0141abf6caaff42a6ac0fcb441c78e22b1c980973b0795f78f9909f507
```

### Compact halo-staging effect

Before compact halo-only staging, the persistent-ILU FGMRES run required approximately 144.09 s.

After replacing full-vector host staging with compact halo-buffer staging:

- FGMRES: 144.09 s -> 140.22 s
- numerical history remained byte-for-byte identical

After the additional V100 ILU team-size tuning:

- FGMRES: 140.22 s -> **139.38 s**

For BiCGSTAB, compact halo staging plus the V100 ILU tuning reduced the validated run to **80.35 s**.

## Optimization experiments not retained

### Fused local FGMRES MultiDot reduction

An experimental FGMRES implementation fused multiple local projection dot products into a single Kokkos reduction while preserving one MPI reduction for the coefficient vector.

The experiment remained numerically correct and converged in the same 6337 iterations, but performance regressed:

- validated scalar-local-reduction FGMRES: 139.38 s
- fused local MultiDot experiment: 151.78 s

This is approximately an 8.9% slowdown, so the fused implementation was rejected and the validated scalar local reductions were restored.

## CPU comparison

The present V100 implementation is numerically validated, but this particular flat-plate case is not yet a GPU speedup case.

Representative previously measured host baselines were:

| Solver configuration | Elapsed time |
|---|---:|
| CPU FGMRES + host ILU | 54.51 s |
| V100 FGMRES + device ILU, current validated path | 139.38 s |
| CPU/host BiCGSTAB reference | 46.82 s |
| V100 BiCGSTAB + device ILU, current validated path | 80.35 s |

The flat-plate validation matrix has relatively narrow ILU dependency levels, which limits available GPU parallelism in the triangular solve. These timings should therefore be treated as validation and optimization data, not as a general statement about GPU performance for larger SU2 systems.

Larger systems with more cells and/or larger equation blocks are expected to provide substantially more work per GPU kernel and higher arithmetic intensity. Such cases, including NEMO thermochemical-nonequilibrium systems, require separate performance validation.

## Current practical coverage

The current Kokkos implementation covers the main linear-algebra path required by many conventional implicit SU2 cases using Krylov solvers:

| Component | Kokkos status |
|---|---|
| Sparse matrix-vector product | Ported and validated |
| Device vector operations | Ported and validated |
| CG | Ported |
| BiCGSTAB | Ported and validated on V100 |
| FGMRES | Ported and validated on V100 |
| ILU application | Ported and validated on V100 |
| Jacobi | Device implementation available; not the production preconditioner for this validation case |
| Compact host-staged MPI halo exchange | Ported and validated |
| Direct GPU-aware MPI halo exchange | Implemented; not the fastest validated V100 path |
| Restarted FGMRES | Uses the FGMRES implementation internally; separate end-to-end regression validation is still required |
| FGCRODR | Not yet fully ported to the Kokkos device-resident path |
| LU-SGS | Not yet ported to the device-resident Kokkos path |
| LINELET | Not yet ported to the device-resident Kokkos path |
| PaStiX variants | External solver path; not replaced by this Kokkos port |

## Relevance to NEMO / hypersonic cases

The Kokkos changes are implemented in SU2's shared linear-algebra infrastructure rather than in a laminar-flow-specific solver. Therefore NEMO configurations that use the same generic matrix, vector, Krylov, and ILU/Jacobi infrastructure can use this Kokkos linear-solver path.

However, a full NEMO hypersonic thermochemical-nonequilibrium validation has not yet been completed. A representative NEMO case should be used to confirm:

- entry into the Kokkos device-resident linear-solver path;
- correctness relative to the standard SU2 result;
- multi-GPU behavior;
- performance as the number of species/equations and mesh size increase.

## Validation status

The strongest current V100 checkpoint is therefore:

```text
Kokkos device-resident Krylov solvers
+ device SpMV
+ persistent level-scheduled device ILU
+ compact halo-only host MPI staging
+ V100 ILU team size 64
```

with validated two-V100 results of:

```text
BiCGSTAB + ILU : 80.35 s, 2887 iterations
FGMRES  + ILU : 139.38 s, 6337 iterations
```

The FGMRES numerical history checksum is:

```text
420d5c0141abf6caaff42a6ac0fcb441c78e22b1c980973b0795f78f9909f507
```

These results establish a reproducible V100 validation point for continued optimization and extension to additional SU2 solver/preconditioner variants and larger CFD/NEMO workloads.
