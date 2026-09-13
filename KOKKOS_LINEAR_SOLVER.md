# Kokkos linear algebra backend

This initial backend replaces SU2's CUDA-specific block-CSR matrix-vector
product with a portable Kokkos kernel. SU2's Krylov algorithms,
preconditioners, and MPI halo exchange remain unchanged and host-resident.

## Current scope

- Passive/primal SU2 builds only.
- Block-CSR matrix-vector products used by the iterative linear solvers.
- Kokkos Serial, CUDA, or SYCL selected when Kokkos itself is built.
- Runtime selection with `ENABLE_KOKKOS= YES` in the SU2 configuration.
- The existing CUDA backend and the Kokkos backend are mutually exclusive.

The matrix and vectors are copied for each matrix-vector product. This is a
correctness milestone, not yet the final high-performance design. The next
milestone is to keep Krylov vectors and matrix values device-resident and port
the vector reductions and Jacobi preconditioner.

## Configure SU2

Install Kokkos first, then make its CMake package visible:

```bash
export CMAKE_PREFIX_PATH=/path/to/kokkos-install:${CMAKE_PREFIX_PATH:-}
export CXX=/path/to/compiler-for-the-selected-kokkos-backend

python3 meson.py setup build-kokkos \
  -Denable-kokkos=true \
  -Denable-autodiff=false \
  -Denable-directdiff=false
./ninja -C build-kokkos
```

Use the same compiler for Kokkos and SU2. For CUDA this is normally Kokkos'
`nvcc_wrapper`; for Intel PVC/SYCL use a supported `icpx -fsycl` toolchain.

## Run

Add the following to the case configuration:

```text
ENABLE_KOKKOS= YES
```

If this option is enabled in a case but SU2 was built without Kokkos, SU2 exits
with an explicit error instead of silently using the host implementation.

## Validation performed

The full SU2 executable was built with Kokkos 4.7.2 Serial and exercised on the
implicit Euler NACA0012 QuickStart case using FGMRES and ILU(0). A five-iteration
host/Kokkos comparison produced identical convergence-history files. Across
the surface solution variables, the maximum absolute difference was
`5.20e-08` and the maximum relative difference was `8.22e-11`.

CUDA, SYCL, MPI, and performance validation still need to be run on the target
systems.
