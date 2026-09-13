# Kokkos linear algebra backend

This initial backend replaces SU2's CUDA-specific block-CSR matrix-vector
product with a portable, hierarchical-parallel Kokkos kernel. SU2's Krylov algorithms,
preconditioners, and MPI halo exchange remain unchanged and host-resident.

## Current scope

- Passive/primal SU2 builds only.
- Block-CSR matrix-vector products used by the iterative linear solvers.
- Bundled and pinned Kokkos 5.2.2 source in `subprojects/Kokkos`.
- Kokkos Serial, CUDA, or SYCL selected from the SU2 Meson command.
- Runtime selection with `ENABLE_KOKKOS= YES` in the SU2 configuration.
- The existing CUDA backend and the Kokkos backend are mutually exclusive.
- SU2's native OpenMP mode cannot yet be combined with this backend; use the
  Kokkos CUDA/SYCL execution space for node parallelism and MPI between ranks.
- Kokkos-enabled SU2 targets use C++20, as required by Kokkos 5.2.

The matrix and vectors are copied for each matrix-vector product. This is a
correctness milestone, not yet the final high-performance design. The next
milestone is to keep Krylov vectors and matrix values device-resident and port
the vector reductions and Jacobi preconditioner.

## Configure SU2

Kokkos is built as part of SU2; no separate Kokkos installation is required:

```bash
export CXX=/path/to/compiler-for-the-selected-kokkos-backend

python3 meson.py setup build-kokkos \
  -Denable-kokkos=true \
  -Dkokkos-backend=serial \
  -Dkokkos-arch=none \
  -Denable-autodiff=false \
  -Denable-directdiff=false
./ninja -C build-kokkos
```

For GH200 use `-Dkokkos-backend=cuda -Dkokkos-arch=HOPPER90` and a CUDA-capable
C++ compiler. For Intel PVC use `-Dkokkos-backend=sycl -Dkokkos-arch=INTEL_PVC`
with a supported `icpx -fsycl` toolchain. The same compiler builds both Kokkos
and SU2 because Kokkos is a bundled CMake subproject.

## Run

Add the following to the case configuration:

```text
ENABLE_KOKKOS= YES
```

If this option is enabled in a case but SU2 was built without Kokkos, SU2 exits
with an explicit error instead of silently using the host implementation.

## Validation performed

The full SU2 executable was built with Kokkos 5.2.2 Serial and exercised on the
implicit Euler NACA0012 QuickStart case using FGMRES and ILU(0). A five-iteration
host/Kokkos comparison produced identical convergence-history files. Across
the compared surface solution variables, both the maximum absolute difference
and maximum relative difference were zero for the Serial backend.

CUDA, SYCL, MPI, and performance validation still need to be run on the target
systems.
