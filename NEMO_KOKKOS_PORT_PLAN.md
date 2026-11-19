# SU2 NEMO → Kokkos: Stage A inventory and port plan

Base: `MadanBK1/SU2` master `da1deb9` (= `kokkos-v100-team64-validated`; contains device CG/BCGSTAB/FGMRES, device Jacobi, level-scheduled ILU apply, halo staging). Local work branch `kokkos-nemo-port` (not pushed).
Workstation WIP (`work/kokkos-nemo-production`, `acb6753a5c`, NEMO chemistry tag `validated-nemo-chemistry-powers-20261006`) is **not on GitHub** and is not included.

## Key findings

### Blockers (ranked)
1. **Gas model is stateful.** `CNEMOGas`/`CSU2TCLib` keep thermodynamic state in members between calls (`rhos`, `T`, `Tve`, `Cvtrs`, `eves`, `ws`, `taus`, even loop counters `iSpecies/jSpecies`). There are hidden ordering dependencies:
   - `ComputeTemperatures` reads `Cvtrs` from a previous `GetSpeciesCvTraRot`.
   - `ComputeEveSourceTerm` reads `ws` from the previous `ComputeNetProductionRates`.
   - `ThermalConductivitiesWBE` reads `mus/phis` from `ViscosityWBE`.
   - `GetChemistryEquilConstants` does `std::string` compares per reaction per point.
   **Fix:** rewrite as stateless `KOKKOS_INLINE_FUNCTION`s over read-only tables (`NEMOGasTables`). Mutation++ stays host-only.
2. **Numerics are virtual objects with per-object scratch.** Each `CNEMONumerics` owns its own `new CSU2TCLib`, uses `new[]` buffers, and returns `su2double**` Jacobians. **Fix:** POD flux functors using stack arrays sized `MAXNVAR`.
3. **Edge loops are serial.** No coloring and no reducer: NEMO doesn't override `GetHasHybridParallel`, and `GetEdgeColoring()` degenerates to one colour because Kokkos builds forbid OpenMP. **Fix:** atomics, or `colorSparsePattern(GetSparsePattern(FV,0))` called directly.
4. **Jacobian is re-uploaded on every solve.** The first `DeviceMatVec` uses `matrix_on_device=false`, so `HtDTransfer` copies the full nnz·nVar². ILU and Jacobi are factored on the host, then mirrored. `dia_ptr` and `edge_ptr` aren't mirrored, and `d_row_ptr` covers only `nPointDomain+1` rows.
5. **`CVertex` is pointer-AoS**, so it needs per-marker flattening. `CEdge::Nodes` has no whole-array accessor. Everything else (`Coord`, `Volume`, edge `Normal`, CSR, `Solution`/`Primitive`, gradients) is contiguous `C2DContainer` and can be copied once.
6. **Kokkos init is implicit** (a static in `allocation_toolbox.hpp`). Finalize runs after `MPI_Finalize`. Device-resident work needs explicit init and finalize.

### Pre-existing issues in the CPU reference (verify before using as ground truth)
- `CSU2TCLib.cpp:1576-1588` `ComputeKeqConstants`: `unsigned short iIndex = int(pwr)-14` wraps when N < 1e14 cm⁻³, which selects the highest-density row. Affects low-density/high-altitude cases.
- `CNEMOEulerSolver::ComputeConsistentExtrapolation` doesn't recompute dPdU/dTdU/dTvedU/Cvve at reconstructed states (TODO in code). The implicit Jacobian uses stale or zero arrays.
- NS + Green-Gauss: the gradient is of Y_s, but MUSCL adds it to ρ_s.
- `CNEMOGas::ComputedPdU` with ionization: dPdU for the last ion (NO+) is only decremented (`-=`), never assigned, so it accumulates across calls. `ComputedTdU` never writes the electron entry. `GetSpeciesCvTraRot` fills only species nEl..nHeavy-1 (NO+ Cv_tr = 0 in AIR-7).
- `msw.cpp:64-65` allocates `Flux` twice (leak). AUSMPWplus is declared but not implemented.

### Data layout
- nVar = Ns+nDim+2 (U = ρ_s, ρu_i, ρE, ρE_ve). nPrimVar = Ns+nDim+8 (Euler) / +10 (NS).
- Primitive V layout: ρ_s | T | Tve | u_i | P | ρ | h | a | ρCv_tr | ρCv_ve | [μ, μ_t].
- Secondary per point: dPdU, dTdU, dTvedU (nVar), eves, Cvves (Ns), Gamma.
- Passive Kokkos build means su2double == double.

## Call chain (steady implicit, MGLEVEL=0)
1. `CFluidIteration::Iterate` → `CMultiGridIntegration::PreSmoothing`.
2. `Preprocessing`:
   - `SetPrimitive_Variables` (Cons2Prim per point);
   - non-physical Allreduce;
   - zero `LinSysRes`/Jacobian;
   - GG/LS gradients + comms;
   - limiter + comms.
3. `SetTime_Step` (edge + vertex + point).
4. `Space_Integration`: Upwind/Centered → Viscous → Source → BCs.
5. `ImplicitEuler_Iteration`:
   - Prepare: diag += Vol/dt, RHS, `SetResidual_RMS` Allreduce;
   - `System.Solve`;
   - Complete: under-relaxation, `AddSolution`, SOLUTION comms.
6. `AdaptCFLNumber`.

## Validation cases
Meshes come from `su2code/TestCases`, not this repo.

| Case | What it covers | Reference |
|---|---|---|
| `invwedge_ausm.cfg` | AIR-5, implicit AUSM + MUSCL, BCGSTAB+ILU | serial: -1.073693, -1.598456, … |
| `visc_wedge/super_cat.cfg` | implicit NS, 1st order, catalytic wall, `DeleteValsRowi` | — |
| `thermalbath.cfg` | 5×5 mesh, N2 finite-rate chemistry, explicit | serial -11.897761 … |

## Port order and gates
Each step runs behind `ENABLE_KOKKOS`, keeps the CPU path, and is validated against CPU on the Serial backend before CUDA.

| # | Step | Gate |
|---|---|---|
| 0 | `NEMOGasTables` + stateless thermo (Eve, Cvve, Cvtr, energies, T/Tve Newton+bisection, P, a, γ, dPdU, dTdU, dTvedU) | Pointwise vs CSU2TCLib ≤1e-12 rel, AIR-5/N2/AIR-7, random states |
| 1 | Device geometry/state mirror (edges, normals, volumes, flattened vertices, CSR + dia_ptr + edge_ptr) | Round-trip equality |
| 2 | Cons2Prim kernel + non-physical reduce | Frozen-solution field match |
| 3 | GG/LS gradients + Venkatakrishnan/BJ limiter | Match CPU |
| 4 | Time step / max eigenvalue | Match CPU |
| 5 | AUSM residual + approx Jacobian (atomics first, then coloring), MUSCL + non-phys check | invwedge_ausm residual history |
| 6 | Device-assembled Jacobian handed to the device solver (no HtoD); device Jacobi build, then device ILU(0) | Same history; transfer bytes ~0 |
| 7 | Production rates + VT relaxation as a fused point kernel + Jacobian | thermalbath 0-D, then invwedge |
| 8 | Transport (Sutherland/Wilke/Gupta-Yos), viscous AvgGrad(+Corrected) + Jacobian, NS walls | super_cat, axi_visccone |
| 9 | Roe/MSW/LAX, remaining BCs, axisymmetric source | Per-case regression |
| 10 | Device halo exchange for Primitive/gradients/Solution; explicit Kokkos init/finalize | Multi-rank match |
| 11 | CUDA V100 → GH200 → SYCL (PVC/Aurora), HIP later | Per-backend validation + profiling (nsys) |

## Status (2026-11-19)
- **Kokkos Serial build** is in place locally (meson, no MPI, passive).
- **invwedge_ausm**, `ENABLE_KOKKOS=NO` and `YES`: the iteration-10 residuals equal the serial reference exactly. Fields agree to ≤1.2e-10 relative to field magnitude (linear-solver summation order).
- **Fix: ILU apply team size.** It was hard-coded to 64, which aborted on Serial (max team size 1). It is now `min(64, exec.concurrency())`, which leaves V100 unchanged. File: `CSysMatrixKokkos.cpp`.
- **Step 0 done (uncommitted):**
  - `SU2_CFD/include/fluid/kokkos/NEMOGasKokkos.hpp`: stateless tables and functions, plus Cons2Prim.
  - `SU2_CFD/src/fluid/kokkos/NEMOGasKokkos.cpp`: builder from CSU2TCLib, through a `friend` added in `CSU2TCLib.hpp`.
  - `UnitTests/SU2_CFD/fluid/NEMOGasKokkos_validation.cpp`.
- **Step 0 results** (N2, AIR-5, AIR-7, ARGON; random and edge-case states):
  - Strict IEEE flags (`-fno-unsafe-math-optimizations -ffp-contract=off`): bitwise identical, Newton and bisection paths included.
  - Production `-ffast-math`: ≤8.2e-12 relative (dTdU, AIR-5). Most quantities ≤1e-13.
  - Non-physical flag: 0 mismatches in 140k calls.
- **Gate for future kernels:** bitwise under strict flags, ≤1e-10 under production flags.
- **Steps 1–2 done:**
  - `CNEMOKokkosData` (`SU2_CFD/{include,src}/solvers/kokkos/`): device mirror of the geometry (edges, normals, coords, volumes, point/edge CSR, per-marker vertices, FV pattern with all nPoint rows + dia_ptr + edge→nnz map) and of the state. Owned by `CNEMOEulerSolver`, created only with `ENABLE_KOKKOS= YES` and SU2_NONEQ. `SU2_KOKKOS_SELFCHECK=1` runs a bitwise round-trip check.
  - Device `SetPrimVar` (Cons2Prim + rollback to Solution_Old + Gamma + parallel_reduce count) in `CNEMOEulerSolver`/`CNEMONSSolver::SetPrimitive_Variables`. The host stays authoritative: every call uploads U, U_old, V and secondaries and downloads the results. Velocity2, NS transport and a replay of the host gas-model state stay on the host.
  - `SU2_KOKKOS_C2P_VERIFY=1` runs an A/B check against the CPU loop on every call: ≤5.8e-16 relative under -ffast-math (ULP level), 0 non-physical count mismatches.
  - check.sh: residual rows are identical. Fields: invwedge 1.7e-10 (linear solver), super_cat 1.7e-15, thermalbath 0. The CPU path is bit-identical to before.
  - check.sh fix: super_cat.cfg has no trailing newline, so `ENABLE_KOKKOS` had been appended to SCREEN_OUTPUT. super_cat had never run the YES path.
