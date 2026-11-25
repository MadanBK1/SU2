# Review notes for the local agent (written by the Claude reviewer; agent: read, never edit)

Newest section first. Do the **Agent: do next** items before continuing LOCAL_AGENT_TASK.md section 4.

## 2026-11-25 — review 2
Reviewed: 9008747f5db620204c0f309966a36918e6cc000e (REVIEW_REQUEST 20261124T213453-9008747f5d, periodic)

**State:**
- No code since review 1 (e45a5d3 .. 9008747): only autosave/progress-log commits and loop-script changes. Rounds 1–8 (19:41–21:34) all ended rc=1 with NO progress; `CNEMOKokkosLAX.hpp` was announced in round 78 but never written.
- rc=1 on every round most likely means the session hits `--max-turns` while reading files. Keep reads small and write code early.
- Nothing merged into su2kokkos (nothing new to gate).

**Reviewer fix (unblocks 4.4-LAX):** added `SU2_CFD/src/solvers/kokkos/CNEMOKokkosLAX.hpp`.
- It ports CCentLax_NEMO::ComputeResidual, incl. the implicit Jacobian, plus GetInviscidProjFlux / GetInviscidProjJac (row-major), with the CPU quirks kept.
- It is not included or dispatched anywhere yet, so the gates are unaffected.
- Checked here: compiles with g++ -std=c++20 -O2 -ffp-contract=off against Kokkos Serial headers.
- LaxInviscidProjFlux is bitwise identical to a verbatim transcription of the CPU GetInviscidProjFlux (2D and 3D, 20k random states).
- Not yet validated against a full SU2 run.

**Findings:**
- lax.cpp:121 — the CPU Jacobian_j last row, momentum columns, uses `dPdU_j[nSpecies+nDim]` instead of `[nSpecies+iDim]`. This is a CPU quirk; it is reproduced on purpose (see the comment in the header).
- lax.cpp:~80 — `MeandPdU` is a persistent member that is never initialised. With IONIZATION, `ComputedPdU` does `-=` on the previous edge's values, which depends on edge order. The device version starts each edge from 0, so LAX + ionization cannot be bitwise. Keep that combination on the host path: in the dispatch, return false if `config->GetIonization()`.
- The CPU calls `fluidmodel->ComputeSpeciesEve/ComputedPdU` at the mean state of every edge, even when explicit. After the host LAX loop, the host gas object holds the last edge's mean state. If a later host stage reads gas members without recomputing them, replay that state the same way the AUSM path does with ReplayHostGasState. Check with G1 before assuming it is not needed.

**Speedup/accuracy suggestions:**
- The LAX edge kernel is lighter than AUSM: no MUSCL, no reconstruction, no CheckNonPhys pass. Reuse the existing ATOMIC/COLOR accumulation from UpwindEdge, and add no new transfers. Lambda (st.Lambda, from SetMax_Eigenvalue) and Solution are already device-resident; only nNeighbor is new, and it is uploaded once with the geometry.
- `Kokkos::pow(x, 0.3)` can differ by 1 ulp on the GPU. That is expected noise; judge it with the noise floor (≤ 10), not bitwise on CUDA.

**Agent: do next**
1. Add `View1D<unsigned short> nneighbor; /*!< (nPoint) geometry->nodes->GetnNeighbor */` to the geometry struct in `SU2_CFD/include/solvers/kokkos/CNEMOKokkosData.hpp`, next to `coord` at line ~103. Fill it in `SU2_CFD/src/solvers/kokkos/CNEMOKokkosData.cpp`, in the "Points." block near line 385, the same way as `geo.volume`. Build serial, commit.
2. Create `SU2_CFD/src/solvers/kokkos/CNEMOKokkosCentered.cpp`, a copy of the UpwindEdge pattern in CNEMOKokkosUpwind.cpp:
   - The edge kernel calls `LAXResidual<ND>(a.gas, ns, nVar, nPrimVar, implicit, config->GetKappa_1st_Flow(), Normal, &U(i,0), &U(j,0), &V(i,0), &V(j,0), dPdU_i, dPdU_j, Lambda(i), Lambda(j), nneighbor(i), nneighbor(j), Flux, Ji, Jj)`, then NaN check, then res/jac accumulation as in UpwindEdge.
   - Add `bool Device_CenteredResidual(...)` to CNEMOEulerSolver.hpp.
   - Call it at the top of `CNEMOEulerSolver::Centered_Residual` (CNEMOEulerSolver.cpp:735) as `if (Device_CenteredResidual(...)) return;`.
   - It returns false unless the scheme is LAX and there is no ionization.
   - Add the file to SU2_CFD/src/meson.build. Build serial, commit.
3. Make a LAX copy of the invwedge config (`CONV_NUM_METHOD_FLOW= LAX-FRIEDRICH`) under TestCases/kokkos_nemo. Run it with `REF_MODE=HOST RES_TOL=0 FIELD_TOL=0`, then write `READY FOR REVIEW: 4.4-LAX`.
   - If item 1 or 2 does not compile after 3 sessions, mark `BLOCKED 4.4-LAX: <error>` and go to 4.5.
   - Read at most 80 lines of a file at a time. Write code within your first ~10 tool calls.


## 2026-11-24 19:40 — review 1
**State:**
- 4.1, 4.2 (Batch A, all gates incl. the 2-rank fix 982c4ea) and 4.3 G5 are done.
- The user ran G7 on the V100: PASS (super_cat bitwise residuals; invwedge noise floor 1.00/1.05). Device-resident is ON.
- Kernel times per call: SetPrimVar 0.6 ms, Upwind 3.6 ms, Source 3.7 ms, Viscous 5.0 ms.
- Merged into su2kokkos: the script env-var paths (791070c).

**Finding:** rounds ~23–77 produced no code for 4.4 (LAX), only progress-log commits. The loop now treats log-only rounds as stuck.

**Agent: do next**
1. LAX (4.4): write code this session.
   - Smallest compilable step first: a `KOKKOS_INLINE_FUNCTION` port of the LAX flux (numerics/NEMO/convection/lax.cpp, ComputeResidual, without the Jacobian) in a new header next to CNEMOKokkosAUSM.hpp.
   - Build the serial target, then commit.
   - Next session: dispatch it in CNEMOKokkosUpwind.cpp (Centered_Residual path) behind the scheme check.
   - If 3 sessions pass without a compiling piece, mark 4.4-LAX BLOCKED and continue with 4.5.
2. Read at most 80 lines of any file at a time (lax.cpp is ~170 lines: read it in two halves).
