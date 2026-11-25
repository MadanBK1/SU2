# Review notes for the local agent (written by the Claude reviewer; agent: read, never edit)

Newest section first. Do the **ROLE CHANGE (Madan, 2026-11-25):** the reviewer now writes the solver code; you build, run, analyze and report. See ROLE at the top of LOCAL_AGENT_TASK.md. **Do NOT write the Roe code**: the reviewer writes it next round.

**Agent: do next** (tests/analysis only)
1. Pull the check.sh guard (already on this branch). Rerun G1 for LAX on the strict binary:
   `BIN=<strict SU2_CFD> REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 TestCases/kokkos_nemo/check.sh invwedge_lax`
   - Report PASS/FAIL and confirm the log has no `expected stage line missing`.
   - Paste the `[KOKKOS_NEMO] stage ...` lines and the `[KOKKOS_NEMO_PERF]` lines of the YES run (`chk_invwedge_lax_YES/log.txt`, max 30 lines) into KOKKOS_PROGRESS.md.
2. Coverage survey, same env, one case at a time: `... check.sh partial_cat`, then `... check.sh invwedge_am`.
   - For each case, report PASS/FAIL, max|dres| and the max field diff.
   - Also report every `[KOKKOS_NEMO] stage ...: host` line with its reason. This tells the developer what to port next.
3. Write `READY FOR REVIEW: role-switch tests` with a ≤ 15-line summary of 1+2.
   - Build first if the binary is stale (`ninja -C build-kokkos-strict`). Keep file reads ≤ 80 lines.
   - If a case crashes 3 sessions running, write `BLOCKED <case>: <last 10 log lines>` and request review.

**Madan: run on the V100 (G7 for LAX, optional):**
```
cd ~/SU2 && git fetch && git checkout su2kokkos && ninja -C build-kokkos-cuda
BIN=$PWD/build-kokkos-cuda/SU2_CFD/src/SU2_CFD REF_BIN=$PWD/build-kokkos-strict/SU2_CFD/src/SU2_CFD \
  REF_MODE=HOST PHYS_SCALE=1 RES_NOISE_ORDERS=10 TestCases/kokkos_nemo/check.sh invwedge_lax
```
(Adjust build dir names to yours; expect PASS within the noise floor, plus the stage line `centered residual (LAX-FRIEDRICH): device`.)


## 2026-11-25 — review 3
Reviewed: a912a139c56e1d6a0e65fbe8addec81e093ef29d (REVIEW_REQUEST 20261125T000426-a912a139c5)

**State:**
- Review-2 item 1 is DONE (in autosave commits bb96ce8..a912a13): `geo.nneighbor` is added, uploaded and self-checked. The code is correct: `GetnNeighbor` returns unsigned short, and CountMismatch compares bitwise.
- Items 2 and 3 were not started. 16 rounds (~4 h) produced one 7-line change; most rounds ended rc=1 with NO progress.
- Nothing merged into su2kokkos: nothing new has passed a gate.

**Reviewer fix e69cbbb (unblocks item 2):** I wrote the whole LAX dispatch. It is NOT built/linked or run yet; it was only checked with g++ -fsyntax-only against Kokkos Serial headers.
- `SU2_CFD/src/solvers/kokkos/CNEMOKokkosCentered.cpp` (new):
  - `LaxEdge<ND,ATOMIC>` = LAXResidual + CheckResidualNaNs + accumulation, the same as UpwindEdge.
  - `CNEMOKokkosData::ComputeCenteredResidual(implicit, kappa0, uploadAll)` uses EnsureInputsOnDevice (Solution + Primitive + dPdU) and uploads only the Lambda halo rows (all rows if `uploadAll` or `!lambdaValid`).
  - It returns the last edge's mean primitive state (nPrimVar doubles) for the host gas-state replay.
- `CNEMOEulerSolver.cpp`:
  - `Device_CenteredUnsupported`: LAX only; ionization is refused.
  - `Device_CenteredResidual`: ResidualToDevice → kernel → optional eager download → host replay `ComputeSpeciesEve(MeanV[nSpecies+1])` + `ComputedPdU` → A/B verify path.
  - It is called at the top of `Centered_Residual` (guarded by `centeredHostOnly`). The stage report now says where the centered residual runs.
- Also: a new `ST_CENTERED` perf stage ("CenteredLAX"), the `lambdaValid` flag (set in ComputeMaxEigenvalue, cleared in ActivateResident), a meson.build entry, and the `invwedge_lax` case in check.sh (uses the existing TestCases/nonequilibrium/invwedge/invwedge_lax.cfg: AIR-5, implicit, BCGSTAB).

**Findings:**
- CNEMOKokkosData.cpp:387/401/418/768 (nneighbor): OK.
- CNEMOEulerSolver.cpp `Device_CenteredResidual`: `uploadAll = verify || !Device_TimeStepUnsupported(config).empty()`. If SetMax_Eigenvalue ran on the host, Lambda is re-uploaded in full, so it is never stale.
- The flux BCs stay on the host for LAX, because the boundary numerics are Roe. So the accumulator goes to the host after the centered stage, and resident mode stays off for centered schemes (`"centered scheme (host)"`). That is expected; G1/G2 still apply.
- If the strict gate shows a diff only after the first host BC, suspect the gas replay. CUpwRoe_NEMO may read FluidModel members that the CPU LAX loop left behind. Compare with `SU2_KOKKOS_STAGE_VERIFY=1` (per-stage LinSysRes/Jacobian).

**Speedup/accuracy suggestions:**
- No new per-iteration full copies: only the Lambda halo rows (none on 1 rank) and nPrimVar doubles come back.
- The LAX kernel needs no MUSCL or limiter, and 2 Jacobian stacks of MAXNV² doubles like AUSM. The register pressure should be similar to UpwindAUSM.
- The next big win for centered cases is a device Roe boundary flux (BC_Far_Field / BC_Outlet with CUpwRoe_NEMO). That would allow resident mode for LAX. It is a later item: do not start it now.

**Agent: do next**
1. Build the serial target (`ninja -C build-kokkos-serial`). If CNEMOKokkosCentered.cpp or CNEMOEulerSolver.cpp fails to compile or link, fix only the reported line (most likely a missing include or a name) and commit. Do not restructure it.
2. Get meshes into MESH_DIR (invwedge.su2). Run `REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 TestCases/kokkos_nemo/check.sh invwedge_lax`. The log line `[KOKKOS_NEMO] stage centered residual (LAX-FRIEDRICH): device` must appear.
   - If it passes, also run the default case list (no args) with the same env, to prove nothing else changed.
   - Then write `READY FOR REVIEW: 4.4-LAX` with both results in KOKKOS_PROGRESS.md.
3. If it fails: rerun with `SU2_KOKKOS_STAGE_VERIFY=1` and paste the CenteredLAX StageCompare lines into KOKKOS_PROGRESS.md (max 20 lines).
   - If 3 sessions pass without a pass, write `BLOCKED 4.4-LAX: <the lines>` and continue with 4.5.
   - Keep every file read ≤ 80 lines. Run the build within your first 5 tool calls.


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
