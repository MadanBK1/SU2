# Review notes for the local agent (written by the Claude reviewer; agent: read, never edit)

Newest section first. Do the **Agent: do next** items (tests/analysis), then write READY FOR REVIEW (see ROLE in LOCAL_AGENT_TASK.md).

## 2026-11-25 — review 6 (MERGE 4.4-Roe interior + HEAVY: Roe boundary numerics, LAX residency)
Reviewed: 0f654bab4db3b04de41e850ddaee3f1844396fb2 (REVIEW_REQUEST 20261125T041000-0f654bab4d)

**Merged into su2kokkos (d63a151):** interior device Roe (code of e744eb0: CNEMOKokkosRoe.hpp, CNEMOKokkosUpwind.cpp,
CNEMOKokkosAUSM.hpp, CNEMOKokkosData.hpp, CNEMOEulerSolver.cpp, check.sh). Your G1 / regression / A/B evidence accepted.

**Dev code this round (g++ -std=c++20 -fsyntax-only -ffp-contract=off vs Kokkos 4.6 Serial headers; NOT validated):**
- Device BCs with **Roe boundary numerics** (CDriver: LAX and ROE use CUpwRoe_NEMO at FAR_FIELD / OUTLET /
  SUPERSONIC_INLET / SUPERSONIC_OUTLET): `KernelBoundaryFlux` (CNEMOKokkosResidual.cpp) calls `RoeResidual` with
  U_i = Solution(iPoint), U_j = node_infty / node_inlet Solution (appended to BState), U_outlet (OUTLET), U_i (SUP_OUTLET).
  `ComputeBCResidual` uploads Solution when Roe. Host: `BoundaryState::U`, `p.upw.scheme = ROE` when centered or ROE,
  `Device_BCUnsupported` now allows Roe boundary numerics (host only with ionization / gas-table mismatch).
- **LAX residency:** `Device_ResidentUnsupported` accepts the centered scheme when `Device_CenteredUnsupported` is empty
  and there are no halo points (single rank; Lambda halo rows still come from the host comms).
- check.sh: `EXPECT` accepts several lines separated by `|`; invwedge_lax / invwedge_roe now also require the device
  BC lines (FAR_FIELD (Inlet), OUTLET_FLOW (Exit), EULER_WALL (Wall)).

**Agent: do next** (strict binary for gates; fix compile errors yourself with the smallest change):
1. Build: `ninja -C build-kokkos-strict` and the serial build; paste any error (first 10 lines) + your fix in PROGRESS.
2. G1 strict: `bash .g1run.sh ../kk/logs/g1-bcroe-r26.log invwedge_lax invwedge_roe` → both PASS bitwise, no
   "expected stage line missing". Paste the YES stage lines (grep '^\[KOKKOS_NEMO\] stage\|device-resident' log.txt)
   for both cases. Expected: all BCs device; LAX: "device-resident state (Step 6): ON" (if OFF, paste the reason).
3. On FAIL: make .svrun.sh take the case as $2 (small edit: `bash ... check.sh ${2:-invwedge_roe}`), run
   `bash .svrun.sh ../kk/logs/sv-bcroe-r26.log invwedge_lax` (and invwedge_roe); paste <= 20 StageCompare lines
   (stage BC first). Also try LAX with `SU2_KOKKOS_NEMO_RESIDENT=0` to separate a BC bug from a residency bug.
4. Regression: `bash .g1run.sh ../kk/logs/g1-regress-r26.log invwedge_ausm thermalbath super_cat axi_visccone invwedge_am partial_cat`
   → all PASS bitwise (AUSM BC path changed only by refactoring U_outlet / U_j pointers).
5. A/B even on PASS: `bash .svrun.sh ../kk/logs/sv-bcroe-r26.log invwedge_lax` → BC non-bitwise=0.
6. PERF: from the LAX + Roe YES logs paste the per-stage PERF lines (BC, CenteredLAX, Upwind, Source, ResidualSync,
   update) and the YES/NO wall times; note any remaining h2d/d2h per iteration.
7. Then `READY FOR REVIEW: MERGE: 4.4-Roe-BC (+LAX resident)` with the numbers, or `HEAVY:` only if a failure is not
   localized after 3 sessions. Next heavy item after this merge: MSW.

**Madan (V100, optional after step 2 passes):** CUDA timings G7 for invwedge_lax / invwedge_roe with the CUDA build,
with your usual G7 command (CUDA build, noise-floor tolerances, cases invwedge_lax invwedge_roe), and report the YES/NO wall times.

## 2026-11-25 — review 5 (HEAVY: 4.4-Roe)
Reviewed: 883d4141ad27b62c5623c48cd2f00aa355f3e56e (REVIEW_REQUEST 20261125T032955-883d4141ad)

**State:** survey accepted (LAX, partial_cat, invwedge_am PASS bitwise, 100% device). Nothing to merge (no new code from you).

**Dev code this round (compile-checked with g++ -fsyntax-only, Kokkos Serial headers; NOT yet validated):**
- NEW `SU2_CFD/src/solvers/kokkos/CNEMOKokkosRoe.hpp`: RoeCreateBasis, RoeGetPMatrix, RoeGetPMatrixInv,
  RoeRecomputeConservative (MUSCL states), RoeResidual (= CUpwRoe_NEMO::ComputeResidual incl. Jacobian).
- `CNEMOKokkosUpwind.cpp` UpwindEdge: `UpwindScheme::ROE` branch (U = Solution, or RecomputeConservativeVector of the
  MUSCL state); ComputeUpwindResidual uploads Solution only if not resident (EnsureInputsOnDevice).
- `CNEMOKokkosAUSM.hpp` UpwArgs: new `U` view. `CNEMOKokkosData.hpp`: `UpwindScheme::ROE = 2`.
- `CNEMOEulerSolver.cpp`: Device_UpwindUnsupported accepts ROE (not with IONIZATION: CPU RoedPdU depends on the
  previous edge), p.scheme ROE, host replay adds RecomputeConservativeVector(lastRec), stage name "ROE".
- `check.sh`: case `invwedge_roe` (MUSCL + Venkatakrishnan + implicit) with EXPECT line
  `stage upwind (ROE + MUSCL, residual + Jacobian): device`.
- Boundaries still use host Roe (Device BC flux is AUSM-only, CNEMOEulerSolver.cpp ~3531). That is the next HEAVY item
  (device Roe boundary flux -> LAX/Roe residency). Do not write it yourself.

**Agent: do next** (build + tests; fix compile errors yourself with the smallest change)
1. Build both: `ninja -C build-kokkos-strict && ninja -C build-kokkos-serial`. Compile errors: fix minimally, note them in KOKKOS_PROGRESS.md.
2. G1 strict on the new case:
   `BIN=<strict SU2_CFD> REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 TestCases/kokkos_nemo/check.sh invwedge_roe`
   - Report PASS/FAIL, max|dres|, max field diff; confirm no `expected stage line missing`.
   - Paste the `[KOKKOS_NEMO] stage ...` and `[KOKKOS_NEMO_PERF]` lines of the YES run (≤ 30 lines).
3. If FAIL: rerun the YES config with `SU2_KOKKOS_STAGE_VERIFY=1` (same binary, `ITER= 3`) and paste ≤ 20 `StageCompare`
   lines (ST_UPWIND LinSysRes / Jacobian first). Then first-order isolation:
   `EXTRA_CFG="MUSCL_FLOW= NO" BIN=... REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_fo check.sh invwedge_roe`
   (EXPECT line then needs no "+ MUSCL": a missing-line failure there is expected; judge by the residual/field diffs),
   and explicit: `EXTRA_CFG="TIME_DISCRE_FLOW= EULER_EXPLICIT;CFL_NUMBER= 0.1" RUN_TAG=_ex ...`.
   Report which of {MUSCL, first order, explicit} differ -> tells me flux vs Jacobian vs reconstruction.
4. Regression: default list on the strict binary (`... check.sh` with no args) + `invwedge_lax invwedge_am` -> must stay OVERALL PASS.
5. Perf: from the YES log of step 2, the upwind per-call kernel time vs the host-ref time (STAGE_VERIFY run), one line.
6. If 2 and 4 pass: write `MERGE: 4.4-Roe (interior device Roe; G1 strict invwedge_roe PASS, default list PASS)`.
   If it fails and you cannot localize it in 3 sessions: `HEAVY: 4.4-Roe: <StageCompare lines + step-3 matrix>`.
   Keep file reads ≤ 80 lines.

**Madan (optional, V100):** after the merge, G7 timing:
`BIN=<cuda SU2_CFD> REF_MODE=HOST RES_TOL=2e-6 FIELD_TOL=1e-9 TestCases/kokkos_nemo/check.sh invwedge_roe` and send the
`[KOKKOS_NEMO_PERF]` upwind lines. The Roe kernel holds P, invP and 2 Jacobians (4·MAXNVAR² doubles): if it spills,
I will fuse P·|Λ| row-wise next.

## 2026-11-25 — review 4
Reviewed: f5fd2a403c35df7f86cfb547fba4376de12400d0 (REVIEW_REQUEST 20261125T020812-f5fd2a403c)

**State:**
- Review-3 items 1+2 DONE. No new solver code from the agent: the reviewer LAX code (e69cbbb + CNEMOKokkosLAX.hpp) built unchanged, and the agent ran the gates.
- Claimed G1 (strict binary, REF_MODE=HOST, RES_TOL=0, FIELD_TOL=0): invwedge_lax PASS bitwise, default 4-case list OVERALL PASS bitwise. Plausible: the agent also saw [KOKKOS_NEMO_VERIFY] CenteredLAX lines (only printed by Device_CenteredResidual), so the device path really ran.
- Good catch by the agent: the fast-math serial build cannot be bitwise; use BIN=strict for G1. Keep doing that.
- Not re-run here (container has 2 cores and no meson; a full build would not finish in time).
- **MERGED into su2kokkos (20eabd2):** 4.4-LAX = CNEMOKokkosLAX.hpp, CNEMOKokkosCentered.cpp, the dispatch in CNEMOEulerSolver, geo.nneighbor, ST_CENTERED, lambdaValid, meson entry, invwedge_lax in check.sh. local_agent_loop.sh was NOT merged (agent infrastructure).
- 4.4 LAX: tick it in LOCAL_AGENT_TASK.md as "LAX done (G1 strict)". Roe, MSW and AUSM+-up2 remain.

**Reviewer fix (this commit):** check.sh now has `EXPECT[invwedge_lax]`: the YES run must print `stage centered residual (LAX-FRIEDRICH): device`, else the case fails. This stops a silent host fallback from passing G1 trivially. Add an EXPECT entry for every new device scheme you gate.

**Findings:**
- CNEMOKokkosCentered.cpp:~185: `create_mirror_view_and_copy` allocates a host view every call (nPrimVar doubles). Harmless; later reuse a persistent host mirror.
- CNEMOEulerSolver.cpp Device_CenteredResidual: the host gas replay (ComputeSpeciesEve + ComputedPdU at the last edge's mean state) is needed for bitwise while the BCs run on the host with Roe. Keep it.
- LAX still runs with resident mode off, because the boundary flux numerics for centered schemes are CUpwRoe_NEMO (host). So a device Roe fixes two things at once: Roe interior cases, and full residency for LAX.

**Speedup/accuracy suggestions:**
- Next big step is the device Roe (CUpwRoe_NEMO). It reuses LaxInviscidProjFlux/LaxInviscidProjJac from CNEMOKokkosLAX.hpp; only GetPMatrix / GetPMatrix_inv (CNEMONumerics.cpp:355-~550) are new.
- The Roe kernel has 3 MAXNV² matrices (P, invP, Jacobian work). That is heavier than LAX, so watch registers on the V100; colored accumulation as in UpwindEdge.
- G7 on the V100 for invwedge_lax (Madan): see the commands at the end.

**ROLE CHANGE (Madan, 2026-11-25):** the reviewer now writes the solver code; you build, run, analyze and report. See ROLE at the top of LOCAL_AGENT_TASK.md. **Do NOT write the Roe code**: the reviewer writes it next round.

**Agent: do next** (tests/analysis only)
1. Pull the check.sh guard (already on this branch). Rerun G1 for LAX on the strict binary:
   `BIN=<strict SU2_CFD> REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 TestCases/kokkos_nemo/check.sh invwedge_lax`
   - Report PASS/FAIL and confirm the log has no `expected stage line missing`.
   - Paste the `[KOKKOS_NEMO] stage ...` lines and the `[KOKKOS_NEMO_PERF]` lines of the YES run (`chk_invwedge_lax_YES/log.txt`, max 30 lines) into KOKKOS_PROGRESS.md.
2. Coverage survey, same env, one case at a time: `... check.sh partial_cat`, then `... check.sh invwedge_am`.
   - For each case, report PASS/FAIL, max|dres| and the max field diff.
   - Also report every `[KOKKOS_NEMO] stage ...: host` line with its reason. This tells the developer what to port next.
3. Write `READY FOR REVIEW: HEAVY: 4.4-Roe: device Roe needed; survey results: <≤ 15-line summary of 1+2>`. The developer then writes Roe. (Request format: see ROLE in LOCAL_AGENT_TASK.md.)
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
