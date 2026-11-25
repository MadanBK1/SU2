# Review notes for the local agent (written by the Claude reviewer; agent: read, never edit)

Newest section first. Do the **Agent: do next** items (tests/analysis), then write READY FOR REVIEW (see ROLE in LOCAL_AGENT_TASK.md).

## 2026-11-25 — review 14 (merged 4.5-SlipWall; dev: viscous SUPERSONIC_INLET)

**Reviewed:** 59fb0105e2. Strict-gate claims checked (REF_MODE=HOST RES_TOL=0 FIELD_TOL=0, OVERALL PASS, code
unchanged since 0d7a7b9). **Merged into su2kokkos (33bd882):** the 0d7a7b9 code files (CNEMOKokkosData.hpp,
CNEMOEulerSolver.cpp, CNEMONSSolver.cpp, CNEMOKokkosResidual.cpp, check.sh, check_noisefloor.sh, smol_maxwell.cfg).

**Dev code this round (g++ -std=c++20 -fsyntax-only -ffp-contract=off vs bundled Kokkos 5.2.2 Serial headers on
CNEMOEulerSolver.cpp and CNEMOKokkosResidual.cpp; NOT validated):**
- Viscous part of `BC_Supersonic_Inlet` on the device. It reuses the viscous FAR_FIELD code in `KernelBoundaryFlux`
  (CAvgGrad_NEMO, reflected normal neighbour, iPoint transport on both sides). The j state is `node_inlet` (V, eve,
  Cvve, dTdU, dTvedU, gradient), so it matches the CPU call exactly.
- `Device_BCUnsupported`: the "viscous contribution" fallback for SUPERSONIC_INLET is removed. Without NS variables it
  still falls back (same rule as FAR_FIELD).
- New gate case `supin_visc` (visc_wedge, `Inlet` = SUPERSONIC_INLET, non-catalytic isothermal wall, AUSM implicit),
  with `EXPECT` lines for SUPERSONIC_INLET / FAR_FIELD / ISOTHERMAL and a noise-floor case.
- Perf note (do not fix now): BState / BStateVisc are one shared buffer. With FAR_FIELD + SUPERSONIC_INLET in one case
  they are re-uploaded on every BC call (~100 doubles, 2x per iteration). If G7 shows it, the fix is one buffer per marker.

**Agent: do next**
1. Build strict + serial (ninja, as before). Fix compile errors yourself with the smallest change.
2. CPU sanity: run `supin_visc` NO side only. If the CPU itself diverges or errors (inlet values: 288.15 K, 10 Pa,
   1700 m/s), lower the velocity to 1600.0 in supin_visc.cfg and note it. That is a cfg edit, not code.
3. G1 strict: `timeout 1500 bash .g1run.sh ../kk/logs/g1-supin-r59.log supin_visc` -> needs OVERALL: PASS bitwise and
   all 3 EXPECT stage lines device, `device-resident state: ON`.
4. If it fails: `timeout 1500 bash .svrun.sh ../kk/logs/sv-supin-r59.log supin_visc`, paste ≤ 20 StageCompare lines.
   Try 3 sessions to localize the problem (bc:SUPERSONIC_INLET stage? viscous term only? implicit Jacobian only?)
   before requesting `HEAVY: 4.5-SupInVisc: <finding>`.
5. Regression: default list + invall (`... g1-regress-r59.log` with no args; `... g1-invall-r59.log invwedge_am
   invwedge_lax invwedge_roe invwedge_msw invwedge_ausmplusup2 super_cat smol_maxwell`), all PASS bitwise.
6. A/B on supin_visc (svrun) and noise floor: `CASE=supin_visc bash TestCases/kokkos_nemo/check_noisefloor.sh`
   (ratio ≤ 10).
7. PERF lines (BC s/call, wall NO/YES) from the YES log into KOKKOS_PROGRESS.md.
8. Bookkeeping: plan matrix row "SUPERSONIC_INLET viscous" -> device-resident (validated); LOCAL_AGENT_TASK 4.5 note.
   Then request `MERGE: 4.5-SupInVisc`.
Next heavy after that: 4.6 (LU-SGS) — request `HEAVY: 4.6-LUSGS` only after the merge. Item 6 (ionization) is still
waiting for Madan's decision.

## 2026-11-25 — review 13 (no merge: stale request; item 6 needs Madan's decision)

Reviewed: 16fffed237 (REVIEW_REQUEST 20261125T145530-16fffed237). **Nothing to merge:** its reason text is the review-12
request again, and 4.4-AUSMup2 is already in su2kokkos (1e8c2e1). Your loop re-posted the old READY line in the
round-54 autosave. Only request after you finish a queue, with a NEW reason line.

**Heavy item 6 (ionization with Roe / LAX / MSW): not ported, decision for Madan.** The CPU result is not defined, so a
noise-floor gate cannot pass either:
- `CNEMOGas::ComputedPdU` (CNEMOGas.cpp:156-236) takes `rho_el = rhos[0]` from the member *before* it reloads rhos
  from V (the previous call's state) and `Tve` from a stale member (not V[TVE]).
- The heavy loop runs over [nEl, nHeavy) = [1, nSpecies-1), so `dPdU[nSpecies-1]` is never assigned, only `-=`.
  RoedPdU / MeandPdU / dPdUst are `new` with no init, so that entry is uninitialised heap memory plus a running sum
  over every edge since start-up. CPU NO-vs-NO is deterministic only by accident, and device-vs-CPU cannot converge.
- Options: (A) keep it on the host (current; reported by Device_UpwindUnsupported). (B) Madan approves an opt-in fix
  (for example `SU2_NEMO_DPDU_FIX=1`: rho_el = V[0], Tve = V[TVE], heavy loop to nSpecies, dPdU zeroed) applied to
  BOTH CNEMOGas::ComputedPdU and NEMOGasKokkos ComputedPdU, gated G1 bitwise against the patched CPU. That is a
  physics change, so not done silently. Until Madan decides, skip item 6. The next heavy item after the slip wall is
  4.6 (request `HEAVY: 4.6-LUSGS` or `HEAVY: 4.6-ILU`).

**Dev code this round:** none. The slip wall (0d7a7b9) is still untested.

**Agent: do next** — the review-12 queue, unchanged (steps 1-8 below: build, CPU sanity, G1 smol_maxwell, A/B on
failure, regression, noise floor, perf, bookkeeping). Also tick 4.4/queue item 6 in LOCAL_AGENT_TASK.md as
"blocked: CPU ComputedPdU ionization undefined, Madan decision (review 13)". Then request `MERGE: 4.5-SlipWall`.

## 2026-11-25 — review 12 (merged 4.4-AUSMup2; dev: 4.5 slip wall)

Reviewed: 24b4086c95 (REVIEW_REQUEST 20261125T141839-24b4086c95).
**Merged into su2kokkos (1e8c2e1):** AUSM+-up2 explicit interior + boundary and the GetKind_TimeIntScheme_Flow
stage-report fix (CNEMOKokkosData.hpp, CNEMOKokkosAUSM.hpp, CNEMOEulerSolver.cpp, check.sh, check_noisefloor.sh,
plan matrix). Your G1 / regression / A/B / noisefloor / perf evidence accepted.

**BC_Inlet: not portable.** CPU `CNEMOEulerSolver::BC_Inlet` calls `SU2_MPI::Error("BC_INLET: Not operational in
NEMO.")` at entry (and its body is incomplete: no Tve/Eve, species densities 0). There is no CPU reference, so it stays
host. Tick it in LOCAL_AGENT_TASK.md 4.5 as "n/a (CPU not operational)". The catalytic heat-flux wall is the same
(CPU not operational, already reported as such).

**Dev code this round (0d7a7b9; g++ -std=c++20 -fsyntax-only -ffp-contract=off vs bundled Kokkos 5.2.2 Serial
headers on CNEMOKokkosResidual.cpp, CNEMOEulerSolver.cpp, CNEMONSSolver.cpp; NOT validated):**
- `KernelSmoluchowski<ND>` (CNEMOKokkosResidual.cpp) = verbatim `CNEMONSSolver::BC_Smoluchowski_Maxwell`. There is no
  Jacobian term (as on the CPU). GasConstant = host `FluidModel->ComputeGasConstant()`, evaluated once per call
  (`BCParams::gasConstant`), plus `prandtlLam`. SetVelocity_Old goes to Solution_Old (resident); otherwise it goes to
  a per-vertex `st.SlipVel` buffer that the host downloads and applies, with the momentum Res_TruncError rows zeroed.
- `BCKind::SMOLUCHOWSKI_MAXWELL` (appended): Device_BCUnsupported (ionization -> host), Device_BCResidual (+ A/B
  verify switch), stage report, residency marker map, dispatch at the top of CNEMONSSolver::BC_Smoluchowski_Maxwell.
- New test cfg `TestCases/nonequilibrium/visc_wedge/smol_maxwell.cfg` (= super_cat.cfg with the wall as
  `MARKER_SMOLUCHOWSKI_MAXWELL= (Wall, 300)`, catalytic lines removed). check.sh case `smol_maxwell`, with EXPECT
  `stage bc:SMOLUCHOWSKI_MAXWELL (Wall): device`. Noise-floor case `smol_maxwell` (HREL=1).

**Agent: do next**
1. Build strict + serial: `timeout 3000 bash ../kk/build.sh ninja strict 4`, then the serial one. Fix compile errors
   yourself with the smallest change (nvcc/strict may flag things the syntax check did not).
2. CPU sanity first: run the NO (host) side of smol_maxwell once. If the CPU itself diverges or aborts on this cfg
   (NaN, "Not operational"), report it in KOKKOS_PROGRESS.md and try `ITER= 5` or `CFL_NUMBER` halved in the cfg. That
   is a cfg edit you may make.
3. G1 strict: `timeout 1500 bash .g1run.sh ../kk/logs/g1-smol-r55.log smol_maxwell` -> need OVERALL PASS bitwise, no
   `expected stage line missing`.
4. On FAIL: `timeout 1500 bash .svrun.sh ../kk/logs/sv-smol-r55.log smol_maxwell`. Paste <= 20 StageCompare lines
   (`grep StageCompare`). Suspects, in order: the ComputeGasConstant host state (print `p.gasConstant` vs the CPU value
   on the first call), then dTven / dTn == 0 branches, then SetVelocity_Old (Solution_Old) in resident mode.
5. Regression: `timeout 3000 bash .g1run.sh ../kk/logs/g1-regress-r55.log` (default list) and
   `... g1-invall-r55.log invwedge_am invwedge_lax invwedge_roe invwedge_msw invwedge_ausmplusup2 super_cat`.
6. A/B: `.svrun.sh` on smol_maxwell (bitwise). Noise floor: `CASE=smol_maxwell bash TestCases/kokkos_nemo/check_noisefloor.sh`.
7. PERF: BC s/call and wall NO/YES from the G1 YES log. Bookkeeping: 4.5 in LOCAL_AGENT_TASK.md (slip wall done,
   BC_Inlet / catalytic heat flux n/a), plus the coverage matrix in NEMO_KOKKOS_PORT_PLAN.md.
8. Request `MERGE: 4.5-SlipWall` only after steps 3, 5 and 6 pass. Request `HEAVY: ...` only for new code, or for a bug
   you cannot localize after 3 sessions. The next heavy item after this is queue 6 (ionization with Roe / LAX), then
   4.6. Request it as `HEAVY: 4.6-<option>` or `HEAVY: ionization-Roe`.

Madan (V100, optional): G7 GPU timings are still outstanding (4.3). Commands are in the earlier notes / LOCAL_AGENT_TASK.md 4.3.

## 2026-11-25 — review 11 (no merge: AUSM+-up2 host fallback was a stage-report bug, fixed)

Reviewed: 2505fadae8 (REVIEW_REQUEST 20261125T132536-2505fadae8). Nothing merged.
Diagnosis: invwedge_ausmplusup2.cfg IS explicit (`TIME_DISCRE_FLOW= EULER_EXPLICIT`; LINEAR_SOLVER is ignored when
explicit). The Device_*Unsupported checks used the *global* `GetKind_TimeIntScheme()`, which is still at its
CConfig default EULER_IMPLICIT (CConfig.cpp:1097) when the setup stage report runs (before SetGlobalParam), so the
report printed "implicit AUSM+-up2" for an explicit case. No implicit up2 port is needed.
Dev code this round: CNEMOEulerSolver.cpp — Device_UpwindUnsupported / Device_CenteredUnsupported /
Device_SourceUnsupported / Device_BCUnsupported now use `GetKind_TimeIntScheme_Flow()` (6 lines; identical at run time
for NEMO, fixes the setup report). Not built here.

**Agent: do next** (one step per session; foreground + `timeout` + `tail`; write each result into PROGRESS at once):
1. Build: `timeout 3000 bash ../kk/build.sh ninja strict 4 > ../kk/logs/build-strict-r52.log 2>&1; echo rc=$?; grep -m 10 -i 'error' ../kk/logs/build-strict-r52.log`, then the same with `serial` (fix compile errors yourself, smallest change).
2. G1: `timeout 1500 bash .g1run.sh ../kk/logs/g1-up2-r52.log invwedge_ausmplusup2 > /dev/null 2>&1; echo rc=$?; tail -n 25 ../kk/logs/g1-up2-r52.log`
   -> need OVERALL PASS, max|dres|=0, field diff 0, no "expected stage line missing"; upwind + 3 BC lines must say device.
3. On bitwise FAIL with device stages: `timeout 1500 bash .svrun.sh ../kk/logs/sv-up2-r52.log invwedge_ausmplusup2 > /dev/null 2>&1; grep -m 20 StageCompare ../kk/logs/sv-up2-r52.log` -> paste; localize the first differing stage (3 sessions max) before asking HEAVY.
4. Regression (the fix touches every case's stage report): G1 on the default list (all invwedge + the other check.sh cases), each with timeout+tail; all must PASS with unchanged stage lines.
5. On PASS: review-8 steps 4–8 (A/B non-bitwise=0, noise floor <=10, PERF lines, bookkeeping), then `READY FOR REVIEW` with `reason: MERGE: 4.4-AUSMup2` (include the stage-report fix). Next heavy after merge: 4.5 BCs (subsonic BC_Inlet first) — request it as `HEAVY: 4.5-BC_Inlet`.

## 2026-11-25 — review 10 (no merge: AUSM+-up2 YES run fell back to host; likely a stale binary)

Reviewed: 1b09c67452 (REVIEW_REQUEST 20261125T105059-1b09c67452, `reason:` empty — always write one). Nothing merged.
Diagnosis: for invwedge_ausmplusup2 (EULER_EXPLICIT, no MUSCL) the dispatch code at bc75572 has no host reason for
upwind or BCs; the 3 BC lines go host only because the upwind stage went host (accumulator rule). PROGRESS never
records the strict build of bc75572 finishing -> the strict binary most likely predates bc75572 (prints "upwind (other …)
host (upwind scheme not ported)"). check.sh now prints the actual `got:` stage lines when an EXPECT line is missing.

**Agent: do next** (one step per session; foreground + `timeout` + `tail`; write each result line into PROGRESS at once):
1. Show the reason: `grep -E '^\[KOKKOS_NEMO\] stage (upwind|bc:)' ../kk/runs/chk_invwedge_ausmplusup2_YES/log.txt` -> paste.
   Then `git log -1 --format=%cd bc75572; ls -l --time-style=full-iso <strict SU2_CFD path used by .g1run.sh>` -> paste.
2. If the line says "other"/"upwind scheme not ported" or the binary is older than bc75572: rebuild,
   `timeout 3000 bash ../kk/build.sh ninja strict 4 > ../kk/logs/build-strict-r47.log 2>&1; echo rc=$?; grep -m 10 -i 'error' ../kk/logs/build-strict-r47.log`
   (fix compile errors yourself, smallest change), then the same with `serial`.
3. Rerun G1: `timeout 1500 bash .g1run.sh ../kk/logs/g1-up2-r47.log invwedge_ausmplusup2 > /dev/null 2>&1; echo rc=$?; tail -n 25 ../kk/logs/g1-up2-r47.log`
   -> need OVERALL PASS, max|dres|=0, no "expected stage line missing".
4. If the binary is fresh and a host reason is still printed: paste the `got:` lines and request
   `HEAVY: 4.4-AUSMup2: host fallback reason "<reason>"`. On a bitwise FAIL with device stages: review-9 step 3 (.svrun + StageCompare).
5. On PASS: review-8 steps 4–8 (9-case regression, A/B non-bitwise=0, noise floor <=10, PERF, bookkeeping) each with timeout+tail,
   then `READY FOR REVIEW: MERGE: 4.4-AUSMup2`. Next heavy after merge: 4.5 BCs (subsonic BC_Inlet first).

## 2026-11-25 — review 9 (no merge: empty request; AUSM+-up2 queue stalled)

Reviewed: a2d341cb73 (REVIEW_REQUEST 20261125T082316-a2d341cb73, `reason:` empty). Rounds 31–38 changed only
KOKKOS_PROGRESS.md (autosaves) — no gate result for invwedge_ausmplusup2 was ever recorded. Nothing merged.
No new dev code this round: finish testing bc75572 (AUSM+-up2) first; 4.5 BCs come after its merge.

**Agent: do next** — one step per session, write the result line into PROGRESS *immediately* after each command:
1. Do NOT run gates in the background and do NOT `cat` whole logs. Run in the foreground with a hard limit and tail:
   `timeout 1500 bash .g1run.sh ../kk/logs/g1-up2-r39.log invwedge_ausmplusup2 > /dev/null 2>&1; echo rc=$?; tail -n 25 ../kk/logs/g1-up2-r39.log`
   Write rc + PASS/FAIL + max|dres| into PROGRESS before anything else.
2. If rc=124 (timeout) or the log has no OVERALL line: run the case alone, strict YES, 11 iters, and paste the last
   15 lines: `cd TestCases/nonequilibrium/invwedge && timeout 600 SU2_KOKKOS_NEMO=1 <strict SU2_CFD> invwedge_ausmplusup2.cfg 2>&1 | tail -n 15`
   (use the exact binary path/env that .g1run.sh uses). Then same with `SU2_KOKKOS_NEMO_HOST=upwind` and with
   `SU2_KOKKOS_NEMO_HOST=bc`; one line each in PROGRESS: does it finish, residual at iter 10, NaN?
3. On FAIL (bitwise): `timeout 1500 bash .svrun.sh ../kk/logs/sv-up2-r39.log invwedge_ausmplusup2 > /dev/null 2>&1; grep -m 20 StageCompare ../kk/logs/sv-up2-r39.log`
   Suspects: kmax/kmin argument order, `3.0 / 16.0 * (...)` order, far-field Gamma_j (node_infty).
4. On PASS continue review-8 steps 4–8 (regression 9-case, A/B, noise floor, PERF, bookkeeping), each with
   `timeout` + `tail`, then `READY FOR REVIEW: MERGE: 4.4-AUSMup2`.
5. If after 3 sessions you still cannot get a result line for step 1/2: write `READY FOR REVIEW: BLOCKED 4.4-AUSMup2: <last 15 log lines>`
   (paste the lines) — never an empty reason. Fix compile errors yourself (smallest change).

## 2026-11-25 — review 8 (MERGE 4.4-MSW; HEAVY: device AUSM+-up2, explicit)

Reviewed: 716f6200f5. **Merged** 4.4-MSW code (cce61ae + check_noisefloor msw case) into `su2kokkos` (effdde0).

**Dev code this round (bc75572; g++ -std=c++20 -fsyntax-only -ffp-contract=off vs Kokkos Serial headers on
CNEMOKokkosUpwind.cpp, CNEMOKokkosResidual.cpp, CNEMOEulerSolver.cpp; NOT validated):**
- `CNEMOKokkosAUSM.hpp` `AUSMResidual`: new branch `UpwindScheme::AUSMPLUSUP2` = verbatim
  CUpwAUSMPLUSUP2_NEMO::ComputeInterfaceQuantities (C*, C^, aF, Mp, modified pressure flux; Kp=0.25, sigma=1, Minf=a.Minf).
- `CNEMOKokkosData.hpp`: enum `AUSMPLUSUP2 = 4`. `CNEMOEulerSolver.cpp`: interior + BC scheme mapping;
  `Device_UpwindUnsupported` / `Device_BCUnsupported` accept up2 **explicit only** (implicit: CPU Jacobian reads the
  never-set M_L/M_R members -> host, documented quirk); up2 SUPERSONIC_INLET -> host (stale boundary Gamma not replayed).
- check.sh case `invwedge_ausmplusup2` (EULER_EXPLICIT, no MUSCL, CFL 0.25) + EXPECT upwind (AUSM+-up2, residual) device
  + 3 BC lines; check_noisefloor.sh case `invwedge_ausmplusup2`.

**Agent: do next** (strict binary for gates; fix compile errors yourself with the smallest change):
1. Build strict + serial (`bash ../kk/build.sh ninja strict 4`, then `serial`); paste first 10 error lines + fix.
2. G1 strict: `bash .g1run.sh ../kk/logs/g1-up2-r30.log invwedge_ausmplusup2` -> PASS bitwise, no
   "expected stage line missing". Paste YES stage lines (`grep '^\[KOKKOS_NEMO\] stage\|device-resident' log.txt`);
   residency should be ON — if OFF paste the reason line.
3. On FAIL: `bash .svrun.sh ../kk/logs/sv-up2-r30.log invwedge_ausmplusup2`; paste <= 20 StageCompare lines (Upwind, BC
   first); isolate with `SU2_KOKKOS_NEMO_HOST=bc`. Look first at: kmax/kmin argument order (CPU std::max/min),
   `3.0 / 16.0 * (...)` order, Gamma_j at the far-field (node_infty Gamma).
4. Regression (AUSMResidual changed): `bash .g1run.sh ../kk/logs/g1-regress-r30.log invwedge_ausm thermalbath super_cat axi_visccone invwedge_lax invwedge_roe invwedge_msw invwedge_am partial_cat` -> all PASS bitwise.
5. A/B even on PASS: `bash .svrun.sh ../kk/logs/sv-up2-r30.log invwedge_ausmplusup2` -> non-bitwise=0.
6. Noise floor: `CASE=invwedge_ausmplusup2 bash TestCases/kokkos_nemo/check_noisefloor.sh` -> ratio <= 10.
7. PERF: wall NO/YES + `[KOKKOS_NEMO_PERF]` Upwind s/call for invwedge_ausmplusup2.
8. Bookkeeping after 2+4 pass: 4.4 AUSM+-up2 (explicit) ticked, coverage matrix row "AUSM+-up2: device (explicit);
   implicit host (CPU quirk)". Then `READY FOR REVIEW: MERGE: 4.4-AUSMup2`.
9. Next heavy after merge: 4.5 BCs (subsonic BC_Inlet first). Request `HEAVY: …` only for new code or a bug you
   cannot localize after 3 sessions. Madan still owes G7 (V100 timings) — not your job.

## 2026-11-25 — review 7 (MERGE 4.4-Roe-BC + LAX resident; HEAVY: device MSW)
Reviewed: 4a613f35d980b99f02dd8e39507a454d33e97164 (REVIEW_REQUEST 20261125T053245-4a613f35d9)

**Merged into su2kokkos (d761e92):** Roe boundary numerics + LAX residency (CNEMOKokkosData.hpp, CNEMOEulerSolver.cpp,
CNEMOKokkosResidual.cpp, check.sh). Your G1 / regression / A/B / perf evidence accepted. G5 (cuda build) PASS noted.

**Dev code this round (cce61ae; g++ -std=c++20 -fsyntax-only -ffp-contract=off vs bundled Kokkos Serial headers on
CNEMOKokkosUpwind.cpp, CNEMOKokkosResidual.cpp, CNEMOEulerSolver.cpp; NOT validated):**
- New `CNEMOKokkosMSW.hpp`: `MSWResidual<ND>` = verbatim CUpwMSW_NEMO::ComputeResidual (flux + implicit Jacobian;
  dPdU at the weighted states; reuses RoeCreateBasis / RoeGetPMatrix / RoeGetPMatrixInv).
- `UpwindScheme::MSW` (=3), `UpwindParams::mswAlpha` (config MSW_ALPHA) -> `UpwArgs::mswAlpha`.
- Interior: `UpwindEdge` uses the Roe conservative-state path (Solution / RecomputeConservativeVector of the MUSCL
  state) for MSW and calls `MSWResidual`. Boundary: `KernelBoundaryFlux` same (CDriver: MSW boundary numerics = MSW).
- Host: `Device_UpwindUnsupported` / `Device_BCUnsupported` accept MSW (host only with ionization / gas-table
  mismatch); host replay of the last reconstructed state also does RecomputeConservativeVector for MSW; stage name "MSW".
- check.sh: new case `invwedge_msw` (TestCases/nonequilibrium/invwedge/invwedge_msw.cfg, MUSCL + EULER_IMPLICIT) with
  EXPECT = upwind (MSW + MUSCL, residual + Jacobian) device + the 3 device BC lines.

**Agent: do next** (strict binary for gates; fix compile errors yourself with the smallest change):
1. Build: `bash ../kk/build.sh ninja strict 4` and the serial build (same script, `serial`); paste any error (first
   10 lines) + your fix in PROGRESS.
2. G1 strict: `bash .g1run.sh ../kk/logs/g1-msw-r28.log invwedge_msw` -> PASS bitwise (max|dres|=0, field diff=0), no
   "expected stage line missing". Paste the YES stage lines (`grep '^\[KOKKOS_NEMO\] stage\|device-resident' log.txt`).
   Residency for MSW should be ON (same rules as Roe); if OFF paste the reason line.
3. On FAIL: `bash .svrun.sh ../kk/logs/sv-msw-r28.log invwedge_msw`; paste <= 20 StageCompare lines (Upwind and BC
   first). If only BC differs, rerun with `SU2_KOKKOS_NEMO_HOST=bc` to isolate (interior-only device). Look first at:
   pow(alpha*dp,2.0) (should be bitwise on Serial), kmin argument order, Jacobian zeroing.
4. Regression: `bash .g1run.sh ../kk/logs/g1-regress-r28.log invwedge_ausm thermalbath super_cat axi_visccone invwedge_lax invwedge_roe invwedge_am partial_cat`
   -> all PASS bitwise (shared kernels changed: UpwindEdge, KernelBoundaryFlux).
5. A/B even on PASS: `bash .svrun.sh ../kk/logs/sv-msw-r28.log invwedge_msw` -> all stages non-bitwise=0.
6. PERF: wall NO/YES + `[KOKKOS_NEMO_PERF]` Upwind s/call for invwedge_msw (expect > Roe: two P/invP pairs per edge).
7. Bookkeeping: tick MSW in LOCAL_AGENT_TASK.md section 4 / 4.4 and the coverage matrix only after 2+4 pass. Then
   `READY FOR REVIEW: MERGE: 4.4-MSW`.
8. If time remains (no new code needed): noise-floor gate on invwedge_msw (`bash TestCases/kokkos_nemo/check_noisefloor.sh invwedge_msw`)
   and record the ratio. Next heavy item after MSW: AUSM+-up2 (queue item 4); LAX multi-rank Lambda halo (item 2)
   only if you can give an np=2 case + failure log.

**Madan (V100, optional):** G7 timings with the cuda build once 4.4-MSW passes:
`bash ../kk/build.sh ninja cuda 4 && bash TestCases/kokkos_nemo/check.sh invwedge_roe invwedge_msw invwedge_lax`
(binary build-kokkos-cuda; record wall NO/YES and PERF lines in KOKKOS_PROGRESS.md).

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
