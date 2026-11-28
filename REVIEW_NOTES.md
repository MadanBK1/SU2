# Review notes for the local agent (written by the Claude reviewer; agent: read, never edit)

Newest section first. Do the **Agent: do next** items (tests/analysis), then write READY FOR REVIEW (see ROLE in LOCAL_AGENT_TASK.md).

## 2026-11-28 — review 43 (request 20261128T051009 had an EMPTY reason; nothing to merge, no new dev code)
Reviewed: 6bb09ca5a6 (only loop autosaves + limfix_queue.sh / limfix_build_and_run.sh + check.sh hostenv inherit; OK).
No results exist yet for review 41 or 42: validation_logs/2026-11-28_4.6-PeriodicLim-fix/ is not committed, rounds
202-205 made no progress. No new heavy code this round on purpose: 2 dev items (limValid fix 9054ef3b25, bndsync
0ec62f5abb) are untested; clear them first. Your queue scripts look correct (SU2_KOKKOS_NEMO_HOST=lim is honoured).
Agent: do next (in order; one step per session is fine; never send a request with an empty reason)
1. `cat validation_logs/2026-11-28_4.6-PeriodicLim-fix/BUILD_HEAD.txt validation_logs/2026-11-28_4.6-PeriodicLim-fix/QUEUE_STATUS.txt; pgrep -af limfix`
   - files missing and nothing running -> launch DETACHED (never in the foreground of your session; it outlives it):
     `setsid nohup bash TestCases/kokkos_nemo/limfix_build_and_run.sh > /dev/null 2>&1 < /dev/null &` then end the session.
   - running -> end the session (next session: step 1 again). Build FAILED -> fix the compile error (smallest change), commit, relaunch.
   - "QUEUE FINISHED" -> step 2.
2. `git add -f validation_logs/2026-11-28_4.6-PeriodicLim-fix/` (all *.log + BUILD_HEAD.txt + QUEUE_STATUS.txt), one INDEX row
   per step, commit. Any FAIL: rerun that step with SU2_KOKKOS_STAGE_VERIFY=1 (log <name>-sv-r1.log), paste <=20 StageCompare lines.
3. Review-42 B1-B6 (below) the same way: write a bnd_queue.sh copy of limfix_queue.sh (OUT=validation_logs/2026-11-28_zc2a-bndsync),
   launch detached, commit logs + INDEX rows.
4. Then ONE request with reason `MERGE: 4.6-PeriodicLim + limValid fix + zc2a-bndsync — <one-line results>` (B7), or
   `MERGE: 4.6-PeriodicLim + limValid fix — …` if only that is green.

## 2026-11-28 — review 42 (request 20261128T015904 was STALE: re-sent the pre-review-41 reason; nothing merged)
Reviewed: 9054ef3b25 (= my review-41 commit; no agent work since). Decision (a)/(b) is moot: review 41 FIXED the limoff
bug in code (limValid). Run the review-41 queue below FIRST (unchanged), then this one. Never re-send an old reason.
Dev code this round (zero-copy 2(a) step 1; g++ -fsyntax-only OK on Residual.cpp + CNEMOEulerSolver.cpp; not built/run):
- CNEMOKokkosData::SetForceMarkers (Residual.cpp) + hpp decl/NumBoundarySyncPoints: the per-iteration BOUNDARY sync
  downloads Primitive rows only at solid-wall / near-field / inlet / outlet / actdisk / engine / MARKER_ANALYZE markers
  (the only markers Pressure/Momentum/Friction_Forces, ComputeVertexTractions, SetAnalyzeSurface read). Far field,
  symmetry, supersonic outlet, periodic, send/receive rows no longer copied. CUSTOM_OUTPUTS, MARKER_PYTHON_CUSTOM or
  SU2_KOKKOS_NEMO_BND_ALL=1 keep all markers. Print: "stage boundary sync (force/analyze markers): device, N of M ...".
- check.sh EXPECT keys axi_visccone_bnd/_bndall, invwedge_lax_bnd/_bndall (RUN_TAG).
NOTE: check.sh compares residuals + restart fields, NOT force/heat coefficients -> step B3 below compares history.

Agent: do next (after review-41 queue; dir validation_logs/2026-11-28_zc2a-bndsync/, S = strict gate as before)
B1. Build strict + serial -> build-bnd-r1.log, build-bnd-serial-r1.log (compile errors: fix yourself).
B2. G1 S RUN_TAG=_bnd on axi_visccone and invwedge_lax -> g1-bnd-r1.log; controls SU2_KOKKOS_NEMO_BND_ALL=1
    RUN_TAG=_bndall same cases -> g1-bndall-r1.log. Both must PASS bitwise; record "N of M marker points" lines.
B3. Force/heat history bitwise: for both cases diff the history files (history.csv / .dat) of the HOST ref run and the
    _bnd device run, every column except time/wall-clock (CD, CL, CMz, CFx/CFy, HF, MaxHF, ...). Must be identical.
    Put the diff summary (or "identical, K columns x N rows") in g1-bnd-hist-r1.log.
B4. Ledger: [KOKKOS_NEMO_LEDGER] "sync boundary Primitive" steady-iteration bytes for _bnd vs _bndall -> paste both
    in KOKKOS_PROGRESS.md and in ledger-bnd-r1.log.
B5. Regress the 13 default cases (G1 S, no RUN_TAG) -> g1-bnd-regress-r1.log.
B6. FAIL: rerun with SU2_KOKKOS_STAGE_VERIFY=1, paste <=20 StageCompare lines. A wrong EXPECT text = fix yourself.
B7. Commit logs + INDEX rows, then ONE request: `MERGE: 4.6-PeriodicLim + limValid fix + zc2a-bndsync — <results>`
    (or two separate MERGE items if only one is green).
Madan (optional, V100): after B5, GPU timing of the
bndsync win: run axi_visccone with the CUDA build, RUN_TAG=_bnd and _bndall, archive both logs (ledger + timers) in
validation_logs/2026-11-28_zc2a-bndsync/g7-bnd-v100-r1.log.

## 2026-11-28 — review 41 (4.6-PeriodicLim NOT merged yet; limoff FAIL = real bug, FIXED in code)
Reviewed: 527782188324d0f0f2a5106d395ae1d4bc6db717. Neither (a) nor (b): the limoff control was right to fail.
Diagnosis: `limValid` (device limiter == host rows) was never cleared per iteration outside ActivateResident. With a
HOST limiter + device upwind (non-resident: PERIODIC_LIM=0, SU2_KOKKOS_NEMO_HOST=lim, unported kinds), the upwind
uploaded the full limiter once (iteration 1), then only halo rows -> stale domain limiter. Same 3.95e-02 numbers as the
4.6-Periodic g1-per-venk-r1 FAIL (wrongly filed as a check-script issue; the _sv rerun hid it because verify uploads all).
Dev code this round (not built/run; 2-line flag change, members already exist):
- kokkos/CNEMOKokkosData.cpp SetPrimVar tail: `limValid = false;` next to `gradPrimValid = gradRecValid = false;`
- include/solvers/kokkos/CNEMOKokkosData.hpp InvalidatePrimitive(): also clears gradPrimValid/gradRecValid/limValid.
Resident path unchanged (ComputeLimiter sets limValid before the upwind each iteration) -> no new per-iteration copies.

Agent: do next (dir validation_logs/2026-11-28_4.6-PeriodicLim-fix/, S = BIN=<strict> REF_MODE=HOST RES_TOL=0 FIELD_TOL=0)
1. Build strict + serial at this head -> build-limfix-r1.log, build-limfix-serial-r1.log. Compile errors: fix yourself.
2. G1 S `SU2_KOKKOS_NEMO_PERIODIC_LIM=0 RUN_TAG=_limoff` invper_lim -> g1-limfix-off-r1.log (must now PASS bitwise, no STAGE_VERIFY).
3. G1 S invper_ausm with EXTRA_CFG "MUSCL_FLOW= YES;SLOPE_LIMITER_FLOW= VENKATAKRISHNAN", SU2_KOKKOS_NEMO_PERIODIC_LIM=0,
   RUN_TAG=_limoff (add EXPECT key invper_ausm_limoff = the invper_lim_limoff string; small edit) -> g1-limfix-pervenkoff-r1.log.
4. Host-limiter control, non-periodic: pick one default check.sh case with MUSCL + BJ/Venkat limiter, run G1 S with
   SU2_KOKKOS_NEMO_HOST=lim (add EXPECT key <case>_hostlim with "stage limiter (...): host") -> g1-limfix-hostlim-r1.log.
5. Re-run G1 S invper_lim (device path) + A/B (SU2_KOKKOS_STAGE_VERIFY=1) invper_lim -> g1-limfix-r1.log, ab-limfix-r1.log.
6. Regress 13 cases (same list as g1-regress-r1) -> g1-limfix-regress-r1.log.
7. Any FAIL: rerun that case with SU2_KOKKOS_STAGE_VERIFY=1, paste <=20 StageCompare lines in KOKKOS_PROGRESS.md.
8. Commit logs + INDEX rows, push, then ONE `reason: MERGE: 4.6-PeriodicLim + limValid fix — <results>`.
   `HEAVY: ...` only for new code or a bug you cannot localize after 3 sessions. Never leave reason empty.

## 2026-11-28 01:40 — PRIORITY ORDER from Madan (overrides older queue orders; roles unchanged: Claude writes code, agent builds/runs/analyzes)
Scope: finish the **NEMO solver end-to-end on the GPU** first.
- **Later:** chemistry/gas-model work (Mutation++ models, Chapman–Enskog), and the speedup/performance study (preconditioners, kernel tuning, benchmarks), per Madan's 2026-11-27 decision.
- **Out of scope:** turbulence (NEMO has no turbulence model in SU2).

**Claude developer queue (code), in order**
1. 4.6-PeriodicLim: finish validation. If it drags on, park it (low value for hypersonics).
2. **Zero-copy steady state:**
   - (a) boundary forces and heat flux computed only on output/monitoring iterations, or integrated on the device (download scalars only);
   - (b) make the device symmetry correction exact, so the per-iteration symmetry-row staging goes away.

   Gate: `[KOKKOS_NEMO_LEDGER]` non-output iterations at scalars only, and G1 strict bitwise.
3. A GPU validation runner (`g7_all.sh`) for all merged features, for Madan to run on the V100 / GH200.

**Agent: do next**
- Unchanged: run the queue in the newest review section below (4.6-PeriodicLim gates), then request review as in ROLE.
- Do not start new items.

## 2026-11-28 — review 40 (MERGED 4.6-PeriodicDT + 4.6-PeriodicAUSMM; queue C unblocked)
Reviewed: e077c252d7 (request had an empty `reason:`; rounds 177-182 made no progress: lim_queue.sh was never written).
Merged into su2kokkos a69acea: 4.6-PeriodicDT (b0008fc) + the AUSM+M hunk of 60a160f (Device_PeriodicUnsupported),
invper_dt/invper_ausmm cfgs + check.sh/noisefloor entries, VALIDATION_LOG rows, port-plan matrix.
NOT merged: 4.6-PeriodicLim hunks of 60a160f (Data/Residual/Stages, Device_PrimitiveLimiter, Device_PeriodicNeighborRows,
invper_lim keys) - still unvalidated. Do NOT re-validate DT/AUSMM.
Dev code this round: TestCases/kokkos_nemo/lim_queue.sh (queue C1-C5 + D regress, same runner as mm_queue.sh; bash -n only).

Agent: do next
1. `bash TestCases/kokkos_nemo/lim_queue.sh` (background; resumable). Status: `cat validation_logs/2026-11-27_4.6-PeriodicLim/QUEUE_STATUS.txt`.
   It builds (build-lim-r1/-serial-r1), cpu-lim-r1, g1-lim-r1, g1-lim-bj/-wang/-off/-pervenk-r1, ab-lim-r1, poison-lim-r1,
   nf-lim-r1, g1-regress-r1 (13 cases). All logs land in that dir.
2. On any FAIL: rerun that step's case with SU2_KOKKOS_STAGE_VERIFY=1 and paste <= 20 StageCompare lines in KOKKOS_PROGRESS.md;
   check Limiter_Primitive / Solution_Max / Solution_Min first. A wrong EXPECT string (stage line text) is a small edit: fix it yourself.
3. Commit the logs + one INDEX.md row per step, push, then ONE REVIEW_REQUEST `reason: MERGE: 4.6-PeriodicLim — <results>`.
4. Compile errors: fix yourself (smallest change). `HEAVY: ...` only for new code or a bug you cannot localize after 3 sessions.
5. Never leave `reason:` empty. If idle, write `reason: HEAVY: next item` (next: periodic least-squares gradient).

## 2026-11-27 — review 39 (MERGED 4.6-PeriodicNS; Dev: 4.6-PeriodicAUSMM + 4.6-PeriodicLim)

Reviewed: 891fe7f045. MERGED 4.6-PeriodicNS into su2kokkos 3340824 (code 5a12c86 + c4f21b5, EXPECT key viscper_ausm_exp
from 690ab0f; VALIDATION_LOG row added) + port-plan matrix 844f294. 4.6-PeriodicDT (b0008fc) NOT merged: still unvalidated.
Dev code this round (syntax-checked g++ -std=c++20 -fsyntax-only, Kokkos Serial, all 4 TUs; not built/run):
- 4.6-PeriodicAUSMM: Device_PeriodicUnsupported no longer gates AUSM+M (CPU SetPressureDiffusionSensor has no periodic
  comms; PERIODIC_SENSOR is JST-only). Control SU2_KOKKOS_NEMO_PERIODIC_AUSMM=0. Case invper_ausmm (meshes/invper_ausmm.cfg).
- 4.6-PeriodicLim: device limiter (BJ / Venkat / Venkat-Wang / SHARP_EDGES / WALL_DISTANCE) with periodic markers:
  CNEMOKokkosData::LimiterPeriodicInit (fieldMax=fieldMin=field, all nPoint), KernelLimiter(periodic) skips per-point init,
  Device_PrimitiveLimiter stages PRIM_MAX/PRIM_MIN periodic rows (+ resident: Primitive rows of the send points' neighbours,
  Device_PeriodicNeighborRows) around PERIODIC_LIM_PRIM_1, and LIMITER periodic rows around PERIODIC_LIM_PRIM_2. New
  HaloQuantity PRIMITIVE / PRIM_MAX / PRIM_MIN. Control SU2_KOKKOS_NEMO_PERIODIC_LIM=0. Case invper_lim (MUSCL + VENKATAKRISHNAN).
  EXPECT key invper_ausm_pervenk now expects resident ON + "stage limiter (VENKATAKRISHNAN): device" (old OFF form obsolete).
Files: CNEMOEulerSolver.cpp/.hpp, kokkos/CNEMOKokkosData.hpp, CNEMOKokkosStages.cpp, CNEMOKokkosResidual.cpp, CNEMOKokkosData.cpp,
check.sh, check_noisefloor.sh, meshes/invper_ausmm.cfg, meshes/invper_lim.cfg.

Agent: do next (write every log DIRECTLY into validation_logs/<dir>/, git add -f + commit + push before any REVIEW_REQUEST)
A. Finish 4.6-PeriodicDT (review 38 steps 2-7, dir validation_logs/2026-11-27_4.6-PeriodicDT/) on a build of THIS head
   (it includes the DT change). Add viscper_ausm to its regression list.
B. 4.6-PeriodicAUSMM, dir validation_logs/2026-11-27_4.6-PeriodicAUSMM/ (S = BIN=<strict> REF_MODE=HOST RES_TOL=0 FIELD_TOL=0):
   1. build strict + serial -> build-mm-r1.log, build-mm-serial-r1.log (compile error: smallest fix yourself).
   2. CPU sanity ENABLE_KOKKOS=NO invper_ausmm.cfg 21 iters -> cpu-mm-r1.log.
   3. S check.sh invper_ausmm -> g1-mm-r1.log. 4. S EXTRA_CFG="TIME_DISCRE_FLOW= EULER_EXPLICIT" RUN_TAG=_exp -> g1-mm-exp-r1.log
      (EXPECT falls back to the plain key). 5. control SU2_KOKKOS_NEMO_PERIODIC_AUSMM=0 RUN_TAG=_mmoff -> g1-mm-off-r1.log.
   6. A/B SU2_KOKKOS_STAGE_VERIFY=1 RUN_TAG=_mmab -> ab-mm-r1.log; poison SU2_KOKKOS_NEMO_POISON=1 RUN_TAG=_mmpois -> poison-mm-r1.log.
   7. noise floor CASE=invper_ausmm K=10 -> nf-mm-r1.log.
C. 4.6-PeriodicLim, dir validation_logs/2026-11-27_4.6-PeriodicLim/:
   1. (same build as B) CPU sanity invper_lim.cfg -> cpu-lim-r1.log.
   2. S check.sh invper_lim -> g1-lim-r1.log. 3. variants: EXTRA_CFG="SLOPE_LIMITER_FLOW= BARTH_JESPERSEN" RUN_TAG=_bj
      -> g1-lim-bj-r1.log; EXTRA_CFG="SLOPE_LIMITER_FLOW= VENKATAKRISHNAN_WANG" RUN_TAG=_wang -> g1-lim-wang-r1.log;
      control SU2_KOKKOS_NEMO_PERIODIC_LIM=0 RUN_TAG=_limoff -> g1-lim-off-r1.log; old variant invper_ausm RUN_TAG=_pervenk with
      its old EXTRA_CFG (see the 4.6-Periodic runner script) -> g1-lim-pervenk-r1.log.
   4. A/B SU2_KOKKOS_STAGE_VERIFY=1 RUN_TAG=_limab -> ab-lim-r1.log (StageCompare "Limiter_Primitive"/"Solution_Max"/
      "Solution_Min" must be bitwise; on failure paste <= 20 StageCompare lines in KOKKOS_PROGRESS.md). Poison RUN_TAG=_limpois -> poison-lim-r1.log.
   5. noise floor CASE=invper_lim K=10 -> nf-lim-r1.log.
D. Regressions once (strict bitwise): default 8 + invper_ausm viscper_ausm invper_dt invper_ausmm invper_lim -> g1-regress-r1.log
   in validation_logs/2026-11-27_4.6-PeriodicLim/.
E. All PASS + logs committed AND pushed -> ONE REVIEW_REQUEST `reason: MERGE: 4.6-PeriodicDT + 4.6-PeriodicAUSMM + 4.6-PeriodicLim — <results>`
   (a subset is fine if one item is stuck: name exactly which items passed). Failure you cannot localize after 3 sessions ->
   `HEAVY: <item>: <stage + iter + StageCompare lines>`.
Next heavy items (reviewer): periodic + multigrid; periodic weighted least squares (PERIODIC_PRIM_LS).

## 2026-11-27 — review 38 (MERGE 4.6-PeriodicNS DEFERRED: logs not committed; Dev: 4.6-PeriodicDT)

Reviewed: 690ab0f096. Gate claims look good and code is unchanged since 5a12c86 (only check.sh/check_noisefloor.sh EXPECT/case
lines added). BUT `git ls-tree origin/su2kokkos_qwenlocal_agent validation_logs/2026-11-27_4.6-PeriodicNS/` is EMPTY: the
12 INDEX rows cite logs that are not in git (same as review 36). Rule: no merge without archived logs. Nothing merged.
Dev code this round (4.6-PeriodicDT, not built/run; same getenv pattern as PeriodicNS):
- CNEMOEulerSolver::Device_PeriodicUnsupported: dual time no longer gated (SU2_KOKKOS_NEMO_PERIODIC_DT=0 restores it).
  Dual-time source is added in Space_Integration before BC_Periodic on both paths; no periodic comms of its own.
- New case meshes/invper_dt.cfg (invper_ausm + DUAL_TIME_STEPPING-2ND_ORDER, dt 1e-6, 4 x 4 inner); check.sh case `invper_dt`
  + EXPECT keys invper_dt, invper_dt_dtoff, invper_dt_dtab, invper_dt_dt1st; check_noisefloor.sh case invper_dt (16 rows).

Agent: do next
1. FIRST archive PeriodicNS logs: copy the 12 logs + runner scripts to validation_logs/2026-11-27_4.6-PeriodicNS/ with the
   exact INDEX names; `git add -f validation_logs/2026-11-27_4.6-PeriodicNS/`; check `git ls-files validation_logs/2026-11-27_4.6-PeriodicNS | wc -l` >= 12;
   commit "Validation: archive 4.6-PeriodicNS gate logs"; `git push`; verify with `git ls-tree origin/su2kokkos_qwenlocal_agent validation_logs/2026-11-27_4.6-PeriodicNS/`.
   Lost log -> rerun that gate as -r(N+1) + INDEX row. Make the runner write logs DIRECTLY into validation_logs/<dir>/ from now on.
2. Build strict + serial -> validation_logs/2026-11-27_4.6-PeriodicDT/build-pdt-r1.log, build-pdt-serial-r1.log. Compile error -> fix (smallest change).
3. CPU sanity: strict, ENABLE_KOKKOS=NO, invper_dt.cfg -> cpu-pdt-r1.log (rc=0, 16 rows). Diverges -> TIME_STEP= 1e-7 (r2).
4. G1 strict: `BIN=<strict> REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 bash TestCases/kokkos_nemo/check.sh invper_dt` -> g1-pdt-r1.log.
5. Variants: (a) EXTRA_CFG="TIME_MARCHING= DUAL_TIME_STEPPING-1ST_ORDER" RUN_TAG=_dt1st -> g1-pdt-1st-r1.log; (b) MUSCL_FLOW= YES;SLOPE_LIMITER_FLOW= VAN_ALBADA_EDGE
   -> g1-pdt-muscl-r1.log; (c) control SU2_KOKKOS_NEMO_PERIODIC_DT=0 RUN_TAG=_dtoff -> g1-pdt-off-r1.log.
6. A/B: SU2_KOKKOS_STAGE_VERIFY=1 RUN_TAG=_dtab -> ab-pdt-r1.log (fail: <= 20 StageCompare lines in progress log). Poison -> poison-pdt-r1.log.
7. Regressions: default 8 + invper_ausm viscper_ausm, strict bitwise -> g1-pdt-regress-r1.log. Noise floor CASE=invper_dt K=10 -> nf-pdt-r1.log.
8. All PASS + logs committed AND pushed -> one REVIEW_REQUEST `reason: MERGE: 4.6-PeriodicNS + 4.6-PeriodicDT — <results>`.
   Unlocalized failure after 3 sessions -> `HEAVY: 4.6-PeriodicDT: <stage + iter + StageCompare lines>`.
Next heavy items: periodic + MG, PERIODIC_SENSOR (AUSM+M / JST), periodic limiters (BJ/Venkat).

## 2026-11-27 — review 37 (MERGED 4.6-Periodic; Dev: 4.6-PeriodicNS)

Reviewed: 1c2ed85537. Logs committed (23 files) + INDEX rows OK. MERGED into su2kokkos b8b11be (code 1c0b4f8 incl. CSolver
guard; VALIDATION_LOG row; plan matrix row). Thanks for archiving.
Dev code this round (4.6-PeriodicNS, not built/run; tiny change, same getenv pattern as SU2_KOKKOS_NEMO_MIXED — not compiled here):
- CNEMOEulerSolver::Device_PeriodicUnsupported: NS no longer gated (env SU2_KOKKOS_NEMO_PERIODIC_NS=0 restores the gate).
  NS-only periodic traffic is PERIODIC_PRIM_GG rotation of the partial GG gradient, already staged in Device_PrimitiveGradient.
- New gate case meshes/viscper_ausm.cfg (invper_ausm + NEMO_NAVIER_STOKES, isothermal 300 K Lower/Upper); check.sh case
  `viscper_ausm` + EXPECT keys `viscper_ausm`, `viscper_ausm_nsoff`.
Logs dir for everything below: validation_logs/2026-11-27_4.6-PeriodicNS/ (INDEX row + `git add -f` per log).

Agent: do next
1. Build strict + serial (`ninja -C build-kokkos-strict -j10 SU2_CFD/src/SU2_CFD > .../build-pns-r1.log 2>&1`, serial -> build-pns-serial-r1.log). Compile error -> fix yourself (smallest change).
2. CPU sanity: strict binary, ENABLE_KOKKOS=NO, viscper_ausm.cfg, 21 iters, must finish rc=0 -> cpu-pns-r1.log. If the CPU itself diverges/NaNs, set CFL_NUMBER= 1 in viscper_ausm.cfg and retry (r2); still failing -> `HEAVY: 4.6-PeriodicNS: CPU case <last 10 lines>`.
3. G1 strict: `BIN=<strict> REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 bash TestCases/kokkos_nemo/check.sh viscper_ausm > .../g1-pns-r1.log 2>&1` (OVERALL PASS, no "expected stage line missing").
4. Variants (same command + EXTRA_CFG, RUN_TAG): (a) `TIME_DISCRE_FLOW= EULER_EXPLICIT` -> g1-pns-exp-r1.log (EXPECT key falls back to viscper_ausm: the viscous stage line then reads "(CAvgGradCorrected_NEMO)"; if that line is reported missing, add EXPECT key `viscper_ausm_exp` with ")" form and RUN_TAG=_exp, rerun r2); (b) `MUSCL_FLOW= YES;SLOPE_LIMITER_FLOW= VAN_ALBADA_EDGE` -> g1-pns-muscl-r1.log; (c) control `SU2_KOKKOS_NEMO_PERIODIC_NS=0 RUN_TAG=_nsoff` -> g1-pns-nsoff-r1.log.
5. A/B: `SU2_KOKKOS_STAGE_VERIFY=1 RUN_TAG=_pnsab` (add EXPECT key `viscper_ausm_pnsab` = same as invper_ausm_perab) -> ab-pns-r1.log; on failure paste <= 20 StageCompare lines into the progress log.
6. Poison variant (as poison-per-r1) on viscper_ausm -> poison-pns-r1.log.
7. Regressions: 8-case default list + invper_ausm, strict bitwise -> g1-pns-regress-r1.log.
8. Noise floor: add `viscper_ausm) CFG=$SU2/TestCases/kokkos_nemo/meshes/viscper_ausm.cfg; HREL=0 ;;` to check_noisefloor.sh, `CASE=viscper_ausm K=10` -> nf-pns-r1.log.
9. All PASS + logs committed -> `REVIEW_REQUEST reason: MERGE: 4.6-PeriodicNS — <results>`. Failure you cannot localize in 3 sessions -> `HEAVY: 4.6-PeriodicNS: <stage + iteration + StageCompare lines>`.
Next heavy items after this: periodic + MG, periodic + dual time, PERIODIC_SENSOR (AUSM+M / JST), periodic limiters.

## 2026-11-27 — review 36 (MERGE 4.6-Periodic REJECTED for now: logs not committed; nothing merged)

Reviewed: 615f573cea. The gate claims look good (G1 strict bitwise, variants, A/B r2, poison, regress 8/8, NF r3; tested
code 7a1da15 == 1c0b4f8 in SU2_CFD/). BUT the cited logs are NOT in git: under validation_logs/2026-11-27_4.6-Periodic/
only QUEUE_STATUS.txt, build-per-*.log, cpu-per-r1*.log are committed. Missing: g1-per-r1, g1-per-exp-r1, g1-per-muscl-r1,
g1-per-lax-r1, g1-per-off-r1, g1-per-venk-r1, g1-per-venk-sv-r1, ab-per-r1, ab-per-r2, ab-per-muscl-r1, ab-per-muscl-r2,
poison-per-r1, g1-per-regress-r1, nf-per-r1, nf-per-r2, nf-per-r3 (.log). Rule: no merge without archived logs.
Dev code this round: none.

Agent: do next
1. Copy the 16 logs above (from wherever per_queue.sh / .rerun3.sh / .nfper.sh wrote them) into
   validation_logs/2026-11-27_4.6-Periodic/ with exactly the INDEX names; `git add -f validation_logs/2026-11-27_4.6-Periodic/`;
   check `git ls-files validation_logs/2026-11-27_4.6-Periodic | wc -l` >= 23; commit "Validation: archive 4.6-Periodic gate logs".
2. If any log is lost, rerun that gate (same command as its INDEX row) with suffix -r(N+1), add an INDEX row, archive it.
3. Then write REVIEW_REQUEST with `reason: MERGE: 4.6-Periodic` (same claims). Next heavy item after merge: 4.6 dynamic/deforming grids.

## 2026-11-27 — review 35 (empty request again; nothing merged; FIXED the segfault in your guard fix)

Reviewed: c252bfd0b5. cpu-per-r1 rc=139 (segfault right after "Begin Solver") is caused by the round-163 guard fix in
CSolver::InitiatePeriodicComms: `GetMarker_All_TagBound(val_periodic_index)` used the periodic PAIR index (1) as a
marker index -> tag of a non-periodic marker -> GetPeriodicRotAngles reads past Periodic_RotAngles -> crash.
Dev code this round: CSolver.cpp:348-363 — loop over all PERIODIC_BOUNDARY markers, error only if any rotation angle != 0.
Checked: g++ -fsyntax-only (no errors in CSolver.cpp). Not yet built or run. Guard is approved in this form.

Agent: do next
1. Build strict + serial: `ninja -C build-kokkos-strict -j10 SU2_CFD/src/SU2_CFD > validation_logs/2026-11-27_4.6-Periodic/build-per-r2.log 2>&1`
   (same for build-kokkos-serial -> build-per-serial-r2.log). Fix compile errors yourself (smallest change). INDEX rows + commit.
2. `rm validation_logs/2026-11-27_4.6-Periodic/.cpu-per-r1.done` and relaunch `bash TestCases/kokkos_nemo/per_queue.sh` as a
   harness background task (setsid dies in your sandbox). Then each session: steps 2-5 of review 34/33 (INDEX row + commit per END line).
3. If cpu-per-r1 fails again: paste the last 30 lines of cpu-per-r1-su2.log into the progress log; segfault -> rerun once under
   `gdb -batch -ex run -ex bt --args <strict binary> run.cfg` in the run dir, save as gdb-per-r1.log, then `HEAVY: 4.6-Periodic: CPU sanity <bt top 3 frames>`.
4. Your request MUST have a reason line: `READY FOR REVIEW: MERGE: 4.6-Periodic — <results>` (include CSolver.cpp guard in the merge).

## 2026-11-27 — review 34 (empty request; nothing merged; FIX: your memory files were too big for your context)

Reviewed: e01ea89d2e. Rounds 151-158 all ended rc=1 with no progress and the request had no reason line. Diagnosis:
KOKKOS_PROGRESS.md had grown to 161 KB and REVIEW_NOTES.md to 97 KB; reading them at session start filled your context
before any work. Fixed (history kept, nothing deleted): old log entries -> KOKKOS_PROGRESS_ARCHIVE.md, old review
sections -> REVIEW_NOTES_ARCHIVE.md. LOCAL_AGENT_TASK.md section 0: read `head -30` + `tail -40` of the progress file and
`head -45` of this file only; never read the archives whole (grep them). Keep progress entries ≤ 5 lines.
No new solver code on purpose: 4.6-Periodic (6dcd2bf) is still untested. Madan: nothing to run.

Agent: do next — the review-33 queue below (unchanged), starting NOW with step 1 (it takes 2 tool calls):
1. `pgrep -f per_queue.sh || setsid nohup bash TestCases/kokkos_nemo/per_queue.sh > /dev/null 2>&1 < /dev/null &`
   then `cat validation_logs/2026-11-27_4.6-Periodic/QUEUE_STATUS.txt`; append a 2-line progress entry, commit, end session.
2. Next sessions: steps 2-5 of review 33 (INDEX row + commit per END line; `git add -f` the logs; do not commit
   `.queue.lock`). Request `READY FOR REVIEW: MERGE: 4.6-Periodic — <results>` only when all steps PASS and are committed;
   `READY FOR REVIEW: HEAVY: 4.6-Periodic: <stage + iteration>` on a failure you cannot localize in 3 sessions.

## 2026-11-27 — review 33 (empty request again; nothing merged; validation runner for 4.6-Periodic)

Reviewed: 349539bd1b. Empty `reason:` again; rounds 143-150 made no progress (step 2 CPU sanity never ran to the end:
the runs die with the session). Nothing to merge. Dev this round (no solver code): `TestCases/kokkos_nemo/per_queue.sh`
— a DETACHED, resumable runner for review-31 steps 2-7 that writes every log directly into
validation_logs/2026-11-27_4.6-Periodic/ and one START/END line per step into QUEUE_STATUS.txt there (skips steps with a
`.<name>.done` marker; flock prevents double launch). check.sh: EXPECT keys for `invper_ausm_peroff` / `invper_ausm_pervenk`
(expect `periodic comms ...: host` + `resident ... OFF (`). Smoke-tested with stub binaries here; not run on SU2.

Agent: do next (each session is SHORT — do only these mechanical steps, no planning):
1. If `pgrep -f per_queue.sh` prints nothing and QUEUE_STATUS.txt has no `QUEUE FINISHED`/`STOP` line, launch it and end the session:
   `setsid nohup bash TestCases/kokkos_nemo/per_queue.sh > /dev/null 2>&1 < /dev/null &`
2. Every session: `cat validation_logs/2026-11-27_4.6-Periodic/QUEUE_STATUS.txt`. For each new END line add ONE INDEX row
   (gate = step name, result = the `::` summary, log = validation_logs/2026-11-27_4.6-Periodic/<name>.log), then
   `git add validation_logs && git commit -m "Validation: 4.6-Periodic <steps>"`. That counts as progress.
3. On `STOP: CPU sanity failed`: read the last 30 lines of cpu-per-r1-su2.log. If it is a periodic matching error, change
   2.0 -> -2.0 in meshes/invper_ausm.cfg, commit, `rm validation_logs/2026-11-27_4.6-Periodic/.cpu-per-r1.done`, relaunch (step 1).
   Any other error: `HEAVY: 4.6-Periodic: CPU sanity <first error line>`.
4. On a FAIL in g1-* (no `OVERALL PASS`, or `expected stage line missing`): wait for ab-per-r1 to finish, paste <= 20
   StageCompare lines with non-bitwise > 0 from ab-per-r1.log into the progress log, note the first differing iteration;
   if you cannot localize it in 3 sessions: `HEAVY: 4.6-Periodic: <stage + iteration>`.
5. When QUEUE FINISHED and g1/variants/ab/poison/regress are all OVERALL PASS and committed: `MERGE: 4.6-Periodic`
   (nf-per-r1 must be NOISEFLOOR PASS too). A request ALWAYS has a reason line; never file an empty one.

