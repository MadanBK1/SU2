# Review notes for the local agent (written by the Claude reviewer; agent: read, never edit)

Newest section first. Do the **Agent: do next** items (tests/analysis), then write READY FOR REVIEW (see ROLE in LOCAL_AGENT_TASK.md).

## 2026-11-27 — review 32 (empty request; nothing merged; no new dev code)

Reviewed: e5d6840abe. REVIEW_REQUEST had an EMPTY `reason:` line and no new results (rounds 138-142: progress-log lines
only; 4.6-Periodic step 2 not archived). Nothing to merge. No new dev code on purpose: 6dcd2bf (4.6-Periodic step 1) is
still untested, and stacking more periodic code (periodic limiter / LS) on it before G1 passes would hide the cause of a
failure. A request needs a reason line (`MERGE: <item>` or `HEAVY: <item>: <what>`); do not file one without it.

Agent: do next — the review-31 queue, unchanged. Concretely, in THIS session (foreground, it fits the session):
1. `bash TestCases/kokkos_nemo/.cpuper.sh > ../kk/logs/cpu-per-r1.log 2>&1` (900 s timeout inside; output goes to
   stdout, so redirect it). Then copy cpu-per-r1.log + kk/runs/cpu_per_r1/log.txt (as cpu-per-r1-su2.log) to
   validation_logs/2026-11-27_4.6-Periodic/, add the INDEX row, commit. Pass = rc=0, 21 residual rows, no
   "periodic"/"matching" error. If it stops in periodic preprocessing: flip 2.0 -> -2.0 in invper_ausm.cfg, rerun as r2, note it.
2. Then steps 3-7 of review 31 (G1 `.g1per.sh` -> g1-per-r1.log first). On FAIL: STAGE_VERIFY=1 rerun, <= 20 StageCompare
   lines, first differing iteration; `HEAVY: 4.6-Periodic: <bug>` after 3 sessions without a localization.
3. Request `MERGE: 4.6-Periodic` only after steps 1-6 pass and their logs are committed.

## 2026-11-27 — review 31 (4.6-Limiters + ResSplit MERGED; new dev code 4.6-Periodic step 1)

Reviewed: 1e7f5b2ebb. Merged into su2kokkos as fc5c06e (+ 622b019 plan wording): code 332d230 (Limiters) + c2ba950
(ResSplit) — CNEMOKokkosData.hpp, meson.build, CNEMOEulerSolver.cpp, kokkos/CNEMOKokkosResidual*.{cpp,hpp,inl},
CNEMOKokkosStages.cpp, check.sh — + VALIDATION_LOG row + plan matrix. Do not re-request either. Limiter step 6
(noise floor, .nflim.sh -> nf-lim-r1.log) is still owed: run it once and archive (no merge request needed).

Dev code this round (6dcd2bf; g++ -fsyntax-only clean, NOT built, NOT validated) — 4.6-Periodic step 1: device-resident
NEMO with periodic markers. The host periodic comms run unchanged on staged device rows of the periodic points:
BC_Periodic override (PERIODIC_RESIDUAL: residual + Delta_Time + Jacobian block rows, `AccumulatorRows`), SetMax_Eigenvalue
(Lambda rows re-uploaded after PERIODIC_MAX_EIG), GG gradient (PERIODIC_PRIM_GG), CompleteImplicitIteration
(PERIODIC_IMPLICIT: Solution + Solution_Old). Scope gate `Device_PeriodicUnsupported`: inviscid, no MG, no AUSM+M,
steady / time stepping. New gate case `invper_ausm` (TestCases/kokkos_nemo/meshes/: perchannel.su2, make_perchannel.py,
invper_ausm.cfg; check.sh links that mesh dir). Startup line `periodic comms (4.6-Periodic): device (staged periodic rows)`.
Common change: CFVMFlowSolverBase::BC_Periodic `final` -> `override` (non-NEMO solvers unchanged).

Agent: do next (logs under validation_logs/2026-11-27_4.6-Periodic/ + one INDEX row each; strict = REF_MODE=HOST RES_TOL=0 FIELD_TOL=0):
1. Build strict + serial -> build-per-r1.log, build-per-serial-r1.log. Fix compile errors yourself (smallest change).
2. Sanity: run invper_ausm once with ENABLE_KOKKOS= NO (plain CPU, 21 iters) -> cpu-per-r1.log. If SU2 stops in the
   periodic preprocessing (no matching vertices), flip the translation sign in invper_ausm.cfg (2.0 -> -2.0), rerun, note it.
3. G1 strict `check.sh invper_ausm` -> g1-per-r1.log (bitwise; EXPECT line must be found).
4. Variants, strict, one log each (RUN_TAG per variant):
   a. explicit: EXTRA_CFG="TIME_DISCRE_FLOW= EULER_EXPLICIT;CFL_NUMBER= 0.5" RUN_TAG=_perexp -> g1-per-exp-r1.log
   b. MUSCL GG: EXTRA_CFG="MUSCL_FLOW= YES;SLOPE_LIMITER_FLOW= VAN_ALBADA_EDGE" RUN_TAG=_permuscl -> g1-per-muscl-r1.log
   c. LAX: EXTRA_CFG="CONV_NUM_METHOD_FLOW= LAX-FRIEDRICH" RUN_TAG=_perlax -> g1-per-lax-r1.log
   d. resident OFF control: SU2_KOKKOS_NEMO_RESIDENT=0 RUN_TAG=_peroff (EXPECT not applicable; must still be bitwise) -> g1-per-off-r1.log
   e. Venkatakrishnan (expected resident OFF "limiter: periodic", still bitwise): EXTRA_CFG="MUSCL_FLOW= YES;SLOPE_LIMITER_FLOW= VENKATAKRISHNAN" RUN_TAG=_pervenk -> g1-per-venk-r1.log
5. A/B: SU2_KOKKOS_STAGE_VERIFY=1 on invper_ausm and variant b -> ab-per-r1.log, ab-per-muscl-r1.log (paste the StageCompare
   lines with non-bitwise > 0, if any). Poison: SU2_KOKKOS_NEMO_POISON=1 on invper_ausm -> poison-per-r1.log (bitwise).
6. Regressions strict: default list + invwedge_lusgs invwedge_mg invwedge_dyn supin_visc -> g1-per-regress-r1.log.
7. Noise floor (production build): `check_noisefloor.sh` with CASE=invper_ausm K=10 -> nf-per-r1.log.
On any FAIL: rerun that case with SU2_KOKKOS_STAGE_VERIFY=1, paste <= 20 StageCompare lines; check whether the first
differing iteration is 1 (setup) or later. Request `MERGE: 4.6-Periodic` only after 1-6 pass and logs are committed;
`HEAVY: 4.6-Periodic: <bug>` only if you cannot localize a failure after 3 sessions.

## 2026-11-27 — dev (Madan request): split CNEMOKokkosResidual.cpp for nvcc compile time

On the GH200 (CUDA 12.4, sm_90) `cicc` spent > 40 min on CNEMOKokkosResidual.cpp. Dev commit (this push; g++ -fsyntax-only
clean on all 5 TUs + object/nm check that all 12 kernel instantiations resolve; NOT built with nvcc, NOT validated):
the source / BC kernel templates moved VERBATIM (line-set identical to the old file) into `CNEMOKokkosResidualKernels.inl`
(namespace nemo_kokkos_res, external linkage), ResArgs + helpers + declarations into `CNEMOKokkosResidualKernels.hpp`,
explicit instantiations in CNEMOKokkosResidualWall.cpp (isothermal/heat-flux/Smoluchowski/symmetry), ...Source.cpp,
...BCFlux2D.cpp, ...BCFlux3D.cpp (meson.build updated). No physics/order change -> strict gate must stay bitwise.

Agent: do next (logs under validation_logs/2026-11-27_ResSplit/ + INDEX rows), BEFORE continuing 4.6-Limiters:
1. Build strict + serial -> build-split-r1.log, build-split-serial-r1.log (meson reconfigures by itself). Fix compile errors yourself.
2. G1 strict default list + invwedge_lusgs invwedge_mg invwedge_dyn supin_visc -> g1-split-regress-r1.log (bitwise).
3. CUDA build on the V100: `time ninja -C build-kokkos-cuda -j10 SU2_CFD/src/SU2_CFD` -> build-split-cuda-r1.log (record the time).
Request `MERGE: ResSplit` after 1-3 pass and logs are committed (may be combined with 4.6-Limiters).

## 2026-11-27 — review 30 (stale request; nothing merged; no new dev code)

Reviewed: bbacf34b2a. The request repeated `MERGE: 4.6-Linelet`, already merged (su2kokkos ffbb893, review 29); it was
filed 12 min after review 29's REVIEW_DONE because the old READY line was still in the last 3 lines of KOKKOS_PROGRESS.md.
The reviewer appended 3 progress lines so it no longer is. If this repeats, restart the loop process (review-26 guard).

Agent: do next — unchanged: the review-29 queue, 4.6-Limiters steps 1-6 (logs under validation_logs/2026-11-27_4.6-Limiters/,
one INDEX row each). Request `MERGE: 4.6-Limiters` only after 1-6 pass and logs are committed.

## 2026-11-27 — review 29 (4.6-Linelet MERGED; new dev code 4.6-Limiters)

Reviewed: 34636cb741. Merged into su2kokkos as ffbb893 (code bb0ab70: CPreconditioner.hpp, CSysMatrix.hpp/.cpp,
CSysMatrixKokkos.cpp, CNEMOEulerSolver.cpp, check.sh, check_noisefloor.sh EXTRA_CFG) + VALIDATION_LOG row + plan matrix.
Do not re-request 4.6-Linelet. V100 G7 for LINELET still OPEN (Madan).

Dev code this round (332d230; g++ -fsyntax-only clean, NOT built, NOT validated) — device SHARP_EDGES / WALL_DISTANCE
slope limiters: `KernelLimiter` multiplies by a per-point factor `st.LimiterGeo` (CNEMOKokkosStages.cpp), uploaded once by
`SetLimiterGeoFactor` from the CPU `CLimiterDetails<...>::geometricFactor` (CNEMOEulerSolver::Device_PrimitiveLimiter);
`Device_LimiterUnsupported` accepts both kinds (dynamic grid -> host). check.sh EXPECT keys `invwedge_ausm_sharp`,
`super_cat_walld`. Note: laminar NEMO has no wall distance (numerical limit) -> WALL_DISTANCE factor is 1 there; the
SHARP_EDGES case (wedge corner) is the real test.

Agent: do next (archive every log under validation_logs/2026-11-27_4.6-Limiters/ + one INDEX row each)
1. Build strict + serial (ninja -j10 SU2_CFD) -> build-lim-r1.log, build-lim-serial-r1.log. Fix compile errors yourself.
2. G1 SHARP_EDGES: `EXTRA_CFG="SLOPE_LIMITER_FLOW= SHARP_EDGES" RUN_TAG=_sharp REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 timeout 1500
   bash TestCases/kokkos_nemo/.g1run.sh ../kk/logs/g1-lim-sharp-r1.log invwedge_ausm` -> OVERALL PASS bitwise + EXPECT lines.
   Also grep the CPU log for a limiter that differs from the VENKATAKRISHNAN run (sanity: factor < 1 near the corner) —
   if the residual history is identical to the default invwedge_ausm run, note it (factor may be 1 everywhere).
3. G1 WALL_DISTANCE viscous: `EXTRA_CFG="MUSCL_FLOW= YES;SLOPE_LIMITER_FLOW= WALL_DISTANCE" RUN_TAG=_walld ... super_cat`
   -> g1-lim-walld-r1.log.
4. A/B: step 2 and 3 with `SU2_KOKKOS_STAGE_VERIFY=1` -> ab-lim-sharp-r1.log, ab-lim-walld-r1.log; need the
   Limiter_Primitive / Solution_Max / Solution_Min StageCompare lines bitwise (paste <= 20 lines on mismatch).
5. Regressions strict (default list + invwedge_lusgs invwedge_mg invwedge_dyn supin_visc) -> g1-lim-regress-r1.log.
6. Noise floor: `CASE=invwedge_ausm EXTRA_CFG="SLOPE_LIMITER_FLOW= SHARP_EDGES" K=10 check_noisefloor.sh` -> nf-lim-r1.log (<= 10).
Request `MERGE: 4.6-Limiters` only after 1-6 pass and logs are committed; `HEAVY: ...` only for new code or a bug you
cannot localize after 3 sessions.

## 2026-11-26 — review 28 (stale request; nothing merged; no new dev code)

Reviewed: 42ae7b9275. The request repeated `READY FOR REVIEW: MERGE: 4.8-LUSGSHyb`, which is ALREADY merged (su2kokkos
f2a9c23, review 27). Only new work since 7559936: the 4.6-Linelet step-1 builds (strict + serial rc=0, archived, 2 INDEX
rows — good). Cause: round 121 wrote no progress entry, so the old READY line is still in the last 3 lines of
KOKKOS_PROGRESS.md, and the running loop process still uses the pre-fix script (bash parses the while loop once; the
review-26 guard is in the file but not active until the loop is restarted).

Agent: do next
0. Append a dated progress entry now (round 122: "4.6-Linelet step 1 done; 4.8-LUSGSHyb merged, not re-requested"), so
   the READY line leaves the tail. Restart `local_agent_loop.sh` at the next round boundary so the guard is active.
1. Continue review 27 "Agent: do next" steps 2-7 for 4.6-Linelet (G1 viscous/inviscid LINELET, A/B PRECOND_VERIFY,
   regressions, noise floor, perf lines) with the log names given there, archived under
   validation_logs/2026-11-26_4.6-Linelet/ + one INDEX row each.
2. Request `MERGE: 4.6-Linelet` only after steps 1-6 pass and the logs are committed. Never re-request an item a review
   note says is merged.

## 2026-11-26 — review 27 (4.8-LUSGSHyb MERGED; new dev code 4.6-Linelet)

Reviewed: ca9282c995. Merged into su2kokkos as f2a9c23 (code 7932194a9a: CSysMatrix.hpp, CSysMatrixKokkos.cpp, check.sh)
+ VALIDATION_LOG row + plan matrix. Do not re-request 4.8-LUSGSHyb. V100 G7 (LU_SGS hybrid vs single) still OPEN (Madan).

Dev code this round (bb0ab70; g++ -fsyntax-only -ffp-contract=off -Wall, Kokkos Serial headers, with/without -DHAVE_KOKKOS:
clean; NOT built, NOT validated) — queue item 7, LINELET preconditioner on the device:
- CSysMatrixKokkos.cpp: `EnsureKokkosLineletStructure` (pattern-only, once: Jacobi points, linelet CSR, d_matrix offsets of
  the lower/upper blocks, flattened working memory; prints `[KOKKOS] LINELET device apply: linelets=...`) and
  `KokkosComputeLineletPreconditioner` (Jacobi kernel on NO_LINELET points with d_invM + one thread per linelet, block
  Thomas exactly as ComputeLineletPreconditioner; halo exchange; A/B with SU2_KOKKOS_PRECOND_VERIFY=1).
- CPreconditioner.hpp: CLineletPreconditioner device branch (only when matrix values are device-authoritative, like LU_SGS).
- CSysMatrix.hpp/.cpp: device members + getters + frees. CNEMOEulerSolver.cpp: resident gate accepts LINELET; startup
  line `stage linear preconditioner (LINELET apply): device`; perf/verify lines `device LINELET applies=` /
  `LINELET device apply vs host apply: compares=N non-bitwise=M`.
- check.sh EXPECT keys: `super_cat_linelet`, `invwedge_lusgs_linelet` (use RUN_TAG=_linelet).

Agent: do next (archive every log under validation_logs/2026-11-26_4.6-Linelet/ + one INDEX row each)
1. Build strict + serial (ninja SU2_CFD in build-kokkos-strict and build-kokkos-serial) -> build-linelet-r1.log,
   build-linelet-serial-r1.log. Fix compile errors yourself (smallest change, note it in KOKKOS_PROGRESS.md).
2. G1 viscous: `EXTRA_CFG="LINEAR_SOLVER_PREC= LINELET" RUN_TAG=_linelet REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 timeout 1500
   bash TestCases/kokkos_nemo/.g1run.sh ../kk/logs/g1-linelet-visc-r1.log super_cat` -> OVERALL PASS bitwise, EXPECT lines found.
3. G1 inviscid: same with `invwedge_lusgs` -> g1-linelet-inv-r1.log.
4. A/B: step 2 with `SU2_KOKKOS_PRECOND_VERIFY=1 RUN_TAG=_linelet` -> ab-linelet-r1.log; need `compares>0 non-bitwise=0`
   for both the LINELET line and the preconditioner build line.
5. Regressions (default list + invwedge_lusgs invwedge_mg invwedge_dyn supin_visc), strict -> g1-linelet-regress-r1.log.
6. Noise floor: `CASE=super_cat EXTRA_CFG="LINEAR_SOLVER_PREC= LINELET" K=10 check_noisefloor.sh` -> nf-linelet-r1.log (<= 10).
7. Perf lines (informational): grep `device LINELET applies=` and `LINELET device apply: linelets=` from steps 2-3 into
   perf-linelet-r1.log.
On a G1 FAIL: rerun with SU2_KOKKOS_STAGE_VERIFY=1 (+ SU2_KOKKOS_PRECOND_VERIFY=1), paste <= 20 StageCompare/VERIFY lines.
If EXTRA_CFG does not override LINEAR_SOLVER_PREC in invwedge_lusgs.cfg, make a copy invwedge_linelet.cfg (small edit) and
add a check.sh case + EXPECT key `invwedge_linelet` with the same stage lines.
Request `MERGE: 4.6-Linelet` only after 1-6 pass and the logs are committed; `HEAVY: ...` only for new code or a bug you
cannot localize after 3 sessions.

Madan: V100 G7 (when convenient) — LINELET apply on GPU: build-kokkos-cuda, then
`EXTRA_CFG="LINEAR_SOLVER_PREC= LINELET" RUN_TAG=_linelet bash TestCases/kokkos_nemo/check.sh super_cat` and grep
`device LINELET applies=` vs `device LU_SGS applies=` / ILU; archive under validation_logs/2026-11-26_4.6-Linelet/gpu/.

## 2026-11-26 — review 26 (stale request; nothing merged; no new dev code)

Reviewed: 5ca315fe41. The request repeated the review-24/25 line `READY FOR REVIEW: MERGE: 4.8-ILUTeam`, but 4.8-ILUTeam
is ALREADY merged (su2kokkos 3897d1a). The only new work since 3468ddec5a is the 4.8-LUSGSHyb step-1 build; steps 2-9
have not run, so there is nothing to merge. Cause: the loop re-requested because the old READY line was still in the last
3 lines of KOKKOS_PROGRESS.md. Fixed in `TestCases/kokkos_nemo/local_agent_loop.sh` (a READY line equal to the previous
REVIEW_REQUEST `reason:` is ignored; periodic requests unchanged) — takes effect when the loop script is restarted.

Agent: do next
0. Append a dated progress entry for round 113 now (so the old READY line leaves the tail), then restart the loop
   script when convenient so the fix is active.
1. Continue review 25 "Agent: do next" steps 2-9 for 4.8-LUSGSHyb (G1 hybrid, WIDE=1, single, A/B, viscous WIDE=1,
   regressions, noise floor, perf lines) with the log names given there, archived under
   validation_logs/2026-11-26_4.8-LUSGSHyb/ + INDEX rows; also archive the step-1 build logs (build-lusgs-r1.log,
   build-lusgs-serial-r1.log) if not yet committed.
2. Request `MERGE: 4.8-LUSGSHyb` only after steps 1-8 pass and the logs are committed. Never re-request an item that a
   review note says is merged.

Madan: V100 G7 commands unchanged from review 25 (ILU team vs row; LU_SGS hybrid vs single, WIDE=64/1024).

## 2026-11-26 — review 25 (4.8-ILUTeam MERGED; new dev code 4.8 LU_SGS hybrid apply)

Reviewed: 3468ddec5a. Merged into su2kokkos as 3897d1a (code c362c12d78: CSysMatrixKokkos.cpp KokkosILU0BuildTeam,
check.sh EXPECT keys) + VALIDATION_LOG row + plan matrix. V100 G7 team-vs-row timing still OPEN (Madan).

Dev code this round (g++ -std=c++20 -fsyntax-only -ffp-contract=off -Wall, Kokkos Serial headers: clean; NOT built,
NOT validated) — 4.8 perf, G7 hot spot "LU_SGS device apply ~0.1 s/apply (level-scheduled single team)":
- `Common/src/linear_algebra/CSysMatrixKokkos.cpp`: row bodies moved verbatim into `DeviceLUSGSForwardRow` /
  `DeviceLUSGSBackwardRow` (same statements); `SegmentLevels` splits the levels: a level with >= W rows gets its own
  RangePolicy launch over the whole device, runs of narrower levels stay in one single-team launch (old kernel, level
  sub-range). Result is independent of segmentation (rows of a level independent, same per-row code) -> must be bitwise.
  Env: `SU2_KOKKOS_LUSGS_APPLY=single` = old apply; `SU2_KOKKOS_LUSGS_WIDE=<rows>` (default 256). Startup line
  `[KOKKOS] LU_SGS device apply: hybrid (wide >= W rows; fwd levels=.. wide=.. segments=..; bwd ...)` or `single-team`.
- `CSysMatrix.hpp`: host vectors lusgs_fwd/bwd_ptr_h, lusgs_fwd/bwd_seg.
- check.sh EXPECT keys `invwedge_lusgs_lusgshyb`, `invwedge_lusgs_lusgsw1`, `invwedge_lusgs_lusgssingle`.

Agent: do next (archive under validation_logs/2026-11-26_4.8-LUSGSHyb/ + one INDEX row each):
1. Build strict + serial -> build-lusgs-r1.log, build-lusgs-serial-r1.log. Fix compile errors yourself (smallest change).
2. G1 default: `REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_lusgshyb bash TestCases/kokkos_nemo/check.sh invwedge_lusgs` -> g1-lusgs-hyb-r1.log (PASS bitwise + hybrid line; paste the startup line in INDEX).
3. All-wide path (forces per-level launches on every level, exercises the new kernels on Serial):
   `SU2_KOKKOS_LUSGS_WIDE=1 REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_lusgsw1 bash TestCases/kokkos_nemo/check.sh invwedge_lusgs` -> g1-lusgs-w1-r1.log.
4. Old path: `SU2_KOKKOS_LUSGS_APPLY=single REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_lusgssingle bash TestCases/kokkos_nemo/check.sh invwedge_lusgs` -> g1-lusgs-single-r1.log.
5. A/B: step 3 again with `SU2_KOKKOS_PRECOND_VERIFY=1 RUN_TAG=_lusgsw1v` (EXPECT falls back to the plain key) -> grep `KOKKOS_NEMO_VERIFY` for the LU_SGS apply compares, must be non-bitwise=0 -> ab-lusgs-w1-r1.log.
6. Viscous LU_SGS: repeat the super_cat LU_SGS run you used for the 2026-11-25 LU-SGS validation, with SU2_KOKKOS_LUSGS_WIDE=1 (strict) -> g1-lusgs-visc-w1-r1.log.
7. Regressions strict: default list + invwedge_mg invwedge_dyn supin_visc -> g1-lusgs-regress-r1.log.
8. Noise floor serial+strict, invwedge_lusgs K=10 -> nf-lusgs-r1.log (ratio <= 10).
9. Record `[KOKKOS_NEMO_PERF] device LU_SGS applies=` lines for steps 2/3/4 in INDEX (Serial timing informational).
On failure: rerun the failing step with SU2_KOKKOS_STAGE_VERIFY=1 and SU2_KOKKOS_PRECOND_VERIFY=1, paste <= 20 StageCompare/VERIFY lines in KOKKOS_PROGRESS.md.
Request `MERGE: 4.8-LUSGSHyb` only after 1-8 pass and the logs are committed; `HEAVY: ...` only for new code or a bug you
cannot localize after 3 sessions.

Madan (V100, after the agent builds the CUDA binary): G7 timings, archive under validation_logs/2026-11-26_4.8-GPU/:
  `RUN_TAG=_g7iluteam bash TestCases/kokkos_nemo/check.sh super_cat` vs `SU2_KOKKOS_ILU_BUILD=row RUN_TAG=_g7ilurow ...` (grep `device preconditioner builds=`),
  `RUN_TAG=_g7lusgshyb bash TestCases/kokkos_nemo/check.sh invwedge_lusgs` vs `SU2_KOKKOS_LUSGS_APPLY=single RUN_TAG=_g7lusgssingle ...`,
  and `SU2_KOKKOS_LUSGS_WIDE=64` / `=1024` variants (grep `device LU_SGS applies=` and the hybrid startup line).

## 2026-11-26 — review 24 (6b.b-TwallIn PARTIAL merge; new dev code 4.8 ILU(0) build perf)

Reviewed: d02d49356a. Merged into su2kokkos as eafe69f (code 89112d6457: CNEMOKokkosData.hpp, CNEMOEulerSolver.cpp,
CNEMONSSolver.cpp, CNEMOKokkosResidual.cpp, check.sh, pycustom_twall.py) + VALIDATION_LOG row + plan matrix, as PARTIAL:
steps 1-4 bitwise; steps 5-6 (pysu2) still OPEN until swig is installed (Madan). Do not re-request 6b.b until 5-6 have run.
Heat flux OUT needs no new code: Friction_Forces runs after SyncHostData(BOUNDARY) (marker data only) — step 5's
wall_heatflux.csv diff checks it.

Dev code this round (g++ -fsyntax-only Serial headers; NOT built, NOT validated) — 4.8 perf, G7 hot spot "ILU build 0.29 s/build"
(super_cat V100: 3.19 s of 6.49 s wall are the 11 ILU builds):
- `Common/src/linear_algebra/CSysMatrixKokkos.cpp` `KokkosBuildILUPreconditioner`: new kernel `SU2::KokkosILU0BuildTeam`,
  one team per row of a level, block entries spread over the team (gemm entry-wise, same statement order -> bitwise on Serial),
  diagonal inverse in `Kokkos::single`. Default ON; `SU2_KOKKOS_ILU_BUILD=row` = old per-row kernel. Startup line
  `[KOKKOS] ILU(0) device build: team-per-row|row-per-thread`.
- check.sh EXPECT keys `super_cat_iluteam`, `super_cat_ilurow`.

Agent: do next (archive under validation_logs/2026-11-26_4.8-ILUTeam/ + one INDEX row each):
1. Build strict + serial -> build-ilu-r1.log, build-ilu-serial-r1.log. Fix compile errors yourself (smallest change).
2. G1: `REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_iluteam bash TestCases/kokkos_nemo/check.sh super_cat` -> g1-ilu-team-r1.log (PASS bitwise + team-per-row line).
3. A/B: same with `SU2_KOKKOS_PRECOND_VERIFY=1 RUN_TAG=_iluteamv` -> must report `compares=11 ... non-bitwise=0` (grep KOKKOS_NEMO_VERIFY) -> ab-ilu-team-r1.log.
4. Old path still works: `SU2_KOKKOS_ILU_BUILD=row REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_ilurow bash TestCases/kokkos_nemo/check.sh super_cat` -> g1-ilu-row-r1.log.
5. Regressions strict: default list + invwedge_mg invwedge_dyn supin_visc -> g1-ilu-regress-r1.log (all PASS bitwise).
6. Noise floor serial+strict on super_cat (your nf script, K=10) -> nf-ilu-r1.log.
7. Record the `[KOKKOS_NEMO_PERF] device preconditioner builds=` line of steps 2 and 4 (serial timing, informational) in INDEX.
On FAIL: rerun step 3 and paste <= 20 verify/StageCompare lines. Then request `MERGE: 4.8-ILUTeam`.
Still pending (6b.b steps 5-6, after Madan installs swig): run them exactly as in review 23, archive under 2026-11-26_6b.b-TwallIn/, then `MERGE: 6b.b-TwallIn-py`.
`HEAVY: …` only for new code or a bug you cannot localize after 3 sessions. Always fill `reason:`.

Madan (V100, after the agent's step 2 passes): G7 timing of super_cat, team vs row build:
  `SU2_KOKKOS_ILU_BUILD=row bash TestCases/kokkos_nemo/check.sh super_cat` and plain `bash TestCases/kokkos_nemo/check.sh super_cat` with the cuda binary;
  copy both chk_super_cat_*YES.log to validation_logs/2026-11-26_G7-V100-ILUTeam/ (compare `device preconditioner builds=` seconds).
  swig for 6b.b: `sudo apt-get install swig` (or `pip install --user swig`), rebuild with `-Denable-pywrapper=true`.

## 2026-11-26 — review 23 (6b.a-DynGrid MERGED; new dev code 6b.b wall-temperature coupling IN)

Reviewed: 4e49fe63ae (empty `reason:` again — treated as MERGE: 6b.a-DynGrid; ALWAYS fill `reason:`).
Merged into su2kokkos as 8af70e6 (code bf2ff4ad8f: CNEMOEulerSolver.hpp/.cpp, CNEMOKokkosData.hpp/.cpp, CNEMOKokkosResidual.cpp,
CNEMOKokkosStages.cpp, check.sh, check_noisefloor.sh, invwedge_dyn.cfg) + VALIDATION_LOG row + plan matrix. 6b.a is DONE — never re-request.

Dev code this round (89112d6; g++ -fsyntax-only with/without -DHAVE_KOKKOS; NOT built, NOT validated) — queue item 6b.b (IN part):
- `CNEMONSSolver::BC_IsothermalNonCatalytic_Wall` / partially-catalytic part of `BC_IsothermalCatalytic_Wall`: with
  MARKER_PYTHON_CUSTOM the wall T is `geometry->GetCustomBoundaryTemperature(marker, iVertex)` (default = MARKER_ISOTHERMAL value).
- Device: `BCParams::customTwall`, `CNEMOKokkosData::customTwall_` (only the marker's nVertex doubles copied per BC call),
  `KernelIsothermal` reads `r.twallVtx(kv)`. Stage line gets ` [py-custom Twall]`.
- check.sh: env `PYDRIVER=<script>` runs both runs through pysu2; EXPECT key `super_cat_pycustom`.
- New `TestCases/kokkos_nemo/pycustom_twall.py`: sets T(x)=T0+DT*ramp each iteration, writes wall_heatflux.csv (heat flux OUT).

Agent: do next (archive every log under validation_logs/2026-11-26_6b.b-TwallIn/ + one INDEX row each):
1. Build strict + serial (your build scripts) -> build-tw-r1.log, build-tw-serial-r1.log. Fix compile errors yourself (smallest change).
2. Identity, SU2_CFD binary: `REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_pycustom EXTRA_CFG="MARKER_PYTHON_CUSTOM= ( Wall )" bash TestCases/kokkos_nemo/check.sh super_cat` -> g1-tw-ident-r1.log, PASS bitwise + `[py-custom Twall]: device` line.
3. Regression: default list + supin_visc smol_maxwell (strict) -> g1-tw-regress-r1.log, all PASS bitwise.
4. CPU no-op check: compare the step-2 NO run restart with a plain super_cat NO run (strict binary, both host) -> must be bitwise identical; note in INDEX (cpu-noop-tw-r1.log).
5. pysu2 (only if your build has the python wrapper, meson `-Denable-pywrapper=true`; if not, rebuild a separate strict build dir with it, log build-tw-py-r1.log):
   `PYTHONPATH=<build>/bin REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 RUN_TAG=_pycustom EXTRA_CFG="MARKER_PYTHON_CUSTOM= ( Wall )" PYDRIVER=$PWD/TestCases/kokkos_nemo/pycustom_twall.py bash TestCases/kokkos_nemo/check.sh super_cat` -> g1-tw-py-r1.log (PASS bitwise expected).
   Then `diff` the two wall_heatflux.csv files (chk_super_cat_pycustom_NO / _YES) -> tw-heatflux-diff-r1.log. Identical = heat flux OUT already correct in resident mode; if they differ, paste the first 10 differing rows and request `HEAVY: 6b.b: heat flux OUT in resident mode`.
6. Also run step 5 with PYCUSTOM_DT=0 (uniform) and compare its NO restart to step 2 NO -> should be bitwise (g1-tw-py-dt0-r1.log).
On any G1 FAIL: rerun with SU2_KOKKOS_STAGE_VERIFY=1, paste <= 20 StageCompare lines. Request `MERGE: 6b.b-TwallIn` after 1-5 pass and logs are committed.
`HEAVY: …` only for new code or a bug you cannot localize after 3 sessions. Always fill `reason:`.

## 2026-11-26 — review 22 (6b.a dynvisc bug fixed; nothing merged)

Reviewed: 7aa0372c39 (request had an empty `reason:`; treated as HEAVY 6b.a-DynGrid dynvisc). Nothing merged (dynvisc open).

Diagnosis: `CNEMONSSolver::Preprocessing` calls `CommonPreprocessing` directly, never `CNEMOEulerSolver::Preprocessing`,
so `Device_RefreshDynamicGeometry` (placed in the Euler Preprocessing) never ran for viscous cases -> device grid velocity /
coords / normals / Volume_nM1 stayed at the constructor upload (grid velocity 0). Inviscid cases were fine. With
SU2_KOKKOS_STAGE_VERIFY=1 resident mode is off and the host results are used, which hid it.

Dev code this round: `CNEMOEulerSolver.cpp` — refresh call moved from `Preprocessing` to the top of `CommonPreprocessing`
(same guard `dynamic_grid && iMesh == MESH_0`; once per (time, outer) iter as before). No other change. Not built.

Agent: do next (archive every log under validation_logs/2026-11-26_6b.a-DynGrid/ + INDEX rows; then `MERGE: 6b.a-DynGrid`):
1. Build strict + serial: `bash .builddyn.sh` -> build-dyn-r2.log, build-dyn-serial-r2.log. Fix compile errors yourself (smallest change).
2. G1 dynvisc: `bash .g1dynvisc.sh` (edit its log name to g1-dynvisc-r2.log) -> expect PASS bitwise (max|dres|=0, field diff 0) and stage lines device, resident ON.
   If FAIL: rerun with SU2_KOKKOS_STAGE_VERIFY=1 AND SU2_KOKKOS_NEMO_DYNGRID=0 separately (sv-dynvisc-r2.log, dynoff-visc-r2.log), paste <= 20 StageCompare lines + max|dres| per run, then `HEAVY: 6b.a-DynGrid: dynvisc r2`.
3. Re-run inviscid gate (regression of the move): `timeout 1500 bash .g1run.sh ../kk/logs/g1-dyn-r2.log invwedge_dyn` -> OVERALL PASS bitwise.
4. Poison viscous: copy .g1dynpoison.sh with RUN_TAG/EXTRA_CFG of .g1dynvisc.sh on super_cat -> poison-dynvisc-r2.log, PASS bitwise.
5. Regressions: `timeout 2400 bash .g1run.sh ../kk/logs/g1-dynregress-r2.log` (default list) and `... g1-dynregress2-r2.log invwedge_mg invwedge_lusgs` -> all PASS bitwise.
6. Noise floor dyn (K=10): `CASE=invwedge_dyn bash .nfsmg.sh ../kk/logs/nf-dyn-r1.log` (serial+strict) -> ratio <= 10.
7. Archive all r1 logs already listed in KOKKOS_PROGRESS (g1-dyn, dynoff, dyn1st, dyntr, poison-dyn, dynvisc r1 FAIL, sv-dynvisc r1) + the r2 logs; one INDEX row each; commit.
8. Small edit: check.sh EXPECT for dyn1st hardcodes "(2nd order)" — add a separate EXPECT entry for the dyn1st variant if check.sh supports per-RUN_TAG keys, else note it as a known artifact in INDEX.
Request `MERGE: 6b.a-DynGrid` only after 1-7 pass and logs are committed; always fill the `reason:` line.

## 2026-11-26 — review 21 (4.6-MG2 merged; new dev code 6b.a dynamic grid)

**Reviewed:** 5c4dccefde (code validated at bbcbd98). 36 logs committed, 11 INDEX rows strict PASS, code diff as expected.
**Merged** into su2kokkos as 85e6f71 (+ VALIDATION_LOG row f51a827, plan matrix 5e7c240): CMultiGridIntegration.hpp/.cpp,
CNEMOEulerSolver.hpp/.cpp, CSolver.hpp, CNEMOKokkosData.hpp, CNEMOKokkosMG.cpp, meson.build, check.sh. 4.6-MG2 is DONE —
never re-request it.

**Dev code this round (5327da3; g++ -fsyntax-only with/without -DHAVE_KOKKOS; NOT built, NOT validated)** — queue item 6b.a:
- `CNEMOKokkosData::RefreshDynamicGeometry` (CNEMOKokkosStages.cpp): re-uploads normals, coords, volumes, periodic volumes,
  symmetry normals, grid velocity, Volume_nM1; called once per (TimeIter, OuterIter) from `CNEMOEulerSolver::Preprocessing`.
- `KernelTimeStep` / `KernelMaxEigenvalue`: grid-velocity terms; `DualTimeResidual`: GCL branch (CNEMOKokkosResidual.cpp).
- `Device_DynamicGridUnsupported` replaces the "dynamic grid" gates (MG / periodic / adjoint + dynamic grid stay host;
  `SU2_KOKKOS_NEMO_DYNGRID=0` = old host path). New startup line `stage dynamic grid (...)`.
- Gate case `invwedge_dyn` (TestCases/nonequilibrium/invwedge/invwedge_dyn.cfg: dual time 2nd order, rigid rotation +
  translation, 4 time iters x 4 inner) + check.sh EXPECT. check_noisefloor.sh synced from su2kokkos.

**Agent: do next** (archive every log under `validation_logs/2026-11-26_6b.a-DynGrid/` + one INDEX row each):
1. Build strict + serial (your .buildmg2.sh pattern) -> `build-dyn-r1.log`, `build-dyn-serial-r1.log`. Fix compile errors
   yourself (smallest change).
2. CPU sanity first: `ENABLE_KOKKOS= NO` run of invwedge_dyn.cfg must run 4 time iterations without error
   (`cpu-dyn-r1.log`). If the cfg itself fails on the CPU (option names, ITER), fix the cfg minimally and note it.
3. G1 strict: `timeout 1500 bash .g1run.sh ../kk/logs/g1-dyn-r1.log invwedge_dyn` -> needs OVERALL PASS, bitwise, and all
   4 EXPECT lines (dynamic grid: device, time step: device, dual-time 2nd: device, resident ON). Archive the NO/YES logs.
4. Control: same with `SU2_KOKKOS_NEMO_DYNGRID=0` (expect bitwise, EXPECT lines missing = expected) -> `g1-dynoff-r1.log`.
5. Variants (EXTRA_CFG, RUN_TAG): `TIME_MARCHING= DUAL_TIME_STEPPING-1ST_ORDER` -> `g1-dyn1st-r1.log`;
   `ROTATION_RATE= 0.0 0.0 0.0` (translation only) -> `g1-dyntr-r1.log`; viscous: super_cat.cfg with
   `EXTRA_CFG="TIME_DOMAIN= YES;TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER;TIME_STEP= 1e-7;TIME_ITER= 3;INNER_ITER= 3;GRID_MOVEMENT= RIGID_MOTION;TRANSLATION_RATE= 20.0 0.0 0.0;ITER= 0"`
   (drop ITER if SU2 rejects it) -> `g1-dynvisc-r1.log`.
6. On any FAIL: rerun with `SU2_KOKKOS_STAGE_VERIFY=1` and paste <= 20 StageCompare lines into KOKKOS_PROGRESS.md
   (`sv-dyn-r1.log`); the first non-bitwise stage localizes it (dt, dual-time, or a stage reading stale geometry).
7. Poison (`SU2_KOKKOS_NEMO_POISON=1`) on invwedge_dyn -> `poison-dyn-r1.log`; regressions: default list + invwedge_mg +
   invwedge_lusgs -> `g1-dynregress-r1.log` (must stay bitwise: static grids take the old code paths).
8. Perf: `[KOKKOS_NEMO_PERF]` lines + wall NO/YES of step 3 into KOKKOS_PROGRESS.md.
Request `MERGE: 6b.a-DynGrid` only after 1-7 pass and the logs are committed; `HEAVY: ...` only for new code or a bug you
cannot localize after 3 sessions. Next heavy item after this: 6b.b (pysu2 per-vertex T_wall in / heat flux out, resident).

## 2026-11-26 — review 20 (stale request again; nothing merged, no new dev code)

**Reviewed:** c5fb22465b. Request repeats `MERGE: 4.6-MG1` — **already merged** (su2kokkos 800d75f, VALIDATION_LOG row
8277424, review 18). Your own round-80 entry says the MG2 queue was starting, then the request re-sent MG1. Fixed
LOCAL_AGENT_TASK.md item 4.6 text (it still said "merge requested MG1"). No new heavy code: MG2 (bbcbd98) must be
validated before more code lands on top of it.

**Agent: do next**
1. Before writing any REVIEW_REQUEST: `grep -n "<item>" REVIEW_NOTES.md | head` — if a review lists it as merged, do not request it.
2. Run the review-18 4.6-MG2 queue (below) end to end: strict + serial build; `timeout 1500 bash .g1run.sh ../kk/logs/g1-mg2-r1.log invwedge_mg`
   (expect `MG level 1 device-resident state: ON` and `MG device transfers=` lines); control `SU2_KOKKOS_NEMO_MG_DEVICE=0` (g1-mg2off-r1.log);
   variants mgpost/mgw/mgcs (g1-mg2{post,w,cs}-r1.log); viscous super_cat MGLEVEL=1 must report coarse OFF (viscous) and stay bitwise (g1-mg2visc-r1.log);
   poison (poison-mg2-r1.log); noise floor (nf-mg2-r1.log); regressions default + `invwedge_lusgs invwedge_roe` (g1-mg2regress{,2}-r1.log).
   Archive all under `validation_logs/2026-11-26_4.6-MG2/` + INDEX rows; check `git ls-tree` count == disk count.
3. On a mismatch rerun with `SU2_KOKKOS_STAGE_VERIFY=1`, paste ≤ 20 StageCompare lines in KOKKOS_PROGRESS.md; first suspect = first MG op diverging
   (compare `MGRestrictSolution` / `MGForcingTerm` / `MGProlongCorrection` ordering vs CMultiGridIntegration.cpp). Try 3 sessions before `HEAVY:`.
4. Fix compile errors yourself (smallest change). Request `MERGE: 4.6-MG2` only after strict PASS + logs committed.

## 2026-11-26 — review 19 (stale request; nothing new to merge)

**Reviewed:** e3ba221662. This request repeats `MERGE: 4.6-MG1`, which was **already merged** in review 18
(su2kokkos 800d75f + VALIDATION_LOG row 8277424). Your round-79 `git pull` was blocked by the proxy, so you likely
do not have review 18 locally. No code merged, no new dev code this round.

**Agent: do next**
1. `git pull origin su2kokkos_qwenlocal_agent` (retry until it succeeds; if the proxy keeps blocking, note it in
   KOKKOS_PROGRESS.md and ask Madan). Confirm `git log --oneline -3` shows `bbcbd98 Dev: 4.6-MG2 ...` and
   `e32cd06 Review 18`.
2. Then do the review-18 "Agent: do next" queue below (4.6-MG2: build, G1 invwedge_mg, variants, control
   `SU2_KOKKOS_NEMO_MG_DEVICE=0`, poison, noise floor, regressions, perf lines), logs under
   `validation_logs/2026-11-26_4.6-MG2/` + INDEX rows, `git ls-tree` count check.
3. Request `MERGE: 4.6-MG2` only after the strict gate passes and the logs are committed; never re-request an item
   that REVIEW_NOTES already lists as merged.

## 2026-11-26 — review 18 (merged 4.6-MG1; dev: 4.6-MG2 coarse levels resident + device transfer operators)

**Reviewed:** db03759d2e (code validated at 413b3a0). Logs committed (37/37), INDEX rows strict PASS, code diff as
expected. **Merged** into su2kokkos as 800d75f (+ VALIDATION_LOG row 8277424): CSolver.hpp, CNEMOEulerSolver.hpp/.cpp,
CMultiGridIntegration.cpp, check.sh, check_noisefloor.sh, invwedge_mg.cfg.

**Dev code this round (g++ -std=c++20 -fsyntax-only -ffp-contract=off -Wall, with/without -DHAVE_KOKKOS, vs bundled
Kokkos Serial headers; NOT built, NOT validated):**
- New `SU2_CFD/src/solvers/kokkos/CNEMOKokkosMG.cpp` (added to SU2_CFD/src/meson.build): device MG transfer ops,
  verbatim CMultiGridIntegration loops, no atomics: `MGSetupCoarse` (children CSR, EULER_WALL projection list),
  `MGSetupSmoothing`, `MGResidualTerm`, `MGRestrictSolution` (+ MultigridProjectEulerWall), `MGForcingTerm`,
  `MGProlongCorrection` (+ projection on Solution_Old), `MGProlongScatter`, `MGSmoothSave` / `MGSmoothIteration`,
  `MGSetCorrection`, `MGFillLocalCFL`; counter `mgDeviceOps`.
- CSolver.hpp: `MGDeviceOp`, virtuals `MG_DeviceTransfer(coarse)` / `MG_DeviceOp(...)` (no-op by default).
- CNEMOEulerSolver: coarse levels (iMesh>0) now device-resident when `Device_MGUnsupported` is empty (serial,
  inviscid, no viscous-wall marker, not FULL_MG, static grid, env `SU2_KOKKOS_NEMO_MG_DEVICE` != 0) and the finer level
  is on the device MG path; startup line `[KOKKOS_NEMO] MG level L device-resident state: ON|OFF (why)`; perf line
  `[KOKKOS_NEMO_PERF] MG device transfers=N (level L)`; `AdaptCFLNumber_Device` now also for resident coarse levels.
- CMultiGridIntegration.cpp: each transfer operator dispatches to the device when both levels are on the device path;
  otherwise the step-1 path (fine `SyncHostData(MG_TRANSFER)`, host loops) is unchanged (viscous / MPI / env=0).
- check.sh: `EXPECT[invwedge_mg]` now requires `MG level 1|2 device-resident state: ON` and `MG device transfers=`.

**Agent: do next** (log dir `validation_logs/2026-11-26_4.6-MG2/`, rows in INDEX.md, `git ls-tree` check before
requesting):
1. Build strict + serial (fix compile errors yourself, smallest change; nvcc device-lambda errors: move the lambda to
   a public member or free function, never change the arithmetic). Archive `build-mg2-r1.log` (tail 200 on failure).
2. G1 strict: `timeout 1500 bash .g1run.sh ../kk/logs/g1-mg2-r1.log invwedge_mg` -> must be OVERALL PASS bitwise
   with the new EXPECT lines. Archive g1 log + invwedge_mg_NO/YES logs.
3. Variants (same scripts as MG1, new log names `-mg2-`): `.g1mgpost.sh`, `.g1mgw.sh`, `.g1mgcs.sh` (correction
   smoothing = MGSmoothIteration), `.g1mgpoison.sh` (SU2_KOKKOS_NEMO_POISON=1: proves no missed host read), all
   bitwise. `.g1mgvisc.sh` (viscous: must stay on step 1, `MG level 1 ... OFF (viscous ...)` and bitwise).
4. Control: `SU2_KOKKOS_NEMO_MG_DEVICE=0` run of invwedge_mg (copy .g1mgoff.sh, RUN_TAG=_mg2off): bitwise PASS,
   OVERALL FAIL only for the missing MG2 EXPECT lines (= step-1 behaviour). Archive.
5. Noise floor: `CASE=invwedge_mg` serial+strict (as .nfsmg.sh) -> ratios <= 10. Archive nf-mg2-r1.log.
6. Regressions: default list + `invwedge_lusgs invwedge_roe` bitwise. Archive.
7. Perf lines: paste `[KOKKOS_NEMO_PERF] MG device transfers=` and wall NO/YES for invwedge_mg, compare with MG1
   (NO 63.84 s / YES 53.86 s) in KOKKOS_PROGRESS.md.
8. On any non-bitwise result: rerun with `SU2_KOKKOS_STAGE_VERIFY=1` is NOT useful here (verify disables resident ->
   step-1 path); instead bisect with `SU2_KOKKOS_NEMO_MG_DEVICE=0` (must be bitwise) and report the first iteration /
   level that differs (≤ 20 lines). Suspects: Euler-wall projection order (corner points), smoothing neighbour order,
   halo rows (nPointDomain vs nPoint), LocalCFL fill, coarse CFL adaptation.
Request `MERGE: 4.6-MG2` only after 2-6 pass and the logs are committed; `HEAVY: 4.6-MG2: ...` only for a bug you
cannot localize after 3 sessions. Madan (V100): after the strict gate passes, GPU timing of invwedge_mg:
`SU2_KOKKOS_NEMO_MG_DEVICE=0` vs default, archive both logs under the same dir.

## 2026-11-26 — review 17 (NOT merged: 4.6-MG1 logs missing from git)

**Reviewed:** 54e09ba7e3. Code diff since 065f855 is clean and as expected (CSolver.hpp, CNEMOEulerSolver.hpp/.cpp,
CMultiGridIntegration.cpp, check.sh, check_noisefloor.sh, invwedge_mg.cfg). INDEX.md rows look like a full strict PASS.
**But `validation_logs/2026-11-26_4.6-MG/` does not exist in git** — `*.log` is in .gitignore, so `git add
validation_logs` silently skipped every log (only INDEX.md went in). Per the rule, no merge until the cited logs are
committed. Fixed this round: `.gitignore` now un-ignores `validation_logs/**/*.log(.gz)`; LOCAL_AGENT_TASK.md
VALIDATION LOGS gets a ls-tree check step. **Dev code this round:** none (no new code needed for MG1).

**Agent: do next**
1. `git pull` (merge, no rebase). Copy the round-75 logs from `../kk/logs/` and the chk_* dirs into
   `validation_logs/2026-11-26_4.6-MG/` exactly as the INDEX rows cite them (cpu-mg-r1, g1-mg-r1, g1-mgoff-r1,
   g1-mg{post,w,cs,visc}-r1, poison-mg-r1, nf-mg-r1, g1-regress-r1, g1-regress2-r1 + every `*_NO.log`/`*_YES.log`
   named in the rows). gzip any > 1 MB (then fix the INDEX path to `.log.gz`).
2. `git add -f validation_logs && git commit -m "Log archive: 4.6-MG round-75 logs"`; verify with
   `git ls-tree -r --name-only HEAD validation_logs/2026-11-26_4.6-MG/ | wc -l` (must equal the number of files
   copied) and `git check-ignore validation_logs/2026-11-26_4.6-MG/g1-mg-r1.log` (must print nothing). Push.
3. If any cited log is lost: rerun that gate (same command, new suffix `-r2`), archive it, add a new INDEX row; do not
   edit the old row except to append "(log lost, see -r2)".
4. Also check the older dirs: every log path cited in INDEX.md must exist in git
   (`grep -o 'backfill[^ ,|]*\|2026-[^ ,|]*\.log[.gz]*' validation_logs/INDEX.md | sort -u | while read f; do
   git cat-file -e HEAD:validation_logs/$f 2>/dev/null || echo MISSING $f; done`); archive what you still have, list
   the rest in KOKKOS_PROGRESS.md.
5. Then write REVIEW_REQUEST with `reason: MERGE: 4.6-MG1` (same gate results, now with committed logs). Next heavy
   after merge stays `HEAVY: 4.6-MG2`.

## 2026-11-26 — review 16 (merged 4.6-LUSGS; dev: 4.6-MG step 1, fine level device-resident with multigrid)

**Reviewed:** 065f855fa2. Strict-gate claims checked in the archived logs (g1-lusgs-r64, g1-lusgs-visc-r64,
g1-regress-r68, g1-invall-r68: tol 0.0, OVERALL PASS, no missing stage line; sv/nf r68 PASS). **Merged into su2kokkos
(91f8df0 + VALIDATION_LOG row 0770d05):** the 02676f8 code files (CPreconditioner.hpp, CSysMatrix.hpp/.cpp,
CSysMatrixKokkos.cpp, CNEMOEulerSolver.cpp, check.sh, check_noisefloor.sh, invwedge_lusgs.cfg); plan matrix updated.
Side note: thermalbath in g1-regress-r68 shows wall=0.01s for NO and YES — check that the case really runs (mesh/cfg
path); if it exits immediately the PASS is empty. Report what you find (item 9 below).

**Dev code this round (g++ -std=c++20 -fsyntax-only -ffp-contract=off, with and without -DHAVE_KOKKOS, vs bundled Kokkos
5.2.2 Serial headers, on CNEMOEulerSolver.cpp and CMultiGridIntegration.cpp; NOT validated):**
- `CSolver::HostSyncScope::MG_TRANSFER` (CSolver.hpp) — new scope, no-op for every other solver.
- `CMultiGridIntegration::MultiGrid_Cycle`: one `SyncHostData(MG_TRANSFER)` on the fine solver right after the fine
  level's second Space_Integration (before SetResidual_Term). The host transfer operators (SetResidual_Term,
  SetRestricted_Solution, SetForcing_Term, GetProlongated/SmoothProlongated/SetProlongated_Correction) then see current
  host data. ENABLE_KOKKOS=NO: identical (virtual no-op).
- `CNEMOEulerSolver::SyncHostData(MG_TRANSFER)`: EnsureHost(Solution, LinSysRes, Res_TruncError), then
  MarkHostWritten(Solution, LinSysRes): Cons2Prim re-uploads Solution, the next residual zero-fills the device LinSysRes.
  Counter -> destructor line `[KOKKOS_NEMO_PERF] MG fine-level host transfers=N`.
- `Device_ResidentUnsupported`: MGLEVEL>0 no longer forces resident OFF. Still OFF for `MGCYCLE= FULLMG_CYCLE`
  ("full multigrid cycle") and with `SU2_KOKKOS_NEMO_MG_RESIDENT=0` (old behaviour, for A/B). Coarse levels unchanged
  (their own non-resident pipeline, "MG level").
- Gate case `invwedge_mg` (= invwedge_ausm + MGLEVEL=2, V cycle, pre-smooth 1/2/3, damping 0.75); EXPECT: resident ON
  + the MG transfer perf line. Noise-floor case added.
- Cost: per MG cycle 3 downloads + 2 uploads of nPoint*nVar on the fine level (instead of a fully host run). Step 2
  (later, HEAVY): coarse levels resident + device restriction/prolongation kernels (agglomeration CSR), no copies.

**Agent: do next** (archive every log under `validation_logs/2026-11-26_4.6-MG/` + INDEX.md rows)
1. Build strict + serial. Fix compile errors yourself (smallest change).
2. CPU baseline first: ENABLE_KOKKOS=NO run of invwedge_mg.cfg (11 iters) -> exits 0, residuals finite. Log
   `cpu-mg-r1.log`. If the CPU NEMO multigrid itself fails, stop and request `HEAVY: 4.6-MG: CPU MG fails: <error>`.
3. G1 strict: `timeout 1500 bash .g1run.sh ../kk/logs/g1-mg-r1.log invwedge_mg` -> OVERALL PASS bitwise, both EXPECT
   lines present. Also paste the YES startup line `device-resident state (Step 6): …` and the coarse-level lines.
4. Same with the old behaviour as control: `EXTRA_CFG="" SU2_KOKKOS_NEMO_MG_RESIDENT=0 RUN_TAG=_mgoff timeout 1500 bash
   .g1run.sh ../kk/logs/g1-mgoff-r1.log invwedge_mg` (EXPECT "ON" will be missing there — expected; report the
   tolerances/diffs only). If 4 fails too, the problem is in the coarse-level (non-resident) pipeline, not this round.
5. Variants (each PASS bitwise): `EXTRA_CFG="MG_POST_SMOOTH= ( 1, 1, 1 )" RUN_TAG=_mgpost` (fine-level post-smoothing
   after the host transfer) -> g1-mgpost-r1.log; `EXTRA_CFG="MGCYCLE= W_CYCLE" RUN_TAG=_mgw` -> g1-mgw-r1.log;
   `EXTRA_CFG="MG_CORRECTION_SMOOTH= ( 2, 2, 2 )" RUN_TAG=_mgcs` -> g1-mgcs-r1.log; viscous:
   `EXTRA_CFG="MGLEVEL= 1" RUN_TAG=_mg super_cat` -> g1-mgvisc-r1.log.
6. Host-read hook check: YES run of invwedge_mg with `SU2_KOKKOS_NEMO_POISON=1` -> results identical to step 3's YES
   (restart diff 0). Log `poison-mg-r1.log`. A difference means a host read of a device group the MG code does that
   MG_TRANSFER does not cover: find which (grep GetNodes in CMultiGridIntegration.cpp for the fine solver) and report.
7. On a G1 failure: rerun YES with `SU2_KOKKOS_STAGE_VERIFY=1` (sv-mg-r1.log, ≤ 20 StageCompare lines) and compare the
   first-iteration history line NO vs YES; 3 sessions, then `HEAVY: 4.6-MG: <finding>`.
8. Regression (unchanged paths): default list + `invwedge_lusgs invwedge_roe` PASS bitwise -> g1-regress-r1.log;
   noise floor `CASE=invwedge_mg bash TestCases/kokkos_nemo/check_noisefloor.sh` (ratio ≤ 10) -> nf-mg-r1.log.
9. thermalbath wall=0.01s (see above): check log.txt of that case; note the finding in KOKKOS_PROGRESS.md.
10. PERF: `MG fine-level host transfers=` line + wall NO/YES into KOKKOS_PROGRESS.md. Bookkeeping: plan matrix row
   "multigrid (MGLEVEL>0)" -> "fine level device-resident, coarse host (step 1)"; LOCAL_AGENT_TASK 4.6 note. Then
   request `MERGE: 4.6-MG1` (logs committed first). Next heavy after that: `HEAVY: 4.6-MG2` (coarse levels resident).

**Madan (V100, optional):** `cd $RUN && BIN=<cuda SU2_CFD> bash TestCases/kokkos_nemo/check.sh invwedge_mg` and the
same for invwedge_lusgs; archive the logs under `validation_logs/2026-11-26_gpu/`.

## 2026-11-25 — review 15 (merged 4.5-SupInVisc; dev: 4.6 device LU_SGS)

**Reviewed:** 632c23260f. Strict-gate claims checked (G1 strict supin_visc + default list + invall 7/7 bitwise, no
missing stage line; code unchanged since d2fee8f). **Merged into su2kokkos (986dc2b):** the d2fee8f code files
(CNEMOEulerSolver.cpp, CNEMOKokkosResidual.cpp, check.sh, check_noisefloor.sh, supin_visc.cfg).

**Dev code this round (g++ -std=c++20 -fsyntax-only -ffp-contract=off -DHAVE_KOKKOS vs bundled Kokkos Serial headers
on CSysMatrixKokkos.cpp, CSysMatrix.cpp, CSysSolve.cpp, CNEMOEulerSolver.cpp; NOT validated):**
- `CSysMatrix::KokkosComputeLU_SGSPreconditioner` (CSysMatrixKokkos.cpp, end of file): transcription of
  `ComputeLU_SGSPreconditioner`. Forward sweep (LowerProduct col_lb=partition begin, b - L.x, Gauss_Elimination on a
  copy of the diagonal block) -> halo exchange -> backward sweep (DiagonalProduct, UpperProduct col_ub=partition end
  + halos, Gauss_Elimination) -> halo exchange. Level-scheduled, one persistent team per sweep (ILU pattern);
  levels + per-row partition bounds built once from omp_partitions (`EnsureKokkosLU_SGSStructure`). Same accumulation
  order as gemv_impl, so it should be bitwise. `DeviceGaussElimination` = non-MKL Gauss_Elimination verbatim.
- `CLU_SGSPreconditioner`: device apply only when `Jacobian.GetKokkosValuesOnDevice()` (resident owner). So the
  REF_MODE=HOST reference still runs the CPU LU_SGS: G1 really tests the device sweep.
- NEMO: `Device_ResidentUnsupported` accepts LU_SGS; new report line
  `[KOKKOS_NEMO] stage linear preconditioner (LU_SGS apply): device|host`; destructor prints
  `[KOKKOS_NEMO_PERF] device LU_SGS applies=…` and, with `SU2_KOKKOS_PRECOND_VERIFY=1`,
  `[KOKKOS_NEMO_VERIFY] LU_SGS device apply vs host apply: compares=N non-bitwise=M` (A/B on every apply).
- Gate case `invwedge_lusgs` (= invwedge_ausm + `LINEAR_SOLVER_PREC= LU_SGS`), EXPECT LU_SGS device + resident ON;
  noise-floor case added.
- Perf note (do not fix now): Gauss_Elimination of the diagonal block is redone per row per sweep (as on the CPU);
  levels for natural ordering can be many (one team, one barrier per level). If G7 shows LU_SGS slow, the next step
  is a level-count report, not a physics change.

**Agent: do next**
1. Build strict + serial (ninja, as before). Fix compile errors yourself with the smallest change.
2. G1 strict: `timeout 1500 bash .g1run.sh ../kk/logs/g1-lusgs-r64.log invwedge_lusgs` -> OVERALL: PASS bitwise,
   both EXPECT lines present (LU_SGS device, resident ON).
3. If it fails: `SU2_KOKKOS_PRECOND_VERIFY=1` YES run of invwedge_lusgs, paste the `[KOKKOS_NEMO_VERIFY] LU_SGS`
   line and ≤ 20 StageCompare lines (`timeout 1500 bash .svrun.sh ../kk/logs/sv-lusgs-r64.log invwedge_lusgs`).
   non-bitwise=0 but G1 fails -> the problem is outside the apply (report it). non-bitwise>0 -> check (a) the
   halo exchange between the sweeps, (b) MAXNVAR local arrays, (c) partitions (print omp_num_parts). 3 sessions,
   then `HEAVY: 4.6-LUSGS: <finding>`.
4. Viscous check: `EXTRA_CFG="LINEAR_SOLVER_PREC= LU_SGS" RUN_TAG=_lusgs timeout 1500 bash .g1run.sh
   ../kk/logs/g1-lusgs-visc-r64.log super_cat` -> PASS bitwise, and grep the YES log for `LU_SGS apply): device`.
5. Regression: default list + `invwedge_roe invwedge_msw smol_maxwell supin_visc` (ILU/Jacobi paths unchanged),
   all PASS bitwise.
6. A/B with `SU2_KOKKOS_PRECOND_VERIFY=1` (non-bitwise=0) and noise floor:
   `CASE=invwedge_lusgs bash TestCases/kokkos_nemo/check_noisefloor.sh` (ratio ≤ 10).
7. PERF: `device LU_SGS applies=… (s)` line + wall NO/YES into KOKKOS_PROGRESS.md.
8. Bookkeeping: plan matrix row LU_SGS -> device-resident (validated); LOCAL_AGENT_TASK 4.6 note. Then request
   `MERGE: 4.6-LUSGS`. Next heavy after that: `HEAVY: 4.6-MG` (multigrid coarse levels) — or periodic markers if
   you find a periodic NEMO cfg first. Item 6 (ionization) still waits for Madan.

7. **Log archive (Madan, 2026-11-26), do this first, then keep doing it (see VALIDATION LOGS in LOCAL_AGENT_TASK.md):**
   backfill: copy every existing `../kk/logs/*.log` into `validation_logs/backfill_2026-11-25/` (gzip > 1 MB), list each in
   `validation_logs/INDEX.md` (item/gate/result as far as KOKKOS_PROGRESS.md says; "unknown" otherwise), commit.
   From now on every run's log is committed with its progress note.
8. Ionization with Roe/LAX/MSW: decided — stays on the host. Remove it from your queue.

**Madan (V100, optional):** after the agent's strict PASS, time it on the GPU:
`cd $RUN && BIN=<cuda SU2_CFD> bash TestCases/kokkos_nemo/check.sh invwedge_lusgs` and
`nsys profile -o lusgs SU2_CFD run.cfg` (look at SU2::KokkosLUSGSForward/Backward).

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
