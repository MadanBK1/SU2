# KOKKOS_PROGRESS — inter-session memory (branch su2kokkos_qwenlocal_agent)

## Item status (mirrors section 4 of LOCAL_AGENT_TASK.md)
- [x] 4.1 Script env-var paths (SU2_ROOT/MESH_DIR/RUN_DIR) — done 2026-11-24
- [x] 4.2 Batch A gates: G1, G2, G3, G4, G6 all PASS after fix 4450650725 — done 2026-11-24 round 6
- [~] 4.3 CUDA V100 build + GPU run — G5 PASS at CURRENT HEAD (round 192: build-cuda-r1 rc=0, 21/21 steps; binary 11-27 23:16 supersedes stale Nov-26 binary; log committed fecb7d8123 + INDEX row); G7 BLOCKED (no /dev/nvidia* in sandbox, re-confirmed rounds 190/191/192); nsys BLOCKED (not installed)
- [ ] 4.4 Device schemes: Roe, MSW, LAX, AUSM+-up2
- [ ] 4.5 BCs: subsonic inlet, Smoluchowski–Maxwell slip, catalytic heat-flux wall
- [~] 4.6 Solver options: ILU(n>0) done (device-resident), LU-SGS merged, MG levels merged, periodic markers MERGED (b8b11be), periodic+NS MERGED (review 39), **periodic+DT + periodic+AUSMM MERGED into su2kokkos a69acea (review 40 — do NOT re-validate)**, periodic+Lim: queue FINISHED 11/12 PASS bitwise (code 9af8641ec1); limoff control FAIL (mixed device-comms+host-limiter path) — reviewer decision (a) gate comms host in control / (b) waive (r3 proves (a) bitwise); READY FOR REVIEW round 197; deforming grids pending
- [ ] 4.7 Turbulence coupling (or documented host-only)
- [ ] 4.8 Performance (after 4.3)
- [ ] 4.9 Keep feature-coverage matrix accurate after every item

## Environment facts (roam148-045, V100, GCC 12 toolset)
- Clone: /home/mpr7366/software/AgenticTest/SU2
- Meshes (done): /home/mpr7366/software/AgenticTest/kk/meshes/{invwedge,5x5Mesh,viscwedge}.su2; runs dir ../kk/runs (empty)
- No build-* directories yet (as of 2026-11-24 round 2). G3 needs baseline binaries from cd2cd46 saved before/during the 35b6f39+ builds.
- Toolchain: ninja at /usr/bin/ninja; gcc-toolset-12 at /opt/rh/gcc-toolset-12; meson NOT on PATH — use ./meson.py (vendored externals/meson exists) or pip-less fallback per section 2; Kokkos submodule extracted (subprojects/Kokkos/bin has kokkos_launch_compiler, nvcc_wrapper); externals/eigen present; CoolProp, MLPCpp, Mutationpp extracted.
- G2/G1 gates run against $SU2_ROOT/build-kokkos-strict and build-kokkos-serial; G6 against build-kokkos-strict-mpi; G3 scripts take OLD= (default /home/claude/bin_pre/...) and NEW= env vars → point OLD at the cd2cd46 binary.

## Log

(Entries before round 148 are in KOKKOS_PROGRESS_ARCHIVE.md — grep it, never read it whole.)

- 2026-11-27 (round 161, 4.6-Periodic step 2 diagnosis):
  - cpu-per-r1 CPU sanity FAIL root cause FOUND: CSolver.cpp:348-350 guard `rotate_periodic && config->GetNEMOProblem()` errors on ANY NEMO+periodic run.
    `rotate_periodic` is set unconditionally at MESH_0 (upstream CFVMFlowSolverBase.inl:272; upstream guard commit 41624d6ad2, 2021) — so it blocks pure translational markers too (rotation matrix is identity for zero angles).
    Translation-sign flip (review-33 step 3 fallback) cannot help: the guard never reads marker data. Mesh matching itself worked ("Matched 42 points, max distance 0").
  - Proposed minimal fix (reviewer's call, solver code): guard only when marker angles non-zero, e.g.
    `if (rotate_periodic && config->GetNEMOProblem()) { a = config->GetPeriodicRotAngles(config->GetMarker_All_TagBound(val_periodic_index)); if (a[0]!=0||a[1]!=0||a[2]!=0) SU2_MPI::Error(...); }`
  - THIS SESSION: applying that patch locally to verify it is the ONLY blocker; strict rebuild + rerun of .cpuper.sh (21 iters, ENABLE_KOKKOS=NO).
  - After reviewer approves the guard fix: `rm validation_logs/2026-11-27_4.6-Periodic/.cpu-per-r1.done` and relaunch `per_queue.sh` (steps 2-7 resume).

- 2026-11-27 (round 148, review-32 queue = review-31 4.6-Periodic, step 2 re-running in foreground-ish bg):
  - State: review 32 (empty request, nothing merged) — do review-31 queue: step 1 DONE+committed (builds rc=0, round 136); step 2 CPU sanity (.cpuper.sh) never archived (round-137 bg run died with session; no cpu-per-r1.out, no kk/runs/cpu_per_r1). Re-running THIS round as bg task b7bj8w307 (`bash TestCases/kokkos_nemo/.cpuper.sh`, strict binary, ENABLE_KOKKOS=NO, 21 iters, translation x=+2.0) -> kk/logs/cpu-per-r1.out + kk/runs/cpu_per_r1/log.txt.
  - PASS criteria: rc=0, 21 residual rows, no periodic/matching error. If preprocessing fails on matching vertices: flip invper_ausm.cfg translation 2.0 -> -2.0, rerun as r2, note it.
  - NEXT on pass: archive cpu-per-r1.log + log.txt -> validation_logs/2026-11-27_4.6-Periodic/ + INDEX row + commit; then step 3 G1 strict `.g1per.sh` -> g1-per-r1.log. Limiter step 6 noise floor (.nflim.sh) still owed (run + archive, no merge request).
- 2026-11-27T04:44:47-05:00 [loop] round 150 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_040551_150.log

- 2026-11-27T06:43:52-05:00 [loop] round 158 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_062426_158.log

- 2026-11-27T06:50:56-05:00 [loop] round 159 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_064413_159.log

- 2026-11-27T06:59:54-05:00 [loop] round 160 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_065116_160.log

- 2026-11-27T07:17:20-05:00 [loop] round 161 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_070015_161.log

LOOP HINT: 2026-11-27T07:17:20-05:00 no progress for 3 rounds on the current item. No code changed (only the progress log). Stop planning: in THIS session write code immediately - the smallest compilable piece (e.g. one device function + its call site), build it, commit it. If you cannot, mark the item BLOCKED and move to the next unchecked item.

- 2026-11-27T07:31:15-05:00 [loop] round 162 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_072740_162.log

- 2026-11-27T07:49:21-05:00 [loop] round 163 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_073135_163.log

- 2026-11-27T08:08:32-05:00 [loop] round 164 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_074941_164.log

- 2026-11-27T08:40:06-05:00 [loop] round 165 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_080852_165.log

- 2026-11-27 (round 166, 4.6-Periodic: resumed queue after guard fix):
  - Guard fix (CSolver.cpp:348-356, angle check before NEMO periodic error) was already committed in 774b3e6bfd (round-163 autosave) and BUILT into build-kokkos-strict (binary 07:44:25 > source 07:43:29); cfg invper_ausm has zero rot angles -> should pass.
  - Removed .cpu-per-r1.done marker; relaunched per_queue.sh (harness bg task bhdclmz93, NOT setsid — detached launch dies in sandbox). Steps 2-7 now running: cpu-per-r1, g1-per, 5 variants, ab/poison, 8-case regress, noise floor.
  - NEXT: cat QUEUE_STATUS.txt; per new END line add INDEX row + `git add -f validation_logs` + commit (never commit .queue.lock). All PASS+committed -> `READY FOR REVIEW: MERGE: 4.6-Periodic` (incl. reviewer approval of the guard fix). CPU sanity FAIL again -> `HEAVY: 4.6-Periodic: CPU sanity <error line>`.

- 2026-11-27T09:45:16-05:00 [loop] round 166 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_084027_166.log
- 2026-11-27 (review-35 steps): built strict+serial after dev segfault fix 1c0b4f8 (CSolver.cpp:348-361 periodic-marker loop); both rc=0, logs build-per-r2.log / build-per-serial-r2.log, INDEX rows added.
  - NEXT: rm .cpu-per-r1.done; relaunch `bash TestCases/kokkos_nemo/per_queue.sh` (harness bg, NOT setsid); per new END line add INDEX row + commit; all PASS -> `READY FOR REVIEW: MERGE: 4.6-Periodic — <results>`.

- 2026-11-27T09:54:37-05:00 [loop] round 167 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_094537_167.log

- 2026-11-27T10:26:40-05:00 [loop] round 168 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_095457_168.log

- 2026-11-27T11:10:32-05:00 [loop] round 169 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_102700_169.log

- 2026-11-27T11:32:01-05:00 [loop] round 170 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_111052_170.log

- 2026-11-27 (round 171, 4.6-Periodic: queue complete, ALL gates PASS):
  - Queue (10:07, code 7a1da1532e): cpu-per PASS; g1-per invper_ausm PASS bitwise; variants exp/muscl/lax/off PASS bitwise; poison PASS; 8-case regression PASS bitwise (372s).
  - venk/ab r1 FAILs were check-script (not solver) — rerun with fixed scripts (.rerun3.sh, 11:10): g1-per-venk-sv, ab-per-r2, ab-per-muscl-r2 all PASS bitwise.
  - Noise floor r1/r2 FAILed (run dirs never materialized); rerun THIS round: nf-per-r3 PASS (ratios 0.64 ITER11 / 1.12 ITER100, K=10).
  - INDEX: 17 rows added (PASS + superseded FAILs kept); committing logs + runner scripts .rerun3.sh/.nfper.sh.
  - NEXT (reviewer): MERGE 4.6-Periodic incl. CSolver.cpp guard fix (1c0b4f8). Then remaining 4.6: dynamic/deforming grids.

READY FOR REVIEW: MERGE: 4.6-Periodic — CPU sanity PASS; G1 strict invper_ausm + variants (exp/muscl/lax/off/venk) + periodic-OFF control all bitwise PASS (max|dres|=0.00e+00); A/B stage-verify bitwise PASS (plain + MUSCL); poison PASS; 8-case default regression bitwise PASS; noise floor PASS (ratios 0.64/1.12, K=10). Includes CSolver.cpp translational-periodic guard fix (1c0b4f8). All logs under validation_logs/2026-11-27_4.6-Periodic/ + INDEX rows.

- 2026-11-27T12:45:19-05:00 [loop] round 171 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_113221_171.log

- 2026-11-27 (round 172, review-36 follow-up: gate logs were present but uncommitted — the only rejection reason):
  - All 16 missing logs verified non-empty in validation_logs/2026-11-27_4.6-Periodic/; tails spot-checked:
    g1-per-r1 / ab-per-r2 / poison-per-r1 / g1-per-regress-r1 OVERALL PASS (max|dres|=0.00e+00); nf-per-r3 NOISEFLOOR PASS (0.64/1.12).
  - `git add -f` the 16 logs + runner scripts .rerun3.sh/.nfper.sh (excluded .queue.lock, .done markers);
    `git ls-files` count = 23 (>= 23 required); committed 080bf8d592 "Validation: archive 4.6-Periodic gate logs".
  - Re-filing the merge request below (same claims as round 171; logs now committed). Next heavy item after merge: 4.6 dynamic/deforming grids.

READY FOR REVIEW: MERGE: 4.6-Periodic — CPU sanity PASS; G1 strict invper_ausm + variants (exp/muscl/lax/off/venk) + periodic-OFF control all bitwise PASS (max|dres|=0.00e+00); A/B stage-verify bitwise PASS (plain + MUSCL); poison PASS; 8-case default regression bitwise PASS; noise floor PASS (ratios 0.64/1.12, K=10). Includes CSolver.cpp translational-periodic guard fix (1c0b4f8). All 23 files under validation_logs/2026-11-27_4.6-Periodic/ now committed (080bf8d592) + INDEX rows.

- 2026-11-27 (round 173, 4.6-PeriodicNS per review 37):
  - Edits: check.sh +EXPECT key `viscper_ausm_pnsab` (= invper_ausm_perab string); check_noisefloor.sh +case `viscper_ausm` (HREL=0).
  - Queue: build strict+serial -> CPU sanity (21 it, strict, KOKKOS=NO) -> G1 strict viscper_ausm -> variants exp/muscl/nsoff -> A/B _pnsab -> poison -> 8-case+invper regress -> noise floor K=10 -> MERGE.
  - Results (all logs committed under validation_logs/2026-11-27_4.6-PeriodicNS/ + 12 INDEX rows; runner pns_queue.sh, code c4f21b5 on 5a12c86):
    - build strict+serial rc=0 ([3/3] CNEMOEulerSolver.cpp + link); CPU sanity 21 iters PASS (rc=0).
    - G1 strict viscper_ausm PASS bitwise; variants exp-r2/muscl/nsoff PASS; A/B _pnsab PASS; poison PASS; regress 9/9 bitwise (349s); NF PASS ratios 0.83/1.33 (K=10).
    - g1-pns-exp-r1 FAIL was EXPECT-key form only (explicit prints "(CAvgGradCorrected_NEMO)" without ", residual + Jacobian"); added key `viscper_ausm_exp`, r2 PASS bitwise.
  - NEXT: reviewer — MERGE 4.6-PeriodicNS. Then review-37 queue continues: periodic+MG, periodic+dual time, PERIODIC_SENSOR (AUSM+M/JST), periodic limiters.

READY FOR REVIEW: MERGE: 4.6-PeriodicNS — CPU sanity PASS (21 iters, rc=0); G1 strict viscper_ausm bitwise PASS (max|dres|=0.00e+00, field diff=0.00e+00); variants EULER_EXPLICIT (r2, new EXPECT key viscper_ausm_exp) + MUSCL/VAN_ALBADA + NS-gate control (SU2_KOKKOS_NEMO_PERIODIC_NS=0) all bitwise PASS; A/B stage-verify bitwise PASS; poison PASS; 9/9-case regression (default 8 + invper_ausm) bitwise PASS (349s); noise floor PASS (ratios 0.83/1.33, K=10). 12 logs + runner scripts under validation_logs/2026-11-27_4.6-PeriodicNS/ (committed) + 12 INDEX rows.

- 2026-11-27T14:47:13-05:00 [loop] round 173 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_134638_173.log

- 2026-11-27T15:52:14-05:00 [loop] round 174 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_144733_174.log
- 2026-11-27 (round 175, review-39 queue A = finish 4.6-PeriodicDT):
  - HEAD 60a160f26b has DT (b0008fc) + PeriodicNS + new dev code AUSMM/Lim (not yet built). Round-174 binaries predate AUSMM/Lim; DT queue never ran (pdt_queue.sh existed, no QUEUE_STATUS.txt).
  - Plan: rebuild strict+serial at THIS head (logs build-pdt-r2*.log) -> launch pdt_queue.sh (steps: cpu-pdt-r1, g1-pdt-r1, variants 1st/muscl/off, ab, poison, 10-case regress incl. viscper_ausm, nf K=10) -> commit logs + INDEX. Then B (AUSMM), C (Lim), D (regress), E (one MERGE request per review 39).
  - Loop hint 07:17 superseded by review-39 queue (test work, no code unless compile error).
- 2026-11-27 (round 175 cont, review-39 queue A COMPLETE — 4.6-PeriodicDT all gates PASS):
  - Rebuilt strict+serial at HEAD 60a160f26b (build-pdt-r2.log / build-pdt-serial-r2.log, 16/16 rc=0; AUSMM/Lim dev code now compiled).
  - pdt_queue.sh: cpu-pdt PASS; g1-pdt invper_dt PASS bitwise; variants dt1st/muscl/dtoff PASS bitwise; ab PASS; poison PASS; regress 10/10 bitwise (360s, incl. viscper_ausm); nf K=10 PASS ratio 0.46.
  - 10 INDEX rows added; logs committed. NEXT: queue B 4.6-PeriodicAUSMM (build-mm logs, cpu, g1, exp, mmoff, ab, poison, nf), then C 4.6-PeriodicLim, D regress, E one MERGE request.

- 2026-11-27T16:29:49-05:00 [loop] round 175 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_155234_175.log
- 2026-11-27 (round 176, review-39 queue B COMPLETE — 4.6-PeriodicAUSMM all gates PASS):
  - mm_queue.sh (launched round 175, finished 16:17:39, code 7704af8288): builds rc=0 (no-op at this head); cpu-mm 21 iters PASS; g1-mm invper_ausmm PASS bitwise (sensor (AUSM+M): device, resident ON); variants exp + control mmoff PASS bitwise; ab PASS; poison PASS; nf K=10 PASS ratios 0.60/0.93.
  - 8 INDEX rows added; logs (17 files) + runner mm_queue.sh committed this session.
  - NEXT: queue C 4.6-PeriodicLim (lim_queue.sh: cpu, g1, bj/wang/limoff/pervenk variants, ab, poison, nf), D regress (default 8 + invper_ausm viscper_ausm invper_dt invper_ausmm invper_lim -> g1-regress-r1 in Lim dir), E ONE MERGE request (review-39 E).

- 2026-11-27T17:04:10-05:00 [loop] round 176 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_163009_176.log

- 2026-11-27T17:21:50-05:00 [loop] round 177 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_170430_177.log

- 2026-11-27T17:37:29-05:00 [loop] round 178 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_172210_178.log

- 2026-11-27T18:00:39-05:00 [loop] round 179 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_173749_179.log

LOOP HINT: 2026-11-27T18:00:39-05:00 no progress for 3 rounds on the current item. No code changed (only the progress log). Stop planning: in THIS session write code immediately - the smallest compilable piece (e.g. one device function + its call site), build it, commit it. If you cannot, mark the item BLOCKED and move to the next unchecked item.

- 2026-11-27T18:21:38-05:00 [loop] round 180 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_181059_180.log

- 2026-11-27T18:38:16-05:00 [loop] round 181 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_182159_181.log

- 2026-11-27T19:43:35-05:00 [loop] round 182 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_183837_182.log

- 2026-11-27T20:02:59-05:00 [loop] round 183 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_194355_183.log

- 2026-11-27T20:23:57-05:00 [loop] round 184 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_200319_184.log

- 2026-11-27T20:48:35-05:00 [loop] round 185 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_202417_185.log

LOOP HINT: 2026-11-27T20:48:35-05:00 no progress for 3 rounds on the current item. No code changed (only the progress log). Stop planning: in THIS session write code immediately - the smallest compilable piece (e.g. one device function + its call site), build it, commit it. If you cannot, mark the item BLOCKED and move to the next unchecked item.

- 2026-11-27T21:20:33-05:00 [loop] round 186 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_205856_186.log

- 2026-11-27 (review-39 queue C+D closed out — Lim + regress logs committed; limoff control localized):
  - State found this session: queues A (DT, 25 logs committed) + B (AUSMM, 11 committed) DONE. Queue C (Lim) had RUN (logs on disk 16:43-16:49) but 0 committed; queue D (regress) PASS but uncommitted. Limoff-verify-r1.log (0 bytes) = a dead round-184 re-run, NOT part of the queue, left untracked.
  - Lim gates (code 9af8641ec1): build strict+serial rc=0; CPU sanity PASS; g1-lim bitwise PASS; variants bj/wang/pervenk bitwise PASS; **g1-lim-off (SU2_KOKKOS_NEMO_PERIODIC_LIM=0) FAIL: max|dres|=3.95e-02, field diff=1.54e-01**; ab/poison/nf (ratios 1.22/0.49) PASS; regress 10/10 bitwise PASS (360s). 12 Lim INDEX rows + QUEUE_STATUS + lim_queue.sh committed this session (stray .claude/.cc-writes + 0-byte log excluded).
  - ROOT CAUSE of limoff (localized, not a port bug): the PERIODIC_LIM=0 control only forces the limiter stage to host while leaving 4.6-Periodic device comms ON (check.sh EXPECT key: "limiter: host ... | device-resident OFF" — no "periodic comms: host"). Contrast the PASSING dtop/mmaoff controls whose EXPECT includes "periodic comms: host". So limoff is a mixed device-comms+host-limiter path that legitimately deviates from the full-host reference. Fix is reviewer's call (either gate comms to host too, or drop/waive this control); NOT something I may fix per ROLE (code change, not compile error).
  - DT + AUSMM are fully green (all bitwise, controls incl. off, poison, NF, regress). Ready to merge.
  - NEXT: reviewer — (a) MERGE 4.6-PeriodicDT + 4.6-PeriodicAUSMM (all logs committed, all gates PASS); (b) decide limoff control for 4.6-PeriodicLim (all other Lim gates + 10/10 regress already PASS bitwise, logs committed) — merge Lim with the control waived/fixed, or tell me to re-run after your fix.
READY FOR REVIEW: MERGE: 4.6-PeriodicDT + 4.6-PeriodicAUSMM — DT: CPU sanity PASS, G1 invper_dt bitwise + dt1st/muscl/dtoff variants bitwise, A/B+poison bitwise, 10/10 regress bitwise (360s), NF ratio 0.46 (K=10). AUSMM: CPU sanity PASS, G1 invper_ausmm bitwise + exp variant bitwise + mmoff control bitwise, A/B+poison bitwise, NF ratios 0.60/0.93 (K=10). All logs committed under validation_logs/2026-11-27_4.6-PeriodicDT/ + /4.6-PeriodicAUSMM/ + INDEX rows. 4.6-PeriodicLim also staged: 11/12 gates PASS bitwise (g1/bj/wang/pervenk/ab/poison/nf + 10/10 regress) but limoff control FAIL (field 1.54e-01; mixed device-comms+host-limiter path — needs your decision, see progress).

- 2026-11-27T22:02:48-05:00 [loop] round 187 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_212053_187.log

- 2026-11-27T22:36:48-05:00 [loop] round 188 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_220308_188.log

- 2026-11-27 (round 189, status check — review-39 queue done, holding for reviewer):
  - Review-39 queue: A (DT) PASS, B (AUSMM) PASS, D (10/10 regress) PASS — all committed. C (Lim): 11/12 gates PASS bitwise; limoff control BLOCKED reviewer-decision (mixed device-comms+host-limiter path; the fix is a code change = reviewer's call, not mine).
  - Review request (MERGE DT+AUSMM + limoff decision) already filed; loop pushed REVIEW_REQUEST (5eb67bf329). Newest REVIEW_NOTES.md section is still review 39 — review 40 not yet in.
  - No new reviewer queue to execute. Next items (multigrid / PERIODIC_PRIM_LS / deforming grid) are reviewer-reserved "heavy items". Holding for review 40 rather than starting a competing dev track (would collide with the reviewer's next dev code).

READY FOR REVIEW: MERGE: 4.6-PeriodicDT + 4.6-PeriodicAUSMM — DT: CPU sanity PASS, G1 invper_dt bitwise + dt1st/muscl/dtoff variants bitwise, A/B+poison bitwise, 10/10 regress bitwise (360s), NF ratio 0.46 (K=10). AUSMM: CPU sanity PASS, G1 invper_ausmm bitwise + exp variant bitwise + mmoff control bitwise, A/B+poison bitwise, NF ratios 0.60/0.93 (K=10). All logs committed under validation_logs/2026-11-27_4.6-PeriodicDT/ + /4.6-PeriodicAUSMM/ + INDEX rows. 4.6-PeriodicLim staged: 11/12 gates PASS bitwise (g1/bj/wang/pervenk/ab/poison/nf + 10/10 regress) but limoff control BLOCKED pending your decision (mixed device-comms+host-limiter path).

- 2026-11-27 (round 190, 4.3 G5 re-verify at current HEAD):
  - Reviewer queue still on review 39 (MERGE DT+AUSMM + limoff decision pending; review 40 not yet in). No new dev code to test.
  - Found CUDA binary stale: build-kokkos-cuda/SU2_CFD 11-26 23:23, but solvers/CNEMOEulerSolver.cpp + kokkos/*.cpp changed 11-27 (DT/NS/AUSMM/Lim code never compiled under real nvcc).
  - THIS SESSION: rebuild build-kokkos-cuda at HEAD via TestCases/kokkos_nemo/.buildcuda.sh (bg task bbg2vy855), log validation_logs/2026-11-27_4.3-CUDA/build-cuda-r1.log. G7 still BLOCKED (no /dev/nvidia* in sandbox, re-confirmed).
  - NEXT: rc=0 -> INDEX row + commit (G5 PASS at this head); TU fail -> -j1 -v diagnose, smallest fix, rebuild.

- 2026-11-27T23:11:25-05:00 [loop] round 190 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_225357_190.log

- 2026-11-27 (round 191, 4.3 G5 rebuild resume):
  - Round 190's bg build (bbg2vy855) never ran: log dir validation_logs/2026-11-27_4.3-CUDA/ absent, binary still Nov 26 23:23.
  - THIS SESSION: ninja dry-run = 21 steps (5 NEMO TUs + link). Relaunching incremental build via .buildcuda.sh (bg task bipknb0qa), log validation_logs/2026-11-27_4.3-CUDA/build-cuda-r1.log.
  - G7 still BLOCKED: no /dev/nvidia* in sandbox (re-confirmed this session). nvcc 12.9 present, sm_70 env set.
  - NEXT: rc=0 -> INDEX row + commit (G5 PASS at this head); TU fail -> -j1 -v, bash -x dispatcher diagnose, smallest fix, rebuild.

- 2026-11-27T23:24:50-05:00 [loop] round 191 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_231145_191.log

- 2026-11-27 (round 192, 4.3 G5 re-verify CLOSED at current HEAD):
  - Round-191 bg build (bipknb0qa) completed 23:16: build-cuda-r1.log ends RC=0, [21/21] link; binary build-kokkos-cuda/SU2_CFD/src/SU2_CFD now 11-27 23:16 (stale Nov-26 23:23 binary superseded; all 11-27 NEMO/DT/NS/AUSMM/Lim dev code now compiled under real nvcc). Log committed in fecb7d8123; INDEX row added this session.
  - G7 re-confirmed BLOCKED (no /dev/nvidia*; nvidia-smi can't reach driver). nsys still not installed.
  - Reviewer still on review 39 (MERGE DT+AUSMM + limoff decision pending; review 40 not yet in). Heavy items (periodic+MG, PERIODIC_PRIM_LS, deforming grids) are reviewer-reserved — holding rather than starting a competing dev track.

READY FOR REVIEW: MERGE: 4.6-PeriodicDT + 4.6-PeriodicAUSMM — DT: CPU sanity PASS, G1 invper_dt bitwise + dt1st/muscl/dtoff variants bitwise, A/B+poison bitwise, 10/10 regress bitwise (360s), NF ratio 0.46 (K=10). AUSMM: CPU sanity PASS, G1 invper_ausmm bitwise + exp variant bitwise + mmoff control bitwise, A/B+poison bitwise, NF ratios 0.60/0.93 (K=10). All logs committed under validation_logs/2026-11-27_4.6-PeriodicDT/ + /4.6-PeriodicAUSMM/ + INDEX rows. 4.6-PeriodicLim staged: 11/12 gates PASS bitwise (g1/bj/wang/pervenk/ab/poison/nf + 10/10 regress) but limoff control BLOCKED pending your decision (mixed device-comms+host-limiter path).

- 2026-11-27T23:44:18-05:00 [loop] round 192 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_232510_192.log

- 2026-11-27 (round 193, 4.9 feature-matrix accuracy — docs-only chunk):
  - Reviewer still on review 39 (MERGE DT+AUSMM + limoff decision; review 40 not yet in). G7 re-confirmed BLOCKED (no /dev/nvidia* in sandbox this session). No new reviewer queue to run.
  - THIS SESSION (4.9, first unchecked actionable item I own): fixed stale NEMO_KOKKOS_PORT_PLAN.md matrix rows: (a) "periodic LS/limiter host fallback" -> periodic limiter (BJ/Venkat/Venkat-Wang) device-resident (4.6-PeriodicLim, validated, limoff control pending reviewer) + periodic weighted LS still host; (b) "periodic ... host fallback" -> periodic translational markers device-resident (invper_ausm+viscper_ausm MERGED 3340824; invper_dt/invper_ausmm/invper_lim validated bitwise, merge requested review 39) + turbomachinery/Giles/actuator-disk still host; (c) MG2 row "merge requested" -> "MERGED into su2kokkos 800d75f" (per task file 4.6).
  - Docs only: no build needed, zero CPU-path impact (markdown). No gates run (n/a for docs).
  - NEXT: hold for review 40 (limoff decision + MERGE DT/AUSMM). Then: reviewer's next heavy items (periodic+MG, PERIODIC_PRIM_LS) land -> test queue. 4.3 G7 still BLOCKED for me (run from user's shell; commands in earlier entries).
READY FOR REVIEW: MERGE: 4.6-PeriodicDT + 4.6-PeriodicAUSMM — DT: CPU sanity PASS, G1 invper_dt bitwise + dt1st/muscl/dtoff variants bitwise, A/B+poison bitwise, 10/10 regress bitwise (360s), NF ratio 0.46 (K=10). AUSMM: CPU sanity PASS, G1 invper_ausmm bitwise + exp variant bitwise + mmoff control bitwise, A/B+poison bitwise, NF ratios 0.60/0.93 (K=10). All logs committed under validation_logs/2026-11-27_4.6-PeriodicDT/ + /4.6-PeriodicAUSMM/ + INDEX rows. 4.6-PeriodicLim staged: 11/12 gates PASS bitwise (g1/bj/wang/pervenk/ab/poison/nf + 10/10 regress) but limoff control BLOCKED pending your decision (mixed device-comms+host-limiter path). Round 193 (this session): 4.9 feature-matrix rows corrected in NEMO_KOKKOS_PORT_PLAN.md (periodic BC/limiter rows now device-resident per validated gates; MG2 marked MERGED 800d75f) — docs only, no gates needed.

- 2026-11-28T00:13:32-05:00 [loop] round 194 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_235620_194.log

- 2026-11-28 (round 195, review 40 landed on origin 3b8bdcf6f0 — limoff StageVerify evidence):
  - Review 40: MERGED 4.6-PeriodicDT + 4.6-PeriodicAUSMM into su2kokkos a69acea (do NOT re-validate them); 4.6-PeriodicLim hunks NOT merged (still unvalidated); reviewer's lim_queue.sh runner = equivalent to local committed one (comment-only diff).
  - Queue C+D already QUEUE FINISHED (round-192 run, code 9af8641ec1): build/cpu/g1/variants/ab/poison/nf PASS bitwise + regress 13 cases PASS (360s); ONLY failure g1-lim-off: max|dres|=3.95e-02, field diff=1.54e-01 (numeric residual-row deviation, not an EXPECT-text issue — residual rows NO vs YES: -0.386878/-0.911641 vs -0.34816/-0.872923).
  - THIS SESSION (limoff evidence, review-40 step 2): limoff re-run with SU2_KOKKOS_STAGE_VERIFY=1 (forces comms host too) via TestCases/kokkos_nemo/.limoff_verify.sh:
    - limoff-verify-r2.log (no RUN_TAG, EXPECT fell back to plain key): stage-line artifact aside, residual/field BITWISE vs full-host: max|dres|=0.00e+00, field diff=0.00e+00; verify line "[KOKKOS_NEMO_VERIFY] MG 0 device-resident Primitive checks=42 mismatching entries=0"; stages: upwind device, bc:EULER_WALL device (eager), resident OFF (STAGE_VERIFY).
    - limoff-verify-r3.log (RUN_TAG=_limoff, EXPECT key invper_lim_limoff): **invper_lim_limoff: PASS, OVERALL: PASS, bitwise** (residual rows identical: -0.386878 -0.911641 -32 -32 -22.948773 ...).
  - CONCLUSION for reviewer: full-host limoff path (limiter host + comms host) is bitwise-clean; the failing g1-lim-off is SPECIFICALLY the mixed path (PERIODIC_LIM=0 keeps device periodic comms ON + limiter host). Options (code = your call): (a) gate comms to host in the PERIODIC_LIM=0 control (then it will PASS bitwise, as r3 shows), or (b) waive/drop the control. All other Lim gates + 13-case regress PASS bitwise, logs committed.

- 2026-11-28T00:39:30-05:00 [loop] round 195 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261128_001352_195.log

- 2026-11-28T00:58:40-05:00 [loop] round 196 ended rc=1 with NO progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261128_003950_196.log

- 2026-11-28 (round 197, 4.6-PeriodicLim queue CLOSED; review 40 synced):
  - Review 40 (origin 3b8bdcf6f0, now merged locally): MERGED 4.6-PeriodicDT + 4.6-PeriodicAUSMM into su2kokkos a69acea (do NOT re-validate); Lim hunks not merged — still awaiting validation sign-off. lim_queue.sh (reviewer) = equivalent to local runner (comment-only diff); queue already QUEUE FINISHED (round-192 run, code 9af8641ec1): 11/12 gates PASS bitwise (build/cpu/g1/bj/wang/pervenk/ab/poison/nf 1.22+0.49 + regress 13 cases, 360s); ONLY g1-lim-off FAIL (mixed path, max|dres|=3.95e-02).
  - limoff StageVerify evidence (round 195) now committed + INDEX rows: full-host limoff path (STAGE_VERIFY=1, comms host) bitwise — incl. r3 `invper_lim_limoff: PASS, OVERALL: PASS, bitwise` → the failing control is SPECIFICALLY the mixed path (PERIODIC_LIM=0 keeps device comms ON). Decision (a) gate periodic comms to host in the PERIODIC_LIM=0 control [r3 proves it then bitwise] or (b) waive/drop the control = reviewer call (solver code = reviewer's per ROLE; conservative choice: do NOT self-apply).
  - THIS SESSION: INDEX rows for limoff-verify-r2/r3 added; evidence logs committed (git add -f); review-40 commit merged into local (REVIEW_NOTES.md now at review 40). G7 still BLOCKED for me (no /dev/nvidia* in sandbox; re-confirmed rounds 190-192; run from user's shell — commands in earlier entries).
  - NEXT: reviewer (a)/(b) decision on limoff → MERGE: 4.6-PeriodicLim. Then reviewer's next heavy items (periodic+MG, PERIODIC_PRIM_LS) land → run their queues (same gate recipe as review-40 lim_queue).
READY FOR REVIEW: MERGE: 4.6-PeriodicLim — queue C+D (review 40) FINISHED at code 9af8641ec1: 11/12 gates PASS bitwise — build strict+serial rc=0, CPU sanity PASS, G1 invper_lim bitwise, BARTH_JESPERSEN + VENKATAKRISHNAN_WANG variants bitwise, old invper_ausm pervenk variant bitwise, A/B stage-verify bitwise (Limiter_Primitive/Solution_Max/Solution_Min), poison bitwise, noisefloor PASS (ratios 1.22/0.49, K=10), regress 13/13 cases bitwise (360s). STUCK (1): g1-lim-off control FAIL (PERIODIC_LIM=0 = mixed device-comms+host-limiter path; max|dres|=3.95e-02, field diff=1.54e-01). Evidence committed: limoff-verify-r2/r3 (STAGE_VERIFY=1 full-host limoff path) BITWISE incl. r3 invper_lim_limoff PASS — so the control is only the mixed path. Decision needed (your call, solver code): (a) gate periodic comms to host in the PERIODIC_LIM=0 control (r3 proves it then bitwise; mmoff/dtoff do exactly this and PASS), or (b) waive/drop the limoff control. All logs committed under validation_logs/2026-11-27_4.6-PeriodicLim/ + INDEX rows 251-263.

- 2026-11-28T02:49:08-05:00 [loop] round 198 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261128_014812_198.log

- 2026-11-28 (round 199, review-41 queue RUNNING): HEAD 52527ec85c contains limValid fix (9054ef3b25) + zc2a-bndsync dev (0ec62f5abb); strict/serial binaries pre-date them -> rebuilding.
  - Queue (review 41): build S+serial; G1 S invper_lim limoff (must now PASS bitwise); invper_ausm MUSCL+Venkat limoff (new EXPECT invper_ausm_limoff); invwedge_roe hostlim control (new EXPECT); invper_lim device + A/B _limab; regress 13.
  - Then review-42 queue (zc2a-bndsync B2-B6) if time; combined MERGE per B7.
