# KOKKOS_PROGRESS — inter-session memory (branch su2kokkos_qwenlocal_agent)

## Item status (mirrors section 4 of LOCAL_AGENT_TASK.md)
- [x] 4.1 Script env-var paths (SU2_ROOT/MESH_DIR/RUN_DIR) — done 2026-11-24
- [x] 4.2 Batch A gates: G1, G2, G3, G4, G6 all PASS after fix 4450650725 — done 2026-11-24 round 6
- [~] 4.3 CUDA V100 build + GPU run — G5 PASS (build-kokkos-cuda 320/320, rc=0); G7 BLOCKED (no /dev/nvidia* in sandbox; dangerouslyDisableSandbox refused); nsys BLOCKED (not installed)
- [ ] 4.4 Device schemes: Roe, MSW, LAX, AUSM+-up2
- [ ] 4.5 BCs: subsonic inlet, Smoluchowski–Maxwell slip, catalytic heat-flux wall
- [~] 4.6 Solver options: ILU(n>0) done (device-resident), LU-SGS merged, MG levels merged, periodic markers MERGED (b8b11be), **periodic+NS ALL GATES PASS 2026-11-27 (merge requested, review 37)**, deforming grids pending
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
