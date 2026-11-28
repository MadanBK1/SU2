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

(Entries before round 215 are in KOKKOS_PROGRESS_ARCHIVE.md — grep it, never read it whole.)

- 2026-11-28 review 44 (reviewer): CURRENT STATE. Untested dev code at HEAD: limValid fix (9054ef3b25) + zc2a-bndsync (0ec62f5abb).
  Rounds 202-214 made no progress (stale loop, empty-reason requests). Validation is ONE detached command:
  `setsid nohup bash TestCases/kokkos_nemo/limfix_build_and_run.sh > /dev/null 2>&1 < /dev/null &` (builds, limfix queue, then bnd_queue).
  Check: `tail -5 validation_logs/2026-11-28_4.6-PeriodicLim-fix/QUEUE_STATUS.txt validation_logs/2026-11-28_zc2a-bndsync/QUEUE_STATUS.txt; pgrep -af queue`
  Next: REVIEW_NOTES.md review 44 steps 1-4.
