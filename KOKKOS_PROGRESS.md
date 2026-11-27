# KOKKOS_PROGRESS — inter-session memory (branch su2kokkos_qwenlocal_agent)

## Item status (mirrors section 4 of LOCAL_AGENT_TASK.md)
- [x] 4.1 Script env-var paths (SU2_ROOT/MESH_DIR/RUN_DIR) — done 2026-11-24
- [x] 4.2 Batch A gates: G1, G2, G3, G4, G6 all PASS after fix 4450650725 — done 2026-11-24 round 6
- [~] 4.3 CUDA V100 build + GPU run — G5 PASS (build-kokkos-cuda 320/320, rc=0); G7 BLOCKED (no /dev/nvidia* in sandbox; dangerouslyDisableSandbox refused); nsys BLOCKED (not installed)
- [ ] 4.4 Device schemes: Roe, MSW, LAX, AUSM+-up2
- [ ] 4.5 BCs: subsonic inlet, Smoluchowski–Maxwell slip, catalytic heat-flux wall
- [ ] 4.6 Solver options: ILU(n>0), LU-SGS, MG levels, periodic markers, deforming grids
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

- 2026-11-27 (round 148, review-32 queue = review-31 4.6-Periodic, step 2 re-running in foreground-ish bg):
  - State: review 32 (empty request, nothing merged) — do review-31 queue: step 1 DONE+committed (builds rc=0, round 136); step 2 CPU sanity (.cpuper.sh) never archived (round-137 bg run died with session; no cpu-per-r1.out, no kk/runs/cpu_per_r1). Re-running THIS round as bg task b7bj8w307 (`bash TestCases/kokkos_nemo/.cpuper.sh`, strict binary, ENABLE_KOKKOS=NO, 21 iters, translation x=+2.0) -> kk/logs/cpu-per-r1.out + kk/runs/cpu_per_r1/log.txt.
  - PASS criteria: rc=0, 21 residual rows, no periodic/matching error. If preprocessing fails on matching vertices: flip invper_ausm.cfg translation 2.0 -> -2.0, rerun as r2, note it.
  - NEXT on pass: archive cpu-per-r1.log + log.txt -> validation_logs/2026-11-27_4.6-Periodic/ + INDEX row + commit; then step 3 G1 strict `.g1per.sh` -> g1-per-r1.log. Limiter step 6 noise floor (.nflim.sh) still owed (run + archive, no merge request).
- 2026-11-27T04:44:47-05:00 [loop] round 150 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261127_040551_150.log
