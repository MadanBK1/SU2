# KOKKOS_PROGRESS — inter-session memory (branch su2kokkos_qwenlocal_agent)

## Item status (mirrors section 4 of LOCAL_AGENT_TASK.md)
- [x] 4.1 Script env-var paths (SU2_ROOT/MESH_DIR/RUN_DIR) — done 2026-11-24
- [ ] 4.2 Batch A gates: G2, G3 (vs cd2cd46 binaries), G6 — NEXT
- [ ] 4.3 CUDA V100 build + GPU run (G5, G7)
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
- 2026-11-24 round 1 (autosave commit 8f5f57d23a): converted all TestCases/kokkos_nemo/*.sh to ${SU2_ROOT|MESH_DIR|RUN_DIR:-/home/claude/...}; missed ab_variants.sh:80; hit max turns before committing.
- 2026-11-24 round 2 (this session):
  - Verified on su2kokkos_qwenlocal_agent, clean tree. `bash -n` on all 8 kokkos_nemo scripts: OK.
  - Fixed last hard-coded path: ab_variants.sh:80 → `d=$RUN/ab${NP:+_np$NP}/$n; ... ln -sf $M/*.su2 .`
  - Decision (recorded per loop rules): item 4.1 is test-harness only (no C++, no binary change), so the section-3 numeric gates do not apply; committed after syntax + env-default verification, as the item itself prescribes ("then commit").
  - Ticked 4.1 in LOCAL_AGENT_TASK.md. Committed d2e840e8b5 (local). Push BLOCKED (see below) — record in my commit message saying "pushed" is wrong; push never succeeded.
- NEXT — round 3, item 4.2 (Batch A gates; Batch A = commit 35b6f39544 WIP):
  1. env in every shell: `export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc SU2_ROOT=$PWD MESH_DIR=$PWD/../kk/meshes RUN_DIR=$PWD/../kk/runs`
  2. Build baseline cd2cd46 binaries for G3 first (avoid `reset --hard`; use `git archive cd2cd46` into ../kk/cd2cd46, or a throwaway clone of branch @ cd2cd46 in ../kk). Build serial+strict(+strict-mpi) there or save the SU2_CFD binaries to ../kk/bin_pre/ (name e.g. SU2_CFD_serial_e1aafef to match the OLD default) and export OLD=... when running G3.
  3. `./meson.py` or system meson + section-2 COMMON flags; ninja -j4 SU2_CFD and UnitTests/nemo_gas_kokkos_validation; redirect to logs, tail -30 only.
  4. Run gates in order: G1 (STRICT: check.sh REF_MODE=HOST RES_TOL=0 FIELD_TOL=0, ab_variants.sh, ab_resident.sh), G2 (PROD: check.sh super_cat thermalbath + check_noisefloor.sh invwedge_ausm & super_cat), G3 (check_no_identity.sh + check_nonnemo_kokkos.sh vs cd2cd46 binaries), G4 (UnitTests/nemo_gas_kokkos_validation 5000 1e-10 in both Serial builds), G6 (NP=2 strict-mpi check.sh).
  5. All pass → non-WIP commit "Batch A gates complete" (+ any fixes), tick 4.2 here and in the task file, push. Failures: fix within the item, re-run failing gate, log pass/fail numbers here.
  - If a build is blocked (e.g. subproject fetch): try section-2 fallback (Eigen into externals/eigen, touch su2preconfig.timestamp, system meson). Still blocked → mark `BLOCKED 4.2: <reason>` and move to 4.3/4.4 (4.3 also needs a build; 4.4 is code work).
BLOCKED push-to-origin: egress to github.com:443 denied by sandbox policy ("user denied", 2026-11-24 round 2); local_agent_loop.sh is in local-only mode ("Do NOT run git push", "user pushes manually for review"). Do not attempt push in future rounds; commit locally, user pushes.

LOOP HINT: round 3 = item 4.2 per NEXT steps above; builds via ./meson.py or system meson (meson NOT on PATH), ninja -j4, log to file, tail -30; commit LOCALLY before ending the session — do not push (local-only loop mode).

- 2026-11-24T02:32:12-05:00 [loop] round 1 ended rc=1 with progress. Log: /home/mpr7366/software/AgenticTest/agent_logs/round_20261124_012128_1.log
- 2026-11-24 round 3 (this session, item 4.2 start):
  - meson setup failed: `meson.py` preconfigure tries `git submodule update externals/codi` → sandbox refuses `.git/config` write ("Device or resource busy"); `externals/meson` is EMPTY (no vendored meson); `pip3 install meson` → pypi.org:443 denied (user denied).
  - FOUND vendored meson 1.8.2 at /home/mpr7366/SU2/externals/meson (exact match to meson.build `>=1.8.2`). Works: `PYTHONPATH=/home/mpr7366/SU2/externals/meson python3 -m mesonbuild.mesonmain setup ...` from repo root (skips preconfigure entirely; submodules already extracted in this clone).
  - Created driver ../kk/build.sh (setup|ninja for serial|strict|strict-mpi; logs to ../kk/logs/*.log; exports SU2_ROOT/MESH_DIR/RUN_DIR/GXX/GCC).
  - NEXT: `bash ../kk/build.sh setup strict`, then `setup serial`, then `ninja strict 4` (background), verify SU2_CFD + nemo_gas_kokkos_validation binaries, then G1..G4 per LOOP HINT.
  - externals/eigen and externals/mel were EMPTY in this clone (submodule dirs without .git/modules); copied from /home/mpr7366/SU2/externals/{eigen,mel}, removed the stray `.git` file each cp dragged (it pointed at missing .git/modules/* and broke `git status` until removed). CLI11/utils present. codi/FADO/medi still empty — OK so long as autodiff/directdiff stay disabled; fill from ~/SU2 if a TU needs them.
  - meson setup: strict + serial + strict-mpi all rc=0 (vendored meson 1.8.2 + touched su2preconfig.timestamp).
  - Compiles: strict + serial in background (ninja -j4, logs ../kk/logs/ninja-*.log). Serial failed once on mel.hpp before the mel copy; restarted.
  - strict build DONE (ninja rc=0): build-kokkos-strict/{SU2_CFD/src/SU2_CFD, UnitTests/nemo_gas_kokkos_validation} exist.
  - G4 strict: PASS (bitwise transport props, AIR-7 GUPTA-YOS 1250 states).
  - build.sh now has `gate` mode: gate g1 | g1-ab | g2 | g4strict | g4serial | g6 (logs to ../kk/logs).
  - G1 (strict, REF_MODE=HOST, RES_TOL=0, FIELD_TOL=0) launched bg b17dw21fv → ../kk/logs/g1-strict.log. Serial build b11xeexsk still compiling.
  - NEXT after G1: g1-ab (ab_variants + ab_resident), G2 (needs serial build done), G4 serial, G3 (needs cd2cd46 baseline binaries — see NEXT above), G6 (needs strict-mpi ninja build: `bash ../kk/build.sh ninja strict-mpi 4`).
- 2026-11-24 round 4 (this session, item 4.2 continue):
  - State check: strict+serial SU2_CFD & UnitTests built (Nov 24 02:51); strict-mpi NOT built; g1-strict.log absent (G1 never finished last round); g4-strict.log PASS (on disk).
  - This round: G1 strict, g1-ab, G2, g4serial; strict-mpi ninja for G6; G3 needs cd2cd46 baseline.
