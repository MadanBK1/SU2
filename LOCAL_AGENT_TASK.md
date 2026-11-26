# Task file for a local coding agent (branch su2kokkos_qwenlocal_agent)

You are continuing a validated port of SU2's two-temperature NEMO solver to Kokkos. The goal is that every compressible NEMO run goes end to end on the GPU. Read `NEMO_KOKKOS_PORT_PLAN.md` first; it has the design, the feature coverage table, the validation policy and the CUDA notes.

## ROLE (from 2026-11-25, Madan's decision) — read this first
- The Claude reviewer does only the **heavy work**: new solver code (kernels, dispatch, ports) and hard bugs. Its usage is limited, so ask it only when needed.
- **You do everything else:** build (strict and serial), run the simulations and gates from the newest REVIEW_NOTES.md queue, analyze logs (pass/fail numbers, device/host stage lines, `[KOKKOS_NEMO_PERF]`, StageCompare), summarize in KOKKOS_PROGRESS.md, fix compile errors yourself (smallest change), make small mechanical edits (config copies, check.sh cases, ticking boxes).
- Do **not** write new kernels/ports or restructure code.
- Request a review (last line of KOKKOS_PROGRESS.md) ONLY in these forms:
  - `READY FOR REVIEW: MERGE: <item> — <gate results>`: the strict gate (BIN=strict REF_MODE=HOST RES_TOL=0 FIELD_TOL=0) passed for the new code + default list.
  - `READY FOR REVIEW: HEAVY: <item>: <what code or which bug>`: you need new code, or a failure you could not localize after 3 sessions (paste ≤ 20 StageCompare/log lines).
- Finish the whole queue before requesting. Routine results do not need a review.

## VALIDATION LOGS (standing rule, Madan 2026-11-26) — keep every run, start to end
- Every gate / A/B / noise-floor / perf / GPU run you do: copy its log into the repo at
  `validation_logs/<YYYY-MM-DD>_<item>/<logname>.log` (e.g. `validation_logs/2026-11-26_4.6-LUSGS/g1-lusgs-r64.log`),
  together with the YES/NO `log.txt` of each case (`chk_<case>_{NO,YES}/log.txt` -> `<logname>__<case>_{NO,YES}.log`).
  Do NOT copy meshes, restart/solution files or binaries. gzip any single log > 1 MB.
- Add one line per run to `validation_logs/INDEX.md`:
  `| date | item | gate | exact command | binary (strict/serial/cuda) + git sha | result (PASS/FAIL, max|dres|, field diff, ratio) | log path |`.
- Failed runs are kept too (mark FAIL); never delete or overwrite a log — new run, new file name.
- Commit the logs + INDEX.md in the same commit as the progress note for that run (`git add validation_logs`).
  `*.log` is gitignored outside validation_logs/ (exception added to .gitignore 2026-11-26); after committing, CHECK
  `git ls-tree -r --name-only HEAD validation_logs/<dir>/ | wc -l` matches the files you copied (use `git add -f` if not).
  A `MERGE:` request is only valid if every log it cites is committed under validation_logs/.

## 0. Restart protocol (do this every session)
1. `cd` to the clone, run `git checkout su2kokkos_qwenlocal_agent`, then `git status` and `git log --oneline -5`. Stop if you are not on that branch.
2. Read `KOKKOS_PROGRESS.md` (create it if missing). It is the only memory you have between sessions.
3. Read the newest section of `REVIEW_NOTES.md` (from the Claude reviewer; never edit that file). Do its **Agent: do next** items (tests/analysis), then write `READY FOR REVIEW` with the results (see ROLE above).
4. After every meaningful step, append to `KOKKOS_PROGRESS.md`: date, what you did, the exact commands, pass/fail numbers, and the next step.
5. Work on **one item per session**. When the item is committed, or your context is getting long, update `KOKKOS_PROGRESS.md`, commit it, and tell the user: "restart me with the same prompt". Never rely on remembering earlier chat.

### Unattended loop mode
The script `TestCases/kokkos_nemo/local_agent_loop.sh` restarts you in a fresh session after every round. In that mode:
- **Each session:** do one bounded chunk, then save progress and commit locally (no push) before ending.
- **When everything is finished:** if every item in section 4 is checked, write `ALL ITEMS DONE` as the last line of `KOKKOS_PROGRESS.md`.
- **Never wait for a person.** If an item is blocked (credentials, GPU busy, a permission refused, a missing tool), mark it `BLOCKED <item>: <reason>` in `KOKKOS_PROGRESS.md` and continue with the next unchecked item. Retry blocked items once everything else is done or blocked. When something is a decision for the user, pick the conservative option and record it.
- **Review handshake:** when an item or a compiled-and-tested chunk is done, write `READY FOR REVIEW: <what>` as the last line of `KOKKOS_PROGRESS.md` and end the session. The loop pushes and waits for the Claude review in `REVIEW_NOTES.md`, then you continue with its **Agent: do next** items. Never edit `REVIEW_NOTES.md`, `REVIEW_REQUEST` or `REVIEW_DONE`.
- **Loop hints:** obey the most recent `LOOP HINT:` line in `KOKKOS_PROGRESS.md`.
- **Stopping:** the loop stops only on `ALL ITEMS DONE` or when the user creates the file `STOP_AGENT`.

### Context budget (small local model)
- **Big files:** never read a whole source file (`CNEMOEulerSolver.cpp` alone is thousands of lines). Use `grep -n`, then read 50–80 lines around the hit.
- **Long output:** redirect builds and tests to log files and show only `tail -30`. Never print full diffs; use `git diff --stat` or a single file through `head`.
- **Save early:** write progress to `KOKKOS_PROGRESS.md` and commit early in each session.

## 1. Rules
- **Folder boundary (hard rule):** only create, modify or delete files inside `~/software/AgenticTest/` (the SU2 clone and `kk/`).
  - Never write, move or delete anything outside it, especially `/home/mpr7366/SU2` (the user's other SU2 tree), `~/.bashrc` and other home dotfiles, and system paths.
  - No `sudo`, and no `rm -rf` on a path containing `..`, `~` on its own, or `/`.
  - Reading outside is fine: compilers, `/opt/rh`, CUDA.
- **Branch:** work only on `su2kokkos_qwenlocal_agent`. **Never commit to or push `su2kokkos`** — that is the reviewed branch, and Claude merges from yours after review. Also never touch `master` or other branches, and never use `reset --hard`, `clean -fd`, force-push or rebase.
- **Commits:** commit **locally only**, and only after the gates in section 3 pass. **Never run `git push`**; the user pushes for review.
- **Untested work:** if gates could not be run, the commit message must start with `WIP` and list which gates were not run.
- **Never commit:** build directories, `subprojects/.wraplock`, or the `subprojects/CoolProp` submodule pointer.
- **CPU path:** `ENABLE_KOKKOS= NO` behaviour must stay bit-identical.
- **CPU quirks:** reproduce CPU behaviour exactly, quirks included (listed in the plan). Don't silently "fix" physics; record new quirks in the plan.
- **Device code:** use plain KOKKOS_INLINE_FUNCTION / KOKKOS_LAMBDA. Keep to CPU statement order, with no compiler-specific arithmetic tricks.
- **nvcc lambda rules:**
  - no lambdas inside private/protected member functions;
  - no lambdas capturing function-local types;
  - no `std::` containers, strings or virtuals in device code;
  - no implicit `this` capture.

## 2. Setup on roam148-045 (V100, Rocky, GCC 12, CUDA 12.9)
```bash
cd ~/software/AgenticTest
git clone -b su2kokkos_qwenlocal_agent https://github.com/MadanBK1/SU2.git && cd SU2
mkdir -p ~/software/AgenticTest/kk/meshes ~/software/AgenticTest/kk/runs
for f in invwedge/invwedge.su2 thermalbath/finitechemistry/5x5Mesh.su2 visc_wedge/viscwedge.su2; do
  curl -sSfL -o ~/software/AgenticTest/kk/meshes/$(basename $f) https://raw.githubusercontent.com/su2code/TestCases/master/nonequilibrium/$f; done
export GXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ GCC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
```
**First code task (item 4.1):** the scripts in `TestCases/kokkos_nemo/*.sh` hard-code cloud paths: `/home/claude/su2`, `/home/claude/meshes` and `/home/claude/runs`. Replace them with env vars that default to the old paths:

| Env var | Replaces | Set it to |
|---|---|---|
| `SU2_ROOT` | `/home/claude/su2` | your clone |
| `MESH_DIR` | `/home/claude/meshes` | `~/software/AgenticTest/kk/meshes` |
| `RUN_DIR` | `/home/claude/runs` | `~/software/AgenticTest/kk/runs` |

Export them in every shell.

### Builds (meson + ninja; NOT cmake --build)
`./meson.py` normally fetches submodules. If a submodule fetch fails (for example gitlab eigen), install the Eigen headers, copy them into `externals/eigen`, `touch su2preconfig.timestamp`, and use the system `meson`.

```bash
COMMON="-Denable-kokkos=true -Denable-autodiff=false -Denable-directdiff=false -Denable-tecio=false -Denable-cgns=false -Dbuildtype=release"
# production Serial (fast-math)
CXX=$GXX CC=$GCC meson setup build-kokkos-serial $COMMON -Dkokkos-backend=serial -Dkokkos-arch=none -Dwith-mpi=disabled
# strict IEEE Serial (exactness reference)
CXX=$GXX CC=$GCC meson setup build-kokkos-strict $COMMON -Dkokkos-backend=serial -Dkokkos-arch=none -Dwith-mpi=disabled -Denable-fast-math=false -Dcpp_args=-ffp-contract=off
# strict MPI (2-rank gate)
CXX=mpicxx CC=mpicc meson setup build-kokkos-strict-mpi $COMMON -Dkokkos-backend=serial -Dkokkos-arch=none -Dwith-mpi=enabled -Denable-fast-math=false -Dcpp_args=-ffp-contract=off
ninja -C build-kokkos-serial -j4 SU2_CFD/src/SU2_CFD UnitTests/nemo_gas_kokkos_validation   # same for the others
```

### CUDA V100 build
The `NVCC_WRAPPER_DEFAULT_COMPILER` line below is required. Without it, `subprojects/Kokkos/bin/kokkos_launch_compiler` line 107 sets the dispatcher as nvcc's host compiler, and the build recurses and fails with no message.
```bash
export SU2_KOKKOS_HOST_COMPILER=$GXX SU2_KOKKOS_CUDA_ARCH=sm_70 NVCC_WRAPPER_DEFAULT_COMPILER=$GXX
CXX=$PWD/meson_scripts/kokkos_compiler_dispatcher CC=$GCC meson setup build-kokkos-cuda $COMMON \
  -Dkokkos-backend=cuda -Dkokkos-arch=VOLTA70 -Denable-cuda=false -Dwith-mpi=disabled
ninja -C build-kokkos-cuda -j4 SU2_CFD/src/SU2_CFD      # export the 3 vars in every shell that runs ninja
```
- **If a TU fails:** run it at `-j1 -v`, capture the exit code, and run `bash -x meson_scripts/kokkos_compiler_dispatcher ...`.
- **GPU memory:** check with `nvidia-smi`. The local LLM (ollama) uses about 17+ GB of the V100. That is fine for correctness runs, but **unload it (`ollama stop <model>`) before any timing/benchmark run**.

## 3. Validation gates (all must pass before a non-WIP commit)
Set `S=$SU2_ROOT/TestCases/kokkos_nemo`, `STRICT=$SU2_ROOT/build-kokkos-strict/SU2_CFD/src/SU2_CFD` and `PROD=$SU2_ROOT/build-kokkos-serial/SU2_CFD/src/SU2_CFD`.

| # | Gate | Command | Pass if |
|---|---|---|---|
| G1 | Strict exactness | `BIN=$STRICT REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 $S/check.sh`, plus `$S/ab_variants.sh` and `$S/ab_resident.sh` (strict). Add a variant for every feature you port. | OVERALL: PASS, bitwise. |
| G2 | Production | `BIN=$PROD PHYS_SCALE=1 RES_NOISE_ORDERS=10 $S/check.sh super_cat thermalbath`, plus `CASE=invwedge_ausm $S/check_noisefloor.sh` and `CASE=super_cat $S/check_noisefloor.sh` | PASS; noise-floor ratio ≤ 10. |
| G3 | CPU identity | `$S/check_no_identity.sh` and `$S/check_nonnemo_kokkos.sh`, both against binaries you built from the commit *before* your change (save them first) | Bit-identical. |
| G4 | Thermo unit driver | `<build>/UnitTests/nemo_gas_kokkos_validation 5000 1e-10` | PASS in both Serial builds. |
| G5 | Device-code check | Build `build-kokkos-cuda` (real nvcc, better than `cuda_compile_check.sh`). | Compiles clean. |
| G6 | MPI | `NP=2 BIN=<strict-mpi SU2_CFD> REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 $S/check.sh` | Bitwise. |
| G7 | GPU (once the CUDA build exists) | `BIN=<cuda SU2_CFD> PHYS_SCALE=1 RES_NOISE_ORDERS=10 $S/check.sh super_cat thermalbath`, plus `check_noisefloor.sh` with the CUDA binary | GPU results are NOT bitwise; use the noise-floor policy. Also check that the log shows `[KOKKOS_NEMO] device-resident state: ON`, and record `[KOKKOS_NEMO_PERF]` lines. Try `SU2_KOKKOS_NEMO_EDGE=atomic` vs `color`. |

## 4. Work items (in order; tick them here AND in KOKKOS_PROGRESS.md)

**END-TO-END GOAL (Madan, 2026-11-26): port EVERYTHING in the NEMO solver to the GPU, end to end (device-resident ON
for every compressible NEMO config). Items once marked "stays on host" are now in scope. Heavy queue for the reviewer
(agent: test each one as it lands; same gates):**
1. Roe boundary numerics + LAX residency (fc56517, testing now).
2. LAX multi-rank residency: device MAX_EIGENVALUE (Lambda) halo exchange.
3. MSW scheme.
4. AUSM+-up2 (deterministically initialised M_L/M_R; document the CPU quirk; G1 vs the patched CPU variant).
5. 4.5 BCs: subsonic BC_Inlet, Smoluchowski-Maxwell slip, catalytic heat-flux wall.
6. Ionization with Roe / LAX / MSW: **DECIDED (Madan, 2026-11-26): stays on the CPU (host).** The CPU ComputedPdU with
   ionization is undefined (review 13), so it is not ported and not patched. Do not work on it; the host fallback reason in
   Device_UpwindUnsupported / Device_CenteredUnsupported is the intended behaviour.
6b. **COUPLING READINESS — TOP PRIORITY after the LU-SGS merge (Madan 2026-11-26; AFOSR aerothermoelasticity center
   proposal, our part = GPU aerothermal solver coupled to other PIs' structures/experiments):**
   a. Dynamic / deforming grids on the device (DYNAMIC_GRID / mesh deformation each coupling step): geometry re-upload
      hook (normals, volumes, grid velocities) without leaving resident mode; gate: a deforming-wedge case, G1 strict.
   b. Coupling interface in resident mode: per-vertex wall temperature IN and per-vertex heat flux / pressure / traction
      OUT through the python wrapper (pysu2) without full-state host copies (only marker data moves); gate: a pysu2 script
      that sets T_wall per vertex each step, G1 strict vs ENABLE_KOKKOS=NO.
   c. NEMO conjugate heat transfer path (fluid–solid heat exchange, SU2 multizone CHT or preCICE-style exchange) —
      design after a and b.
   d. Continuum-breakdown output for hybrid CFD–DSMC (Madan 2026-11-26: our computational part includes rarefied flow
      with DSMC): device kernel for the gradient-length local Knudsen number Kn_GLL = (λ/Q)|∇Q| (Q = ρ, T, Tve, |u|),
      written as a volume/surface output field, to mark where DSMC (e.g. SPARTA) must replace continuum NEMO. Gate: G1 strict
      (output field bitwise vs ENABLE_KOKKOS=NO).
7. 4.6 solver options (remaining): LU-SGS, ILU(n>0) / GPU preconditioners, multigrid coarse levels, periodic markers,
   dynamic / deforming grids.
8. Host-touch removal: device residual norms, output / restart every N iterations, CUDA-aware MPI halos.
9. 4.7 turbulence coupling (SA / SST device kernels for NEMO).
10. Then 4.8 performance.
**DEFERRED (Madan 2026-11-26): items 11–12 are a later stage. Do NOT start them, and do not request HEAVY for them, until Madan says so.**
11. **SHAPE OPTIMIZATION, stage 1 (later stage):** finite-difference shape
   optimization with every primal on the GPU. A `TestCases/kokkos_nemo/shapeopt/` FFD wedge (and sphere-cone) case:
   DEFINITION_DV FFD_CONTROL_POINT_2D (≈4–8 DVs), OBJECTIVE_FUNCTION DRAG and TOTAL_HEATFLUX, `shape_optimization.py
   -g FINDIFF`, ENABLE_KOKKOS= YES, warm start from the previous design's restart. Developer: config + any GPU-path fixes
   for deformed meshes / objective outputs in resident mode. Agent: run it, gate each design's objective vs ENABLE_KOKKOS=NO
   (strict: bitwise; production: noise floor), archive all logs.
12. **SHAPE OPTIMIZATION, stage 2 (REQUIRED): adjoint gradients.** This SU2 has no NEMO adjoint. Plan: enable SU2's
   generic discrete adjoint (CDiscAdjSolver / CoDiPack) for NEMO_EULER / NEMO_NAVIER_STOKES on the CPU AD build
   (SU2_CFD_AD; native SU2TCLib gas only, Mutation++ not differentiable), then run hybrid: converged GPU primal ->
   restart -> CPU adjoint. Gate: adjoint gradient vs the stage-1 finite-difference gradient (agreement to FD accuracy).
   A GPU adjoint is a later research item, not in this queue.
- [x] **4.1 Script paths:** add env-var paths to the scripts (section 2), then commit.
- [x] **4.2 Finish Batch A** (commit `35b6f39`, WIP). Run G2, G3 (against `cd2cd46` binaries) and G6. Fix any failures, then commit "Batch A gates complete".
- [ ] **4.3 CUDA V100 build and GPU run:** run G5 and G7.
  - On a runtime failure, check `cudaLimitStackSize`; the big-stack kernels are `KernelBoundaryFlux` (~15.5 KB) and `KernelViscousEdges` (~12.4 KB).
  - Profile: `nsys profile -o nemo SU2_CFD run.cfg` for 20 iterations of invwedge_ausm.
  - Report the per-stage times, and wall time vs `ENABLE_KOKKOS=NO`.
- [x] **4.4 Device-side schemes:** ~~Roe~~ (done, G1 strict bitwise, merged d761e92), ~~MSW~~ (done, G1 strict bitwise + 8-case regression, merge pending), ~~LAX~~ (LAX done, G1 strict bitwise incl. stage-line guard, merged to su2kokkos 20eabd2) ~~AUSM+-up2 (explicit)~~ (done 2026-11-25, G1 strict bitwise + A/B bitwise + noisefloor PASS; implicit stays host — CPU quirk: Jacobian reads uninitialised M_L/M_R, documented).
- [x] **4.5 BCs:** Smoluchowski–Maxwell slip **done 2026-11-25** (G1 strict bitwise + regression + A/B bitwise + noisefloor PASS, merged to su2kokkos 33bd882); viscous **SUPERSONIC_INLET done 2026-11-25** (G1 strict bitwise supin_visc + regression + A/B bitwise + noisefloor PASS, merge requested `MERGE: 4.5-SupInVisc`); `BC_Inlet` **n/a** (CPU not operational — `SU2_MPI::Error` at entry); catalytic heat-flux wall **n/a** (CPU not operational, already reported as such).
- [ ] **4.6 Solver options:** ILU(n>0) device factorization, LU-SGS, the multigrid coarse levels (`MGLEVEL>0`), periodic markers, and dynamic/deforming grids (geometry re-upload hook).
  - **LU-SGS apply done 2026-11-25** (G1 strict invwedge_lusgs bitwise + viscous super_cat LU_SGS bitwise + default/targeted regressions bitwise + A/B `non-bitwise=0` + noisefloor PASS ratios 0.71/0.95; merge requested `MERGE: 4.6-LUSGS`). ILU(n>0) already device-resident (see plan matrix).
  - **Multigrid step 1 done 2026-11-26** (MGLEVEL>0: fine level device-resident via `SyncHostData(MG_TRANSFER)`; G1 strict invwedge_mg bitwise + mgoff control + 3 smoothing/cycle variants + viscous super_cat MGLEVEL=1 + poison=0 + regressions + noisefloor PASS ratios 0.71/1.00; merge requested `MERGE: 4.6-MG1`). Remaining: coarse levels resident (step 2, HEAVY), periodic markers, dynamic/deforming grids.
- [ ] **4.7 Turbulence coupling** for NEMO (SA/SST device kernels), or leave it host-only with a clear notice.
- [ ] **4.8 Performance** (only after 4.3): 32-bit indices, LayoutLeft for coalesced access, fusing Cons2Prim, transport and gradient, reducing the 81-entry atomics per Jacobian block, and a polynomial/block-Jacobi preconditioner option.
- [ ] **4.8b Speedup with accuracy:** this is a standing goal. Every performance change must keep G1 bitwise (strict) and the noise-floor ratio ≤ 10. Record wall time and `[KOKKOS_NEMO_PERF]` before and after. The user runs GPU timings (G7) from their own shell, so leave exact commands for them in `KOKKOS_PROGRESS.md`.
- [ ] **4.9** Keep the "Feature coverage matrix" in `NEMO_KOKKOS_PORT_PLAN.md` accurate after every item.

Out of scope on the device: Mutation++ (an external host-only library). Leave it on the host path with a notice.
