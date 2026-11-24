# Task file for a local coding agent (branch su2kokkos_qwenlocal_agent)

You are continuing a validated port of SU2's two-temperature NEMO solver to Kokkos. The goal is that every compressible NEMO run goes end to end on the GPU. Read `NEMO_KOKKOS_PORT_PLAN.md` first; it has the design, the feature coverage table, the validation policy and the CUDA notes.

## 0. Restart protocol (do this every session)
1. `cd` to the clone, run `git checkout su2kokkos_qwenlocal_agent`, then `git status` and `git log --oneline -5`. Stop if you are not on that branch.
2. Read `KOKKOS_PROGRESS.md` (create it if missing). It is the only memory you have between sessions.
3. Continue the first unchecked item in section 4.
4. After every meaningful step, append to `KOKKOS_PROGRESS.md`: date, what you did, the exact commands, pass/fail numbers, and the next step.
5. Work on **one item per session**. When the item is committed, or your context is getting long, update `KOKKOS_PROGRESS.md`, commit it, and tell the user: "restart me with the same prompt". Never rely on remembering earlier chat.

### Unattended loop mode
The script `TestCases/kokkos_nemo/local_agent_loop.sh` restarts you in a fresh session after every round. In that mode:
- **Each session:** do one bounded chunk, then save progress, commit and push before ending.
- **When everything is finished:** if every item in section 4 is checked, write `ALL ITEMS DONE` as the last line of `KOKKOS_PROGRESS.md`.
- **When you need a person:** if you are blocked (credentials, GPU busy, a permission refused, or a decision only the user can make), write `NEEDS HUMAN: <reason>` as the last line. The loop stops when it sees either marker.

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
- **Commits:** commit only after the gates in section 3 pass. Push with `git push origin su2kokkos_qwenlocal_agent`.
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
- [ ] **4.1 Script paths:** add env-var paths to the scripts (section 2), then commit.
- [ ] **4.2 Finish Batch A** (commit `35b6f39`, WIP). Run G2, G3 (against `cd2cd46` binaries) and G6. Fix any failures, then commit "Batch A gates complete".
- [ ] **4.3 CUDA V100 build and GPU run:** run G5 and G7.
  - On a runtime failure, check `cudaLimitStackSize`; the big-stack kernels are `KernelBoundaryFlux` (~15.5 KB) and `KernelViscousEdges` (~12.4 KB).
  - Profile: `nsys profile -o nemo SU2_CFD run.cfg` for 20 iterations of invwedge_ausm.
  - Report the per-stage times, and wall time vs `ENABLE_KOKKOS=NO`.
- [ ] **4.4 Device-side schemes:** Roe, MSW, LAX and AUSM+-up2. The CPU AUSM+-up2 Jacobian reads uninitialized members: document it, then match a deterministically initialized CPU variant.
- [ ] **4.5 BCs:** subsonic `BC_Inlet`, Smoluchowski–Maxwell slip (its gas-state-dependent `ComputeGasConstant`), and the catalytic heat-flux wall.
- [ ] **4.6 Solver options:** ILU(n>0) device factorization, LU-SGS, the multigrid coarse levels (`MGLEVEL>0`), periodic markers, and dynamic/deforming grids (geometry re-upload hook).
- [ ] **4.7 Turbulence coupling** for NEMO (SA/SST device kernels), or leave it host-only with a clear notice.
- [ ] **4.8 Performance** (only after 4.3): 32-bit indices, LayoutLeft for coalesced access, fusing Cons2Prim, transport and gradient, reducing the 81-entry atomics per Jacobian block, and a polynomial/block-Jacobi preconditioner option.
- [ ] **4.9** Keep the "Feature coverage matrix" in `NEMO_KOKKOS_PORT_PLAN.md` accurate after every item.

Out of scope on the device: Mutation++ (an external host-only library). Leave it on the host path with a notice.
