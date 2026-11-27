# NEMO Kokkos port — validation record (su2kokkos)

One row per change merged into `su2kokkos`, oldest first. The raw logs of every run (passes and failures) are committed on
branch `su2kokkos_qwenlocal_agent` under `validation_logs/` (index: `validation_logs/INDEX.md`); detailed reviews are in
that branch's `REVIEW_NOTES.md`.

Gates: **G1** strict Kokkos Serial build (no fast-math, `-ffp-contract=off`), `REF_MODE=HOST RES_TOL=0 FIELD_TOL=0
check.sh`, bitwise residuals + restart fields; **A/B** `SU2_KOKKOS_STAGE_VERIFY=1` per-stage device-vs-host compare;
**NF** noise floor (`check_noisefloor.sh`, ratio ≤ 10); **G6** 2 MPI ranks bitwise; **G7** CUDA V100 run (noise floor).

## Before the log archive (2026-11-19 .. 2026-11-24)
Steps 5–10 and Batch A were validated by earlier reviewer sessions; results are recorded in `NEMO_KOKKOS_PORT_PLAN.md`
("Validation" paragraphs) and in the commit messages. No raw logs were archived for these.

| date | su2kokkos commit | change | validation (as recorded) |
|---|---|---|---|
| 2026-11-19 | e87cfe6 | Step 5: device upwind AUSM / AUSM+M residual + Jacobian | strict A/B 13 variants bitwise (plan) |
| 2026-11-19 | e1aafef | Step 7: device source terms + Euler BCs | G1 strict bitwise (plan) |
| 2026-11-19 | 86a1354, 79f5853, 389cbea | Step 6: Kokkos linear-solver handoff, host-read hooks, device-resident state | ab_resident.sh bitwise (plan) |
| 2026-11-20 | e7f2986, 7ec2df7 | Step 8: device transport + NS device-resident | thermo driver strict bitwise; ab_variants/ab_resident 13 NS variants bitwise; G1 5 cases bitwise (plan) |
| 2026-11-20 | 0ba576a, 7f2e722, 83a6d2a | Step 10: MPI halo exchange, edge colouring | G6 2-rank bitwise (plan) |
| 2026-11-20 | cd2cd46 | CUDA compile check + nvcc fixes | G5 compile |
| 2026-11-23/24 | 35b6f39, 982c4ea | Batch A (AUSM+M sensor, CFL adapt, axisymmetric, supersonic BCs, RK/dual time) + MPI fix | G2/G3/G6 by local agent (agent KOKKOS_PROGRESS.md) |

## Reviewed merges (local agent runs the gates)
| date | su2kokkos commit | item | validated agent code | validation (agent logs) |
|---|---|---|---|---|
| 2026-11-25 | 20eabd2, f284c05 | 4.4-LAX device LAX-FRIEDRICH | f5fd2a4 | G1 invwedge_lax + default 4 cases bitwise (g1lax-strict.log, g1-default-strict.log) |
| 2026-11-25 | d63a151 | 4.4-Roe interior | e744eb0 | G1 invwedge_roe bitwise, 6-case regression, A/B bitwise |
| 2026-11-25 | d761e92 | 4.4-Roe-BC + LAX residency | agent 4a613f3 | G1 invwedge_lax + invwedge_roe bitwise, 6-case regression, A/B 0 non-bitwise |
| 2026-11-25 | effdde0 | 4.4-MSW | cce61ae | G1 invwedge_msw bitwise, 8-case regression, A/B 0, NF PASS |
| 2026-11-25 | 1e8c2e1 | 4.4-AUSM+-up2 (explicit) | agent 24b4086 | G1 bitwise (case + default + all invwedge), A/B bitwise, NF PASS |
| 2026-11-25 | 33bd882 | 4.5 Smoluchowski–Maxwell slip wall | 0d7a7b9 | G1 smol_maxwell bitwise, default + invall bitwise, A/B bitwise, NF 0.61/0.50 |
| 2026-11-25 | 986dc2b | 4.5 viscous supersonic inlet | d2fee8f | G1 supin_visc bitwise, default + invall 7/7 bitwise, A/B bitwise, NF 0.50/0.72 (g1-supin-r59.log, g1-regress-r59.log, g1-invall-r59.log, sv-supin-r61.log, nf-supin-r61.log) |
| 2026-11-26 | 91f8df0 | 4.6 device LU_SGS preconditioner apply | 02676f8 (reviewed head 065f855) | G1 invwedge_lusgs bitwise, super_cat+LU_SGS bitwise, default + invall 4/4 bitwise, A/B compares=106 non-bitwise=0, NF 0.71/0.95 (backfill_2026-11-25/g1-lusgs-r64.log, g1-lusgs-visc-r64.log, g1-regress-r68.log, g1-invall-r68.log, sv-lusgs-r68.log, nf-lusgs-r68.log) |
| 2026-11-26 | 800d75f | 4.6-MG1 multigrid: fine level device-resident, coarse levels host | 413b3a0 (reviewed head db03759) | G1 strict invwedge_mg bitwise (wall NO 63.84s / YES 53.86s, MG transfers=11), MG_RESIDENT=0 control bitwise, mgpost/mgw/mgcs + super_cat MGLEVEL=1 bitwise, poison bitwise, NF 0.71/1.00, default + lusgs/roe regressions bitwise (validation_logs/2026-11-26_4.6-MG/g1-mg-r1.log, g1-mgoff-r1.log, g1-mgpost-r1.log, g1-mgw-r1.log, g1-mgcs-r1.log, g1-mgvisc-r1.log, poison-mg-r1.log, nf-mg-r1.log, g1-regress-r1.log, g1-regress2-r1.log) |
| 2026-11-26 | 85e6f71 | 4.6-MG2 multigrid: coarse levels device-resident + device transfer ops | bbcbd98 (reviewed head 5c4dcce) | G1 strict invwedge_mg bitwise (wall NO 64.55s / YES 53.28s, MG level 1+2 device-resident ON, device transfers=66), MG_DEVICE=0 control bitwise, mgpost/mgw/mgcs + super_cat MGLEVEL=1 bitwise (coarse OFF viscous), poison bitwise, NF 1.12/1.00, default 4/4 + lusgs/roe 2/2 bitwise (2026-11-26_4.6-MG2/g1-mg2-r1.log, g1-mg2off-r1.log, g1-mg2{post,w,cs,visc}-r1.log, poison-mg2-r1.log, nf-mg2-r1.log, g1-mg2regress{,2}-r1.log) |

## GPU runs (G7, CUDA V100, run by Madan)
| date | su2kokkos commit | cases | result | logs |
|---|---|---|---|---|
| 2026-11-26 | 8277424 (+ d9e0b2a noise-floor cases) | super_cat, thermalbath, invwedge_ausm, invwedge_roe, invwedge_lusgs | device-resident ON for all. super_cat 2.3e-15 / thermalbath 4.4e-16 vs CPU. Noise floor (K=10): invwedge_ausm 1.21 / 1.53 PASS; invwedge_lusgs 1.04 / 0.77 PASS; invwedge_roe ITER 11 0.59 PASS, ITER 100 43.8 FAIL, judged chaotic: the case does not converge (rms stalls ~-1.2, ~92k non-physical reconstructed states; CPU strict-vs-prod also O(1) apart in 82% of points). STAGE_VERIFY over all 100 iterations: every per-column stage diff <= 2e-13 (the "max rel diff" 0.125 / 0.25 come from entries near zero). Wall (11 it, 25.8k points, incl. start-up): super_cat 11.50 -> 6.49 s, invwedge_ausm 10.99 -> 7.33 s, invwedge_roe 12.22 -> 8.35 s, invwedge_lusgs 12.83 -> 11.49 s (LU_SGS apply ~0.1 s each: perf item). | ~/gpu_runs on roam148-045 -> agent branch validation_logs/2026-11-26_G7-V100/ |

## Decisions
- 2026-11-26 (Madan): IONIZATION= YES with Roe / LAX / MSW stays on the CPU (host). The CPU `CNEMOGas::ComputedPdU`
  with ionization reads stale members and an uninitialised dPdU entry, so its result is undefined; not ported, not patched.

From 2026-11-26 every merge adds a row here citing the archived log paths under `validation_logs/`.
| 2026-11-26 | (this merge) | 6b.a-DynGrid: dynamic grid device path (RefreshDynamicGeometry, grid-velocity dt/eigenvalue, dual-time GCL; refresh in CommonPreprocessing for NS) — agent code bf2ff4ad8f, reviewed at 4e49fe63ae | strict REF_MODE=HOST RES_TOL=0 FIELD_TOL=0: invwedge_dyn, dyn1st, dyntr, DYNGRID=0 control, poison inviscid+viscous, super_cat dynvisc (r2), default list + invwedge_mg/lusgs regressions all PASS bitwise; noise floor invwedge_dyn K=10 ratio 0.19 PASS | agent branch validation_logs/2026-11-26_6b.a-DynGrid/ (g1-dynvisc-r2, g1-dyn-r2, poison-dynvisc-r2, g1-dynregress-r2, g1-dynregress2-r2, nf-dyn-r1, build-dyn-r2; r1 incl. failing dynvisc) |
| 2026-11-26 | (this merge) | 6b.b-TwallIn (PARTIAL): per-vertex python-custom wall temperature on NEMO isothermal walls (CPU + device) — agent code 89112d6457, reviewed at d02d49356a | strict REF_MODE=HOST RES_TOL=0 FIELD_TOL=0: super_cat MARKER_PYTHON_CUSTOM identity, default list + supin_visc + smol_maxwell (6/6), CPU no-op + cross-tag cmp — all PASS bitwise; build strict+serial OK. NOT yet run: pywrapper-driven Twall (steps 5-6, swig missing) | validation_logs/2026-11-26_6b.b-TwallIn/{build-tw-r1,build-tw-serial-r1,g1-tw-ident-r1,g1-tw-regress-r1,cpu-noop-tw-r1,cpu-noop-tw-cmp-r1}.log (agent branch) |
| 2026-11-26 | (this merge) | 4.8-ILUTeam: team-per-row device ILU(0) build `KokkosILU0BuildTeam` (default ON; `SU2_KOKKOS_ILU_BUILD=row` fallback) — agent code c362c12d78, reviewed at 3468ddec5a | strict REF_MODE=HOST RES_TOL=0 FIELD_TOL=0: super_cat team (builds=11) + PRECOND_VERIFY A/B compares=11 non-bitwise=0 + row fallback + default list + invwedge_mg/invwedge_dyn/supin_visc (7/7) all PASS bitwise; noise floor super_cat K=10 0.39/0.77 PASS; build strict+serial OK. V100 G7 team-vs-row timing NOT yet run | validation_logs/2026-11-26_4.8-ILUTeam/{build-ilu-r1,build-ilu-serial-r1,g1-ilu-team-r1,ab-ilu-team-r1,g1-ilu-row-r1,g1-ilu-regress-r1,nf-ilu-r1}.log (agent branch) |
| 2026-11-26 | (this merge) | 4.8-LUSGSHyb: hybrid device LU_SGS apply (per-level launches for wide levels, single-team segments for narrow; `SU2_KOKKOS_LUSGS_WIDE`, `SU2_KOKKOS_LUSGS_APPLY=single` fallback) — agent code 7932194a9a, reviewed at ca9282c995 | strict REF_MODE=HOST RES_TOL=0 FIELD_TOL=0: invwedge_lusgs hybrid default / WIDE=1 / single-team bitwise, PRECOND_VERIFY A/B compares=106 non-bitwise=0, super_cat LU_SGS WIDE=1 bitwise, 7-case regressions bitwise, noise floor K=10 ratios 0.78/0.86; build strict+serial OK — all PASS. V100 G7 timing OPEN | validation_logs/2026-11-26_4.8-LUSGSHyb/ (agent branch; 9 INDEX rows) |
| 2026-11-27 | (this merge) | 4.6-Linelet: device LINELET preconditioner apply (Jacobi points + one thread per linelet, CPU statement order; NEMO dispatch) — agent code bb0ab70, reviewed at 34636cb741 | strict REF_MODE=HOST RES_TOL=0 FIELD_TOL=0: G1 super_cat LINELET + invwedge_lusgs LINELET bitwise, PRECOND_VERIFY A/B compares (build 11, apply 76) non-bitwise=0, regressions 8/8 bitwise, noise floor K=10 0.33/0.75 PASS; build strict+serial OK. V100 G7 timing open | validation_logs/2026-11-26_4.6-Linelet/ (build-linelet-r1, g1-linelet-regress-r1, nf-linelet-r1, perf-linelet-r1, …; INDEX rows 4.6-Linelet) on su2kokkos_qwenlocal_agent |
| 2026-11-27 | (this merge) | 4.6-Limiters: device SHARP_EDGES / WALL_DISTANCE slope limiters (host-evaluated geometric factor uploaded once) + ResSplit: CNEMOKokkosResidual.cpp split into 5 TUs (nvcc compile time) — agent code 332d230 + c2ba950 (unchanged to 1e7f5b2), reviewed at 1e7f5b2ebb | strict REF_MODE=HOST RES_TOL=0 FIELD_TOL=0: G1 invwedge_ausm_sharp + super_cat_walld PASS bitwise; A/B SU2_KOKKOS_STAGE_VERIFY Limiter compares=33 non-bitwise=0 (both); regressions 8/8 bitwise on split build (invwedge_ausm thermalbath super_cat axi_visccone invwedge_lusgs invwedge_mg invwedge_dyn supin_visc); builds strict+serial rc=0; CUDA sm_70 build rc=0 (~4.4 min -j10). Noise floor (nf-lim-r1) not yet run. | agent validation_logs/2026-11-27_4.6-Limiters/ (g1-lim-sharp-r1, g1-lim-walld-r1, ab-lim-sharp-r1, ab-lim-walld-r1, build-lim-*), validation_logs/2026-11-27_ResSplit/ (build-split-r1, build-split-serial-r1, build-split-cuda-r1, g1-split-regress-r1) |
