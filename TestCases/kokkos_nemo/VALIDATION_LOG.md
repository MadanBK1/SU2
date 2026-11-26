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

## Decisions
- 2026-11-26 (Madan): IONIZATION= YES with Roe / LAX / MSW stays on the CPU (host). The CPU `CNEMOGas::ComputedPdU`
  with ionization reads stale members and an uninitialised dPdU entry, so its result is undefined; not ported, not patched.

From 2026-11-26 every merge adds a row here citing the archived log paths under `validation_logs/`.
