# Validation logs index

One row per committed log. New runs are added as they happen (see VALIDATION LOGS in LOCAL_AGENT_TASK.md).

| date | item | gate | exact command | binary + git sha | result | log path |
|---|---|---|---|---|---|---|
| 2026-11-25 | unknown | build/configure | unknown | serial, sha unknown | OK (build finished) | backfill_2026-11-25/build-serial-r52.log |
| 2026-11-25 | unknown | build/configure | unknown | serial, sha unknown | OK (build finished) | backfill_2026-11-25/build-serial-r55.log |
| 2026-11-25 | unknown | build/configure | unknown | serial, sha unknown | OK (already built) | backfill_2026-11-25/build-serial-r57.log |
| 2026-11-25 | unknown | build/configure | unknown | serial, sha unknown | OK (build finished) | backfill_2026-11-25/build-serial-r59.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (build finished) | backfill_2026-11-25/build-strict-r52.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (build finished) | backfill_2026-11-25/build-strict-r55.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (already built) | backfill_2026-11-25/build-strict-r57.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (build finished) | backfill_2026-11-25/build-strict-r59.log |
| 2026-11-25 | baseline gates (G2-G7) | A/B verify (G1 ab_resident, early) | unknown | strict, sha unknown | PASS (A/B, mismatching entries=0, non-bitwise=0) | backfill_2026-11-25/g1-abv.log |
| 2026-11-25 | 4.4-Roe-BC | G1 strict bitwise gate | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-bcroe-r26-driver.log |
| 2026-11-25 | 4.4-Roe-BC | G1 strict bitwise gate | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-bcroe-r26.log |
| 2026-11-25 | regressions | G1 strict bitwise gate | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-default-strict-r21.log |
| 2026-11-25 | regressions | G1 strict bitwise gate | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-default-strict.log |
| 2026-11-25 | regressions | G1 strict, invwedge_all list | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-invall-r52.log |
| 2026-11-25 | regressions | G1 strict, invwedge_all list | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-invall-r57.log |
| 2026-11-25 | regressions | G1 strict, invwedge_all list | timeout 4000 bash .g1run.sh ../kk/logs/g1-invall-r59.log invwedge_am invwedge_lax invwedge_roe invwedge_msw invwedge_ausmplusup2 super_cat smol_maxwell | strict, sha unknown | PASS | backfill_2026-11-25/g1-invall-r59.log |
| 2026-11-25 | regressions | G1 strict, invwedge_all list | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-invall-r61.log |
| 2026-11-25 | regressions | G1 strict, invwedge_all list | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-invall-r64.log |
| 2026-11-25 | regressions | G1 strict, invwedge_all list | timeout 4000 bash .g1run.sh ../kk/logs/g1-invall-r67.log invwedge_roe invwedge_msw smol_maxwell supin_visc | strict, sha unknown | PASS | backfill_2026-11-25/g1-invall-r67.log |
| 2026-11-25 | regressions | G1 strict, invwedge_all list | timeout 4000 bash .g1run.sh ../kk/logs/g1-invall-r68.log invwedge_roe invwedge_msw smol_maxwell supin_visc | strict, sha unknown | PASS | backfill_2026-11-25/g1-invall-r68.log |
| 2026-11-25 | 4.6-LUSGS | G1 strict bitwise gate | timeout 1500 bash .g1run.sh ../kk/logs/g1-lusgs-r64.log invwedge_lusgs | strict, sha unknown | PASS | backfill_2026-11-25/g1-lusgs-r64.log |
| 2026-11-25 | 4.6-LUSGS | G1 strict bitwise gate | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-lusgs-r64.tall.log |
| 2026-11-25 | 4.6-LUSGS | G1 strict bitwise gate | EXTRA_CFG="LINEAR_SOLVER_PREC= LU_SGS" RUN_TAG=_lusgs timeout 1500 bash .g1run.sh ../kk/logs/g1-lusgs-visc-r64.log super_cat | strict, sha unknown | PASS | backfill_2026-11-25/g1-lusgs-visc-r64.log |
| 2026-11-25 | 4.4-MSW | G1 strict bitwise gate | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-msw-r28.log |
| 2026-11-25 | partial-cat survey | G1 strict bitwise gate | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-partialcat-r52.log |
| 2026-11-25 | regressions | G1 strict, default regression list | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-regress-r25.log |
| 2026-11-25 | regressions | G1 strict, default regression list | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-regress-r26.log |
| 2026-11-25 | regressions | G1 strict, default regression list | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-regress-r28-driver.log |
| 2026-11-25 | regressions | G1 strict, default regression list | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-regress-r28.log |
| 2026-11-25 | regressions | G1 strict, default regression list | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-regress-r52.log |
| 2026-11-25 | regressions | G1 strict, default regression list | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-regress-r57.log |
| 2026-11-25 | regressions | G1 strict, default regression list | timeout 2400 bash .g1run.sh ../kk/logs/g1-regress-r59.log | strict, sha unknown | PASS | backfill_2026-11-25/g1-regress-r59.log |
| 2026-11-25 | regressions | G1 strict, default regression list | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-regress-r61.log |
| 2026-11-25 | regressions | G1 strict, default regression list | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-regress-r64.log |
| 2026-11-25 | regressions | G1 strict, default regression list | timeout 2400 bash .g1run.sh ../kk/logs/g1-regress-r67.log | strict, sha unknown | PASS | backfill_2026-11-25/g1-regress-r67.log |
| 2026-11-25 | regressions | G1 strict, default regression list | timeout 2400 bash .g1run.sh ../kk/logs/g1-regress-r68.log | strict, sha unknown | PASS | backfill_2026-11-25/g1-regress-r68.log |
| 2026-11-25 | smol_maxwell | G1 strict bitwise gate | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-smol-r57.log |
| 2026-11-25 | unknown | G1 strict bitwise gate | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1-strict.log |
| 2026-11-25 | 4.5-SupInVisc | G1 strict bitwise gate | timeout 1500 bash .g1run.sh ../kk/logs/g1-supin-r59.log supin_visc | strict, sha unknown | PASS | backfill_2026-11-25/g1-supin-r59.log |
| 2026-11-25 | 4.4-AUSMup2 | G1 strict bitwise gate | unknown | strict, sha unknown | FAIL | backfill_2026-11-25/g1-up2-r30.log |
| 2026-11-25 | 4.4-AUSMup2 | G1 strict bitwise gate | unknown | strict, sha unknown | FAIL | backfill_2026-11-25/g1-up2-r32.log |
| 2026-11-25 | 4.4-AUSMup2 | G1 strict bitwise gate | unknown | strict, sha unknown | FAIL | backfill_2026-11-25/g1-up2-r33.log |
| 2026-11-25 | 4.4-AUSMup2 | G1 strict bitwise gate | unknown | strict, sha unknown | FAIL | backfill_2026-11-25/g1-up2-r35.log |
| 2026-11-25 | 4.4-AUSMup2 | G1 strict bitwise gate | timeout 1500 bash .g1run.sh ../kk/logs/g1-up2-r39.log invwedge_ausmplusup2 | strict, sha unknown | FAIL | backfill_2026-11-25/g1-up2-r39.log |
| 2026-11-25 | 4.4-AUSMup2 | G1 strict bitwise gate | timeout 1500 bash .g1run.sh ../kk/logs/g1-up2-r52.log invwedge_ausmplusup2 | strict, sha unknown | PASS | backfill_2026-11-25/g1-up2-r52.log |
| 2026-11-25 | 4.3-LAX | unknown | bash .g1run.sh ../kk/logs/g1lax-r24.log invwedge_lax | strict, sha unknown | PASS | backfill_2026-11-25/g1lax-r24.log |
| 2026-11-25 | 4.3-LAX | unknown | unknown | serial, sha unknown | FAIL | backfill_2026-11-25/g1lax-serial.log |
| 2026-11-25 | 4.3-LAX | unknown | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1lax-strict-r21.log |
| 2026-11-25 | 4.3-LAX | unknown | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1lax-strict.log |
| 2026-11-25 | 4.3-LAX | unknown | unknown | serial, sha unknown | FAIL | backfill_2026-11-25/g1laxv-serial.log |
| 2026-11-25 | 4.3-LAX | unknown | unknown | serial, sha unknown | FAIL | backfill_2026-11-25/g1laxv2-serial.log |
| 2026-11-25 | 4.4-Roe-BC | unknown | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1roe-strict-r25.log |
| 2026-11-25 | 4.4-Roe-BC | unknown | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g1roe-sv-r25.log |
| 2026-11-25 | baseline gates (G2-G7) | G2 production (noise) | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g2-check.log |
| 2026-11-25 | baseline gates (G2-G7) | G2 production (noise) | unknown | strict, sha unknown | PASS; ratios 1.00/1.05 | backfill_2026-11-25/g2-nf-inv.log |
| 2026-11-25 | baseline gates (G2-G7) | G2 production (noise) | unknown | strict, sha unknown | PASS; ratios 0.36/0.82 | backfill_2026-11-25/g2-nf-sc.log |
| 2026-11-25 | baseline gates (G2-G7) | G3 CPU identity (no-identity) | unknown | strict, sha unknown | PASS (no-identity) | backfill_2026-11-25/g3-noid.log |
| 2026-11-25 | baseline gates (G2-G7) | G3 CPU identity (non-NEMO Kokkos) | unknown | strict, sha unknown | PASS (non-NEMO Kokkos identity) | backfill_2026-11-25/g3-non.log |
| 2026-11-25 | baseline gates (G2-G7) | G4 thermo unit driver | unknown | serial, sha unknown | PASS (thermo unit driver) | backfill_2026-11-25/g4-serial.log |
| 2026-11-25 | baseline gates (G2-G7) | G4 thermo unit driver | unknown | strict, sha unknown | PASS (thermo unit driver) | backfill_2026-11-25/g4-strict.log |
| 2026-11-25 | baseline gates (G2-G7) | G6 MPI | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g6-mpi.log |
| 2026-11-25 | baseline gates (G2-G7) | G7 noise floor / GPU | unknown | strict, sha unknown | PASS | backfill_2026-11-25/g7-check.log |
| 2026-11-25 | baseline gates (G2-G7) | G7 noise floor / GPU | unknown | strict, sha unknown | PASS; ratios 1.00/1.05 | backfill_2026-11-25/g7-nf-inv.log |
| 2026-11-25 | baseline gates (G2-G7) | G7 noise floor / GPU | unknown | strict, sha unknown | PASS; ratio 0.36 | backfill_2026-11-25/g7-nf-sc.log |
| 2026-11-25 | 4.6-LUSGS | noise floor (K=10) | unknown | strict, sha unknown | PASS; ratios 0.71/0.95 | backfill_2026-11-25/nf-lusgs-r67.log |
| 2026-11-25 | 4.6-LUSGS | noise floor (K=10) | timeout 2400 bash .nfslusgs.sh ../kk/logs/nf-lusgs-r68.log | strict, sha unknown | PASS; ratios 0.71/0.95 | backfill_2026-11-25/nf-lusgs-r68.log |
| 2026-11-25 | 4.4-MSW | noise floor (K=10) | unknown | strict, sha unknown | PASS; ratios 0.11/1.05 | backfill_2026-11-25/nf-msw-r29.log |
| 2026-11-25 | smol_maxwell | noise floor (K=10) | unknown | strict, sha unknown | PASS; ratios 0.61/0.50 | backfill_2026-11-25/nf-smol-r57.log |
| 2026-11-25 | 4.5-SupInVisc | noise floor (K=10) | unknown | strict, sha unknown | PASS; ratios 0.50/0.72 | backfill_2026-11-25/nf-supin-r61.log |
| 2026-11-25 | 4.4-AUSMup2 | noise floor (K=10) | unknown | strict, sha unknown | PASS; ratios 0.64/1.17 | backfill_2026-11-25/nf-up2-r53.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (build finished) | backfill_2026-11-25/ninja-baseline-mpi.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (build finished) | backfill_2026-11-25/ninja-baseline.log |
| 2026-11-25 | unknown | build/configure | unknown | cuda, sha unknown | OK (build finished) | backfill_2026-11-25/ninja-cuda-r27.log |
| 2026-11-25 | unknown | build/configure | unknown | cuda, sha unknown | OK (build finished) | backfill_2026-11-25/ninja-cuda.log |
| 2026-11-25 | unknown | build/configure | unknown | serial, sha unknown | unknown | backfill_2026-11-25/ninja-serial-nnb.log |
| 2026-11-25 | unknown | build/configure | unknown | serial, sha unknown | OK (already built) | backfill_2026-11-25/ninja-serial-r20.log |
| 2026-11-25 | unknown | build/configure | unknown | serial, sha unknown | OK (build finished) | backfill_2026-11-25/ninja-serial-r25.log |
| 2026-11-25 | unknown | build/configure | unknown | serial, sha unknown | OK (already built) | backfill_2026-11-25/ninja-serial-r26b.log |
| 2026-11-25 | unknown | build/configure | unknown | serial, sha unknown | OK (already built) | backfill_2026-11-25/ninja-serial-r28.log |
| 2026-11-25 | unknown | build/configure | unknown | serial, sha unknown | OK (build finished) | backfill_2026-11-25/ninja-serial-r30.log |
| 2026-11-25 | unknown | build/configure | unknown | serial, sha unknown | OK (build finished) | backfill_2026-11-25/ninja-serial.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (build finished) | backfill_2026-11-25/ninja-strict-mpi.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (build finished) | backfill_2026-11-25/ninja-strict-r20.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (build finished) | backfill_2026-11-25/ninja-strict-r25.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (already built) | backfill_2026-11-25/ninja-strict-r26b.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (build finished) | backfill_2026-11-25/ninja-strict-r28.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (build finished) | backfill_2026-11-25/ninja-strict-r30.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (build finished) | backfill_2026-11-25/ninja-strict.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (cmake done) | backfill_2026-11-25/setup-baseline-mpi.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (cmake done) | backfill_2026-11-25/setup-baseline.log |
| 2026-11-25 | unknown | build/configure | unknown | cuda, sha unknown | OK (cmake done) | backfill_2026-11-25/setup-cuda.log |
| 2026-11-25 | unknown | build/configure | unknown | serial, sha unknown | OK (cmake done) | backfill_2026-11-25/setup-serial.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (cmake done) | backfill_2026-11-25/setup-strict-mpi.log |
| 2026-11-25 | unknown | build/configure | unknown | strict, sha unknown | OK (cmake done) | backfill_2026-11-25/setup-strict.log |
| 2026-11-25 | invwedge survey | log analysis | unknown | strict, sha unknown | PASS | backfill_2026-11-25/survey-invwdam-r24.log |
| 2026-11-25 | partial-cat survey | log analysis | unknown | strict, sha unknown | PASS | backfill_2026-11-25/survey-partialcat-r24.log |
| 2026-11-25 | 4.4-Roe-BC | A/B verify (SU2_KOKKOS_PRECOND_VERIFY=1) | unknown | strict, sha unknown | PASS | backfill_2026-11-25/sv-bcroe-lax-r26.log |
| 2026-11-25 | 4.4-Roe-BC | A/B verify (SU2_KOKKOS_PRECOND_VERIFY=1) | unknown | strict, sha unknown | PASS | backfill_2026-11-25/sv-bcroe-roe-r26.log |
| 2026-11-25 | 4.6-LUSGS | A/B verify (SU2_KOKKOS_PRECOND_VERIFY=1) | unknown | strict, sha unknown | PASS | backfill_2026-11-25/sv-lusgs-r67.log |
| 2026-11-25 | 4.6-LUSGS | A/B verify (SU2_KOKKOS_PRECOND_VERIFY=1) | timeout 1500 bash .svlusgs.sh ../kk/logs/sv-lusgs-r68.log | strict, sha unknown | PASS | backfill_2026-11-25/sv-lusgs-r68.log |
| 2026-11-25 | 4.4-MSW | A/B verify (SU2_KOKKOS_PRECOND_VERIFY=1) | unknown | strict, sha unknown | PASS | backfill_2026-11-25/sv-msw-r28.log |
| 2026-11-25 | smol_maxwell | A/B verify (SU2_KOKKOS_PRECOND_VERIFY=1) | unknown | strict, sha unknown | PASS | backfill_2026-11-25/sv-smol-r57.log |
| 2026-11-25 | 4.5-SupInVisc | A/B verify (SU2_KOKKOS_PRECOND_VERIFY=1) | unknown | strict, sha unknown | PASS | backfill_2026-11-25/sv-supin-r59.log |
| 2026-11-25 | 4.5-SupInVisc | A/B verify (SU2_KOKKOS_PRECOND_VERIFY=1) | unknown | strict, sha unknown | PASS | backfill_2026-11-25/sv-supin-r61.log |
| 2026-11-25 | 4.4-AUSMup2 | A/B verify (SU2_KOKKOS_PRECOND_VERIFY=1) | timeout 1500 bash .svrun.sh ../kk/logs/sv-up2-r53.log invwedge_ausmplusup2 | strict, sha unknown | PASS | backfill_2026-11-25/sv-up2-r53.log |
| 2026-11-26 | 4.6-MG | CPU baseline (step 2) | bash .cpumg.sh (strict SU2_CFD, ENABLE_KOKKOS=NO, invwedge_mg.cfg, 10 iters) | strict, 413b3a0038 | PASS (exit=0, residuals finite/converging) | 2026-11-26_4.6-MG/cpu-mg-r1.log |
| 2026-11-26 | 4.6-MG | G1 strict bitwise | timeout 1500 bash .g1run.sh ../kk/logs/g1-mg-r1.log invwedge_mg | strict, 413b3a0038 | PASS bitwise (max\|dres\|=0, field diff 0; wall NO 63.84s/YES 53.86s; resident ON, MG transfers=11) | 2026-11-26_4.6-MG/g1-mg-r1.log, invwedge_mg_NO.log, invwedge_mg_YES.log |
| 2026-11-26 | 4.6-MG | G1 control (MG_RESIDENT=0) | bash .g1mgoff.sh (SU2_KOKKOS_NEMO_MG_RESIDENT=0, RUN_TAG=_mgoff) | strict, 413b3a0038 | PASS bitwise (max\|dres\|=0, field diff 0; OVERALL FAIL = expected: 2 EXPECT lines missing = old behaviour) | 2026-11-26_4.6-MG/g1-mgoff-r1.log, invwedge_mg_mgoff_NO.log, invwedge_mg_mgoff_YES.log |
| 2026-11-26 | 4.6-MG | G1 variant MG_POST_SMOOTH=(1,1,1) | bash .g1mgpost.sh | strict, 413b3a0038 | PASS bitwise (max\|dres\|=0, field diff 0) | 2026-11-26_4.6-MG/g1-mgpost-r1.log, invwedge_mg_mgpost_NO.log, invwedge_mg_mgpost_YES.log |
| 2026-11-26 | 4.6-MG | G1 variant MGCYCLE=W_CYCLE | bash .g1mgw.sh | strict, 413b3a0038 | PASS bitwise (max\|dres\|=0, field diff 0) | 2026-11-26_4.6-MG/g1-mgw-r1.log, invwedge_mg_mgw_NO.log, invwedge_mg_mgw_YES.log |
| 2026-11-26 | 4.6-MG | G1 variant MG_CORRECTION_SMOOTH=(2,2,2) | bash .g1mgcs.sh | strict, 413b3a0038 | PASS bitwise (max\|dres\|=0, field diff 0) | 2026-11-26_4.6-MG/g1-mgcs-r1.log, invwedge_mg_mgcs_NO.log, invwedge_mg_mgcs_YES.log |
| 2026-11-26 | 4.6-MG | G1 viscous super_cat MGLEVEL=1 | bash .g1mgvisc.sh (EXTRA_CFG="MGLEVEL= 1", RUN_TAG=_mg) | strict, 413b3a0038 | PASS bitwise (max\|dres\|=0, field diff 0) | 2026-11-26_4.6-MG/g1-mgvisc-r1.log, super_cat_mg_NO.log, super_cat_mg_YES.log |
| 2026-11-26 | 4.6-MG | host-read hook (SU2_KOKKOS_NEMO_POISON=1) | bash .g1mgpoison.sh | strict, 413b3a0038 | PASS bitwise (max field diff 0, identical to G1 YES — no missed MG host read) | 2026-11-26_4.6-MG/poison-mg-r1.log, invwedge_mg_mgpoison_NO.log, invwedge_mg_mgpoison_YES.log |
| 2026-11-26 | 4.6-MG | noise floor (K=10) | bash .nfsmg.sh (CASE=invwedge_mg, serial+strict) | serial+strict, 413b3a0038 | PASS (ITER 11 ratio 0.71, ITER 100 ratio 1.00, both <= 10) | 2026-11-26_4.6-MG/nf-mg-r1.log (case logs shared with invwedge_mg_*.log) |
| 2026-11-26 | 4.6-MG | regression default list | timeout 2400 bash .g1run.sh ../kk/logs/g1-regress-r1.log | strict, 413b3a0038 | PASS bitwise (invwedge_ausm, thermalbath, super_cat, axi_visccone; max\|dres\|=0, field diff 0) | 2026-11-26_4.6-MG/g1-regress-r1.log, regress1__*_NO.log, regress1__*_YES.log |
| 2026-11-26 | 4.6-MG | regression targeted (lusgs/roe paths) | timeout 2400 bash .g1run.sh ../kk/logs/g1-regress2-r1.log invwedge_lusgs invwedge_roe | strict, 413b3a0038 | PASS bitwise (2/2; max\|dres\|=0, field diff 0) | 2026-11-26_4.6-MG/g1-regress2-r1.log, regress2__*_NO.log, regress2__*_YES.log |
| 2026-11-26 | 4.3-G7 | G7 GPU check (super_cat thermalbath invwedge_ausm invwedge_roe invwedge_lusgs) | Madan on V100 (roam148-045): BIN=<cuda SU2_CFD su2kokkos 8277424> REF_MODE=HOST PHYS_SCALE=1 RES_NOISE_ORDERS=10 check.sh super_cat thermalbath invwedge_ausm invwedge_roe invwedge_lusgs | cuda, 8277424 (su2kokkos) | super_cat PASS (max\|dres\|=0, field 2.26e-15), thermalbath PASS (field 4.38e-16); invwedge_ausm/roe/lusgs FAIL vs 1e-9 tol (chaotic non-converging cases at ITER 100 — judge by noise floor); all 5 cases device-resident | 2026-11-26_G7-V100/g7-2026-11-26.log, chk_super_cat_NO.log, chk_super_cat_YES.log, chk_thermalbath_NO.log, chk_thermalbath_YES.log, chk_invwedge_ausm_NO.log, chk_invwedge_ausm_YES.log, chk_invwedge_roe_NO.log, chk_invwedge_roe_YES.log, chk_invwedge_lusgs_NO.log, chk_invwedge_lusgs_YES.log |
| 2026-11-26 | 4.3-G7 | G7 noise floor invwedge_ausm | Madan on V100: CASE=invwedge_ausm check_noisefloor.sh (CUDA binary) | cuda, 8277424 (su2kokkos) | NOISEFLOOR PASS (ITER 11 ratio 1.21, ITER 100 ratio 1.53, K 10) | 2026-11-26_G7-V100/nf-ausm-gpu.log |
| 2026-11-26 | 4.3-G7 | G7 noise floor invwedge_lusgs | Madan on V100: CASE=invwedge_lusgs check_noisefloor.sh (CUDA binary) | cuda, 8277424 (su2kokkos) | NOISEFLOOR PASS (ITER 11 ratio 1.04, ITER 100 ratio 0.77, K 10) | 2026-11-26_G7-V100/nf-invwedge_lusgs-gpu.log |
| 2026-11-26 | 4.3-G7 | G7 noise floor invwedge_roe | Madan on V100: CASE=invwedge_roe check_noisefloor.sh (CUDA binary) | cuda, 8277424 (su2kokkos) | FAIL (ITER 11 ratio 0.59 PASS; ITER 100 ratio 43.75 FAIL — case chaotic, does not converge at ITER 100) | 2026-11-26_G7-V100/nf-invwedge_roe-gpu.log |
| 2026-11-26 | 4.3-G7 | G7 stage-verify invwedge_roe (10 iter) | Madan on V100: SU2_KOKKOS_STAGE_VERIFY=1 check.sh invwedge_roe (CUDA binary, 10 iters) | cuda, 8277424 (su2kokkos) | FAIL vs 1e-9 tol (expected: GPU not bitwise; max\|dres\|=5.95e-01; case chaotic) | 2026-11-26_G7-V100/sv-roe-gpu.log, chk_invwedge_roe_sv_NO.log, chk_invwedge_roe_sv_YES.log |
| 2026-11-26 | 4.3-G7 | G7 stage-verify invwedge_roe (100 iter) | Madan on V100: SU2_KOKKOS_STAGE_VERIFY=1 check.sh invwedge_roe (CUDA binary, 100 iters) | cuda, 8277424 (su2kokkos) | FAIL vs 1e-9 tol (expected: max\|dres\|=1.95e-02; case chaotic, does not converge) | 2026-11-26_G7-V100/sv100-roe-gpu.log, chk_invwedge_roe_sv100_NO.log, chk_invwedge_roe_sv100_YES.log |
| 2026-11-26 | 4.3-G7 | G7 LU_SGS device-vs-host apply verify invwedge_lusgs | Madan on V100: SU2_KOKKOS_PRECOND_VERIFY=1 check.sh invwedge_lusgs (CUDA binary) | cuda, 8277424 (su2kokkos) | FAIL vs 2e-06 tol (expected: GPU not bitwise; max\|dres\|=3.44e-03) | 2026-11-26_G7-V100/pv-lusgs-gpu.log, chk_invwedge_lusgs_pv_NO.log, chk_invwedge_lusgs_pv_YES.log |
