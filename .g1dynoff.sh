#!/bin/bash
# review-21 step 4 (6b.a-DynGrid): control — dynamic grid OFF (SU2_KOKKOS_NEMO_DYNGRID=0),
# old host path restored; expect bitwise PASS, OVERALL may show missing "stage dynamic grid"
# EXPECT lines (expected: DYNGRID off -> those stages not on device).
export EXTRA_CFG="" SU2_KOKKOS_NEMO_DYNGRID=0 RUN_TAG=_dynoff
exec timeout 1500 bash .g1run.sh ../kk/logs/g1-dynoff-r1.log invwedge_dyn
