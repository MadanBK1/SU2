#!/bin/bash
# review-21 step 7 (6b.a-DynGrid): poison check — stale host copies NaN'd; a missed
# geometry re-upload or stale read changes results.
export SU2_KOKKOS_NEMO_POISON=1 RUN_TAG=_dynpoison
exec timeout 1500 bash .g1run.sh ../kk/logs/poison-dyn-r1.log invwedge_dyn
