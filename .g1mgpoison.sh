#!/bin/bash
# G1 poison check per review-16 step 6: stale host copies NaN'd; a missed host sync changes results.
export SU2_KOKKOS_NEMO_POISON=1 RUN_TAG=_mgpoison
exec timeout 1500 bash .g1run.sh ../kk/logs/poison-mg-r1.log invwedge_mg
