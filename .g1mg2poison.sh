#!/bin/bash
# review-20 queue: MG2 poison check — stale host copies NaN'd; missed host sync changes results.
export SU2_KOKKOS_NEMO_POISON=1 RUN_TAG=_mg2poison
exec timeout 1500 bash .g1run.sh ../kk/logs/poison-mg2-r1.log invwedge_mg
