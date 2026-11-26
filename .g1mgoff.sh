#!/bin/bash
# round 73 (review 16, step 4): control — old behaviour, MG fine level NOT resident.
# EXPECT "ON" line will be missing (expected); report tolerances/diffs only.
set -u
export SU2_KOKKOS_NEMO_MG_RESIDENT=0
export RUN_TAG=_mgoff
export EXTRA_CFG=""
bash "$(dirname "$0")/.g1run.sh" "$@"
