#!/bin/bash
# G1 driver for super_cat with LINEAR_SOLVER_PREC= LU_SGS (review-15 queue step 4).
set -u
export EXTRA_CFG="LINEAR_SOLVER_PREC= LU_SGS"
export RUN_TAG=_lusgs
exec bash .g1run.sh "$@"
