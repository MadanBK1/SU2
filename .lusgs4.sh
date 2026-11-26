#!/bin/bash
# Step 4 driver (review-15 queue): G1 strict viscous super_cat with LU_SGS preconditioner.
set -u
SU2=/home/mpr7366/software/AgenticTest/SU2
KK=/home/mpr7366/software/AgenticTest/kk
LOG=$1; shift
CASES="$*"
EXTRA_CFG="LINEAR_SOLVER_PREC= LU_SGS" RUN_TAG=_lusgs \
bash $SU2/.g1run.sh "$LOG" $CASES
