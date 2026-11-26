#!/bin/bash
# review-16 step 5c: MG correction smoothing variant.
export EXTRA_CFG="MG_CORRECTION_SMOOTH= ( 2, 2, 2 )" RUN_TAG=_mgcs
exec timeout 1500 bash .g1run.sh ../kk/logs/g1-mgcs-r1.log invwedge_mg
