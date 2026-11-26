#!/bin/bash
# review-16 step 5a: MG post-smoothing variant (fine-level post-smoothing after host transfer).
export EXTRA_CFG="MG_POST_SMOOTH= ( 1, 1, 1 )" RUN_TAG=_mgpost
exec timeout 1500 bash .g1run.sh ../kk/logs/g1-mgpost-r1.log invwedge_mg
