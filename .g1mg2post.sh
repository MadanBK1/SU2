#!/bin/bash
# review-20 queue: MG2 + post-smoothing variant (mgpost).
export EXTRA_CFG="MG_POST_SMOOTH= ( 1, 1, 1 )" RUN_TAG=_mg2post
exec timeout 1500 bash .g1run.sh ../kk/logs/g1-mg2post-r1.log invwedge_mg
