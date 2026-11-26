#!/bin/bash
# review-20 queue: MG2 + W-cycle variant.
export EXTRA_CFG="MGCYCLE= W_CYCLE" RUN_TAG=_mg2w
exec timeout 1500 bash .g1run.sh ../kk/logs/g1-mg2w-r1.log invwedge_mg
