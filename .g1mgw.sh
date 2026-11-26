#!/bin/bash
# review-16 step 5b: W-cycle variant.
export EXTRA_CFG="MGCYCLE= W_CYCLE" RUN_TAG=_mgw
exec timeout 1500 bash .g1run.sh ../kk/logs/g1-mgw-r1.log invwedge_mg
