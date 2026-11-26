#!/bin/bash
# review-20 queue: MG2 + correction-smoothing variant (mgcs).
export EXTRA_CFG="MG_CORRECTION_SMOOTH= ( 2, 2, 2 )" RUN_TAG=_mg2cs
exec timeout 1500 bash .g1run.sh ../kk/logs/g1-mg2cs-r1.log invwedge_mg
