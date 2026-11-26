#!/bin/bash
# review-21 step 5 (6b.a-DynGrid): variant — translation only (zero rotation).
export EXTRA_CFG="ROTATION_RATE= 0.0 0.0 0.0" RUN_TAG=_dyntr
exec timeout 1500 bash .g1run.sh ../kk/logs/g1-dyntr-r1.log invwedge_dyn
