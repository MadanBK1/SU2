#!/bin/bash
# review-21 step 5 (6b.a-DynGrid): variant — 1st-order dual time.
export EXTRA_CFG="TIME_MARCHING= DUAL_TIME_STEPPING-1ST_ORDER" RUN_TAG=_dyn1st
exec timeout 1500 bash .g1run.sh ../kk/logs/g1-dyn1st-r1.log invwedge_dyn
