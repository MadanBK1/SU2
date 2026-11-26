#!/bin/bash
# review-16 step 5d: viscous super_cat with MGLEVEL=1.
export EXTRA_CFG="MGLEVEL= 1" RUN_TAG=_mg
exec timeout 1500 bash .g1run.sh ../kk/logs/g1-mgvisc-r1.log super_cat
