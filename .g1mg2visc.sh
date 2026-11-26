#!/bin/bash
# review-20 queue: viscous super_cat MGLEVEL=1 — coarse must report OFF (viscous), stay bitwise.
export EXTRA_CFG="MGLEVEL= 1" RUN_TAG=_mg2visc
exec timeout 1500 bash .g1run.sh ../kk/logs/g1-mg2visc-r1.log super_cat
