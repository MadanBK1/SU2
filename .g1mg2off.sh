#!/bin/bash
# review-20 queue step: MG2 control — coarse levels OFF (SU2_KOKKOS_NEMO_MG_DEVICE=0),
# step-1 path (fine resident + host transfer loops) restored; expect bitwise PASS,
# OVERALL FAIL only via missing "device-resident state: ON" EXPECT lines.
export EXTRA_CFG="" SU2_KOKKOS_NEMO_MG_DEVICE=0 RUN_TAG=_mg2off
exec timeout 1500 bash .g1run.sh ../kk/logs/g1-mg2off-r1.log invwedge_mg
