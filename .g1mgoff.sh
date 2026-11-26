#!/bin/bash
# G1 control: force old behaviour (MG resident OFF) per review-16 step 4.
export EXTRA_CFG="" SU2_KOKKOS_NEMO_MG_RESIDENT=0 RUN_TAG=_mgoff
exec timeout 1500 bash .g1run.sh ../kk/logs/g1-mgoff-r1.log invwedge_mg
