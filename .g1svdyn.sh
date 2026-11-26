#!/bin/bash
# review-21 step 6 (6b.a-DynGrid): StageVerify on the FAILING viscous variant.
export EXTRA_CFG="ITER=;TIME_DOMAIN= YES;TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER;TIME_STEP= 1e-7;TIME_ITER= 3;INNER_ITER= 3;GRID_MOVEMENT= RIGID_MOTION;TRANSLATION_RATE= 20.0 0.0 0.0"
export RUN_TAG=_sv
export SU2_KOKKOS_STAGE_VERIFY=1
exec timeout 1500 bash .g1run.sh ../kk/logs/sv-dyn-r1.log super_cat
