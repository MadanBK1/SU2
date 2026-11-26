#!/bin/bash
# review-21 step 6 comparison: StageVerify A/B on the PASSING inviscid invwedge_dyn.
# Purpose: does the inviscid case show the same TimeStep ~0.9% delta as viscous super_cat?
# If yes -> TimeStep delta is a benign/accepted quirk; viscous FAIL has a different cause.
# If no  -> the viscous TimeStep delta is a new dynamic-grid+viscous bug.
export RUN_TAG=_sv
export SU2_KOKKOS_STAGE_VERIFY=1
exec timeout 1500 bash .g1run.sh ../kk/logs/sv-dyn-inv-r1.log invwedge_dyn
