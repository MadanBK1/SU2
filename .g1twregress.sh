#!/bin/bash
# review-23 step 3 (6b.b-TwallIn): regressions — default list + supin_visc + smol_maxwell (strict)
# The py-custom Twall code path must be a no-op for all non-pycustom cases: all bitwise.
exec timeout 3600 bash .g1run.sh ../kk/logs/g1-tw-regress-r1.log invwedge_ausm thermalbath super_cat axi_visccone supin_visc smol_maxwell
