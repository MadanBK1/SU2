#!/bin/bash
# review-21 step 7 (6b.a-DynGrid): regressions — default list + invwedge_mg + invwedge_lusgs
# (static grids must stay bitwise: they take the old code paths).
exec timeout 3000 bash .g1run.sh ../kk/logs/g1-dynregress-r1.log invwedge_ausm thermalbath super_cat axi_visccone invwedge_mg invwedge_lusgs
