#!/usr/bin/env python3
"""6b.b coupling gate: per-vertex wall temperature IN through pysu2 on a NEMO isothermal marker.

Usage (from check.sh with PYDRIVER set):  python3 pycustom_twall.py run.cfg
The cfg must list the wall in MARKER_ISOTHERMAL and MARKER_PYTHON_CUSTOM. Every iteration the script sets
T_wall(x) = T0 + DT * (x - xmin) / (xmax - xmin) on PYCUSTOM_MARKER (default Wall) via SetMarkerCustomTemperature,
then runs one iteration. PYCUSTOM_DT=0 reproduces the uniform MARKER_ISOTHERMAL value (identity test).
At the end it writes wall_heatflux.csv (iVertex, x, y, Twall, q_n) from GetMarkerNormalHeatFlux (heat-flux OUT).
"""
import os, sys
from mpi4py import MPI
import pysu2

cfg = sys.argv[1]
comm = MPI.COMM_WORLD
marker = os.environ.get('PYCUSTOM_MARKER', 'Wall')
T0 = float(os.environ.get('PYCUSTOM_T0', '300'))
DT = float(os.environ.get('PYCUSTOM_DT', '300'))

driver = pysu2.CSinglezoneDriver(cfg, 1, comm)
ids = driver.GetMarkerIndices()
mid = ids.get(marker, None)
nv = driver.GetNumberMarkerNodes(mid) if mid is not None else 0
xs = [driver.MarkerCoordinates(mid)(v, 0) for v in range(nv)] if nv else []
lo = comm.allreduce(min(xs) if xs else 1e300, op=MPI.MIN)
hi = comm.allreduce(max(xs) if xs else -1e300, op=MPI.MAX)
span = (hi - lo) if hi > lo else 1.0
Tw = [T0 + DT * (x - lo) / span for x in xs]
iflow = 0  # FLOW_SOL (the NEMO flow solver slot)

for it in range(driver.GetNumberTimeIter()):
    for v in range(nv):
        driver.SetMarkerCustomTemperature(mid, v, Tw[v])
    driver.BoundaryConditionsUpdate()
    driver.Preprocess(it)
    driver.Run()
    driver.Postprocess()
    driver.Update()
    stop = driver.Monitor(it)
    driver.Output(it)
    if stop:
        break

if nv:
    with open('wall_heatflux.csv' if comm.size == 1 else f'wall_heatflux_{comm.rank}.csv', 'w') as f:
        f.write('iVertex,x,y,Twall,qn\n')
        for v in range(nv):
            f.write(f'{v},{xs[v]:.17g},{driver.MarkerCoordinates(mid)(v, 1):.17g},{Tw[v]:.17g},'
                    f'{driver.GetMarkerNormalHeatFlux(iflow, mid, v):.17g}\n')
driver.Finalize()
