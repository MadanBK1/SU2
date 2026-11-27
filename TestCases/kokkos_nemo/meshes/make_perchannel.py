#!/usr/bin/env python3
"""4.6-Periodic gate mesh: 2D channel [0,2]x[0,1] with a sin^2 bump (h=0.04) on the lower wall, x in [0.5,1.5].
Markers: Lower / Upper (MARKER_EULER), PerLeft (x=0) / PerRight (x=2) translational periodic pair.
Usage: python3 make_perchannel.py [nx ny] > perchannel.su2   (committed mesh: nx=60 ny=20)."""
import math, sys
nx, ny = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (60, 20)
L, H, h = 2.0, 1.0, 0.04
def yb(x): return h * math.sin(math.pi * (x - 0.5)) ** 2 if 0.5 <= x <= 1.5 else 0.0
pid = lambda i, j: j * (nx + 1) + i
out = ["NDIME= 2", "NELEM= %d" % (nx * ny)]
k = 0
for j in range(ny):
    for i in range(nx):
        out.append("9 %d %d %d %d %d" % (pid(i, j), pid(i + 1, j), pid(i + 1, j + 1), pid(i, j + 1), k)); k += 1
out.append("NPOIN= %d" % ((nx + 1) * (ny + 1)))
for j in range(ny + 1):
    for i in range(nx + 1):
        x = L * i / nx; y0 = yb(x); y = y0 + (H - y0) * j / ny
        out.append("%.15e %.15e %d" % (x, y, pid(i, j)))
mk = [("Lower", [(pid(i, 0), pid(i + 1, 0)) for i in range(nx)]),
      ("Upper", [(pid(i + 1, ny), pid(i, ny)) for i in range(nx)]),
      ("PerLeft", [(pid(0, j + 1), pid(0, j)) for j in range(ny)]),
      ("PerRight", [(pid(nx, j), pid(nx, j + 1)) for j in range(ny)])]
out.append("NMARK= %d" % len(mk))
for name, el in mk:
    out += ["MARKER_TAG= " + name, "MARKER_ELEMS= %d" % len(el)] + ["3 %d %d" % e for e in el]
print("\n".join(out))
