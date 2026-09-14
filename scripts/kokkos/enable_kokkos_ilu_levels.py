#!/usr/bin/env python3
"""Enable ILU dependency levels for Kokkos builds.

SU2 historically constructs levels_ilu only when OpenMP has more than one host
thread and LINEAR_SOLVER_ILU_LEVELS is enabled. Device triangular solves also
need a level schedule, including builds configured with -Dwith-omp=false.

This staged patch deliberately changes only level construction. It does not
change CPU ILU factorization or application semantics.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Common/src/linear_algebra/CSysMatrix.cpp"
MARKER = "KOKKOS ILU LEVEL SCHEDULE"

text = SOURCE.read_text()

if MARKER in text:
    print("Kokkos ILU level scheduling patch is already present.")
    sys.exit(0)

old = '''    if (omp_get_max_threads() > 1 && config->GetLinear_Solver_ILU_levels()) {
      levels_ilu = computeLevels(csr_ilu);
    }
'''

new = '''    /*--- KOKKOS ILU LEVEL SCHEDULE
     * Host ILU historically builds dependency levels only for multi-threaded
     * OpenMP.  A Kokkos triangular solve requires the same dependency graph
     * even when SU2 was configured with -Dwith-omp=false, so build the level
     * structure whenever Kokkos is enabled.  The CPU ILU algorithms remain
     * unchanged and may continue to ignore levels_ilu when appropriate. ---*/
    if (config->GetKokkos() ||
        (omp_get_max_threads() > 1 && config->GetLinear_Solver_ILU_levels())) {
      levels_ilu = computeLevels(csr_ilu);
    }
'''

count = text.count(old)
if count != 1:
    sys.exit(f"ERROR: expected exactly one ILU level-construction anchor, found {count}")

text = text.replace(old, new, 1)
SOURCE.write_text(text)
print(f"Patched {SOURCE}")
print("Enabled ILU dependency levels for Kokkos, including with-omp=false builds.")
