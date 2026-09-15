#!/usr/bin/env python3
"""Fix Kokkos ILU backward substitution to match CPU halo semantics.

The CPU BackwardSolve stops processing upper-triangular entries when the
column index reaches nPointDomain, because halo values are communicated only
after the local triangular solve. The initial Kokkos port omitted that guard,
so it consumed stale/uninitialized halo values during backward substitution.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Common/src/linear_algebra/CSysMatrixKokkos.cpp"
MARKER = "KOKKOS ILU HALO COLUMN GUARD"

text = SOURCE.read_text()

if MARKER in text:
    print("Kokkos ILU halo-column guard is already present.")
    sys.exit(0)

old = '''          for (auto index = diagonal[i] + 1; index < rows[i + 1]; ++index) {\n            const auto j = columns[index];\n            const auto out_j = j * block_size;\n'''

if text.count(old) != 1:
    sys.exit(f"ERROR: expected exactly one Kokkos ILU backward-loop anchor, found {text.count(old)}")

new = '''          for (auto index = diagonal[i] + 1; index < rows[i + 1]; ++index) {\n            const auto j = columns[index];\n            /*--- KOKKOS ILU HALO COLUMN GUARD\n             * Match CPU BackwardSolve(iPoint, nPointDomain): halo columns are\n             * not part of the local triangular solve and are communicated only\n             * after the solve completes. ---*/\n            if (j >= nPointDomain) break;\n            const auto out_j = j * block_size;\n'''

text = text.replace(old, new, 1)
SOURCE.write_text(text)

print(f"Patched {SOURCE}")
print("Restored CPU-equivalent ILU backward halo-column handling in the Kokkos path.")
