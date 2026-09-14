#!/usr/bin/env python3
"""Make enable_kokkos_device_ilu_apply.py robust to CSysMatrix.hpp formatting."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PATCHER = ROOT / "scripts/kokkos/enable_kokkos_device_ilu_apply.py"

text = PATCHER.read_text()

start_marker = "# ---------------------------------------------------------------------------\n# CSysMatrix.hpp: declare device ILU application next to the sync method.\n# ---------------------------------------------------------------------------\n"
end_marker = "# ---------------------------------------------------------------------------\n# CPreconditioner.hpp: generic device-resident hook and ILU override.\n# ---------------------------------------------------------------------------\n"

if start_marker not in text or end_marker not in text:
    sys.exit("ERROR: could not locate CSysMatrix.hpp patch section in device ILU patcher")

start = text.index(start_marker) + len(start_marker)
end = text.index(end_marker, start)

replacement = '''sync_decl = "  void SyncKokkosILUPreconditioner();"
if mh.count(sync_decl) != 1:
    sys.exit(f"ERROR: expected exactly one SyncKokkosILUPreconditioner declaration, found {mh.count(sync_decl)}")

sync_replacement = sync_decl + """

  /*--- KOKKOS DEVICE ILU APPLY ---*/
  /*! \\brief Apply the already-built ILU factors entirely in Kokkos device memory. */
  void KokkosComputeILUPreconditioner(const CSysVector<ScalarType>& vec, CSysVector<ScalarType>& prod,
                                      CGeometry* geometry, const CConfig* config) const;"""

mh = mh.replace(sync_decl, sync_replacement, 1)

'''

text = text[:start] + replacement + text[end:]
PATCHER.write_text(text)

print(f"Patched {PATCHER}")
print("Device ILU apply patcher now uses a token-based SyncKokkosILUPreconditioner anchor.")
