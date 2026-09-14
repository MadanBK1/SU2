#!/usr/bin/env python3
"""Make enable_kokkos_device_ilu_apply.py robust to CSysMatrix.hpp formatting.

The staged device-storage patch is already validated locally.  This helper only
replaces the brittle exact-text match around SyncKokkosILUPreconditioner() in
the device-ILU apply patcher with a token/line based insertion.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PATCHER = ROOT / "scripts/kokkos/enable_kokkos_device_ilu_apply.py"

text = PATCHER.read_text()

old = r'''anchor = '''#ifdef HAVE_KOKKOS
  /*! \\brief Mirror completed ILU factors and dependency metadata to Kokkos device memory. */
  void SyncKokkosILUPreconditioner();
#endif
'''
if mh.count(anchor) != 1:
    sys.exit(f"ERROR: CSysMatrix.hpp sync declaration anchor count = {mh.count(anchor)}")
replacement = '''#ifdef HAVE_KOKKOS
  /*! \\brief Mirror completed ILU factors and dependency metadata to Kokkos device memory. */
  void SyncKokkosILUPreconditioner();

  /*--- KOKKOS DEVICE ILU APPLY ---*/
  /*! \\brief Apply the already-built ILU factors entirely in Kokkos device memory. */
  void KokkosComputeILUPreconditioner(const CSysVector<ScalarType>& vec, CSysVector<ScalarType>& prod,
                                      CGeometry* geometry, const CConfig* config) const;
#endif
'''
mh = mh.replace(anchor, replacement, 1)
'''

new = r'''sync_decl = "  void SyncKokkosILUPreconditioner();"
if mh.count(sync_decl) != 1:
    sys.exit(f"ERROR: expected exactly one SyncKokkosILUPreconditioner declaration, found {mh.count(sync_decl)}")
replacement = sync_decl + '''

  /*--- KOKKOS DEVICE ILU APPLY ---*/
  /*! \\brief Apply the already-built ILU factors entirely in Kokkos device memory. */
  void KokkosComputeILUPreconditioner(const CSysVector<ScalarType>& vec, CSysVector<ScalarType>& prod,
                                      CGeometry* geometry, const CConfig* config) const;'''
mh = mh.replace(sync_decl, replacement, 1)
'''

if old not in text:
    if "sync_decl = \"  void SyncKokkosILUPreconditioner();\"" in text:
        print("Device ILU apply sync-anchor fix is already present.")
        sys.exit(0)
    sys.exit("ERROR: expected brittle sync-anchor block was not found in device ILU apply patcher")

PATCHER.write_text(text.replace(old, new, 1))
print(f"Patched {PATCHER}")
print("Device ILU apply patcher now locates SyncKokkosILUPreconditioner() independent of surrounding formatting.")
