#!/usr/bin/env python3
"""Decouple Kokkos Krylov device residency from direct GPU-aware MPI.

This staged patch enables device-resident Krylov solvers whenever Kokkos is
active.  When KOKKOS_GPU_AWARE_MPI=NO, SpMV halo communication remains on the
existing host-staged SU2 MPI path; if the caller requested a device-resident
result, the communicated vector is copied back to device before returning.

The script is idempotent and intentionally narrow so it can be compiled and
benchmarked before committing the source changes.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "Common/include/linear_algebra/CMatrixVectorProduct.hpp"
SOURCE = ROOT / "Common/src/linear_algebra/CSysMatrixKokkos.cpp"

header = HEADER.read_text(encoding="utf-8-sig")
source = SOURCE.read_text()

old_support = '''  inline bool SupportsKokkosDeviceResident() const override {
#ifdef HAVE_KOKKOS
    /*--- Direct device residency currently requires the device-buffer halo path.
     * Host-staged MPI still needs the SpMV result on host. ---*/
    return config->GetKokkos() && config->GetKokkosGPUAwareMPI();
#else
    return false;
#endif
  }
'''

new_support = '''  inline bool SupportsKokkosDeviceResident() const override {
#ifdef HAVE_KOKKOS
    /*--- Krylov device residency is independent of MPI transport.  With
     * KOKKOS_GPU_AWARE_MPI=YES the halo stays in device buffers; otherwise
     * the existing host-staged halo path is used and the communicated result
     * is restored to device before returning to a resident Krylov solver. ---*/
    return config->GetKokkos();
#else
    return false;
#endif
  }
'''

if old_support in header:
    header = header.replace(old_support, new_support, 1)
elif new_support not in header:
    sys.exit("ERROR: unexpected SupportsKokkosDeviceResident implementation")

old_host = '''  } else {
    /*--- Host-staged communication intrinsically requires the product on host. ---*/
    prod.DtHTransfer();
    CSysMatrixComms::Initiate(prod, geometry, config);
    CSysMatrixComms::Complete(prod, geometry, config);
  }
'''

new_host = '''  } else {
    /*--- Host-staged MPI requires host data for packing/communication.  A
     * device-resident Krylov caller can nevertheless continue on device once
     * the halo has been completed, so restore the communicated vector to the
     * accelerator unless the caller explicitly requested host output. ---*/
    prod.DtHTransfer();
    CSysMatrixComms::Initiate(prod, geometry, config);
    CSysMatrixComms::Complete(prod, geometry, config);
    if (kokkos_spmv_mode.active && !output_to_host) prod.HtDTransfer();
  }
'''

if old_host in source:
    source = source.replace(old_host, new_host, 1)
elif new_host not in source:
    sys.exit("ERROR: unexpected host-staged Kokkos SpMV communication block")

HEADER.write_text(header)
SOURCE.write_text(source)
print(f"Patched {HEADER}")
print(f"Patched {SOURCE}")
print("Enabled device-resident Krylov solvers with either direct or host-staged MPI halos.")
