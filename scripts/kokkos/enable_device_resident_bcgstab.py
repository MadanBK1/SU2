#!/usr/bin/env python3
"""Enable device-resident Kokkos BCGSTAB while preserving the host fallback.

This is staged as a source patcher so the implementation can be compiled and
validated on the target GPU before committing the CSysSolve.cpp change.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Common/src/linear_algebra/CSysSolve.cpp"
MARKER = "KOKKOS DEVICE-RESIDENT BCGSTAB"

text = SOURCE.read_text()

include_line = '#include "../../include/linear_algebra/KokkosVectorOps.hpp"\n'
if include_line not in text:
    sys.exit("ERROR: KokkosVectorOps.hpp include is missing; apply the device-resident CG patch first")

if MARKER in text:
    print("Device-resident BCGSTAB patch is already present.")
    sys.exit(0)

anchor = '''      bcg_ready = true;
    }
    END_SU2_OMP_SAFE_GLOBAL_ACCESS
  }

  /*--- Calculate the initial residual, compute norm, and check if system is already solved ---*/
'''

if text.count(anchor) != 1:
    sys.exit(f"ERROR: expected exactly one BCGSTAB insertion anchor, found {text.count(anchor)}")

resident = r'''      bcg_ready = true;
    }
    END_SU2_OMP_SAFE_GLOBAL_ACCESS
  }

#ifdef HAVE_KOKKOS
  /*--- KOKKOS DEVICE-RESIDENT BCGSTAB
   * Keep Krylov vectors and SpMV results on the accelerator. Until the
   * preconditioners themselves are ported, a non-identity preconditioner is an
   * explicit synchronization boundary. The original host BCGSTAB remains below
   * as the fallback for non-Kokkos and host-staged MPI configurations. ---*/
  if (config->GetKokkos() && mat_vec.SupportsKokkosDeviceResident()) {
    bool matrix_on_device = false;

    KokkosLinearAlgebra::HostToDevice(b);
    KokkosLinearAlgebra::HostToDevice(x);

    auto DeviceMatVec = [&](const CSysVector<ScalarType>& in, CSysVector<ScalarType>& out) {
      mat_vec.KokkosDeviceResident(in, out, matrix_on_device, true, false);
      matrix_on_device = true;
    };

    auto ApplyPreconditioner = [&](const CSysVector<ScalarType>& in, CSysVector<ScalarType>& out) {
      if (precond.IsIdentity()) {
        KokkosLinearAlgebra::Copy(in, out);
      } else {
        KokkosLinearAlgebra::DeviceToHost(in);
        precond(in, out);
        KokkosLinearAlgebra::HostToDevice(out);
      }
    };

    /*--- Initial residual r = b - A*x. ---*/
    if (!xIsZero) {
      DeviceMatVec(x, A_x);
      KokkosLinearAlgebra::Copy(b, r);
      KokkosLinearAlgebra::AXPY(r, ScalarType(-1), A_x);
    } else {
      KokkosLinearAlgebra::Copy(b, r);
    }

    if (config->GetComm_Level() == COMM_FULL) {
      norm0 = KokkosLinearAlgebra::Norm(b);
      norm_r = xIsZero ? norm0 : KokkosLinearAlgebra::Norm(r);

      if (tol_type == LinearToleranceType::RELATIVE) norm0 = norm_r;

      if ((norm_r < tol * norm0) || (norm_r < eps)) {
        if (masterRank) {
          SU2_OMP_MASTER
          cout << "CSysSolve::BCGSTAB(): system solved by initial guess." << endl;
          END_SU2_OMP_MASTER
        }
        return 0;
      }

      if (monitoring && masterRank) {
        SU2_OMP_MASTER {
          WriteHeader("BCGSTAB", tol, norm_r);
          WriteHistory(i, norm_r / norm0);
        }
        END_SU2_OMP_MASTER
      }
    }

    ScalarType alpha = ScalarType(1);
    ScalarType omega = ScalarType(1);
    ScalarType rho = ScalarType(1);
    ScalarType rho_prime = ScalarType(1);

    KokkosLinearAlgebra::Fill(p, ScalarType(0));
    KokkosLinearAlgebra::Fill(v, ScalarType(0));
    KokkosLinearAlgebra::Copy(r, r_0);

    for (i = 0; i < m; i++) {
      rho_prime = rho;
      rho = KokkosLinearAlgebra::Dot(r, r_0);

      const ScalarType beta = (rho / rho_prime) * (alpha / omega);

      /*--- p = beta*(p - omega*v) + r. ---*/
      KokkosLinearAlgebra::AXPY(p, -omega, v);
      KokkosLinearAlgebra::Scale(p, beta);
      KokkosLinearAlgebra::AXPY(p, ScalarType(1), r);

      ApplyPreconditioner(p, z);
      DeviceMatVec(z, v);

      const ScalarType r_0_v = KokkosLinearAlgebra::Dot(r_0, v);
      alpha = rho / r_0_v;

      KokkosLinearAlgebra::AXPY(x, alpha, z);
      KokkosLinearAlgebra::AXPY(r, -alpha, v);

      ApplyPreconditioner(r, z);
      DeviceMatVec(z, A_x);

      const ScalarType omega_denom = KokkosLinearAlgebra::Dot(A_x, A_x);
      if (omega_denom == ScalarType(0)) break;
      omega = KokkosLinearAlgebra::Dot(A_x, r) / omega_denom;

      KokkosLinearAlgebra::AXPY(x, omega, z);
      KokkosLinearAlgebra::AXPY(r, -omega, A_x);

      if (config->GetComm_Level() == COMM_FULL) {
        norm_r = KokkosLinearAlgebra::Norm(r);
        if (norm_r < tol * norm0) break;
        if (monitoring && masterRank && ((i + 1) % monitorFreq == 0)) {
          SU2_OMP_MASTER
          WriteHistory(i + 1, norm_r / norm0);
          END_SU2_OMP_MASTER
        }
      }
    }

    if (monitoring && config->GetComm_Level() == COMM_FULL) {
      if (masterRank) {
        SU2_OMP_MASTER
        WriteFinalResidual("BCGSTAB", i, norm_r / norm0);
        END_SU2_OMP_MASTER
      }

      if (recomputeRes) {
        DeviceMatVec(x, A_x);
        KokkosLinearAlgebra::Copy(b, r);
        KokkosLinearAlgebra::AXPY(r, ScalarType(-1), A_x);
        const ScalarType true_res = KokkosLinearAlgebra::Norm(r);

        if ((fabs(true_res - norm_r) > tol * 10.0) && masterRank) {
          SU2_OMP_MASTER
          WriteWarning(norm_r, true_res, tol);
          END_SU2_OMP_MASTER
        }
      }
    }

    KokkosLinearAlgebra::DeviceToHost(x);

    residual = norm_r / norm0;
    return i;
  }
#endif

  /*--- Calculate the initial residual, compute norm, and check if system is already solved ---*/
'''

text = text.replace(anchor, resident, 1)
SOURCE.write_text(text)
print(f"Patched {SOURCE}")
print("Added device-resident BCGSTAB while preserving the original host BCGSTAB fallback.")
