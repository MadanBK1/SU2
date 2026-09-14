#!/usr/bin/env python3
"""Enable the first device-resident Kokkos Krylov solver path (CG).

This script is intentionally staged as a source patcher so the implementation can
be compiled and validated on the target GPU before it is committed into
CSysSolve.cpp.  It is idempotent and refuses to patch an unexpected source tree.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Common/src/linear_algebra/CSysSolve.cpp"
MARKER = "KOKKOS DEVICE-RESIDENT CG"

text = SOURCE.read_text()

include_anchor = '#include "../../include/linear_algebra/CPreconditioner.hpp"\n'
include_line = '#include "../../include/linear_algebra/KokkosVectorOps.hpp"\n'
if include_line not in text:
    if include_anchor not in text:
        sys.exit("ERROR: could not find CPreconditioner include anchor")
    text = text.replace(include_anchor, include_anchor + include_line, 1)

if MARKER in text:
    SOURCE.write_text(text)
    print("Device-resident CG patch is already present.")
    sys.exit(0)

anchor = '''      cg_ready = true;
    }
    END_SU2_OMP_SAFE_GLOBAL_ACCESS
  }

  /*--- Calculate the initial residual, compute norm, and check if system is already solved ---*/
'''

if text.count(anchor) != 1:
    sys.exit(f"ERROR: expected exactly one CG insertion anchor, found {text.count(anchor)}")

resident_cg = r'''      cg_ready = true;
    }
    END_SU2_OMP_SAFE_GLOBAL_ACCESS
  }

#ifdef HAVE_KOKKOS
  /*--- KOKKOS DEVICE-RESIDENT CG
   * Keep Krylov vectors on the accelerator.  Matrix values, RHS, and initial
   * solution are uploaded once per linear solve.  Until the preconditioners
   * themselves are ported, non-identity preconditioners form an explicit
   * synchronization boundary: r is copied to host, the existing preconditioner
   * is applied, and z is copied back to device.  Identity preconditioning stays
   * entirely on device. ---*/
  if (config->GetKokkos() && mat_vec.SupportsKokkosDeviceResident()) {
    bool matrix_on_device = false;

    KokkosLinearAlgebra::HostToDevice(b);
    KokkosLinearAlgebra::HostToDevice(x);

    auto DeviceMatVec = [&](const CSysVector<ScalarType>& in, CSysVector<ScalarType>& out) {
      mat_vec.KokkosDeviceResident(in, out, matrix_on_device, true, false);
      matrix_on_device = true;
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
        if (masterRank && (lin_sol_mode != LINEAR_SOLVER_MODE::MESH_DEFORM)) {
          SU2_OMP_MASTER
          cout << "CSysSolve::ConjugateGradient(): system solved by initial guess." << endl;
          END_SU2_OMP_MASTER
        }
        return 0;
      }

      if (monitoring && masterRank) {
        SU2_OMP_MASTER {
          WriteHeader("CG", tol, norm_r);
          WriteHistory(i, norm_r / norm0);
        }
        END_SU2_OMP_MASTER
      }
    }

    auto ApplyPreconditioner = [&](const CSysVector<ScalarType>& in, CSysVector<ScalarType>& out) {
      if (precond.IsIdentity()) {
        KokkosLinearAlgebra::Copy(in, out);
      } else {
        KokkosLinearAlgebra::DeviceToHost(in);
        precond(in, out);
        KokkosLinearAlgebra::HostToDevice(out);
      }
    };

    ApplyPreconditioner(r, z);
    KokkosLinearAlgebra::Copy(z, p);
    ScalarType r_dot_z = KokkosLinearAlgebra::Dot(r, z);

    for (i = 0; i < m; i++) {
      DeviceMatVec(p, A_x);

      const ScalarType denom = KokkosLinearAlgebra::Dot(A_x, p);
      const ScalarType alpha = r_dot_z / denom;

      KokkosLinearAlgebra::AXPY(x, alpha, p);
      KokkosLinearAlgebra::AXPY(r, -alpha, A_x);

      if (config->GetComm_Level() == COMM_FULL) {
        norm_r = KokkosLinearAlgebra::Norm(r);
        if (norm_r < tol * norm0) break;
        if (monitoring && masterRank && ((i + 1) % monitorFreq == 0)) {
          SU2_OMP_MASTER
          WriteHistory(i + 1, norm_r / norm0);
          END_SU2_OMP_MASTER
        }
      }

      ApplyPreconditioner(r, z);

      const ScalarType previous_r_dot_z = r_dot_z;
      r_dot_z = KokkosLinearAlgebra::Dot(r, z);
      const ScalarType beta = r_dot_z / previous_r_dot_z;

      /*--- p = z + beta*p. ---*/
      KokkosLinearAlgebra::AXPBY(p, ScalarType(1), z, beta);
    }

    if (monitoring && config->GetComm_Level() == COMM_FULL) {
      if (masterRank) {
        SU2_OMP_MASTER
        WriteFinalResidual("CG", i, norm_r / norm0);
        END_SU2_OMP_MASTER
      }

      if (recomputeRes) {
        DeviceMatVec(x, A_x);
        KokkosLinearAlgebra::Copy(b, r);
        KokkosLinearAlgebra::AXPY(r, ScalarType(-1), A_x);
        const ScalarType true_res = KokkosLinearAlgebra::Norm(r);

        if (fabs(true_res - norm_r) > tol * 10.0) {
          if (masterRank) {
            SU2_OMP_MASTER
            WriteWarning(norm_r, true_res, tol);
            END_SU2_OMP_MASTER
          }
        }
      }
    }

    /*--- SU2 outside the solver still consumes the host-side solution. ---*/
    KokkosLinearAlgebra::DeviceToHost(x);

    residual = norm_r / norm0;
    return i;
  }
#endif

  /*--- Calculate the initial residual, compute norm, and check if system is already solved ---*/
'''

text = text.replace(anchor, resident_cg, 1)
SOURCE.write_text(text)
print(f"Patched {SOURCE}")
print("Added device-resident CG while preserving the original host CG fallback.")
