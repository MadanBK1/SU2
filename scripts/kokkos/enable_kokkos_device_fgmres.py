#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[2]
cpp = root / "Common/src/linear_algebra/CSysSolve.cpp"
s = cpp.read_text()

if "KOKKOS DEVICE-RESIDENT FGMRES" in s:
    print("Device-resident FGMRES branch already present; no changes made.")
    raise SystemExit(0)

anchor = '''  su2matrix<ScalarType> H(m + 1, m);\n  H = ScalarType(0);\n\n  /*--- Calculate the norm of the rhs vector. ---*/\n'''

if s.count(anchor) != 1:
    raise SystemExit(f"{cpp}: FGMRES insertion anchor count={s.count(anchor)}")

block = r'''  su2matrix<ScalarType> H(m + 1, m);
  H = ScalarType(0);

#ifdef HAVE_KOKKOS
  /*--- KOKKOS DEVICE-RESIDENT FGMRES
   * Keep Krylov bases, preconditioned vectors, SpMV results, orthogonalization,
   * and the final solution update on the accelerator. The Hessenberg matrix,
   * Givens rotations, and reduced triangular solve remain on the host because
   * they contain only O(m^2) scalars. The orthogonalization below preserves
   * SU2's existing two-pass classical Gram-Schmidt (CGS2) semantics. ---*/
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
      } else if (precond.SupportsKokkosDeviceResident()) {
        precond.KokkosDeviceResident(in, out);
      } else {
        KokkosLinearAlgebra::DeviceToHost(in);
        precond(in, out);
        KokkosLinearAlgebra::HostToDevice(out);
      }
    };

    ScalarType norm0 = KokkosLinearAlgebra::Norm(b);

    /*--- CPU FGMRES stores the negative residual first, then divides by -beta.
     * Preserve that sequence here for equivalent Arnoldi initialization. ---*/
    if (!xIsZero) {
      DeviceMatVec(x, V[0]);
      KokkosLinearAlgebra::AXPY(V[0], ScalarType(-1), b);
    } else {
      KokkosLinearAlgebra::Copy(b, V[0]);
      KokkosLinearAlgebra::Scale(V[0], ScalarType(-1));
    }

    ScalarType beta = xIsZero ? norm0 : KokkosLinearAlgebra::Norm(V[0]);

    SU2_OMP_MASTER
    ResetDeflation();
    END_SU2_OMP_MASTER

    if (tol_type == LinearToleranceType::RELATIVE) norm0 = beta;

    if (beta < tol * norm0 || beta < eps) {
      if (masterRank) {
        SU2_OMP_MASTER
        cout << "CSysSolve::FGMRES(): system solved by initial guess." << endl;
        END_SU2_OMP_MASTER
      }
      residual = beta;
      return 0;
    }

    KokkosLinearAlgebra::Scale(V[0], ScalarType(-1) / beta);
    g[0] = beta;

    unsigned long i = 0;
    if (monitoring && masterRank) {
      SU2_OMP_MASTER {
        WriteHeader("FGMRES", tol, beta);
        WriteHistory(i, beta / norm0);
      }
      END_SU2_OMP_MASTER
    }

    for (i = 0; i < m; ++i) {
      if (beta < tol * norm0) break;

      if (flexible) {
        ApplyPreconditioner(V[i], Z[i]);
        DeviceMatVec(Z[i], V[i + 1]);
      } else {
        DeviceMatVec(V[i], V[i + 1]);
      }

      /*--- SU2 CGS2 pass 1: h = V^T w; w -= V h. ---*/
      for (unsigned long k = 0; k <= i; ++k) {
        H(k, i) = KokkosLinearAlgebra::Dot(V[k], V[i + 1]);
      }
      for (unsigned long k = 0; k <= i; ++k) {
        KokkosLinearAlgebra::AXPY(V[i + 1], -H(k, i), V[k]);
      }

      /*--- SU2 CGS2 pass 2: dh = V^T w; w -= V dh; H += dh. ---*/
      for (unsigned long k = 0; k <= i; ++k) {
        const ScalarType dh = KokkosLinearAlgebra::Dot(V[k], V[i + 1]);
        H(k, i) += dh;
        KokkosLinearAlgebra::AXPY(V[i + 1], -dh, V[k]);
      }

      const ScalarType nrm = KokkosLinearAlgebra::Norm(V[i + 1]);
      if (nrm <= ScalarType(0) || nrm != nrm) {
        H(i + 1, i) = ScalarType(0);
        if (masterRank) {
          SU2_OMP_MASTER
          cout << "WARNING: FGMRES orthogonalization failed, linear solver diverged." << endl;
          END_SU2_OMP_MASTER
        }
        break;
      }

      H(i + 1, i) = nrm;
      KokkosLinearAlgebra::Scale(V[i + 1], ScalarType(1) / nrm);

      for (unsigned long k = 0; k < i; ++k) ApplyGivens(sn[k], cs[k], H(k, i), H(k + 1, i));
      GenerateGivens(H(i, i), H(i + 1, i), sn[i], cs[i]);
      ApplyGivens(sn[i], cs[i], g[i], g[i + 1]);

      beta = fabs(g[i + 1]);

      if (monitoring && masterRank && ((i + 1) % monitorFreq == 0)) {
        SU2_OMP_MASTER
        WriteHistory(i + 1, beta / norm0);
        END_SU2_OMP_MASTER
      }
    }

    SolveReduced(i, H, g, y);

    /*--- LinearCombination(..., true) in the host path means
     * x += sum_k basis[k] * y[k]. Keep that update resident. ---*/
    if (flexible) {
      for (unsigned long k = 0; k < i; ++k) KokkosLinearAlgebra::AXPY(x, y[k], Z[k]);
    } else {
      for (unsigned long k = 0; k < i; ++k) KokkosLinearAlgebra::AXPY(x, y[k], V[k]);
    }

    if (monitoring && config->GetComm_Level() == COMM_FULL) {
      if (masterRank) {
        SU2_OMP_MASTER
        WriteFinalResidual("FGMRES", i, beta / norm0);
        END_SU2_OMP_MASTER
      }

      if (recomputeRes) {
        DeviceMatVec(x, V[0]);
        KokkosLinearAlgebra::AXPY(V[0], ScalarType(-1), b);
        const ScalarType true_res = KokkosLinearAlgebra::Norm(V[0]);

        if (fabs(true_res - beta) > tol * 10) {
          if (masterRank) {
            SU2_OMP_MASTER
            WriteWarning(beta, true_res, tol);
            END_SU2_OMP_MASTER
          }
        }
      }
    }

    KokkosLinearAlgebra::DeviceToHost(x);
    residual = beta / norm0;
    return i;
  }
#endif

  /*--- Calculate the norm of the rhs vector. ---*/
'''

cpp.write_text(s.replace(anchor, block, 1))
print(f"Patched {cpp}")
print("Enabled correctness-oriented device-resident FGMRES with two-pass CGS2.")
