#!/usr/bin/env python3
"""Enable level-scheduled Kokkos ILU application while retaining CPU factorization.

Requires the staged ILU dependency-level and device-storage patches.  The host
BuildILUPreconditioner() remains authoritative; only repeated forward/backward
triangular application is moved to the Kokkos execution space.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
MATRIX_HPP = ROOT / "Common/include/linear_algebra/CSysMatrix.hpp"
PREC_HPP = ROOT / "Common/include/linear_algebra/CPreconditioner.hpp"
MATRIX_KK = ROOT / "Common/src/linear_algebra/CSysMatrixKokkos.cpp"
SOLVE_CPP = ROOT / "Common/src/linear_algebra/CSysSolve.cpp"
MARKER = "KOKKOS DEVICE ILU APPLY"

mh = MATRIX_HPP.read_text()
ph = PREC_HPP.read_text()
kk = MATRIX_KK.read_text()
ss = SOLVE_CPP.read_text()

if MARKER in mh or MARKER in ph or MARKER in kk or MARKER in ss:
    print("Kokkos device ILU apply patch is already present.")
    sys.exit(0)

# The preceding storage stage must already have been applied locally.
required = [
    (mh, "d_ILU_matrix", "CSysMatrix.hpp device ILU storage"),
    (mh, "SyncKokkosILUPreconditioner", "CSysMatrix.hpp ILU sync declaration"),
    (kk, "SyncKokkosILUPreconditioner", "CSysMatrixKokkos.cpp ILU sync implementation"),
]
for text, token, desc in required:
    if token not in text:
        sys.exit(f"ERROR: missing prerequisite: {desc}")

# ---------------------------------------------------------------------------
# CSysMatrix.hpp: declare device ILU application next to the sync method.
# ---------------------------------------------------------------------------
anchor = '''#ifdef HAVE_KOKKOS
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

# ---------------------------------------------------------------------------
# CPreconditioner.hpp: generic device-resident hook and ILU override.
# ---------------------------------------------------------------------------
anchor = '''  virtual bool IsIdentity() const { return false; }

  /*!
   * \\brief Factory method.
   */
'''
if ph.count(anchor) != 1:
    sys.exit(f"ERROR: CPreconditioner base hook anchor count = {ph.count(anchor)}")
replacement = '''  virtual bool IsIdentity() const { return false; }

#ifdef HAVE_KOKKOS
  /*--- KOKKOS DEVICE ILU APPLY
   * Optional accelerator-resident preconditioner interface.  False by default
   * so every existing preconditioner preserves its host implementation. ---*/
  virtual bool SupportsKokkosDeviceResident() const { return false; }
  virtual void KokkosDeviceResident(const CSysVector<ScalarType>& u, CSysVector<ScalarType>& v) const {}
#endif

  /*!
   * \\brief Factory method.
   */
'''
ph = ph.replace(anchor, replacement, 1)

anchor = '''  inline void Build() override { sparse_matrix.BuildILUPreconditioner(); }
};
'''
if ph.count(anchor) != 1:
    sys.exit(f"ERROR: CILUPreconditioner tail anchor count = {ph.count(anchor)}")
replacement = '''  inline void Build() override { sparse_matrix.BuildILUPreconditioner(); }

#ifdef HAVE_KOKKOS
  /*--- KOKKOS DEVICE ILU APPLY ---*/
  inline bool SupportsKokkosDeviceResident() const override { return config->GetKokkos(); }
  inline void KokkosDeviceResident(const CSysVector<ScalarType>& u, CSysVector<ScalarType>& v) const override {
    sparse_matrix.KokkosComputeILUPreconditioner(u, v, geometry, config);
  }
#endif
};
'''
ph = ph.replace(anchor, replacement, 1)

# ---------------------------------------------------------------------------
# CSysMatrixKokkos.cpp: level-scheduled forward/backward solves.
# Insert immediately before HtDTransfer().
# ---------------------------------------------------------------------------
anchor = '''template <class ScalarType>
void CSysMatrix<ScalarType>::HtDTransfer(bool trigger) const {
'''
if kk.count(anchor) != 1:
    sys.exit(f"ERROR: CSysMatrixKokkos.cpp HtD anchor count = {kk.count(anchor)}")

method = r'''/*--- KOKKOS DEVICE ILU APPLY --------------------------------------------------
 * CPU BuildILUPreconditioner() remains unchanged.  Its completed factors are
 * mirrored by SyncKokkosILUPreconditioner(); this routine applies those factors
 * using dependency levels. Rows in one level are independent, while levels are
 * fenced to preserve the exact triangular dependency ordering portably across
 * CUDA, HIP, and SYCL execution spaces. ---*/
template <class ScalarType>
void CSysMatrix<ScalarType>::KokkosComputeILUPreconditioner(const CSysVector<ScalarType>& vec,
                                                            CSysVector<ScalarType>& prod,
                                                            CGeometry* geometry,
                                                            const CConfig* config) const {
  if (d_ILU_matrix == nullptr || d_row_ptr_ilu == nullptr || d_dia_ptr_ilu == nullptr ||
      d_col_ind_ilu == nullptr || d_ilu_level_ptr == nullptr || d_ilu_level_rows == nullptr ||
      levels_ilu.empty()) {
    SU2_MPI::Error("Kokkos ILU device storage is not initialized.", CURRENT_FUNCTION);
  }

#ifndef NDEBUG
  if (vec.GetNVar() != nVar || prod.GetNVar() != nVar)
    SU2_MPI::Error("Incompatible vector dimensions in Kokkos ILU apply.", CURRENT_FUNCTION);
#endif

  using execution_space = Kokkos::DefaultExecutionSpace;
  using range_policy = Kokkos::RangePolicy<execution_space, Kokkos::IndexType<unsigned long>>;

  const auto factors = d_ILU_matrix;
  const auto rows = d_row_ptr_ilu;
  const auto diagonal = d_dia_ptr_ilu;
  const auto columns = d_col_ind_ilu;
  const auto level_ptr = d_ilu_level_ptr;
  const auto level_rows = d_ilu_level_rows;
  const auto input = vec.GetDevicePointer();
  const auto output = prod.GetDevicePointer();
  const auto block_size = nVar;
  const auto block_entries = nVar * nVar;
  const auto n_levels = levels_ilu.getOuterSize();

  /*--- Forward substitution: y_i = b_i - sum_{j<i} L_ij y_j. ---*/
  for (unsigned long level = 0; level < n_levels; ++level) {
    const auto begin = levels_ilu.outerPtr()[level];
    const auto end = levels_ilu.outerPtr()[level + 1];

    Kokkos::parallel_for(
        "SU2::KokkosILUForwardLevel", range_policy(begin, end), KOKKOS_LAMBDA(const unsigned long k) {
          const auto i = level_rows[k];
          const auto out_i = i * block_size;

          for (unsigned long i_var = 0; i_var < block_size; ++i_var)
            output[out_i + i_var] = input[out_i + i_var];

          for (auto index = rows[i]; index < diagonal[i]; ++index) {
            const auto j = columns[index];
            const auto out_j = j * block_size;
            const auto block = factors + index * block_entries;

            for (unsigned long i_var = 0; i_var < block_size; ++i_var) {
              ScalarType value = 0;
              for (unsigned long j_var = 0; j_var < block_size; ++j_var)
                value += block[i_var * block_size + j_var] * output[out_j + j_var];
              output[out_i + i_var] -= value;
            }
          }
        });
    Kokkos::fence("SU2::Kokkos ILU forward level complete");
  }

  /*--- Backward substitution: x_i = inv(U_ii) *
   *     (y_i - sum_{j>i} U_ij x_j). ---*/
  for (unsigned long level_plus_one = n_levels; level_plus_one > 0; --level_plus_one) {
    const auto level = level_plus_one - 1;
    const auto begin = levels_ilu.outerPtr()[level];
    const auto end = levels_ilu.outerPtr()[level + 1];

    Kokkos::parallel_for(
        "SU2::KokkosILUBackwardLevel", range_policy(begin, end), KOKKOS_LAMBDA(const unsigned long k) {
          const auto i = level_rows[k];
          const auto out_i = i * block_size;
          ScalarType rhs[20];
          ScalarType result[20];

          for (unsigned long i_var = 0; i_var < block_size; ++i_var) rhs[i_var] = output[out_i + i_var];

          for (auto index = diagonal[i] + 1; index < rows[i + 1]; ++index) {
            const auto j = columns[index];
            const auto out_j = j * block_size;
            const auto block = factors + index * block_entries;

            for (unsigned long i_var = 0; i_var < block_size; ++i_var) {
              ScalarType value = 0;
              for (unsigned long j_var = 0; j_var < block_size; ++j_var)
                value += block[i_var * block_size + j_var] * output[out_j + j_var];
              rhs[i_var] -= value;
            }
          }

          const auto inv_diag = factors + diagonal[i] * block_entries;
          for (unsigned long i_var = 0; i_var < block_size; ++i_var) {
            ScalarType value = 0;
            for (unsigned long j_var = 0; j_var < block_size; ++j_var)
              value += inv_diag[i_var * block_size + j_var] * rhs[j_var];
            result[i_var] = value;
          }
          for (unsigned long i_var = 0; i_var < block_size; ++i_var) output[out_i + i_var] = result[i_var];
        });
    Kokkos::fence("SU2::Kokkos ILU backward level complete");
  }

  /*--- Match the existing preconditioner semantics: communicate solved halo
   * entries. Keep the result resident when the caller is a device Krylov solver. ---*/
  if (config->GetKokkosGPUAwareMPI()) {
    KokkosGPUAwareHaloExchange(prod, geometry, false);
  } else {
    prod.DtHTransfer();
    CSysMatrixComms::Initiate(prod, geometry, config);
    CSysMatrixComms::Complete(prod, geometry, config);
    prod.HtDTransfer();
  }
}

'''
kk = kk.replace(anchor, method + anchor, 1)

# ---------------------------------------------------------------------------
# Resident Krylov solvers: prefer the device preconditioner hook.
# There are currently two identical fallback lambdas (CG and BCGSTAB).
# ---------------------------------------------------------------------------
old = '''      if (precond.IsIdentity()) {
        KokkosLinearAlgebra::Copy(in, out);
      } else {
        KokkosLinearAlgebra::DeviceToHost(in);
        precond(in, out);
        KokkosLinearAlgebra::HostToDevice(out);
      }
'''
count = ss.count(old)
if count != 2:
    sys.exit(f"ERROR: expected 2 resident preconditioner fallback blocks, found {count}")
new = '''      if (precond.IsIdentity()) {
        KokkosLinearAlgebra::Copy(in, out);
#ifdef HAVE_KOKKOS
      } else if (precond.SupportsKokkosDeviceResident()) {
        /*--- KOKKOS DEVICE ILU APPLY ---*/
        precond.KokkosDeviceResident(in, out);
#endif
      } else {
        KokkosLinearAlgebra::DeviceToHost(in);
        precond(in, out);
        KokkosLinearAlgebra::HostToDevice(out);
      }
'''
ss = ss.replace(old, new)

MATRIX_HPP.write_text(mh)
PREC_HPP.write_text(ph)
MATRIX_KK.write_text(kk)
SOLVE_CPP.write_text(ss)

print(f"Patched {MATRIX_HPP}")
print(f"Patched {PREC_HPP}")
print(f"Patched {MATRIX_KK}")
print(f"Patched {SOLVE_CPP}")
print("Enabled level-scheduled device ILU application for resident Kokkos CG/BCGSTAB.")
print("CPU ILU factorization and all non-Kokkos preconditioner fallbacks remain unchanged.")
