#!/usr/bin/env python3
"""Stage Kokkos device storage/upload for SU2 ILU factors and dependency levels.

This patch intentionally does not change preconditioner execution. CPU ILU
factorization/application remains active; after each BuildILUPreconditioner()
the completed factors and static CSR/level metadata are mirrored to device
memory so the subsequent device triangular-solve patch can be compiled and
validated independently.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HPP = ROOT / "Common/include/linear_algebra/CSysMatrix.hpp"
CPP = ROOT / "Common/src/linear_algebra/CSysMatrix.cpp"
KOKKOS = ROOT / "Common/src/linear_algebra/CSysMatrixKokkos.cpp"
MARKER = "KOKKOS ILU DEVICE STORAGE"

hpp = HPP.read_text()
cpp = CPP.read_text()
kk = KOKKOS.read_text()

if MARKER in hpp or MARKER in cpp or MARKER in kk:
    print("Kokkos ILU device-storage patch is already present.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# CSysMatrix.hpp: persistent device mirrors and upload method declaration.
# ---------------------------------------------------------------------------
anchor = '''  ScalarType* ILU_matrix;           /*!< \\brief Entries of the ILU sparse matrix. */
  unsigned long nnz_ilu;            /*!< \\brief Number of possible nonzero entries in the matrix (ILU). */
  const unsigned long* row_ptr_ilu; /*!< \\brief Pointers to the first element in each row (ILU). */
  const unsigned long* dia_ptr_ilu; /*!< \\brief Pointers to the diagonal element in each row (ILU). */
  const unsigned long* col_ind_ilu; /*!< \\brief Column index for each of the elements in val() (ILU). */
  unsigned short ilu_fill_in;       /*!< \\brief Fill in level for the ILU preconditioner. */
'''
if hpp.count(anchor) != 1:
    sys.exit(f"ERROR: CSysMatrix.hpp ILU field anchor count = {hpp.count(anchor)}")
replacement = anchor + r'''

#ifdef HAVE_KOKKOS
  /*--- KOKKOS ILU DEVICE STORAGE
   * Persistent mirrors of the completed host ILU factors and their sparse /
   * level-scheduling metadata. The structure is immutable after matrix
   * initialization; only d_ILU_matrix is refreshed when ILU is rebuilt. ---*/
  ScalarType* d_ILU_matrix = nullptr;
  const unsigned long* d_row_ptr_ilu = nullptr;
  const unsigned long* d_dia_ptr_ilu = nullptr;
  const unsigned long* d_col_ind_ilu = nullptr;
  const unsigned long* d_ilu_level_ptr = nullptr;
  const unsigned long* d_ilu_level_rows = nullptr;
#endif
'''
hpp = hpp.replace(anchor, replacement, 1)

anchor = '''  /*! \\brief Kokkos block-CSR sparse matrix-vector product. */
  void KokkosMatrixVectorProduct(const CSysVector<ScalarType>& vec, CSysVector<ScalarType>& prod,
                                 CGeometry* geometry, const CConfig* config) const;
'''
if hpp.count(anchor) != 1:
    sys.exit(f"ERROR: Kokkos SpMV declaration anchor count = {hpp.count(anchor)}")
replacement = anchor + r'''

#ifdef HAVE_KOKKOS
  /*! \\brief Mirror completed ILU factors and dependency metadata to Kokkos device memory. */
  void SyncKokkosILUPreconditioner();
#endif
'''
hpp = hpp.replace(anchor, replacement, 1)

# ---------------------------------------------------------------------------
# CSysMatrix.cpp: free mirrors and synchronize factors after host build.
# ---------------------------------------------------------------------------
anchor = '''  if (useDevice) {
    GPUMemoryAllocation::gpu_free(d_matrix);
    GPUMemoryAllocation::gpu_free(d_row_ptr);
    GPUMemoryAllocation::gpu_free(d_col_ind);
  }
'''
if cpp.count(anchor) != 1:
    sys.exit(f"ERROR: CSysMatrix.cpp destructor anchor count = {cpp.count(anchor)}")
replacement = anchor + r'''

#ifdef HAVE_KOKKOS
  GPUMemoryAllocation::gpu_free(d_ILU_matrix);
  GPUMemoryAllocation::gpu_free(d_row_ptr_ilu);
  GPUMemoryAllocation::gpu_free(d_dia_ptr_ilu);
  GPUMemoryAllocation::gpu_free(d_col_ind_ilu);
  GPUMemoryAllocation::gpu_free(d_ilu_level_ptr);
  GPUMemoryAllocation::gpu_free(d_ilu_level_rows);
#endif
'''
cpp = cpp.replace(anchor, replacement, 1)

# Insert synchronization immediately before ComputeILUPreconditioner().
# Do not depend on the exact number of blank lines between functions.
next_marker = "template <class ScalarType>\nvoid CSysMatrix<ScalarType>::ComputeILUPreconditioner"
next_pos = cpp.find(next_marker)
if next_pos < 0:
    sys.exit("ERROR: ComputeILUPreconditioner marker not found")

build_pos = cpp.rfind("void CSysMatrix<ScalarType>::BuildILUPreconditioner()", 0, next_pos)
if build_pos < 0:
    sys.exit("ERROR: BuildILUPreconditioner marker not found before ComputeILUPreconditioner")

# Find the final closing brace of BuildILUPreconditioner by scanning from its
# opening brace. This is insensitive to comments and blank-line formatting.
open_pos = cpp.find("{", build_pos)
if open_pos < 0:
    sys.exit("ERROR: opening brace for BuildILUPreconditioner not found")

depth = 0
close_pos = None
for i in range(open_pos, next_pos):
    ch = cpp[i]
    if ch == "{":
        depth += 1
    elif ch == "}":
        depth -= 1
        if depth == 0:
            close_pos = i
            break

if close_pos is None:
    sys.exit("ERROR: closing brace for BuildILUPreconditioner not found")

sync = '''\n#ifdef HAVE_KOKKOS\n  if (useDevice) SyncKokkosILUPreconditioner();\n#endif\n'''
cpp = cpp[:close_pos] + sync + cpp[close_pos:]

# ---------------------------------------------------------------------------
# CSysMatrixKokkos.cpp: portable allocation and deep-copy implementation.
# ---------------------------------------------------------------------------
include_anchor = '#include "../../include/geometry/CGeometry.hpp"\n'
include_line = '#include "../../include/toolboxes/allocation_toolbox.hpp"\n'
if include_line not in kk:
    if kk.count(include_anchor) != 1:
        sys.exit("ERROR: allocation_toolbox include anchor not found")
    kk = kk.replace(include_anchor, include_anchor + include_line, 1)

anchor = '''template <class ScalarType>
void CSysMatrix<ScalarType>::HtDTransfer(bool trigger) const {
'''
if kk.count(anchor) != 1:
    sys.exit(f"ERROR: CSysMatrixKokkos.cpp insertion anchor count = {kk.count(anchor)}")
method = r'''/*--- KOKKOS ILU DEVICE STORAGE ------------------------------------------------
 * Keep CPU factorization semantics unchanged. The completed factors are
 * mirrored after each BuildILUPreconditioner(); sparse and level metadata are
 * allocated/copied only once because the matrix graph is immutable. ---*/
template <class ScalarType>
void CSysMatrix<ScalarType>::SyncKokkosILUPreconditioner() {
  if (ILU_matrix == nullptr || nnz_ilu == 0 || levels_ilu.empty()) return;

  const auto block_size = nVar * nEqn;
  const auto factor_count = nnz_ilu * block_size;
  const auto level_count = levels_ilu.getOuterSize();
  const auto level_rows = levels_ilu.getNumNonZeros();

  if (d_ILU_matrix == nullptr)
    d_ILU_matrix = GPUMemoryAllocation::gpu_alloc<ScalarType>(factor_count * sizeof(ScalarType));

  if (d_row_ptr_ilu == nullptr)
    d_row_ptr_ilu = GPUMemoryAllocation::gpu_alloc_cpy<const unsigned long>(
        row_ptr_ilu, (nPoint + 1) * sizeof(unsigned long));
  if (d_dia_ptr_ilu == nullptr)
    d_dia_ptr_ilu = GPUMemoryAllocation::gpu_alloc_cpy<const unsigned long>(
        dia_ptr_ilu, nPoint * sizeof(unsigned long));
  if (d_col_ind_ilu == nullptr)
    d_col_ind_ilu = GPUMemoryAllocation::gpu_alloc_cpy<const unsigned long>(
        col_ind_ilu, nnz_ilu * sizeof(unsigned long));
  if (d_ilu_level_ptr == nullptr)
    d_ilu_level_ptr = GPUMemoryAllocation::gpu_alloc_cpy<const unsigned long>(
        levels_ilu.outerPtr(), (level_count + 1) * sizeof(unsigned long));
  if (d_ilu_level_rows == nullptr)
    d_ilu_level_rows = GPUMemoryAllocation::gpu_alloc_cpy<const unsigned long>(
        levels_ilu.innerIdx(), level_rows * sizeof(unsigned long));

  Kokkos::deep_copy(DeviceView<ScalarType>(d_ILU_matrix, factor_count),
                    HostView<const ScalarType>(ILU_matrix, factor_count));
}

'''
kk = kk.replace(anchor, method + anchor, 1)

HPP.write_text(hpp)
CPP.write_text(cpp)
KOKKOS.write_text(kk)

print(f"Patched {HPP}")
print(f"Patched {CPP}")
print(f"Patched {KOKKOS}")
print("Staged persistent Kokkos device mirrors for ILU factors and level metadata.")
print("Preconditioner execution is intentionally unchanged in this stage.")
