#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[2]

hpp = root / "Common/include/linear_algebra/CSysMatrix.hpp"
pre = root / "Common/include/linear_algebra/CPreconditioner.hpp"
cpp = root / "Common/src/linear_algebra/CSysMatrix.cpp"
kk  = root / "Common/src/linear_algebra/CSysMatrixKokkos.cpp"


def replace_once(path: Path, old: str, new: str, label: str):
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: {label} anchor count={count}")
    path.write_text(text.replace(old, new, 1))

# 1) Persistent device mirror for Jacobi inverse blocks.
replace_once(
    hpp,
    '  ScalarType* invM; /*!< \\brief Inverse of (Jacobi) preconditioner. */\n',
    '  ScalarType* invM; /*!< \\brief Inverse of (Jacobi) preconditioner. */\n\n'
    '#ifdef HAVE_KOKKOS\n'
    '  /*--- KOKKOS JACOBI DEVICE STORAGE: persistent mirror of invM. ---*/\n'
    '  ScalarType* d_invM = nullptr;\n'
    '#endif\n',
    'invM declaration',
)

# 2) Public Kokkos Jacobi hooks near existing Kokkos matrix/preconditioner declarations.
replace_once(
    hpp,
    '#ifdef HAVE_KOKKOS\n'
    '  /*! \\brief Mirror completed ILU factors and dependency metadata to Kokkos device memory. */\n'
    '  void SyncKokkosILUPreconditioner();\n',
    '#ifdef HAVE_KOKKOS\n'
    '  /*! \\brief Mirror completed Jacobi inverse diagonal blocks to Kokkos device memory. */\n'
    '  void SyncKokkosJacobiPreconditioner();\n\n'
    '  /*! \\brief Apply the Jacobi preconditioner entirely in Kokkos device memory. */\n'
    '  void KokkosComputeJacobiPreconditioner(const CSysVector<ScalarType>& vec, CSysVector<ScalarType>& prod,\n'
    '                                          CGeometry* geometry, const CConfig* config) const;\n\n'
    '  /*! \\brief Mirror completed ILU factors and dependency metadata to Kokkos device memory. */\n'
    '  void SyncKokkosILUPreconditioner();\n',
    'Kokkos declarations',
)

# 3) Device-resident Jacobi interface.
replace_once(
    pre,
    '  inline void Build() override { sparse_matrix.BuildJacobiPreconditioner(); }\n'
    '};\n',
    '  inline void Build() override { sparse_matrix.BuildJacobiPreconditioner(); }\n\n'
    '#ifdef HAVE_KOKKOS\n'
    '  /*--- KOKKOS DEVICE JACOBI APPLY ---*/\n'
    '  inline bool SupportsKokkosDeviceResident() const override { return config->GetKokkos(); }\n'
    '  inline void KokkosDeviceResident(const CSysVector<ScalarType>& u, CSysVector<ScalarType>& v) const override {\n'
    '    sparse_matrix.KokkosComputeJacobiPreconditioner(u, v, geometry, config);\n'
    '  }\n'
    '#endif\n'
    '};\n',
    'CJacobiPreconditioner',
)

# 4) Sync Jacobi factors after host build. Keep CPU factorization semantics unchanged.
replace_once(
    cpp,
    '  for (unsigned long iPoint = 0; iPoint < nPointDomain; iPoint++)\n'
    '    InverseDiagonalBlock(iPoint, &(invM[iPoint * nVar * nVar]));\n'
    '  END_SU2_OMP_FOR\n'
    '}\n\n'
    'template <class ScalarType>\n'
    'void CSysMatrix<ScalarType>::ComputeJacobiPreconditioner',
    '  for (unsigned long iPoint = 0; iPoint < nPointDomain; iPoint++)\n'
    '    InverseDiagonalBlock(iPoint, &(invM[iPoint * nVar * nVar]));\n'
    '  END_SU2_OMP_FOR\n\n'
    '#ifdef HAVE_KOKKOS\n'
    '  if (useDevice) SyncKokkosJacobiPreconditioner();\n'
    '#endif\n'
    '}\n\n'
    'template <class ScalarType>\n'
    'void CSysMatrix<ScalarType>::ComputeJacobiPreconditioner',
    'BuildJacobiPreconditioner',
)

# 5) Free device Jacobi storage in destructor. Anchor before existing Kokkos ILU frees if present.
text = cpp.read_text()
if 'GPUMemoryAllocation::gpu_free(d_invM);' not in text:
    anchor = '  if (useDevice) {\n    GPUMemoryAllocation::gpu_free(d_matrix);\n'
    if anchor not in text:
        raise SystemExit(f"{cpp}: destructor device anchor not found")
    text = text.replace(
        anchor,
        '  if (useDevice) {\n'
        '#ifdef HAVE_KOKKOS\n'
        '    if (d_invM != nullptr) GPUMemoryAllocation::gpu_free(d_invM);\n'
        '#endif\n'
        '    GPUMemoryAllocation::gpu_free(d_matrix);\n',
        1,
    )
    cpp.write_text(text)

# 6) Implement device Jacobi storage + single-kernel apply before ILU storage section.
text = kk.read_text()
marker = '/*--- KOKKOS ILU DEVICE STORAGE ------------------------------------------------\n'
if marker not in text:
    raise SystemExit(f"{kk}: ILU storage marker not found")
if 'KokkosComputeJacobiPreconditioner' not in text:
    code = r'''/*--- KOKKOS JACOBI DEVICE STORAGE ---------------------------------------------
 * BuildJacobiPreconditioner() retains SU2's host block inversion. The completed
 * inverse diagonal blocks are mirrored to persistent device memory. ---*/
template <class ScalarType>
void CSysMatrix<ScalarType>::SyncKokkosJacobiPreconditioner() {
  if (invM == nullptr || nPointDomain == 0) return;

  const auto count = nPointDomain * nVar * nVar;
  if (d_invM == nullptr)
    d_invM = GPUMemoryAllocation::gpu_alloc<ScalarType>(count * sizeof(ScalarType));

  Kokkos::deep_copy(DeviceView<ScalarType>(d_invM, count), HostView<const ScalarType>(invM, count));
}

/*--- KOKKOS DEVICE JACOBI APPLY -----------------------------------------------
 * One independent block GEMV per owned point. No triangular dependencies. ---*/
template <class ScalarType>
void CSysMatrix<ScalarType>::KokkosComputeJacobiPreconditioner(const CSysVector<ScalarType>& vec,
                                                               CSysVector<ScalarType>& prod,
                                                               CGeometry* geometry,
                                                               const CConfig* config) const {
  if (d_invM == nullptr)
    SU2_MPI::Error("Kokkos Jacobi device storage is not initialized.", CURRENT_FUNCTION);

#ifndef NDEBUG
  if (vec.GetNVar() != nVar || prod.GetNVar() != nVar)
    SU2_MPI::Error("Incompatible vector dimensions in Kokkos Jacobi apply.", CURRENT_FUNCTION);
#endif

  using execution_space = Kokkos::DefaultExecutionSpace;
  using range_policy = Kokkos::RangePolicy<execution_space, Kokkos::IndexType<unsigned long>>;

  execution_space exec;
  const auto inv_diag = d_invM;
  const auto input = vec.GetDevicePointer();
  const auto output = prod.GetDevicePointer();
  const auto block_size = nVar;
  const auto block_entries = nVar * nVar;
  const auto domain_points = nPointDomain;

  Kokkos::parallel_for(
      "SU2::KokkosJacobiApply", range_policy(exec, 0ul, domain_points),
      KOKKOS_LAMBDA(const unsigned long i) {
        const auto offset = i * block_size;
        const auto block = inv_diag + i * block_entries;

        for (unsigned long i_var = 0; i_var < block_size; ++i_var) {
          ScalarType value = 0;
          for (unsigned long j_var = 0; j_var < block_size; ++j_var)
            value += block[i_var * block_size + j_var] * input[offset + j_var];
          output[offset + i_var] = value;
        }
      });

  exec.fence("SU2::Kokkos Jacobi apply complete");

  /*--- Preserve existing Jacobi semantics: communicate solved halo entries. ---*/
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
    text = text.replace(marker, code + marker, 1)
    kk.write_text(text)

print(f"Patched {hpp}")
print(f"Patched {pre}")
print(f"Patched {cpp}")
print(f"Patched {kk}")
print("Enabled persistent device Jacobi storage and single-kernel Kokkos Jacobi apply.")
