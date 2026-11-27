/*!
 * \file CNEMOKokkosResidualKernels.hpp
 * \brief Internal header of the device NEMO source / BC residual kernels (ResArgs, helpers, declarations).
 *        Split out of CNEMOKokkosResidual.cpp to cut nvcc compile time; code verbatim (see that file for the
 *        exactness notes).
 *
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#pragma once

#include "../../../include/solvers/kokkos/CNEMOKokkosData.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "../../../../Common/include/CConfig.hpp"
#include "../../../../Common/include/geometry/CGeometry.hpp"
#include "../../../include/variables/CNEMOEulerVariable.hpp"
#include "../../../include/variables/CNEMONSVariable.hpp"

#include "CNEMOKokkosAUSM.hpp"
#include "CNEMOKokkosLAX.hpp"
#include "CNEMOKokkosRoe.hpp"
#include "CNEMOKokkosMSW.hpp"
#include "CNEMOKokkosViscous.hpp"

/*--- External linkage so the kernels can live in separate translation units (explicit instantiations for ND = 2, 3 in
 *    CNEMOKokkosResidual{Wall,Source,BCFlux2D,BCFlux3D}.cpp; definitions in CNEMOKokkosResidualKernels.inl). ---*/
namespace nemo_kokkos_res {

/*--- Arguments of the source / BC kernels (UpwArgs carries the AUSM parameters and the state views). ---*/
struct ResArgs {
  UpwArgs a;
  View2D<double> U, dTdU, dTvedU, Cvves, charac, vtxNormal;
  View1D<double> volume, bstate;
  View1D<unsigned long> vtxNode;
  unsigned long nPointDomain = 0;
  bool frozen = false, monoatomic = false, vtLimiting = false;
  double P_Exit = 0.0;
  /*--- Step 8: viscous far-field part and NS walls. ---*/
  nemo_visc::ViscConst vc;
  View3D<double> grad;
  View2D<double> coord, Ds, Uold, trunc;
  View1D<double> mu, k, kve, bstateVisc;
  View1D<long> vtxNormalNeighbor;
  View1D<unsigned long> rowPtr, colInd;
  int eddyIdx = 0;
  bool viscous = false, catalytic = false, supercat = false, writeWallState = false;
  double Twall = 0.0, catEff = 0.0, heatFlux = 0.0, RuSI = 0.0;
  double Yst[MAXNS] = {0.0};
  /*--- 6b.b: per-vertex (MARKER_PYTHON_CUSTOM) wall temperature, indexed by the packed vertex kv. ---*/
  bool customTwall = false;
  View1D<double> twallVtx;
  /*--- 4.5 BC_Smoluchowski_Maxwell. ---*/
  double gasConstant = 0.0, prandtlLam = 0.0;
  View2D<double> slip;
  /*--- Batch A: axisymmetric source, stale boundary-numerics state (AUSM+M supersonic BCs). ---*/
  View3D<double> auxGrad;
  bool axisym = false, axiViscous = false;
  int nHeavy = 0;
  double prandtlTurb = 0.0;
  bool staleNumerics = false;
  double staleGammaI = 0.0, staleGammaJ = 0.0, staleSensor = 0.0;
};

template <int ND>
void KernelIsothermal(const ResArgs& r, unsigned long v0, unsigned long v1);
template <int ND>
void KernelHeatFlux(const ResArgs& r, unsigned long v0, unsigned long v1);
template <int ND>
void KernelSmoluchowski(const ResArgs& r, unsigned long v0, unsigned long v1);
template <int ND>
void KernelSource(const ResArgs& r, unsigned long& eChm, unsigned long& eVib, unsigned long& eAxi);
template <int ND>
void KernelSymmetry(const ResArgs& r, unsigned long v0, unsigned long v1);
template <int ND>
void KernelBoundaryFlux(const ResArgs& r, int kind, unsigned long v0, unsigned long v1, long& lastSub);

}  // namespace nemo_kokkos_res

namespace {

using nemo_kokkos_res::ResArgs;
using nemo_kokkos_res::KernelIsothermal;
using nemo_kokkos_res::KernelHeatFlux;
using nemo_kokkos_res::KernelSmoluchowski;
using nemo_kokkos_res::KernelSource;
using nemo_kokkos_res::KernelSymmetry;
using nemo_kokkos_res::KernelBoundaryFlux;

/*! \brief res_i += alpha * block (CSysVector::AddBlock / SubtractBlock semantics: vec = vec + alpha*block). */
KOKKOS_INLINE_FUNCTION void AddRow(const View2D<double>& res, unsigned long i, int nVar, const double* b, double alpha) {
  for (int v = 0; v < nVar; v++) res(i, v) = res(i, v) + alpha * b[v];
}

/*! \brief CSysMatrix::AddBlock2Diag(i, J, alpha): mat_ii = mat_ii + alpha*J (row-major nVar x nVar). */
KOKKOS_INLINE_FUNCTION void AddDiag(const UpwArgs& a, unsigned long i, const double* J, double alpha) {
  const unsigned long bs = static_cast<unsigned long>(a.nVar) * a.nVar;
  double* m = &a.jac(a.dia_ptr(i) * bs);
  for (unsigned long k = 0; k < bs; ++k) m[k] = m[k] + alpha * J[k];
}

/*! \brief CNumerics::CheckResidualNaNs (jacobian_j == nullptr). */
KOKKOS_INLINE_FUNCTION bool ResidualNaN(bool implicit, int nVar, const double* r, const double* J) {
  bool err = false;
  for (int iVar = 0; iVar < nVar; iVar++) {
    if (Kokkos::isnan(r[iVar])) err = true;
    if (implicit)
      for (int jVar = 0; jVar < nVar; jVar++)
        if (Kokkos::isnan(J[iVar * nVar + jVar])) err = true;
  }
  return err;
}

/*! \brief CSysMatrix::DeleteValsRowi(i, row) on the device accumulator. */
KOKKOS_INLINE_FUNCTION void DeleteValsRowi(const ResArgs& r, unsigned long i, int row) {
  const unsigned long nv = r.a.nVar;
  for (auto index = r.rowPtr(i); index < r.rowPtr(i + 1); index++) {
    for (unsigned long iVar = 0; iVar < nv; iVar++) r.a.jac(index * nv * nv + row * nv + iVar) = 0.0;
    if (r.colInd(index) == i) r.a.jac(index * nv * nv + row * nv + row) = 1.0;
  }
}

/*! \brief SetVelocity_Old(i, 0) and the momentum rows of LinSysRes / Res_TruncError (wall BCs). */
KOKKOS_INLINE_FUNCTION void WallMomentumRows(const ResArgs& r, unsigned long iPoint, int ND) {
  const int nSpecies = r.a.ns;
  const int RHO = nSpecies + ND + 3;
  for (int iDim = 0; iDim < ND; iDim++) {
    if (r.writeWallState) r.Uold(iPoint, nSpecies + iDim) = 0.0 * r.a.V(iPoint, RHO);
  }
  for (int iDim = 0; iDim < ND; iDim++) r.a.res(iPoint, nSpecies + iDim) = 0.0;
  if (r.writeWallState)
    for (int iDim = 0; iDim < ND; iDim++) r.trunc(iPoint, nSpecies + iDim) = 0.0;
}

}  // namespace
