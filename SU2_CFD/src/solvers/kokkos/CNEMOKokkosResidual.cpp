/*!
 * \file CNEMOKokkosResidual.cpp
 * \brief Device (Kokkos) NEMO residual stages of Step 7 (NEMO_KOKKOS_PORT_PLAN.md):
 *        - the device residual accumulator (st.Res + st.JacVal) and its host <-> device moves;
 *        - CNEMOEulerSolver::Source_Residual (finite-rate chemistry + V-T relaxation, no axisymmetry) as one fused
 *          point kernel;
 *        - the Euler boundary conditions BC_Sym_Plane / BC_Euler_Wall, BC_Far_Field (inviscid), BC_Outlet,
 *          BC_Supersonic_Inlet / BC_Supersonic_Outlet (AUSM) as vertex kernels over the flattened marker arrays.
 *
 * Exactness: every kernel evaluates the CPU statements in source order (CSource_NEMO, CSU2TCLib, the BC bodies) and
 * adds into the accumulator in the CPU order per entry: points / vertices are unique within one kernel (one marker
 * per launch, markers launched in the CIntegration order), so no atomics are needed and the per-entry summation order
 * is upwind -> source (chemistry, then V-T) -> BCs exactly as on the CPU, on every backend.
 *
 * Host state the CPU loops leave behind and that is reproduced: NaN counters (source), the solver FluidModel state
 * after BC_Outlet (replayed on the host by the caller from the returned last subsonic V_outlet), CharacPrimVar
 * (downloaded on request). Not reproduced (documented): the scratch members of the numerics objects (their gas
 * model state after the source loop, Residual/Jacobian_i scratch of the solver), which nothing reads before
 * overwriting.
 *
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#ifdef HAVE_KOKKOS

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
#include "CNEMOKokkosViscous.hpp"

namespace {

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
  /*--- Batch A: axisymmetric source, stale boundary-numerics state (AUSM+M supersonic BCs). ---*/
  View3D<double> auxGrad;
  bool axisym = false, axiViscous = false;
  int nHeavy = 0;
  double prandtlTurb = 0.0;
  bool staleNumerics = false;
  double staleGammaI = 0.0, staleGammaJ = 0.0, staleSensor = 0.0;
};

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

/*!
 * \brief CNEMONSSolver::BC_Isothermal_Wall: BC_IsothermalNonCatalytic_Wall then (catalytic marker) the species /
 *        energy part of BC_IsothermalCatalytic_Wall, fused per vertex (each vertex reads other points but writes only
 *        the row / diagonal block of its own point, so the per-entry order equals the CPU's two marker loops).
 */
template <int ND>
void KernelIsothermal(const ResArgs& r, unsigned long v0, unsigned long v1) {
  Kokkos::parallel_for(
      "NEMO::BC_Isothermal_Wall", Kokkos::RangePolicy<exec_space>(v0, v1), KOKKOS_LAMBDA(const unsigned long kv) {
        const UpwArgs& a = r.a;
        const NEMOGasTables& g = a.gas;
        const int nSpecies = a.ns, nVar = a.nVar, nDim = ND;
        const NEMOPrimIndices idx(nSpecies, ND);
        const bool implicit = a.implicit;
        const auto iPoint = r.vtxNode(kv);
        if (iPoint >= r.nPointDomain) return;

        /*====================== BC_IsothermalNonCatalytic_Wall ======================*/
        const double C = 5;
        const double Twall = r.Twall;
        double Normal[ND], UnitNormal[ND];
        for (int iDim = 0; iDim < nDim; iDim++) Normal[iDim] = r.vtxNormal(kv, iDim);
        double sq = 0.0;
        for (int iDim = 0; iDim < nDim; ++iDim) sq += Normal[iDim] * Normal[iDim];
        const double Area = Kokkos::sqrt(sq);
        for (int iDim = 0; iDim < nDim; iDim++) UnitNormal[iDim] = Normal[iDim] / Area;

        const auto Point_Normal = static_cast<unsigned long>(r.vtxNormalNeighbor(kv));
        double d2 = 0.0;
        for (int iDim = 0; iDim < nDim; iDim++) {
          const double dd = r.coord(iPoint, iDim) - r.coord(Point_Normal, iDim);
          d2 += dd * dd;
        }
        const double dist_ij = Kokkos::sqrt(d2);

        double Res_Visc[MAXNV];
        WallMomentumRows(r, iPoint, ND);
        for (int iVar = 0; iVar < nVar; iVar++) Res_Visc[iVar] = 0.0;

        const double Ti = a.V(iPoint, idx.T());
        const double Tj = a.V(Point_Normal, idx.T());
        const double Tvei = a.V(iPoint, idx.TVE());
        const double Tvej = a.V(Point_Normal, idx.TVE());
        const double ktr = r.k(iPoint);
        const double kve = r.kve(iPoint);

        Res_Visc[nSpecies + nDim] = ((ktr * (Ti - Tj) + kve * (Tvei - Tvej)) +
                                     (ktr * (Twall - Ti) + kve * (Twall - Tvei)) * C) * Area / dist_ij;
        Res_Visc[nSpecies + nDim + 1] = (kve * (Tvei - Tvej) + kve * (Twall - Tvei) * C) * Area / dist_ij;

        double Jacobian_i[MAXNV * MAXNV];
        if (implicit) {
          for (int q = 0; q < nVar * nVar; q++) Jacobian_i[q] = 0.0;
          const double* dTdU = &r.dTdU(iPoint, 0);
          const double* dTvedU = &r.dTvedU(iPoint, 0);
          double theta = 0.0;
          for (int iDim = 0; iDim < nDim; ++iDim) theta += UnitNormal[iDim] * UnitNormal[iDim];
          for (int iVar = 0; iVar < nVar; iVar++) {
            Jacobian_i[(nSpecies + nDim) * nVar + iVar] =
                -(ktr * theta / dist_ij * dTdU[iVar] + kve * theta / dist_ij * dTvedU[iVar]) * Area;
            Jacobian_i[(nSpecies + nDim + 1) * nVar + iVar] = -kve * theta / dist_ij * dTvedU[iVar] * Area;
          }
        }

        AddRow(a.res, iPoint, nVar, Res_Visc, -1.0);
        if (implicit) {
          AddDiag(a, iPoint, Jacobian_i, -1.0);
          /*--- CPU: rows 1..nDim (not the momentum rows nSpecies..), reproduced. ---*/
          for (int iVar = 1; iVar <= nDim; iVar++) DeleteValsRowi(r, iPoint, iVar);
        }

        if (!r.catalytic) return;

        /*====================== BC_IsothermalCatalytic_Wall (species / energy) ======================*/
        const double RuSI = r.RuSI;
        const double Ru = 1000.0 * RuSI;
        const auto jPoint = Point_Normal;
        const double dij = dist_ij;
        for (int iVar = 0; iVar < nVar; iVar++) Res_Visc[iVar] = 0.0;

        const double* Vi = &a.V(iPoint, 0);
        const double* Vj = &a.V(jPoint, 0);
        const double* Di = &r.Ds(iPoint, 0);
        const double* eves = &a.eves(iPoint, 0);
        double hs[MAXNS];
        ComputeSpeciesEnthalpy(g, Vi[idx.T()], Vi[idx.TVE()], eves, hs);
        const double rho = Vi[idx.RHO()];
        const int RHO_INDEX = idx.RHO();

        if (r.supercat) {
          const double* Yst = r.Yst;
          double dYdn = 0.0, SdYdn = 0.0;
          for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
            dYdn = (Yst[iSpecies] - Vj[iSpecies] / Vj[RHO_INDEX]) / dij;
            SdYdn += rho * Di[iSpecies] * dYdn;
          }
          for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
            dYdn = (Yst[iSpecies] - Vj[iSpecies] / Vj[RHO_INDEX]) / dij;
            Res_Visc[iSpecies] = -(-rho * Di[iSpecies] * dYdn + Yst[iSpecies] * SdYdn) * Area;
          }

          if (implicit) {
            const double* dTdU = &r.dTdU(iPoint, 0);
            const double* dTvedU = &r.dTvedU(iPoint, 0);
            const double* Cvve = &r.Cvves(iPoint, 0);
            double dVdU[MAXNV * MAXNV], Jj[MAXNV * MAXNV];
#define DV(rr, cc) dVdU[(rr) * nVar + (cc)]
#define JJ(rr, cc) Jj[(rr) * nVar + (cc)]
#define JI(rr, cc) Jacobian_i[(rr) * nVar + (cc)]
            for (int q = 0; q < nVar * nVar; q++) dVdU[q] = Jj[q] = Jacobian_i[q] = 0.0;

            for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
              for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++) DV(iSpecies, jSpecies) += -1.0 / rho * Yst[iSpecies];
              DV(iSpecies, iSpecies) += 1.0 / rho;
            }
            for (int iVar = 0; iVar < nVar; iVar++) {
              DV(nSpecies + nDim, iVar) = dTdU[iVar];
              DV(nSpecies + nDim + 1, iVar) = dTvedU[iVar];
            }

            for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
              for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++)
                JJ(iSpecies, jSpecies) += -Yst[iSpecies] * rho * Di[jSpecies] / dij;
              JJ(iSpecies, iSpecies) += rho * Di[iSpecies] / dij - SdYdn;
            }
            for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
              for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++)
                JJ(nSpecies + nDim, iSpecies) += JJ(jSpecies, iSpecies) * hs[iSpecies];
              JJ(nSpecies + nDim, nSpecies + nDim) += Res_Visc[iSpecies] / Area * (Ru / g.MolarMass(iSpecies) + g.Cvtrs(iSpecies));
              JJ(nSpecies + nDim, nSpecies + nDim + 1) += Res_Visc[iSpecies] / Area * Cvve[iSpecies];
            }
            for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
              for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++)
                JJ(nSpecies + nDim + 1, iSpecies) += JJ(jSpecies, iSpecies) * eves[iSpecies];
              JJ(nSpecies + nDim + 1, nSpecies + nDim + 1) += Res_Visc[iSpecies] / Area * Cvve[iSpecies];
            }
            for (int iVar = 0; iVar < nVar; iVar++)
              for (int jVar = 0; jVar < nVar; jVar++)
                for (int kVar = 0; kVar < nVar; kVar++) JI(iVar, jVar) += JJ(iVar, kVar) * DV(kVar, jVar) * Area;
#undef DV
#undef JJ
#undef JI
            /*--- Jacobian.SubtractBlock(iPoint, iPoint, Jacobian_i) ---*/
            AddDiag(a, iPoint, Jacobian_i, -1.0);
          }
        } else {
          /*--- Partially catalytic (explicit only, checked by the caller). ---*/
          const double Tw = Twall;
          const double gam = r.catEff;
          const double factor = gam * rho * Kokkos::sqrt(RuSI * Tw / 2 / g.Pi) * Area;
          for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
            const int Index = g.CatRecomb(iSpecies, 1);
            Res_Visc[iSpecies] = double(g.CatRecomb(iSpecies, 0)) * factor * Vi[Index] / Vi[RHO_INDEX] *
                                 Kokkos::sqrt(1 / g.MolarMass(Index));
          }
        }

        for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
          Res_Visc[nSpecies + nDim] += (Res_Visc[iSpecies] * hs[iSpecies]) * Area;
          Res_Visc[nSpecies + nDim + 1] += (Res_Visc[iSpecies] * eves[iSpecies]) * Area;
        }
        AddRow(a.res, iPoint, nVar, Res_Visc, -1.0);
      });
}

/*! \brief CNEMONSSolver::BC_HeatFluxNonCatalytic_Wall. */
template <int ND>
void KernelHeatFlux(const ResArgs& r, unsigned long v0, unsigned long v1) {
  Kokkos::parallel_for(
      "NEMO::BC_HeatFlux_Wall", Kokkos::RangePolicy<exec_space>(v0, v1), KOKKOS_LAMBDA(const unsigned long kv) {
        const UpwArgs& a = r.a;
        const int nSpecies = a.ns, nVar = a.nVar, nDim = ND;
        const NEMOPrimIndices idx(nSpecies, ND);
        const auto iPoint = r.vtxNode(kv);
        if (iPoint >= r.nPointDomain) return;

        const double pcontrol = 1.0;
        double Normal[ND];
        for (int iDim = 0; iDim < nDim; iDim++) Normal[iDim] = r.vtxNormal(kv, iDim);
        double sq = 0.0;
        for (int iDim = 0; iDim < nDim; ++iDim) sq += Normal[iDim] * Normal[iDim];
        const double Area = Kokkos::sqrt(sq);

        double Res_Visc[MAXNV];
        for (int iVar = 0; iVar < nVar; iVar++) Res_Visc[iVar] = 0.0;

        double dTdn = 0.0;
        double dTvedn = 0.0;
        for (int iDim = 0; iDim < nDim; iDim++) {
          dTdn += r.grad(iPoint, idx.T(), iDim) * Normal[iDim];
          dTvedn += r.grad(iPoint, idx.TVE(), iDim) * Normal[iDim];
        }
        const double ktr = r.k(iPoint);
        const double kve = r.kve(iPoint);

        Res_Visc[nSpecies + nDim] += pcontrol * (ktr * dTdn + kve * dTvedn) + r.heatFlux * Area;
        Res_Visc[nSpecies + nDim + 1] += pcontrol * (kve * dTvedn) + r.heatFlux * Area;

        WallMomentumRows(r, iPoint, ND);
        AddRow(a.res, iPoint, nVar, Res_Visc, -1.0);

        if (a.implicit) {
          for (int iVar = nSpecies; iVar < nSpecies + nDim; iVar++) DeleteValsRowi(r, iPoint, iVar);
        }
      });
}


/*!
 * \brief Source kernel: CNEMOEulerSolver::Source_Residual loop body (monoatomic / frozen / implicit options,
 *        CSource_NEMO::ComputeChemistry then ComputeVibRelaxation, both on the same point state; the chemistry "ws"
 *        is passed to the V-T term explicitly, as the CPU passes it through the gas-model member).
 */
template <int ND>
void KernelSource(const ResArgs& r, unsigned long& eChm, unsigned long& eVib, unsigned long& eAxi) {
  Kokkos::parallel_reduce(
      "NEMO::SourceChemVT", Kokkos::RangePolicy<exec_space>(0, r.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint, unsigned long& nChm, unsigned long& nVib, unsigned long& nAxi) {
        const UpwArgs& a = r.a;
        const NEMOGasTables& g = a.gas;
        const int nSpecies = a.ns, nVar = a.nVar;
        const bool implicit = a.implicit;

        const double* V = &a.V(iPoint, 0);
        const double* eve = &a.eves(iPoint, 0);
        const double* cvve = &r.Cvves(iPoint, 0);
        const double* dTdU = implicit ? &r.dTdU(iPoint, 0) : nullptr;
        const double* dTvedU = implicit ? &r.dTvedU(iPoint, 0) : nullptr;
        const double Volume = r.volume(iPoint);

        double residual[MAXNV], jacobian[MAXNV * MAXNV];
        double ws[MAXNS];
        for (int s = 0; s < nSpecies; s++) ws[s] = 0.0;

        /*--- Compute finite rate chemistry (CSource_NEMO::ComputeChemistry) ---*/
        if (!r.monoatomic && !r.frozen) {
          for (int iVar = 0; iVar < nVar; iVar++) residual[iVar] = 0.0;
          if (implicit)
            for (int k = 0; k < nVar * nVar; k++) jacobian[k] = 0.0;

          ComputeNetProductionRates(g, ND, implicit, V, eve, cvve, dTdU, dTvedU, ws, jacobian);

          for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) residual[iSpecies] = ws[iSpecies] * Volume;
          if (implicit)
            for (int k = 0; k < nVar * nVar; k++) jacobian[k] *= Volume;

          if (!ResidualNaN(implicit, nVar, residual, jacobian)) {
            AddRow(a.res, iPoint, nVar, residual, -1.0);
            if (implicit) AddDiag(a, iPoint, jacobian, -1.0);
          } else {
            nChm++;
          }
        }

        /*--- Compute vibrational energy relaxation (CSource_NEMO::ComputeVibRelaxation) ---*/
        if (!r.monoatomic) {
          const double res_min = -1E6;
          const double res_max = 1E6;
          for (int iVar = 0; iVar < nVar; iVar++) residual[iVar] = 0.0;
          if (implicit)
            for (int k = 0; k < nVar * nVar; k++) jacobian[k] = 0.0;

          const double T = V[nSpecies];
          const double Tve = V[nSpecies + 1];
          double taus[MAXNS], eve_eq[MAXNS], eve_v[MAXNS];
          const double VTterm = ComputeEveSourceTerm(g, V, T, Tve, ws, taus, eve_eq, eve_v);
          if (implicit) GetEveSourceTermJacobian(g, ND, V, T, eve, cvve, dTdU, dTvedU, taus, eve_eq, jacobian);

          residual[nSpecies + ND + 1] = VTterm * Volume;
          if (implicit)
            for (int k = 0; k < nVar * nVar; k++) jacobian[k] *= Volume;

          /*--- Relax/limit vt transfer ---*/
          if (r.vtLimiting) {
            if (residual[nSpecies + ND + 1] > res_max) residual[nSpecies + ND + 1] = res_max;
            if (residual[nSpecies + ND + 1] < res_min) residual[nSpecies + ND + 1] = res_min;
          }

          if (!ResidualNaN(implicit, nVar, residual, jacobian)) {
            AddRow(a.res, iPoint, nVar, residual, -1.0);
            if (implicit) AddDiag(a, iPoint, jacobian, -1.0);
          } else {
            nVib++;
          }
        }

        /*--- Batch A: axisymmetric source terms (CSource_NEMO::ComputeAxisymmetric, 2D), CPU statement order. ---*/
        if (r.axisym) {
          const int RHO_INDEX = nSpecies + ND + 3, H_INDEX = nSpecies + ND + 4, T_INDEX = nSpecies;
          const int TVE_INDEX = nSpecies + 1, VEL_INDEX = nSpecies + 2, RHOCVTR_INDEX = nSpecies + ND + 6;
          const double TWO3 = 2.0 / 3.0;
          const double* U_i = &r.U(iPoint, 0);
          const double* dPdU_i = implicit ? &a.dPdU(iPoint, 0) : nullptr;
          const double Coord_y = r.coord(iPoint, 1);

          const double rho = V[RHO_INDEX];
          const double RuSI = r.RuSI;
          const double Ru = 1000.0 * RuSI;
          const double rhou = U_i[nSpecies];
          const double rhov = U_i[nSpecies + 1];
          const double H = V[H_INDEX];
          const double rhoEve = U_i[nVar - 1];
          double hs[MAXNS];
          ComputeSpeciesEnthalpy(g, V[T_INDEX], V[TVE_INDEX], eve, hs);

          for (int iVar = 0; iVar < nVar; iVar++) residual[iVar] = 0.0;

          double yinv = 0.0;
          if (Coord_y != 0.0) yinv = 1.0 / Coord_y;
          else yinv = 0.0;

          double Y[MAXNS];
          for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) Y[iSpecies] = V[iSpecies] / rho;

          for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) residual[iSpecies] = yinv * rhov * Y[iSpecies] * Volume;
          residual[nSpecies] = yinv * rhov * rhou / rho * Volume;
          residual[nSpecies + 1] = yinv * rhov * rhov / rho * Volume;
          residual[nSpecies + 2] = yinv * rhov * H * Volume;
          residual[nSpecies + 3] = yinv * rhov * rhoEve / rho * Volume;

          if (implicit) {
            for (int k = 0; k < nVar * nVar; k++) jacobian[k] = 0.0;
            double dYdr[MAXNS * MAXNS];
            for (int k = 0; k < nSpecies * nSpecies; k++) dYdr[k] = 0.0;

            for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
              for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++) {
                dYdr[iSpecies * nSpecies + jSpecies] += -1 / rho * Y[iSpecies];
              }
              dYdr[iSpecies * nSpecies + iSpecies] += 1 / rho;
            }

            for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
              for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++) {
                jacobian[(iSpecies) * nVar + (jSpecies)] = dYdr[iSpecies * nSpecies + jSpecies] * rhov;
              }
              jacobian[(iSpecies) * nVar + (nSpecies + 1)] = Y[iSpecies];
            }

            for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) jacobian[(nSpecies) * nVar + (iSpecies)] = -rhou * rhov / (rho * rho);
            jacobian[(nSpecies) * nVar + (nSpecies)] = rhov / rho;
            jacobian[(nSpecies) * nVar + (nSpecies + 1)] = rhou / rho;

            for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) jacobian[(nSpecies + 1) * nVar + (iSpecies)] = -rhov * rhov / (rho * rho);
            jacobian[(nSpecies + 1) * nVar + (nSpecies + 1)] = 2 * rhov / rho;

            for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++)
              jacobian[(nSpecies + ND) * nVar + (iSpecies)] = -H * rhov / rho + dPdU_i[iSpecies] * rhov / rho;
            jacobian[(nSpecies + ND) * nVar + (nSpecies)] = dPdU_i[nSpecies] * rhov / rho;
            jacobian[(nSpecies + ND) * nVar + (nSpecies + 1)] = H + dPdU_i[nSpecies + 1] * rhov / rho;
            jacobian[(nSpecies + ND) * nVar + (nSpecies + ND)] = (1 + dPdU_i[nSpecies + ND]) * rhov / rho;
            jacobian[(nSpecies + ND) * nVar + (nSpecies + ND + 1)] = dPdU_i[nSpecies + ND + 1] * rhov / rho;

            for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) jacobian[(nSpecies + ND + 1) * nVar + (iSpecies)] = -rhoEve * rhov / (rho * rho);
            jacobian[(nSpecies + ND + 1) * nVar + (nSpecies + 1)] = rhoEve / rho;
            jacobian[(nSpecies + ND + 1) * nVar + (nSpecies + ND + 1)] = rhov / rho;

            for (int k = 0; k < nVar * nVar; k++) jacobian[k] *= yinv * Volume;
          }

          if (r.axiViscous) {
            const double turb_ke_i = 0.0; /*--- !rans ---*/
            const double ktr = r.k(iPoint);
            const double kve = r.kve(iPoint);
            const double Laminar_Viscosity_i = r.mu(iPoint);
            const double Eddy_Viscosity_i = a.V(iPoint, r.eddyIdx);
            const double Prandtl_Turb = r.prandtlTurb;

            double Vector = 0.0;
            for (int iSpecies = 0; iSpecies < r.nHeavy; iSpecies++)
              Vector += rho * r.Ds(iPoint, iSpecies) * r.grad(iPoint, iSpecies, 1);

            double Mass = 0.0;
            for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) Mass += V[iSpecies] / rho * g.MolarMass(iSpecies);

            const double heat_capacity_cp_i = V[RHOCVTR_INDEX] / rho + Ru / Mass;
            const double total_viscosity_i = Laminar_Viscosity_i + Eddy_Viscosity_i;
            const double total_conductivity_i = ktr + kve + heat_capacity_cp_i * Eddy_Viscosity_i / Prandtl_Turb;
            const double u = V[VEL_INDEX];
            const double v = V[VEL_INDEX + 1];
            const double qy_t = -total_conductivity_i * r.grad(iPoint, T_INDEX, 1);
            const double qy_ve = -kve * r.grad(iPoint, TVE_INDEX, 1);

            double sumJhs_y, sumJeve_y;
            sumJhs_y = sumJeve_y = 0.0;
            for (int iSpecies = 0; iSpecies < r.nHeavy; iSpecies++) {
              sumJhs_y += -(rho * r.Ds(iPoint, iSpecies) * r.grad(iPoint, iSpecies, 1) - V[iSpecies] * Vector) * hs[iSpecies];
              sumJeve_y += -(rho * r.Ds(iPoint, iSpecies) * r.grad(iPoint, iSpecies, 1) - V[iSpecies] * Vector) * eve[iSpecies];
            }

            for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) residual[iSpecies] -= 0.0;
            residual[nSpecies] -= Volume * (yinv * total_viscosity_i * (r.grad(iPoint, nSpecies + 2, 1) + r.grad(iPoint, nSpecies + 3, 0)) -
                                            TWO3 * r.auxGrad(iPoint, 0, 0));
            residual[nSpecies + 1] -= Volume * (yinv * total_viscosity_i * 2 * (r.grad(iPoint, nSpecies + 3, 1) - v * yinv) -
                                                TWO3 * r.auxGrad(iPoint, 0, 1));
            residual[nSpecies + 2] -=
                Volume * (yinv * (-sumJhs_y +
                                  total_viscosity_i * (u * (r.grad(iPoint, nSpecies + 3, 0) + r.grad(iPoint, nSpecies + 2, 1)) +
                                                       v * TWO3 * (2 * r.grad(iPoint, nSpecies + 3, 1) - r.grad(iPoint, nSpecies + 2, 0) -
                                                                   v * yinv + rho * turb_ke_i)) -
                                  qy_t) -
                          TWO3 * (r.auxGrad(iPoint, 1, 1) + r.auxGrad(iPoint, 2, 0)));
            residual[nSpecies + 3] -= Volume * (yinv * (-sumJeve_y - qy_ve));
          }

          if (!ResidualNaN(implicit, nVar, residual, jacobian)) {
            AddRow(a.res, iPoint, nVar, residual, 1.0);
            if (implicit) AddDiag(a, iPoint, jacobian, 1.0);
          } else {
            nAxi++;
          }
        }
      },
      Kokkos::Sum<unsigned long>(eChm), Kokkos::Sum<unsigned long>(eVib), Kokkos::Sum<unsigned long>(eAxi));
}

/*! \brief CNEMOEulerSolver::BC_Sym_Plane (NEMO override; also BC_Euler_Wall) for the vertices [v0, v1). */
template <int ND>
void KernelSymmetry(const ResArgs& r, unsigned long v0, unsigned long v1) {
  Kokkos::parallel_for(
      "NEMO::BC_Sym_Plane", Kokkos::RangePolicy<exec_space>(v0, v1), KOKKOS_LAMBDA(const unsigned long k) {
        const UpwArgs& a = r.a;
        const int nSpecies = a.ns, nVar = a.nVar, nDim = ND;
        const NEMOPrimIndices idx(nSpecies, ND);
        const auto iPoint = r.vtxNode(k);
        if (iPoint >= r.nPointDomain) return;

        double Normal[ND], UnitNormal[ND];
        for (int iDim = 0; iDim < nDim; iDim++) Normal[iDim] = r.vtxNormal(k, iDim);

        double sq = 0.0;
        for (int iDim = 0; iDim < nDim; ++iDim) sq += Normal[iDim] * Normal[iDim];
        const double Area = Kokkos::sqrt(sq);

        for (int iDim = 0; iDim < nDim; iDim++) UnitNormal[iDim] = -Normal[iDim] / Area;

        /*--- Retrieve the pressure on the vertex ---*/
        const double P = a.V(iPoint, idx.P());

        /*--- Apply the flow-tangency b.c. to the convective flux ---*/
        double Residual[MAXNV];
        for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) Residual[iSpecies] = 0.0;
        for (int iDim = 0; iDim < nDim; iDim++) Residual[nSpecies + iDim] = P * UnitNormal[iDim] * Area;
        Residual[nSpecies + nDim] = 0.0;
        Residual[nSpecies + nDim + 1] = 0.0;

        AddRow(a.res, iPoint, nVar, Residual, 1.0);

        if (!a.implicit) return;

        double J[MAXNV * MAXNV];
#define JS(rr, cc) J[(rr) * nVar + (cc)]
        for (int q = 0; q < nVar * nVar; q++) J[q] = 0.0;

        const double rho = a.V(iPoint, idx.RHO());
        const double rhoE = r.U(iPoint, nSpecies + nDim);
        const double rhoEve = r.U(iPoint, nSpecies + nDim + 1);
        const double* dPdU = &a.dPdU(iPoint, 0);
        double Velocity[ND];
        for (int iDim = 0; iDim < nDim; iDim++) Velocity[iDim] = a.V(iPoint, idx.VEL() + iDim);

        for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
          const double cs = a.V(iPoint, idx.RHOS() + iSpecies) / a.V(iPoint, idx.RHO());
          for (int iDim = 0; iDim < nDim; iDim++) {
            JS(nSpecies + iDim, iSpecies) = dPdU[iSpecies] * UnitNormal[iDim];
            JS(iSpecies, nSpecies + iDim) = cs * UnitNormal[iDim];
          }
        }

        for (int iDim = 0; iDim < nDim; iDim++) {
          for (int jDim = 0; jDim < nDim; jDim++) {
            JS(nSpecies + iDim, nSpecies + jDim) =
                Velocity[iDim] * UnitNormal[jDim] + dPdU[nSpecies + jDim] * UnitNormal[iDim];
          }
          JS(nSpecies + iDim, nSpecies + nDim) = dPdU[nSpecies + nDim] * UnitNormal[iDim];
          JS(nSpecies + iDim, nSpecies + nDim + 1) = dPdU[nSpecies + nDim + 1] * UnitNormal[iDim];

          JS(nSpecies + nDim, nSpecies + iDim) = (rhoE + P) / rho * UnitNormal[iDim];
          JS(nSpecies + nDim + 1, nSpecies + iDim) = rhoEve / rho * UnitNormal[iDim];
        }
#undef JS
        for (int q = 0; q < nVar * nVar; q++) J[q] = J[q] * Area;

        AddDiag(a, iPoint, J, 1.0);
      });
}

/*!
 * \brief Convective boundary flux with the boundary numerics (AUSM family, same device function as the edges):
 *        FAR_FIELD (j = node_infty), OUTLET (j = characteristic outlet state, CNEMOEulerSolver::BC_Outlet),
 *        SUPERSONIC_INLET (j = node_inlet), SUPERSONIC_OUTLET (j = i). LinSysRes.AddBlock + AddBlock2Diag(J_i).
 * \param[out] lastSub - OUTLET: largest vertex index on the subsonic branch (-1 if none).
 */
template <int ND>
void KernelBoundaryFlux(const ResArgs& r, int kind, unsigned long v0, unsigned long v1, long& lastSub) {
  using K = CNEMOKokkosData::BCKind;
  Kokkos::parallel_reduce(
      "NEMO::BC_Flux", Kokkos::RangePolicy<exec_space>(v0, v1),
      KOKKOS_LAMBDA(const unsigned long k, long& last) {
        const UpwArgs& a = r.a;
        const NEMOGasTables& g = a.gas;
        const int nSpecies = a.ns, nVar = a.nVar, nDim = ND, nPrimVar = a.nPrimVar;
        const NEMOPrimIndices idx(nSpecies, ND);
        const auto iPoint = r.vtxNode(k);
        if (iPoint >= r.nPointDomain) return;

        /*--- Normal vector for this vertex (negate for outward convention) ---*/
        double Normal[ND];
        for (int iDim = 0; iDim < nDim; iDim++) Normal[iDim] = -r.vtxNormal(k, iDim);

        const double zero[MAXNV] = {0.0};
        const double* V_i = &a.V(iPoint, 0);
        const double* eve_i = &a.eves(iPoint, 0);
        const double* dPdU_i = a.implicit ? &a.dPdU(iPoint, 0) : zero;
        double Gamma_i = a.Gamma(iPoint);
        double Sensor_i = a.Sensor.size() ? a.Sensor(iPoint) : 0.0;

        /*--- Boundary state (node_infty / node_inlet): V | eve | dPdU | Gamma. ---*/
        const double* Vb = r.bstate.size() ? &r.bstate(0) : zero;
        const double* eveb = r.bstate.size() ? &r.bstate(nPrimVar) : zero;
        const double* dPdUb = (r.bstate.size() && a.implicit) ? &r.bstate(nPrimVar + nSpecies) : zero;
        const double Gammab = r.bstate.size() ? r.bstate(nPrimVar + nSpecies + nVar) : 0.0;

        const double *V_j = Vb, *eve_j = eveb, *dPdU_j = dPdUb;
        double Gamma_j = Gammab;

        if (kind == static_cast<int>(K::SUPERSONIC_OUTLET)) {
          V_j = V_i;
          eve_j = eve_i;
          dPdU_j = dPdU_i;
          Gamma_j = Gamma_i;
        } else if (kind == static_cast<int>(K::OUTLET)) {
          /*====================== CNEMOEulerSolver::BC_Outlet, fictitious state ======================*/
          double* V_outlet = &r.charac(k, 0);
          const double* U_domain = &r.U(iPoint, 0);
          const double* V_domain = V_i;
          double U_outlet[MAXNV], rhos[MAXNS], Ys[MAXNS];

          double sq = 0.0;
          for (int iDim = 0; iDim < nDim; ++iDim) sq += Normal[iDim] * Normal[iDim];
          const double Area = Kokkos::sqrt(sq);
          double UnitNormal[ND];
          for (int iDim = 0; iDim < nDim; iDim++) UnitNormal[iDim] = Normal[iDim] / Area;

          for (int iVar = 0; iVar < nVar; iVar++) U_outlet[iVar] = 0.0;
          for (int iVar = 0; iVar < nPrimVar; iVar++) V_outlet[iVar] = 0.0;

          const double Gamma = Gamma_i;
          const double Gamma_Minus_One = Gamma - 1.0;
          const double P_Exit = r.P_Exit;

          double Density = V_domain[idx.RHO()];
          double Velocity2 = 0.0;
          double Vn = 0.0;
          double Velocity[ND];
          for (int iDim = 0; iDim < nDim; iDim++) {
            Velocity[iDim] = V_domain[idx.VEL() + iDim];
            Velocity2 += Velocity[iDim] * Velocity[iDim];
            Vn += Velocity[iDim] * UnitNormal[iDim];
          }
          const double Temperature = V_domain[idx.T()];
          const double Tve = V_domain[idx.TVE()];
          double Pressure = V_domain[idx.P()];
          double SoundSpeed = V_domain[idx.A()];
          const double Mach_Exit = Kokkos::sqrt(Velocity2) / SoundSpeed;

          for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) Ys[iSpecies] = V_domain[iSpecies] / Density;

          if (Mach_Exit >= 1.0) {
            for (int iVar = 0; iVar < nVar; iVar++) U_outlet[iVar] = U_domain[iVar];
            for (int iVar = 0; iVar < nPrimVar; iVar++) V_outlet[iVar] = V_domain[iVar];
          } else {
            const double Entropy = Pressure * Kokkos::pow(1.0 / Density, Gamma);
            const double Riemann = Vn + 2.0 * SoundSpeed / Gamma_Minus_One;

            Density = Kokkos::pow(P_Exit / Entropy, 1.0 / Gamma);
            Pressure = P_Exit;
            SoundSpeed = Kokkos::sqrt(Gamma * P_Exit / Density);
            const double Vn_Exit = Riemann - 2.0 * SoundSpeed / Gamma_Minus_One;
            Velocity2 = 0.0;
            for (int iDim = 0; iDim < nDim; iDim++) {
              Velocity[iDim] = Velocity[iDim] + (Vn_Exit - Vn) * UnitNormal[iDim];
              Velocity2 += Velocity[iDim] * Velocity[iDim];
            }

            for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
              V_outlet[iSpecies] = Ys[iSpecies] * Density;
              rhos[iSpecies] = V_outlet[iSpecies];
            }

            V_outlet[idx.T()] = V_domain[idx.T()];
            V_outlet[idx.TVE()] = V_domain[idx.TVE()];
            for (int iDim = 0; iDim < nDim; iDim++) V_outlet[idx.VEL() + iDim] = Velocity[iDim];

            V_outlet[idx.P()] = Pressure;
            V_outlet[idx.RHO()] = Density;
            V_outlet[idx.A()] = SoundSpeed;

            /*--- Set mixture state and compute quantities (SetTDStateRhosTTv(rhos, Temperature, Tve)) ---*/
            V_outlet[idx.RHOCVTR()] = ComputerhoCvtr(g, rhos);
            V_outlet[idx.RHOCVVE()] = ComputerhoCvve(g, rhos, Tve);
            double energies[2];
            ComputeMixtureEnergies(g, rhos, Temperature, Tve, energies);

            for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) U_outlet[iSpecies] = V_outlet[iSpecies];
            for (int iDim = 0; iDim < nDim; iDim++) U_outlet[nSpecies + iDim] = Velocity[iDim] * Density;
            U_outlet[nVar - 2] = (energies[0] + 0.5 * Velocity2) * Density;
            U_outlet[nVar - 1] = energies[1] * Density;

            last = kmaxl(last, long(k));
          }

          /*--- Setting Last remaining variables ---*/
          V_outlet[idx.H()] = (U_outlet[nVar - 2] + Pressure) / Density;
          V_j = V_outlet;
        }

        /*--- Batch A: BC_Supersonic_Inlet sets neither Gamma nor Sensor of the boundary numerics, BC_Supersonic_Outlet
         *    no Sensor: AUSM+M reads the values the last earlier BC left there (host replay, staleNumerics). ---*/
        if (r.staleNumerics) {
          if (kind == static_cast<int>(K::SUPERSONIC_INLET)) {
            Gamma_i = r.staleGammaI;
            Gamma_j = r.staleGammaJ;
          }
          Sensor_i = r.staleSensor;
        }
        const double Sensor_j = Sensor_i; /*--- SetSensor(Sensor(iPoint), Sensor(iPoint)) on the CPU ---*/
        double Flux[MAXNV], Jacobian_i[MAXNV * MAXNV], Jacobian_j[MAXNV * MAXNV];
        AUSMResidual<ND>(a, Normal, V_i, V_j, eve_i, eve_j, dPdU_i, dPdU_j, Gamma_i, Gamma_j, Sensor_i, Sensor_j, Flux,
                         Jacobian_i, Jacobian_j);

        AddRow(a.res, iPoint, nVar, Flux, 1.0);
        if (a.implicit) AddDiag(a, iPoint, Jacobian_i, 1.0);

        /*--- Step 8: viscous contribution of BC_Far_Field (CAvgGrad_NEMO, node_infty, reflected normal neighbour;
         *    transport properties of iPoint on both sides). ---*/
        if (r.viscous) {
          using namespace nemo_visc;
          const auto Point_Normal = static_cast<unsigned long>(r.vtxNormalNeighbor(k));
          double Coord_i[ND], Coord_Reflected[ND];
          for (int iDim = 0; iDim < nDim; iDim++) {
            Coord_i[iDim] = r.coord(iPoint, iDim);
            Coord_Reflected[iDim] = 2 * r.coord(iPoint, iDim) - r.coord(Point_Normal, iDim);
          }
          const double* bv = &r.bstateVisc(0);
          const int ofsGrad = nSpecies + 2 * nVar;
          ViscPoint pi, pj;
          pi.V = V_i;
          pi.G = &r.grad(iPoint, 0, 0);
          pi.eve = eve_i;
          pi.cvve = &r.Cvves(iPoint, 0);
          pi.dTdU = a.implicit ? &r.dTdU(iPoint, 0) : zero;
          pi.dTvedU = a.implicit ? &r.dTvedU(iPoint, 0) : zero;
          pi.Ds = &r.Ds(iPoint, 0);
          pi.mu = r.mu(iPoint);
          pi.eddy = a.V(iPoint, r.eddyIdx);
          pi.k = r.k(iPoint);
          pi.kve = r.kve(iPoint);
          pj = pi;
          pj.V = Vb;
          pj.G = bv + ofsGrad;
          pj.eve = eveb;
          pj.cvve = bv;
          pj.dTdU = a.implicit ? bv + nSpecies : zero;
          pj.dTvedU = a.implicit ? bv + nSpecies + nVar : zero;

          double FluxV[MAXNV], JiV[MAXNV * MAXNV], JjV[MAXNV * MAXNV];
          AvgGradResidual<ND>(r.vc, false, Coord_i, Coord_Reflected, Normal, pi, pj, FluxV, JiV, JjV);
          AddRow(a.res, iPoint, nVar, FluxV, -1.0);
          if (a.implicit) AddDiag(a, iPoint, JiV, -1.0);
        }
      },
      Kokkos::Max<long>(lastSub));
}

}  // namespace

/*============================== residual accumulator ==============================*/

void CNEMOKokkosData::EnsureAccumulator(bool impl) {
  if (st.Res.size() == 0) st.Res = View2D<double>("NEMO::Residual", nPoint, nVar);
  if (impl) {
    if (!geo.has_pattern || geo.nnz == 0)
      SU2_MPI::Error("CNEMOKokkosData: implicit device residual needs the FV sparse pattern.", CURRENT_FUNCTION);
    if (st.JacVal.size() == 0) st.JacVal = View1D<double>("NEMO::Jacobian", geo.nnz * size_t(nVar) * nVar);
  }
}

void CNEMOKokkosData::AccumulatorToDevice(bool impl, const double* hostRes, const double* hostJac) {
  if (accOnDevice) return;
  auto& P = stagePerf[ST_RESSYNC];
  Kokkos::Timer timer;
  EnsureAccumulator(impl);
  accImplicit = impl;
  if (hostAccZero) {
    Kokkos::deep_copy(st.Res, 0.0);
    if (impl) Kokkos::deep_copy(st.JacVal, 0.0);
  } else {
    const size_t nr = size_t(nPoint) * nVar;
    Kokkos::deep_copy(st.Res, Kokkos::View<const double**, Kokkos::LayoutRight, HostSpace, Unmanaged>(hostRes, nPoint, nVar));
    P.bytes_h2d += double(nr) * sizeof(double);
    if (impl) {
      Kokkos::deep_copy(st.JacVal, Kokkos::View<const double*, HostSpace, Unmanaged>(hostJac, st.JacVal.size()));
      P.bytes_h2d += double(st.JacVal.size()) * sizeof(double);
    }
  }
  Kokkos::fence();
  accOnDevice = true;
  hostAccZero = false;
  MarkDeviceWritten(G_LINSYS); /*--- resident mode: st.Res is the device LinSysRes ---*/
  P.h2d += timer.seconds();
}

void CNEMOKokkosData::AccumulatorToHost(double* hostRes, double* hostJac) {
  if (!accOnDevice) return;
  auto& P = stagePerf[ST_RESSYNC];
  Kokkos::Timer timer;
  Kokkos::deep_copy(Kokkos::View<double**, Kokkos::LayoutRight, HostSpace, Unmanaged>(hostRes, nPoint, nVar), st.Res);
  P.bytes_d2h += double(st.Res.size()) * sizeof(double);
  if (accImplicit) {
    Kokkos::deep_copy(Kokkos::View<double*, HostSpace, Unmanaged>(hostJac, st.JacVal.size()), st.JacVal);
    P.bytes_d2h += double(st.JacVal.size()) * sizeof(double);
  }
  accOnDevice = false;
  hostAccZero = false;
  P.d2h += timer.seconds();
  P.calls++;
}

void CNEMOKokkosData::MarkHostAccumulatorZero() {
  if (accOnDevice) accDropped++;
  accOnDevice = false;
  hostAccZero = true;
}

void CNEMOKokkosData::EnsureInputsOnDevice(bool upload, bool needSolution, bool needSensor) {
  if (upload || !primDeviceValid || !secDeviceValid) {
    UploadPrimitive();
    primDeviceValid = secDeviceValid = true;
  }
  if (needSolution && (upload || !consDeviceValid)) {
    if (static_cast<unsigned long>(nodes->Solution.rows()) != nPoint ||
        static_cast<int>(nodes->Solution.cols()) != nVar)
      SU2_MPI::Error("CNEMOKokkosData: unexpected shape of the host Solution.", CURRENT_FUNCTION);
    Kokkos::deep_copy(st.Solution, Kokkos::View<const double**, Kokkos::LayoutRight, HostSpace, Unmanaged>(
                                       nodes->Solution.data(), nPoint, nVar));
    consDeviceValid = true;
  }
  if (needSensor && ResidentActive()) {
    EnsureDevice(G_SENSOR); /*--- device SetPressureDiffusionSensor (batch A) ---*/
  } else if (needSensor && (upload || !sensorValid)) {
    if (st.Sensor.size() == 0) st.Sensor = View1D<double>("NEMO::Sensor", nPoint);
    Kokkos::deep_copy(st.Sensor, Kokkos::View<const double*, HostSpace, Unmanaged>(nodes->GetSensor().data(), nPoint));
    sensorValid = true;
  }
}

/*============================== source terms ==============================*/

CNEMOKokkosData::SourceResult CNEMOKokkosData::ComputeSourceResidual(const SourceParams& p, bool uploadInputs) {
  auto& P = stagePerf[ST_SOURCE];
  Kokkos::Timer timer;
  SourceResult out;
  if (!accOnDevice) SU2_MPI::Error("CNEMOKokkosData: source stage without a device accumulator.", CURRENT_FUNCTION);
  /*--- Batch A: the axisymmetric terms also read U (rho u, rho v, rho Eve) and, viscous, the primitive gradient, the
   *    transport properties and the AuxVar gradient. ---*/
  EnsureInputsOnDevice(uploadInputs, p.axisymmetric, false);
  if (p.axiViscous) {
    if (ResidentActive()) {
      EnsureDevice(G_GRADIENT);
      EnsureDevice(G_AUXVAR);
    } else {
      if (uploadInputs || !gradPrimValid) {
        AllocateGradients();
        Kokkos::deep_copy(st.GradPrim, Kokkos::View<const double***, Kokkos::LayoutRight, HostSpace, Unmanaged>(
                                           nodes->GetGradient_Primitive().data(), nPoint, nPrimVarGrad, nDim));
        P.bytes_h2d += double(st.GradPrim.size()) * sizeof(double);
        gradPrimValid = true;
        if (!separateRecGradient) gradRecValid = true;
      }
      if (uploadInputs || !transportDeviceValid) UploadTransport();
      if (uploadInputs) UploadAuxVar();
    }
  }
  Kokkos::fence();
  P.h2d += timer.seconds();
  timer.reset();

  ResArgs r;
  r.a.dia_ptr = geo.dia_ptr;
  r.a.V = st.Primitive;
  r.a.eves = st.eves;
  r.a.res = st.Res;
  r.a.jac = st.JacVal;
  r.a.gas = gas;
  r.a.nDim = nDim;
  r.a.ns = nSpecies;
  r.a.nVar = nVar;
  r.a.nPrimVar = nPrimVar;
  r.a.implicit = p.implicit;
  r.Cvves = st.Cvves;
  r.dTdU = st.dTdU;
  r.dTvedU = st.dTvedU;
  r.volume = geo.volume;
  r.nPointDomain = geo.nPointDomain;
  r.frozen = p.frozen;
  r.monoatomic = p.monoatomic;
  r.vtLimiting = p.vtLimiting;
  r.axisym = p.axisymmetric;
  r.axiViscous = p.axiViscous;
  r.nHeavy = p.nHeavyNumerics;
  r.prandtlTurb = p.prandtlTurb;
  r.RuSI = p.RuSI;
  r.eddyIdx = nSpecies + nDim + 9;
  if (p.axisymmetric) {
    r.a.dPdU = st.dPdU;
    r.U = st.Solution;
    r.coord = geo.coord;
  }
  if (p.axiViscous) {
    r.grad = st.GradPrim;
    r.auxGrad = st.AuxGrad;
    r.Ds = st.DiffCoeff;
    r.mu = st.LamVisc;
    r.k = st.ThermalCond;
    r.kve = st.ThermalCondVe;
  }

  if (nDim == 2) KernelSource<2>(r, out.eChm, out.eVib, out.eAxi);
  else KernelSource<3>(r, out.eChm, out.eVib, out.eAxi);
  P.kernel += timer.seconds();
  P.calls++;
  return out;
}

/*============================== boundary conditions ==============================*/

CNEMOKokkosData::BCResult CNEMOKokkosData::ComputeBCResidual(const BCParams& p, const BoundaryState& bs,
                                                             bool uploadInputs) {
  auto& P = stagePerf[ST_BC];
  Kokkos::Timer timer;
  BCResult out;
  if (!accOnDevice) SU2_MPI::Error("CNEMOKokkosData: BC stage without a device accumulator.", CURRENT_FUNCTION);

  const bool wall = (p.kind == BCKind::ISOTHERMAL || p.kind == BCKind::HEAT_FLUX);
  const bool flux = (p.kind != BCKind::SYMMETRY) && !wall;
  const bool ausmPlusM = flux && (p.upw.scheme == UpwindScheme::AUSMPLUSM);
  const bool needState = (p.kind == BCKind::FAR_FIELD || p.kind == BCKind::OUTLET || p.kind == BCKind::SUPERSONIC_INLET);
  const bool viscFar = (p.kind == BCKind::FAR_FIELD) && p.viscous;
  EnsureInputsOnDevice(uploadInputs, p.kind == BCKind::SYMMETRY || p.kind == BCKind::OUTLET, ausmPlusM);

  /*--- Step 8: NS inputs (gradient, transport), host-authoritative mode only (resident: always on the device). ---*/
  if (wall || viscFar) {
    if (!viscousVars) SU2_MPI::Error("CNEMOKokkosData: viscous BC without NS variables.", CURRENT_FUNCTION);
    const bool needGrad = viscFar || p.kind == BCKind::HEAT_FLUX;
    if (needGrad && (uploadInputs || !gradPrimValid)) {
      if (ResidentActive()) SU2_MPI::Error("CNEMOKokkosData: device-resident gradient not valid.", CURRENT_FUNCTION);
      AllocateGradients();
      Kokkos::deep_copy(st.GradPrim, Kokkos::View<const double***, Kokkos::LayoutRight, HostSpace, Unmanaged>(
                                         nodes->GetGradient_Primitive().data(), nPoint, nPrimVarGrad, nDim));
      P.bytes_h2d += double(st.GradPrim.size()) * sizeof(double);
      gradPrimValid = true;
      if (!separateRecGradient) gradRecValid = true;
    }
    if (uploadInputs || !transportDeviceValid) {
      if (ResidentActive()) SU2_MPI::Error("CNEMOKokkosData: device-resident transport not valid.", CURRENT_FUNCTION);
      UploadTransport();
    }
    /*--- Host-authoritative mode: the caller updates the host Solution_Old / Res_TruncError rows itself. ---*/
    if (wall && ResidentActive()) {
      EnsureDevice(G_SOLUTION_OLD);
      EnsureDevice(G_TRUNCERR);
    }
  }
  if (viscFar) {
    /*--- node_infty Cvve | dTdU | dTvedU | gradient, uploaded when it changes (constant in practice). ---*/
    const size_t ng = size_t(nPrimVarGrad) * nDim;
    const size_t n = size_t(nSpecies) + 2 * size_t(nVar) + ng;
    std::vector<double> h(n, 0.0);
    std::copy(bs.Cvve, bs.Cvve + nSpecies, h.begin());
    if (p.upw.implicit) {
      std::copy(bs.dTdU, bs.dTdU + nVar, h.begin() + nSpecies);
      std::copy(bs.dTvedU, bs.dTvedU + nVar, h.begin() + nSpecies + nVar);
    }
    std::copy(bs.grad, bs.grad + ng, h.begin() + nSpecies + 2 * nVar);
    if (st.BStateVisc.size() != n) st.BStateVisc = View1D<double>("NEMO::BoundaryStateViscous", n);
    if (uploadInputs || bstateViscCache.size() != n ||
        std::memcmp(h.data(), bstateViscCache.data(), n * sizeof(double)) != 0) {
      Kokkos::deep_copy(st.BStateVisc, Kokkos::View<const double*, HostSpace, Unmanaged>(h.data(), n));
      bstateViscCache = h;
      P.bytes_h2d += double(n) * sizeof(double);
    }
  }

  if (needState) {
    /*--- V | eve | dPdU | Gamma, uploaded only when it differs from the cached copy (constant in practice). ---*/
    const size_t n = size_t(nPrimVar) + nSpecies + nVar + 1;
    std::vector<double> h(n, 0.0);
    std::copy(bs.V, bs.V + nPrimVar, h.begin());
    std::copy(bs.eve, bs.eve + nSpecies, h.begin() + nPrimVar);
    if (p.upw.implicit && bs.dPdU) std::copy(bs.dPdU, bs.dPdU + nVar, h.begin() + nPrimVar + nSpecies);
    h[n - 1] = bs.Gamma;
    if (st.BState.size() != n) st.BState = View1D<double>("NEMO::BoundaryState", n);
    if (uploadInputs || bstateCache.size() != n || std::memcmp(h.data(), bstateCache.data(), n * sizeof(double)) != 0) {
      Kokkos::deep_copy(st.BState, Kokkos::View<const double*, HostSpace, Unmanaged>(h.data(), n));
      bstateCache = h;
      P.bytes_h2d += double(n) * sizeof(double);
    }
  }
  if (p.kind == BCKind::OUTLET && st.Charac.size() == 0) {
    const auto nv = Kokkos::create_mirror_view_and_copy(HostSpace(), geo.marker_ptr)(geo.nMarker);
    st.Charac = View2D<double>("NEMO::CharacPrimVar", nv, nPrimVar);
  }
  Kokkos::fence();
  P.h2d += timer.seconds();
  timer.reset();

  ResArgs r;
  r.a.dia_ptr = geo.dia_ptr;
  r.a.V = st.Primitive;
  r.a.dPdU = st.dPdU;
  r.a.eves = st.eves;
  r.a.Gamma = st.Gamma;
  r.a.Sensor = ausmPlusM ? st.Sensor : View1D<double>();
  r.a.res = st.Res;
  r.a.jac = st.JacVal;
  r.a.gas = gas;
  r.a.nDim = nDim;
  r.a.ns = nSpecies;
  r.a.nVar = nVar;
  r.a.nPrimVar = nPrimVar;
  r.a.nPrimVarGrad = nPrimVarGrad;
  r.a.nElNum = p.upw.nElNumerics;
  r.a.scheme = static_cast<int>(p.upw.scheme);
  r.a.implicit = p.upw.implicit;
  r.a.Minf = p.upw.Minf;
  r.a.pi = PI_NUMBER;
  r.U = st.Solution;
  r.charac = st.Charac;
  r.vtxNormal = geo.vtx_normal;
  r.vtxNode = geo.vtx_node;
  r.bstate = needState ? st.BState : View1D<double>();
  r.nPointDomain = geo.nPointDomain;
  r.P_Exit = p.P_Exit;
  r.staleNumerics = p.staleNumerics;
  r.staleGammaI = p.staleGammaI;
  r.staleGammaJ = p.staleGammaJ;
  r.staleSensor = p.staleSensor;
  /*--- Step 8. ---*/
  r.viscous = viscFar;
  if (wall || viscFar) {
    r.vc.gas = gas;
    r.vc.ns = nSpecies;
    r.vc.nVar = nVar;
    r.vc.nPrimVar = nPrimVar;
    r.vc.nPrimVarGrad = nPrimVarGrad;
    r.vc.nHeavy = nSpecies - p.upw.nElNumerics;
    r.vc.implicit = p.upw.implicit;
    r.vc.Ru = 1000.0 * UNIVERSAL_GAS_CONSTANT;
    r.grad = st.GradPrim;
    r.coord = geo.coord;
    r.Ds = st.DiffCoeff;
    r.mu = st.LamVisc;
    r.k = st.ThermalCond;
    r.kve = st.ThermalCondVe;
    r.Cvves = st.Cvves;
    r.dTdU = st.dTdU;
    r.dTvedU = st.dTvedU;
    r.bstateVisc = viscFar ? st.BStateVisc : View1D<double>();
    r.vtxNormalNeighbor = geo.vtx_normal_neighbor;
    r.rowPtr = geo.row_ptr;
    r.colInd = geo.col_ind;
    r.eddyIdx = nSpecies + nDim + 9;
    r.catalytic = p.catalytic;
    r.supercat = p.supercatalytic;
    for (int s2 = 0; s2 < nSpecies; s2++) r.Yst[s2] = p.Yst[s2];
    r.Twall = p.Twall;
    r.catEff = p.catEff;
    r.heatFlux = p.heatFlux;
    r.RuSI = UNIVERSAL_GAS_CONSTANT;
    r.writeWallState = wall && ResidentActive();
    if (r.writeWallState) {
      r.Uold = st.Solution_Old;
      r.trunc = st.TruncErr;
    }
  }

  /*--- Marker range (host copy of marker_ptr cached on first use). ---*/
  if (h_markerPtr.empty()) {
    auto hm = Kokkos::create_mirror_view_and_copy(HostSpace(), geo.marker_ptr);
    h_markerPtr.assign(hm.data(), hm.data() + geo.nMarker + 1);
  }
  const unsigned long v0 = h_markerPtr[p.marker], v1 = h_markerPtr[p.marker + 1];

  long lastSub = -1;
  if (p.kind == BCKind::ISOTHERMAL) {
    if (nDim == 2) KernelIsothermal<2>(r, v0, v1);
    else KernelIsothermal<3>(r, v0, v1);
  } else if (p.kind == BCKind::HEAT_FLUX) {
    if (nDim == 2) KernelHeatFlux<2>(r, v0, v1);
    else KernelHeatFlux<3>(r, v0, v1);
  } else if (!flux) {
    if (nDim == 2) KernelSymmetry<2>(r, v0, v1);
    else KernelSymmetry<3>(r, v0, v1);
  } else {
    if (nDim == 2) KernelBoundaryFlux<2>(r, static_cast<int>(p.kind), v0, v1, lastSub);
    else KernelBoundaryFlux<3>(r, static_cast<int>(p.kind), v0, v1, lastSub);
  }
  Kokkos::fence();
  P.kernel += timer.seconds();
  timer.reset();
  if (r.writeWallState) {
    MarkDeviceWritten(G_SOLUTION_OLD);
    MarkDeviceWritten(G_TRUNCERR);
  }

  if (p.kind == BCKind::OUTLET) {
    out.lastSubsonic = (lastSub >= long(v0)) ? lastSub - long(v0) : -1;
    outletCalls++;
    if (out.lastSubsonic >= 0) outletSubsonicCalls++;
    if (out.lastSubsonic >= 0) {
      out.lastVOutlet.resize(nPrimVar);
      Kokkos::deep_copy(Kokkos::View<double*, HostSpace, Unmanaged>(out.lastVOutlet.data(), nPrimVar),
                        Kokkos::subview(st.Charac, size_t(lastSub), Kokkos::ALL()));
      P.bytes_d2h += double(nPrimVar) * sizeof(double);
    }
    if (p.downloadCharac && v1 > v0) {
      out.charac.resize((v1 - v0) * nPrimVar);
      Kokkos::deep_copy(Kokkos::View<double**, Kokkos::LayoutRight, HostSpace, Unmanaged>(out.charac.data(), v1 - v0,
                                                                                          nPrimVar),
                        Kokkos::subview(st.Charac, std::make_pair(v0, v1), Kokkos::ALL()));
      P.bytes_d2h += double(out.charac.size()) * sizeof(double);
    }
  }
  P.d2h += timer.seconds();
  P.calls++;
  return out;
}


/*====================== Step 6: device-resident state, implicit / explicit solution update ======================*/

namespace {

using RR = CNEMOKokkosData::ResidualReduction;

KOKKOS_INLINE_FUNCTION void RRInit(RR& v) {
  for (int k = 0; k < su2kokkos::nemo::MAXNVAR; ++k) {
    v.rms[k] = 0.0;
    v.max[k] = 0.0;
    v.idx[k] = 0;
  }
}
/*--- Ordered chunks (Serial: a single chunk) give the CPU result; the index tie-break keeps the first maximum
 *    (CPU: strict ">") for any join order. ---*/
KOKKOS_INLINE_FUNCTION void RRJoin(RR& d, const RR& s, int nVar) {
  for (int k = 0; k < nVar; ++k) {
    d.rms[k] += s.rms[k];
    if (s.max[k] > d.max[k] || (s.max[k] == d.max[k] && s.idx[k] < d.idx[k])) {
      d.max[k] = s.max[k];
      d.idx[k] = s.idx[k];
    }
  }
}
/*--- CSolver::ResidualReductions_PerThread. ---*/
KOKKOS_INLINE_FUNCTION void RRAdd(RR& v, unsigned long iPoint, int iVar, double res) {
  res = Kokkos::fabs(res);
  v.rms[iVar] += res * res;
  if (res > v.max[iVar]) {
    v.max[iVar] = res;
    v.idx[iVar] = iPoint;
  }
}

/*--- PrepareImplicitIteration_impl (DummyPrec: AddVal2Diag / SetVal2Diag) fused with the RHS loop (per point
 *    independent, same statements per entry). ---*/
struct PrepareFunctor {
  using value_type = RR;
  View1D<double> vol, pvol, dt, jac;
  View1D<unsigned long> dia;
  View2D<double> res, sol, trunc;
  int nVar = 0;
  bool implicit = false;

  KOKKOS_INLINE_FUNCTION void init(value_type& v) const { RRInit(v); }
  KOKKOS_INLINE_FUNCTION void join(value_type& d, const value_type& s) const { RRJoin(d, s, nVar); }
  KOKKOS_INLINE_FUNCTION void operator()(const unsigned long iPoint, value_type& v) const {
    const unsigned long nv = nVar;
    const double dti = dt(iPoint);
    if (implicit) {
      const unsigned long base = dia(iPoint) * nv * nv;
      if (dti != 0.0) {
        const double Vol = vol(iPoint) + pvol(iPoint);
        const double Delta = Vol / dti;
        for (unsigned long iVar = 0; iVar < nv; iVar++) jac(base + iVar * (nv + 1)) += Delta;
      } else {
        for (unsigned long k = 0; k < nv * nv; k++) jac(base + k) = 0.0;
        for (unsigned long iVar = 0; iVar < nv; iVar++) jac(base + iVar * (nv + 1)) = 1.0;
      }
    }
    if (dti == 0.0) {
      for (int iVar = 0; iVar < nVar; iVar++) {
        res(iPoint, iVar) = 0.0;
        trunc(iPoint, iVar) = 0.0;
      }
    }
    for (int iVar = 0; iVar < nVar; iVar++) {
      res(iPoint, iVar) = -(res(iPoint, iVar) + trunc(iPoint, iVar));
      sol(iPoint, iVar) = 0.0;
      RRAdd(v, iPoint, iVar, res(iPoint, iVar));
    }
  }
};

/*--- Explicit_Iteration_impl<EULER_EXPLICIT> with the Identity residual "preconditioner". ---*/
struct ExplicitFunctor {
  using value_type = RR;
  View1D<double> vol, pvol, dt;
  View2D<double> res, trunc, U, Uold;
  int nVar = 0;

  KOKKOS_INLINE_FUNCTION void init(value_type& v) const { RRInit(v); }
  KOKKOS_INLINE_FUNCTION void join(value_type& d, const value_type& s) const { RRJoin(d, s, nVar); }
  KOKKOS_INLINE_FUNCTION void operator()(const unsigned long iPoint, value_type& v) const {
    const double Vol = vol(iPoint) + pvol(iPoint);
    const double Delta = dt(iPoint) / Vol;
    for (int iVar = 0; iVar < nVar; iVar++) {
      const double Res = res(iPoint, iVar) + trunc(iPoint, iVar);
      /*--- CVariable::AddSolution: Solution = Solution_Old + du (Solution_Old differs from Solution where the NS
       *    wall BCs zeroed its velocity, SetVelocity_Old). ---*/
      U(iPoint, iVar) = Uold(iPoint, iVar) + -Res * Delta;
      RRAdd(v, iPoint, iVar, Res);
    }
  }
};

/*--- Batch A: Explicit_Iteration_impl<RUNGE_KUTTA_EXPLICIT> / <CLASSICAL_RK4_EXPLICIT>, Identity preconditioner. ---*/
struct ExplicitRKFunctor {
  using value_type = RR;
  View1D<double> vol, pvol, dt;
  View2D<double> res, trunc, U, Uold, Unew;
  int nVar = 0, step = 0;
  bool rk4 = false;
  double alpha = 0.0, timeCoeff = 0.0, funcCoeff = 0.0;

  KOKKOS_INLINE_FUNCTION void init(value_type& v) const { RRInit(v); }
  KOKKOS_INLINE_FUNCTION void join(value_type& d, const value_type& s) const { RRJoin(d, s, nVar); }
  KOKKOS_INLINE_FUNCTION void operator()(const unsigned long iPoint, value_type& v) const {
    const double Vol = vol(iPoint) + pvol(iPoint);
    const double Delta = dt(iPoint) / Vol;
    for (int iVar = 0; iVar < nVar; iVar++) {
      const double Res = res(iPoint, iVar) + trunc(iPoint, iVar);
      if (!rk4) {
        /*--- AddSolution(-Res*Delta*RK_AlphaCoeff): Solution = Solution_Old + du. ---*/
        U(iPoint, iVar) = Uold(iPoint, iVar) + -Res * Delta * alpha;
      } else {
        const double tmp_time = -1.0 * timeCoeff * Delta;
        const double tmp_func = -1.0 * funcCoeff * Delta;
        if (step < 3) {
          U(iPoint, iVar) = Uold(iPoint, iVar) + tmp_time * Res;     /*--- AddSolution ---*/
          Unew(iPoint, iVar) = Unew(iPoint, iVar) + tmp_func * Res;  /*--- AddSolution_New ---*/
        } else {
          U(iPoint, iVar) = Unew(iPoint, iVar) + tmp_func * Res;
        }
      }
      RRAdd(v, iPoint, iVar, Res);
    }
  }
};

/*--- Batch A: point loop of CSolver::AdaptCFLNumber (no turbulence / species solver), std::min / std::max semantics. ---*/
struct CFLStats {
  double mn, mx, sum;
};
struct AdaptCFLFunctor {
  using value_type = CFLStats;
  View1D<double> cfl, ur;
  CNEMOKokkosData::CFLAdaptParams p;

  KOKKOS_INLINE_FUNCTION void init(value_type& v) const {
    v.mn = 1e30;
    v.mx = 0.0;
    v.sum = 0.0;
  }
  KOKKOS_INLINE_FUNCTION void join(value_type& d, const value_type& s) const {
    d.mn = (d.mn < s.mn) ? d.mn : s.mn;
    d.mx = (s.mx < d.mx) ? d.mx : s.mx;
    d.sum += s.sum;
  }
  KOKKOS_INLINE_FUNCTION void operator()(const unsigned long iPoint, value_type& v) const {
    double CFL = cfl(iPoint);
    const double underRelaxationFlow = ur(iPoint);
    const double underRelaxationTurb = 1.0;
    const double underRelaxation = (underRelaxationTurb < underRelaxationFlow) ? underRelaxationTurb : underRelaxationFlow;

    double CFLFactor = 1.0;
    if (underRelaxation < 0.1 || p.reduceCFL) {
      CFLFactor = p.factorDecrease;
    } else if ((underRelaxation >= 0.1 && underRelaxation < 1.0) || !p.canIncrease) {
      CFLFactor = 1.0;
    } else {
      CFLFactor = p.factorIncrease;
    }

    if (CFL * CFLFactor <= p.cflMin) {
      CFL = p.cflMin;
      CFLFactor = p.mgFactor;
    } else if (CFL * CFLFactor >= p.cflMax) {
      CFL = p.cflMax;
      CFLFactor = p.mgFactor;
    }

    if (p.resetCFL) {
      CFL = p.cflMin;
      CFLFactor = p.mgFactor;
    }

    CFL *= CFLFactor;
    cfl(iPoint) = CFL;

    if (p.stats) {
      v.mn = (v.mn < CFL) ? v.mn : CFL; /*--- min(CFL, myCFLMin) ---*/
      v.mx = (CFL < v.mx) ? v.mx : CFL; /*--- max(CFL, myCFLMax) ---*/
      v.sum += CFL;
    }
  }
};

template <class DView>
double GroupBytes(const DView& d) {
  return double(d.span()) * sizeof(typename DView::value_type);
}

}  // namespace

void CNEMOKokkosData::EnableResident(double* linSysRes, double* linSysSol, double* jacValues, double* hostRes,
                                     double* hostSol) {
  resident = true;
  hostLinSysRes = hostRes;
  hostLinSysSol = hostSol;
  /*--- The accumulator / linear-system views alias the CSysVector / CSysMatrix device storage (same row-major /
   *    CSysMatrix block layout), so the device solver reads them without any copy. ---*/
  st.Res = View2D<double>(linSysRes, nPoint, nVar); /*--- unmanaged (pointer constructor) ---*/
  st.Sol = View2D<double>(linSysSol, nPoint, nVar);
  if (jacValues != nullptr) st.JacVal = View1D<double>(jacValues, geo.nnz * size_t(nVar) * nVar);
  st.TruncErr = View2D<double>("NEMO::Res_TruncError", nPoint, nVar);
  st.UnderRelax = View1D<double>("NEMO::UnderRelaxation", nPoint);
  for (auto& r : residency) r = Residency::HOST;
  const char* env = std::getenv("SU2_KOKKOS_NEMO_POISON");
  poison = (env != nullptr && std::atoi(env) != 0);

  /*--- Unique marker-vertex points (force / traction computations read their Primitive). ---*/
  auto hm = Kokkos::create_mirror_view_and_copy(HostSpace(), geo.vtx_node);
  std::vector<char> seen(nPoint, 0);
  for (size_t k = 0; k < hm.size(); ++k)
    if (!seen[hm(k)]) {
      seen[hm(k)] = 1;
      h_bndPoints.push_back(hm(k));
    }
  std::sort(h_bndPoints.begin(), h_bndPoints.end());
  st.bndPoints = View1D<unsigned long>("NEMO::bndPoints", h_bndPoints.size());
  st.bndStage = View2D<double>("NEMO::bndStage", h_bndPoints.size(), nPrimVar);
  Kokkos::deep_copy(st.bndPoints, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(h_bndPoints.data(),
                                                                                           h_bndPoints.size()));
}

void CNEMOKokkosData::ActivateResident() {
  if (!resident || residentActive) return;
  residentActive = true;
  for (auto& r : residency) r = Residency::HOST;
  /*--- Device copies of the accumulator inputs that SetPrimVar's previous uploads may have left: none are trusted. ---*/
  primDeviceValid = secDeviceValid = consDeviceValid = false;
  gradPrimValid = gradRecValid = limValid = sensorValid = false;
  transportDeviceValid = bndViscHost = false;
}

void CNEMOKokkosData::DownloadGroup(Group g) {
  auto& P = syncPerf[g];
  Kokkos::Timer timer;
  double b = 0.0;
  const auto nd = geo.nPointDomain;
  auto as2d = [&](const View1D<double>& v) {
    return Kokkos::View<double**, Kokkos::LayoutRight, mem_space, Unmanaged>(v.data(), v.size(), 1);
  };
  auto rows = [&](double* host, const auto& d, size_t n) {
    if (d.size() == 0) return;
    const size_t cnt = n * (d.size() / d.extent(0));
    Kokkos::deep_copy(Kokkos::View<double*, HostSpace, Unmanaged>(host, cnt),
                      Kokkos::View<const double*, mem_space, Unmanaged>(d.data(), cnt));
    b += double(cnt) * sizeof(double);
  };
  switch (g) {
    case G_SOLUTION: rows(nodes->Solution.data(), st.Solution, nPoint); break;
    case G_SOLUTION_OLD: rows(nodes->Solution_Old.data(), st.Solution_Old, nPoint); break;
    case G_PRIMITIVE:
      rows(nodes->Primitive.data(), st.Primitive, nPoint);
      if (implicit) {
        rows(nodes->dPdU.data(), st.dPdU, nPoint);
        rows(nodes->dTdU.data(), st.dTdU, nPoint);
        rows(nodes->dTvedU.data(), st.dTvedU, nPoint);
      }
      rows(nodes->eves.data(), st.eves, nPoint);
      rows(nodes->Cvves.data(), st.Cvves, nPoint);
      rows(nodes->Gamma.data(), Kokkos::View<double**, Kokkos::LayoutRight, mem_space, Unmanaged>(st.Gamma.data(),
                                                                                                    nPoint, 1), nPoint);
      if (viscousVars) {
        auto* ns = static_cast<CNEMONSVariable*>(nodes);
        rows(ns->DiffusionCoeff.data(), st.DiffCoeff, nPoint);
        rows(ns->Enthalpys.data(), st.Enthalpys, nPoint);
        rows(ns->LaminarViscosity.data(), as2d(st.LamVisc), nPoint);
        rows(ns->ThermalCond.data(), as2d(st.ThermalCond), nPoint);
        rows(ns->ThermalCond_ve.data(), as2d(st.ThermalCondVe), nPoint);
      }
      /*--- Host remainder of SetPrimVar (SetPrimitive_Variables): Velocity2 from the downloaded Primitive. ---*/
      for (unsigned long i = 0; i < nPoint; ++i) nodes->SetVelocity2(i);
      bndPrimHost = true;
      break;
    case G_GRADIENT:
      if (st.GradPrim.size()) rows(nodes->GetGradient_Primitive().data(), st.GradPrim, nd);
      if (separateRecGradient && st.GradRec.size()) rows(nodes->GetGradient_Reconstruction().data(), st.GradRec, nd);
      break;
    case G_LIMITER:
      rows(nodes->GetLimiter_Primitive().data(), st.Limiter, nd);
      rows(nodes->GetSolution_Max().data(), st.PrimMax, nd);
      rows(nodes->GetSolution_Min().data(), st.PrimMin, nd);
      break;
    case G_TIMESTEP: {
      auto as2 = [&](const View1D<double>& v) {
        return Kokkos::View<double**, Kokkos::LayoutRight, mem_space, Unmanaged>(v.data(), v.size(), 1);
      };
      if (st.DeltaTime.size()) rows(nodes->Delta_Time.data(), as2(st.DeltaTime), nd);
      if (st.MaxLambdaInv.size()) rows(nodes->Max_Lambda_Inv.data(), as2(st.MaxLambdaInv), nd);
      if (st.MaxLambdaVisc.size()) rows(nodes->Max_Lambda_Visc.data(), as2(st.MaxLambdaVisc), nd);
      break;
    }
    case G_LOCALCFL:
      if (st.LocalCFL.size())
        rows(nodes->LocalCFL.data(),
             Kokkos::View<double**, Kokkos::LayoutRight, mem_space, Unmanaged>(st.LocalCFL.data(), nPoint, 1), nPoint);
      break;
    case G_NONPHYS:
      if (st.NonPhys.size()) {
        auto hnp = Kokkos::create_mirror_view_and_copy(HostSpace(), st.NonPhys);
        auto hnc = Kokkos::create_mirror_view_and_copy(HostSpace(), st.NonPhysCount);
        for (unsigned long i = 0; i < nPoint; ++i) {
          nodes->Non_Physical(i) = (hnp(i) != 0);
          nodes->Non_Physical_Counter(i) = hnc(i);
        }
        b += double(nPoint) * 3;
      }
      break;
    case G_UNDERRELAX:
      rows(nodes->UnderRelaxation.data(),
           Kokkos::View<double**, Kokkos::LayoutRight, mem_space, Unmanaged>(st.UnderRelax.data(), nPoint, 1), nPoint);
      break;
    case G_TRUNCERR: rows(nodes->Res_TruncError.data(), st.TruncErr, nPoint); break;
    case G_LINSYS:
      rows(hostLinSysRes, st.Res, nPoint);
      rows(hostLinSysSol, st.Sol, nPoint);
      break;
    case G_SENSOR:
      if (st.Sensor.size()) rows(nodes->Sensor.data(), as2d(st.Sensor), nPoint);
      break;
    case G_AUXVAR:
      if (st.AuxVar.size()) {
        rows(nodes->AuxVar.data(), st.AuxVar, nPoint);
        rows(nodes->Grad_AuxVar.data(), st.AuxGrad, nPoint);
      }
      break;
    case G_TIMEN:
      if (st.SolTimeN.size()) {
        rows(nodes->Solution_time_n.data(), st.SolTimeN, nPoint);
        rows(nodes->Solution_time_n1.data(), st.SolTimeN1, nPoint);
      }
      break;
    default: break;
  }
  P.bytes_d2h += b;
  P.d2h += timer.seconds();
  P.calls++;
}

void CNEMOKokkosData::UploadGroup(Group g) {
  auto& P = syncPerf[g];
  Kokkos::Timer timer;
  double b = 0.0;
  auto rows = [&](const auto& d, const double* host, size_t n) {
    if (d.size() == 0) return;
    const size_t cnt = n * (d.size() / d.extent(0));
    Kokkos::deep_copy(Kokkos::View<double*, mem_space, Unmanaged>(d.data(), cnt),
                      Kokkos::View<const double*, HostSpace, Unmanaged>(host, cnt));
    b += double(cnt) * sizeof(double);
  };
  auto as2 = [&](const View1D<double>& v) {
    return Kokkos::View<double**, Kokkos::LayoutRight, mem_space, Unmanaged>(v.data(), v.size(), 1);
  };
  switch (g) {
    case G_SOLUTION: rows(st.Solution, nodes->Solution.data(), nPoint); break;
    case G_SOLUTION_OLD: rows(st.Solution_Old, nodes->Solution_Old.data(), nPoint); break;
    case G_PRIMITIVE:
      rows(st.Primitive, nodes->Primitive.data(), nPoint);
      if (implicit) {
        rows(st.dPdU, nodes->dPdU.data(), nPoint);
        rows(st.dTdU, nodes->dTdU.data(), nPoint);
        rows(st.dTvedU, nodes->dTvedU.data(), nPoint);
      }
      rows(st.eves, nodes->eves.data(), nPoint);
      rows(st.Cvves, nodes->Cvves.data(), nPoint);
      rows(as2(st.Gamma), nodes->Gamma.data(), nPoint);
      if (viscousVars) {
        auto* ns = static_cast<CNEMONSVariable*>(nodes);
        rows(st.DiffCoeff, ns->DiffusionCoeff.data(), nPoint);
        rows(st.Enthalpys, ns->Enthalpys.data(), nPoint);
        rows(as2(st.LamVisc), ns->LaminarViscosity.data(), nPoint);
        rows(as2(st.ThermalCond), ns->ThermalCond.data(), nPoint);
        rows(as2(st.ThermalCondVe), ns->ThermalCond_ve.data(), nPoint);
        transportDeviceValid = true;
      }
      break;
    case G_GRADIENT:
      AllocateGradients();
      rows(st.GradPrim, nodes->GetGradient_Primitive().data(), nPoint);
      if (separateRecGradient) rows(st.GradRec, nodes->GetGradient_Reconstruction().data(), nPoint);
      break;
    case G_LIMITER:
      rows(st.Limiter, nodes->GetLimiter_Primitive().data(), nPoint);
      rows(st.PrimMax, nodes->GetSolution_Max().data(), nPoint);
      rows(st.PrimMin, nodes->GetSolution_Min().data(), nPoint);
      break;
    case G_TIMESTEP:
      if (st.DeltaTime.size()) rows(as2(st.DeltaTime), nodes->Delta_Time.data(), nPoint);
      if (st.MaxLambdaInv.size()) rows(as2(st.MaxLambdaInv), nodes->Max_Lambda_Inv.data(), nPoint);
      if (st.MaxLambdaVisc.size()) rows(as2(st.MaxLambdaVisc), nodes->Max_Lambda_Visc.data(), nPoint);
      break;
    case G_LOCALCFL:
      if (st.LocalCFL.size() == 0) st.LocalCFL = View1D<double>("NEMO::LocalCFL", nPoint);
      rows(as2(st.LocalCFL), nodes->LocalCFL.data(), nPoint);
      break;
    case G_NONPHYS:
      if (st.NonPhys.size()) {
        auto hnp = Kokkos::create_mirror_view(st.NonPhys);
        auto hnc = Kokkos::create_mirror_view(st.NonPhysCount);
        for (unsigned long i = 0; i < nPoint; ++i) {
          hnp(i) = nodes->Non_Physical(i) ? 1 : 0;
          hnc(i) = nodes->Non_Physical_Counter(i);
        }
        Kokkos::deep_copy(st.NonPhys, hnp);
        Kokkos::deep_copy(st.NonPhysCount, hnc);
        b += double(nPoint) * 3;
      }
      break;
    case G_UNDERRELAX: rows(as2(st.UnderRelax), nodes->UnderRelaxation.data(), nPoint); break;
    case G_TRUNCERR: rows(st.TruncErr, nodes->Res_TruncError.data(), nPoint); break;
    case G_LINSYS:
      rows(st.Res, hostLinSysRes, nPoint);
      rows(st.Sol, hostLinSysSol, nPoint);
      break;
    case G_SENSOR:
      if (nodes->Sensor.size() < nPoint) break; /*--- not allocated (not AUSM+M) ---*/
      if (st.Sensor.size() == 0) st.Sensor = View1D<double>("NEMO::Sensor", nPoint);
      rows(as2(st.Sensor), nodes->Sensor.data(), nPoint);
      sensorValid = true;
      break;
    case G_AUXVAR:
      if (nodes->GetnAuxVar() != 3) break; /*--- not axisymmetric ---*/
      AllocateAuxVar();
      rows(st.AuxVar, nodes->AuxVar.data(), nPoint);
      rows(st.AuxGrad, nodes->Grad_AuxVar.data(), nPoint);
      auxDeviceValid = true;
      break;
    case G_TIMEN:
      if (size_t(nodes->Solution_time_n.size()) < size_t(nPoint) * nVar) break; /*--- not dual time ---*/
      if (st.SolTimeN.size() == 0) {
        st.SolTimeN = View2D<double>("NEMO::Solution_time_n", nPoint, nVar);
        st.SolTimeN1 = View2D<double>("NEMO::Solution_time_n1", nPoint, nVar);
      }
      rows(st.SolTimeN, nodes->Solution_time_n.data(), nPoint);
      rows(st.SolTimeN1, nodes->Solution_time_n1.data(), nPoint);
      break;
    default: break;
  }
  P.bytes_h2d += b;
  P.h2d += timer.seconds();
  P.calls++;
}

void CNEMOKokkosData::PoisonGroup(Group g) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  auto fill = [&](double* p, size_t n) { std::fill(p, p + n, nan); };
  const auto nd = geo.nPointDomain;
  switch (g) {
    case G_SOLUTION: fill(nodes->Solution.data(), nodes->Solution.size()); break;
    case G_SOLUTION_OLD: fill(nodes->Solution_Old.data(), nodes->Solution_Old.size()); break;
    case G_PRIMITIVE:
      fill(nodes->Primitive.data(), nodes->Primitive.size());
      if (implicit) {
        fill(nodes->dPdU.data(), nodes->dPdU.size());
        fill(nodes->dTdU.data(), nodes->dTdU.size());
        fill(nodes->dTvedU.data(), nodes->dTvedU.size());
      }
      fill(nodes->eves.data(), nodes->eves.size());
      fill(nodes->Cvves.data(), nodes->Cvves.size());
      fill(nodes->Gamma.data(), nodes->Gamma.size());
      fill(nodes->Velocity2.data(), nodes->Velocity2.size());
      if (viscousVars) {
        auto* ns = static_cast<CNEMONSVariable*>(nodes);
        fill(ns->DiffusionCoeff.data(), ns->DiffusionCoeff.size());
        fill(ns->Enthalpys.data(), ns->Enthalpys.size());
        fill(ns->LaminarViscosity.data(), ns->LaminarViscosity.size());
        fill(ns->ThermalCond.data(), ns->ThermalCond.size());
        fill(ns->ThermalCond_ve.data(), ns->ThermalCond_ve.size());
      }
      break;
    case G_GRADIENT:
      fill(nodes->GetGradient_Primitive().data(), nd * size_t(nPrimVarGrad) * nDim);
      if (separateRecGradient) fill(nodes->GetGradient_Reconstruction().data(), nd * size_t(nPrimVarGrad) * nDim);
      break;
    case G_LIMITER:
      fill(nodes->GetLimiter_Primitive().data(), nd * size_t(nPrimVarGrad));
      fill(nodes->GetSolution_Max().data(), nd * size_t(nPrimVarGrad));
      fill(nodes->GetSolution_Min().data(), nd * size_t(nPrimVarGrad));
      break;
    case G_TIMESTEP:
      fill(nodes->Delta_Time.data(), nd);
      fill(nodes->Max_Lambda_Inv.data(), nd);
      if (st.MaxLambdaVisc.size()) fill(nodes->Max_Lambda_Visc.data(), nd);
      break;
    case G_LOCALCFL: fill(nodes->LocalCFL.data(), nPoint); break;
    case G_NONPHYS:
      for (unsigned long i = 0; i < nPoint; ++i) {
        nodes->Non_Physical(i) = true;
        nodes->Non_Physical_Counter(i) = 0xFFFF;
      }
      break;
    case G_UNDERRELAX: fill(nodes->UnderRelaxation.data(), nPoint); break;
    case G_TRUNCERR: fill(nodes->Res_TruncError.data(), nodes->Res_TruncError.size()); break;
    case G_LINSYS:
      fill(hostLinSysRes, size_t(nPoint) * nVar);
      fill(hostLinSysSol, size_t(nPoint) * nVar);
      break;
    case G_SENSOR: fill(nodes->Sensor.data(), nodes->Sensor.size()); break;
    case G_AUXVAR:
      fill(nodes->AuxVar.data(), nodes->AuxVar.size());
      fill(nodes->Grad_AuxVar.data(), nodes->Grad_AuxVar.size());
      break;
    case G_TIMEN:
      fill(nodes->Solution_time_n.data(), nodes->Solution_time_n.size());
      fill(nodes->Solution_time_n1.data(), nodes->Solution_time_n1.size());
      break;
    default: break;
  }
}

void CNEMOKokkosData::EnsureHost(Group g) {
  if (!ResidentActive() || residency[g] != Residency::DEVICE) return;
  DownloadGroup(g);
  residency[g] = Residency::BOTH;
}

void CNEMOKokkosData::EnsureDevice(Group g) {
  if (!ResidentActive() || residency[g] != Residency::HOST) return;
  UploadGroup(g);
  residency[g] = Residency::BOTH;
}

void CNEMOKokkosData::MarkDeviceWritten(Group g) {
  if (!ResidentActive()) return;
  residency[g] = Residency::DEVICE;
  if (g == G_PRIMITIVE) bndPrimHost = false;
  if (g == G_PRIMITIVE || g == G_GRADIENT) bndViscHost = false;
  if (poison) PoisonGroup(g);
}

void CNEMOKokkosData::SetViscousWallMarkers(const std::vector<int>& markers) {
  if (!viscousVars) return;
  if (h_markerPtr.empty()) {
    auto hm = Kokkos::create_mirror_view_and_copy(HostSpace(), geo.marker_ptr);
    h_markerPtr.assign(hm.data(), hm.data() + geo.nMarker + 1);
  }
  auto hn = Kokkos::create_mirror_view_and_copy(HostSpace(), geo.vtx_node);
  std::vector<char> seen(nPoint, 0);
  h_viscWallPoints.clear();
  for (int m : markers)
    for (auto k = h_markerPtr[m]; k < h_markerPtr[m + 1]; ++k)
      if (!seen[hn(k)]) {
        seen[hn(k)] = 1;
        h_viscWallPoints.push_back(hn(k));
      }
  std::sort(h_viscWallPoints.begin(), h_viscWallPoints.end());
  const size_t rowLen = size_t(nPrimVarGrad) * nDim + 3 + 2 * size_t(nSpecies);
  st.viscWallPoints = View1D<unsigned long>("NEMO::viscWallPoints", h_viscWallPoints.size());
  st.viscWallStage = View2D<double>("NEMO::viscWallStage", h_viscWallPoints.size(), rowLen);
  Kokkos::deep_copy(st.viscWallPoints, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(
                                           h_viscWallPoints.data(), h_viscWallPoints.size()));
}

void CNEMOKokkosData::EnsureHostBoundaryPrimitive() {
  if (!ResidentActive()) return;
  /*--- NS (Friction_Forces): gradient + transport rows of the viscous-wall points. ---*/
  const bool viscStale = (residency[G_PRIMITIVE] == Residency::DEVICE || residency[G_GRADIENT] == Residency::DEVICE);
  if (viscousVars && !bndViscHost && viscStale && !h_viscWallPoints.empty()) {
    auto& P = syncPerf[N_GROUPS];
    Kokkos::Timer timer;
    const auto pts = st.viscWallPoints;
    const auto stage = st.viscWallStage;
    const auto G = st.GradPrim;
    const auto mu = st.LamVisc, k = st.ThermalCond, kve = st.ThermalCondVe;
    const auto Ds = st.DiffCoeff, hs = st.Enthalpys;
    const int ng = nPrimVarGrad * nDim, ns = nSpecies;
    const bool haveGrad = (G.size() != 0);
    Kokkos::parallel_for(
        "NEMO::GatherViscousWall", Kokkos::RangePolicy<exec_space>(0, h_viscWallPoints.size()),
        KOKKOS_LAMBDA(const size_t q) {
          const auto i = pts(q);
          if (haveGrad)
            for (int v = 0; v < ng; ++v) stage(q, v) = G.data()[i * ng + v];
          stage(q, ng) = mu(i);
          stage(q, ng + 1) = k(i);
          stage(q, ng + 2) = kve(i);
          for (int s = 0; s < ns; ++s) {
            stage(q, ng + 3 + s) = Ds(i, s);
            stage(q, ng + 3 + ns + s) = hs(i, s);
          }
        });
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), st.viscWallStage);
    auto* nsv = static_cast<CNEMONSVariable*>(nodes);
    double* hg = nodes->GetGradient_Primitive().data();
    for (size_t q = 0; q < h_viscWallPoints.size(); ++q) {
      const auto i = h_viscWallPoints[q];
      if (haveGrad && residency[G_GRADIENT] == Residency::DEVICE) std::memcpy(hg + i * ng, &h(q, 0), ng * sizeof(double));
      if (residency[G_PRIMITIVE] == Residency::DEVICE) {
        nsv->LaminarViscosity(i) = h(q, ng);
        nsv->ThermalCond(i) = h(q, ng + 1);
        nsv->ThermalCond_ve(i) = h(q, ng + 2);
        for (int s = 0; s < ns; ++s) {
          nsv->DiffusionCoeff(i, s) = h(q, ng + 3 + s);
          nsv->Enthalpys(i, s) = h(q, ng + 3 + ns + s);
        }
      }
    }
    P.bytes_d2h += double(h.size()) * sizeof(double);
    bndViscHost = true;
    P.d2h += timer.seconds();
  }
  if (residency[G_PRIMITIVE] != Residency::DEVICE || bndPrimHost) return;
  auto& P = syncPerf[N_GROUPS];
  Kokkos::Timer timer;
  const auto nb = h_bndPoints.size();
  if (nb > 0) {
    const auto pts = st.bndPoints;
    const auto stage = st.bndStage;
    const auto V = st.Primitive;
    const int npv = nPrimVar;
    Kokkos::parallel_for(
        "NEMO::GatherBoundaryPrimitive", Kokkos::RangePolicy<exec_space>(0, nb), KOKKOS_LAMBDA(const size_t k) {
          for (int v = 0; v < npv; ++v) stage(k, v) = V(pts(k), v);
        });
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), st.bndStage);
    for (size_t k = 0; k < nb; ++k)
      std::memcpy(&nodes->Primitive(h_bndPoints[k], 0), &h(k, 0), size_t(nPrimVar) * sizeof(double));
    P.bytes_d2h += double(nb) * nPrimVar * sizeof(double);
  }
  bndPrimHost = true;
  P.d2h += timer.seconds();
  P.calls++;
}

void CNEMOKokkosData::EnsureHostAll() {
  if (!ResidentActive()) return;
  outputThisIteration = true;
  for (int g = 0; g < N_GROUPS; ++g) EnsureHost(static_cast<Group>(g));
}

/*====================== Step 10: MPI halo exchange of device-resident arrays ======================*/

double* CNEMOKokkosData::HaloDevicePtr(HaloQuantity q, size_t& rowLen, Group& g) {
  switch (q) {
    case HaloQuantity::SOLUTION:
      rowLen = nVar;
      g = G_SOLUTION;
      return st.Solution.data();
    case HaloQuantity::GRADIENT_PRIMITIVE:
      rowLen = size_t(nPrimVarGrad) * nDim;
      g = G_GRADIENT;
      return st.GradPrim.data();
    case HaloQuantity::GRADIENT_RECONSTRUCTION:
      rowLen = size_t(nPrimVarGrad) * nDim;
      g = G_GRADIENT;
      return st.GradRec.data(); /*--- == GradPrim if Gradient_Reconstruction is not a separate container ---*/
    case HaloQuantity::LIMITER:
      rowLen = nPrimVarGrad;
      g = G_LIMITER;
      return st.Limiter.data();
    case HaloQuantity::SENSOR:
      rowLen = 1;
      g = G_SENSOR;
      return st.Sensor.data();
    case HaloQuantity::AUXVAR_GRADIENT:
      rowLen = size_t(3) * nDim;
      g = G_AUXVAR;
      return st.AuxGrad.data();
    default: break;
  }
  SU2_MPI::Error("CNEMOKokkosData: unknown halo quantity.", CURRENT_FUNCTION);
  return nullptr;
}

double* CNEMOKokkosData::HaloHostPtr(HaloQuantity q) {
  switch (q) {
    case HaloQuantity::SOLUTION: return nodes->Solution.data();
    case HaloQuantity::GRADIENT_PRIMITIVE: return nodes->GetGradient_Primitive().data();
    case HaloQuantity::GRADIENT_RECONSTRUCTION: return nodes->GetGradient_Reconstruction().data();
    case HaloQuantity::LIMITER: return nodes->GetLimiter_Primitive().data();
    case HaloQuantity::SENSOR: return nodes->Sensor.data();
    case HaloQuantity::AUXVAR_GRADIENT: return nodes->Grad_AuxVar.data();
    default: break;
  }
  return nullptr;
}

void CNEMOKokkosData::HaloPlan(CGeometry* geometry) {
  /*--- Geometry metadata, uploaded once per (solver, geometry). A different geometry object (e.g. after a mesh
   *    re-partition) or different halo counts rebuild the plan. ---*/
  const auto nSend = geometry->nP2PSend > 0 ? static_cast<unsigned long>(geometry->nPoint_P2PSend[geometry->nP2PSend]) : 0ul;
  const auto nRecv = geometry->nP2PRecv > 0 ? static_cast<unsigned long>(geometry->nPoint_P2PRecv[geometry->nP2PRecv]) : 0ul;
  if (haloGeometry == geometry && haloSend == nSend && haloRecv == nRecv) return;
  haloGeometry = geometry;
  haloSend = nSend;
  haloRecv = nRecv;
  haloSendIdx = View1D<unsigned long>("NEMO::haloSendIdx", nSend);
  haloRecvIdx = View1D<unsigned long>("NEMO::haloRecvIdx", nRecv);
  if (nSend) Kokkos::deep_copy(haloSendIdx, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(geometry->Local_Point_P2PSend, nSend));
  if (nRecv) Kokkos::deep_copy(haloRecvIdx, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(geometry->Local_Point_P2PRecv, nRecv));
  /*--- Buffers sized for the widest row (gradient). ---*/
  const size_t maxRow = std::max<size_t>(nVar, size_t(nPrimVarGrad) * nDim);
  haloSendBuf = View1D<double>("NEMO::haloSendBuf", nSend * maxRow);
  haloRecvBuf = View1D<double>("NEMO::haloRecvBuf", nRecv * maxRow);
  haloHostSend.assign(nSend * maxRow, 0.0);
  haloHostRecv.assign(nRecv * maxRow, 0.0);
}

void CNEMOKokkosData::HaloExchange(CGeometry* geometry, HaloQuantity q, bool gpuAware) {
  if (!ResidentActive()) return;
#ifdef HAVE_MPI
  Kokkos::Timer timer;
  HaloPlan(geometry);
  if (haloSend == 0 && haloRecv == 0) return;
  size_t rowLen = 0;
  Group g = G_SOLUTION;
  double* dev = HaloDevicePtr(q, rowLen, g);
  if (dev == nullptr) SU2_MPI::Error("CNEMOKokkosData: halo exchange of an unallocated device array.", CURRENT_FUNCTION);
  /*--- The device copy must be current (it is, after the stage that produced it). ---*/
  EnsureDevice(g);
  auto& P = syncPerf[SYNC_HALO];
  auto& H = haloPerf[static_cast<int>(q)];

  const int nMsgR = geometry->nP2PRecv, nMsgS = geometry->nP2PSend;
  std::vector<SU2_MPI::Request> reqR(nMsgR), reqS(nMsgS);
  double* recvBase = gpuAware ? haloRecvBuf.data() : haloHostRecv.data();
  double* sendBase = gpuAware ? haloSendBuf.data() : haloHostSend.data();

  /*--- Receives first (CGeometry::PostP2PRecvs: source = Neighbors_P2PRecv, tag = source + 1). ---*/
  for (int iMsg = 0; iMsg < nMsgR; ++iMsg) {
    const auto offset = static_cast<unsigned long>(geometry->nPoint_P2PRecv[iMsg]) * rowLen;
    const int count = static_cast<int>((geometry->nPoint_P2PRecv[iMsg + 1] - geometry->nPoint_P2PRecv[iMsg]) * rowLen);
    const int source = geometry->Neighbors_P2PRecv[iMsg];
    SU2_MPI::Irecv(recvBase + offset, count, MPI_DOUBLE, source, source + 1, SU2_MPI::GetComm(), &reqR[iMsg]);
  }

  /*--- Device gather of the send rows (point-major, the CPU packing order of the row). ---*/
  if (haloSend > 0) {
    const auto idx = haloSendIdx;
    const auto buf = haloSendBuf;
    const size_t rl = rowLen;
    Kokkos::parallel_for(
        "NEMO::HaloPack", Kokkos::RangePolicy<exec_space>(0, haloSend * rl), KOKKOS_LAMBDA(const unsigned long k) {
          buf(k) = dev[idx(k / rl) * rl + k % rl];
        });
    Kokkos::fence("NEMO::HaloPack");
    if (!gpuAware) {
      Kokkos::deep_copy(Kokkos::View<double*, HostSpace, Unmanaged>(haloHostSend.data(), haloSend * rl),
                        Kokkos::subview(haloSendBuf, std::make_pair(size_t(0), haloSend * rl)));
      P.bytes_d2h += double(haloSend) * rl * sizeof(double);
    }
  }
  /*--- Sends (CGeometry::PostP2PSends: dest = Neighbors_P2PSend, tag = rank + 1). ---*/
  const int rank = SU2_MPI::GetRank();
  for (int iMsg = 0; iMsg < nMsgS; ++iMsg) {
    const auto offset = static_cast<unsigned long>(geometry->nPoint_P2PSend[iMsg]) * rowLen;
    const int count = static_cast<int>((geometry->nPoint_P2PSend[iMsg + 1] - geometry->nPoint_P2PSend[iMsg]) * rowLen);
    SU2_MPI::Isend(sendBase + offset, count, MPI_DOUBLE, geometry->Neighbors_P2PSend[iMsg], rank + 1,
                   SU2_MPI::GetComm(), &reqS[iMsg]);
  }
  if (nMsgR > 0) SU2_MPI::Waitall(nMsgR, reqR.data(), MPI_STATUSES_IGNORE);

  /*--- Device scatter of the receive rows. ---*/
  if (haloRecv > 0) {
    const size_t rl = rowLen;
    if (!gpuAware) {
      Kokkos::deep_copy(Kokkos::subview(haloRecvBuf, std::make_pair(size_t(0), haloRecv * rl)),
                        Kokkos::View<const double*, HostSpace, Unmanaged>(haloHostRecv.data(), haloRecv * rl));
      P.bytes_h2d += double(haloRecv) * rl * sizeof(double);
    }
    const auto idx = haloRecvIdx;
    const auto buf = haloRecvBuf;
    Kokkos::parallel_for(
        "NEMO::HaloUnpack", Kokkos::RangePolicy<exec_space>(0, haloRecv * rl), KOKKOS_LAMBDA(const unsigned long k) {
          dev[idx(k / rl) * rl + k % rl] = buf(k);
        });
    Kokkos::fence("NEMO::HaloUnpack");
  }
  if (nMsgS > 0) SU2_MPI::Waitall(nMsgS, reqS.data(), MPI_STATUSES_IGNORE);

  /*--- The host copy (if it was current) no longer has the received rows. ---*/
  if (residency[g] != Residency::DEVICE) MarkDeviceWritten(g);
  P.calls++;
  H.calls++;
  H.bytesMsg += double(haloSend) * rowLen * sizeof(double);
  H.time += timer.seconds();
#else
  (void)geometry;
  (void)q;
  (void)gpuAware;
#endif
}

void CNEMOKokkosData::DownloadHaloRows(HaloQuantity q, const unsigned long* rows, unsigned long n) {
  if (!ResidentActive() || n == 0) return;
  /*--- Legacy path: send rows into the host container (device gather + one compact DtoH), host comms next. ---*/
  size_t rowLen = 0;
  Group g = G_SOLUTION;
  double* dev = HaloDevicePtr(q, rowLen, g);
  double* host = HaloHostPtr(q);
  EnsureDevice(g);
  auto& P = syncPerf[SYNC_HALO];
  View1D<unsigned long> idx("NEMO::haloIdx", n);
  View2D<double> stage("NEMO::haloStage", n, rowLen);
  Kokkos::deep_copy(idx, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(rows, n));
  Kokkos::parallel_for(
      "NEMO::GatherRows", Kokkos::RangePolicy<exec_space>(0, n), KOKKOS_LAMBDA(const unsigned long k) {
        for (size_t v = 0; v < rowLen; ++v) stage(k, v) = dev[idx(k) * rowLen + v];
      });
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), stage);
  for (unsigned long k = 0; k < n; ++k) std::memcpy(host + rows[k] * rowLen, &h(k, 0), rowLen * sizeof(double));
  P.bytes_d2h += double(n) * rowLen * sizeof(double);
  haloPerf[static_cast<int>(q)].bytesMsg += double(n) * rowLen * sizeof(double);
}

void CNEMOKokkosData::UploadHaloRows(HaloQuantity q, const unsigned long* rows, unsigned long n) {
  if (!ResidentActive() || n == 0) return;
  size_t rowLen = 0;
  Group g = G_SOLUTION;
  double* dev = HaloDevicePtr(q, rowLen, g);
  const double* host = HaloHostPtr(q);
  auto& P = syncPerf[SYNC_HALO];
  View1D<unsigned long> idx("NEMO::haloIdx", n);
  View2D<double> stage("NEMO::haloStage", n, rowLen);
  auto h = Kokkos::create_mirror_view(stage);
  for (unsigned long k = 0; k < n; ++k) std::memcpy(&h(k, 0), host + rows[k] * rowLen, rowLen * sizeof(double));
  Kokkos::deep_copy(stage, h);
  Kokkos::deep_copy(idx, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(rows, n));
  Kokkos::parallel_for(
      "NEMO::ScatterRows", Kokkos::RangePolicy<exec_space>(0, n), KOKKOS_LAMBDA(const unsigned long k) {
        for (size_t v = 0; v < rowLen; ++v) dev[idx(k) * rowLen + v] = stage(k, v);
      });
  Kokkos::fence();
  P.bytes_h2d += double(n) * rowLen * sizeof(double);
  if (residency[g] != Residency::DEVICE) MarkDeviceWritten(g);
  P.calls++;
  haloPerf[static_cast<int>(q)].calls++;
}

void CNEMOKokkosData::PrepareImplicit(bool impl, ResidualReduction& red) {
  Kokkos::Timer timer;
  EnsureDevice(G_TRUNCERR);
  if (impl && st.JacVal.size() == 0) SU2_MPI::Error("CNEMOKokkosData: no device Jacobian.", CURRENT_FUNCTION);
  PrepareFunctor f;
  f.vol = geo.volume;
  f.pvol = geo.periodic_volume;
  f.dt = st.DeltaTime;
  f.jac = st.JacVal;
  f.dia = geo.dia_ptr;
  f.res = st.Res;
  f.sol = st.Sol;
  f.trunc = st.TruncErr;
  f.nVar = nVar;
  f.implicit = impl;
  Kokkos::parallel_reduce("NEMO::PrepareImplicitIteration", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain), f,
                          red);
  /*--- ImplicitEuler_Iteration: halo rows of the RHS and of the initial guess are zero. ---*/
  if (geo.nPoint > geo.nPointDomain) {
    const auto res = st.Res, sol = st.Sol;
    const int nv = nVar;
    Kokkos::parallel_for(
        "NEMO::ZeroHaloRows", Kokkos::RangePolicy<exec_space>(geo.nPointDomain, geo.nPoint),
        KOKKOS_LAMBDA(const unsigned long i) {
          for (int v = 0; v < nv; ++v) res(i, v) = sol(i, v) = 0.0;
        });
  }
  Kokkos::fence();
  /*--- The accumulator has been consumed: st.Res now holds the RHS of the linear system. ---*/
  accOnDevice = false;
  hostAccZero = false;
  MarkDeviceWritten(G_LINSYS);
  MarkDeviceWritten(G_TRUNCERR);
  updatePerf.prepare += timer.seconds();
  updatePerf.nPrepare++;
}

void CNEMOKokkosData::ExplicitEulerUpdate(ResidualReduction& red) {
  Kokkos::Timer timer;
  EnsureDevice(G_TRUNCERR);
  EnsureDevice(G_SOLUTION);
  EnsureDevice(G_SOLUTION_OLD);
  ExplicitFunctor f;
  f.vol = geo.volume;
  f.pvol = geo.periodic_volume;
  f.dt = st.DeltaTime;
  f.res = st.Res;
  f.trunc = st.TruncErr;
  f.U = st.Solution;
  f.Uold = st.Solution_Old;
  f.nVar = nVar;
  Kokkos::parallel_reduce("NEMO::ExplicitEulerIteration", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain), f,
                          red);
  Kokkos::fence();
  accOnDevice = false;
  hostAccZero = false;
  MarkDeviceWritten(G_LINSYS); /*--- the residual stays in st.Res (volume output reads LinSysRes) ---*/
  MarkDeviceWritten(G_SOLUTION);
  updatePerf.explicitUpd += timer.seconds();
  updatePerf.nExplicit++;
}

void CNEMOKokkosData::CompleteImplicit(double allowableRatio) {
  Kokkos::Timer timer;
  EnsureDevice(G_SOLUTION);
  EnsureDevice(G_SOLUTION_OLD);
  const auto U = st.Solution, Uold = st.Solution_Old, sol = st.Sol;
  const auto UR = st.UnderRelax;
  const int nv = nVar, ns = nSpecies;
  const double eps = EPS;
  /*--- CNEMOEulerSolver::ComputeUnderRelaxationFactor then AddSolution(UR * LinSysSol), per point (the factor of a
   *    point reads only that point's Solution and LinSysSol, before its update). ---*/
  Kokkos::parallel_for(
      "NEMO::CompleteImplicitIteration", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint) {
        double localUnderRelaxation = 1.0;
        double num = 0.0;
        double denom = 0.0;
        for (int iVar = 0; iVar < nv; iVar++) {
          if (iVar < ns) {
            num += Kokkos::fabs(sol(iPoint, iVar));
            denom += Kokkos::fabs(U(iPoint, iVar));
            if (iVar == (ns - 1)) {
              const double ratio = (num / (denom + eps));
              if (ratio > allowableRatio) {
                const double r = allowableRatio / ratio;
                localUnderRelaxation = (localUnderRelaxation < r) ? localUnderRelaxation : r; /*--- std::min(r, ur) ---*/
              }
            }
            /*--- The CPU energy branch (iVar == nVar-2) is nested in iVar < nSpecies and never taken. ---*/
            if (iVar == (nv - 2)) {
              const double ratio = Kokkos::fabs(sol(iPoint, iVar)) / (Kokkos::fabs(U(iPoint, iVar)) + eps);
              if (ratio > allowableRatio) {
                const double r = allowableRatio / ratio;
                localUnderRelaxation = (localUnderRelaxation < r) ? localUnderRelaxation : r;
              }
            }
          }
        }
        if (localUnderRelaxation < 1e-10) localUnderRelaxation = 0.0;
        UR(iPoint) = localUnderRelaxation;
        /*--- CVariable::AddSolution: Solution = Solution_Old + UR*du (Solution_Old differs from Solution where the
         *    NS wall BCs zeroed its velocity, SetVelocity_Old). ---*/
        for (int iVar = 0; iVar < nv; iVar++) U(iPoint, iVar) = Uold(iPoint, iVar) + localUnderRelaxation * sol(iPoint, iVar);
      });
  Kokkos::fence();
  MarkDeviceWritten(G_SOLUTION);
  MarkDeviceWritten(G_UNDERRELAX);
  updatePerf.complete += timer.seconds();
  updatePerf.nComplete++;
}

void CNEMOKokkosData::SetOldSolution() {
  Kokkos::Timer timer;
  EnsureDevice(G_SOLUTION);
  Kokkos::deep_copy(st.Solution_Old, st.Solution);
  if (rk4) {
    /*--- CMultiGridIntegration calls Set_NewSolution right after Set_OldSolution (same Solution). ---*/
    if (st.SolutionNew.size() == 0) st.SolutionNew = View2D<double>("NEMO::Solution_New", nPoint, nVar);
    Kokkos::deep_copy(st.SolutionNew, st.Solution);
  }
  Kokkos::fence();
  MarkDeviceWritten(G_SOLUTION_OLD);
  updatePerf.oldSol += timer.seconds();
  updatePerf.nOldSol++;
}

double CNEMOKokkosData::ResidualSquaredNorm(unsigned long nElmDomain) const {
  Kokkos::Timer timer;
  const double* r = st.Res.data();
  double sum = 0.0;
  Kokkos::parallel_reduce(
      "NEMO::ResidualSquaredNorm", Kokkos::RangePolicy<exec_space>(0, nElmDomain),
      KOKKOS_LAMBDA(const unsigned long k, double& a) { a += r[k] * r[k]; }, sum);
  updatePerf.norm += timer.seconds();
  updatePerf.nNorm++;
  return sum;
}

void CNEMOKokkosData::TimeSteppingFill(double cfl, double dt) {
  const auto c = st.LocalCFL, d = st.DeltaTime;
  Kokkos::parallel_for(
      "NEMO::TimeSteppingFill", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long i) {
        c(i) = cfl;
        d(i) = dt;
      });
  Kokkos::fence();
  MarkDeviceWritten(G_LOCALCFL);
  MarkDeviceWritten(G_TIMESTEP);
}

void CNEMOKokkosData::ExplicitRKUpdate(bool rk4Update, unsigned short iRKStep, double alpha, double timeCoeff,
                                       double funcCoeff, ResidualReduction& red) {
  Kokkos::Timer timer;
  EnsureDevice(G_TRUNCERR);
  EnsureDevice(G_SOLUTION);
  EnsureDevice(G_SOLUTION_OLD);
  if (rk4Update && st.SolutionNew.size() == 0)
    SU2_MPI::Error("CNEMOKokkosData: classical RK4 without the device Solution_New.", CURRENT_FUNCTION);
  ExplicitRKFunctor f;
  f.vol = geo.volume;
  f.pvol = geo.periodic_volume;
  f.dt = st.DeltaTime;
  f.res = st.Res;
  f.trunc = st.TruncErr;
  f.U = st.Solution;
  f.Uold = st.Solution_Old;
  f.Unew = st.SolutionNew;
  f.nVar = nVar;
  f.step = iRKStep;
  f.rk4 = rk4Update;
  f.alpha = alpha;
  f.timeCoeff = timeCoeff;
  f.funcCoeff = funcCoeff;
  Kokkos::parallel_reduce("NEMO::ExplicitRKIteration", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain), f, red);
  Kokkos::fence();
  accOnDevice = false;
  hostAccZero = false;
  MarkDeviceWritten(G_LINSYS);
  MarkDeviceWritten(G_SOLUTION);
  updatePerf.explicitUpd += timer.seconds();
  updatePerf.nExplicit++;
}

void CNEMOKokkosData::AdaptCFL(const CFLAdaptParams& p, double& cflMin, double& cflMax, double& cflSum) {
  auto& P = stagePerf[ST_CFL];
  Kokkos::Timer timer;
  EnsureDevice(G_LOCALCFL);
  EnsureDevice(G_UNDERRELAX);
  if (st.LocalCFL.size() == 0) SU2_MPI::Error("CNEMOKokkosData: CFL adaptation without device LocalCFL.", CURRENT_FUNCTION);
  AdaptCFLFunctor f;
  f.cfl = st.LocalCFL;
  f.ur = st.UnderRelax;
  f.p = p;
  CFLStats v;
  Kokkos::parallel_reduce("NEMO::AdaptCFLNumber", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain), f, v);
  Kokkos::fence();
  if (p.stats) {
    cflMin = v.mn;
    cflMax = v.mx;
    cflSum = v.sum;
  }
  MarkDeviceWritten(G_LOCALCFL);
  P.kernel += timer.seconds();
  P.calls++;
}

void CNEMOKokkosData::DualTimeResidual(bool secondOrder, bool impl, double timeStep, bool uploadInputs) {
  auto& P = stagePerf[ST_DUALTIME];
  Kokkos::Timer timer;
  if (!accOnDevice) SU2_MPI::Error("CNEMOKokkosData: dual-time stage without a device accumulator.", CURRENT_FUNCTION);
  if (ResidentActive()) {
    EnsureDevice(G_SOLUTION);
    EnsureDevice(G_TIMEN);
  } else {
    EnsureInputsOnDevice(uploadInputs, true, false);
    /*--- Host-authoritative: Solution_time_n(1) change once per time step on the host, uploaded every call. ---*/
    UploadGroup(G_TIMEN);
  }
  if (st.SolTimeN.size() == 0) SU2_MPI::Error("CNEMOKokkosData: dual time without Solution_time_n.", CURRENT_FUNCTION);
  Kokkos::fence();
  P.h2d += timer.seconds();
  timer.reset();
  const auto U = st.Solution, Un = st.SolTimeN, Unm1 = st.SolTimeN1, res = st.Res;
  const auto vol = geo.volume;
  const auto dia = geo.dia_ptr;
  const auto jac = st.JacVal;
  const int nv = nVar;
  const double TimeStep = timeStep;
  /*--- CFVMFlowSolverBase::SetResidual_DualTime, static mesh: per point independent. ---*/
  Kokkos::parallel_for(
      "NEMO::DualTimeResidual", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint) {
        const double Volume_nP1 = vol(iPoint);
        for (int iVar = 0; iVar < nv; iVar++) {
          if (!secondOrder)
            res(iPoint, iVar) += (U(iPoint, iVar) - Un(iPoint, iVar)) * Volume_nP1 / TimeStep;
          else
            res(iPoint, iVar) += (3.0 * U(iPoint, iVar) - 4.0 * Un(iPoint, iVar) + 1.0 * Unm1(iPoint, iVar)) *
                                 Volume_nP1 / (2.0 * TimeStep);
        }
        if (impl) {
          const double val = !secondOrder ? Volume_nP1 / TimeStep : (Volume_nP1 * 3.0) / (2.0 * TimeStep);
          const unsigned long base = dia(iPoint) * nv * nv;
          for (int iVar = 0; iVar < nv; iVar++) jac(base + iVar * (nv + 1)) += val;
        }
      });
  Kokkos::fence();
  P.kernel += timer.seconds();
  P.calls++;
}

void CNEMOKokkosData::DualTimeShift(double cfl) {
  auto& P = stagePerf[ST_DUALTIME];
  Kokkos::Timer timer;
  EnsureDevice(G_SOLUTION);
  EnsureDevice(G_TIMEN);
  if (st.SolTimeN.size() == 0) SU2_MPI::Error("CNEMOKokkosData: dual time without Solution_time_n.", CURRENT_FUNCTION);
  if (st.LocalCFL.size() == 0) st.LocalCFL = View1D<double>("NEMO::LocalCFL", nPoint);
  /*--- CIntegration::SetDualTime_Solver: Set_Solution_time_n1, Set_Solution_time_n, UnderRelaxation, LocalCFL on all
   *    points (copies / constants: bitwise). ---*/
  Kokkos::deep_copy(st.SolTimeN1, st.SolTimeN);
  Kokkos::deep_copy(st.SolTimeN, st.Solution);
  Kokkos::deep_copy(st.UnderRelax, 1.0);
  Kokkos::deep_copy(st.LocalCFL, cfl);
  Kokkos::fence();
  MarkDeviceWritten(G_TIMEN);
  MarkDeviceWritten(G_UNDERRELAX);
  MarkDeviceWritten(G_LOCALCFL);
  P.kernel += timer.seconds();
}

double CNEMOKokkosData::DualTimeMinDt(double num, double init) {
  EnsureDevice(G_LOCALCFL);
  const auto c = st.LocalCFL;
  double m = init;
  Kokkos::parallel_reduce(
      "NEMO::DualTimeMinDt", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long i, double& v) {
        const double x = num / c(i);
        v = (x < v) ? x : v; /*--- min(v, x) ---*/
      },
      Kokkos::Min<double>(m));
  /*--- Kokkos::Min starts from +max: fold in the CPU initial value (min is order independent). ---*/
  return (m < init) ? m : init;
}

void CNEMOKokkosData::DualTimeExplicitDt(double dtMax) {
  EnsureDevice(G_TIMESTEP);
  const auto d = st.DeltaTime;
  Kokkos::parallel_for(
      "NEMO::DualTimeExplicitDt", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long i) {
        d(i) = (d(i) < dtMax) ? d(i) : dtMax; /*--- min(dtMax, Delta_Time) ---*/
      });
  Kokkos::fence();
  MarkDeviceWritten(G_TIMESTEP);
}

double CNEMOKokkosData::DeviceGamma(unsigned long iPoint) {
  double g = 0.0;
  Kokkos::deep_copy(Kokkos::View<double, HostSpace, Unmanaged>(&g), Kokkos::subview(st.Gamma, iPoint));
  syncPerf[G_PRIMITIVE].bytes_d2h += sizeof(double);
  return g;
}

void CNEMOKokkosData::ReplayGasStateLastPoint(bool full) {
  if (nPoint == 0) return;
  auto& P = syncPerf[G_PRIMITIVE];
  const auto i = nPoint - 1;
  std::vector<double> V(nPrimVar), fdp(2);
  unsigned char status = 0;
  Kokkos::deep_copy(Kokkos::View<double*, HostSpace, Unmanaged>(V.data(), nPrimVar),
                    Kokkos::subview(st.Primitive, i, Kokkos::ALL()));
  Kokkos::deep_copy(Kokkos::View<double*, HostSpace, Unmanaged>(fdp.data(), 2), Kokkos::subview(st.FirstDP, i, Kokkos::ALL()));
  Kokkos::deep_copy(Kokkos::View<unsigned char, HostSpace, Unmanaged>(&status), Kokkos::subview(st.Status, i));
  P.bytes_d2h += double(nPrimVar + 2) * sizeof(double) + 1;
  su2kokkos::nemo::ReplayHostGasState(*gasModel, nDim, V.data(), status, fdp.data(), full);
}

#endif  // HAVE_KOKKOS
