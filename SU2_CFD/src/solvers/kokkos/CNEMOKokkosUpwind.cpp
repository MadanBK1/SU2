/*!
 * \file CNEMOKokkosUpwind.cpp
 * \brief Device (Kokkos) NEMO upwind residual, Step 5 of NEMO_KOKKOS_PORT_PLAN.md:
 *        CNEMOEulerSolver::Upwind_Residual for the AUSM family (AUSM, AUSM+M) incl. MUSCL reconstruction,
 *        CheckNonPhys + sticky Non_Physical, consistent extrapolation, flux, approximate AUSM Jacobian
 *        (CUpwAUSM_SLAU_Base_NEMO::ComputeJacobian) and residual / block-CSR Jacobian accumulation.
 *
 * Structure (all statements in CPU source order, nothing compiler-specific pinned):
 *  1. edge kernel:  MUSCL reconstruction of both states, raw CheckNonPhys flags (MUSCL only);
 *  2. point kernel: CVariable::SetNon_Physical / GetNon_Physical replayed per point over its edges in increasing
 *                   edge order (exactly the CPU edge-loop order seen by that point), giving the flag each edge reads
 *                   back and the final Non_Physical / Non_Physical_Counter; + reduction of the non-physical counter
 *                   and of the last (edge, side) that calls ComputeConsistentExtrapolation (host gas-state replay);
 *  3. edge kernel:  states (reconstructed or cell values), Eve/Gamma at the reconstructed state, AUSM / AUSM+M flux
 *                   and Jacobian with MAXNVAR stack arrays, CheckResidualNaNs, Kokkos atomics into Res / JacVal
 *                   (UpdateBlocks: ii += Ji, ij += Jj, ji -= Ji, jj -= Jj; AddBlock(i), SubtractBlock(j)).
 * On the Serial backend kernel 3 runs the edges sequentially, so every residual/Jacobian entry receives its
 * contributions in the CPU order -> bitwise in strict IEEE builds. On GPUs the atomic order is not deterministic
 * (ULP-level, run-to-run); an edge-coloured variant is a TODO (see the plan).
 *
 * CPU semantics reproduced on purpose (not fixed silently, see NEMO_KOKKOS_PORT_PLAN.md):
 *  - ComputeConsistentExtrapolation computes only Eve and Gamma (TODO in the CPU code): for a reconstructed state
 *    dPdU/dTdU/dTvedU/Cvve are the zero-initialised local arrays of Upwind_Residual, so the AUSM Jacobian of a
 *    reconstructed state uses dPdU = 0 (cell values use the node dPdU).
 *  - Primitive entries >= nPrimVarGrad of a reconstructed state are 0 (static arrays, never written).
 *  - The reconstruction also extrapolates P, rho, h, a, rhoCv_tr, rhoCv_ve independently (not re-derived).
 *  - CheckResidualNaNs: an edge with a NaN in the flux or (implicit) a Jacobian is skipped entirely.
 *  - The AUSM Jacobian splits electrons/heavy species with CNEMONumerics::nEl = config IONIZATION, while the gas
 *    model (CSU2TCLib, e.g. AIR-7 forces ionization) may have nEl = 1: the numerics' count is used.
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
#include <string>
#include <vector>

#include "../../../../Common/include/CConfig.hpp"
#include "../../../../Common/include/geometry/CGeometry.hpp"
#include "../../../include/variables/CNEMOEulerVariable.hpp"

namespace {

using namespace su2kokkos::nemo;
using exec_space = CNEMOKokkosData::exec_space;
using HostSpace = Kokkos::HostSpace;
using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
template <class T>
using View1D = CNEMOKokkosData::View1D<T>;
template <class T>
using View2D = CNEMOKokkosData::View2D<T>;
template <class T>
using View3D = CNEMOKokkosData::View3D<T>;

constexpr int MAXNV = MAXNVAR;      /*!< >= nVar = nSpecies + nDim + 2 */
constexpr int MAXNP = MAXNPRIMVAR;  /*!< >= nPrimVar */

/*--- std::min / std::max semantics. ---*/
KOKKOS_INLINE_FUNCTION double kmax(double a, double b) { return (a < b) ? b : a; }
KOKKOS_INLINE_FUNCTION double kmin(double a, double b) { return (b < a) ? b : a; }
KOKKOS_INLINE_FUNCTION long kmaxl(long a, long b) { return (a < b) ? b : a; }

struct UpwArgs {
  /*--- geometry ---*/
  View2D<unsigned long> edge_nodes;
  View2D<double> edge_normal, coord;
  View1D<unsigned long> dia_ptr;
  View2D<unsigned long> edge_nnz;
  /*--- state ---*/
  View2D<double> V, dPdU, eves;
  View1D<double> Gamma, Sensor;
  View3D<double> grad;
  View2D<double> lim;
  /*--- outputs ---*/
  View2D<double> res;
  View1D<double> jac;
  View2D<unsigned char> eflag;
  View1D<double> lastRec;
  NEMOGasTables gas;
  int nDim, ns, nVar, nPrimVar, nPrimVarGrad;
  int nElNum; /*!< electrons as the numerics count them (config IONIZATION), not the gas model's nEl */
  int scheme;
  bool implicit, muscl, limiter, vanAlbada;
  double kappa, musclRamp, Minf, eps, pi;
};

/*! \brief CNEMOEulerSolver::CheckNonPhys. */
KOKKOS_INLINE_FUNCTION bool CheckNonPhys(int ns, int nd, const double* V) {
  const NEMOPrimIndices idx(ns, nd);
  bool nonPhys = false;
  const double Tmin = 50.0, Tmax = 8E4;
  const double Tvemin = 50.0, Tvemax = 8E4;
  for (int s = 0; s < ns; s++)
    if (V[idx.RHOS() + s] < 0.0) nonPhys = true;
  if (V[idx.P()] < 0.0) nonPhys = true;
  if (V[idx.T()] < Tmin || V[idx.T()] > Tmax) nonPhys = true;
  if (V[idx.TVE()] < Tvemin || V[idx.TVE()] > Tvemax) nonPhys = true;
  if (V[idx.A()] < 0.0) nonPhys = true;
  return nonPhys;
}

/*!
 * \brief The MUSCL block of Upwind_Residual: Project_Grad (CSolver::MUSCL_Reconstruction + umusclProjection),
 *        edge limiter lim_ij, reconstructed Primitive_i/j (entries >= nPrimVarGrad stay 0 as on the CPU).
 */
template <int ND>
KOKKOS_INLINE_FUNCTION void Reconstruct(const UpwArgs& a, unsigned long iPoint, unsigned long jPoint, double* Primitive_i,
                                        double* Primitive_j) {
  const int nvg = a.nPrimVarGrad;
  double Vector_ij[ND] = {0.0};
  for (int iDim = 0; iDim < ND; iDim++) Vector_ij[iDim] = a.coord(jPoint, iDim) - a.coord(iPoint, iDim);

  double Project_Grad_i[MAXNP] = {0.0}, Project_Grad_j[MAXNP] = {0.0};
  double lim_i = 2.0;
  double lim_j = 2.0;

  for (int iVar = 0; iVar < nvg; iVar++) {
    const double V_ij = a.V(jPoint, iVar) - a.V(iPoint, iVar);

    auto muscl = [&](unsigned long p) {
      double project_grad = 0.0;
      for (int iDim = 0; iDim < ND; ++iDim) project_grad += a.grad(p, iVar, iDim) * Vector_ij[iDim];
      const double cent = 0.5 * V_ij;
      const double upw = project_grad - cent;
      return a.musclRamp * ((1.0 - a.kappa) * upw + (1.0 + a.kappa) * cent);
    };
    Project_Grad_i[iVar] = muscl(iPoint);
    Project_Grad_j[iVar] = muscl(jPoint);

    if (a.limiter) {
      if (a.vanAlbada) {
        auto va = [&](double proj, double delta) {
          return delta * (proj + delta) / (proj * proj + delta * delta + a.eps);
        };
        const double va_lim_i = va(Project_Grad_i[iVar], V_ij);
        const double va_lim_j = va(Project_Grad_j[iVar], V_ij);
        lim_i = kmin(lim_i, va_lim_i);
        lim_j = kmin(lim_j, va_lim_j);
      } else {
        lim_i = kmin(lim_i, a.lim(iPoint, iVar));
        lim_j = kmin(lim_j, a.lim(jPoint, iVar));
      }
    } else {
      lim_i = lim_j = 1.0;
    }
  }
  const double lim_ij = kmin(lim_i, lim_j);

  for (int iVar = 0; iVar < nvg; iVar++) {
    Primitive_i[iVar] = a.V(iPoint, iVar) + 0.5 * lim_ij * Project_Grad_i[iVar];
    Primitive_j[iVar] = a.V(jPoint, iVar) - 0.5 * lim_ij * Project_Grad_j[iVar];
  }
  for (int iVar = nvg; iVar < a.nPrimVar; iVar++) Primitive_i[iVar] = Primitive_j[iVar] = 0.0;
}

/*!
 * \brief CUpwAUSM_SLAU_Base_NEMO::ComputeResidual (+ ComputeInterfaceQuantities of CUpwAUSM_NEMO / CUpwAUSMPLUSM_NEMO,
 *        + ComputeJacobian if implicit). Jacobians are row-major nVar x nVar.
 */
template <int ND>
KOKKOS_INLINE_FUNCTION void AUSMResidual(const UpwArgs& a, const double* Normal, const double* V_i, const double* V_j,
                                         const double* eve_i, const double* eve_j, const double* dPdU_i,
                                         const double* dPdU_j, double Gamma_i, double Gamma_j, double Sensor_i,
                                         double Sensor_j, double* Flux, double* Jacobian_i, double* Jacobian_j) {
  const NEMOGasTables& g = a.gas;
  const int nSpecies = a.ns, nDim = ND, nVar = a.nVar, nEl = a.nElNum;
  const NEMOPrimIndices idx(nSpecies, ND);
  const int RHOS_INDEX = idx.RHOS(), VEL_INDEX = idx.VEL(), P_INDEX = idx.P(), RHO_INDEX = idx.RHO(),
            H_INDEX = idx.H(), A_INDEX = idx.A(), RHOCVTR_INDEX = idx.RHOCVTR();

  /*--- Compute geometric quantities ---*/
  double sq = 0.0;
  for (int iDim = 0; iDim < nDim; ++iDim) sq += Normal[iDim] * Normal[iDim];
  const double Area = Kokkos::sqrt(sq);

  double UnitNormal[ND];
  for (int iDim = 0; iDim < nDim; iDim++) UnitNormal[iDim] = Normal[iDim] / Area;

  /*--- Pull stored primitive variables ---*/
  double rhos_i[MAXNS], rhos_j[MAXNS], Velocity_i[ND], Velocity_j[ND];
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    rhos_i[iSpecies] = V_i[RHOS_INDEX + iSpecies];
    rhos_j[iSpecies] = V_j[RHOS_INDEX + iSpecies];
  }
  for (int iDim = 0; iDim < nDim; iDim++) {
    Velocity_i[iDim] = V_i[VEL_INDEX + iDim];
    Velocity_j[iDim] = V_j[VEL_INDEX + iDim];
  }

  const double Pressure_i = V_i[P_INDEX];
  const double Enthalpy_i = V_i[H_INDEX];
  const double Density_i = V_i[RHO_INDEX];
  const double SoundSpeed_i = V_i[A_INDEX];

  const double Pressure_j = V_j[P_INDEX];
  const double Enthalpy_j = V_j[H_INDEX];
  const double Density_j = V_j[RHO_INDEX];
  const double SoundSpeed_j = V_j[A_INDEX];

  double e_ve_i = 0, e_ve_j = 0;
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    e_ve_i += (V_i[RHOS_INDEX + iSpecies] * eve_i[iSpecies]) / Density_i;
    e_ve_j += (V_j[RHOS_INDEX + iSpecies] * eve_j[iSpecies]) / Density_j;
  }

  /*--- Projected velocities ---*/
  double ProjVelocity_i = 0.0, ProjVelocity_j = 0.0;
  for (int iDim = 0; iDim < nDim; ++iDim) ProjVelocity_i += Velocity_i[iDim] * UnitNormal[iDim];
  for (int iDim = 0; iDim < nDim; ++iDim) ProjVelocity_j += Velocity_j[iDim] * UnitNormal[iDim];

  /*--- Compute mass and pressure fluxes of specific scheme (ComputeInterfaceQuantities) ---*/
  double PressureFlux[ND], A_F[2], M_F, M_L, M_R;

  if (a.scheme == static_cast<int>(CNEMOKokkosData::UpwindScheme::AUSM)) {
    /*--- CUpwAUSM_NEMO ---*/
    A_F[0] = SoundSpeed_i;
    A_F[1] = SoundSpeed_j;
    M_L = ProjVelocity_i / A_F[0];
    M_R = ProjVelocity_j / A_F[1];

    double M_LP, M_RM, P_LP, P_RM;

    if (Kokkos::fabs(M_L) <= 1.0) {
      M_LP = 0.25 * (M_L + 1.0) * (M_L + 1.0);
      P_LP = 0.25 * Pressure_i * (M_L + 1.0) * (M_L + 1.0) * (2.0 - M_L);
    } else {
      M_LP = 0.5 * (M_L + Kokkos::fabs(M_L));
      P_LP = 0.5 * Pressure_i * (M_L + Kokkos::fabs(M_L)) / M_L;
    }

    if (Kokkos::fabs(M_R) <= 1.0) {
      M_RM = -0.25 * (M_R - 1.0) * (M_R - 1.0);
      P_RM = 0.25 * Pressure_j * (M_R - 1.0) * (M_R - 1.0) * (2.0 + M_R);
    } else {
      M_RM = 0.5 * (M_R - Kokkos::fabs(M_R));
      P_RM = 0.5 * Pressure_j * (M_R - Kokkos::fabs(M_R)) / M_R;
    }

    M_F = (M_LP + M_RM);
    for (int iDim = 0; iDim < nDim; iDim++) PressureFlux[iDim] = (P_LP + P_RM) * UnitNormal[iDim];
  } else {
    /*--- CUpwAUSMPLUSM_NEMO ---*/
    const double Minf = a.Minf, beta = 1.0 / 8.0, PI_NUMBER = a.pi;
    double sq_veli = 0.0, sq_velj = 0.0;
    for (int iDim = 0; iDim < nDim; ++iDim) sq_veli += Velocity_i[iDim] * Velocity_i[iDim];
    for (int iDim = 0; iDim < nDim; ++iDim) sq_velj += Velocity_j[iDim] * Velocity_j[iDim];

    const double Enthalpy_norm = 0.5 * (Enthalpy_i + Enthalpy_j);
    const double Gamma_ij = 0.5 * (Gamma_i + Gamma_j);
    const double A_s = Kokkos::sqrt(2.0 * Enthalpy_norm * (Gamma_ij - 1.0) / (Gamma_ij + 1.0));

    double A_FF;
    if (0.5 * (ProjVelocity_i + ProjVelocity_j) >= 0.0)
      A_FF = A_s * A_s / kmax(Kokkos::fabs(ProjVelocity_i), A_s);
    else
      A_FF = A_s * A_s / kmax(Kokkos::fabs(ProjVelocity_j), A_s);

    A_F[0] = A_F[1] = A_FF;
    M_L = ProjVelocity_i / A_F[0];
    M_R = ProjVelocity_j / A_F[1];

    const double Density_F = 0.5 * (Density_i + Density_j);
    const double MF_sq = 0.5 * (sq_veli + sq_velj) / (A_FF * A_FF);

    const double param1 = kmax(MF_sq, Minf * Minf);
    const double Mrefsq = (kmin(1.0, param1));
    const double fa = 2.0 * Kokkos::sqrt(Mrefsq) - Mrefsq;

    const double alpha = 3.0 / 16.0 * (-4.0 + 5.0 * fa * fa);
    const double f = 0.5 * (1 - Kokkos::cos(PI_NUMBER * kmin(1.0, kmax(Kokkos::fabs(M_L), Kokkos::fabs(M_R)))));

    const double h = kmin(Sensor_i, Sensor_j);
    const double gg = 0.5 * (1 + Kokkos::cos(PI_NUMBER * h));
    const double f0 = kmin(1.0, kmax(f, Minf * Minf));

    const double M_p = -0.5 * (1.0 - f) * (Pressure_j - Pressure_i) / (Density_F * A_FF * A_FF) * (1.0 - gg);

    double M_LP, M_RM, P_LP, P_RM;

    if (Kokkos::fabs(M_L) <= 1.0) {
      M_LP = 0.25 * (M_L + 1.0) * (M_L + 1.0) + beta * (M_L * M_L - 1.0) * (M_L * M_L - 1.0);
      P_LP = (0.25 * (M_L + 1.0) * (M_L + 1.0) * (2.0 - M_L) + alpha * M_L * (M_L * M_L - 1.0) * (M_L * M_L - 1.0));
    } else {
      M_LP = 0.5 * (M_L + Kokkos::fabs(M_L));
      P_LP = 0.5 * (M_L + Kokkos::fabs(M_L)) / M_L;
    }

    if (Kokkos::fabs(M_R) <= 1.0) {
      M_RM = -0.25 * (M_R - 1.0) * (M_R - 1.0) - beta * (M_R * M_R - 1.0) * (M_R * M_R - 1.0);
      P_RM = (0.25 * (M_R - 1.0) * (M_R - 1.0) * (2.0 + M_R) - alpha * M_R * (M_R * M_R - 1.0) * (M_R * M_R - 1.0));
    } else {
      M_RM = 0.5 * (M_R - Kokkos::fabs(M_R));
      P_RM = 0.5 * (M_R - Kokkos::fabs(M_R)) / M_R;
    }

    M_F = M_LP + M_RM + M_p;

    const double pFi = f0 * (P_LP + P_RM - 1.0) * 0.5 * (Pressure_i + Pressure_j);

    double P_un[ND] = {0.0};
    for (int iDim = 0; iDim < nDim; iDim++) {
      const double Vel_L = (Velocity_i[iDim] + ProjVelocity_i * UnitNormal[iDim]);
      const double Vel_R = (Velocity_j[iDim] - ProjVelocity_j * UnitNormal[iDim]);
      P_un[iDim] = -gg * Gamma_ij * 0.5 * (Pressure_i + Pressure_j) / A_FF * P_LP * P_RM * (Vel_R - Vel_L);
    }

    for (int iDim = 0; iDim < nDim; iDim++) {
      PressureFlux[iDim] =
          (0.5 * (Pressure_j + Pressure_i) + 0.5 * (P_LP - P_RM) * (Pressure_i - Pressure_j) + pFi) * UnitNormal[iDim] +
          P_un[iDim];
    }
  }

  const double MassFlux_i = M_F * A_F[0];
  const double MassFlux_j = M_F * A_F[1];

  const double DissFlux_i = Kokkos::fabs(MassFlux_i);
  const double DissFlux_j = Kokkos::fabs(MassFlux_j);

  /*--- Assign left & right convective flux vectors ---*/
  double Fc_L[MAXNV], Fc_R[MAXNV];
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    Fc_L[iSpecies] = rhos_i[iSpecies];
    Fc_R[iSpecies] = rhos_j[iSpecies];
  }
  for (int iDim = 0; iDim < nDim; iDim++) {
    Fc_L[nSpecies + iDim] = Density_i * Velocity_i[iDim];
    Fc_R[nSpecies + iDim] = Density_j * Velocity_j[iDim];
  }
  Fc_L[nSpecies + nDim] = Density_i * Enthalpy_i;
  Fc_R[nSpecies + nDim] = Density_j * Enthalpy_j;
  Fc_L[nSpecies + nDim + 1] = Density_i * e_ve_i;
  Fc_R[nSpecies + nDim + 1] = Density_j * e_ve_j;

  /*--- Compute numerical flux ---*/
  for (int iVar = 0; iVar < nVar; iVar++)
    Flux[iVar] = 0.5 * ((MassFlux_i + DissFlux_i) * Fc_L[iVar] + (MassFlux_j - DissFlux_j) * Fc_R[iVar]) * Area;

  for (int iDim = 0; iDim < nDim; iDim++) Flux[nSpecies + iDim] += PressureFlux[iDim] * Area;

  if (!a.implicit) return;

  /*====================== ComputeJacobian ======================*/
  const double Ru = g.Ru;
  const double rhoCvtr_i = V_i[RHOCVTR_INDEX];
  const double rhoCvtr_j = V_j[RHOCVTR_INDEX];
#define JI(r, c) Jacobian_i[(r) * nVar + (c)]
#define JJ(r, c) Jacobian_j[(r) * nVar + (c)]

  for (int iVar = 0; iVar < nVar; iVar++) {
    for (int jVar = 0; jVar < nVar; jVar++) {
      JI(iVar, jVar) = 0.0;
      JJ(iVar, jVar) = 0.0;
    }
  }

  const double* Fc_LR = (M_F >= 0.0) ? Fc_L : Fc_R;
  double A_LR = 0;
  if (M_F >= 0.0)
    A_LR = A_F[0];
  else
    A_LR = A_F[1];

  double da_L[MAXNV], da_R[MAXNV];

  // Electrons only
  for (int iSpecies = 0; iSpecies < nEl; iSpecies++) {
    da_L[iSpecies] = 1.0 / (2.0 * SoundSpeed_i * Density_i) * (1 + dPdU_i[nSpecies + nDim]) *
                     (dPdU_i[iSpecies] - Pressure_i / Density_i);
    da_R[iSpecies] = 1.0 / (2.0 * SoundSpeed_j * Density_j) * (1 + dPdU_j[nSpecies + nDim]) *
                     (dPdU_j[iSpecies] - Pressure_j / Density_j);
  }

  // Heavy species
  for (int iSpecies = nEl; iSpecies < nSpecies; iSpecies++) {
    const double Ms = g.MolarMass(iSpecies), Cvtr = g.Cvtrs(iSpecies);
    da_L[iSpecies] =
        1.0 / (2.0 * SoundSpeed_i) *
        (1 / rhoCvtr_i * (Ru / Ms - Cvtr * dPdU_i[nSpecies + nDim]) * Pressure_i / Density_i +
         1.0 / Density_i * (1.0 + dPdU_i[nSpecies + nDim]) * (dPdU_i[iSpecies] - Pressure_i / Density_i));
    da_R[iSpecies] =
        1.0 / (2.0 * SoundSpeed_j) *
        (1 / rhoCvtr_j * (Ru / Ms - Cvtr * dPdU_j[nSpecies + nDim]) * Pressure_j / Density_j +
         1.0 / Density_j * (1.0 + dPdU_j[nSpecies + nDim]) * (dPdU_j[iSpecies] - Pressure_j / Density_j));
  }

  /*--- Sound speed derivatives: Momentum ---*/
  for (int iDim = 0; iDim < nDim; iDim++) {
    da_L[nSpecies + iDim] = -1.0 / (2.0 * Density_i * SoundSpeed_i) *
                            ((1.0 + dPdU_i[nSpecies + nDim]) * dPdU_i[nSpecies + nDim]) * Velocity_i[iDim];
    da_R[nSpecies + iDim] = -1.0 / (2.0 * Density_j * SoundSpeed_j) *
                            ((1.0 + dPdU_j[nSpecies + nDim]) * dPdU_j[nSpecies + nDim]) * Velocity_j[iDim];
  }

  /*--- Sound speed derivatives: Energy ---*/
  da_L[nSpecies + nDim] =
      1.0 / (2.0 * Density_i * SoundSpeed_i) * ((1.0 + dPdU_i[nSpecies + nDim]) * dPdU_i[nSpecies + nDim]);
  da_R[nSpecies + nDim] =
      1.0 / (2.0 * Density_j * SoundSpeed_j) * ((1.0 + dPdU_j[nSpecies + nDim]) * dPdU_j[nSpecies + nDim]);

  /*--- Sound speed derivatives: Vib-el energy ---*/
  da_L[nSpecies + nDim + 1] =
      1.0 / (2.0 * Density_i * SoundSpeed_i) * ((1.0 + dPdU_i[nSpecies + nDim]) * dPdU_i[nSpecies + nDim + 1]);
  da_R[nSpecies + nDim + 1] =
      1.0 / (2.0 * Density_j * SoundSpeed_j) * ((1.0 + dPdU_j[nSpecies + nDim]) * dPdU_j[nSpecies + nDim + 1]);

  /*--- pow(x, 2) is evaluated as x*x (GCC folds pow(x, 2.0) to x*x, which is exact rounding of x^2). ---*/
  const double a2_i = SoundSpeed_i * SoundSpeed_i, a2_j = SoundSpeed_j * SoundSpeed_j;

  /*--- Left state Jacobian ---*/
  if (M_F >= 0) {
    for (int iVar = 0; iVar < nSpecies + nDim; iVar++) {
      for (int jVar = 0; jVar < nVar; jVar++) {
        JI(iVar, jVar) += M_F * Fc_L[iVar] * da_L[jVar];
      }
      JI(iVar, iVar) += M_F * SoundSpeed_i;
    }
    for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
      JI(nSpecies + nDim, iSpecies) += M_F * (dPdU_i[iSpecies] * SoundSpeed_i + Density_i * Enthalpy_i * da_L[iSpecies]);
    }
    for (int iDim = 0; iDim < nDim; iDim++) {
      JI(nSpecies + nDim, nSpecies + iDim) +=
          M_F *
          (-dPdU_i[nSpecies + nDim] * Velocity_i[iDim] * SoundSpeed_i + Density_i * Enthalpy_i * da_L[nSpecies + iDim]);
    }
    JI(nSpecies + nDim, nSpecies + nDim) +=
        M_F * ((1.0 + dPdU_i[nSpecies + nDim]) * SoundSpeed_i + Density_i * Enthalpy_i * da_L[nSpecies + nDim]);
    JI(nSpecies + nDim, nSpecies + nDim + 1) +=
        M_F * (dPdU_i[nSpecies + nDim + 1] * SoundSpeed_i + Density_i * Enthalpy_i * da_L[nSpecies + nDim + 1]);
    for (int jVar = 0; jVar < nVar; jVar++) {
      JI(nSpecies + nDim + 1, jVar) += M_F * Fc_L[nSpecies + nDim + 1] * da_L[jVar];
    }
    JI(nSpecies + nDim + 1, nSpecies + nDim + 1) += M_F * SoundSpeed_i;
  }

  /*--- Calculate derivatives of the split pressure flux ---*/
  if ((M_F >= 0) || ((M_F < 0) && (Kokkos::fabs(M_F) <= 1.0))) {
    double dM_LP[MAXNV], dP_LP[MAXNV];
    if (Kokkos::fabs(M_L) <= 1.0) {
      /*--- Mach number ---*/
      for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++)
        dM_LP[iSpecies] =
            0.5 * (M_L + 1.0) * (-ProjVelocity_i / (Density_i * SoundSpeed_i) - ProjVelocity_i * da_L[iSpecies] / (a2_i));
      for (int iDim = 0; iDim < nDim; iDim++)
        dM_LP[nSpecies + iDim] = 0.5 * (M_L + 1.0) *
                                 (-ProjVelocity_i / (a2_i)*da_L[nSpecies + iDim] +
                                  UnitNormal[iDim] / (Density_i * SoundSpeed_i));
      dM_LP[nSpecies + nDim] = 0.5 * (M_L + 1.0) * (-ProjVelocity_i / (a2_i)*da_L[nSpecies + nDim]);
      dM_LP[nSpecies + nDim + 1] = 0.5 * (M_L + 1.0) * (-ProjVelocity_i / (a2_i)*da_L[nSpecies + nDim + 1]);

      /*--- Pressure ---*/
      for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++)
        dP_LP[iSpecies] = 0.25 * (M_L + 1.0) *
                          (dPdU_i[iSpecies] * (M_L + 1.0) * (2.0 - M_L) +
                           Pressure_i *
                               (-ProjVelocity_i / (Density_i * SoundSpeed_i) -
                                ProjVelocity_i * da_L[iSpecies] / (a2_i)) *
                               (3.0 - 3.0 * M_L));
      for (int iDim = 0; iDim < nDim; iDim++)
        dP_LP[nSpecies + iDim] = 0.25 * (M_L + 1.0) *
                                 (-Velocity_i[iDim] * dPdU_i[nSpecies + nDim] * (M_L + 1.0) * (2.0 - M_L) +
                                  Pressure_i *
                                      (-ProjVelocity_i / (a2_i)*da_L[nSpecies + iDim] +
                                       UnitNormal[iDim] / (Density_i * SoundSpeed_i)) *
                                      (3.0 - 3.0 * M_L));
      dP_LP[nSpecies + nDim] =
          0.25 * (M_L + 1.0) *
          (dPdU_i[nSpecies + nDim] * (M_L + 1.0) * (2.0 - M_L) +
           Pressure_i * (-ProjVelocity_i / (a2_i)*da_L[nSpecies + nDim]) * (3.0 - 3.0 * M_L));
      dP_LP[nSpecies + nDim + 1] =
          0.25 * (M_L + 1.0) *
          (dPdU_i[nSpecies + nDim + 1] * (M_L + 1.0) * (2.0 - M_L) +
           Pressure_i * (-ProjVelocity_i / (a2_i)*da_L[nSpecies + nDim + 1]) * (3.0 - 3.0 * M_L));
    } else {
      /*--- Mach number ---*/
      for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++)
        dM_LP[iSpecies] = -ProjVelocity_i / (Density_i * SoundSpeed_i) - ProjVelocity_i * da_L[iSpecies] / (a2_i);
      for (int iDim = 0; iDim < nDim; iDim++)
        dM_LP[nSpecies + iDim] =
            -ProjVelocity_i / (a2_i)*da_L[nSpecies + iDim] + UnitNormal[iDim] / (Density_i * SoundSpeed_i);
      dM_LP[nSpecies + nDim] = -ProjVelocity_i / (a2_i)*da_L[nSpecies + nDim];
      dM_LP[nSpecies + nDim + 1] = -ProjVelocity_i / (a2_i)*da_L[nSpecies + nDim + 1];

      /*--- Pressure ---*/
      for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) dP_LP[iSpecies] = dPdU_i[iSpecies];
      for (int iDim = 0; iDim < nDim; iDim++) dP_LP[nSpecies + iDim] = (-Velocity_i[iDim] * dPdU_i[nSpecies + nDim]);
      dP_LP[nSpecies + nDim] = dPdU_i[nSpecies + nDim];
      dP_LP[nSpecies + nDim + 1] = dPdU_i[nSpecies + nDim + 1];
    }

    /*--- dM contribution ---*/
    for (int iVar = 0; iVar < nVar; iVar++) {
      for (int jVar = 0; jVar < nVar; jVar++) {
        JI(iVar, jVar) += dM_LP[jVar] * Fc_LR[iVar] * A_LR;
      }
    }

    /*--- Jacobian contribution: dP terms ---*/
    for (int iDim = 0; iDim < nDim; iDim++) {
      for (int iVar = 0; iVar < nVar; iVar++) {
        JI(nSpecies + iDim, iVar) += dP_LP[iVar] * UnitNormal[iDim];
      }
    }
  }

  /*--- Right state Jacobian ---*/
  if (M_F < 0) {
    for (int iVar = 0; iVar < nSpecies + nDim; iVar++) {
      for (int jVar = 0; jVar < nVar; jVar++) {
        JJ(iVar, jVar) += M_F * Fc_R[iVar] * da_R[jVar];
      }
      JJ(iVar, iVar) += M_F * SoundSpeed_j;
    }
    for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
      JJ(nSpecies + nDim, iSpecies) += M_F * (dPdU_j[iSpecies] * SoundSpeed_j + Density_j * Enthalpy_j * da_R[iSpecies]);
    }
    for (int iDim = 0; iDim < nDim; iDim++) {
      JJ(nSpecies + nDim, nSpecies + iDim) +=
          M_F *
          (-dPdU_j[nSpecies + nDim] * Velocity_j[iDim] * SoundSpeed_j + Density_j * Enthalpy_j * da_R[nSpecies + iDim]);
    }
    JJ(nSpecies + nDim, nSpecies + nDim) +=
        M_F * ((1.0 + dPdU_j[nSpecies + nDim]) * SoundSpeed_j + Density_j * Enthalpy_j * da_R[nSpecies + nDim]);
    JJ(nSpecies + nDim, nSpecies + nDim + 1) +=
        M_F * (dPdU_j[nSpecies + nDim + 1] * SoundSpeed_j + Density_j * Enthalpy_j * da_R[nSpecies + nDim + 1]);
    for (int jVar = 0; jVar < nVar; jVar++) {
      JJ(nSpecies + nDim + 1, jVar) += M_F * Fc_R[nSpecies + nDim + 1] * da_R[jVar];
    }
    JJ(nSpecies + nDim + 1, nSpecies + nDim + 1) += M_F * SoundSpeed_j;
  }

  /*--- Calculate derivatives of the split pressure flux ---*/
  if ((M_F < 0) || ((M_F >= 0) && (Kokkos::fabs(M_F) <= 1.0))) {
    double dM_RM[MAXNV], dP_RM[MAXNV];
    if (Kokkos::fabs(M_R) <= 1.0) {
      /*--- Mach ---*/
      for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++)
        dM_RM[iSpecies] =
            -0.5 * (M_R - 1.0) * (-ProjVelocity_j / (Density_j * SoundSpeed_j) - ProjVelocity_j * da_R[iSpecies] / (a2_j));
      for (int iDim = 0; iDim < nDim; iDim++)
        dM_RM[nSpecies + iDim] = -0.5 * (M_R - 1.0) *
                                 (-ProjVelocity_j / (a2_j)*da_R[nSpecies + iDim] +
                                  UnitNormal[iDim] / (Density_j * SoundSpeed_j));
      dM_RM[nSpecies + nDim] = -0.5 * (M_R - 1.0) * (-ProjVelocity_j / (a2_j)*da_R[nSpecies + nDim]);
      dM_RM[nSpecies + nDim + 1] = -0.5 * (M_R - 1.0) * (-ProjVelocity_j / (a2_j)*da_R[nSpecies + nDim + 1]);

      /*--- Pressure ---*/
      for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++)
        dP_RM[iSpecies] = 0.25 * (M_R - 1.0) *
                          (dPdU_j[iSpecies] * (M_R - 1.0) * (2.0 + M_R) +
                           Pressure_j *
                               (-ProjVelocity_j / (Density_j * SoundSpeed_j) -
                                ProjVelocity_j * da_R[iSpecies] / (a2_j)) *
                               (3.0 + 3.0 * M_R));
      for (int iDim = 0; iDim < nDim; iDim++)
        dP_RM[nSpecies + iDim] = 0.25 * (M_R - 1.0) *
                                 ((-Velocity_j[iDim] * dPdU_j[nSpecies + nDim]) * (M_R - 1.0) * (2.0 + M_R) +
                                  Pressure_j *
                                      (-ProjVelocity_j / (a2_j)*da_R[nSpecies + iDim] +
                                       UnitNormal[iDim] / (Density_j * SoundSpeed_j)) *
                                      (3.0 + 3.0 * M_R));
      dP_RM[nSpecies + nDim] =
          0.25 * (M_R - 1.0) *
          (dPdU_j[nSpecies + nDim] * (M_R - 1.0) * (2.0 + M_R) +
           Pressure_j * (-ProjVelocity_j / (a2_j)*da_R[nSpecies + nDim]) * (3.0 + 3.0 * M_R));
      dP_RM[nSpecies + nDim + 1] =
          0.25 * (M_R - 1.0) *
          (dPdU_j[nSpecies + nDim + 1] * (M_R - 1.0) * (2.0 + M_R) +
           Pressure_j * (-ProjVelocity_j / (a2_j)*da_R[nSpecies + nDim + 1]) * (3.0 + 3.0 * M_R));

    } else {
      /*--- Mach ---*/
      for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++)
        dM_RM[iSpecies] = -ProjVelocity_j / (Density_j * SoundSpeed_j) - ProjVelocity_j * da_R[iSpecies] / (a2_j);
      for (int iDim = 0; iDim < nDim; iDim++)
        dM_RM[nSpecies + iDim] =
            -ProjVelocity_j / (a2_j)*da_R[nSpecies + iDim] + UnitNormal[iDim] / (Density_j * SoundSpeed_j);
      dM_RM[nSpecies + nDim] = -ProjVelocity_j / (a2_j)*da_R[nSpecies + nDim];
      dM_RM[nSpecies + nDim + 1] = -ProjVelocity_j / (a2_j)*da_R[nSpecies + nDim + 1];

      /*--- Pressure ---*/
      for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) dP_RM[iSpecies] = dPdU_j[iSpecies];
      for (int iDim = 0; iDim < nDim; iDim++) dP_RM[nSpecies + iDim] = -Velocity_j[iDim] * dPdU_j[nSpecies + nDim];
      dP_RM[nSpecies + nDim] = dPdU_j[nSpecies + nDim];
      dP_RM[nSpecies + nDim + 1] = dPdU_j[nSpecies + nDim + 1];
    }

    /*--- Jacobian contribution: dM terms ---*/
    for (int iVar = 0; iVar < nVar; iVar++) {
      for (int jVar = 0; jVar < nVar; jVar++) {
        JJ(iVar, jVar) += dM_RM[jVar] * Fc_LR[iVar] * A_LR;
      }
    }

    /*--- Jacobian contribution: dP terms ---*/
    for (int iDim = 0; iDim < nDim; iDim++) {
      for (int iVar = 0; iVar < nVar; iVar++) {
        JJ(nSpecies + iDim, iVar) += dP_RM[iVar] * UnitNormal[iDim];
      }
    }
  }

  /*--- Integrate over dual-face area ---*/
  for (int iVar = 0; iVar < nVar; iVar++) {
    for (int jVar = 0; jVar < nVar; jVar++) {
      JI(iVar, jVar) *= Area;
      JJ(iVar, jVar) *= Area;
    }
  }
#undef JI
#undef JJ
}

/*! \brief Kernel 1: raw CheckNonPhys of both reconstructed states (MUSCL only). */
template <int ND>
void KernelReconstructionCheck(const UpwArgs& a, unsigned long nEdge) {
  Kokkos::parallel_for(
      "NEMO::UpwindCheckNonPhys", Kokkos::RangePolicy<exec_space>(0, nEdge), KOKKOS_LAMBDA(const unsigned long e) {
        const auto iPoint = a.edge_nodes(e, 0), jPoint = a.edge_nodes(e, 1);
        double Primitive_i[MAXNP], Primitive_j[MAXNP];
        Reconstruct<ND>(a, iPoint, jPoint, Primitive_i, Primitive_j);
        a.eflag(e, 0) = CheckNonPhys(a.ns, ND, Primitive_i) ? 1 : 0;
        a.eflag(e, 1) = CheckNonPhys(a.ns, ND, Primitive_j) ? 1 : 0;
      });
}

/*!
 * \brief Kernel 2: CVariable::SetNon_Physical then GetNon_Physical, per point over its (edge, side) list in increasing
 *        edge order. Overwrites eflag(e, side) with the value the CPU reads back for that edge.
 */
void KernelStickyNonPhys(const CNEMOKokkosData::Geometry& geo, const View2D<unsigned char>& eflag,
                         const View1D<unsigned char>& np, const View1D<unsigned short>& cnt) {
  const auto ptr = geo.pside_ptr, idx = geo.pside_idx;
  Kokkos::parallel_for(
      "NEMO::UpwindStickyNonPhys", Kokkos::RangePolicy<exec_space>(0, geo.nPoint), KOKKOS_LAMBDA(const unsigned long p) {
        unsigned char NP = np(p);
        unsigned short C = cnt(p);
        for (auto k = ptr(p); k < ptr(p + 1); ++k) {
          const auto key = idx(k);
          const auto e = key >> 1;
          const int side = int(key & 1ul);
          if (eflag(e, side)) {
            NP = 1;
            C = 0;
          } else {
            C++;
            if (C > 20) NP = 0;
          }
          eflag(e, side) = NP;
        }
        np(p) = NP;
        cnt(p) = C;
      });
}

/*! \brief Kernel 3: flux + Jacobian per edge, CheckResidualNaNs, atomic accumulation. */
template <int ND>
void KernelUpwindFlux(const UpwArgs& a, unsigned long nEdge, long lastKey) {
  Kokkos::parallel_for(
      "NEMO::UpwindAUSM", Kokkos::RangePolicy<exec_space>(0, nEdge), KOKKOS_LAMBDA(const unsigned long iEdge) {
        const int nVar = a.nVar, ns = a.ns;
        const auto iPoint = a.edge_nodes(iEdge, 0), jPoint = a.edge_nodes(iEdge, 1);
        const double* Normal = &a.edge_normal(iEdge, 0);

        /*--- Cell values (first order or non-physical reconstruction). ---*/
        const double* V_i = &a.V(iPoint, 0);
        const double* V_j = &a.V(jPoint, 0);
        const double* eve_i = &a.eves(iPoint, 0);
        const double* eve_j = &a.eves(jPoint, 0);
        const double zero[MAXNV] = {0.0};
        const double* dPdU_i = a.implicit ? &a.dPdU(iPoint, 0) : zero;
        const double* dPdU_j = a.implicit ? &a.dPdU(jPoint, 0) : zero;
        double Gamma_i = a.Gamma(iPoint), Gamma_j = a.Gamma(jPoint);

        double Primitive_i[MAXNP], Primitive_j[MAXNP];
        double Eve_i[MAXNS], Eve_j[MAXNS];

        if (a.muscl) {
          Reconstruct<ND>(a, iPoint, jPoint, Primitive_i, Primitive_j);
          const bool chk_err_i = a.eflag(iEdge, 0), chk_err_j = a.eflag(iEdge, 1);

          /*--- ComputeConsistentExtrapolation: Eve at Tve and Gamma of the reconstructed state only
           *    (dPdU/dTdU/dTvedU/Cvve are not computed: the zero-initialised CPU arrays are used). ---*/
          if (!chk_err_i) {
            ComputeSpeciesEve(a.gas, Primitive_i[ns + 1], Eve_i);
            Gamma_i = ComputeGamma(a.gas, Primitive_i);
            V_i = Primitive_i;
            eve_i = Eve_i;
            dPdU_i = zero;
            if (long(2 * iEdge) == lastKey)
              for (int v = 0; v < a.nPrimVar; ++v) a.lastRec(v) = Primitive_i[v];
          }
          if (!chk_err_j) {
            ComputeSpeciesEve(a.gas, Primitive_j[ns + 1], Eve_j);
            Gamma_j = ComputeGamma(a.gas, Primitive_j);
            V_j = Primitive_j;
            eve_j = Eve_j;
            dPdU_j = zero;
            if (long(2 * iEdge + 1) == lastKey)
              for (int v = 0; v < a.nPrimVar; ++v) a.lastRec(v) = Primitive_j[v];
          }
        }
        const double Sensor_i = a.Sensor.size() ? a.Sensor(iPoint) : 0.0;
        const double Sensor_j = a.Sensor.size() ? a.Sensor(jPoint) : 0.0;

        double Flux[MAXNV];
        double Jacobian_i[MAXNV * MAXNV], Jacobian_j[MAXNV * MAXNV];
        AUSMResidual<ND>(a, Normal, V_i, V_j, eve_i, eve_j, dPdU_i, dPdU_j, Gamma_i, Gamma_j, Sensor_i, Sensor_j, Flux,
                         Jacobian_i, Jacobian_j);

        /*--- CNumerics::CheckResidualNaNs ---*/
        bool err = false;
        for (int iVar = 0; iVar < nVar; iVar++) {
          if (Kokkos::isnan(Flux[iVar])) err = true;
          if (a.implicit) {
            for (int jVar = 0; jVar < nVar; jVar++) {
              if (Kokkos::isnan(Jacobian_i[iVar * nVar + jVar])) err = true;
              if (Kokkos::isnan(Jacobian_j[iVar * nVar + jVar])) err = true;
            }
          }
        }
        if (err) return;

        /*--- LinSysRes.AddBlock(iPoint), SubtractBlock(jPoint). ---*/
        for (int iVar = 0; iVar < nVar; iVar++) Kokkos::atomic_add(&a.res(iPoint, iVar), Flux[iVar]);
        for (int iVar = 0; iVar < nVar; iVar++) Kokkos::atomic_sub(&a.res(jPoint, iVar), Flux[iVar]);

        if (!a.implicit) return;

        /*--- Jacobian.UpdateBlocks(iEdge, iPoint, jPoint, Ji, Jj). ---*/
        const unsigned long bs = static_cast<unsigned long>(nVar) * nVar;
        double* bii = &a.jac(a.dia_ptr(iPoint) * bs);
        double* bjj = &a.jac(a.dia_ptr(jPoint) * bs);
        double* bij = &a.jac(a.edge_nnz(iEdge, 0) * bs);
        double* bji = &a.jac(a.edge_nnz(iEdge, 1) * bs);
        for (unsigned long k = 0; k < bs; ++k) {
          Kokkos::atomic_add(&bii[k], Jacobian_i[k]);
          Kokkos::atomic_add(&bij[k], Jacobian_j[k]);
          Kokkos::atomic_sub(&bji[k], Jacobian_i[k]);
          Kokkos::atomic_sub(&bjj[k], Jacobian_j[k]);
        }
      });
}

}  // namespace

void CNEMOKokkosData::UploadUpwindStatic() {
  if (geo.pside_ptr.size() != 0) return;
  const auto np = geo.nPoint, ne = geo.nEdge;
  auto hn = Kokkos::create_mirror_view_and_copy(HostSpace(), geo.edge_nodes);
  std::vector<unsigned long> count(np + 1, 0), list(2 * ne);
  for (auto e = 0ul; e < ne; ++e) {
    count[hn(e, 0) + 1]++;
    count[hn(e, 1) + 1]++;
  }
  for (auto i = 0ul; i < np; ++i) count[i + 1] += count[i];
  std::vector<unsigned long> fill(count.begin(), count.end() - 1);
  /*--- Increasing edge order per point by construction (edges visited in order). ---*/
  for (auto e = 0ul; e < ne; ++e) {
    list[fill[hn(e, 0)]++] = (e << 1);
    list[fill[hn(e, 1)]++] = (e << 1) | 1ul;
  }
  geo.pside_ptr = View1D<unsigned long>("NEMO::pside_ptr", np + 1);
  geo.pside_idx = View1D<unsigned long>("NEMO::pside_idx", 2 * ne);
  Kokkos::deep_copy(geo.pside_ptr, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(count.data(), np + 1));
  Kokkos::deep_copy(geo.pside_idx, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(list.data(), 2 * ne));
}

CNEMOKokkosData::UpwindResult CNEMOKokkosData::ComputeUpwindResidual(const UpwindParams& p, bool uploadAll,
                                                                     double* hostRes, double* hostJac) {
  auto& P = stagePerf[ST_UPWIND];
  Kokkos::Timer timer;
  UpwindResult out;

  if (p.implicit && (!geo.has_pattern || geo.nnz == 0))
    SU2_MPI::Error("CNEMOKokkosData: implicit upwind needs the FV sparse pattern.", CURRENT_FUNCTION);
  UploadUpwindStatic();
  if (st.Res.size() == 0) {
    st.Res = View2D<double>("NEMO::UpwindRes", nPoint, nVar);
    st.NonPhys = View1D<unsigned char>("NEMO::Non_Physical", nPoint);
    st.NonPhysCount = View1D<unsigned short>("NEMO::Non_Physical_Counter", nPoint);
    st.EdgeNonPhys = View2D<unsigned char>("NEMO::EdgeNonPhys", geo.nEdge, 2);
    st.LastRec = View1D<double>("NEMO::LastRec", nPrimVar);
  }
  if (p.implicit && st.JacVal.size() == 0)
    st.JacVal = View1D<double>("NEMO::UpwindJacobian", geo.nnz * size_t(nVar) * nVar);

  /*--- Inputs: Primitive + secondaries (normally resident from SetPrimVar). ---*/
  if (uploadAll || !primDeviceValid || !secDeviceValid) {
    UploadPrimitive();
    P.bytes_h2d += double(nPoint) * (nPrimVar + nSpecies * 2 + 1 + (implicit ? 3 * nVar : 0)) * sizeof(double);
    primDeviceValid = secDeviceValid = true;
  }
  /*--- Gradient / limiter: resident from the device stages; domain rows only, so halos come from the host. ---*/
  const size_t gRow = size_t(nPrimVarGrad) * nDim;
  auto uploadRange = [&](double* dptr, const double* hptr, size_t first, size_t count) {
    if (count == 0) return;
    Kokkos::deep_copy(Kokkos::View<double*, mem_space, Unmanaged>(dptr + first, count),
                      Kokkos::View<const double*, HostSpace, Unmanaged>(hptr + first, count));
    P.bytes_h2d += double(count) * sizeof(double);
  };
  if (p.muscl) {
    AllocateGradients();
    const double* hg = nodes->GetGradient_Reconstruction().data();
    if (uploadAll || !gradRecValid) {
      uploadRange(st.GradRec.data(), hg, 0, nPoint * gRow);
      gradRecValid = true;
      if (!separateRecGradient) gradPrimValid = true;
    } else {
      uploadRange(st.GradRec.data(), hg, geo.nPointDomain * gRow, (nPoint - geo.nPointDomain) * gRow);
    }
    if (p.limiter && !p.vanAlbada) {
      if (st.Limiter.size() == 0) {
        st.Limiter = View2D<double>("NEMO::Limiter_Primitive", nPoint, nPrimVarGrad);
        st.PrimMax = View2D<double>("NEMO::Solution_Max", nPoint, nPrimVarGrad);
        st.PrimMin = View2D<double>("NEMO::Solution_Min", nPoint, nPrimVarGrad);
      }
      const double* hl = nodes->GetLimiter_Primitive().data();
      if (uploadAll || !limValid) {
        uploadRange(st.Limiter.data(), hl, 0, nPoint * size_t(nPrimVarGrad));
        limValid = true;
      } else {
        uploadRange(st.Limiter.data(), hl, geo.nPointDomain * nPrimVarGrad,
                    (nPoint - geo.nPointDomain) * size_t(nPrimVarGrad));
      }
    }
    /*--- Non_Physical / counter: host authoritative (CVariable), tiny. ---*/
    auto hnp = Kokkos::create_mirror_view(st.NonPhys);
    auto hnc = Kokkos::create_mirror_view(st.NonPhysCount);
    for (unsigned long i = 0; i < nPoint; ++i) {
      hnp(i) = nodes->Non_Physical(i) ? 1 : 0;
      hnc(i) = nodes->Non_Physical_Counter(i);
    }
    Kokkos::deep_copy(st.NonPhys, hnp);
    Kokkos::deep_copy(st.NonPhysCount, hnc);
    P.bytes_h2d += double(nPoint) * 3;
  }
  if (p.scheme == UpwindScheme::AUSMPLUSM) {
    if (st.Sensor.size() == 0) st.Sensor = View1D<double>("NEMO::Sensor", nPoint);
    uploadRange(st.Sensor.data(), nodes->GetSensor().data(), 0, nPoint);
  }
  Kokkos::deep_copy(st.Res, 0.0);
  if (p.implicit) Kokkos::deep_copy(st.JacVal, 0.0);
  Kokkos::fence();
  P.h2d += timer.seconds();
  timer.reset();

  UpwArgs a;
  a.edge_nodes = geo.edge_nodes;
  a.edge_normal = geo.edge_normal;
  a.coord = geo.coord;
  a.dia_ptr = geo.dia_ptr;
  a.edge_nnz = geo.edge_nnz;
  a.V = st.Primitive;
  a.dPdU = st.dPdU;
  a.eves = st.eves;
  a.Gamma = st.Gamma;
  a.Sensor = (p.scheme == UpwindScheme::AUSMPLUSM) ? st.Sensor : View1D<double>();
  a.grad = st.GradRec;
  a.lim = st.Limiter;
  a.res = st.Res;
  a.jac = st.JacVal;
  a.eflag = st.EdgeNonPhys;
  a.lastRec = st.LastRec;
  a.gas = gas;
  a.nDim = nDim;
  a.ns = nSpecies;
  a.nVar = nVar;
  a.nPrimVar = nPrimVar;
  a.nPrimVarGrad = nPrimVarGrad;
  a.nElNum = p.nElNumerics;
  a.scheme = static_cast<int>(p.scheme);
  a.implicit = p.implicit;
  a.muscl = p.muscl;
  a.limiter = p.limiter;
  a.vanAlbada = p.vanAlbada;
  a.kappa = p.kappa;
  a.musclRamp = p.musclRamp;
  a.Minf = p.Minf;
  a.eps = EPS;
  a.pi = PI_NUMBER;

  const unsigned long ne = geo.nEdge;
  long lastKey = -1;
  if (p.muscl) {
    if (nDim == 2) KernelReconstructionCheck<2>(a, ne);
    else KernelReconstructionCheck<3>(a, ne);
    KernelStickyNonPhys(geo, st.EdgeNonPhys, st.NonPhys, st.NonPhysCount);

    /*--- counter_local and the last (edge, side) with a consistent extrapolation (CPU: i before j). ---*/
    const auto ef = st.EdgeNonPhys;
    unsigned long count = 0;
    Kokkos::parallel_reduce(
        "NEMO::UpwindNonPhysCount", Kokkos::RangePolicy<exec_space>(0, ne),
        KOKKOS_LAMBDA(const unsigned long e, unsigned long& c, long& last) {
          c += ef(e, 0) + ef(e, 1);
          if (!ef(e, 1)) last = kmaxl(last, long(2 * e + 1));
          else if (!ef(e, 0)) last = kmaxl(last, long(2 * e));
        },
        Kokkos::Sum<unsigned long>(count), Kokkos::Max<long>(lastKey));
    if (lastKey < 0) lastKey = -1;
    out.nonPhysical = count;
  }
  if (nDim == 2) KernelUpwindFlux<2>(a, ne, lastKey);
  else KernelUpwindFlux<3>(a, ne, lastKey);
  Kokkos::fence();
  P.kernel += timer.seconds();
  timer.reset();

  /*--- DtoH + host accumulation (host LinSysRes / Jacobian are zero here in the CPU stage order, so the sums equal
   *    the device sums bitwise; the "+=" keeps it correct if a stage ever runs before the upwind residual). ---*/
  {
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), st.Res);
    const size_t n = size_t(nPoint) * nVar;
    for (size_t k = 0; k < n; ++k) hostRes[k] += h.data()[k];
    P.bytes_d2h += double(n) * sizeof(double);
  }
  if (p.implicit) {
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), st.JacVal);
    const size_t n = h.size();
    for (size_t k = 0; k < n; ++k) hostJac[k] += h.data()[k];
    P.bytes_d2h += double(n) * sizeof(double);
  }
  if (p.muscl) {
    auto hnp = Kokkos::create_mirror_view_and_copy(HostSpace(), st.NonPhys);
    auto hnc = Kokkos::create_mirror_view_and_copy(HostSpace(), st.NonPhysCount);
    for (unsigned long i = 0; i < nPoint; ++i) {
      nodes->Non_Physical(i) = (hnp(i) != 0);
      nodes->Non_Physical_Counter(i) = hnc(i);
    }
    P.bytes_d2h += double(nPoint) * 3;
    if (lastKey >= 0) {
      auto hr = Kokkos::create_mirror_view_and_copy(HostSpace(), st.LastRec);
      out.haveLastRec = true;
      out.lastRec.assign(hr.data(), hr.data() + nPrimVar);
    }
  }
  P.d2h += timer.seconds();
  P.calls++;
  return out;
}

void CNEMOKokkosData::GetNonPhysical(std::vector<double>& flag, std::vector<double>& count) const {
  flag.resize(nPoint);
  count.resize(nPoint);
  for (unsigned long i = 0; i < nPoint; ++i) {
    flag[i] = nodes->Non_Physical(i) ? 1.0 : 0.0;
    count[i] = nodes->Non_Physical_Counter(i);
  }
}

void CNEMOKokkosData::SetNonPhysical(const std::vector<double>& flag, const std::vector<double>& count) {
  for (unsigned long i = 0; i < nPoint; ++i) {
    nodes->Non_Physical(i) = (flag[i] != 0.0);
    nodes->Non_Physical_Counter(i) = static_cast<unsigned short>(count[i]);
  }
}

bool CNEMOKokkosData::HostForced(const char* stage) {
  static const std::string env = [] {
    const char* e = std::getenv("SU2_KOKKOS_NEMO_HOST");
    return std::string(e ? e : "");
  }();
  return env.find("all") != std::string::npos || env.find(stage) != std::string::npos;
}

#endif  // HAVE_KOKKOS
