/*!
 * \file CNEMOKokkosAUSM.hpp
 * \brief Internal header of the device NEMO residual kernels: shared argument struct, MUSCL reconstruction,
 *        CheckNonPhys and the AUSM / AUSM+M flux + approximate Jacobian (CUpwAUSM_SLAU_Base_NEMO), used by the
 *        edge kernels (CNEMOKokkosUpwind.cpp, Step 5) and the boundary-vertex kernels (CNEMOKokkosResidual.cpp,
 *        Step 7). Moved verbatim from CNEMOKokkosUpwind.cpp. Include only from those translation units.
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#pragma once

#ifdef HAVE_KOKKOS

#include "../../../include/solvers/kokkos/CNEMOKokkosData.hpp"

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
  View2D<double> U; /*!< Solution (Roe only) */
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


}  // namespace

#endif  // HAVE_KOKKOS
