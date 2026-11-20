/*!
 * \file CNEMOKokkosViscous.hpp
 * \brief Internal header of the device NEMO viscous kernels (Step 8): CAvgGrad_NEMO / CAvgGradCorrected_NEMO
 *        ComputeResidual with CNEMONumerics::GetViscousProjFlux and GetViscousProjJacs (ComputeViscousJacs_impl) as
 *        POD device functions over MAXNVAR stack arrays, statement by statement as the CPU code. Used by the viscous
 *        edge kernel and the wall / far-field boundary kernels. Include only from the kokkos solver translation units.
 *
 * CPU semantics reproduced on purpose (pre-existing, not fixed):
 *  - CAvgGrad_NEMO passes the SQUARED distance as dist_ij to the Jacobian (CAvgGradCorrected_NEMO the distance).
 *  - CAvgGradCorrected_NEMO returns (Flux, Jacobian_j, Jacobian_j): the edge update uses Jacobian_j for both blocks.
 *  - The species diffusion terms use CNEMONumerics::nHeavy (config IONIZATION), the Jacobian uses the laminar
 *    viscosity only (no eddy viscosity), and the mean "primitive" holds mean mass fractions at the species indices.
 *
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#pragma once

#ifdef HAVE_KOKKOS

#include "../../../include/solvers/kokkos/CNEMOKokkosData.hpp"

namespace nemo_visc {

using namespace su2kokkos::nemo;
constexpr int MAXV = MAXNVAR;
constexpr int MAXP = MAXNPRIMVAR;

/*! \brief Inputs of one viscous flux evaluation (pointers into device views or stack buffers). */
struct ViscPoint {
  const double* V;      /*!< primitive (rho_s layout) */
  const double* G;      /*!< gradient rows (nPrimVarGrad x ND), row-major */
  const double* eve;
  const double* cvve;
  const double* dTdU;   /*!< nullptr if explicit */
  const double* dTvedU;
  const double* Ds;
  double mu, eddy, k, kve;
};

struct ViscConst {
  NEMOGasTables gas;
  int ns, nVar, nPrimVar, nPrimVarGrad, nHeavy;
  bool implicit;
  double Ru; /*!< 1000*UNIVERSAL_GAS_CONSTANT (ComputeViscousJacs_impl) */
};

/*! \brief CNEMONumerics::GetViscousProjFlux (val_primvar holds Y_s at the species indices). */
template <int ND>
KOKKOS_INLINE_FUNCTION void ViscousProjFlux(const ViscConst& c, const double* val_primvar,
                                            const double (*val_gradprimvar)[ND], const double* val_eve,
                                            const double* val_normal, const double* val_diffusioncoeff,
                                            double val_lam_viscosity, double val_eddy_viscosity,
                                            double val_therm_conductivity, double val_therm_conductivity_ve,
                                            double* Proj_Flux_Tensor) {
  const int nSpecies = c.ns, nVar = c.nVar, nDim = ND, nHeavy = c.nHeavy;
  const NEMOPrimIndices idx(nSpecies, ND);
  const int RHOS_INDEX = 0, T_INDEX = idx.T(), TVE_INDEX = idx.TVE(), VEL_INDEX = idx.VEL(), RHO_INDEX = idx.RHO();

  double Flux_Tensor[MAXV][ND];
  for (int iVar = 0; iVar < nVar; iVar++) {
    Proj_Flux_Tensor[iVar] = 0.0;
    for (int iDim = 0; iDim < nDim; iDim++) Flux_Tensor[iVar][iDim] = 0.0;
  }

  const double* Ds = val_diffusioncoeff;
  const double mu = val_lam_viscosity + val_eddy_viscosity;
  const double ktr = val_therm_conductivity;
  const double kve = val_therm_conductivity_ve;
  const double rho = val_primvar[RHO_INDEX];
  const double T = val_primvar[T_INDEX];
  const double Tve = val_primvar[TVE_INDEX];
  const double* V = val_primvar;
  const auto GV = val_gradprimvar;
  double hs[MAXNS];
  ComputeSpeciesEnthalpy(c.gas, T, Tve, val_eve, hs);

  double Vector[ND] = {0.0};
  for (int iDim = 0; iDim < nDim; iDim++) {
    for (int iSpecies = 0; iSpecies < nHeavy; iSpecies++) {
      Vector[iDim] += rho * Ds[iSpecies] * GV[RHOS_INDEX + iSpecies][iDim];
    }
  }

  /*--- CNumerics::ComputeStressTensor(nDim, tau, GV + VEL_INDEX, mu) (density = turb_ke = 0). ---*/
  double tau[ND][ND];
  {
    double divVel = 0.0;
    for (int iDim = 0; iDim < nDim; iDim++) divVel += GV[VEL_INDEX + iDim][iDim];
    const double density = 0.0, turb_ke = 0.0;
    const double pTerm = 2. / 3. * (divVel * mu + density * turb_ke);
    for (int iDim = 0; iDim < nDim; iDim++) {
      for (int jDim = 0; jDim < nDim; jDim++)
        tau[iDim][jDim] = mu * (GV[VEL_INDEX + iDim][jDim] + GV[VEL_INDEX + jDim][iDim]);
      tau[iDim][iDim] -= pTerm;
    }
  }

  for (int iDim = 0; iDim < nDim; iDim++) {
    for (int iSpecies = 0; iSpecies < nHeavy; iSpecies++) {
      Flux_Tensor[iSpecies][iDim] = rho * Ds[iSpecies] * GV[RHOS_INDEX + iSpecies][iDim] - V[RHOS_INDEX + iSpecies] * Vector[iDim];
    }

    Flux_Tensor[nSpecies + nDim][iDim] = 0.0;
    for (int jDim = 0; jDim < nDim; jDim++) {
      Flux_Tensor[nSpecies + jDim][iDim] = tau[iDim][jDim];
      Flux_Tensor[nSpecies + nDim][iDim] += tau[iDim][jDim] * val_primvar[VEL_INDEX + jDim];
    }

    for (int iSpecies = 0; iSpecies < nHeavy; iSpecies++) {
      Flux_Tensor[nSpecies + nDim][iDim] += Flux_Tensor[iSpecies][iDim] * hs[iSpecies];
      Flux_Tensor[nSpecies + nDim + 1][iDim] += Flux_Tensor[iSpecies][iDim] * val_eve[iSpecies];
    }

    Flux_Tensor[nSpecies + nDim][iDim] += ktr * GV[T_INDEX][iDim] + kve * GV[TVE_INDEX][iDim];
    Flux_Tensor[nSpecies + nDim + 1][iDim] += kve * GV[TVE_INDEX][iDim];
  }

  for (int iVar = 0; iVar < nVar; iVar++) {
    for (int iDim = 0; iDim < nDim; iDim++) {
      Proj_Flux_Tensor[iVar] += Flux_Tensor[iVar][iDim] * val_normal[iDim];
    }
  }
}

/*!
 * \brief CNEMONumerics::ComputeViscousJacs_impl (static or dynamic sizes give the same per-entry arithmetic).
 * \param[in,out] Jac_i, Jac_j - nVar x nVar row-major, accumulated into ("+=", zeroed by the caller).
 */
template <int ND>
KOKKOS_INLINE_FUNCTION void ViscousProjJacs(const ViscConst& c, const double* V_i, const double* V_j,
                                            const double* dTdU_i, const double* dTdU_j, const double* dTvedU_i,
                                            const double* dTvedU_j, const double* val_Mean_PrimVar,
                                            const double* val_Mean_Eve, const double* val_Mean_Cvve,
                                            const double* val_diffusion_coeff, double val_laminar_viscosity,
                                            double val_thermal_conductivity, double val_thermal_conductivity_ve,
                                            double val_dist_ij, const double* val_normal, double val_dS,
                                            const double* val_Fv, double* Jac_i, double* Jac_j) {
  const int nSpecies = c.ns, nVar = c.nVar, nDim = ND;
  const NEMOPrimIndices idx(nSpecies, ND);
  const int RHOS_INDEX = 0, T_INDEX = idx.T(), TVE_INDEX = idx.TVE(), VEL_INDEX = idx.VEL(), RHO_INDEX = idx.RHO();

  double dFdVi[MAXV * MAXV], dFdVj[MAXV * MAXV], dVdUi[MAXV * MAXV], dVdUj[MAXV * MAXV];
  double dJdr_i[MAXNS * MAXNS], dJdr_j[MAXNS * MAXNS];
  double Ys_i[MAXNS], Ys_j[MAXNS];
#define FVI(r, cc) dFdVi[(r) * nVar + (cc)]
#define FVJ(r, cc) dFdVj[(r) * nVar + (cc)]
#define VUI(r, cc) dVdUi[(r) * nVar + (cc)]
#define VUJ(r, cc) dVdUj[(r) * nVar + (cc)]
#define JRI(r, cc) dJdr_i[(r) * nSpecies + (cc)]
#define JRJ(r, cc) dJdr_j[(r) * nSpecies + (cc)]
  for (int k = 0; k < nVar * nVar; k++) dFdVi[k] = dFdVj[k] = dVdUi[k] = dVdUj[k] = 0.0;
  for (int k = 0; k < nSpecies * nSpecies; k++) dJdr_i[k] = dJdr_j[k] = 0.0;

  const double dij = val_dist_ij;
  double theta = 0.0;
  for (int iDim = 0; iDim < nDim; ++iDim) theta += val_normal[iDim] * val_normal[iDim];

  const double rho_i = V_i[RHO_INDEX];
  const double rho_j = V_j[RHO_INDEX];
  const double T = val_Mean_PrimVar[T_INDEX];
  const double Tve = val_Mean_PrimVar[TVE_INDEX];
  const double mu = val_laminar_viscosity;
  const double ktr = val_thermal_conductivity;
  const double kve = val_thermal_conductivity_ve;
  const double Ru = c.Ru;
  const double* Ds = val_diffusion_coeff;
  double hs[MAXNS];
  ComputeSpeciesEnthalpy(c.gas, T, Tve, val_Mean_Eve, hs);

  double sumY = 0.0;
  double sumY_i = 0.0;
  double sumY_j = 0.0;
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    Ys_i[iSpecies] = V_i[RHOS_INDEX + iSpecies] / V_i[RHO_INDEX];
    Ys_j[iSpecies] = V_j[RHOS_INDEX + iSpecies] / V_j[RHO_INDEX];

    sumY_i += Ds[iSpecies] * theta / dij * Ys_i[iSpecies];
    sumY_j += Ds[iSpecies] * theta / dij * Ys_j[iSpecies];
    sumY += Ds[iSpecies] * theta / dij * (Ys_j[iSpecies] - Ys_i[iSpecies]);
  }

  double vel[ND];
  for (int iDim = 0; iDim < nDim; iDim++) vel[iDim] = val_Mean_PrimVar[VEL_INDEX + iDim];

  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++) {
      // first term
      JRJ(iSpecies, jSpecies) += 0.5 * (Ds[iSpecies] * theta / dij * (Ys_j[iSpecies] * rho_i / rho_j + Ys_i[iSpecies]));
      JRI(iSpecies, jSpecies) += -0.5 * (Ds[iSpecies] * theta / dij * (Ys_j[iSpecies] + Ys_i[iSpecies] * rho_j / rho_i));

      // second term
      JRJ(iSpecies, jSpecies) +=
          0.25 * (Ys_i[iSpecies] - rho_i / rho_j * Ys_j[iSpecies]) * sumY +
          0.25 * (Ys_i[iSpecies] + Ys_j[iSpecies]) * (rho_i + rho_j) * Ds[jSpecies] * theta / (dij * rho_j) -
          0.25 * (Ys_i[iSpecies] + Ys_j[iSpecies]) * (rho_i + rho_j) * sumY_j / rho_j;

      JRI(iSpecies, jSpecies) +=
          0.25 * (-rho_j / rho_i * Ys_i[iSpecies] + Ys_j[iSpecies]) * sumY -
          0.25 * (Ys_i[iSpecies] + Ys_j[iSpecies]) * (rho_i + rho_j) * Ds[jSpecies] * theta / (dij * rho_i) +
          0.25 * (Ys_i[iSpecies] + Ys_j[iSpecies]) * (rho_i + rho_j) * sumY_i / rho_i;
    }

    // first term
    JRJ(iSpecies, iSpecies) += -0.5 * Ds[iSpecies] * theta / dij * (1 + rho_i / rho_j);
    JRI(iSpecies, iSpecies) += 0.5 * Ds[iSpecies] * theta / dij * (1 + rho_j / rho_i);

    // second term
    JRJ(iSpecies, iSpecies) += 0.25 * (1.0 + rho_i / rho_j) * sumY;
    JRI(iSpecies, iSpecies) += 0.25 * (1.0 + rho_j / rho_i) * sumY;
  }

  /*--- Calculate transformation matrix ---*/
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    VUI(iSpecies, iSpecies) = 1.0;
    VUJ(iSpecies, iSpecies) = 1.0;
  }
  for (int iDim = 0; iDim < nDim; iDim++) {
    for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
      VUI(nSpecies + iDim, iSpecies) = -V_i[VEL_INDEX + iDim] / V_i[RHO_INDEX];
      VUJ(nSpecies + iDim, iSpecies) = -V_j[VEL_INDEX + iDim] / V_j[RHO_INDEX];
    }
    VUI(nSpecies + iDim, nSpecies + iDim) = 1.0 / V_i[RHO_INDEX];
    VUJ(nSpecies + iDim, nSpecies + iDim) = 1.0 / V_j[RHO_INDEX];
  }
  for (int iVar = 0; iVar < nVar; iVar++) {
    VUI(nSpecies + nDim, iVar) = dTdU_i[iVar];
    VUJ(nSpecies + nDim, iVar) = dTdU_j[iVar];
    VUI(nSpecies + nDim + 1, iVar) = dTvedU_i[iVar];
    VUJ(nSpecies + nDim + 1, iVar) = dTvedU_j[iVar];
  }

  if constexpr (ND == 2) {
    const double thetax = theta + val_normal[0] * val_normal[0] / 3.0;
    const double thetay = theta + val_normal[1] * val_normal[1] / 3.0;
    const double etaz = val_normal[0] * val_normal[1] / 3.0;
    const double pix = mu / dij * (thetax * vel[0] + etaz * vel[1]);
    const double piy = mu / dij * (etaz * vel[0] + thetay * vel[1]);

    FVJ(nSpecies, nSpecies) = mu * thetax / dij * val_dS;
    FVJ(nSpecies, nSpecies + 1) = mu * etaz / dij * val_dS;

    FVJ(nSpecies + 1, nSpecies) = mu * etaz / dij * val_dS;
    FVJ(nSpecies + 1, nSpecies + 1) = mu * thetay / dij * val_dS;

    FVJ(nSpecies + 2, nSpecies) = pix * val_dS;
    FVJ(nSpecies + 2, nSpecies + 1) = piy * val_dS;
    FVJ(nSpecies + 2, nSpecies + 2) = ktr * theta / dij * val_dS;
    FVJ(nSpecies + 2, nSpecies + 3) = kve * theta / dij * val_dS;

    FVJ(nSpecies + 3, nSpecies + 3) = kve * theta / dij * val_dS;

    for (int iVar = 0; iVar < nVar; iVar++)
      for (int jVar = 0; jVar < nVar; jVar++) FVI(iVar, jVar) = -FVJ(iVar, jVar);

    FVI(nSpecies + 2, nSpecies) += 0.5 * val_Fv[nSpecies];
    FVJ(nSpecies + 2, nSpecies) += 0.5 * val_Fv[nSpecies];
    FVI(nSpecies + 2, nSpecies + 1) += 0.5 * val_Fv[nSpecies + 1];
    FVJ(nSpecies + 2, nSpecies + 1) += 0.5 * val_Fv[nSpecies + 1];
    for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
      FVI(nSpecies + 2, nSpecies + 2) += 0.5 * val_Fv[iSpecies] * (Ru / c.gas.MolarMass(iSpecies) + c.gas.Cvtrs(iSpecies));
      FVJ(nSpecies + 2, nSpecies + 2) += 0.5 * val_Fv[iSpecies] * (Ru / c.gas.MolarMass(iSpecies) + c.gas.Cvtrs(iSpecies));
      FVI(nSpecies + 2, nSpecies + 3) += 0.5 * val_Fv[iSpecies] * val_Mean_Cvve[iSpecies];
      FVJ(nSpecies + 2, nSpecies + 3) += 0.5 * val_Fv[iSpecies] * val_Mean_Cvve[iSpecies];
      FVI(nSpecies + 3, nSpecies + 3) += 0.5 * val_Fv[iSpecies] * val_Mean_Cvve[iSpecies];
      FVJ(nSpecies + 3, nSpecies + 3) += 0.5 * val_Fv[iSpecies] * val_Mean_Cvve[iSpecies];
    }

    for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
      for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++) {
        FVJ(iSpecies, jSpecies) += -JRJ(iSpecies, jSpecies) * val_dS;
        FVI(iSpecies, jSpecies) += -JRI(iSpecies, jSpecies) * val_dS;
        FVJ(nSpecies + 2, iSpecies) += -JRJ(jSpecies, iSpecies) * hs[jSpecies] * val_dS;
        FVI(nSpecies + 2, iSpecies) += -JRI(jSpecies, iSpecies) * hs[jSpecies] * val_dS;
        FVJ(nSpecies + 3, iSpecies) += -JRJ(jSpecies, iSpecies) * val_Mean_Eve[jSpecies] * val_dS;
        FVI(nSpecies + 3, iSpecies) += -JRI(jSpecies, iSpecies) * val_Mean_Eve[jSpecies] * val_dS;
      }
    }
  } else {
    const double thetax = theta + val_normal[0] * val_normal[0] / 3.0;
    const double thetay = theta + val_normal[1] * val_normal[1] / 3.0;
    const double thetaz = theta + val_normal[2] * val_normal[2] / 3.0;
    const double etax = val_normal[1] * val_normal[2] / 3.0;
    const double etay = val_normal[0] * val_normal[2] / 3.0;
    const double etaz = val_normal[0] * val_normal[1] / 3.0;
    const double pix = mu / dij * (thetax * vel[0] + etaz * vel[1] + etay * vel[2]);
    const double piy = mu / dij * (etaz * vel[0] + thetay * vel[1] + etax * vel[2]);
    const double piz = mu / dij * (etay * vel[0] + etax * vel[1] + thetaz * vel[2]);

    FVJ(nSpecies, nSpecies) = mu * thetax / dij * val_dS;
    FVJ(nSpecies, nSpecies + 1) = mu * etaz / dij * val_dS;
    FVJ(nSpecies, nSpecies + 2) = mu * etay / dij * val_dS;

    FVJ(nSpecies + 1, nSpecies) = mu * etaz / dij * val_dS;
    FVJ(nSpecies + 1, nSpecies + 1) = mu * thetay / dij * val_dS;
    FVJ(nSpecies + 1, nSpecies + 2) = mu * etax / dij * val_dS;

    FVJ(nSpecies + 2, nSpecies) = mu * etay / dij * val_dS;
    FVJ(nSpecies + 2, nSpecies + 1) = mu * etax / dij * val_dS;
    FVJ(nSpecies + 2, nSpecies + 2) = mu * thetaz / dij * val_dS;

    FVJ(nSpecies + 3, nSpecies) = pix * val_dS;
    FVJ(nSpecies + 3, nSpecies + 1) = piy * val_dS;
    FVJ(nSpecies + 3, nSpecies + 2) = piz * val_dS;
    FVJ(nSpecies + 3, nSpecies + 3) = ktr * theta / dij * val_dS;
    FVJ(nSpecies + 3, nSpecies + 4) = kve * theta / dij * val_dS;

    FVJ(nSpecies + 4, nSpecies + 4) = kve * theta / dij * val_dS;

    for (int iVar = 0; iVar < nVar; iVar++)
      for (int jVar = 0; jVar < nVar; jVar++) FVI(iVar, jVar) = -FVJ(iVar, jVar);

    for (int iDim = 0; iDim < nDim; iDim++) {
      FVI(nSpecies + 3, nSpecies + iDim) += 0.5 * val_Fv[nSpecies + iDim];
      FVJ(nSpecies + 3, nSpecies + iDim) += 0.5 * val_Fv[nSpecies + iDim];
    }
    for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
      FVI(nSpecies + 3, nSpecies + 3) += 0.5 * val_Fv[iSpecies] * (Ru / c.gas.MolarMass(iSpecies) + c.gas.Cvtrs(iSpecies));
      FVJ(nSpecies + 3, nSpecies + 3) += 0.5 * val_Fv[iSpecies] * (Ru / c.gas.MolarMass(iSpecies) + c.gas.Cvtrs(iSpecies));
      FVI(nSpecies + 3, nSpecies + 4) += 0.5 * val_Fv[iSpecies] * val_Mean_Cvve[iSpecies];
      FVJ(nSpecies + 3, nSpecies + 4) += 0.5 * val_Fv[iSpecies] * val_Mean_Cvve[iSpecies];
      FVI(nSpecies + 4, nSpecies + 4) += 0.5 * val_Fv[iSpecies] * val_Mean_Cvve[iSpecies];
      FVJ(nSpecies + 4, nSpecies + 4) += 0.5 * val_Fv[iSpecies] * val_Mean_Cvve[iSpecies];
    }

    for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
      for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++) {
        FVJ(iSpecies, jSpecies) += -JRJ(iSpecies, jSpecies) * val_dS;
        FVI(iSpecies, jSpecies) += -JRI(iSpecies, jSpecies) * val_dS;
        FVJ(nSpecies + 3, iSpecies) += -JRJ(jSpecies, iSpecies) * hs[jSpecies] * val_dS;
        FVI(nSpecies + 3, iSpecies) += -JRI(jSpecies, iSpecies) * hs[jSpecies] * val_dS;
        FVJ(nSpecies + 4, iSpecies) += -JRJ(jSpecies, iSpecies) * val_Mean_Eve[jSpecies] * val_dS;
        FVI(nSpecies + 4, iSpecies) += -JRI(jSpecies, iSpecies) * val_Mean_Eve[jSpecies] * val_dS;
      }
    }
  }

  /*--- dFv/dUij = dFv/dVij * dVij/dUij ---*/
  for (int iVar = 0; iVar < nVar; iVar++)
    for (int jVar = 0; jVar < nVar; jVar++)
      for (int kVar = 0; kVar < nVar; kVar++) {
        Jac_i[iVar * nVar + jVar] += FVI(iVar, kVar) * VUI(kVar, jVar);
        Jac_j[iVar * nVar + jVar] += FVJ(iVar, kVar) * VUJ(kVar, jVar);
      }
#undef FVI
#undef FVJ
#undef VUI
#undef VUJ
#undef JRI
#undef JRJ
}

/*!
 * \brief CAvgGrad_NEMO::ComputeResidual (corrected = false) / CAvgGradCorrected_NEMO::ComputeResidual (true).
 * \param[out] Jac_i, Jac_j - the residual's jacobian_i / jacobian_j (corrected: both are the CPU Jacobian_j).
 */
template <int ND>
KOKKOS_INLINE_FUNCTION void AvgGradResidual(const ViscConst& c, bool corrected, const double* Coord_i,
                                            const double* Coord_j, const double* Normal, const ViscPoint& pi,
                                            const ViscPoint& pj, double* Flux, double* Jac_i, double* Jac_j) {
  const int nSpecies = c.ns, nVar = c.nVar, nDim = ND, nPrimVar = c.nPrimVar, nPrimVarGrad = c.nPrimVarGrad;
  const NEMOPrimIndices idx(nSpecies, ND);
  const int RHO_INDEX = idx.RHO();
  const double* V_i = pi.V;
  const double* V_j = pj.V;
  auto Gi = [&](int v, int d) { return pi.G[v * ND + d]; };
  auto Gj = [&](int v, int d) { return pj.G[v * ND + d]; };

  double sq = 0.0;
  for (int iDim = 0; iDim < nDim; ++iDim) sq += Normal[iDim] * Normal[iDim];
  const double Area = Kokkos::sqrt(sq);
  double UnitNormal[ND];
  for (int iDim = 0; iDim < nDim; iDim++) UnitNormal[iDim] = Normal[iDim] / Area;

  double Edge_Vector[ND];
  double dist_ij_2 = 0.0;
  if (corrected) {
    for (int iDim = 0; iDim < nDim; iDim++) Edge_Vector[iDim] = Coord_j[iDim] - Coord_i[iDim];
    for (int iDim = 0; iDim < nDim; ++iDim) dist_ij_2 += Edge_Vector[iDim] * Edge_Vector[iDim];
  }

  double PrimVar_i[MAXP], PrimVar_j[MAXP], Mean_PrimVar[MAXP];
  double Mean_GradPrimVar[MAXP][ND];
  double Mean_Eve[MAXNS], Mean_Cvve[MAXNS], Mean_Diffusion_Coeff[MAXNS];

  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++)
    Mean_Diffusion_Coeff[iSpecies] = 0.5 * (pi.Ds[iSpecies] + pj.Ds[iSpecies]);
  const double Mean_Laminar_Viscosity = 0.5 * (pi.mu + pj.mu);
  const double Mean_Eddy_Viscosity = 0.5 * (pi.eddy + pj.eddy);
  const double Mean_Thermal_Conductivity = 0.5 * (pi.k + pj.k);
  const double Mean_Thermal_Conductivity_ve = 0.5 * (pi.kve + pj.kve);

  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    PrimVar_i[iSpecies] = V_i[iSpecies] / V_i[RHO_INDEX];
    PrimVar_j[iSpecies] = V_j[iSpecies] / V_j[RHO_INDEX];
    Mean_PrimVar[iSpecies] = 0.5 * (PrimVar_i[iSpecies] + PrimVar_j[iSpecies]);
    for (int iDim = 0; iDim < nDim; iDim++) {
      Mean_GradPrimVar[iSpecies][iDim] =
          0.5 * (1.0 / V_i[RHO_INDEX] * (Gi(iSpecies, iDim) - PrimVar_i[iSpecies] * Gi(RHO_INDEX, iDim)) +
                 1.0 / V_j[RHO_INDEX] * (Gj(iSpecies, iDim) - PrimVar_j[iSpecies] * Gj(RHO_INDEX, iDim)));
    }
  }
  for (int iVar = nSpecies; iVar < nPrimVar; iVar++) {
    PrimVar_i[iVar] = V_i[iVar];
    PrimVar_j[iVar] = V_j[iVar];
    Mean_PrimVar[iVar] = 0.5 * (PrimVar_i[iVar] + PrimVar_j[iVar]);
  }
  for (int iVar = nSpecies; iVar < nPrimVarGrad; iVar++) {
    for (int iDim = 0; iDim < nDim; iDim++) {
      Mean_GradPrimVar[iVar][iDim] = 0.5 * (Gi(iVar, iDim) + Gj(iVar, iDim));
    }
  }
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    Mean_Eve[iSpecies] = 0.5 * (pi.eve[iSpecies] + pj.eve[iSpecies]);
    Mean_Cvve[iSpecies] = 0.5 * (pi.cvve[iSpecies] + pj.cvve[iSpecies]);
  }

  if (corrected) {
    for (int iVar = 0; iVar < nPrimVarGrad; iVar++) {
      double Proj_Mean_GradPrimVar_Edge = 0.0;
      for (int iDim = 0; iDim < nDim; ++iDim) Proj_Mean_GradPrimVar_Edge += Mean_GradPrimVar[iVar][iDim] * Edge_Vector[iDim];
      for (int iDim = 0; iDim < nDim; iDim++) {
        Mean_GradPrimVar[iVar][iDim] -=
            (Proj_Mean_GradPrimVar_Edge - (PrimVar_j[iVar] - PrimVar_i[iVar])) * Edge_Vector[iDim] / dist_ij_2;
      }
    }
  }

  double Proj_Flux_Tensor[MAXV];
  ViscousProjFlux<ND>(c, Mean_PrimVar, Mean_GradPrimVar, Mean_Eve, Normal, Mean_Diffusion_Coeff,
                      Mean_Laminar_Viscosity, Mean_Eddy_Viscosity, Mean_Thermal_Conductivity,
                      Mean_Thermal_Conductivity_ve, Proj_Flux_Tensor);

  for (int iVar = 0; iVar < nVar; iVar++) Flux[iVar] = Proj_Flux_Tensor[iVar];

  if (!c.implicit) return;

  double dist_ij;
  if (corrected) {
    dist_ij = Kokkos::sqrt(dist_ij_2);
  } else {
    /*--- CAvgGrad_NEMO: GeometryToolbox::Distance(nDim, Coord_j, Coord_i, d), dist_ij = SquaredNorm(d). ---*/
    double d[ND];
    for (int iDim = 0; iDim < nDim; iDim++) d[iDim] = Coord_j[iDim] - Coord_i[iDim];
    dist_ij = 0.0;
    for (int iDim = 0; iDim < nDim; ++iDim) dist_ij += d[iDim] * d[iDim];
  }

  double JiCPU[MAXV * MAXV];
  for (int k = 0; k < nVar * nVar; k++) {
    JiCPU[k] = 0.0;
    Jac_j[k] = 0.0;
  }
  ViscousProjJacs<ND>(c, V_i, V_j, pi.dTdU, pj.dTdU, pi.dTvedU, pj.dTvedU, Mean_PrimVar, Mean_Eve, Mean_Cvve,
                      Mean_Diffusion_Coeff, Mean_Laminar_Viscosity, Mean_Thermal_Conductivity,
                      Mean_Thermal_Conductivity_ve, dist_ij, UnitNormal, Area, Proj_Flux_Tensor, JiCPU, Jac_j);
  if (corrected) {
    for (int k = 0; k < nVar * nVar; k++) Jac_i[k] = Jac_j[k];
  } else {
    for (int k = 0; k < nVar * nVar; k++) Jac_i[k] = JiCPU[k];
  }
}

/*! \brief CNumerics::CheckResidualNaNs with both Jacobians. */
KOKKOS_INLINE_FUNCTION bool ResidualNaN2(bool implicit, int nVar, const double* r, const double* Ji, const double* Jj) {
  bool err = false;
  for (int iVar = 0; iVar < nVar; iVar++) {
    if (Kokkos::isnan(r[iVar])) err = true;
    if (implicit) {
      for (int jVar = 0; jVar < nVar; jVar++) {
        if (Kokkos::isnan(Ji[iVar * nVar + jVar])) err = true;
        if (Kokkos::isnan(Jj[iVar * nVar + jVar])) err = true;
      }
    }
  }
  return err;
}

}  // namespace nemo_visc

#endif  // HAVE_KOKKOS
