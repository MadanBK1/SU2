/*!
 * \file CNEMOKokkosLAX.hpp
 * \brief Internal header (reviewer scaffold for item 4.4-LAX, NOT dispatched yet): device port of
 *        CCentLax_NEMO::ComputeResidual (numerics/NEMO/convection/lax.cpp) incl. the implicit Jacobian, and of
 *        CNEMONumerics::GetInviscidProjFlux / GetInviscidProjJac (numerics/NEMO/CNEMONumerics.cpp).
 *        Statement order and CPU quirks are kept (see the notes below). Include only from the edge-kernel TUs,
 *        after CNEMOKokkosAUSM.hpp (uses MAXNV / MAXNP and the gas helpers).
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * CPU quirks reproduced:
 *  - GetInviscidProjFlux(2D) reads val_V[VEL_INDEX+2] (= P) into an unused w: no effect, not reproduced.
 *  - Jacobian_j last row, momentum columns use dPdU_j[nSpecies+nDim] (not [nSpecies+iDim]).
 *  - The vib.-el. row of GetInviscidProjJac uses "=" (not "+=") for the momentum / eve columns.
 * CPU behaviour NOT reproducible in parallel:
 *  - MeandPdU is a persistent, never-initialised member on the CPU. With IONIZATION, CNEMOGas::ComputedPdU updates
 *    entries with "-=" on the previous edge's values (edge-order dependent). Here it starts at 0 per edge, so only
 *    non-ionized LAX cases can be gated bitwise; keep ionized LAX on the host path (or document the difference).
 */

#pragma once

#ifdef HAVE_KOKKOS

namespace {

/*! \brief CNEMONumerics::GetInviscidProjFlux (val_U = conservative, val_V = primitive, Normal not unit). */
template <int ND>
KOKKOS_INLINE_FUNCTION void LaxInviscidProjFlux(int nSpecies, int nVar, const double* val_U, const double* val_V,
                                                const double* val_normal, double* val_Proj_Flux) {
  const NEMOPrimIndices idx(nSpecies, ND);
  for (int iVar = 0; iVar < nVar; iVar++) val_Proj_Flux[iVar] = 0.0;

  const double rho = val_V[idx.RHO()];
  const double P = val_V[idx.P()];
  const double H = val_V[idx.H()];
  const double rhoEve = val_U[nSpecies + ND + 1];
  const double* rhos = &val_V[idx.RHOS()];
  double vel[ND];
  for (int iDim = 0; iDim < ND; iDim++) vel[iDim] = val_V[idx.VEL() + iDim];

  /*--- Direction d = 0 assigns, d > 0 accumulates (same as the unrolled CPU code). ---*/
  for (int d = 0; d < ND; d++) {
    const double n = val_normal[d];
    double f[MAXNV];
    for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) f[iSpecies] = (rhos[iSpecies] * vel[d]) * n;
    for (int iDim = 0; iDim < ND; iDim++) {
      if (iDim == d)
        f[nSpecies + iDim] = (rho * vel[d] * vel[iDim] + P) * n;
      else
        f[nSpecies + iDim] = (rho * vel[d] * vel[iDim]) * n;
    }
    f[nSpecies + ND] = (rho * vel[d] * H) * n;
    f[nSpecies + ND + 1] = (rhoEve * vel[d]) * n;
    if (d == 0)
      for (int iVar = 0; iVar < nVar; iVar++) val_Proj_Flux[iVar] = f[iVar];
    else
      for (int iVar = 0; iVar < nVar; iVar++) val_Proj_Flux[iVar] += f[iVar];
  }
}

/*! \brief CNEMONumerics::GetInviscidProjJac, row-major nVar x nVar. */
template <int ND>
KOKKOS_INLINE_FUNCTION void LaxInviscidProjJac(int nSpecies, int nVar, const double* val_U, const double* val_V,
                                               const double* val_dPdU, const double* val_normal, double val_scale,
                                               double* J) {
  const NEMOPrimIndices idx(nSpecies, ND);
  const int nDim = ND;
  const double* rhos = &val_V[idx.RHOS()];
#define PJ(r, c) J[(r) * nVar + (c)]
  for (int iVar = 0; iVar < nVar; iVar++)
    for (int jVar = 0; jVar < nVar; jVar++) PJ(iVar, jVar) = 0.0;

  const double rho = val_V[idx.RHO()];
  const double H = val_V[idx.H()];
  const double rhoEve = val_U[nSpecies + nDim + 1];

  double u[ND];
  for (int iDim = 0; iDim < nDim; iDim++) u[iDim] = val_V[idx.VEL() + iDim];

  double proj_vel = 0.0;
  for (int iDim = 0; iDim < nDim; iDim++) proj_vel += u[iDim] * val_normal[iDim];

  /*--- Species density rows ---*/
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++) PJ(iSpecies, jSpecies) += -(rhos[iSpecies] / rho) * proj_vel;
    PJ(iSpecies, iSpecies) += proj_vel;
    for (int iDim = 0; iDim < nDim; iDim++) {
      PJ(iSpecies, nSpecies + iDim) += (rhos[iSpecies] / rho) * val_normal[iDim];
      PJ(nSpecies + iDim, iSpecies) += val_dPdU[iSpecies] * val_normal[iDim] - proj_vel * u[iDim];
    }
    PJ(nSpecies + nDim, iSpecies) += (val_dPdU[iSpecies] - H) * proj_vel;
    PJ(nSpecies + nDim + 1, iSpecies) += -proj_vel * rhoEve / rho;
  }

  /*--- Momentum rows ---*/
  for (int iDim = 0; iDim < nDim; iDim++) {
    for (int jDim = 0; jDim < nDim; jDim++)
      PJ(nSpecies + iDim, nSpecies + jDim) += val_dPdU[nSpecies + jDim] * val_normal[iDim] + u[iDim] * val_normal[jDim];
    PJ(nSpecies + iDim, nSpecies + iDim) += proj_vel;
    PJ(nSpecies + iDim, nSpecies + nDim) += val_dPdU[nSpecies + nDim] * val_normal[iDim];
    PJ(nSpecies + iDim, nSpecies + nDim + 1) += val_dPdU[nSpecies + nDim + 1] * val_normal[iDim];
  }

  /*--- Total energy row ---*/
  for (int iDim = 0; iDim < nDim; iDim++)
    PJ(nSpecies + nDim, nSpecies + iDim) += val_dPdU[nSpecies + iDim] * proj_vel + H * val_normal[iDim];
  PJ(nSpecies + nDim, nSpecies + nDim) += (1 + val_dPdU[nSpecies + nDim]) * proj_vel;
  PJ(nSpecies + nDim, nSpecies + nDim + 1) += val_dPdU[nSpecies + nDim + 1] * proj_vel;

  /*--- Vib.-el. energy row ("=" as on the CPU) ---*/
  for (int iDim = 0; iDim < nDim; iDim++) PJ(nSpecies + nDim + 1, nSpecies + iDim) = rhoEve / rho * val_normal[iDim];
  PJ(nSpecies + nDim + 1, nSpecies + nDim + 1) = proj_vel;

  for (int iVar = 0; iVar < nVar; iVar++)
    for (int jVar = 0; jVar < nVar; jVar++) PJ(iVar, jVar) = val_scale * PJ(iVar, jVar);
#undef PJ
}

/*!
 * \brief CCentLax_NEMO::ComputeResidual. Inputs are the cell values of Centered_Residual (no reconstruction):
 *        U = Solution, V = Primitive, dPdU = nodes dPdU (only read if implicit), Lambda = nodes Lambda
 *        (SetMax_Eigenvalue), nNeighbor = geometry->nodes->GetnNeighbor. kappa0 = config->GetKappa_1st_Flow().
 *        Jacobian_i/j are row-major nVar x nVar and only written if implicit.
 */
template <int ND>
KOKKOS_INLINE_FUNCTION void LAXResidual(const NEMOGasTables& g, int nSpecies, int nVar, int nPrimVar, bool implicit,
                                        double kappa0, const double* Normal, const double* U_i, const double* U_j,
                                        const double* V_i, const double* V_j, const double* dPdU_i,
                                        const double* dPdU_j, double Lambda_i, double Lambda_j,
                                        unsigned short Neighbor_i, unsigned short Neighbor_j, double* Flux,
                                        double* Jacobian_i, double* Jacobian_j) {
  const int nDim = ND;
  const NEMOPrimIndices idx(nSpecies, ND);
  const double Param_p = 0.3;
  const double EPS_ = 1.0E-16; /*!< option_structure.hpp EPS */

  /*--- Calculate geometrical quantities (UnitNormal is unused by the CPU code too) ---*/
  double sq = 0.0;
  for (int iDim = 0; iDim < nDim; iDim++) sq += Normal[iDim] * Normal[iDim];
  const double Area = Kokkos::sqrt(sq);

  const double Density_i = V_i[idx.RHO()], Density_j = V_j[idx.RHO()];
  const double Enthalpy_i = V_i[idx.H()], Enthalpy_j = V_j[idx.H()];
  const double SoundSpeed_i = V_i[idx.A()], SoundSpeed_j = V_j[idx.A()];

  /*--- Mean quantities ---*/
  double MeanU[MAXNV], MeanV[MAXNP];
  for (int iVar = 0; iVar < nVar; iVar++) MeanU[iVar] = 0.5 * (U_i[iVar] + U_j[iVar]);
  for (int iVar = 0; iVar < nPrimVar; iVar++) MeanV[iVar] = 0.5 * (V_i[iVar] + V_j[iVar]);

  /*--- Projected flux tensor ---*/
  double ProjFlux[MAXNV];
  LaxInviscidProjFlux<ND>(nSpecies, nVar, MeanU, MeanV, Normal, ProjFlux);

  if (implicit) {
    /*--- NonEq specific variables (only the Jacobian reads them) ---*/
    double mean_eves[MAXNS];
    double MeandPdU[MAXNV] = {0.0}; /*!< deterministic: see the header note on ionization */
    ComputeSpeciesEve(g, MeanV[idx.TVE()], mean_eves);
    ComputedPdU(g, nDim, MeanV, mean_eves, MeandPdU);

    /*--- scale = 0.5 because val_res ~ 0.5*(fc_i+fc_j)*Normal ---*/
    LaxInviscidProjJac<ND>(nSpecies, nVar, MeanU, MeanV, MeandPdU, Normal, 0.5, Jacobian_i);
    for (int k = 0; k < nVar * nVar; k++) Jacobian_j[k] = Jacobian_i[k];
  }

  /*--- Local spectral radius and stretching factor ---*/
  double ProjVelocity_i = 0.0, ProjVelocity_j = 0.0;
  for (int iDim = 0; iDim < nDim; iDim++) ProjVelocity_i += V_i[idx.VEL() + iDim] * Normal[iDim];
  for (int iDim = 0; iDim < nDim; iDim++) ProjVelocity_j += V_j[idx.VEL() + iDim] * Normal[iDim];

  const double Local_Lambda_i = (Kokkos::fabs(ProjVelocity_i) + SoundSpeed_i * Area);
  const double Local_Lambda_j = (Kokkos::fabs(ProjVelocity_j) + SoundSpeed_j * Area);
  const double MeanLambda = 0.5 * (Local_Lambda_i + Local_Lambda_j);

  const double Phi_i = Kokkos::pow(Lambda_i / (4.0 * MeanLambda + EPS_), Param_p);
  const double Phi_j = Kokkos::pow(Lambda_j / (4.0 * MeanLambda + EPS_), Param_p);
  const double StretchingFactor = 4.0 * Phi_i * Phi_j / (Phi_i + Phi_j + EPS_);

  /*--- Differences of conservative variables (rho*H instead of rho*E in the energy row) ---*/
  double Diff_U[MAXNV];
  for (int iVar = 0; iVar < nVar; iVar++) Diff_U[iVar] = U_i[iVar] - U_j[iVar];
  Diff_U[nSpecies + nDim] = Density_i * Enthalpy_i - Density_j * Enthalpy_j;

  /*--- Dissipation coefficient ---*/
  const double sc0 =
      3.0 * (double(Neighbor_i) + double(Neighbor_j)) / (double(Neighbor_i) * double(Neighbor_j));
  const double Epsilon_0 = kappa0 * sc0 * double(nDim) / 3.0;

  for (int iVar = 0; iVar < nVar; iVar++)
    Flux[iVar] = ProjFlux[iVar] + Epsilon_0 * Diff_U[iVar] * StretchingFactor * MeanLambda;

  if (!implicit) return;

  const double cte = Epsilon_0 * StretchingFactor * MeanLambda;
#define JI(r, c) Jacobian_i[(r) * nVar + (c)]
#define JJ(r, c) Jacobian_j[(r) * nVar + (c)]
  for (int iVar = 0; iVar < nSpecies + nDim; iVar++) {
    JI(iVar, iVar) += cte;
    JJ(iVar, iVar) -= cte;
  }

  /*--- Last rows: differences of rho*Enthalpy, not rho*Energy ---*/
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) JI(nSpecies + nDim, iSpecies) += cte * dPdU_i[iSpecies];
  for (int iDim = 0; iDim < nDim; iDim++) JI(nSpecies + nDim, nSpecies + iDim) += cte * dPdU_i[nSpecies + iDim];
  JI(nSpecies + nDim, nSpecies + nDim) += cte * (1 + dPdU_i[nSpecies + nDim]);
  JI(nSpecies + nDim, nSpecies + nDim + 1) += cte * dPdU_i[nSpecies + nDim + 1];
  JI(nSpecies + nDim + 1, nSpecies + nDim + 1) += cte;

  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) JJ(nSpecies + nDim, iSpecies) -= cte * dPdU_j[iSpecies];
  for (int iDim = 0; iDim < nDim; iDim++)
    JJ(nSpecies + nDim, nSpecies + iDim) -= cte * dPdU_j[nSpecies + nDim]; /*!< CPU quirk: [nSpecies+nDim] */
  JJ(nSpecies + nDim, nSpecies + nDim) -= cte * (1 + dPdU_j[nSpecies + nDim]);
  JJ(nSpecies + nDim, nSpecies + nDim + 1) -= cte * dPdU_j[nSpecies + nDim + 1];
  JJ(nSpecies + nDim + 1, nSpecies + nDim + 1) -= cte;
#undef JI
#undef JJ
}

}  // namespace

#endif  // HAVE_KOKKOS
