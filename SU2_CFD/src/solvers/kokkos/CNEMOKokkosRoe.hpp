/*!
 * \file CNEMOKokkosRoe.hpp
 * \brief Internal header (item 4.4-Roe): device port of CUpwRoe_NEMO::ComputeResidual (numerics/NEMO/convection/
 *        roe.cpp) incl. the implicit Jacobian, CNEMONumerics::GetPMatrix / GetPMatrix_inv (CNEMONumerics.cpp),
 *        CNumerics::CreateBasis and CNEMOEulerSolver::RecomputeConservativeVector (MUSCL states).
 *        Statement order and CPU quirks are kept. Include only from the edge-kernel TUs, after CNEMOKokkosAUSM.hpp
 *        and CNEMOKokkosLAX.hpp (uses MAXNV / MAXNP, the gas helpers and LaxInviscidProjFlux / LaxInviscidProjJac).
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * CPU quirks reproduced:
 *  - Epsilon[nSpecies+iDim] (iDim < nDim-1) uses Lambda[iDim], not Lambda[nSpecies+iDim].
 *  - GetPMatrix / GetPMatrix_inv in 2D accumulate with "+=" on the zeroed matrix (incl. "+= 0.0"), 3D P uses "=".
 *  - Lambda gets a second fabs pass after the entropy fix.
 * CPU behaviour NOT reproducible in parallel:
 *  - RoedPdU is a persistent, never-initialised member. With IONIZATION, CNEMOGas::ComputedPdU reads rho_el from the
 *    previous call and updates entries with "-=" (edge-order dependent). Here it starts at 0 per edge, so the
 *    dispatch keeps Roe + ionization on the host (Device_UpwindUnsupported).
 */

#pragma once

#ifdef HAVE_KOKKOS

namespace {

/*! \brief CNumerics::CreateBasis (val_Normal is the UNIT normal in CUpwRoe_NEMO). */
template <int ND>
KOKKOS_INLINE_FUNCTION void RoeCreateBasis(const double* val_Normal, double* l, double* m) {
  const int nDim = ND;
  double modm, modl;
  if (nDim == 2) {
    l[0] = -val_Normal[1];
    l[1] = val_Normal[0];
    m[0] = 0.0;
    m[1] = 0.0;
    modl = 0.0;
    for (int iDim = 0; iDim < nDim; iDim++) modl += l[iDim] * l[iDim];
    modl = Kokkos::sqrt(modl);
    for (int iDim = 0; iDim < nDim; iDim++) l[iDim] = l[iDim] / modl;
  } else {
    l[0] = 0.0;
    l[1] = -val_Normal[2];
    l[2] = val_Normal[1];
    if (l[0] == 0.0 && l[1] == 0.0 && l[2] == 0.0) {
      l[0] = -val_Normal[2];
      l[1] = 0.0;
      l[2] = val_Normal[0];
    }
    m[0] = val_Normal[1] * l[2] - val_Normal[2] * l[1];
    m[1] = val_Normal[2] * l[0] - val_Normal[0] * l[2];
    m[2] = val_Normal[0] * l[1] - val_Normal[1] * l[0];
    modm = 0;
    modl = 0;
    for (int iDim = 0; iDim < nDim; iDim++) {
      modm += m[iDim] * m[iDim];
      modl += l[iDim] * l[iDim];
    }
    modm = Kokkos::sqrt(modm);
    modl = Kokkos::sqrt(modl);
    for (int iDim = 0; iDim < nDim; iDim++) {
      l[iDim] = l[iDim] / modl;
      m[iDim] = m[iDim] / modm;
    }
  }
}

/*! \brief CNEMONumerics::GetPMatrix, row-major nVar x nVar. */
template <int ND>
KOKKOS_INLINE_FUNCTION void RoeGetPMatrix(int nSpecies, int nVar, const double* U, const double* V,
                                          const double* val_dPdU, const double* val_normal, const double* l,
                                          const double* m, double* Pt) {
  const int nDim = ND;
  const NEMOPrimIndices idx(nSpecies, ND);
  const int RHOS_INDEX = idx.RHOS(), VEL_INDEX = idx.VEL(), RHO_INDEX = idx.RHO(), H_INDEX = idx.H(),
            A_INDEX = idx.A();
#define PT(r, c) Pt[(r) * nVar + (c)]
  for (int iVar = 0; iVar < nVar; iVar++)
    for (int jVar = 0; jVar < nVar; jVar++) PT(iVar, jVar) = 0.0;

  double sqvel = 0.0;
  const double rho = V[RHO_INDEX];
  const double eve = U[nSpecies + nDim + 1] / rho;
  double vU, vV, vW;
  vU = vV = vW = 0.0;
  for (int iDim = 0; iDim < nDim; iDim++) {
    vU += V[VEL_INDEX + iDim] * val_normal[iDim];
    vV += V[VEL_INDEX + iDim] * l[iDim];
    vW += V[VEL_INDEX + iDim] * m[iDim];
    sqvel += V[VEL_INDEX + iDim] * V[VEL_INDEX + iDim];
  }
  const double a = V[A_INDEX];
  const double a2 = V[A_INDEX] * V[A_INDEX];

  if (nDim == 2) {
    for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
      PT(iSpecies, iSpecies) += 1.0 / a2;
      PT(iSpecies, nSpecies) += 0.0;
      PT(iSpecies, nSpecies + 1) += V[RHOS_INDEX + iSpecies] / (2.0 * rho * a2);
      PT(iSpecies, nSpecies + 2) += V[RHOS_INDEX + iSpecies] / (2.0 * rho * a2);
      PT(iSpecies, nSpecies + 3) += 0.0;

      PT(nSpecies, iSpecies) += V[VEL_INDEX] / a2;
      PT(nSpecies + 1, iSpecies) += V[VEL_INDEX + 1] / a2;
      PT(nSpecies + 2, iSpecies) +=
          (val_dPdU[nSpecies + nDim] * sqvel - val_dPdU[iSpecies]) / (val_dPdU[nSpecies + nDim] * a2);
      PT(nSpecies + 3, iSpecies) += 0.0;
    }

    for (int iDim = 0; iDim < nDim; iDim++) {
      PT(nSpecies + iDim, nSpecies) += l[iDim];
      PT(nSpecies + iDim, nSpecies + 1) += (V[VEL_INDEX + iDim] + a * val_normal[iDim]) / (2.0 * a2);
      PT(nSpecies + iDim, nSpecies + 2) += (V[VEL_INDEX + iDim] - a * val_normal[iDim]) / (2.0 * a2);
      PT(nSpecies + iDim, nSpecies + 3) += 0.0;
    }

    PT(nSpecies + 2, nSpecies) += vV;
    PT(nSpecies + 2, nSpecies + 1) += ((V[H_INDEX]) + a * vU) / (2.0 * a2);
    PT(nSpecies + 2, nSpecies + 2) += ((V[H_INDEX]) - a * vU) / (2.0 * a2);
    PT(nSpecies + 2, nSpecies + 3) += -val_dPdU[nSpecies + nDim + 1] / (val_dPdU[nSpecies + nDim] * a2);

    PT(nSpecies + 3, nSpecies) += 0.0;
    PT(nSpecies + 3, nSpecies + 1) += eve / (2.0 * a2);
    PT(nSpecies + 3, nSpecies + 2) += eve / (2.0 * a2);
    PT(nSpecies + 3, nSpecies + 3) += 1.0 / a2;

  } else {
    for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
      PT(iSpecies, iSpecies) = 1.0 / a2;
      PT(iSpecies, nSpecies) = 0.0;
      PT(iSpecies, nSpecies + 1) = 0.0;
      PT(iSpecies, nSpecies + 2) = V[RHOS_INDEX + iSpecies] / (2.0 * rho * a2);
      PT(iSpecies, nSpecies + 3) = V[RHOS_INDEX + iSpecies] / (2.0 * rho * a2);
      PT(iSpecies, nSpecies + 4) = 0.0;

      PT(nSpecies, iSpecies) = V[VEL_INDEX] / a2;
      PT(nSpecies + 1, iSpecies) = V[VEL_INDEX + 1] / a2;
      PT(nSpecies + 2, iSpecies) = V[VEL_INDEX + 2] / a2;
      PT(nSpecies + 3, iSpecies) =
          (val_dPdU[nSpecies + 3] * sqvel - val_dPdU[iSpecies]) / (val_dPdU[nSpecies + 3] * a2);
      PT(nSpecies + 4, iSpecies) = 0.0;
    }

    for (int iDim = 0; iDim < nDim; iDim++) {
      PT(nSpecies + iDim, nSpecies) = l[iDim];
      PT(nSpecies + iDim, nSpecies + 1) = m[iDim];
      PT(nSpecies + iDim, nSpecies + 2) = (V[VEL_INDEX + iDim] + a * val_normal[iDim]) / (2.0 * a2);
      PT(nSpecies + iDim, nSpecies + 3) = (V[VEL_INDEX + iDim] - a * val_normal[iDim]) / (2.0 * a2);
      PT(nSpecies + iDim, nSpecies + 4) = 0.0;
    }

    PT(nSpecies + 3, nSpecies) = vV;
    PT(nSpecies + 3, nSpecies + 1) = vW;
    PT(nSpecies + 3, nSpecies + 2) = ((V[H_INDEX]) + a * vU) / (2.0 * a2);
    PT(nSpecies + 3, nSpecies + 3) = ((V[H_INDEX]) - a * vU) / (2.0 * a2);
    PT(nSpecies + 3, nSpecies + 4) = -val_dPdU[nSpecies + nDim + 1] / (val_dPdU[nSpecies + nDim] * a2);

    PT(nSpecies + 4, nSpecies) = 0.0;
    PT(nSpecies + 4, nSpecies + 1) = 0.0;
    PT(nSpecies + 4, nSpecies + 2) = eve / (2.0 * a2);
    PT(nSpecies + 4, nSpecies + 3) = eve / (2.0 * a2);
    PT(nSpecies + 4, nSpecies + 4) = 1.0 / a2;
  }
#undef PT
}

/*! \brief CNEMONumerics::GetPMatrix_inv, row-major nVar x nVar. */
template <int ND>
KOKKOS_INLINE_FUNCTION void RoeGetPMatrixInv(int nSpecies, int nVar, const double* U, const double* V,
                                             const double* val_dPdU, const double* val_normal, const double* l,
                                             const double* m, double* Ip) {
  const int nDim = ND;
  const NEMOPrimIndices idx(nSpecies, ND);
  const int RHOS_INDEX = idx.RHOS(), VEL_INDEX = idx.VEL(), RHO_INDEX = idx.RHO(), A_INDEX = idx.A();
#define IP(r, c) Ip[(r) * nVar + (c)]
  for (int iVar = 0; iVar < nVar; iVar++)
    for (int jVar = 0; jVar < nVar; jVar++) IP(iVar, jVar) = 0.0;

  const double rho = V[RHO_INDEX];
  const double eve = U[nSpecies + nDim + 1] / rho;
  double vU, vV, vW;
  vU = vV = vW = 0.0;
  for (int iDim = 0; iDim < nDim; iDim++) {
    vU += V[VEL_INDEX + iDim] * val_normal[iDim];
    vV += V[VEL_INDEX + iDim] * l[iDim];
    vW += V[VEL_INDEX + iDim] * m[iDim];
  }
  const double a = V[A_INDEX];
  const double a2 = V[A_INDEX] * V[A_INDEX];
  const double* dP = val_dPdU;

  if (nDim == 2) {
    for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
      for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++)
        IP(iSpecies, jSpecies) += -(V[RHOS_INDEX + iSpecies] / rho) * dP[jSpecies];
      IP(iSpecies, iSpecies) += a2;
      IP(iSpecies, nSpecies) += dP[nSpecies + nDim] * V[VEL_INDEX] * (V[RHOS_INDEX + iSpecies] / rho);
      IP(iSpecies, nSpecies + 1) += dP[nSpecies + nDim] * V[VEL_INDEX + 1] * (V[RHOS_INDEX + iSpecies] / rho);
      IP(iSpecies, nSpecies + 2) += -dP[nSpecies + nDim] * (V[RHOS_INDEX + iSpecies] / rho);
      IP(iSpecies, nSpecies + 3) += -dP[nSpecies + nDim + 1] * (V[RHOS_INDEX + iSpecies] / rho);

      IP(nSpecies, iSpecies) += -vV;
      IP(nSpecies + 1, iSpecies) += dP[iSpecies] - vU * a;
      IP(nSpecies + 2, iSpecies) += dP[iSpecies] + vU * a;
      IP(nSpecies + 3, iSpecies) += -eve * dP[iSpecies];
    }

    IP(nSpecies, nSpecies) += l[0];
    IP(nSpecies, nSpecies + 1) += l[1];
    IP(nSpecies, nSpecies + 2) += 0.0;
    IP(nSpecies, nSpecies + 3) += 0.0;

    IP(nSpecies + 1, nSpecies) += a * val_normal[0] - dP[nSpecies + nDim] * V[VEL_INDEX];
    IP(nSpecies + 1, nSpecies + 1) += a * val_normal[1] - dP[nSpecies + nDim] * V[VEL_INDEX + 1];
    IP(nSpecies + 1, nSpecies + 2) += dP[nSpecies + nDim];
    IP(nSpecies + 1, nSpecies + 3) += dP[nSpecies + nDim + 1];

    IP(nSpecies + 2, nSpecies) += -a * val_normal[0] - dP[nSpecies + nDim] * V[VEL_INDEX];
    IP(nSpecies + 2, nSpecies + 1) += -a * val_normal[1] - dP[nSpecies + nDim] * V[VEL_INDEX + 1];
    IP(nSpecies + 2, nSpecies + 2) += dP[nSpecies + nDim];
    IP(nSpecies + 2, nSpecies + 3) += dP[nSpecies + nDim + 1];

    IP(nSpecies + 3, nSpecies) += dP[nSpecies + nDim] * V[VEL_INDEX] * eve;
    IP(nSpecies + 3, nSpecies + 1) += dP[nSpecies + nDim] * V[VEL_INDEX + 1] * eve;
    IP(nSpecies + 3, nSpecies + 2) += -dP[nSpecies + nDim] * eve;
    IP(nSpecies + 3, nSpecies + 3) += a2 - dP[nSpecies + nDim + 1] * eve;

  } else {
    for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
      for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++)
        IP(iSpecies, jSpecies) += -(V[RHOS_INDEX + iSpecies] / rho) * dP[jSpecies];
      IP(iSpecies, iSpecies) += a2;
      IP(iSpecies, nSpecies) += dP[nSpecies + nDim] * V[VEL_INDEX] * (V[RHOS_INDEX + iSpecies] / rho);
      IP(iSpecies, nSpecies + 1) += dP[nSpecies + nDim] * V[VEL_INDEX + 1] * (V[RHOS_INDEX + iSpecies] / rho);
      IP(iSpecies, nSpecies + 2) += dP[nSpecies + nDim] * V[VEL_INDEX + 2] * (V[RHOS_INDEX + iSpecies] / rho);
      IP(iSpecies, nSpecies + 3) += -dP[nSpecies + nDim] * (V[RHOS_INDEX + iSpecies] / rho);
      IP(iSpecies, nSpecies + 4) += -dP[nSpecies + nDim + 1] * (V[RHOS_INDEX + iSpecies] / rho);

      IP(nSpecies, iSpecies) += -vV;
      IP(nSpecies + 1, iSpecies) += -vW;
      IP(nSpecies + 2, iSpecies) += dP[iSpecies] - vU * a;
      IP(nSpecies + 3, iSpecies) += dP[iSpecies] + vU * a;
      IP(nSpecies + 4, iSpecies) += -eve * dP[iSpecies];
    }

    IP(nSpecies, nSpecies) += l[0];
    IP(nSpecies, nSpecies + 1) += l[1];
    IP(nSpecies, nSpecies + 2) += l[2];
    IP(nSpecies, nSpecies + 3) += 0.0;
    IP(nSpecies, nSpecies + 4) += 0.0;

    IP(nSpecies + 1, nSpecies) += m[0];
    IP(nSpecies + 1, nSpecies + 1) += m[1];
    IP(nSpecies + 1, nSpecies + 2) += m[2];
    IP(nSpecies + 1, nSpecies + 3) += 0.0;
    IP(nSpecies + 1, nSpecies + 4) += 0.0;

    IP(nSpecies + 2, nSpecies) += a * val_normal[0] - dP[nSpecies + nDim] * V[VEL_INDEX];
    IP(nSpecies + 2, nSpecies + 1) += a * val_normal[1] - dP[nSpecies + nDim] * V[VEL_INDEX + 1];
    IP(nSpecies + 2, nSpecies + 2) += a * val_normal[2] - dP[nSpecies + nDim] * V[VEL_INDEX + 2];
    IP(nSpecies + 2, nSpecies + 3) += dP[nSpecies + nDim];
    IP(nSpecies + 2, nSpecies + 4) += dP[nSpecies + nDim + 1];

    IP(nSpecies + 3, nSpecies) += -a * val_normal[0] - dP[nSpecies + nDim] * V[VEL_INDEX];
    IP(nSpecies + 3, nSpecies + 1) += -a * val_normal[1] - dP[nSpecies + nDim] * V[VEL_INDEX + 1];
    IP(nSpecies + 3, nSpecies + 2) += -a * val_normal[2] - dP[nSpecies + nDim] * V[VEL_INDEX + 2];
    IP(nSpecies + 3, nSpecies + 3) += dP[nSpecies + nDim];
    IP(nSpecies + 3, nSpecies + 4) += dP[nSpecies + nDim + 1];

    IP(nSpecies + 4, nSpecies) += dP[nSpecies + nDim] * V[VEL_INDEX] * eve;
    IP(nSpecies + 4, nSpecies + 1) += dP[nSpecies + nDim] * V[VEL_INDEX + 1] * eve;
    IP(nSpecies + 4, nSpecies + 2) += dP[nSpecies + nDim] * V[VEL_INDEX + 2] * eve;
    IP(nSpecies + 4, nSpecies + 3) += -dP[nSpecies + nDim] * eve;
    IP(nSpecies + 4, nSpecies + 4) += a2 - dP[nSpecies + nDim + 1] * eve;
  }
#undef IP
}

/*! \brief CNEMOEulerSolver::RecomputeConservativeVector (MUSCL state of Roe / MSW). */
template <int ND>
KOKKOS_INLINE_FUNCTION void RoeRecomputeConservative(const NEMOGasTables& g, int nSpecies, double* U,
                                                     const double* V) {
  const int nDim = ND;
  const NEMOPrimIndices idx(nSpecies, ND);
  double rhos[MAXNS];
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    U[iSpecies] = V[iSpecies];
    rhos[iSpecies] = V[iSpecies];
  }
  double sqvel = 0.0;
  for (int iDim = 0; iDim < nDim; iDim++) {
    U[nSpecies + iDim] = V[idx.RHO()] * V[idx.VEL() + iDim];
    sqvel += V[idx.VEL() + iDim] * V[idx.VEL() + iDim];
  }
  double Energies[2];
  ComputeMixtureEnergies(g, rhos, V[idx.T()], V[idx.TVE()], Energies);
  U[nSpecies + nDim] = V[idx.RHO()] * (Energies[0] + 0.5 * sqvel);
  U[nSpecies + nDim + 1] = V[idx.RHO()] * (Energies[1]);
}

/*!
 * \brief CUpwRoe_NEMO::ComputeResidual. U/V are the states set by Upwind_Residual (cell or MUSCL), dPdU the node
 *        values (zero for a MUSCL state, only read if implicit). Jacobian_i/j row-major nVar x nVar.
 */
template <int ND>
KOKKOS_INLINE_FUNCTION void RoeResidual(const NEMOGasTables& g, int nSpecies, int nVar, int nPrimVar, bool implicit,
                                        const double* Normal, const double* U_i, const double* U_j,
                                        const double* V_i, const double* V_j, const double* dPdU_i,
                                        const double* dPdU_j, double* Flux, double* Jacobian_i,
                                        double* Jacobian_j) {
  const int nDim = ND;
  const NEMOPrimIndices idx(nSpecies, ND);
  const int RHO_INDEX = idx.RHO(), TVE_INDEX = idx.TVE(), VEL_INDEX = idx.VEL(), P_INDEX = idx.P(),
            A_INDEX = idx.A();

  /*--- Face area and unit normal ---*/
  double sq = 0.0;
  for (int iDim = 0; iDim < nDim; iDim++) sq += Normal[iDim] * Normal[iDim];
  const double Area = Kokkos::sqrt(sq);
  double UnitNormal[ND];
  for (int iDim = 0; iDim < nDim; iDim++) UnitNormal[iDim] = Normal[iDim] / Area;

  /*--- Roe averaged variables ---*/
  const double R = Kokkos::sqrt(Kokkos::fabs(V_j[RHO_INDEX] / V_i[RHO_INDEX]));

  double RoeU[MAXNV], RoeV[MAXNP];
  for (int iVar = 0; iVar < nVar; iVar++) RoeU[iVar] = (R * U_j[iVar] + U_i[iVar]) / (R + 1);
  for (int iVar = 0; iVar < nPrimVar; iVar++) RoeV[iVar] = (R * V_j[iVar] + V_i[iVar]) / (R + 1);

  double roe_eves[MAXNS];
  ComputeSpeciesEve(g, RoeV[TVE_INDEX], roe_eves);

  /*--- Derivatives of pressure (deterministic start: see the header note on ionization) ---*/
  double RoedPdU[MAXNV] = {0.0};
  ComputedPdU(g, nDim, RoeV, roe_eves, RoedPdU);

  /*--- Dual grid tangent vectors ---*/
  double l[3] = {0.0, 0.0, 0.0}, m[3] = {0.0, 0.0, 0.0};
  RoeCreateBasis<ND>(UnitNormal, l, m);

  /*--- Inviscid projected fluxes ---*/
  double ProjFlux_i[MAXNV], ProjFlux_j[MAXNV];
  LaxInviscidProjFlux<ND>(nSpecies, nVar, U_i, V_i, Normal, ProjFlux_i);
  LaxInviscidProjFlux<ND>(nSpecies, nVar, U_j, V_j, Normal, ProjFlux_j);

  /*--- P, invP ---*/
  double P_Tensor[MAXNV * MAXNV], invP_Tensor[MAXNV * MAXNV];
  RoeGetPMatrix<ND>(nSpecies, nVar, RoeU, RoeV, RoedPdU, UnitNormal, l, m, P_Tensor);
  RoeGetPMatrixInv<ND>(nSpecies, nVar, RoeU, RoeV, RoedPdU, UnitNormal, l, m, invP_Tensor);

  /*--- Projected velocities ---*/
  double ProjVelocity = 0.0, ProjVelocity_i = 0.0, ProjVelocity_j = 0.0;
  for (int iDim = 0; iDim < nDim; iDim++) ProjVelocity += RoeV[VEL_INDEX + iDim] * UnitNormal[iDim];
  for (int iDim = 0; iDim < nDim; iDim++) ProjVelocity_i += V_i[VEL_INDEX + iDim] * UnitNormal[iDim];
  for (int iDim = 0; iDim < nDim; iDim++) ProjVelocity_j += V_j[VEL_INDEX + iDim] * UnitNormal[iDim];

  const double RoeSoundSpeed = Kokkos::sqrt((1.0 + RoedPdU[nSpecies + nDim]) * RoeV[P_INDEX] / RoeV[RHO_INDEX]);

  /*--- Eigenvalues ---*/
  double Lambda[MAXNV], Epsilon[MAXNV];
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) Lambda[iSpecies] = ProjVelocity;
  for (int iDim = 0; iDim < nDim - 1; iDim++) Lambda[nSpecies + iDim] = ProjVelocity;
  Lambda[nSpecies + nDim - 1] = ProjVelocity + RoeSoundSpeed;
  Lambda[nSpecies + nDim] = ProjVelocity - RoeSoundSpeed;
  Lambda[nSpecies + nDim + 1] = ProjVelocity;

  /*--- Harten and Hyman (1983) entropy correction ---*/
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++)
    Epsilon[iSpecies] =
        4.0 * kmax(0.0, kmax(Lambda[iSpecies] - ProjVelocity_i, ProjVelocity_j - Lambda[iSpecies]));
  for (int iDim = 0; iDim < nDim - 1; iDim++) /*--- CPU quirk: Lambda[iDim] ---*/
    Epsilon[nSpecies + iDim] = 4.0 * kmax(0.0, kmax(Lambda[iDim] - ProjVelocity_i, ProjVelocity_j - Lambda[iDim]));
  Epsilon[nSpecies + nDim - 1] =
      4.0 * kmax(0.0, kmax(Lambda[nSpecies + nDim - 1] - (ProjVelocity_i + V_i[A_INDEX]),
                           (ProjVelocity_j + V_j[A_INDEX]) - Lambda[nSpecies + nDim - 1]));
  Epsilon[nSpecies + nDim] = 4.0 * kmax(0.0, kmax(Lambda[nSpecies + nDim] - (ProjVelocity_i - V_i[A_INDEX]),
                                                  (ProjVelocity_j - V_j[A_INDEX]) - Lambda[nSpecies + nDim]));
  Epsilon[nSpecies + nDim + 1] = 4.0 * kmax(0.0, kmax(Lambda[nSpecies + nDim + 1] - ProjVelocity_i,
                                                      ProjVelocity_j - Lambda[nSpecies + nDim + 1]));
  for (int iVar = 0; iVar < nVar; iVar++)
    if (Kokkos::fabs(Lambda[iVar]) < Epsilon[iVar])
      Lambda[iVar] = (Lambda[iVar] * Lambda[iVar] + Epsilon[iVar] * Epsilon[iVar]) / (2.0 * Epsilon[iVar]);
    else
      Lambda[iVar] = Kokkos::fabs(Lambda[iVar]);

  for (int iVar = 0; iVar < nVar; iVar++) Lambda[iVar] = Kokkos::fabs(Lambda[iVar]);

  /*--- Inviscid projected Jacobians (scale 0.5) ---*/
  if (implicit) {
    LaxInviscidProjJac<ND>(nSpecies, nVar, U_i, V_i, dPdU_i, Normal, 0.5, Jacobian_i);
    LaxInviscidProjJac<ND>(nSpecies, nVar, U_j, V_j, dPdU_j, Normal, 0.5, Jacobian_j);
  }

  /*--- Difference of conserved variables ---*/
  double Diff_U[MAXNV];
  for (int iVar = 0; iVar < nVar; iVar++) Diff_U[iVar] = U_j[iVar] - U_i[iVar];

  /*--- Roe's flux approximation ---*/
  for (int iVar = 0; iVar < nVar; iVar++) {
    Flux[iVar] = 0.5 * (ProjFlux_i[iVar] + ProjFlux_j[iVar]);
    for (int jVar = 0; jVar < nVar; jVar++) {
      /*--- |Proj_ModJac_Tensor| = P x |Lambda| x inverse P ---*/
      double Proj_ModJac_Tensor_ij = 0.0;
      for (int kVar = 0; kVar < nVar; kVar++)
        Proj_ModJac_Tensor_ij += P_Tensor[iVar * nVar + kVar] * Lambda[kVar] * invP_Tensor[kVar * nVar + jVar];

      Flux[iVar] -= 0.5 * Proj_ModJac_Tensor_ij * Diff_U[jVar] * Area;
      if (implicit) {
        Jacobian_i[iVar * nVar + jVar] += 0.5 * Proj_ModJac_Tensor_ij * Area;
        Jacobian_j[iVar * nVar + jVar] -= 0.5 * Proj_ModJac_Tensor_ij * Area;
      }
    }
  }
}

}  // namespace

#endif  // HAVE_KOKKOS
