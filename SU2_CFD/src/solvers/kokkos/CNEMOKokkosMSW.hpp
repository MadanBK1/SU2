/*!
 * \file CNEMOKokkosMSW.hpp
 * \brief Internal header (item 4.4-MSW): device port of CUpwMSW_NEMO::ComputeResidual (numerics/NEMO/convection/
 *        msw.cpp) incl. the implicit Jacobian. Statement order and CPU quirks are kept. Include only from the
 *        edge-kernel TUs, after CNEMOKokkosRoe.hpp (uses RoeCreateBasis / RoeGetPMatrix / RoeGetPMatrixInv).
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * CPU behaviour kept:
 *  - epsilon = 0 is still added (sqrt(x*x + 0*0)), w = 0.5/(pow(alpha*dp, 2) + 1).
 *  - The node dPdU is not used: dPdU is computed at the weighted (*) states, also for the Jacobian.
 *  - CreateBasis is called twice (same result).
 * CPU behaviour NOT reproducible in parallel:
 *  - dPdUst_i / dPdUst_j are persistent members; with IONIZATION CNEMOGas::ComputedPdU reads the previous call's
 *    state. Here they start at 0 per edge, so the dispatch keeps MSW + ionization on the host.
 */
#pragma once
#ifdef HAVE_KOKKOS

namespace {

/*!
 * \brief CUpwMSW_NEMO::ComputeResidual. U/V are the states set by Upwind_Residual (cell values, or the MUSCL state
 *        with RecomputeConservativeVector). Jacobian_i/j row-major nVar x nVar (written only if implicit).
 */
template <int ND>
KOKKOS_INLINE_FUNCTION void MSWResidual(const NEMOGasTables& g, int nSpecies, int nVar, int nPrimVar, bool implicit,
                                        double alpha, const double* Normal, const double* U_i, const double* U_j,
                                        const double* V_i, const double* V_j, double* Flux, double* Jacobian_i,
                                        double* Jacobian_j) {
  const int nDim = ND;
  const NEMOPrimIndices idx(nSpecies, ND);
  const int TVE_INDEX = idx.TVE(), VEL_INDEX = idx.VEL(), P_INDEX = idx.P(), A_INDEX = idx.A();

  /*--- Set parameters in the numerical method ---*/
  const double epsilon = 0.0;

  /*--- Calculate supporting geometry parameters ---*/
  double sq = 0.0;
  for (int iDim = 0; iDim < nDim; iDim++) sq += Normal[iDim] * Normal[iDim];
  const double Area = Kokkos::sqrt(sq);
  double UnitNormal[ND];
  for (int iDim = 0; iDim < nDim; iDim++) UnitNormal[iDim] = Normal[iDim] / Area;

  /*--- Initialize flux & Jacobian vectors ---*/
  double Fc_i[MAXNV], Fc_j[MAXNV];
  for (int iVar = 0; iVar < nVar; iVar++) {
    Fc_i[iVar] = 0.0;
    Fc_j[iVar] = 0.0;
  }
  if (implicit) {
    for (int iVar = 0; iVar < nVar; iVar++) {
      for (int jVar = 0; jVar < nVar; jVar++) {
        Jacobian_i[iVar * nVar + jVar] = 0.0;
        Jacobian_j[iVar * nVar + jVar] = 0.0;
      }
    }
  }

  /*--- Load variables from nodes i & j ---*/
  const double Pressure_i = V_i[P_INDEX];
  const double Pressure_j = V_j[P_INDEX];

  /*--- Calculate velocity quantities (GeometryToolbox::DotProduct: sum from 0) ---*/
  double ProjVelocity_i = 0.0, ProjVelocity_j = 0.0;
  for (int iDim = 0; iDim < nDim; iDim++) ProjVelocity_i += V_i[VEL_INDEX + iDim] * UnitNormal[iDim];
  for (int iDim = 0; iDim < nDim; iDim++) ProjVelocity_j += V_j[VEL_INDEX + iDim] * UnitNormal[iDim];

  /*--- Calculate the state weighting function ---*/
  const double dp = Kokkos::fabs(Pressure_j - Pressure_i) / kmin(Pressure_j, Pressure_i);
  const double w = 0.5 * (1.0 / (Kokkos::pow(alpha * dp, 2.0) + 1.0));
  const double onemw = 1.0 - w;

  /*--- Calculate weighted state vector (*) for i & j ---*/
  double Ust_i[MAXNV], Ust_j[MAXNV], Vst_i[MAXNP], Vst_j[MAXNP];
  for (int iVar = 0; iVar < nVar; iVar++) {
    Ust_i[iVar] = onemw * U_i[iVar] + w * U_j[iVar];
    Ust_j[iVar] = onemw * U_j[iVar] + w * U_i[iVar];
  }
  for (int iVar = 0; iVar < nPrimVar; iVar++) {
    Vst_i[iVar] = onemw * V_i[iVar] + w * V_j[iVar];
    Vst_j[iVar] = onemw * V_j[iVar] + w * V_i[iVar];
  }
  const double ProjVelst_i = onemw * ProjVelocity_i + w * ProjVelocity_j;
  const double ProjVelst_j = onemw * ProjVelocity_j + w * ProjVelocity_i;

  double eves_st_i[MAXNS], eves_st_j[MAXNS];
  ComputeSpeciesEve(g, Vst_i[TVE_INDEX], eves_st_i);
  ComputeSpeciesEve(g, Vst_j[TVE_INDEX], eves_st_j);

  /*--- Deterministic start (see the header note on ionization) ---*/
  double dPdUst_i[MAXNV] = {0.0}, dPdUst_j[MAXNV] = {0.0};
  ComputedPdU(g, nDim, Vst_i, eves_st_i, dPdUst_i);
  ComputedPdU(g, nDim, Vst_j, eves_st_j, dPdUst_j);

  /*--- Flow eigenvalues at i (Lambda+) ---*/
  double Lambda_i[MAXNV], Lambda_j[MAXNV];
  for (int iVar = 0; iVar < nSpecies + nDim - 1; iVar++)
    Lambda_i[iVar] = 0.5 * (ProjVelst_i + Kokkos::sqrt(ProjVelst_i * ProjVelst_i + epsilon * epsilon));
  Lambda_i[nSpecies + nDim - 1] =
      0.5 * (ProjVelst_i + Vst_i[A_INDEX] +
             Kokkos::sqrt((ProjVelst_i + Vst_i[A_INDEX]) * (ProjVelst_i + Vst_i[A_INDEX]) + epsilon * epsilon));
  Lambda_i[nSpecies + nDim] =
      0.5 * (ProjVelst_i - Vst_i[A_INDEX] +
             Kokkos::sqrt((ProjVelst_i - Vst_i[A_INDEX]) * (ProjVelst_i - Vst_i[A_INDEX]) + epsilon * epsilon));
  Lambda_i[nSpecies + nDim + 1] = 0.5 * (ProjVelst_i + Kokkos::sqrt(ProjVelst_i * ProjVelst_i + epsilon * epsilon));

  /*--- Compute projected P, invP, and Lambda ---*/
  double l[3] = {0.0, 0.0, 0.0}, m[3] = {0.0, 0.0, 0.0};
  double P_Tensor[MAXNV * MAXNV], invP_Tensor[MAXNV * MAXNV];
  RoeCreateBasis<ND>(UnitNormal, l, m);
  RoeGetPMatrix<ND>(nSpecies, nVar, Ust_i, Vst_i, dPdUst_i, UnitNormal, l, m, P_Tensor);
  RoeGetPMatrixInv<ND>(nSpecies, nVar, Ust_i, Vst_i, dPdUst_i, UnitNormal, l, m, invP_Tensor);

  /*--- Projected flux (f+) at i ---*/
  for (int iVar = 0; iVar < nVar; iVar++) {
    for (int jVar = 0; jVar < nVar; jVar++) {
      double Proj_ModJac_Tensor_i = 0.0;
      /*--- Compute Proj_ModJac_Tensor = P x Lambda+ x inverse P ---*/
      for (int kVar = 0; kVar < nVar; kVar++)
        Proj_ModJac_Tensor_i += P_Tensor[iVar * nVar + kVar] * Lambda_i[kVar] * invP_Tensor[kVar * nVar + jVar];
      Fc_i[iVar] += Proj_ModJac_Tensor_i * U_i[jVar] * Area;
      if (implicit) Jacobian_i[iVar * nVar + jVar] += Proj_ModJac_Tensor_i * Area;
    }
  }

  /*--- Flow eigenvalues at j (Lambda-) ---*/
  for (int iVar = 0; iVar < nSpecies + nDim - 1; iVar++)
    Lambda_j[iVar] = 0.5 * (ProjVelst_j - Kokkos::sqrt(ProjVelst_j * ProjVelst_j + epsilon * epsilon));
  Lambda_j[nSpecies + nDim - 1] =
      0.5 * (ProjVelst_j + Vst_j[A_INDEX] -
             Kokkos::sqrt((ProjVelst_j + Vst_j[A_INDEX]) * (ProjVelst_j + Vst_j[A_INDEX]) + epsilon * epsilon));
  Lambda_j[nSpecies + nDim] =
      0.5 * (ProjVelst_j - Vst_j[A_INDEX] -
             Kokkos::sqrt((ProjVelst_j - Vst_j[A_INDEX]) * (ProjVelst_j - Vst_j[A_INDEX]) + epsilon * epsilon));
  Lambda_j[nSpecies + nDim + 1] = 0.5 * (ProjVelst_j - Kokkos::sqrt(ProjVelst_j * ProjVelst_j + epsilon * epsilon));

  /*--- Compute projected P, invP, and Lambda ---*/
  RoeCreateBasis<ND>(UnitNormal, l, m);
  RoeGetPMatrix<ND>(nSpecies, nVar, Ust_j, Vst_j, dPdUst_j, UnitNormal, l, m, P_Tensor);
  RoeGetPMatrixInv<ND>(nSpecies, nVar, Ust_j, Vst_j, dPdUst_j, UnitNormal, l, m, invP_Tensor);

  /*--- Projected flux (f-) ---*/
  for (int iVar = 0; iVar < nVar; iVar++) {
    for (int jVar = 0; jVar < nVar; jVar++) {
      double Proj_ModJac_Tensor_j = 0.0;
      /*--- Compute Proj_ModJac_Tensor = P x Lambda- x inverse P ---*/
      for (int kVar = 0; kVar < nVar; kVar++)
        Proj_ModJac_Tensor_j += P_Tensor[iVar * nVar + kVar] * Lambda_j[kVar] * invP_Tensor[kVar * nVar + jVar];
      Fc_j[iVar] += Proj_ModJac_Tensor_j * U_j[jVar] * Area;
      if (implicit) Jacobian_j[iVar * nVar + jVar] += Proj_ModJac_Tensor_j * Area;
    }
  }

  /*--- Flux splitting ---*/
  for (int iVar = 0; iVar < nVar; iVar++) Flux[iVar] = Fc_i[iVar] + Fc_j[iVar];
}

}  // namespace

#endif  // HAVE_KOKKOS
