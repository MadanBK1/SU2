/*!
 * \file CNEMOKokkosResidualKernels.inl
 * \brief Definitions of the device NEMO source / BC residual kernels (included only by the instantiation TUs).
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

namespace nemo_kokkos_res {

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
        const double Twall = r.customTwall ? r.twallVtx(kv) : r.Twall;
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
 * \brief CNEMONSSolver::BC_Smoluchowski_Maxwell (no ionization: the CPU aborts there). No Jacobian contribution
 *        (as the CPU). GasConstant = FluidModel->ComputeGasConstant() of the host gas-model state (constant over the
 *        CPU loop), passed in. SetVelocity_Old(iPoint, Vector) is written to Solution_Old (resident) or to the
 *        per-vertex slip buffer (host-authoritative: the host applies it).
 */
template <int ND>
void KernelSmoluchowski(const ResArgs& r, unsigned long v0, unsigned long v1) {
  Kokkos::parallel_for(
      "NEMO::BC_Smoluchowski_Maxwell", Kokkos::RangePolicy<exec_space>(v0, v1), KOKKOS_LAMBDA(const unsigned long kv) {
        const UpwArgs& a = r.a;
        const int nSpecies = a.ns, nVar = a.nVar, nDim = ND;
        const NEMOPrimIndices idx(nSpecies, ND);
        const auto iPoint = r.vtxNode(kv);
        if (iPoint >= r.nPointDomain) return;

        /*--- Define 'proportional control' constant ---*/
        const double C = 1;

        /*---Define under-relaxation factors --- */
        const double alpha_V = 0.1;
        const double alpha_T = 1.0;

        const double Twall = r.Twall;
        const double TMAC = 1.0;
        const double TAC = 1.0;
        const double Prandtl_Lam = r.prandtlLam;
        const double PI_NUMBER = a.pi;

        const int T_INDEX = idx.T(), VEL_INDEX = idx.VEL(), TVE_INDEX = idx.TVE();

        /*--- Compute dual-grid area and boundary normal ---*/
        double Normal[ND];
        for (int iDim = 0; iDim < nDim; iDim++) Normal[iDim] = r.vtxNormal(kv, iDim);
        double sq = 0.0;
        for (int iDim = 0; iDim < nDim; ++iDim) sq += Normal[iDim] * Normal[iDim];
        const double Area = Kokkos::sqrt(sq);

        double UnitNormal[ND];
        for (int iDim = 0; iDim < nDim; iDim++) UnitNormal[iDim] = Normal[iDim] / Area;

        /*--- Compute closest normal neighbor ---*/
        const auto jPoint = static_cast<unsigned long>(r.vtxNormalNeighbor(kv));

        /*--- Compute distance between wall & normal neighbor ---*/
        double d2 = 0.0;
        for (int iDim = 0; iDim < nDim; iDim++) {
          const double dd = r.coord(iPoint, iDim) - r.coord(jPoint, iDim);
          d2 += dd * dd;
        }
        const double dij = Kokkos::sqrt(d2);

        /*--- Calculate Pressure ---*/
        const double Pi = a.V(iPoint, idx.P());

        /*--- Calculate the gradient of temperature ---*/
        const double Ti = a.V(iPoint, T_INDEX);
        const double Tj = a.V(jPoint, T_INDEX);
        const double Tvei = a.V(iPoint, TVE_INDEX);
        const double Tvej = a.V(jPoint, TVE_INDEX);

        /*--- Rename variables for convenience ---*/
        const double ktr = r.k(iPoint);
        const double kve = r.kve(iPoint);

        /*--- Retrieve Cv*density ---*/
        const double rhoCvtr = a.V(iPoint, idx.RHOCVTR());
        const double rhoCvve = a.V(iPoint, idx.RHOCVVE());

        /*--- Retrieve Flow Data ---*/
        const double Viscosity = r.mu(iPoint);
        const double Density = a.V(iPoint, idx.RHO());
        const double Gamma = a.Gamma(iPoint);

        /*--- Calculate specific gas constant --- */
        const double GasConstant = r.gasConstant;

        /*--- Calculate temperature gradients normal to surface---*/
        double dTn = 0.0, dTven = 0.0;
        for (int iDim = 0; iDim < nDim; iDim++) dTn += r.grad(iPoint, T_INDEX, iDim) * UnitNormal[iDim];
        for (int iDim = 0; iDim < nDim; iDim++) dTven += r.grad(iPoint, TVE_INDEX, iDim) * UnitNormal[iDim];

        /*--- Calculate molecular mean free path ---*/
        const double Lambda = Viscosity / Density * Kokkos::sqrt(PI_NUMBER / (2.0 * GasConstant * Ti));

        /*--- Calculate Temperature Slip ---*/
        double Tslip = ((2.0 - TAC) / TAC) * 2.0 * Gamma / (Gamma + 1.0) / Prandtl_Lam * Lambda * dTn + Twall;

        double Tslip_ve = Twall;
        if (dTven != 0) Tslip_ve = (Tslip - Twall) * (kve * rhoCvtr / dTn) / (ktr * rhoCvve / dTven) + Twall;

        /*--- Calculate temperature gradients tangent to surface ---*/
        double Vector_Tangent_dT[ND], Vector_Tangent_dTve[ND];
        for (int iDim = 0; iDim < nDim; iDim++) {
          Vector_Tangent_dT[iDim] = r.grad(iPoint, T_INDEX, iDim) - dTn * UnitNormal[iDim];
          Vector_Tangent_dTve[iDim] = r.grad(iPoint, TVE_INDEX, iDim) - dTven * UnitNormal[iDim];
        }

        /*--- Calculate Heatflux tangent to surface ---*/
        double Vector_Tangent_HF[ND];
        for (int iDim = 0; iDim < nDim; iDim++)
          Vector_Tangent_HF[iDim] = -ktr * Vector_Tangent_dT[iDim] - kve * Vector_Tangent_dTve[iDim];

        /*--- Initialize viscous residual to zero ---*/
        double Res_Visc[MAXNV];
        for (int iVar = 0; iVar < nVar; iVar++) Res_Visc[iVar] = 0.0;

        /*--- Compute wall shear stress (CNumerics::ComputeStressTensor, density = turb_ke = 0) ---*/
        double Tau[ND][ND];
        {
          double divVel = 0.0;
          for (int iDim = 0; iDim < nDim; iDim++) divVel += r.grad(iPoint, VEL_INDEX + iDim, iDim);
          const double density = 0.0, turb_ke = 0.0;
          const double pTerm = 2. / 3. * (divVel * Viscosity + density * turb_ke);
          for (int iDim = 0; iDim < nDim; iDim++) {
            for (int jDim = 0; jDim < nDim; jDim++)
              Tau[iDim][jDim] =
                  Viscosity * (r.grad(iPoint, VEL_INDEX + iDim, jDim) + r.grad(iPoint, VEL_INDEX + jDim, iDim));
            Tau[iDim][iDim] -= pTerm;
          }
        }

        /*--- GeometryToolbox::TangentProjection ---*/
        double TauTangent[ND];
        for (int iDim = 0; iDim < nDim; iDim++) {
          double dot = 0.0;
          for (int jDim = 0; jDim < nDim; jDim++) dot += Tau[iDim][jDim] * UnitNormal[jDim];
          TauTangent[iDim] = dot;
        }
        double normalProj = 0.0;
        for (int iDim = 0; iDim < nDim; iDim++) normalProj += TauTangent[iDim] * UnitNormal[iDim];
        for (int iDim = 0; iDim < nDim; iDim++) TauTangent[iDim] -= normalProj * UnitNormal[iDim];

        /*--- Store the Slip Velocity at the wall */
        double Vector[ND];
        for (int iDim = 0; iDim < nDim; iDim++)
          Vector[iDim] = Lambda / Viscosity * (2.0 - TMAC) / TMAC * (TauTangent[iDim]) -
                         3.0 / 4.0 * (Gamma - 1.0) / Gamma * Prandtl_Lam / Pi * Vector_Tangent_HF[iDim];

        /*--- Apply under-relaxation ---*/
        Tslip = (1.0 - alpha_T) * Ti + (alpha_T)*Tslip;
        Tslip_ve = (1.0 - alpha_T) * Tvei + (alpha_T)*Tslip_ve;

        for (int iDim = 0; iDim < nDim; iDim++) {
          Vector[iDim] = (1.0 - alpha_V) * a.V(iPoint, VEL_INDEX + iDim) + (alpha_V)*Vector[iDim];
        }

        /*--- nodes->SetVelocity_Old(iPoint, Vector) ---*/
        if (r.writeWallState) {
          for (int iDim = 0; iDim < nDim; iDim++) r.Uold(iPoint, nSpecies + iDim) = Vector[iDim] * a.V(iPoint, idx.RHO());
        } else {
          for (int iDim = 0; iDim < nDim; iDim++) r.slip(kv, iDim) = Vector[iDim];
        }

        for (int iDim = 0; iDim < nDim; iDim++) {
          a.res(iPoint, nSpecies + iDim) = 0.0;
          if (r.writeWallState) r.trunc(iPoint, nSpecies + iDim) = 0.0;
        }

        /*--- Apply to the linear system ---*/
        Res_Visc[nSpecies + nDim] =
            ((ktr * (Ti - Tj) + kve * (Tvei - Tvej)) + (ktr * (Tslip - Ti) + kve * (Tslip_ve - Tvei)) * C) * Area / dij;
        Res_Visc[nSpecies + nDim + 1] = (kve * (Tvei - Tvej) + kve * (Tslip_ve - Tvei) * C) * Area / dij;

        AddRow(a.res, iPoint, nVar, Res_Visc, -1.0);
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

        /*--- Conservative states (Roe boundary numerics only): SetConservative(Solution(iPoint), U_boundary). ---*/
        const bool msw = (a.scheme == static_cast<int>(CNEMOKokkosData::UpwindScheme::MSW));
        const bool roe = msw || (a.scheme == static_cast<int>(CNEMOKokkosData::UpwindScheme::ROE));
        const double* U_i = roe ? &r.U(iPoint, 0) : zero;
        const double* U_j = (roe && r.bstate.size()) ? &r.bstate(nPrimVar + nSpecies + nVar + 1) : zero;
        double U_outlet[MAXNV];

        if (kind == static_cast<int>(K::SUPERSONIC_OUTLET)) {
          U_j = U_i;
          V_j = V_i;
          eve_j = eve_i;
          dPdU_j = dPdU_i;
          Gamma_j = Gamma_i;
        } else if (kind == static_cast<int>(K::OUTLET)) {
          /*====================== CNEMOEulerSolver::BC_Outlet, fictitious state ======================*/
          double* V_outlet = &r.charac(k, 0);
          const double* U_domain = &r.U(iPoint, 0);
          const double* V_domain = V_i;
          double rhos[MAXNS], Ys[MAXNS];

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
          U_j = U_outlet;
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
        if (msw)
          MSWResidual<ND>(g, nSpecies, nVar, nPrimVar, a.implicit, a.mswAlpha, Normal, U_i, U_j, V_i, V_j, Flux,
                          Jacobian_i, Jacobian_j);
        else if (roe)
          RoeResidual<ND>(g, nSpecies, nVar, nPrimVar, a.implicit, Normal, U_i, U_j, V_i, V_j, dPdU_i, dPdU_j, Flux,
                          Jacobian_i, Jacobian_j);
        else
          AUSMResidual<ND>(a, Normal, V_i, V_j, eve_i, eve_j, dPdU_i, dPdU_j, Gamma_i, Gamma_j, Sensor_i, Sensor_j,
                           Flux, Jacobian_i, Jacobian_j);

        AddRow(a.res, iPoint, nVar, Flux, 1.0);
        if (a.implicit) AddDiag(a, iPoint, Jacobian_i, 1.0);

        /*--- Step 8: viscous contribution of BC_Far_Field / BC_Supersonic_Inlet (CAvgGrad_NEMO, node_infty /
         *    node_inlet, reflected normal neighbour;
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

}  // namespace nemo_kokkos_res
