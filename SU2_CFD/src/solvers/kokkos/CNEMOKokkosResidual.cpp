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
#include <cstring>
#include <string>
#include <vector>

#include "../../../../Common/include/CConfig.hpp"
#include "../../../../Common/include/geometry/CGeometry.hpp"
#include "../../../include/variables/CNEMOEulerVariable.hpp"

#include "CNEMOKokkosAUSM.hpp"

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

/*!
 * \brief Source kernel: CNEMOEulerSolver::Source_Residual loop body (monoatomic / frozen / implicit options,
 *        CSource_NEMO::ComputeChemistry then ComputeVibRelaxation, both on the same point state; the chemistry "ws"
 *        is passed to the V-T term explicitly, as the CPU passes it through the gas-model member).
 */
template <int ND>
void KernelSource(const ResArgs& r, unsigned long& eChm, unsigned long& eVib) {
  Kokkos::parallel_reduce(
      "NEMO::SourceChemVT", Kokkos::RangePolicy<exec_space>(0, r.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint, unsigned long& nChm, unsigned long& nVib) {
        const UpwArgs& a = r.a;
        const NEMOGasTables& g = a.gas;
        const int nSpecies = a.ns, nVar = a.nVar;
        const bool implicit = a.implicit;
        if (r.monoatomic) return;

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
        if (!r.frozen) {
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
        {
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
      },
      Kokkos::Sum<unsigned long>(eChm), Kokkos::Sum<unsigned long>(eVib));
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
        const double Gamma_i = a.Gamma(iPoint);
        const double Sensor_i = a.Sensor.size() ? a.Sensor(iPoint) : 0.0;

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

        const double Sensor_j = Sensor_i; /*--- SetSensor(Sensor(iPoint), Sensor(iPoint)) on the CPU ---*/
        double Flux[MAXNV], Jacobian_i[MAXNV * MAXNV], Jacobian_j[MAXNV * MAXNV];
        AUSMResidual<ND>(a, Normal, V_i, V_j, eve_i, eve_j, dPdU_i, dPdU_j, Gamma_i, Gamma_j, Sensor_i, Sensor_j, Flux,
                         Jacobian_i, Jacobian_j);

        AddRow(a.res, iPoint, nVar, Flux, 1.0);
        if (a.implicit) AddDiag(a, iPoint, Jacobian_i, 1.0);
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
  if (needSensor && (upload || !sensorValid)) {
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
  EnsureInputsOnDevice(uploadInputs, false, false);
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

  if (nDim == 2) KernelSource<2>(r, out.eChm, out.eVib);
  else KernelSource<3>(r, out.eChm, out.eVib);
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

  const bool flux = (p.kind != BCKind::SYMMETRY);
  const bool ausmPlusM = flux && (p.upw.scheme == UpwindScheme::AUSMPLUSM);
  const bool needState = (p.kind == BCKind::FAR_FIELD || p.kind == BCKind::OUTLET || p.kind == BCKind::SUPERSONIC_INLET);
  EnsureInputsOnDevice(uploadInputs, p.kind == BCKind::SYMMETRY || p.kind == BCKind::OUTLET, ausmPlusM);

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

  /*--- Marker range (host copy of marker_ptr cached on first use). ---*/
  if (h_markerPtr.empty()) {
    auto hm = Kokkos::create_mirror_view_and_copy(HostSpace(), geo.marker_ptr);
    h_markerPtr.assign(hm.data(), hm.data() + geo.nMarker + 1);
  }
  const unsigned long v0 = h_markerPtr[p.marker], v1 = h_markerPtr[p.marker + 1];

  long lastSub = -1;
  if (!flux) {
    if (nDim == 2) KernelSymmetry<2>(r, v0, v1);
    else KernelSymmetry<3>(r, v0, v1);
  } else {
    if (nDim == 2) KernelBoundaryFlux<2>(r, static_cast<int>(p.kind), v0, v1, lastSub);
    else KernelBoundaryFlux<3>(r, static_cast<int>(p.kind), v0, v1, lastSub);
  }
  Kokkos::fence();
  P.kernel += timer.seconds();
  timer.reset();

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

#endif  // HAVE_KOKKOS
