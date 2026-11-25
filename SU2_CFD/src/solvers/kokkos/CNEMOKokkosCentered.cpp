/*!
 * \file CNEMOKokkosCentered.cpp
 * \brief Device (Kokkos) NEMO centered residual, item 4.4-LAX of NEMO_KOKKOS_PORT_PLAN.md:
 *        CNEMOEulerSolver::Centered_Residual for LAX-FRIEDRICH (CCentLax_NEMO::ComputeResidual, incl. the implicit
 *        Jacobian), CheckResidualNaNs and residual / block-CSR Jacobian accumulation (same pattern as UpwindEdge).
 *
 * Reviewer scaffold (review 3): compiled with -fsyntax-only against Kokkos Serial, NOT yet validated by a run.
 * The CPU loop leaves the host FluidModel at the mean state of the LAST edge (ComputeSpeciesEve + ComputedPdU are
 * called for every edge, explicit or not); the device stage stores that mean primitive state in st.LastRec so the
 * caller can replay it on the host (CNEMOEulerSolver::Device_CenteredResidual).
 * Ionization is excluded by the caller (CPU MeandPdU is a persistent, edge-order dependent member).
 *
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#ifdef HAVE_KOKKOS

#include "../../../include/solvers/kokkos/CNEMOKokkosData.hpp"

#include <vector>

#include "../../../../Common/include/CConfig.hpp"
#include "../../../../Common/include/geometry/CGeometry.hpp"
#include "../../../include/variables/CNEMOEulerVariable.hpp"

#include "CNEMOKokkosAUSM.hpp"
#include "CNEMOKokkosLAX.hpp"

namespace {

/*! \brief Kernel arguments (plain struct of views, copied by value into the kernels: no implicit this capture). */
struct LaxArgs {
  View2D<unsigned long> edge_nodes;
  View2D<double> edge_normal;
  View1D<unsigned short> nneighbor;
  View1D<unsigned long> dia_ptr;
  View2D<unsigned long> edge_nnz;
  View2D<double> U, V, dPdU;
  View1D<double> Lambda;
  View2D<double> res;
  View1D<double> jac;
  View1D<double> lastMeanV;
  su2kokkos::nemo::NEMOGasTables gas;
  int ns = 0, nVar = 0, nPrimVar = 0;
  bool implicit = false;
  double kappa0 = 0.0;
  unsigned long lastEdge = 0;
};

template <int ND, bool ATOMIC>
KOKKOS_INLINE_FUNCTION void LaxEdge(const LaxArgs& a, const unsigned long iEdge) {
  const int nVar = a.nVar;
  const auto iPoint = a.edge_nodes(iEdge, 0), jPoint = a.edge_nodes(iEdge, 1);
  const double* Normal = &a.edge_normal(iEdge, 0);
  const double zero[MAXNV] = {0.0};
  const double* dPdU_i = a.implicit ? &a.dPdU(iPoint, 0) : zero;
  const double* dPdU_j = a.implicit ? &a.dPdU(jPoint, 0) : zero;

  /*--- Host gas-state replay: mean primitive state of the last edge (CPU calls the gas model for every edge). ---*/
  if (iEdge == a.lastEdge)
    for (int v = 0; v < a.nPrimVar; ++v) a.lastMeanV(v) = 0.5 * (a.V(iPoint, v) + a.V(jPoint, v));

  double Flux[MAXNV];
  double Jacobian_i[MAXNV * MAXNV], Jacobian_j[MAXNV * MAXNV];
  LAXResidual<ND>(a.gas, a.ns, nVar, a.nPrimVar, a.implicit, a.kappa0, Normal, &a.U(iPoint, 0), &a.U(jPoint, 0),
                  &a.V(iPoint, 0), &a.V(jPoint, 0), dPdU_i, dPdU_j, a.Lambda(iPoint), a.Lambda(jPoint),
                  a.nneighbor(iPoint), a.nneighbor(jPoint), Flux, Jacobian_i, Jacobian_j);

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
  for (int iVar = 0; iVar < nVar; iVar++) nemo_edge::Add<ATOMIC>(&a.res(iPoint, iVar), Flux[iVar]);
  for (int iVar = 0; iVar < nVar; iVar++) nemo_edge::Sub<ATOMIC>(&a.res(jPoint, iVar), Flux[iVar]);

  if (!a.implicit) return;

  /*--- Jacobian.UpdateBlocks(iEdge, iPoint, jPoint, Ji, Jj). ---*/
  const unsigned long bs = static_cast<unsigned long>(nVar) * nVar;
  double* bii = &a.jac(a.dia_ptr(iPoint) * bs);
  double* bjj = &a.jac(a.dia_ptr(jPoint) * bs);
  double* bij = &a.jac(a.edge_nnz(iEdge, 0) * bs);
  double* bji = &a.jac(a.edge_nnz(iEdge, 1) * bs);
  for (unsigned long k = 0; k < bs; ++k) {
    nemo_edge::Add<ATOMIC>(&bii[k], Jacobian_i[k]);
    nemo_edge::Add<ATOMIC>(&bij[k], Jacobian_j[k]);
    nemo_edge::Sub<ATOMIC>(&bji[k], Jacobian_i[k]);
    nemo_edge::Sub<ATOMIC>(&bjj[k], Jacobian_j[k]);
  }
}

template <int ND>
void KernelLaxFlux(const LaxArgs& a, unsigned long nEdge, const CNEMOKokkosData::EdgeColoring& col) {
  if (col.Atomic()) {
    Kokkos::parallel_for(
        "NEMO::CenteredLAX", Kokkos::RangePolicy<exec_space>(0, nEdge),
        KOKKOS_LAMBDA(const unsigned long iEdge) { LaxEdge<ND, true>(a, iEdge); });
    return;
  }
  const auto edges = col.edges;
  const unsigned long gs = col.groupSize;
  for (size_t c = 0; c + 1 < col.colorPtr.size(); ++c) {
    const unsigned long first = col.colorPtr[c], n = col.colorPtr[c + 1] - first;
    Kokkos::parallel_for(
        "NEMO::CenteredLAX_color", Kokkos::RangePolicy<exec_space>(0, (n + gs - 1) / gs),
        KOKKOS_LAMBDA(const unsigned long g) {
          const unsigned long end = (g + 1) * gs < n ? (g + 1) * gs : n;
          for (unsigned long k = g * gs; k < end; ++k) LaxEdge<ND, false>(a, edges(first + k));
        });
  }
}

}  // namespace

std::vector<double> CNEMOKokkosData::ComputeCenteredResidual(bool implicit, double kappa0, bool uploadAll) {
  auto& P = stagePerf[ST_CENTERED];
  Kokkos::Timer timer;

  if (implicit && (!geo.has_pattern || geo.nnz == 0))
    SU2_MPI::Error("CNEMOKokkosData: implicit centered residual needs the FV sparse pattern.", CURRENT_FUNCTION);
  if (!accOnDevice) SU2_MPI::Error("CNEMOKokkosData: centered stage without a device accumulator.", CURRENT_FUNCTION);
  if (geo.nneighbor.size() != nPoint) SU2_MPI::Error("CNEMOKokkosData: nneighbor not uploaded.", CURRENT_FUNCTION);
  if (st.LastRec.size() == 0) st.LastRec = View1D<double>("NEMO::LastRec", nPrimVar);

  /*--- Inputs: Primitive + secondaries + Solution (normally resident from SetPrimVar). ---*/
  EnsureInputsOnDevice(uploadAll, true, false);

  /*--- Lambda: domain rows resident from ComputeMaxEigenvalue, halo rows from the host MAX_EIGENVALUE comms. ---*/
  if (st.Lambda.size() == 0) st.Lambda = View1D<double>("NEMO::Lambda", nPoint);
  {
    const unsigned long first = (uploadAll || !lambdaValid) ? 0ul : geo.nPointDomain;
    const unsigned long count = nPoint - first;
    if (count) {
      Kokkos::deep_copy(Kokkos::View<double*, mem_space, Unmanaged>(st.Lambda.data() + first, count),
                        Kokkos::View<const double*, HostSpace, Unmanaged>(nodes->Lambda.data() + first, count));
      P.bytes_h2d += double(count) * sizeof(double);
    }
  }
  Kokkos::fence();
  P.h2d += timer.seconds();
  timer.reset();

  LaxArgs a;
  a.edge_nodes = geo.edge_nodes;
  a.edge_normal = geo.edge_normal;
  a.nneighbor = geo.nneighbor;
  a.dia_ptr = geo.dia_ptr;
  a.edge_nnz = geo.edge_nnz;
  a.U = st.Solution;
  a.V = st.Primitive;
  a.dPdU = st.dPdU;
  a.Lambda = st.Lambda;
  a.res = st.Res;
  a.jac = st.JacVal;
  a.lastMeanV = st.LastRec;
  a.gas = gas;
  a.ns = nSpecies;
  a.nVar = nVar;
  a.nPrimVar = nPrimVar;
  a.implicit = implicit;
  a.kappa0 = kappa0;
  const unsigned long ne = geo.nEdge;
  a.lastEdge = ne ? ne - 1 : 0;

  if (nDim == 2) KernelLaxFlux<2>(a, ne, edgeColoring);
  else KernelLaxFlux<3>(a, ne, edgeColoring);
  Kokkos::fence();
  P.kernel += timer.seconds();
  timer.reset();

  /*--- Residual / Jacobian stay in the device accumulator; only the replay state (nPrimVar doubles) comes back. ---*/
  std::vector<double> lastMeanV;
  if (ne) {
    auto hr = Kokkos::create_mirror_view_and_copy(HostSpace(), st.LastRec);
    lastMeanV.assign(hr.data(), hr.data() + nPrimVar);
    P.bytes_d2h += double(nPrimVar) * sizeof(double);
  }
  P.d2h += timer.seconds();
  P.calls++;
  return lastMeanV;
}

#endif  // HAVE_KOKKOS
