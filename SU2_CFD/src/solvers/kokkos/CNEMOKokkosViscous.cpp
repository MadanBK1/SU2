/*!
 * \file CNEMOKokkosViscous.cpp
 * \brief Device (Kokkos) NEMO Navier-Stokes viscous residual (Step 8 of NEMO_KOKKOS_PORT_PLAN.md):
 *        CNEMONSSolver::Viscous_Residual with CAvgGradCorrected_NEMO (the MESH_0 viscous numerics) per edge,
 *        CheckResidualNaNs, LinSysRes.SubtractBlock(i) / AddBlock(j) and Jacobian.UpdateBlocks(iEdge, i, j, J_i, J_j)
 *        into the device residual accumulator with atomics (Serial: edge order = CPU order -> bitwise in strict IEEE
 *        builds; GPUs: order-nondeterministic at ULP level, TODO edge colouring as for the upwind kernel).
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#ifdef HAVE_KOKKOS

#include "../../../include/solvers/kokkos/CNEMOKokkosData.hpp"

#include "../../../../Common/include/CConfig.hpp"
#include "../../../include/variables/CNEMOEulerVariable.hpp"
#include "../../../include/variables/CNEMONSVariable.hpp"

#include "CNEMOKokkosViscous.hpp"

namespace {

using exec_space = CNEMOKokkosData::exec_space;
template <class T>
using View1D = CNEMOKokkosData::View1D<T>;
template <class T>
using View2D = CNEMOKokkosData::View2D<T>;
template <class T>
using View3D = CNEMOKokkosData::View3D<T>;

struct EdgeArgs {
  nemo_visc::ViscConst c;
  View2D<unsigned long> edge_nodes, edge_nnz;
  View2D<double> edge_normal, coord, V, eves, Cvves, dTdU, dTvedU, Ds, res;
  View3D<double> grad;
  View1D<double> mu, k, kve, jac;
  View1D<unsigned long> dia_ptr;
  int eddyIdx = 0;
};

template <int ND>
void KernelViscousEdges(const EdgeArgs& a, unsigned long nEdge) {
  Kokkos::parallel_for(
      "NEMO::ViscousAvgGradCorrected", Kokkos::RangePolicy<exec_space>(0, nEdge), KOKKOS_LAMBDA(const unsigned long iEdge) {
        using namespace nemo_visc;
        const int nVar = a.c.nVar;
        const bool implicit = a.c.implicit;
        const auto iPoint = a.edge_nodes(iEdge, 0), jPoint = a.edge_nodes(iEdge, 1);
        const double zero[MAXV] = {0.0};

        auto point = [&](unsigned long p) {
          ViscPoint q;
          q.V = &a.V(p, 0);
          q.G = &a.grad(p, 0, 0);
          q.eve = &a.eves(p, 0);
          q.cvve = &a.Cvves(p, 0);
          q.dTdU = implicit ? &a.dTdU(p, 0) : zero;
          q.dTvedU = implicit ? &a.dTvedU(p, 0) : zero;
          q.Ds = &a.Ds(p, 0);
          q.mu = a.mu(p);
          q.eddy = a.V(p, a.eddyIdx);
          q.k = a.k(p);
          q.kve = a.kve(p);
          return q;
        };
        const ViscPoint pi = point(iPoint), pj = point(jPoint);

        double Flux[MAXV], Jac_i[MAXV * MAXV], Jac_j[MAXV * MAXV];
        AvgGradResidual<ND>(a.c, true, &a.coord(iPoint, 0), &a.coord(jPoint, 0), &a.edge_normal(iEdge, 0), pi, pj, Flux,
                            Jac_i, Jac_j);

        if (ResidualNaN2(implicit, nVar, Flux, Jac_i, Jac_j)) return;

        /*--- LinSysRes.SubtractBlock(iPoint), AddBlock(jPoint). ---*/
        for (int iVar = 0; iVar < nVar; iVar++) Kokkos::atomic_sub(&a.res(iPoint, iVar), Flux[iVar]);
        for (int iVar = 0; iVar < nVar; iVar++) Kokkos::atomic_add(&a.res(jPoint, iVar), Flux[iVar]);

        if (!implicit) return;

        /*--- Jacobian.UpdateBlocks(iEdge, iPoint, jPoint, jacobian_i, jacobian_j). ---*/
        const unsigned long bs = static_cast<unsigned long>(nVar) * nVar;
        double* bii = &a.jac(a.dia_ptr(iPoint) * bs);
        double* bjj = &a.jac(a.dia_ptr(jPoint) * bs);
        double* bij = &a.jac(a.edge_nnz(iEdge, 0) * bs);
        double* bji = &a.jac(a.edge_nnz(iEdge, 1) * bs);
        for (unsigned long q = 0; q < bs; ++q) {
          Kokkos::atomic_add(&bii[q], Jac_i[q]);
          Kokkos::atomic_add(&bij[q], Jac_j[q]);
          Kokkos::atomic_sub(&bji[q], Jac_i[q]);
          Kokkos::atomic_sub(&bjj[q], Jac_j[q]);
        }
      });
}

}  // namespace

void CNEMOKokkosData::ComputeViscousResidual(const ViscousParams& p, bool uploadInputs) {
  auto& P = stagePerf[ST_VISCOUS];
  Kokkos::Timer timer;
  if (!accOnDevice) SU2_MPI::Error("CNEMOKokkosData: viscous stage without a device accumulator.", CURRENT_FUNCTION);
  if (!viscousVars) SU2_MPI::Error("CNEMOKokkosData: viscous stage without NS variables.", CURRENT_FUNCTION);
  EnsureInputsOnDevice(uploadInputs, false, false);
  if (uploadInputs || !gradPrimValid) {
    if (ResidentActive()) SU2_MPI::Error("CNEMOKokkosData: device-resident gradient not valid.", CURRENT_FUNCTION);
    AllocateGradients();
    const double* hg = nodes->GetGradient_Primitive().data();
    Kokkos::deep_copy(st.GradPrim, Kokkos::View<const double***, Kokkos::LayoutRight, Kokkos::HostSpace,
                                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>(hg, nPoint, nPrimVarGrad, nDim));
    P.bytes_h2d += double(st.GradPrim.size()) * sizeof(double);
    gradPrimValid = true;
    if (!separateRecGradient) gradRecValid = true;
  } else if (!ResidentActive() && geo.nPoint > geo.nPointDomain) {
    /*--- Step 10 fix (MPI, host-authoritative mode): the device gradient stage downloads only domain rows and the
     *    host comms fill the halo rows of the host gradient; the edge kernel reads both end points -> refresh the
     *    device halo rows (as the upwind stage does for the reconstruction gradient). Resident mode: the halo rows
     *    were exchanged on the device. ---*/
    const size_t gRow = size_t(nPrimVarGrad) * nDim, first = geo.nPointDomain * gRow;
    const size_t count = (geo.nPoint - geo.nPointDomain) * gRow;
    Kokkos::deep_copy(Kokkos::View<double*, mem_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(st.GradPrim.data() + first, count),
                      Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                          nodes->GetGradient_Primitive().data() + first, count));
    P.bytes_h2d += double(count) * sizeof(double);
  }
  if (uploadInputs || !transportDeviceValid) {
    if (ResidentActive()) SU2_MPI::Error("CNEMOKokkosData: device-resident transport not valid.", CURRENT_FUNCTION);
    UploadTransport();
  }
  Kokkos::fence();
  P.h2d += timer.seconds();
  timer.reset();

  EdgeArgs a;
  a.c.gas = gas;
  a.c.ns = nSpecies;
  a.c.nVar = nVar;
  a.c.nPrimVar = nPrimVar;
  a.c.nPrimVarGrad = nPrimVarGrad;
  a.c.nHeavy = nSpecies - p.nElNumerics;
  a.c.implicit = p.implicit;
  a.c.Ru = 1000.0 * UNIVERSAL_GAS_CONSTANT;
  a.edge_nodes = geo.edge_nodes;
  a.edge_nnz = geo.edge_nnz;
  a.edge_normal = geo.edge_normal;
  a.coord = geo.coord;
  a.dia_ptr = geo.dia_ptr;
  a.V = st.Primitive;
  a.eves = st.eves;
  a.Cvves = st.Cvves;
  a.dTdU = st.dTdU;
  a.dTvedU = st.dTvedU;
  a.Ds = st.DiffCoeff;
  a.grad = st.GradPrim;
  a.mu = st.LamVisc;
  a.k = st.ThermalCond;
  a.kve = st.ThermalCondVe;
  a.res = st.Res;
  a.jac = st.JacVal;
  a.eddyIdx = nSpecies + nDim + 9;

  if (nDim == 2) KernelViscousEdges<2>(a, geo.nEdge);
  else KernelViscousEdges<3>(a, geo.nEdge);
  Kokkos::fence();
  P.kernel += timer.seconds();
  P.calls++;
}

#endif  // HAVE_KOKKOS
