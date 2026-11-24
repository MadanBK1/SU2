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

#include "CNEMOKokkosAUSM.hpp"

namespace {

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

/*! \brief Kernel 3 body: flux + Jacobian of one edge, CheckResidualNaNs, accumulation (atomic, or plain writes when the
 *         caller guarantees that no other concurrent edge touches the same points: edge colouring). */
template <int ND, bool ATOMIC>
KOKKOS_INLINE_FUNCTION void UpwindEdge(const UpwArgs& a, const unsigned long iEdge, const long lastKey) {
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

/*! \brief Kernel 3: all edges (atomics, edge order on Serial), or colour by colour (Step 10 Task B: deterministic). */
template <int ND>
void KernelUpwindFlux(const UpwArgs& a, unsigned long nEdge, long lastKey, const CNEMOKokkosData::EdgeColoring& col) {
  if (col.Atomic()) {
    Kokkos::parallel_for(
        "NEMO::UpwindAUSM", Kokkos::RangePolicy<exec_space>(0, nEdge),
        KOKKOS_LAMBDA(const unsigned long iEdge) { UpwindEdge<ND, true>(a, iEdge, lastKey); });
    return;
  }
  const auto edges = col.edges;
  const unsigned long gs = col.groupSize;
  for (size_t c = 0; c + 1 < col.colorPtr.size(); ++c) {
    const unsigned long first = col.colorPtr[c], n = col.colorPtr[c + 1] - first;
    Kokkos::parallel_for(
        "NEMO::UpwindAUSM_color", Kokkos::RangePolicy<exec_space>(0, (n + gs - 1) / gs),
        KOKKOS_LAMBDA(const unsigned long g) {
          const unsigned long end = (g + 1) * gs < n ? (g + 1) * gs : n;
          for (unsigned long k = g * gs; k < end; ++k) UpwindEdge<ND, false>(a, edges(first + k), lastKey);
        });
  }
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

CNEMOKokkosData::UpwindResult CNEMOKokkosData::ComputeUpwindResidual(const UpwindParams& p, bool uploadAll) {
  auto& P = stagePerf[ST_UPWIND];
  Kokkos::Timer timer;
  UpwindResult out;

  if (p.implicit && (!geo.has_pattern || geo.nnz == 0))
    SU2_MPI::Error("CNEMOKokkosData: implicit upwind needs the FV sparse pattern.", CURRENT_FUNCTION);
  if (!accOnDevice) SU2_MPI::Error("CNEMOKokkosData: upwind stage without a device accumulator.", CURRENT_FUNCTION);
  UploadUpwindStatic();
  if (st.NonPhys.size() == 0) {
    st.NonPhys = View1D<unsigned char>("NEMO::Non_Physical", nPoint);
    st.NonPhysCount = View1D<unsigned short>("NEMO::Non_Physical_Counter", nPoint);
    st.EdgeNonPhys = View2D<unsigned char>("NEMO::EdgeNonPhys", geo.nEdge, 2);
    st.LastRec = View1D<double>("NEMO::LastRec", nPrimVar);
  }

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
    } else if (!ResidentActive()) { /*--- resident: halo rows were staged by the gradient comms ---*/
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
      } else if (!ResidentActive()) {
        uploadRange(st.Limiter.data(), hl, geo.nPointDomain * nPrimVarGrad,
                    (nPoint - geo.nPointDomain) * size_t(nPrimVarGrad));
      }
    }
    /*--- Non_Physical / counter: host authoritative (CVariable), tiny; device authoritative when resident. ---*/
    if (ResidentActive()) {
      EnsureDevice(G_NONPHYS);
    } else {
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
  }
  if (p.scheme == UpwindScheme::AUSMPLUSM) {
    if (ResidentActive() && !uploadAll) {
      EnsureDevice(G_SENSOR); /*--- device SetPressureDiffusionSensor + halo exchange (batch A) ---*/
    } else {
      if (st.Sensor.size() == 0) st.Sensor = View1D<double>("NEMO::Sensor", nPoint);
      uploadRange(st.Sensor.data(), nodes->GetSensor().data(), 0, nPoint);
      sensorValid = true;
    }
  }
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
  if (nDim == 2) KernelUpwindFlux<2>(a, ne, lastKey, edgeColoring);
  else KernelUpwindFlux<3>(a, ne, lastKey, edgeColoring);
  Kokkos::fence();
  P.kernel += timer.seconds();
  timer.reset();

  /*--- The residual / Jacobian stay in the device accumulator (Step 7): downloaded once after the last device stage
   *    (CNEMOEulerSolver::Device_ResidualToHost), or before the next host stage. ---*/
  if (p.muscl) {
    if (ResidentActive()) {
      MarkDeviceWritten(G_NONPHYS);
    } else {
      auto hnp = Kokkos::create_mirror_view_and_copy(HostSpace(), st.NonPhys);
      auto hnc = Kokkos::create_mirror_view_and_copy(HostSpace(), st.NonPhysCount);
      for (unsigned long i = 0; i < nPoint; ++i) {
        nodes->Non_Physical(i) = (hnp(i) != 0);
        nodes->Non_Physical_Counter(i) = hnc(i);
      }
      P.bytes_d2h += double(nPoint) * 3;
    }
    if (lastKey >= 0) {
      auto hr = Kokkos::create_mirror_view_and_copy(HostSpace(), st.LastRec);
      out.haveLastRec = true;
      out.lastRec.assign(hr.data(), hr.data() + nPrimVar);
      P.bytes_d2h += double(nPrimVar) * sizeof(double);
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
