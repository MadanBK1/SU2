/*!
 * \file CNEMOKokkosData.cpp
 * \brief Device (Kokkos) mirror of NEMO geometry/state and the device SetPrimVar (Cons2Prim).
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#ifdef HAVE_KOKKOS

#include "../../../include/solvers/kokkos/CNEMOKokkosData.hpp"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <type_traits>

#include "../../../../Common/include/CConfig.hpp"
#include "../../../../Common/include/geometry/CGeometry.hpp"
#include "../../../../Common/include/toolboxes/allocation_toolbox.hpp"
#include "../../../include/fluid/CSU2TCLib.hpp"
#include "../../../include/variables/CNEMOEulerVariable.hpp"
#include "../../../include/variables/CNEMONSVariable.hpp"

#include <algorithm>
#include <cmath>

static_assert(std::is_same_v<su2double, double>, "CNEMOKokkosData requires a passive (double) build.");

namespace {

using HostSpace = Kokkos::HostSpace;
using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

/*! \brief Unmanaged host view over a contiguous row-major host buffer, matching the device view's shape. */
template <class DView, class T>
auto HostWrap(const DView& d, T* ptr) {
  using value_t = typename DView::non_const_value_type;
  static_assert(std::is_same_v<std::remove_const_t<T>, value_t>);
  if constexpr (DView::rank == 1) {
    return Kokkos::View<value_t*, HostSpace, Unmanaged>(const_cast<value_t*>(ptr), d.extent(0));
  } else {
    return Kokkos::View<value_t**, Kokkos::LayoutRight, HostSpace, Unmanaged>(const_cast<value_t*>(ptr), d.extent(0),
                                                                             d.extent(1));
  }
}

template <class DView, class T>
double HtoD(const DView& d, const T* ptr) {
  if (d.size() == 0) return 0.0;
  Kokkos::deep_copy(d, HostWrap(d, ptr));
  return double(d.span()) * sizeof(typename DView::value_type);
}

template <class DView, class T>
double DtoH(T* ptr, const DView& d) {
  if (d.size() == 0) return 0.0;
  auto h = HostWrap(d, ptr);
  Kokkos::deep_copy(h, d);
  return double(d.span()) * sizeof(typename DView::value_type);
}

template <class Mat>
void CheckShape(const Mat& m, unsigned long rows, unsigned long cols, const char* name) {
  if (static_cast<unsigned long>(m.rows()) != rows || static_cast<unsigned long>(m.cols()) != cols)
    SU2_MPI::Error(std::string("CNEMOKokkosData: unexpected shape of host container ") + name, CURRENT_FUNCTION);
}

/*! \brief Compare a device view with a host buffer bitwise, return the number of differing entries. */
template <class DView, class T>
unsigned long CountMismatch(const DView& d, const T* ref, const char* name) {
  if (d.size() == 0) return 0;
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), d);
  using value_t = typename DView::non_const_value_type;
  const value_t* hp = h.data();
  unsigned long n = 0;
  for (size_t k = 0; k < d.size(); ++k) n += (std::memcmp(&hp[k], &ref[k], sizeof(value_t)) != 0);
  if (n) std::cout << "[KOKKOS_NEMO_SELFCHECK] " << name << ": " << n << " mismatches" << std::endl;
  return n;
}

}  // namespace

CNEMOKokkosData::CNEMOKokkosData(CGeometry* geometry, const CConfig* config, CNEMOEulerVariable* nodes_,
                                 CSU2TCLib* gasModel_, unsigned short iMesh_)
    : nodes(nodes_), gasModel(gasModel_), iMesh(iMesh_) {
  GPUMemoryAllocation::detail::ensure_kokkos_initialized();

  nDim = geometry->GetnDim();
  nSpecies = config->GetnSpecies();
  nVar = static_cast<int>(nodes->nVar);
  nPrimVar = static_cast<int>(nodes->nPrimVar);
  nPoint = geometry->GetnPoint();
  implicit = nodes->implicit;
  monoatomic = nodes->monoatomic;
  Tve_Freestream = monoatomic ? nodes->Tve_Freestream : 0.0;

  if (nVar != nSpecies + nDim + 2 || nPrimVar > su2kokkos::nemo::MAXNPRIMVAR)
    SU2_MPI::Error("CNEMOKokkosData: unexpected NEMO variable sizes.", CURRENT_FUNCTION);

  gas = su2kokkos::nemo::BuildNEMOGasTables(*gasModel);
  if (gas.nSpecies != nSpecies) SU2_MPI::Error("CNEMOKokkosData: gas/config nSpecies mismatch.", CURRENT_FUNCTION);

  UploadGeometry(geometry);

  /*--- State views. ---*/
  st.Solution = View2D<double>("NEMO::Solution", nPoint, nVar);
  st.Solution_Old = View2D<double>("NEMO::Solution_Old", nPoint, nVar);
  st.Primitive = View2D<double>("NEMO::Primitive", nPoint, nPrimVar);
  if (implicit) {
    st.dPdU = View2D<double>("NEMO::dPdU", nPoint, nVar);
    st.dTdU = View2D<double>("NEMO::dTdU", nPoint, nVar);
    st.dTvedU = View2D<double>("NEMO::dTvedU", nPoint, nVar);
  }
  st.eves = View2D<double>("NEMO::eves", nPoint, nSpecies);
  st.Cvves = View2D<double>("NEMO::Cvves", nPoint, nSpecies);
  st.Gamma = View1D<double>("NEMO::Gamma", nPoint);
  st.Status = View1D<unsigned char>("NEMO::Cons2PrimStatus", nPoint);
  st.FirstDP = View2D<double>("NEMO::FirstDensityPressure", nPoint, 2);
  h_Status.assign(nPoint, 0);
  h_FirstDP.assign(2 * nPoint, 0.0);

  /*--- Initial state mirror (not needed by SetPrimVar, which re-uploads, but keeps the device copy valid). ---*/
  UploadConservative();
  UploadPrimitive();
  perf = Perf();

  const char* env = std::getenv("SU2_KOKKOS_SELFCHECK");
  if (env != nullptr && std::atoi(env) != 0) {
    const auto bad = SelfCheck(geometry);
    if (SU2_MPI::GetRank() == MASTER_NODE)
      std::cout << "[KOKKOS_NEMO_SELFCHECK] MG level " << iMesh << ": round-trip "
                << (bad == 0 ? "PASS" : "FAIL") << " (" << bad << " mismatches)" << std::endl;
    if (bad != 0) SU2_MPI::Error("CNEMOKokkosData: device mirror self-check failed.", CURRENT_FUNCTION);
  }

  if (SU2_MPI::GetRank() == MASTER_NODE && iMesh == MESH_0)
    std::cout << "[KOKKOS_NEMO] device Cons2Prim enabled (exec space " << exec_space::name() << ", nPoint " << nPoint
              << ", nEdge " << geo.nEdge << ", nSpecies " << nSpecies << (implicit ? ", implicit" : ", explicit")
              << ")" << std::endl;
}

CNEMOKokkosData::~CNEMOKokkosData() { PrintPerf(); }

bool CNEMOKokkosData::VerifyEnabled() {
  static const bool on = [] {
    const char* env = std::getenv("SU2_KOKKOS_C2P_VERIFY");
    return env != nullptr && std::atoi(env) != 0;
  }();
  return on;
}

void CNEMOKokkosData::Snapshot() {
  if (verifyArrays.empty()) {
    auto add = [&](const char* name, auto& c) { verifyArrays.push_back({name, c.data(), size_t(c.size())}); };
    add("Solution", nodes->Solution);
    add("Primitive", nodes->Primitive);
    if (implicit) {
      add("dPdU", nodes->dPdU);
      add("dTdU", nodes->dTdU);
      add("dTvedU", nodes->dTvedU);
    }
    add("eves", nodes->eves);
    add("Cvves", nodes->Cvves);
    add("Gamma", nodes->Gamma);
    add("Velocity2", nodes->Velocity2);
    if (auto* ns = dynamic_cast<CNEMONSVariable*>(nodes)) {
      add("DiffusionCoeff", ns->DiffusionCoeff);
      add("LaminarViscosity", ns->LaminarViscosity);
      add("ThermalCond", ns->ThermalCond);
      add("ThermalCond_ve", ns->ThermalCond_ve);
      add("Enthalpys", ns->Enthalpys);
    }
    verifyCopies.resize(verifyArrays.size());
  }
  for (size_t k = 0; k < verifyArrays.size(); ++k)
    verifyCopies[k].assign(verifyArrays[k].ptr, verifyArrays[k].ptr + verifyArrays[k].size);
}

void CNEMOKokkosData::SwapSnapshot() {
  for (size_t k = 0; k < verifyArrays.size(); ++k)
    std::swap_ranges(verifyArrays[k].ptr, verifyArrays[k].ptr + verifyArrays[k].size, verifyCopies[k].begin());
}

void CNEMOKokkosData::CompareSnapshot(unsigned long nonPhysDevice, unsigned long nonPhysHost) {
  /*--- Host arrays now hold the CPU results, the copies hold the device results. ---*/
  unsigned long bad = (nonPhysDevice != nonPhysHost);
  double maxRel = 0.0;
  for (size_t k = 0; k < verifyArrays.size(); ++k) {
    const auto& a = verifyArrays[k];
    const auto& d = verifyCopies[k];
    double amax = 0.0, dmax = 0.0;
    unsigned long n = 0;
    for (size_t i = 0; i < a.size; ++i) {
      const double h = a.ptr[i], v = d[i];
      if (std::memcmp(&h, &v, sizeof(double)) != 0) {
        ++n;
        dmax = std::max(dmax, std::fabs(h - v));
      }
      amax = std::max(amax, std::fabs(h));
    }
    const double rel = dmax / std::max(amax, 1e-300);
    if (n && verifyCalls < 3)
      std::cout << "[KOKKOS_NEMO_VERIFY] " << a.name << ": " << n << " non-bitwise entries, max rel " << rel
                << std::endl;
    bad += n;
    maxRel = std::max(maxRel, rel);
  }
  verifyMismatch += bad;
  verifyMaxRel = std::max(verifyMaxRel, maxRel);
  ++verifyCalls;
}

void CNEMOKokkosData::UploadGeometry(CGeometry* geometry) {
  geo.nPoint = geometry->GetnPoint();
  geo.nPointDomain = geometry->GetnPointDomain();
  geo.nEdge = geometry->GetnEdge();
  geo.nDim = geometry->GetnDim();
  geo.nMarker = geometry->GetnMarker();
  const auto np = geo.nPoint, ne = geo.nEdge;
  const int nd = geo.nDim;
  const auto* pts = geometry->nodes;
  const auto* edg = geometry->edges;

  /*--- Edges (host Nodes array may be padded for SIMD, copy element-wise). ---*/
  geo.edge_nodes = View2D<unsigned long>("NEMO::edge_nodes", ne, 2);
  geo.edge_normal = View2D<double>("NEMO::edge_normal", ne, nd);
  {
    auto hn = Kokkos::create_mirror_view(geo.edge_nodes);
    auto hN = Kokkos::create_mirror_view(geo.edge_normal);
    for (auto e = 0ul; e < ne; ++e) {
      hn(e, 0) = edg->GetNode(e, 0);
      hn(e, 1) = edg->GetNode(e, 1);
      const su2double* n = edg->GetNormal(e);
      for (int d = 0; d < nd; ++d) hN(e, d) = n[d];
    }
    Kokkos::deep_copy(geo.edge_nodes, hn);
    Kokkos::deep_copy(geo.edge_normal, hN);
  }

  /*--- Points. ---*/
  geo.coord = View2D<double>("NEMO::coord", np, nd);
  geo.volume = View1D<double>("NEMO::volume", np);
  geo.periodic_volume = View1D<double>("NEMO::periodic_volume", np);
  geo.point_ptr = View1D<unsigned long>("NEMO::point_ptr", np + 1);
  geo.pedge_ptr = View1D<unsigned long>("NEMO::pedge_ptr", np + 1);
  {
    auto hc = Kokkos::create_mirror_view(geo.coord);
    auto hv = Kokkos::create_mirror_view(geo.volume);
    auto hpv = Kokkos::create_mirror_view(geo.periodic_volume);
    auto hptr = Kokkos::create_mirror_view(geo.point_ptr);
    hptr(0) = 0;
    for (auto i = 0ul; i < np; ++i) {
      for (int d = 0; d < nd; ++d) hc(i, d) = pts->GetCoord(i, d);
      hv(i) = pts->GetVolume(i);
      hpv(i) = pts->GetPeriodicVolume(i);
      hptr(i + 1) = hptr(i) + pts->GetnPoint(i);
    }
    const auto nnb = hptr(np);
    geo.point_idx = View1D<unsigned long>("NEMO::point_idx", nnb);
    geo.pedge_idx = View1D<long>("NEMO::pedge_idx", nnb);
    auto hidx = Kokkos::create_mirror_view(geo.point_idx);
    auto heidx = Kokkos::create_mirror_view(geo.pedge_idx);
    for (auto i = 0ul; i < np; ++i) {
      for (auto k = 0ul; k < pts->GetnPoint(i); ++k) {
        hidx(hptr(i) + k) = pts->GetPoint(i, k);
        heidx(hptr(i) + k) = pts->GetEdge(i, k);
      }
    }
    Kokkos::deep_copy(geo.coord, hc);
    Kokkos::deep_copy(geo.volume, hv);
    Kokkos::deep_copy(geo.periodic_volume, hpv);
    Kokkos::deep_copy(geo.point_ptr, hptr);
    Kokkos::deep_copy(geo.pedge_ptr, hptr); /*--- Point and Edge CSR share the outer pointers. ---*/
    Kokkos::deep_copy(geo.point_idx, hidx);
    Kokkos::deep_copy(geo.pedge_idx, heidx);
  }

  /*--- Boundary vertices, flattened in a CSR-by-marker layout. ---*/
  geo.marker_ptr = View1D<unsigned long>("NEMO::marker_ptr", geo.nMarker + 1);
  {
    auto hm = Kokkos::create_mirror_view(geo.marker_ptr);
    hm(0) = 0;
    for (int m = 0; m < geo.nMarker; ++m) hm(m + 1) = hm(m) + geometry->GetnVertex(m);
    const auto nv = hm(geo.nMarker);
    geo.vtx_node = View1D<unsigned long>("NEMO::vtx_node", nv);
    geo.vtx_normal = View2D<double>("NEMO::vtx_normal", nv, nd);
    geo.vtx_normal_neighbor = View1D<long>("NEMO::vtx_normal_neighbor", nv);
    auto hn = Kokkos::create_mirror_view(geo.vtx_node);
    auto hN = Kokkos::create_mirror_view(geo.vtx_normal);
    auto hnn = Kokkos::create_mirror_view(geo.vtx_normal_neighbor);
    for (int m = 0; m < geo.nMarker; ++m) {
      for (auto v = 0ul; v < geometry->GetnVertex(m); ++v) {
        const auto k = hm(m) + v;
        const auto* vtx = geometry->vertex[m][v];
        hn(k) = vtx->GetNode();
        for (int d = 0; d < nd; ++d) hN(k, d) = vtx->GetNormal(d);
        hnn(k) = static_cast<long>(vtx->GetNormal_Neighbor());
      }
    }
    Kokkos::deep_copy(geo.marker_ptr, hm);
    Kokkos::deep_copy(geo.vtx_node, hn);
    Kokkos::deep_copy(geo.vtx_normal, hN);
    Kokkos::deep_copy(geo.vtx_normal_neighbor, hnn);
  }

  /*--- FV sparse pattern (fill 0) + edge -> non-zero map. Built lazily on the host if absent
   *    (explicit schemes do not build it otherwise). ---*/
  geo.has_pattern = (ne > 0);
  if (geo.has_pattern) {
    const auto& pattern = geometry->GetSparsePattern(ConnectivityType::FiniteVolume, 0);
    const auto& emap = geometry->GetEdgeToSparsePatternMap();
    const auto nrow = pattern.getOuterSize();
    const auto nnz = pattern.getNumNonZeros();
    if (nrow != np) SU2_MPI::Error("CNEMOKokkosData: sparse pattern rows != nPoint.", CURRENT_FUNCTION);
    geo.row_ptr = View1D<unsigned long>("NEMO::row_ptr", nrow + 1);
    geo.col_ind = View1D<unsigned long>("NEMO::col_ind", nnz);
    geo.dia_ptr = View1D<unsigned long>("NEMO::dia_ptr", nrow);
    geo.edge_nnz = View2D<unsigned long>("NEMO::edge_nnz", ne, 2);
    HtoD(geo.row_ptr, pattern.outerPtr());
    HtoD(geo.col_ind, pattern.innerIdx());
    HtoD(geo.dia_ptr, pattern.diagPtr());
    auto he = Kokkos::create_mirror_view(geo.edge_nnz);
    for (auto e = 0ul; e < ne; ++e) {
      he(e, 0) = emap(e, 0);
      he(e, 1) = emap(e, 1);
    }
    Kokkos::deep_copy(geo.edge_nnz, he);
  }
}

void CNEMOKokkosData::UploadConservative() {
  CheckShape(nodes->Solution, nPoint, nVar, "Solution");
  CheckShape(nodes->Solution_Old, nPoint, nVar, "Solution_Old");
  perf.bytes_h2d += HtoD(st.Solution, nodes->Solution.data());
  perf.bytes_h2d += HtoD(st.Solution_Old, nodes->Solution_Old.data());
}

void CNEMOKokkosData::UploadPrimitive() {
  CheckShape(nodes->Primitive, nPoint, nPrimVar, "Primitive");
  CheckShape(nodes->eves, nPoint, nSpecies, "eves");
  CheckShape(nodes->Cvves, nPoint, nSpecies, "Cvves");
  perf.bytes_h2d += HtoD(st.Primitive, nodes->Primitive.data());
  if (implicit) {
    CheckShape(nodes->dPdU, nPoint, nVar, "dPdU");
    CheckShape(nodes->dTdU, nPoint, nVar, "dTdU");
    CheckShape(nodes->dTvedU, nPoint, nVar, "dTvedU");
    perf.bytes_h2d += HtoD(st.dPdU, nodes->dPdU.data());
    perf.bytes_h2d += HtoD(st.dTdU, nodes->dTdU.data());
    perf.bytes_h2d += HtoD(st.dTvedU, nodes->dTvedU.data());
  }
  perf.bytes_h2d += HtoD(st.eves, nodes->eves.data());
  perf.bytes_h2d += HtoD(st.Cvves, nodes->Cvves.data());
  /*--- Gamma is write-only in Cons2Prim; uploaded only to keep the mirror complete. ---*/
  perf.bytes_h2d += HtoD(st.Gamma, nodes->Gamma.data());
}

void CNEMOKokkosData::DownloadState() {
  perf.bytes_d2h += DtoH(nodes->Solution.data(), st.Solution);
  perf.bytes_d2h += DtoH(nodes->Primitive.data(), st.Primitive);
  if (implicit) {
    perf.bytes_d2h += DtoH(nodes->dPdU.data(), st.dPdU);
    perf.bytes_d2h += DtoH(nodes->dTdU.data(), st.dTdU);
    perf.bytes_d2h += DtoH(nodes->dTvedU.data(), st.dTvedU);
  }
  perf.bytes_d2h += DtoH(nodes->eves.data(), st.eves);
  perf.bytes_d2h += DtoH(nodes->Cvves.data(), st.Cvves);
  perf.bytes_d2h += DtoH(nodes->Gamma.data(), st.Gamma);
  perf.bytes_d2h += DtoH(h_Status.data(), st.Status);
  perf.bytes_d2h += DtoH(h_FirstDP.data(), st.FirstDP);
}

unsigned long CNEMOKokkosData::SetPrimVar() {
  Kokkos::Timer timer;

  /*--- HtoD (deep_copy from host memory is synchronous with respect to the host). ---*/
  UploadConservative();
  UploadPrimitive();
  perf.h2d += timer.seconds();
  timer.reset();

  /*--- Kernel: copy everything the lambda needs to locals (no implicit capture of this). ---*/
  const auto g = gas;
  const int ns = nSpecies, nd = nDim, nv = nVar;
  const bool impl = implicit, mono = monoatomic;
  const double TveFS = Tve_Freestream;
  const auto U = st.Solution, Uold = st.Solution_Old, V = st.Primitive;
  const auto dPdU = st.dPdU, dTdU = st.dTdU, dTvedU = st.dTvedU, eves = st.eves, Cvves = st.Cvves;
  const auto Gamma = st.Gamma;
  const auto Status = st.Status;
  const auto FirstDP = st.FirstDP;

  unsigned long nonPhysical = 0;
  Kokkos::parallel_reduce(
      "NEMO::SetPrimVar", Kokkos::RangePolicy<exec_space>(0, nPoint),
      KOKKOS_LAMBDA(const unsigned long i, unsigned long& count) {
        using namespace su2kokkos::nemo;
        double* Ui = &U(i, 0);
        double* Vi = &V(i, 0);
        double* dP = impl ? &dPdU(i, 0) : nullptr;
        double* dT = impl ? &dTdU(i, 0) : nullptr;
        double* dTve = impl ? &dTvedU(i, 0) : nullptr;
        double* ev = &eves(i, 0);
        double* cv = &Cvves(i, 0);
        double gam;

        /*--- Same sequence as CNEMOEulerVariable::SetPrimVar. ---*/
        const bool nonPhys = Cons2Prim(g, nd, impl, mono, TveFS, Ui, Vi, dP, dT, dTve, ev, cv, gam);
        unsigned char status = 0;
        const bool firstFull = Cons2PrimReachedFullPath(ns, mono, Vi);
        if (firstFull) status |= C2P_FIRST_FULL;

        bool finalFull = firstFull;
        if (nonPhys) {
          status |= C2P_FIRST_NONPHYS;
          if (firstFull) {
            /*--- Gas Density/Pressure members as SetTDStateRhosTTv left them (for the host replay). ---*/
            double rho = 0.0;
            for (int s = 0; s < ns; s++) rho += Vi[s];
            FirstDP(i, 0) = rho;
            FirstDP(i, 1) = ComputePressure(g, Vi, Vi[ns], Vi[ns + 1]);
          }
          for (int iVar = 0; iVar < nv; iVar++) Ui[iVar] = Uold(i, iVar);
          Cons2Prim(g, nd, impl, mono, TveFS, Ui, Vi, dP, dT, dTve, ev, cv, gam);
          finalFull = Cons2PrimReachedFullPath(ns, mono, Vi);
        }
        if (finalFull) status |= C2P_FINAL_FULL;

        Gamma(i) = gam;
        Status(i) = status;
        count += nonPhys ? 1 : 0;
      },
      nonPhysical);
  /*--- parallel_reduce into a host scalar blocks until completion: no extra fence needed for the timer. ---*/
  perf.kernel += timer.seconds();
  timer.reset();

  DownloadState();
  perf.d2h += timer.seconds();
  perf.calls++;

  return nonPhysical;
}

void CNEMOKokkosData::ReplayGasState(unsigned long iPoint, bool full) const {
  su2kokkos::nemo::ReplayHostGasState(*gasModel, nDim, nodes->Primitive[iPoint], h_Status[iPoint],
                                      &h_FirstDP[2 * iPoint], full);
}

unsigned long CNEMOKokkosData::SelfCheck(CGeometry* geometry) const {
  unsigned long bad = 0;
  const auto np = geo.nPoint, ne = geo.nEdge;
  const int nd = geo.nDim;
  const auto* pts = geometry->nodes;
  const auto* edg = geometry->edges;

  /*--- Geometry: rebuild reference host arrays independently of UploadGeometry. ---*/
  {
    std::vector<unsigned long> en(2 * ne);
    std::vector<double> nrm(ne * nd);
    for (auto e = 0ul; e < ne; ++e) {
      en[2 * e] = edg->GetNode(e, 0);
      en[2 * e + 1] = edg->GetNode(e, 1);
      for (int d = 0; d < nd; ++d) nrm[e * nd + d] = edg->GetNormal(e)[d];
    }
    bad += CountMismatch(geo.edge_nodes, en.data(), "edge_nodes");
    bad += CountMismatch(geo.edge_normal, nrm.data(), "edge_normal");
  }
  {
    std::vector<double> c(np * nd), v(np), pv(np);
    std::vector<unsigned long> ptr(np + 1, 0), idx;
    std::vector<long> eidx;
    for (auto i = 0ul; i < np; ++i) {
      for (int d = 0; d < nd; ++d) c[i * nd + d] = pts->GetCoord(i, d);
      v[i] = pts->GetVolume(i);
      pv[i] = pts->GetPeriodicVolume(i);
      for (auto k = 0ul; k < pts->GetnPoint(i); ++k) {
        idx.push_back(pts->GetPoint(i, k));
        eidx.push_back(pts->GetEdge(i, k));
      }
      ptr[i + 1] = idx.size();
    }
    bad += CountMismatch(geo.coord, c.data(), "coord");
    bad += CountMismatch(geo.volume, v.data(), "volume");
    bad += CountMismatch(geo.periodic_volume, pv.data(), "periodic_volume");
    bad += CountMismatch(geo.point_ptr, ptr.data(), "point_ptr");
    bad += CountMismatch(geo.pedge_ptr, ptr.data(), "pedge_ptr");
    bad += CountMismatch(geo.point_idx, idx.data(), "point_idx");
    bad += CountMismatch(geo.pedge_idx, eidx.data(), "pedge_idx");
    /*--- Consistency: edge k of point i connects i and its k-th neighbour. ---*/
    for (auto i = 0ul; i < np; ++i)
      for (auto k = ptr[i]; k < ptr[i + 1]; ++k) {
        const auto e = static_cast<unsigned long>(eidx[k]);
        const auto a = edg->GetNode(e, 0), b = edg->GetNode(e, 1);
        if (!((a == i && b == idx[k]) || (b == i && a == idx[k]))) ++bad;
      }
  }
  {
    std::vector<unsigned long> mptr(geo.nMarker + 1, 0), vn;
    std::vector<double> vN;
    std::vector<long> vnn;
    for (int m = 0; m < geo.nMarker; ++m) {
      for (auto k = 0ul; k < geometry->GetnVertex(m); ++k) {
        const auto* vtx = geometry->vertex[m][k];
        vn.push_back(vtx->GetNode());
        for (int d = 0; d < nd; ++d) vN.push_back(vtx->GetNormal(d));
        vnn.push_back(static_cast<long>(vtx->GetNormal_Neighbor()));
      }
      mptr[m + 1] = vn.size();
    }
    bad += CountMismatch(geo.marker_ptr, mptr.data(), "marker_ptr");
    bad += CountMismatch(geo.vtx_node, vn.data(), "vtx_node");
    bad += CountMismatch(geo.vtx_normal, vN.data(), "vtx_normal");
    bad += CountMismatch(geo.vtx_normal_neighbor, vnn.data(), "vtx_normal_neighbor");
  }
  if (geo.has_pattern) {
    const auto& pattern = geometry->GetSparsePattern(ConnectivityType::FiniteVolume, 0);
    const auto& emap = geometry->GetEdgeToSparsePatternMap();
    bad += CountMismatch(geo.row_ptr, pattern.outerPtr(), "row_ptr");
    bad += CountMismatch(geo.col_ind, pattern.innerIdx(), "col_ind");
    bad += CountMismatch(geo.dia_ptr, pattern.diagPtr(), "dia_ptr");
    std::vector<unsigned long> em(2 * ne);
    for (auto e = 0ul; e < ne; ++e) {
      em[2 * e] = emap(e, 0);
      em[2 * e + 1] = emap(e, 1);
      /*--- Consistency: (ij) is in row i at column j, (ji) in row j at column i. ---*/
      const auto i = edg->GetNode(e, 0), j = edg->GetNode(e, 1);
      if (pattern.innerIdx()[em[2 * e]] != j || pattern.innerIdx()[em[2 * e + 1]] != i) ++bad;
      if (em[2 * e] < pattern.outerPtr()[i] || em[2 * e] >= pattern.outerPtr()[i + 1]) ++bad;
    }
    bad += CountMismatch(geo.edge_nnz, em.data(), "edge_nnz");
  }

  /*--- State. ---*/
  bad += CountMismatch(st.Solution, nodes->Solution.data(), "Solution");
  bad += CountMismatch(st.Solution_Old, nodes->Solution_Old.data(), "Solution_Old");
  bad += CountMismatch(st.Primitive, nodes->Primitive.data(), "Primitive");
  if (implicit) {
    bad += CountMismatch(st.dPdU, nodes->dPdU.data(), "dPdU");
    bad += CountMismatch(st.dTdU, nodes->dTdU.data(), "dTdU");
    bad += CountMismatch(st.dTvedU, nodes->dTvedU.data(), "dTvedU");
  }
  bad += CountMismatch(st.eves, nodes->eves.data(), "eves");
  bad += CountMismatch(st.Cvves, nodes->Cvves.data(), "Cvves");
  bad += CountMismatch(st.Gamma, nodes->Gamma.data(), "Gamma");

  /*--- Gas tables: scalar consistency with the host model (values validated by the Step-0 driver). ---*/
  if (gas.nSpecies != nSpecies || gas.nDim != nDim) ++bad;
  return bad;
}

void CNEMOKokkosData::PrintPerf() const {
  if (SU2_MPI::GetRank() != MASTER_NODE || perf.calls == 0) return;
  const double c = double(perf.calls);
  std::cout << std::scientific << std::setprecision(3) << "[KOKKOS_NEMO_PERF] MG " << iMesh
            << " SetPrimVar calls=" << perf.calls << " nPoint=" << nPoint << " | h2d " << perf.h2d << " s ("
            << perf.h2d / c << "/call, " << perf.bytes_h2d / c / 1e6 << " MB/call)"
            << " | kernel " << perf.kernel << " s (" << perf.kernel / c << "/call)"
            << " | d2h " << perf.d2h << " s (" << perf.d2h / c << "/call, " << perf.bytes_d2h / c / 1e6
            << " MB/call)"
            << " | host post " << perf.host_post << " s (" << perf.host_post / c << "/call)" << std::endl;
  if (verifyCalls > 0)
    std::cout << "[KOKKOS_NEMO_VERIFY] MG " << iMesh << " calls=" << verifyCalls
              << " non-bitwise entries=" << verifyMismatch << " max rel diff=" << verifyMaxRel << std::endl;
  std::cout << std::defaultfloat;
}

#endif  // HAVE_KOKKOS
