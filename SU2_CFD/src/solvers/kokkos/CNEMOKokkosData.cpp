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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>

#include "../../../../Common/include/CConfig.hpp"
#include "../../../../Common/include/geometry/CGeometry.hpp"
#include "../../../../Common/include/toolboxes/allocation_toolbox.hpp"
#include "../../../../Common/include/toolboxes/graph_toolbox.hpp"
#include "../../../../Common/include/linear_algebra/CSysVector.hpp"
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
  nPrimVarGrad = static_cast<int>(nodes->nPrimVarGrad);
  nPoint = geometry->GetnPoint();
  viscousVars = (dynamic_cast<CNEMONSVariable*>(nodes) != nullptr);
  separateRecGradient = (&nodes->GetGradient_Reconstruction() != &nodes->GetGradient_Primitive());
  implicit = nodes->implicit;
  monoatomic = nodes->monoatomic;
  Tve_Freestream = monoatomic ? nodes->Tve_Freestream : 0.0;

  if (nVar != nSpecies + nDim + 2 || nPrimVar > su2kokkos::nemo::MAXNPRIMVAR || nPrimVarGrad > nPrimVar)
    SU2_MPI::Error("CNEMOKokkosData: unexpected NEMO variable sizes.", CURRENT_FUNCTION);

  gas = su2kokkos::nemo::BuildNEMOGasTables(*gasModel);
  if (gas.nSpecies != nSpecies) SU2_MPI::Error("CNEMOKokkosData: gas/config nSpecies mismatch.", CURRENT_FUNCTION);

  UploadGeometry(geometry);
  UploadBoundaryInfo(geometry, config);
  BuildEdgeColoring(geometry, config);
  /*--- 6b.a: grid velocity / Volume_nM1 views and geo.dynamic (refreshed by the solver every time step). ---*/
  if (config->GetDynamic_Grid() && iMesh == MESH_0) RefreshDynamicGeometry(geometry);

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

  /*--- Step 8: NS transport properties on the device (viscous CSU2TCLib, supported model). ---*/
  if (viscousVars) {
    st.LamVisc = View1D<double>("NEMO::LaminarViscosity", nPoint);
    st.ThermalCond = View1D<double>("NEMO::ThermalCond", nPoint);
    st.ThermalCondVe = View1D<double>("NEMO::ThermalCond_ve", nPoint);
    st.DiffCoeff = View2D<double>("NEMO::DiffusionCoeff", nPoint, nSpecies);
    st.Enthalpys = View2D<double>("NEMO::Enthalpys", nPoint, nSpecies);
    deviceTransport = (gas.transModel != su2kokkos::nemo::NEMOGasTables::TRANSPORT_NONE) && !HostForced("transport");
    if (deviceTransport) st.TransDP = View2D<double>("NEMO::TransportDensityPressure", nPoint, 2);
  }

  /*--- Initial state mirror (not needed by SetPrimVar, which re-uploads, but keeps the device copy valid). ---*/
  UploadConservative();
  UploadPrimitive();
  primDeviceValid = true;
  secDeviceValid = true;
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
              << ")" << std::endl
              << "[KOKKOS_NEMO] edge assembly (upwind, viscous): " << EdgeColoringReport() << std::endl;
}

CNEMOKokkosData::~CNEMOKokkosData() { PrintPerf(); }

void CNEMOKokkosData::BuildEdgeColoring(CGeometry* geometry, const CConfig* config) {
  using nemo_edge::EdgeMode;
  auto& C = edgeColoring;
  const char* env = std::getenv("SU2_KOKKOS_NEMO_EDGE");
  const std::string mode = env ? env : "";
  C.mode = (mode == "color") ? EdgeMode::COLOR : (mode == "level") ? EdgeMode::LEVEL : EdgeMode::ATOMIC;
  if (!mode.empty() && mode != "color" && mode != "level" && mode != "atomic")
    SU2_MPI::Error("SU2_KOKKOS_NEMO_EDGE must be atomic, color or level.", CURRENT_FUNCTION);
  if (C.Atomic()) return;

  Kokkos::Timer timer;
  const unsigned long nEdge = geometry->GetnEdge();
  constexpr bool hostBackend = Kokkos::SpaceAccessibility<Kokkos::HostSpace, mem_space>::accessible;
  C.groupSize = hostBackend ? std::max<unsigned long>(1ul, config->GetEdgeColoringGroupSize()) : 1ul;
  if (const char* g = std::getenv("SU2_KOKKOS_NEMO_EDGE_GROUP")) C.groupSize = std::max(1l, std::atol(g));
  std::vector<unsigned long> list;

  if (C.mode == EdgeMode::LEVEL) {
    /*--- Order-preserving levels: every point sees its edges in increasing edge order across the levels. ---*/
    C.groupSize = 1;
    std::vector<unsigned long> lastLevel(geometry->GetnPoint(), 0), level(nEdge);
    unsigned long nLevel = 0;
    for (unsigned long e = 0; e < nEdge; ++e) {
      const auto i = geometry->edges->GetNode(e, 0), j = geometry->edges->GetNode(e, 1);
      const auto l = std::max(lastLevel[i], lastLevel[j]) + 1;
      lastLevel[i] = lastLevel[j] = l;
      level[e] = l - 1;
      nLevel = std::max(nLevel, l);
    }
    C.colorPtr.assign(nLevel + 1, 0);
    for (unsigned long e = 0; e < nEdge; ++e) C.colorPtr[level[e] + 1]++;
    for (unsigned long l = 0; l < nLevel; ++l) C.colorPtr[l + 1] += C.colorPtr[l];
    list.resize(nEdge);
    std::vector<unsigned long> fill(C.colorPtr.begin(), C.colorPtr.end() - 1);
    for (unsigned long e = 0; e < nEdge; ++e) list[fill[level[e]]++] = e;
  } else {
    /*--- Balanced colouring of the edge -> node pattern (CGeometry::GetEdgeColoring without the OpenMP shortcut);
     *    a failed colouring (> 255 colours) halves the group size. ---*/
    su2vector<unsigned long> outerPtr(nEdge + 1), innerIdx(nEdge * 2);
    for (unsigned long e = 0; e < nEdge; ++e) {
      outerPtr(e) = 2 * e;
      innerIdx(2 * e) = geometry->edges->GetNode(e, 0);
      innerIdx(2 * e + 1) = geometry->edges->GetNode(e, 1);
    }
    outerPtr(nEdge) = 2 * nEdge;
    CCompressedSparsePatternUL pattern(std::move(outerPtr), std::move(innerIdx));
    CCompressedSparsePatternUL coloring;
    while (true) {
      coloring = colorSparsePattern(pattern, C.groupSize, false, true);
      if (!coloring.empty() || C.groupSize == 1) break;
      C.groupSize = std::max(1ul, C.groupSize / 2);
    }
    if (coloring.empty()) SU2_MPI::Error("CNEMOKokkosData: edge colouring failed.", CURRENT_FUNCTION);
    const auto nColor = coloring.getOuterSize();
    C.colorPtr.assign(nColor + 1, 0);
    list.reserve(nEdge);
    for (unsigned long c = 0; c < nColor; ++c) {
      for (const auto e : coloring.getInnerIter(c)) list.push_back(e);
      C.colorPtr[c + 1] = list.size();
    }
    C.efficiency = coloringEfficiency(coloring, std::max(1, int(exec_space().concurrency())), int(C.groupSize));
    C.efficiencyGPU = coloringEfficiency(coloring, 1 << 16, int(C.groupSize));
  }
  if (list.size() != nEdge) SU2_MPI::Error("CNEMOKokkosData: edge colouring lost edges.", CURRENT_FUNCTION);

  /*--- Check: the edges of a colour (whole groups) never share a point across groups. ---*/
  std::vector<long> owner(geometry->GetnPoint(), -1);
  C.minSize = nEdge;
  C.maxSize = 0;
  for (unsigned long c = 0; c + 1 < C.colorPtr.size(); ++c) {
    const auto n = C.colorPtr[c + 1] - C.colorPtr[c];
    C.minSize = std::min(C.minSize, n);
    C.maxSize = std::max(C.maxSize, n);
    for (unsigned long k = 0; k < n; ++k) {
      const long grp = long(C.colorPtr[c] + (k / C.groupSize) * C.groupSize);
      const auto e = list[C.colorPtr[c] + k];
      for (int side = 0; side < 2; ++side) {
        auto& o = owner[geometry->edges->GetNode(e, side)];
        if (o >= long(C.colorPtr[c]) && o != grp)
          SU2_MPI::Error("CNEMOKokkosData: invalid edge colouring (point shared within a colour).", CURRENT_FUNCTION);
        o = grp;
      }
    }
  }
  if (C.mode == EdgeMode::LEVEL) {
    /*--- coloringEfficiency for the level pattern (ideal / real work with unit chunks). ---*/
    auto eff = [&](double nThreads) {
      double real = 0;
      for (unsigned long c = 0; c + 1 < C.colorPtr.size(); ++c)
        real += std::ceil(double(C.colorPtr[c + 1] - C.colorPtr[c]) / nThreads);
      return (double(nEdge) / nThreads) / std::max(1.0, real);
    };
    C.efficiency = eff(std::max(1, int(exec_space().concurrency())));
    C.efficiencyGPU = eff(double(1 << 16));
  }
  C.edges = View1D<unsigned long>("NEMO::coloredEdges", nEdge);
  Kokkos::deep_copy(C.edges, Kokkos::View<const unsigned long*, Kokkos::HostSpace,
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>(list.data(), nEdge));
  C.buildSeconds = timer.seconds();
}

std::string CNEMOKokkosData::EdgeColoringReport() const {
  const auto& C = edgeColoring;
  std::ostringstream os;
  if (C.Atomic()) {
    os << "atomic (Kokkos atomics; SU2_KOKKOS_NEMO_EDGE=color|level for the deterministic coloured assembly)";
    return os.str();
  }
  const double mean = C.NumColors() ? double(geo.nEdge) / C.NumColors() : 0.0;
  os << (C.mode == nemo_edge::EdgeMode::COLOR ? "color" : "level (order-preserving)") << ": " << C.NumColors()
     << " colours, group size " << C.groupSize << ", edges per colour min/mean/max " << C.minSize << "/" << mean << "/"
     << C.maxSize << ", efficiency " << C.efficiency << " (exec concurrency " << exec_space().concurrency() << "), "
     << C.efficiencyGPU << " (2^16 threads), built in " << C.buildSeconds << " s";
  return os.str();
}

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
  geo.nneighbor = View1D<unsigned short>("NEMO::nneighbor", np);
  geo.periodic_volume = View1D<double>("NEMO::periodic_volume", np);
  geo.point_ptr = View1D<unsigned long>("NEMO::point_ptr", np + 1);
  geo.pedge_ptr = View1D<unsigned long>("NEMO::pedge_ptr", np + 1);
  {
    auto hc = Kokkos::create_mirror_view(geo.coord);
    auto hv = Kokkos::create_mirror_view(geo.volume);
    auto hnnb = Kokkos::create_mirror_view(geo.nneighbor);
    auto hpv = Kokkos::create_mirror_view(geo.periodic_volume);
    auto hptr = Kokkos::create_mirror_view(geo.point_ptr);
    hptr(0) = 0;
    for (auto i = 0ul; i < np; ++i) {
      for (int d = 0; d < nd; ++d) hc(i, d) = pts->GetCoord(i, d);
      hv(i) = pts->GetVolume(i);
      hnnb(i) = pts->GetnNeighbor(i);
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
    Kokkos::deep_copy(geo.nneighbor, hnnb);
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
    geo.nnz = nnz;
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
  if (deviceTransport) {
    auto* ns = static_cast<CNEMONSVariable*>(nodes);
    perf.bytes_d2h += DtoH(ns->DiffusionCoeff.data(), st.DiffCoeff);
    perf.bytes_d2h += DtoH(ns->LaminarViscosity.data(), st.LamVisc);
    perf.bytes_d2h += DtoH(ns->ThermalCond.data(), st.ThermalCond);
    perf.bytes_d2h += DtoH(ns->ThermalCond_ve.data(), st.ThermalCondVe);
    perf.bytes_d2h += DtoH(ns->Enthalpys.data(), st.Enthalpys);
  }
}

void CNEMOKokkosData::UploadTransport() {
  if (!viscousVars) return;
  auto* ns = static_cast<CNEMONSVariable*>(nodes);
  CheckShape(ns->DiffusionCoeff, nPoint, nSpecies, "DiffusionCoeff");
  CheckShape(ns->Enthalpys, nPoint, nSpecies, "Enthalpys");
  double b = 0.0;
  b += HtoD(st.DiffCoeff, ns->DiffusionCoeff.data());
  b += HtoD(st.LamVisc, ns->LaminarViscosity.data());
  b += HtoD(st.ThermalCond, ns->ThermalCond.data());
  b += HtoD(st.ThermalCondVe, ns->ThermalCond_ve.data());
  b += HtoD(st.Enthalpys, ns->Enthalpys.data());
  stagePerf[ST_VISCOUS].bytes_h2d += b;
  transportDeviceValid = true;
}

unsigned long CNEMOKokkosData::SetPrimVar() {
  Kokkos::Timer timer;

  /*--- HtoD (deep_copy from host memory is synchronous with respect to the host). Device-resident mode: only what
   *    the host changed since the last device write (first call; nothing in steady state). ---*/
  if (ResidentActive()) {
    EnsureDevice(G_SOLUTION); /*--- bytes counted in syncPerf ---*/
    EnsureDevice(G_SOLUTION_OLD);
    EnsureDevice(G_PRIMITIVE);
  } else {
    UploadConservative();
    UploadPrimitive();
  }
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
  const auto TransDP = st.TransDP;
  const bool transport = deviceTransport;

  unsigned long nonPhysical = 0, noGasState = 0;
  Kokkos::parallel_reduce(
      "NEMO::SetPrimVar", Kokkos::RangePolicy<exec_space>(0, nPoint),
      KOKKOS_LAMBDA(const unsigned long i, unsigned long& count, unsigned long& noState) {
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

        /*--- Step 8: gas Density / Pressure members the transport routines of this point read (last
         *    SetTDStateRhosTTv: final attempt, else first attempt, else an earlier point -> carry kernel). ---*/
        if (transport) {
          if (finalFull) {
            double rho = 0.0;
            for (int s = 0; s < ns; s++) rho += Vi[s];
            TransDP(i, 0) = rho;
            TransDP(i, 1) = ComputePressure(g, Vi, Vi[ns], Vi[ns + 1]);
          } else if (firstFull) {
            TransDP(i, 0) = FirstDP(i, 0);
            TransDP(i, 1) = FirstDP(i, 1);
          } else {
            noState++;
          }
        }
      },
      Kokkos::Sum<unsigned long>(nonPhysical), Kokkos::Sum<unsigned long>(noGasState));
  if (transport) {
    perf.kernel += timer.seconds();
    timer.reset();
    /*--- Rare (both attempts non-physical): sequential carry of the previous state, starting from the host gas. ---*/
    if (noGasState > 0) {
      double dp0[2];
      su2kokkos::nemo::HostGasDensityPressure(*gasModel, dp0);
      const double d0 = dp0[0], p0 = dp0[1];
      const auto np = nPoint;
      Kokkos::parallel_for(
          "NEMO::TransportStateCarry", Kokkos::RangePolicy<exec_space>(0, 1), KOKKOS_LAMBDA(const int) {
            double d = d0, p = p0;
            for (unsigned long i = 0; i < np; i++) {
              if (Status(i) & (su2kokkos::nemo::C2P_FINAL_FULL | su2kokkos::nemo::C2P_FIRST_FULL)) {
                d = TransDP(i, 0);
                p = TransDP(i, 1);
              } else {
                TransDP(i, 0) = d;
                TransDP(i, 1) = p;
              }
            }
          });
    }
    ComputeTransportStage();
  }
  /*--- parallel_reduce into a host scalar blocks until completion: no extra fence needed for the timer. ---*/
  perf.kernel += timer.seconds();
  timer.reset();

  if (ResidentActive()) {
    /*--- Device authoritative; host copies are refreshed on demand (EnsureHost). ---*/
    MarkDeviceWritten(G_SOLUTION);
    MarkDeviceWritten(G_PRIMITIVE);
  } else {
    DownloadState();
  }
  perf.d2h += timer.seconds();
  perf.calls++;

  /*--- The device Primitive now equals the host one (host SetVelocity2 recomputes the same U/rho velocity,
   *    NS transport only writes columns >= nPrimVarGrad), so Steps 3-4 read it without re-uploading. ---*/
  primDeviceValid = true;
  transportDeviceValid = deviceTransport;
  bndViscHost = false;
  secDeviceValid = true;
  consDeviceValid = true; /*--- device Solution (clipped / rolled back) == downloaded host Solution ---*/
  sensorValid = false;    /*--- SetPressureDiffusionSensor recomputes it on the host after SetPrimVar ---*/
  gradPrimValid = gradRecValid = false;
  /*--- 4.6-PeriodicLim fix: the limiter depends on the new Primitive too. Without this, a HOST limiter (periodic
   *    with SU2_KOKKOS_NEMO_PERIODIC_LIM=0, SU2_KOKKOS_NEMO_HOST=lim, unported kinds) left limValid = true after the
   *    first upwind upload and the device upwind kept reading the iteration-1 domain limiter rows (only the halo rows
   *    were re-uploaded). The device limiter sets it again in ComputeLimiter before the upwind stage. ---*/
  limValid = false;

  return nonPhysical;
}

void CNEMOKokkosData::ComputeTransportStage() {
  /*--- CNEMONSVariable::SetTransportProperties per point, on the gas state (rho_s, T, Tve, Density, Pressure). ---*/
  const auto g = gas;
  const int ns = nSpecies, nd = nDim;
  const auto V = st.Primitive, eves = st.eves, TransDP = st.TransDP;
  const auto Ds = st.DiffCoeff, hs = st.Enthalpys;
  const auto mu = st.LamVisc, k = st.ThermalCond, kve = st.ThermalCondVe;
  const int lamIdx = nSpecies + nDim + 8;
  Kokkos::parallel_for(
      "NEMO::TransportProperties", Kokkos::RangePolicy<exec_space>(0, nPoint), KOKKOS_LAMBDA(const unsigned long i) {
        using namespace su2kokkos::nemo;
        const double* Vi = &V(i, 0);
        const GasState s{Vi, Vi[ns], Vi[ns + 1], TransDP(i, 0), TransDP(i, 1)};
        double D[MAXNS], h[MAXNS], kk[2], Mu;
        ComputeTransport(g, s, D, Mu, kk);
        ComputeSpeciesEnthalpy(g, Vi[ns], Vi[ns + 1], &eves(i, 0), h);
        for (int sp = 0; sp < ns; sp++) {
          Ds(i, sp) = D[sp];
          hs(i, sp) = h[sp];
        }
        mu(i) = Mu;
        k(i) = kk[0];
        kve(i) = kk[1];
        V(i, lamIdx) = Mu;
        (void)nd;
      });
  Kokkos::fence();
}

void CNEMOKokkosData::ReplayTransportLastPoint() {
  if (nPoint == 0 || !deviceTransport) return;
  auto& P = syncPerf[G_PRIMITIVE];
  const auto i = nPoint - 1;
  std::vector<double> V(nPrimVar), dp(2), ev(nSpecies);
  unsigned char status = 0;
  Kokkos::deep_copy(Kokkos::View<double*, HostSpace, Unmanaged>(V.data(), nPrimVar),
                    Kokkos::subview(st.Primitive, i, Kokkos::ALL()));
  Kokkos::deep_copy(Kokkos::View<double*, HostSpace, Unmanaged>(dp.data(), 2), Kokkos::subview(st.TransDP, i, Kokkos::ALL()));
  Kokkos::deep_copy(Kokkos::View<double*, HostSpace, Unmanaged>(ev.data(), nSpecies), Kokkos::subview(st.eves, i, Kokkos::ALL()));
  Kokkos::deep_copy(Kokkos::View<unsigned char, HostSpace, Unmanaged>(&status), Kokkos::subview(st.Status, i));
  P.bytes_d2h += double(nPrimVar + 2 + nSpecies) * sizeof(double) + 1;
  su2kokkos::nemo::ReplayHostTransportState(*gasModel, nDim, V.data(), status, dp.data(), ev.data());
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
    std::vector<unsigned short> nnb(np);
    std::vector<unsigned long> ptr(np + 1, 0), idx;
    std::vector<long> eidx;
    for (auto i = 0ul; i < np; ++i) {
      for (int d = 0; d < nd; ++d) c[i * nd + d] = pts->GetCoord(i, d);
      v[i] = pts->GetVolume(i);
      nnb[i] = pts->GetnNeighbor(i);
      pv[i] = pts->GetPeriodicVolume(i);
      for (auto k = 0ul; k < pts->GetnPoint(i); ++k) {
        idx.push_back(pts->GetPoint(i, k));
        eidx.push_back(pts->GetEdge(i, k));
      }
      ptr[i + 1] = idx.size();
    }
    bad += CountMismatch(geo.coord, c.data(), "coord");
    bad += CountMismatch(geo.volume, v.data(), "volume");
    bad += CountMismatch(geo.nneighbor, nnb.data(), "nneighbor");
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

namespace {
const char* GroupName(int g) {
  static const char* names[] = {"Solution",  "Solution_Old", "Primitive+secondaries", "Gradient",
                                "Limiter",   "TimeStep",     "LocalCFL",              "Non_Physical",
                                "UnderRelaxation", "Res_TruncError", "LinSysRes+LinSysSol", "Sensor",
                                "AuxVar+Grad_AuxVar", "Solution_time_n+n1",
                                "boundary Primitive (+ viscous-wall gradient/transport)", "halo rows (MPI)"};
  static_assert(sizeof(names) / sizeof(names[0]) == CNEMOKokkosData::N_SYNC, "group names");
  return names[g];
}
const char* StageName(int s) {
  static const char* names[] = {"Gradient", "Limiter", "TimeStep", "MaxEigenvalue", "Upwind",
                                "Source",   "BC",      "ResidualSync", "Viscous", "PressureSensor",
                                "AxisymAuxGradient", "DualTimeSource", "AdaptCFL", "CenteredLAX"};
  static_assert(sizeof(names) / sizeof(names[0]) == CNEMOKokkosData::N_STAGES, "stage names");
  return names[s];
}
}  // namespace

std::vector<double> CNEMOKokkosData::LedgerCategories() const {
  /*--- (h2d, d2h) pairs: SetPrimVar, stages, residency syncs; then the linear-algebra backend categories. ---*/
  std::vector<double> c;
  c.push_back(perf.bytes_h2d);
  c.push_back(perf.bytes_d2h);
  for (int k = 0; k < N_STAGES; ++k) {
    c.push_back(stagePerf[k].bytes_h2d);
    c.push_back(stagePerf[k].bytes_d2h);
  }
  for (int k = 0; k < N_SYNC; ++k) {
    c.push_back(syncPerf[k].bytes_h2d);
    c.push_back(syncPerf[k].bytes_d2h);
  }
  for (int k = 0; k < KokkosTransferLedger::N_CATEGORIES; ++k)
    c.push_back(KokkosTransferLedger::Get(static_cast<KokkosTransferLedger::Category>(k)));
  return c;
}

double CNEMOKokkosData::TotalH2D() const {
  const auto c = LedgerCategories();
  const size_t nPairs = 1 + N_STAGES + N_SYNC;
  double t = 0.0;
  for (size_t k = 0; k < nPairs; ++k) t += c[2 * k];
  using L = KokkosTransferLedger::Category;
  for (auto k : {L::VECTOR_H2D, L::MATRIX_H2D, L::PRECOND_H2D, L::HALO_H2D}) t += c[2 * nPairs + k];
  return t;
}

double CNEMOKokkosData::TotalD2H() const {
  const auto c = LedgerCategories();
  const size_t nPairs = 1 + N_STAGES + N_SYNC;
  double t = 0.0;
  for (size_t k = 0; k < nPairs; ++k) t += c[2 * k + 1];
  using L = KokkosTransferLedger::Category;
  for (auto k : {L::VECTOR_D2H, L::MATRIX_D2H, L::PRECOND_D2H, L::HALO_D2H}) t += c[2 * nPairs + k];
  return t;
}

void CNEMOKokkosData::LedgerIterationBoundary(bool) {
  /*--- Called at the start of every (non-output) SetPrimitive_Variables: closes the previous iteration. The output
   *    of the previous iteration (EnsureHostAll) happened before this call, so it is attributed to that iteration. ---*/
  const auto c = LedgerCategories();
  if (!ledgerLast.empty()) {
    std::vector<double> d(c.size());
    for (size_t k = 0; k < c.size(); ++k) d[k] = c[k] - ledgerLast[k];
    ledgerCat.push_back(d);
    ledgerOutput.push_back(outputThisIteration);
  }
  ledgerLast = c;
  outputThisIteration = false;
}

void CNEMOKokkosData::PrintPerf() const {
  if (perf.calls == 0) return; /*--- same on every rank (one SetPrimVar per iteration) ---*/

  /*--- Step 10: per-rank ledger of the steady iteration (the same iteration index on every rank), gathered to the
   *    master: collective, so before the rank filter. ---*/
  const int nRanks = SU2_MPI::GetSize();
  std::vector<double> rankLedger;
  long rankPick = -1;
  if (nRanks > 1 && !ledgerCat.empty()) {
    for (long it = long(ledgerCat.size()) - 1; it >= 1; --it)
      if (!ledgerOutput[it]) {
        rankPick = it;
        break;
      }
    const size_t nc = ledgerCat[0].size();
    std::vector<double> mine(nc + 1, 0.0);
    if (rankPick >= 0) std::copy(ledgerCat[rankPick].begin(), ledgerCat[rankPick].end(), mine.begin());
    mine[nc] = 0.0;
    for (const auto& H : haloPerf) mine[nc] += H.bytesMsg;
    /*--- MPI halo message bytes per iteration (one SOLUTION exchange per iteration). ---*/
    mine[nc] /= std::max<double>(1.0, double(haloPerf[static_cast<int>(HaloQuantity::SOLUTION)].calls));
    rankLedger.resize((nc + 1) * nRanks);
    SU2_MPI::Gather(mine.data(), int(nc + 1), MPI_DOUBLE, rankLedger.data(), int(nc + 1), MPI_DOUBLE, MASTER_NODE,
                    SU2_MPI::GetComm());
  }
  if (SU2_MPI::GetRank() != MASTER_NODE) return;
  const double c = double(perf.calls);
  std::cout << std::scientific << std::setprecision(3) << "[KOKKOS_NEMO_PERF] MG " << iMesh
            << " SetPrimVar calls=" << perf.calls << " nPoint=" << nPoint << " | h2d " << perf.h2d << " s ("
            << perf.h2d / c << "/call, " << perf.bytes_h2d / c / 1e6 << " MB/call)"
            << " | kernel " << perf.kernel << " s (" << perf.kernel / c << "/call)"
            << " | d2h " << perf.d2h << " s (" << perf.d2h / c << "/call, " << perf.bytes_d2h / c / 1e6
            << " MB/call)"
            << " | host post " << perf.host_post << " s (" << perf.host_post / c << "/call)" << std::endl;
  auto names = [](int s) { return StageName(s); };
  for (int s = 0; s < N_STAGES; ++s) {
    const auto& P = stagePerf[s];
    if (P.calls == 0) continue;
    const double n = double(P.calls);
    std::cout << "[KOKKOS_NEMO_PERF] MG " << iMesh << " " << names(s) << " calls=" << P.calls << " | h2d " << P.h2d
              << " s (" << P.bytes_h2d / n / 1e6 << " MB/call) | kernel " << P.kernel << " s (" << P.kernel / n
              << "/call) | d2h " << P.d2h << " s (" << P.bytes_d2h / n / 1e6 << " MB/call) | host " << P.host
              << " s (" << P.host / n << "/call)" << std::endl;
    if (P.verifyCalls > 0)
      std::cout << "[KOKKOS_NEMO_VERIFY] MG " << iMesh << " " << names(s) << " compares=" << P.verifyCalls
                << " entries=" << P.verifyEntries << " non-bitwise=" << P.verifyNonBitwise
                << " max rel diff=" << P.verifyMaxRel << std::endl;
    if (P.hostRefCalls > 0)
      std::cout << "[KOKKOS_NEMO_PERF] MG " << iMesh << " " << names(s) << " host reference (A/B rerun) "
                << P.hostRef << " s (" << P.hostRef / P.hostRefCalls << "/call)" << std::endl;
  }
  if (outletCalls > 0)
    std::cout << "[KOKKOS_NEMO_PERF] MG " << iMesh << " BC_Outlet calls=" << outletCalls
              << ", with subsonic outlet vertices=" << outletSubsonicCalls << std::endl;
  if (accDropped > 0)
    std::cout << "[KOKKOS_NEMO] WARNING MG " << iMesh << ": " << accDropped
              << " pending device residual accumulators were discarded (missing host sync)" << std::endl;
  if (residencyChecks > 0)
    std::cout << "[KOKKOS_NEMO_VERIFY] MG " << iMesh << " device-resident Primitive checks=" << residencyChecks
              << " mismatching entries=" << residencyMismatch << std::endl;
  if (verifyCalls > 0)
    std::cout << "[KOKKOS_NEMO_VERIFY] MG " << iMesh << " calls=" << verifyCalls
              << " non-bitwise entries=" << verifyMismatch << " max rel diff=" << verifyMaxRel << std::endl;

  /*--- Step 6: device update kernels, residency syncs, per-iteration transfer ledger. ---*/
  const auto& U = updatePerf;
  if (U.nPrepare + U.nExplicit + U.nOldSol > 0)
    std::cout << "[KOKKOS_NEMO_PERF] MG " << iMesh << " Update | prepare " << U.prepare << " s ("
              << (U.nPrepare ? U.prepare / U.nPrepare : 0.0) << "/call) | complete " << U.complete << " s ("
              << (U.nComplete ? U.complete / U.nComplete : 0.0) << "/call) | explicit " << U.explicitUpd << " s ("
              << (U.nExplicit ? U.explicitUpd / U.nExplicit : 0.0) << "/call) | old solution " << U.oldSol
              << " s | residual norm " << U.norm << " s (" << U.nNorm << " calls)" << std::endl;
  for (int g = 0; g < N_SYNC; ++g) {
    const auto& P = syncPerf[g];
    if (P.calls == 0) continue;
    std::cout << "[KOKKOS_NEMO_PERF] MG " << iMesh << " sync " << GroupName(g) << " calls=" << P.calls << " | h2d "
              << P.bytes_h2d / 1e6 << " MB (" << P.h2d << " s) | d2h " << P.bytes_d2h / 1e6 << " MB (" << P.d2h
              << " s)" << std::endl;
  }
  if (!ledgerCat.empty()) {
    const size_t nPairs = 1 + N_STAGES + N_SYNC;
    auto label = [&](size_t k) -> std::string {
      if (k < 2 * nPairs) {
        const size_t pr = k / 2;
        std::string n = (pr == 0) ? "SetPrimVar" : (pr <= N_STAGES) ? StageName(int(pr - 1))
                                                                     : std::string("sync ") + GroupName(int(pr - 1 - N_STAGES));
        return n + ((k % 2) ? " DtoH" : " HtoD");
      }
      return std::string("linear algebra ") +
             KokkosTransferLedger::Name(static_cast<KokkosTransferLedger::Category>(k - 2 * nPairs));
    };
    auto isH2D = [&](size_t k) {
      if (k < 2 * nPairs) return (k % 2) == 0;
      const auto c = static_cast<KokkosTransferLedger::Category>(k - 2 * nPairs);
      using L = KokkosTransferLedger::Category;
      return c == L::VECTOR_H2D || c == L::MATRIX_H2D || c == L::PRECOND_H2D || c == L::HALO_H2D;
    };
    /*--- Steady state: the last closed iteration that wrote no output (iteration index >= 1). ---*/
    long pick = -1;
    for (long it = long(ledgerCat.size()) - 1; it >= 1; --it)
      if (!ledgerOutput[it]) {
        pick = it;
        break;
      }
    double maxH = 0, maxD = 0, outD = 0;
    unsigned long nOut = 0;
    for (size_t it = 1; it < ledgerCat.size(); ++it) {
      double h = 0, d = 0;
      for (size_t k = 0; k < ledgerCat[it].size(); ++k) (isH2D(k) ? h : d) += ledgerCat[it][k];
      if (ledgerOutput[it]) {
        outD += d;
        nOut++;
      } else {
        maxH = std::max(maxH, h);
        maxD = std::max(maxD, d);
      }
    }
    std::cout << "[KOKKOS_NEMO_LEDGER] MG " << iMesh << " " << ledgerCat.size() << " closed iterations ("
              << (resident ? "device-resident" : "host-authoritative") << "); non-output iterations >= 1: max HtoD "
              << maxH << " B, max DtoH " << maxD << " B; output iterations: " << nOut << ", DtoH " << outD
              << " B in total" << std::endl;
    if (pick >= 0) {
      double h = 0, d = 0;
      for (size_t k = 0; k < ledgerCat[pick].size(); ++k) (isH2D(k) ? h : d) += ledgerCat[pick][k];
      std::cout << "[KOKKOS_NEMO_LEDGER] MG " << iMesh << " steady iteration " << pick << ": HtoD " << h
                << " B, DtoH " << d << " B" << std::endl;
      for (size_t k = 0; k < ledgerCat[pick].size(); ++k)
        if (ledgerCat[pick][k] != 0.0)
          std::cout << "[KOKKOS_NEMO_LEDGER]     " << label(k) << ": " << ledgerCat[pick][k] << " B" << std::endl;
    }
  }
  if (!rankLedger.empty() && rankPick >= 0) {
    const size_t nPairs = 1 + N_STAGES + N_SYNC;
    const size_t nc = rankLedger.size() / nRanks - 1;
    for (int r = 0; r < nRanks; ++r) {
      const double* c = &rankLedger[r * (nc + 1)];
      double h = 0, d = 0;
      for (size_t k = 0; k < nc; ++k) {
        bool h2d;
        if (k < 2 * nPairs) {
          h2d = (k % 2) == 0;
        } else {
          using L = KokkosTransferLedger::Category;
          const auto cat = static_cast<L>(k - 2 * nPairs);
          h2d = cat == L::VECTOR_H2D || cat == L::MATRIX_H2D || cat == L::PRECOND_H2D || cat == L::HALO_H2D;
        }
        (h2d ? h : d) += c[k];
      }
      const double haloNemoH = c[2 * (1 + N_STAGES + SYNC_HALO)], haloNemoD = c[2 * (1 + N_STAGES + SYNC_HALO) + 1];
      const double haloLinH = c[2 * nPairs + KokkosTransferLedger::HALO_H2D];
      const double haloLinD = c[2 * nPairs + KokkosTransferLedger::HALO_D2H];
      std::cout << "[KOKKOS_NEMO_LEDGER] MG " << iMesh << " rank " << r << " steady iteration " << rankPick
                << ": HtoD " << h << " B, DtoH " << d << " B; of which halo rows: NEMO HtoD " << haloNemoH
                << " B / DtoH " << haloNemoD << " B, linear algebra HtoD " << haloLinH << " B / DtoH " << haloLinD
                << " B; other " << (h + d - haloNemoH - haloNemoD - haloLinH - haloLinD)
                << " B; NEMO halo MPI messages " << c[nc] << " B/iteration (avg)" << std::endl;
    }
  }
  for (int q = 0; q < static_cast<int>(HaloQuantity::N); ++q) {
    static const char* hn[] = {"SOLUTION", "PRIMITIVE_GRADIENT", "PRIMITIVE_GRAD_REC", "PRIMITIVE_LIMITER", "SENSOR",
                               "AUXVAR_GRADIENT", "SOLUTION_OLD", "LAMBDA", "DELTA_TIME",
                               "PRIMITIVE", "SOLUTION_MAX", "SOLUTION_MIN"};
    if (haloPerf[q].calls > 0)
      std::cout << "[KOKKOS_NEMO_PERF] MG " << iMesh << " halo " << hn[q] << " calls=" << haloPerf[q].calls
                << " | " << haloPerf[q].time << " s | MPI send " << haloPerf[q].bytesMsg / 1e6 << " MB (rank 0)"
                << std::endl;
  }
  std::cout << std::defaultfloat;
}

#endif  // HAVE_KOKKOS
