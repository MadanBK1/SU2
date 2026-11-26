/*!
 * \file CNEMOKokkosMG.cpp
 * \brief 4.6-MG2: device multigrid transfer operators of the NEMO Kokkos port (verbatim CMultiGridIntegration loops).
 * \note Every kernel is a point loop (coarse point -> its children, or fine point -> its neighbours) with the CPU
 *       statement and accumulation order per entry: no atomics, deterministic, bitwise with the CPU on Serial.
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * Copyright 2012-2025, SU2 Contributors (cf. AUTHORS.md)
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#include "../../../include/solvers/kokkos/CNEMOKokkosData.hpp"

#include <algorithm>
#include <vector>

#include "../../../../Common/include/CConfig.hpp"
#include "../../../../Common/include/geometry/CGeometry.hpp"
#include "../../../../Common/include/toolboxes/geometry_toolbox.hpp"

namespace {
using exec_space = CNEMOKokkosData::exec_space;
using HostSpace = Kokkos::HostSpace;
using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
template <class T>
using View1D = CNEMOKokkosData::View1D<T>;
template <class T>
using View2D = CNEMOKokkosData::View2D<T>;

template <class T>
View1D<T> ToDevice1D(const char* name, const std::vector<T>& h) {
  View1D<T> d(name, h.size());
  if (!h.empty()) Kokkos::deep_copy(d, Kokkos::View<const T*, HostSpace, Unmanaged>(h.data(), h.size()));
  return d;
}

/*--- CFVMFlowSolverBase::MultigridProjectEulerWall (static grid: gridVel = 0, so (sol - gridVel) == sol bitwise) on
 *    s = Solution or Solution_Old: the entries of one point in marker / vertex order (sequential, as on the CPU). ---*/
void ProjectEulerWall(const View2D<double>& s, const View1D<unsigned long>& pts, const View1D<unsigned long>& ptr,
                      const View2D<double>& nrm, int nDim, int iVel) {
  if (pts.size() == 0) return;
  Kokkos::parallel_for(
      "NEMO::MG_ProjectEulerWall", Kokkos::RangePolicy<exec_space>(0, pts.size()), KOKKOS_LAMBDA(const unsigned long u) {
        const auto i = pts(u);
        for (auto e = ptr(u); e < ptr(u + 1); ++e) {
          double momentum_n = 0.0;
          for (int d = 0; d < nDim; d++) momentum_n += s(i, iVel + d) * nrm(e, d);
          for (int d = 0; d < nDim; d++) s(i, iVel + d) -= momentum_n * nrm(e, d);
        }
      });
}
}  // namespace

void CNEMOKokkosData::MGSetupCoarse(CGeometry* geoCoarse, const CConfig* config) {
  if (mgCoarseReady) return;
  mgCoarseReady = true;
  const auto nd = geo.nPointDomain;
  std::vector<unsigned long> ptr(nd + 1, 0), idx;
  for (auto I = 0ul; I < nd; ++I) {
    const auto n = geoCoarse->nodes->GetnChildren_CV(I);
    for (auto k = 0u; k < n; ++k) idx.push_back(geoCoarse->nodes->GetChildren_CV(I, k));
    ptr[I + 1] = idx.size();
  }
  mgChildPtr = ToDevice1D("NEMO::MG_childPtr", ptr);
  mgChildIdx = ToDevice1D("NEMO::MG_childIdx", idx);

  /*--- EULER_WALL projection entries, CPU loop order (marker, vertex); same normal computation (host). ---*/
  const int nd_ = geo.nDim;
  std::vector<std::vector<std::vector<double>>> perPoint(nPoint);
  std::vector<unsigned long> order;
  for (auto iMarker = 0u; iMarker < config->GetnMarker_All(); iMarker++) {
    if (config->GetMarker_All_KindBC(iMarker) != EULER_WALL) continue;
    for (auto iVertex = 0ul; iVertex < geoCoarse->nVertex[iMarker]; iVertex++) {
      const auto iPoint = geoCoarse->vertex[iMarker][iVertex]->GetNode();
      if (!geoCoarse->nodes->GetDomain(iPoint)) continue;
      su2double UnitNormal[su2kokkos::nemo::MAXNDIM] = {0.0};
      bool found = false;
      if (iMarker < geoCoarse->symmetryNormals.size()) {
        const auto it = geoCoarse->symmetryNormals[iMarker].find(iVertex);
        if (it != geoCoarse->symmetryNormals[iMarker].end()) {
          for (auto iDim = 0; iDim < nd_; iDim++) UnitNormal[iDim] = it->second[iDim];
          found = true;
        }
      }
      if (!found) {
        su2double Normal[su2kokkos::nemo::MAXNDIM] = {0.0};
        geoCoarse->vertex[iMarker][iVertex]->GetNormal(Normal);
        const su2double Area = GeometryToolbox::Norm(nd_, Normal);
        if (Area < EPS) continue;
        for (auto iDim = 0; iDim < nd_; iDim++) UnitNormal[iDim] = Normal[iDim] / Area;
      }
      if (perPoint[iPoint].empty()) order.push_back(iPoint);
      perPoint[iPoint].emplace_back(UnitNormal, UnitNormal + nd_);
    }
  }
  std::sort(order.begin(), order.end());
  std::vector<unsigned long> wptr(1, 0);
  std::vector<double> wn;
  for (const auto p : order) {
    for (const auto& n : perPoint[p]) wn.insert(wn.end(), n.begin(), n.end());
    wptr.push_back(wptr.back() + perPoint[p].size());
  }
  mgWallPts = ToDevice1D("NEMO::MG_wallPts", order);
  mgWallPtr = ToDevice1D("NEMO::MG_wallPtr", wptr);
  mgWallNormal = View2D<double>("NEMO::MG_wallNormal", wptr.back(), nd_);
  if (!wn.empty())
    Kokkos::deep_copy(mgWallNormal, Kokkos::View<const double**, Kokkos::LayoutRight, HostSpace, Unmanaged>(
                                        wn.data(), wptr.back(), nd_));
}

void CNEMOKokkosData::MGSetupSmoothing(CGeometry* geometry, const CConfig* config) {
  if (mgSmoothReady) return;
  mgSmoothReady = true;
  /*--- SmoothProlongated_Correction resets the vertices of every marker except INTERNAL / NEARFIELD / PERIODIC to
   *    Residual_Old: the same value whatever the order, so a unique point list. ---*/
  std::vector<char> seen(nPoint, 0);
  std::vector<unsigned long> pts;
  for (auto iMarker = 0u; iMarker < geometry->GetnMarker(); iMarker++) {
    const auto kb = config->GetMarker_All_KindBC(iMarker);
    if (kb == INTERNAL_BOUNDARY || kb == NEARFIELD_BOUNDARY || kb == PERIODIC_BOUNDARY) continue;
    for (auto iVertex = 0ul; iVertex < geometry->GetnVertex(iMarker); iVertex++) {
      const auto iPoint = geometry->vertex[iMarker][iVertex]->GetNode();
      if (!seen[iPoint]) {
        seen[iPoint] = 1;
        pts.push_back(iPoint);
      }
    }
  }
  std::sort(pts.begin(), pts.end());
  mgSmoothBnd = ToDevice1D("NEMO::MG_smoothBnd", pts);
  mgResOld = View2D<double>("NEMO::MG_Residual_Old", nPoint, nVar);
  mgResSum = View2D<double>("NEMO::MG_Residual_Sum", nPoint, nVar);
}

void CNEMOKokkosData::MGResidualTerm() {
  ActivateResident();
  EnsureDevice(G_TRUNCERR);
  EnsureDevice(G_LINSYS);
  const auto res = st.Res;
  const auto trunc = st.TruncErr;
  const int nv = nVar;
  Kokkos::parallel_for(
      "NEMO::MG_SetResidual_Term", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long i) {
        for (int v = 0; v < nv; v++) res(i, v) += trunc(i, v);
      });
  Kokkos::fence();
  MarkDeviceWritten(G_LINSYS);
  ++mgDeviceOps;
}

void CNEMOKokkosData::MGRestrictSolution(CNEMOKokkosData& c, int iVel) {
  ActivateResident();
  c.ActivateResident();
  EnsureDevice(G_SOLUTION);
  c.EnsureDevice(G_SOLUTION); /*--- halo rows keep their value ---*/
  const auto sF = st.Solution, sC = c.st.Solution;
  const auto volF = geo.volume, volC = c.geo.volume;
  const auto cptr = c.mgChildPtr, cidx = c.mgChildIdx;
  const int nv = c.nVar;
  /*--- CSolver::MultigridRestriction. ---*/
  Kokkos::parallel_for(
      "NEMO::MG_Restriction", Kokkos::RangePolicy<exec_space>(0, c.geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long I) {
        for (int v = 0; v < nv; v++) sC(I, v) = 0.0;
        const double scale = 1 / volC(I);
        for (auto k = cptr(I); k < cptr(I + 1); ++k) {
          const auto f = cidx(k);
          const double w = volF(f) * scale;
          for (int v = 0; v < nv; ++v) sC(I, v) += w * sF(f, v);
        }
      });
  /*--- (no viscous walls) MultigridProjectEulerWall(coarse, Solution). ---*/
  ProjectEulerWall(sC, c.mgWallPts, c.mgWallPtr, c.mgWallNormal, c.geo.nDim, iVel);
  Kokkos::fence();
  c.MarkDeviceWritten(G_SOLUTION);
  ++mgDeviceOps;
}

void CNEMOKokkosData::MGForcingTerm(CNEMOKokkosData& c, double factor) {
  ActivateResident();
  c.ActivateResident();
  EnsureDevice(G_LINSYS);
  c.EnsureDevice(G_LINSYS);
  c.EnsureDevice(G_TRUNCERR); /*--- halo rows keep their value ---*/
  const auto rF = st.Res, rC = c.st.Res, tC = c.st.TruncErr;
  const auto cptr = c.mgChildPtr, cidx = c.mgChildIdx;
  const int nv = c.nVar;
  /*--- Per entry: SetRes_TruncErrorZero, Residual = sum factor * fine LinSysRes, AddRes_TruncError, (no viscous walls),
   *    SubtractRes_TruncError(coarse LinSysRes): the CPU sequence of each entry. ---*/
  Kokkos::parallel_for(
      "NEMO::MG_SetForcing_Term", Kokkos::RangePolicy<exec_space>(0, c.geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long I) {
        for (int v = 0; v < nv; v++) {
          double r = 0.0;
          for (auto k = cptr(I); k < cptr(I + 1); ++k) r += factor * rF(cidx(k), v);
          tC(I, v) = 0.0;
          tC(I, v) += r;
          tC(I, v) -= rC(I, v);
        }
      });
  Kokkos::fence();
  c.MarkDeviceWritten(G_TRUNCERR);
  ++mgDeviceOps;
}

void CNEMOKokkosData::MGProlongCorrection(CNEMOKokkosData& c, int iVel) {
  ActivateResident();
  c.ActivateResident();
  EnsureDevice(G_SOLUTION);
  c.EnsureDevice(G_SOLUTION);
  c.EnsureDevice(G_SOLUTION_OLD); /*--- halo rows keep their value ---*/
  const auto sF = st.Solution, sC = c.st.Solution, oC = c.st.Solution_Old;
  const auto volF = geo.volume, volC = c.geo.volume;
  const auto cptr = c.mgChildPtr, cidx = c.mgChildIdx;
  const int nv = c.nVar;
  Kokkos::parallel_for(
      "NEMO::MG_GetProlongated_Correction", Kokkos::RangePolicy<exec_space>(0, c.geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long I) {
        const double Area_Parent = volC(I);
        for (int v = 0; v < nv; v++) {
          double s = 0.0;
          for (auto k = cptr(I); k < cptr(I + 1); ++k) {
            const auto f = cidx(k);
            const double weight = volF(f) / Area_Parent;
            s -= sF(f, v) * weight;
          }
          s += sC(I, v);
          oC(I, v) = s;
        }
      });
  /*--- MultigridProjectEulerWall(coarse, Solution_Old); (no viscous walls). ---*/
  ProjectEulerWall(oC, c.mgWallPts, c.mgWallPtr, c.mgWallNormal, c.geo.nDim, iVel);
  Kokkos::fence();
  c.MarkDeviceWritten(G_SOLUTION_OLD);
  ++mgDeviceOps;
}

void CNEMOKokkosData::MGProlongScatter(CNEMOKokkosData& c) {
  c.EnsureDevice(G_SOLUTION_OLD);
  EnsureDevice(G_LINSYS);
  const auto rF = st.Res, oC = c.st.Solution_Old;
  const auto cptr = c.mgChildPtr, cidx = c.mgChildIdx;
  const int nv = c.nVar;
  Kokkos::parallel_for(
      "NEMO::MG_ProlongScatter", Kokkos::RangePolicy<exec_space>(0, c.geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long I) {
        for (auto k = cptr(I); k < cptr(I + 1); ++k) {
          const auto f = cidx(k);
          for (int v = 0; v < nv; ++v) rF(f, v) = oC(I, v);
        }
      });
  Kokkos::fence();
  MarkDeviceWritten(G_LINSYS);
  ++mgDeviceOps;
}

void CNEMOKokkosData::MGSmoothSave() {
  EnsureDevice(G_LINSYS);
  Kokkos::deep_copy(mgResOld, st.Res);
  Kokkos::fence();
}

void CNEMOKokkosData::MGSmoothIteration(double coeff) {
  const auto res = st.Res, rOld = mgResOld, rSum = mgResSum;
  const auto pptr = geo.point_ptr, pidx = geo.point_idx;
  const int nv = nVar;
  Kokkos::parallel_for(
      "NEMO::MG_SmoothSum", Kokkos::RangePolicy<exec_space>(0, nPoint), KOKKOS_LAMBDA(const unsigned long i) {
        for (int v = 0; v < nv; v++) {
          double s = 0.0;
          for (auto k = pptr(i); k < pptr(i + 1); ++k) s += res(pidx(k), v);
          rSum(i, v) = s;
        }
      });
  Kokkos::parallel_for(
      "NEMO::MG_SmoothUpdate", Kokkos::RangePolicy<exec_space>(0, nPoint), KOKKOS_LAMBDA(const unsigned long i) {
        const double factor = 1.0 / (1.0 + coeff * double(pptr(i + 1) - pptr(i)));
        for (int v = 0; v < nv; v++) res(i, v) = (rOld(i, v) + coeff * rSum(i, v)) * factor;
      });
  const auto bnd = mgSmoothBnd;
  if (bnd.size() > 0)
    Kokkos::parallel_for(
        "NEMO::MG_SmoothBoundary", Kokkos::RangePolicy<exec_space>(0, bnd.size()),
        KOKKOS_LAMBDA(const unsigned long u) {
          const auto i = bnd(u);
          for (int v = 0; v < nv; v++) res(i, v) = rOld(i, v);
        });
  Kokkos::fence();
  MarkDeviceWritten(G_LINSYS);
  ++mgDeviceOps;
}

void CNEMOKokkosData::MGSetCorrection(double factor) {
  EnsureDevice(G_LINSYS);
  EnsureDevice(G_SOLUTION);
  const auto res = st.Res, sol = st.Solution;
  const int nv = nVar;
  Kokkos::parallel_for(
      "NEMO::MG_SetProlongated_Correction", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long i) {
        for (int v = 0; v < nv; v++) {
          if (res(i, v) != res(i, v)) res(i, v) = 0.0;
          const double correction = factor * res(i, v);
          sol(i, v) += correction;
        }
      });
  Kokkos::fence();
  MarkDeviceWritten(G_LINSYS);
  MarkDeviceWritten(G_SOLUTION);
  ++mgDeviceOps;
}

void CNEMOKokkosData::MGFillLocalCFL(double cfl) {
  ActivateResident();
  if (st.LocalCFL.size() == 0) st.LocalCFL = View1D<double>("NEMO::LocalCFL", nPoint);
  const auto c = st.LocalCFL;
  Kokkos::parallel_for(
      "NEMO::MG_LocalCFL", Kokkos::RangePolicy<exec_space>(0, nPoint), KOKKOS_LAMBDA(const unsigned long i) { c(i) = cfl; });
  Kokkos::fence();
  MarkDeviceWritten(G_LOCALCFL);
  ++mgDeviceOps;
}
