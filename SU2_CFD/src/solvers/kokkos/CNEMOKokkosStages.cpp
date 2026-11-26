/*!
 * \file CNEMOKokkosStages.cpp
 * \brief Device (Kokkos) NEMO preprocessing stages, Steps 3-4 of NEMO_KOKKOS_PORT_PLAN.md:
 *        primitive gradients (Green-Gauss, (weighted) least squares), slope limiters
 *        (Barth-Jespersen, Venkatakrishnan, Venkatakrishnan-Wang), local time step and max eigenvalue.
 *
 * All kernels are gather-based point loops over the point->neighbour CSR (the same neighbour order as
 * CPoint::GetPoints/GetEdge), followed by the point's boundary vertices in marker-major order and then
 * the symmetry corrections, i.e. per point the same sequence of floating point operations as the CPU
 * implementations (computeGradientsGreenGauss.hpp, computeGradientsLeastSquares.hpp,
 * correctGradientsSymmetry.hpp, computeLimiters_impl.hpp, CFVMFlowSolverBase::SetTime_Step_impl /
 * SetMax_Eigenvalue_impl). Race-free and deterministic; min/max reductions are order independent.
 * Comms (MPI halos, periodic) stay on the host and are done by the caller after the download.
 *
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#ifdef HAVE_KOKKOS

#include "../../../include/solvers/kokkos/CNEMOKokkosData.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include "../../../../Common/include/CConfig.hpp"
#include "../../../../Common/include/geometry/CGeometry.hpp"
#include "../../../../Common/include/toolboxes/geometry_toolbox.hpp"
#include "../../../include/variables/CNEMOEulerVariable.hpp"
#include "../../../include/variables/CNEMONSVariable.hpp"

namespace {

using exec_space = CNEMOKokkosData::exec_space;
using HostSpace = Kokkos::HostSpace;
using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
using Geo = CNEMOKokkosData::Geometry;
template <class T>
using View1D = CNEMOKokkosData::View1D<T>;
template <class T>
using View2D = CNEMOKokkosData::View2D<T>;
template <class T>
using View3D = CNEMOKokkosData::View3D<T>;

constexpr int MAXNVG = su2kokkos::nemo::MAXNPRIMVAR; /*!< >= nPrimVarGrad = nSpecies + nDim + 8 */

/*--- Same semantics as std::max / std::min (first argument returned on ties / unordered). ---*/
KOKKOS_INLINE_FUNCTION double kmax(double a, double b) { return (a < b) ? b : a; }
KOKKOS_INLINE_FUNCTION double kmin(double a, double b) { return (b < a) ? b : a; }

/*! \brief DtoH of the first nRows rows of a LayoutRight device view into a contiguous host buffer. */
template <class DView>
double DownloadRows(double* host, const DView& d, size_t nRows) {
  if (d.size() == 0 || nRows == 0) return 0.0;
  const size_t rowSize = d.size() / d.extent(0);
  Kokkos::View<double*, HostSpace, Unmanaged> h(host, nRows * rowSize);
  Kokkos::View<double*, CNEMOKokkosData::mem_space, Unmanaged> dflat(d.data(), nRows * rowSize);
  Kokkos::deep_copy(h, dflat);
  return double(nRows * rowSize) * sizeof(double);
}

template <class DView>
double UploadRows(const DView& d, const double* host, size_t nRows) {
  if (d.size() == 0 || nRows == 0) return 0.0;
  const size_t rowSize = d.size() / d.extent(0);
  Kokkos::View<const double*, HostSpace, Unmanaged> h(host, nRows * rowSize);
  Kokkos::View<double*, CNEMOKokkosData::mem_space, Unmanaged> dflat(d.data(), nRows * rowSize);
  Kokkos::deep_copy(dflat, h);
  return double(nRows * rowSize) * sizeof(double);
}

/*
 * Note on exactness: every kernel evaluates the same statements in the same order as the CPU template (results
 * accumulated in the output views, same temporaries, loop nests and pass structure: edges, then boundary
 * vertices, then symmetry corrections). Nothing compiler-specific is pinned, so this is portable to nvcc/icpx/hipcc.
 * In a strict IEEE build (-Denable-fast-math=false, -ffp-contract=off) the results are bitwise equal to the CPU;
 * under -ffast-math they agree to a few ULP (each code shape is re-associated / FMA-contracted differently).
 */

/*!
 * \brief detail::correctGradient (correctGradientsSymmetry.hpp) for varBegin = 0, varEnd = nvg, applied to the
 *        row-major (nvg x ND) gradient matrix of one point in device memory.
 */
template <int ND>
KOKKOS_INLINE_FUNCTION void CorrectGradientSym(int nvg, int idxVel, const double* n, double* G) {
  if (idxVel >= 0 && idxVel + ND <= nvg) {
    /*--- Normal gradient of velocity components and gradient of normal velocity. ---*/
    double normalGrad[ND] = {}, gradNormalVel[ND] = {};

    for (int iDim = 0; iDim < ND; iDim++) {
      double dp = 0.0;
      for (int k = 0; k < ND; ++k) dp += G[(idxVel + iDim) * ND + k] * n[k];
      normalGrad[iDim] = dp;

      for (int jDim = 0; jDim < ND; jDim++) gradNormalVel[jDim] += n[iDim] * G[(idxVel + iDim) * ND + jDim];
    }

    double normalGradNormalVel = 0.0;
    for (int k = 0; k < ND; ++k) normalGradNormalVel += n[k] * gradNormalVel[k];

    for (int iDim = 0; iDim < ND; iDim++) {
      for (int jDim = 0; jDim < ND; jDim++) {
        G[(idxVel + iDim) * ND + jDim] -= normalGrad[iDim] * n[jDim] + n[iDim] * gradNormalVel[jDim];
        G[(idxVel + iDim) * ND + jDim] += 2 * n[iDim] * normalGradNormalVel * n[jDim];
      }
    }
  }

  for (int iVar = 0; iVar < nvg; iVar++) {
    if (idxVel != -1 && iVar >= idxVel && iVar < idxVel + ND) continue;

    double normalGrad = 0.0;
    for (int k = 0; k < ND; ++k) normalGrad += n[k] * G[iVar * ND + k];
    for (int iDim = 0; iDim < ND; iDim++) G[iVar * ND + iDim] -= normalGrad * n[iDim];
  }
}

/*--- Field accessor: primitives, or the NEMO-NS Primitive_Aux (rho_s -> Y_s = rho_s / rho). ---*/
struct PrimField {
  View2D<double> V;
  int ns, rhoIdx;
  bool massFractions;
  KOKKOS_INLINE_FUNCTION double operator()(unsigned long i, int v) const {
    return (massFractions && v < ns) ? V(i, v) / V(i, rhoIdx) : V(i, v);
  }
};

/*! \brief Symmetry-plane / Euler-wall corrections (correctGradientsSymmetry), pass over the point's vertices. */
template <int ND>
void KernelSymmetry(const Geo& geo, int nvg, int idxVel, const View3D<double>& grad) {
  const auto pvtx_ptr = geo.pvtx_ptr, pvtx_idx = geo.pvtx_idx;
  const auto vtx_marker = geo.vtx_marker;
  const auto mflags = geo.marker_flags;
  const auto snrm = geo.vtx_sym_normal;
  Kokkos::parallel_for(
      "NEMO::GradientSymmetry", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long i) {
        for (auto k = pvtx_ptr(i); k < pvtx_ptr(i + 1); ++k) {
          const auto kv = pvtx_idx(k);
          if (!(mflags(vtx_marker(kv)) & CNEMOKokkosData::MARKER_SYMMETRY)) continue;
          double unitNormal[ND] = {};
          for (int d = 0; d < ND; ++d) unitNormal[d] = snrm(kv, d);
          CorrectGradientSym<ND>(nvg, idxVel, unitNormal, &grad(i, 0, 0));
        }
      });
}

/*!
 * \brief Green-Gauss gradient (detail::computeGradientsGreenGauss): edge pass, boundary pass, symmetry pass.
 */
template <int ND>
void KernelGreenGauss(const Geo& geo, const PrimField field, int nvg, int idxVel, bool sym, const View3D<double>& gradient) {
  const auto ptr = geo.point_ptr, idx = geo.point_idx;
  const auto pedge = geo.pedge_idx;
  const auto enrm = geo.edge_normal;
  const auto vol = geo.volume, pvol = geo.periodic_volume;
  const auto pvtx_ptr = geo.pvtx_ptr, pvtx_idx = geo.pvtx_idx;
  const auto vtx_marker = geo.vtx_marker;
  const auto mflags = geo.marker_flags;
  const auto vnrm = geo.vtx_normal;

  Kokkos::parallel_for(
      "NEMO::GradientGreenGauss", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint) {
        /*--- Clear the gradient. --*/
        for (int iVar = 0; iVar < nvg; ++iVar)
          for (int iDim = 0; iDim < ND; ++iDim) gradient(iPoint, iVar, iDim) = 0.0;

        /*--- Handle averaging and division by volume in one constant. ---*/
        const double halfOnVol = 0.5 / (vol(iPoint) + pvol(iPoint));

        /*--- Add a contribution due to each neighbor. ---*/
        for (auto k = ptr(iPoint); k < ptr(iPoint + 1); ++k) {
          const auto iEdge = static_cast<unsigned long>(pedge(k));
          const auto jPoint = idx(k);

          const double dir = (iPoint < jPoint) ? 1.0 : -1.0;
          const double weight = dir * halfOnVol;

          const double* area = &enrm(iEdge, 0);

          for (int iVar = 0; iVar < nvg; ++iVar) {
            const double flux = weight * (field(iPoint, iVar) + field(jPoint, iVar));
            for (int iDim = 0; iDim < ND; ++iDim) gradient(iPoint, iVar, iDim) += flux * area[iDim];
          }
        }
      });

  /*--- Boundary faces, in the CPU marker-loop order for each point. ---*/
  Kokkos::parallel_for(
      "NEMO::GradientGreenGaussBoundary", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint) {
        double flux[MAXNVG] = {0.0};
        for (auto k = pvtx_ptr(iPoint); k < pvtx_ptr(iPoint + 1); ++k) {
          const auto kv = pvtx_idx(k);
          if (!(mflags(vtx_marker(kv)) & CNEMOKokkosData::MARKER_GG_BOUNDARY)) continue;

          const double volume = vol(iPoint) + pvol(iPoint);
          const double* area = &vnrm(kv, 0);

          for (int iVar = 0; iVar < nvg; iVar++) flux[iVar] = field(iPoint, iVar) / volume;

          for (int iVar = 0; iVar < nvg; iVar++)
            for (int iDim = 0; iDim < ND; iDim++) gradient(iPoint, iVar, iDim) -= flux[iVar] * area[iDim];
        }
      });

  if (sym) KernelSymmetry<ND>(geo, nvg, idxVel, gradient);
}

/*!
 * \brief (Weighted) least squares (detail::computeGradientsLeastSquares, non-periodic branch) + solveLeastSquares,
 *        then (optionally) the symmetry pass. Rmatrix is accumulated in device memory as on the CPU.
 *
 * Plain source-order arithmetic, statement by statement as the CPU template (portable: no compiler-specific
 * evaluation order is pinned). Bitwise vs CPU in strict IEEE builds (-Denable-fast-math=false, -ffp-contract=off);
 * ULP-level under -ffast-math, where the compiler re-associates/contracts each code shape differently.
 */
template <int ND>
void KernelLeastSquares(const Geo& geo, const PrimField field, int nvg, int idxVel, bool weighted, double eps,
                        bool sym, const View3D<double>& gradient, const View3D<double>& Rmatrix) {
  const auto ptr = geo.point_ptr, idx = geo.point_idx;
  const auto coord = geo.coord;

  Kokkos::parallel_for(
      "NEMO::GradientLeastSquares", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint) {
        const double* coord_i = &coord(iPoint, 0);

        /*--- Clear gradient and Rmatrix. ---*/
        for (int iVar = 0; iVar < nvg; ++iVar)
          for (int iDim = 0; iDim < ND; ++iDim) gradient(iPoint, iVar, iDim) = 0.0;

        for (int iDim = 0; iDim < ND; ++iDim)
          for (int jDim = 0; jDim < ND; ++jDim) Rmatrix(iPoint, iDim, jDim) = 0.0;

        for (auto k = ptr(iPoint); k < ptr(iPoint + 1); ++k) {
          const auto jPoint = idx(k);
          const double* coord_j = &coord(jPoint, 0);

          /*--- Distance vector from iPoint to jPoint ---*/
          double dist_ij[ND] = {0.0};
          for (int iDim = 0; iDim < ND; iDim++) dist_ij[iDim] = coord_j[iDim] - coord_i[iDim];

          /*--- Compute inverse weight, default 1 (unweighted). ---*/
          double weight = 1.0;
          if (weighted) {
            /*--- GeometryToolbox::SquaredNorm. ---*/
            double sq = 0.0;
            for (int iDim = 0; iDim < ND; ++iDim) sq += dist_ij[iDim] * dist_ij[iDim];
            weight = sq;
          }

          /*--- Summations for entries of upper triangular matrix R. ---*/
          if (weight > 0.0) {
            weight = 1.0 / weight;

            for (int iDim = 0; iDim < ND; ++iDim)
              for (int jDim = iDim; jDim < ND; ++jDim)
                Rmatrix(iPoint, iDim, jDim) += dist_ij[iDim] * dist_ij[jDim] * weight;

            if (ND == 3) Rmatrix(iPoint, ND - 1, 1) += dist_ij[0] * dist_ij[ND - 1] * weight;

            /*--- Entries of c:= transpose(A)*b ---*/
            for (int iVar = 0; iVar < nvg; ++iVar) {
              const double delta_ij = weight * (field(jPoint, iVar) - field(iPoint, iVar));
              for (int iDim = 0; iDim < ND; ++iDim) gradient(iPoint, iVar, iDim) += dist_ij[iDim] * delta_ij;
            }
          }
        }

        /*--- detail::solveLeastSquares<ND, false>. ---*/
        double r11 = Rmatrix(iPoint, 0, 0);
        double r12 = Rmatrix(iPoint, 0, 1);
        double r22 = Rmatrix(iPoint, 1, 1);
        double r13 = 0.0, r23 = 0.0, r33 = 1.0;

        r11 = Kokkos::sqrt(kmax(r11, eps));
        r12 /= r11;
        r22 = Kokkos::sqrt(kmax(r22 - r12 * r12, eps));

        if (ND == 3) {
          r13 = Rmatrix(iPoint, 0, ND - 1);
          r33 = Rmatrix(iPoint, ND - 1, ND - 1);
          const double r23_a = Rmatrix(iPoint, 1, ND - 1);
          const double r23_b = Rmatrix(iPoint, ND - 1, 1);

          r13 /= r11;
          r23 = r23_a / r22 - r23_b * r12 / (r11 * r22);
          r33 = Kokkos::sqrt(kmax(r33 - r23 * r23 - r13 * r13, eps));
        }

        /*--- Compute determinant (pow(x, 2)). ---*/
        const double det = r11 * r22 * r33;
        const double detR2 = det * det;

        /*--- S matrix := inv(R)*traspose(inv(R)) ---*/
        double Smatrix[ND][ND] = {{0.0}};

        if (detR2 > eps) {
          if (ND == 2) {
            Smatrix[0][0] = (r12 * r12 + r22 * r22) / detR2;
            Smatrix[0][1] = -r11 * r12 / detR2;
            Smatrix[1][1] = r11 * r11 / detR2;
          } else {
            const double z11 = r22 * r33;
            const double z12 = -r12 * r33;
            const double z13 = r12 * r23 - r13 * r22;
            const double z22 = r11 * r33;
            const double z23 = -r11 * r23;
            const double z33 = r11 * r22;

            Smatrix[0][0] = (z11 * z11 + z12 * z12 + z13 * z13) / detR2;
            Smatrix[0][1] = (z12 * z22 + z13 * z23) / detR2;
            Smatrix[0][ND - 1] = (z13 * z33) / detR2;
            Smatrix[1][1] = (z22 * z22 + z23 * z23) / detR2;
            Smatrix[1][ND - 1] = (z23 * z33) / detR2;
            Smatrix[ND - 1][ND - 1] = (z33 * z33) / detR2;
          }
        }

        /*--- Computation of the gradient: S*c ---*/
        for (int iVar = 0; iVar < nvg; ++iVar) {
          double Cvector[ND] = {0.0};

          for (int iDim = 0; iDim < ND; ++iDim)
            for (int jDim = 0; jDim < ND; ++jDim)
              Cvector[iDim] += Smatrix[iDim < jDim ? iDim : jDim][iDim < jDim ? jDim : iDim] * gradient(iPoint, iVar, jDim);

          for (int iDim = 0; iDim < ND; ++iDim) gradient(iPoint, iVar, iDim) = Cvector[iDim];
        }
      });

  if (sym) KernelSymmetry<ND>(geo, nvg, idxVel, gradient);
}

/*!
 * \brief computeLimiters_impl (non-periodic) with a Venkatakrishnan-type limiter function, one thread per point.
 *        eps2 is per variable (constant for BJ / Venkatakrishnan). fieldMax/Min accumulate in memory as on the CPU.
 */
template <int ND>
void KernelLimiter(const Geo& geo, const View2D<double>& field, int nvg, double umusclKappa,
                   const View1D<double>& eps2, const View3D<double>& gradient, const View2D<double>& limiter,
                   const View2D<double>& fieldMax, const View2D<double>& fieldMin) {
  const auto ptr = geo.point_ptr, idx = geo.point_idx;
  const auto coord = geo.coord;

  Kokkos::parallel_for(
      "NEMO::Limiter", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint) {
        const double* coord_i = &coord(iPoint, 0);

        for (int iVar = 0; iVar < nvg; ++iVar) {
          /*--- Initialize min/max now for iPoint if not periodic. ---*/
          fieldMax(iPoint, iVar) = field(iPoint, iVar);
          fieldMin(iPoint, iVar) = field(iPoint, iVar);
        }

        /*--- Initialize min/max projection out of iPoint. ---*/
        double projMax[MAXNVG], projMin[MAXNVG];
        for (int iVar = 0; iVar < nvg; ++iVar) projMax[iVar] = projMin[iVar] = 0.0;

        /*--- Compute max/min projection and values over direct neighbors. ---*/
        for (auto k = ptr(iPoint); k < ptr(iPoint + 1); ++k) {
          const auto jPoint = idx(k);
          const double* coord_j = &coord(jPoint, 0);

          /*--- Distance vector from iPoint to face (middle of the edge). ---*/
          double dist_ij[ND] = {0.0};
          for (int iDim = 0; iDim < ND; ++iDim) dist_ij[iDim] = 0.5 * (coord_j[iDim] - coord_i[iDim]);

          /*--- Project each variable, update min/max. ---*/
          for (int iVar = 0; iVar < nvg; ++iVar) {
            double proj = 0.0;
            for (int iDim = 0; iDim < ND; ++iDim) proj += dist_ij[iDim] * gradient(iPoint, iVar, iDim);

            const double cent = 0.5 * (field(jPoint, iVar) - field(iPoint, iVar));
            /*--- LimiterHelpers::umusclProjection(proj, cent, kappa). ---*/
            {
              const double c2 = 0.5 * cent;
              const double upw = proj - c2;
              proj = (1.0 - umusclKappa) * upw + (1.0 + umusclKappa) * c2;
            }

            projMax[iVar] = kmax(projMax[iVar], proj);
            projMin[iVar] = kmin(projMin[iVar], proj);

            fieldMax(iPoint, iVar) = kmax(fieldMax(iPoint, iVar), field(jPoint, iVar));
            fieldMin(iPoint, iVar) = kmin(fieldMin(iPoint, iVar), field(jPoint, iVar));
          }
        }

        /*--- Compute the geometric factor (1 for these limiters). ---*/
        const double geoFactor = 1.0;

        for (int iVar = 0; iVar < nvg; ++iVar) {
          const double e2 = eps2(iVar);
          /*--- LimiterHelpers::venkatFunction. ---*/
          auto venkat = [e2](double proj, double delta) {
            const double y = delta * (delta + proj) + e2;
            return (y + delta * proj) / (y + 2 * proj * proj);
          };
          const double limMax = venkat(projMax[iVar], fieldMax(iPoint, iVar) - field(iPoint, iVar));
          const double limMin = venkat(projMin[iVar], fieldMin(iPoint, iVar) - field(iPoint, iVar));
          limiter(iPoint, iVar) = geoFactor * kmin(limMax, limMin);
        }
      });
}

struct DtArgs {
  View2D<double> V;
  View1D<double> cfl, lamInv, lamVisc, dt, mu, k, kve;
  int ns, eddyIdx;
  bool viscous;
  double maxDeltaTime;
};

/*--- CNEMOEulerVariable::GetProjVel, CNEMOEulerSolver::SetTime_Step functors. Namespace scope (not local to
 *    KernelTimeStep): nvcc rejects extended-lambda captures of function-local types. ---*/
template <int ND>
struct DtFn {
  DtArgs a;
  int vel, sos, rho, cvtr, cvve;
  KOKKOS_INLINE_FUNCTION double ProjVel(unsigned long p, const double* n) const {
    double ProjVel = 0.0;
    for (int iDim = 0; iDim < ND; iDim++) ProjVel += a.V(p, vel + iDim) * n[iDim];
    return ProjVel;
  }
  KOKKOS_INLINE_FUNCTION static double lambda(double lamVisc, double eddyVisc, double rho_, double k_, double cv) {
    const double Lambda_1 = (4.0 / 3.0) * (lamVisc + eddyVisc);
    return (Lambda_1 + k_ / cv) / rho_;
  }
  KOKKOS_INLINE_FUNCTION double lambdaVisc(unsigned long i, unsigned long j) const {
    const auto& V = a.V;
    const double lamVisc = 0.5 * (a.mu(i) + a.mu(j));
    const double eddyVisc = 0.5 * (V(i, a.eddyIdx) + V(j, a.eddyIdx));
    const double thermalCond = 0.5 * (a.k(i) + a.k(j) + a.kve(i) + a.kve(j));
    const double density = 0.5 * (V(i, rho) + V(j, rho));
    const double cv = 0.5 * (V(i, cvtr) + V(i, cvve) + V(j, cvtr) + V(j, cvve)) / density;
    return lambda(lamVisc, eddyVisc, density, thermalCond, cv);
  }
  KOKKOS_INLINE_FUNCTION double lambdaVisc(unsigned long i) const {
    const auto& V = a.V;
    const double lamVisc = a.mu(i);
    const double eddyVisc = V(i, a.eddyIdx);
    const double thermalCond = a.k(i) + a.kve(i);
    const double density = V(i, rho);
    const double cv = (V(i, cvtr) + V(i, cvve)) / density;
    return lambda(lamVisc, eddyVisc, density, thermalCond, cv);
  }
};

/*!
 * \brief SetTime_Step_impl (grid-velocity terms when geo.dynamic): point/edge pass, boundary pass, dt pass with MinMax reduction.
 */
template <int ND>
void KernelTimeStep(const Geo& geo, const DtArgs a, double& minDt, double& maxDt) {
  const auto ptr = geo.point_ptr, idx = geo.point_idx;
  const auto pedge = geo.pedge_idx;
  const auto enrm = geo.edge_normal;
  const auto vol = geo.volume;
  const auto pvtx_ptr = geo.pvtx_ptr, pvtx_idx = geo.pvtx_idx;
  const auto vtx_marker = geo.vtx_marker;
  const auto mflags = geo.marker_flags;
  const auto vnrm = geo.vtx_normal;
  const su2kokkos::nemo::NEMOPrimIndices pidx(a.ns, ND);

  const DtFn<ND> fn{a, pidx.VEL(), pidx.A(), pidx.RHO(), pidx.RHOCVTR(), pidx.RHOCVVE()};
  const bool viscous = a.viscous;
  const bool dyn = geo.dynamic;
  const auto gvel = geo.grid_vel;

  Kokkos::parallel_for(
      "NEMO::TimeStepEdges", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint) {
        /*--- Set maximum eigenvalues to zero. ---*/
        fn.a.lamInv(iPoint) = 0.0;
        if (viscous) fn.a.lamVisc(iPoint) = 0.0;

        for (auto k = ptr(iPoint); k < ptr(iPoint + 1); ++k) {
          const auto jPoint = idx(k);
          const auto iEdge = static_cast<unsigned long>(pedge(k));
          const double* Normal = &enrm(iEdge, 0);
          double Area2 = 0.0;
          for (int iDim = 0; iDim < ND; ++iDim) Area2 += Normal[iDim] * Normal[iDim];

          /*--- Mean Values ---*/
          double Mean_ProjVel = 0.5 * (fn.ProjVel(iPoint, Normal) + fn.ProjVel(jPoint, Normal));
          const double Mean_SoundSpeed =
              0.5 * (fn.a.V(iPoint, fn.sos) + fn.a.V(jPoint, fn.sos)) * Kokkos::sqrt(Area2);

          /*--- Adjustment for grid movement ---*/
          if (dyn) {
            for (int iDim = 0; iDim < ND; iDim++)
              Mean_ProjVel -= 0.5 * (gvel(iPoint, iDim) + gvel(jPoint, iDim)) * Normal[iDim];
          }

          /*--- Inviscid contribution ---*/
          double Lambda = Kokkos::fabs(Mean_ProjVel) + Mean_SoundSpeed;
          fn.a.lamInv(iPoint) += Lambda;

          /*--- Viscous contribution ---*/
          if (!viscous) continue;

          Lambda = fn.lambdaVisc(iPoint, jPoint) * Area2;
          fn.a.lamVisc(iPoint) += Lambda;
        }
      });

  Kokkos::parallel_for(
      "NEMO::TimeStepBoundary", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint) {
        for (auto k = pvtx_ptr(iPoint); k < pvtx_ptr(iPoint + 1); ++k) {
          const auto kv = pvtx_idx(k);
          if (!(mflags(vtx_marker(kv)) & CNEMOKokkosData::MARKER_DT_BOUNDARY)) continue;

          const double* Normal = &vnrm(kv, 0);
          double Area2 = 0.0;
          for (int iDim = 0; iDim < ND; ++iDim) Area2 += Normal[iDim] * Normal[iDim];

          /*--- Mean Values ---*/
          double ProjVel = fn.ProjVel(iPoint, Normal);
          const double SoundSpeed = fn.a.V(iPoint, fn.sos) * Kokkos::sqrt(Area2);

          /*--- Adjustment for grid movement (GeometryToolbox::DotProduct, then one subtraction) ---*/
          if (dyn) {
            double dot = 0.0;
            for (int iDim = 0; iDim < ND; ++iDim) dot += Normal[iDim] * gvel(iPoint, iDim);
            ProjVel -= dot;
          }

          /*--- Inviscid contribution ---*/
          double Lambda = Kokkos::fabs(ProjVel) + SoundSpeed;
          fn.a.lamInv(iPoint) += Lambda;

          /*--- Viscous contribution ---*/
          if (!viscous) continue;

          Lambda = fn.lambdaVisc(iPoint) * Area2;
          fn.a.lamVisc(iPoint) += Lambda;
        }
      });

  using MM = Kokkos::MinMax<double>;
  typename MM::value_type red;
  Kokkos::parallel_reduce(
      "NEMO::TimeStepLocal", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint, typename MM::value_type& r) {
        const double K_v = 0.25;
        const double Vol = vol(iPoint);

        if (Vol != 0.0) {
          double Local_Delta_Time = fn.a.cfl(iPoint) * Vol / fn.a.lamInv(iPoint);

          if (viscous) {
            const double dt_visc = fn.a.cfl(iPoint) * K_v * Vol * Vol / fn.a.lamVisc(iPoint);
            Local_Delta_Time = kmin(Local_Delta_Time, dt_visc);
          }

          r.min_val = kmin(r.min_val, Local_Delta_Time);
          r.max_val = kmax(r.max_val, Local_Delta_Time);

          fn.a.dt(iPoint) = kmin(Local_Delta_Time, fn.a.maxDeltaTime);
        } else {
          fn.a.dt(iPoint) = 0.0;
        }
      },
      MM(red));

  minDt = kmin(1e30, red.min_val);
  maxDt = kmax(0.0, red.max_val);
}

/*! \brief SetMax_Eigenvalue_impl (grid-velocity terms when geo.dynamic): point/edge pass, boundary pass. */
template <int ND>
void KernelMaxEigenvalue(const Geo& geo, const View2D<double>& V, int ns, const View1D<double>& lambda) {
  const auto ptr = geo.point_ptr, idx = geo.point_idx;
  const auto pedge = geo.pedge_idx;
  const auto enrm = geo.edge_normal;
  const auto pvtx_ptr = geo.pvtx_ptr, pvtx_idx = geo.pvtx_idx;
  const auto vtx_marker = geo.vtx_marker;
  const auto mflags = geo.marker_flags;
  const auto vnrm = geo.vtx_normal;
  const su2kokkos::nemo::NEMOPrimIndices pidx(ns, ND);
  const int vel = pidx.VEL(), sos = pidx.A();
  const bool dyn = geo.dynamic;
  const auto gvel = geo.grid_vel;

  Kokkos::parallel_for(
      "NEMO::MaxEigenvalueEdges", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint) {
        auto projVel = [&](unsigned long p, const double* n) {
          double ProjVel = 0.0;
          for (int iDim = 0; iDim < ND; iDim++) ProjVel += V(p, vel + iDim) * n[iDim];
          return ProjVel;
        };
        lambda(iPoint) = 0.0;
        for (auto k = ptr(iPoint); k < ptr(iPoint + 1); ++k) {
          const auto jPoint = idx(k);
          const auto iEdge = static_cast<unsigned long>(pedge(k));
          const double* Normal = &enrm(iEdge, 0);
          double Area2 = 0.0;
          for (int iDim = 0; iDim < ND; ++iDim) Area2 += Normal[iDim] * Normal[iDim];
          const double Area = Kokkos::sqrt(Area2);

          double Mean_ProjVel = 0.5 * (projVel(iPoint, Normal) + projVel(jPoint, Normal));
          const double Mean_SoundSpeed = 0.5 * (V(iPoint, sos) + V(jPoint, sos)) * Area;

          /*--- Adjustment for grid movement ---*/
          if (dyn) {
            for (int iDim = 0; iDim < ND; iDim++)
              Mean_ProjVel -= 0.5 * (gvel(iPoint, iDim) + gvel(jPoint, iDim)) * Normal[iDim];
          }

          lambda(iPoint) += Kokkos::fabs(Mean_ProjVel) + Mean_SoundSpeed;
        }
      });

  Kokkos::parallel_for(
      "NEMO::MaxEigenvalueBoundary", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint) {
        for (auto k = pvtx_ptr(iPoint); k < pvtx_ptr(iPoint + 1); ++k) {
          const auto kv = pvtx_idx(k);
          if (!(mflags(vtx_marker(kv)) & CNEMOKokkosData::MARKER_DT_BOUNDARY)) continue;
          const double* Normal = &vnrm(kv, 0);
          double Area2 = 0.0;
          for (int iDim = 0; iDim < ND; ++iDim) Area2 += Normal[iDim] * Normal[iDim];
          const double Area = Kokkos::sqrt(Area2);

          double Mean_ProjVel = 0.0;
          for (int iDim = 0; iDim < ND; iDim++) Mean_ProjVel += V(iPoint, vel + iDim) * Normal[iDim];
          const double Mean_SoundSpeed = V(iPoint, sos) * Area;

          /*--- Adjustment for grid movement (GeometryToolbox::DotProduct, then one subtraction) ---*/
          if (dyn) {
            double dot = 0.0;
            for (int iDim = 0; iDim < ND; ++iDim) dot += Normal[iDim] * gvel(iPoint, iDim);
            Mean_ProjVel -= dot;
          }

          lambda(iPoint) += Kokkos::fabs(Mean_ProjVel) + Mean_SoundSpeed;
        }
      });
}

}  // namespace

/*========================================================================================================*/

void CNEMOKokkosData::UploadBoundaryInfo(CGeometry* geometry, const CConfig* config) {
  const auto np = geo.nPoint;
  const int nd = geo.nDim, nm = geo.nMarker;

  /*--- Domain points must come first (true for SU2 partitions); otherwise the device stages are off. ---*/
  geo.domain_first = true;
  for (auto i = 0ul; i < np; ++i)
    if (geometry->nodes->GetDomain(i) != (i < geo.nPointDomain)) geo.domain_first = false;

  geo.marker_flags = View1D<unsigned char>("NEMO::marker_flags", nm);
  auto hf = Kokkos::create_mirror_view(geo.marker_flags);
  for (int m = 0; m < nm; ++m) {
    const auto kind = config->GetMarker_All_KindBC(m);
    unsigned char f = 0;
    if (kind != INTERNAL_BOUNDARY && kind != NEARFIELD_BOUNDARY && kind != PERIODIC_BOUNDARY) f |= MARKER_GG_BOUNDARY;
    if (kind != INTERNAL_BOUNDARY && kind != SYMMETRY_PLANE && kind != EULER_WALL && kind != NEARFIELD_BOUNDARY &&
        kind != PERIODIC_BOUNDARY)
      f |= MARKER_DT_BOUNDARY;
    if (kind == SYMMETRY_PLANE || kind == EULER_WALL) f |= MARKER_SYMMETRY;
    hf(m) = f;
  }
  Kokkos::deep_copy(geo.marker_flags, hf);

  unsigned long nv = 0;
  for (int m = 0; m < nm; ++m) nv += geometry->GetnVertex(m);

  geo.vtx_marker = View1D<int>("NEMO::vtx_marker", nv);
  geo.vtx_sym_normal = View2D<double>("NEMO::vtx_sym_normal", nv, nd);
  geo.pvtx_ptr = View1D<unsigned long>("NEMO::pvtx_ptr", np + 1);
  geo.pvtx_idx = View1D<unsigned long>("NEMO::pvtx_idx", nv);
  auto hm = Kokkos::create_mirror_view(geo.vtx_marker);
  auto hs = Kokkos::create_mirror_view(geo.vtx_sym_normal);
  auto hp = Kokkos::create_mirror_view(geo.pvtx_ptr);
  auto hi = Kokkos::create_mirror_view(geo.pvtx_idx);
  std::vector<unsigned long> count(np + 1, 0);

  unsigned long k = 0;
  for (int m = 0; m < nm; ++m) {
    for (auto v = 0ul; v < geometry->GetnVertex(m); ++v, ++k) {
      hm(k) = m;
      count[geometry->vertex[m][v]->GetNode() + 1]++;
      for (int d = 0; d < nd; ++d) hs(k, d) = 0.0;
      if (!(hf(m) & MARKER_SYMMETRY)) continue;
      /*--- Exactly as correctGradientsSymmetry computes the normal. ---*/
      su2double unitNormal[3] = {};
      const auto it = geometry->symmetryNormals[m].find(v);
      if (it != geometry->symmetryNormals[m].end()) {
        for (int d = 0; d < nd; d++) unitNormal[d] = it->second[d];
      } else {
        geometry->vertex[m][v]->GetNormal(unitNormal);
        const su2double area = GeometryToolbox::Norm(nd, unitNormal);
        for (int d = 0; d < nd; d++) unitNormal[d] /= area;
      }
      for (int d = 0; d < nd; ++d) hs(k, d) = unitNormal[d];
    }
  }
  for (auto i = 0ul; i < np; ++i) count[i + 1] += count[i];
  for (auto i = 0ul; i <= np; ++i) hp(i) = count[i];
  /*--- Fill in flattened (marker-major) order: per point the vertices keep the CPU marker-loop order. ---*/
  k = 0;
  for (int m = 0; m < nm; ++m)
    for (auto v = 0ul; v < geometry->GetnVertex(m); ++v, ++k) hi(count[geometry->vertex[m][v]->GetNode()]++) = k;

  /*--- Domain points touched by the symmetry correction (for UploadSymmetryRows). ---*/
  h_symPoints.clear();
  for (auto i = 0ul; i < geo.nPointDomain; ++i)
    for (auto k = hp(i); k < hp(i + 1); ++k)
      if (hf(hm(hi(k))) & MARKER_SYMMETRY) {
        h_symPoints.push_back(i);
        break;
      }
  symPoints = View1D<unsigned long>("NEMO::symPoints", h_symPoints.size());
  if (!h_symPoints.empty()) {
    Kokkos::View<unsigned long*, HostSpace, Unmanaged> hsp(h_symPoints.data(), h_symPoints.size());
    Kokkos::deep_copy(symPoints, hsp);
  }

  Kokkos::deep_copy(geo.vtx_marker, hm);
  Kokkos::deep_copy(geo.vtx_sym_normal, hs);
  Kokkos::deep_copy(geo.pvtx_ptr, hp);
  Kokkos::deep_copy(geo.pvtx_idx, hi);
}

void CNEMOKokkosData::RefreshDynamicGeometry(CGeometry* geometry) {
  const auto np = geo.nPoint, ne = geo.nEdge;
  const int nd = geo.nDim, nm = geo.nMarker;
  auto* pts = geometry->nodes; /*--- GetGridVel is non-const ---*/
  const auto* edg = geometry->edges;
  if (geometry->GetnPoint() != np || geometry->GetnEdge() != ne || geometry->GetnMarker() != nm)
    SU2_MPI::Error("CNEMOKokkosData: dynamic grid changed the topology.", CURRENT_FUNCTION);

  if (geo.grid_vel.size() == 0) {
    geo.grid_vel = View2D<double>("NEMO::grid_vel", np, nd);
    geo.volume_nm1 = View1D<double>("NEMO::volume_nm1", np);
  }

  /*--- Edge normals (same element-wise copy as UploadGeometry). ---*/
  {
    auto hN = Kokkos::create_mirror_view(geo.edge_normal);
    for (auto e = 0ul; e < ne; ++e) {
      const su2double* n = edg->GetNormal(e);
      for (int d = 0; d < nd; ++d) hN(e, d) = n[d];
    }
    Kokkos::deep_copy(geo.edge_normal, hN);
  }

  /*--- Points: coordinates, volumes, periodic volumes, Volume_nM1, grid velocity. ---*/
  {
    auto hc = Kokkos::create_mirror_view(geo.coord);
    auto hv = Kokkos::create_mirror_view(geo.volume);
    auto hpv = Kokkos::create_mirror_view(geo.periodic_volume);
    auto hvm1 = Kokkos::create_mirror_view(geo.volume_nm1);
    auto hgv = Kokkos::create_mirror_view(geo.grid_vel);
    for (auto i = 0ul; i < np; ++i) {
      for (int d = 0; d < nd; ++d) hc(i, d) = pts->GetCoord(i, d);
      hv(i) = pts->GetVolume(i);
      hpv(i) = pts->GetPeriodicVolume(i);
      hvm1(i) = pts->GetVolume_nM1(i);
      const su2double* gv = pts->GetGridVel(i);
      for (int d = 0; d < nd; ++d) hgv(i, d) = gv[d];
    }
    Kokkos::deep_copy(geo.coord, hc);
    Kokkos::deep_copy(geo.volume, hv);
    Kokkos::deep_copy(geo.periodic_volume, hpv);
    Kokkos::deep_copy(geo.volume_nm1, hvm1);
    Kokkos::deep_copy(geo.grid_vel, hgv);
  }

  /*--- Vertex normals and the symmetry unit normals (same statements as UploadGeometry / UploadBoundaryInfo). ---*/
  {
    auto hN = Kokkos::create_mirror_view(geo.vtx_normal);
    auto hs = Kokkos::create_mirror_view(geo.vtx_sym_normal);
    auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), geo.marker_flags);
    unsigned long k = 0;
    for (int m = 0; m < nm; ++m) {
      for (auto v = 0ul; v < geometry->GetnVertex(m); ++v, ++k) {
        for (int d = 0; d < nd; ++d) hN(k, d) = geometry->vertex[m][v]->GetNormal(d);
        for (int d = 0; d < nd; ++d) hs(k, d) = 0.0;
        if (!(hf(m) & MARKER_SYMMETRY)) continue;
        su2double unitNormal[3] = {};
        const auto it = geometry->symmetryNormals[m].find(v);
        if (it != geometry->symmetryNormals[m].end()) {
          for (int d = 0; d < nd; d++) unitNormal[d] = it->second[d];
        } else {
          geometry->vertex[m][v]->GetNormal(unitNormal);
          const su2double area = GeometryToolbox::Norm(nd, unitNormal);
          for (int d = 0; d < nd; d++) unitNormal[d] /= area;
        }
        for (int d = 0; d < nd; ++d) hs(k, d) = unitNormal[d];
      }
    }
    Kokkos::deep_copy(geo.vtx_normal, hN);
    Kokkos::deep_copy(geo.vtx_sym_normal, hs);
  }
  Kokkos::fence();
  geo.dynamic = true;
  dynGeoRefreshes++;
}

void CNEMOKokkosData::AllocateGradients() {
  if (st.GradPrim.size() == 0) st.GradPrim = View3D<double>("NEMO::Gradient_Primitive", nPoint, nPrimVarGrad, nDim);
  if (separateRecGradient) {
    if (st.GradRec.size() == 0)
      st.GradRec = View3D<double>("NEMO::Gradient_Reconstruction", nPoint, nPrimVarGrad, nDim);
  } else {
    st.GradRec = st.GradPrim;
  }
}

void CNEMOKokkosData::UploadPrimitiveOnly() {
  /*--- Stage A/B: first check that the resident device Primitive really equals the host one (the columns the
   *    stages read: [0, nPrimVarGrad) and, for NS, the eddy viscosity). ---*/
  if (StageVerifyEnabled() && primDeviceValid) {
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), st.Primitive);
    unsigned long bad = 0;
    int firstCol = -1;
    for (unsigned long i = 0; i < nPoint; ++i)
      for (int v = 0; v < nPrimVar; ++v) {
        if (v >= nPrimVarGrad && !(viscousVars && v == nSpecies + nDim + 9)) continue;
        if (std::memcmp(&h(i, v), &nodes->Primitive(i, v), sizeof(double)) != 0) {
          if (firstCol < 0) firstCol = v;
          ++bad;
        }
      }
    residencyChecks++;
    residencyMismatch += bad;
    if (bad && SU2_MPI::GetRank() == MASTER_NODE)
      std::cout << "[KOKKOS_NEMO_VERIFY] MG " << iMesh << " resident device Primitive != host: " << bad
                << " entries (first column " << firstCol << ")" << std::endl;
  }
  Kokkos::View<const double**, Kokkos::LayoutRight, HostSpace, Unmanaged> h(nodes->Primitive.data(), nPoint,
                                                                             nPrimVar);
  Kokkos::deep_copy(st.Primitive, h);
  primDeviceValid = true;
}

void CNEMOKokkosData::ComputeGradient(GradMethod method, bool massFractions, bool reconstruction, int idxVel,
                                      bool deviceSymmetry, double* hostGrad) {
  auto& P = stagePerf[ST_GRADIENT];
  Kokkos::Timer timer;
  AllocateGradients();
  if (!primDeviceValid) {
    UploadPrimitiveOnly();
    P.bytes_h2d += double(nPoint) * nPrimVar * sizeof(double);
  }
  P.h2d += timer.seconds();
  timer.reset();

  const PrimField f{st.Primitive, nSpecies, nSpecies + nDim + 3, massFractions};
  const auto& grad = reconstruction ? st.GradRec : st.GradPrim;

  if (method == GradMethod::GREEN_GAUSS) {
    if (nDim == 2) KernelGreenGauss<2>(geo, f, nPrimVarGrad, idxVel, deviceSymmetry, grad);
    else KernelGreenGauss<3>(geo, f, nPrimVarGrad, idxVel, deviceSymmetry, grad);
  } else {
    const bool weighted = (method == GradMethod::WEIGHTED_LEAST_SQUARES);
    const double eps = std::pow(std::numeric_limits<passivedouble>::epsilon(), 2);
    if (st.Rmatrix.size() == 0) st.Rmatrix = View3D<double>("NEMO::Rmatrix", nPoint, nDim, nDim);
    if (nDim == 2) KernelLeastSquares<2>(geo, f, nPrimVarGrad, idxVel, weighted, eps, deviceSymmetry, grad, st.Rmatrix);
    else KernelLeastSquares<3>(geo, f, nPrimVarGrad, idxVel, weighted, eps, deviceSymmetry, grad, st.Rmatrix);
  }
  Kokkos::fence();
  P.kernel += timer.seconds();
  timer.reset();

  if (ResidentActive()) MarkDeviceWritten(G_GRADIENT); /*--- downloaded on demand (EnsureHost) ---*/
  else P.bytes_d2h += DownloadRows(hostGrad, grad, geo.nPointDomain);
  P.d2h += timer.seconds();
  P.calls++;

  if (reconstruction || !separateRecGradient) gradRecValid = true;
  if (!reconstruction || !separateRecGradient) gradPrimValid = true;
}

void CNEMOKokkosData::DownloadSymmetryRows(bool reconstruction, double* hostGrad) {
  const size_t ns = h_symPoints.size();
  if (ns == 0) return;
  auto& P = stagePerf[ST_GRADIENT];
  Kokkos::Timer timer;
  const size_t row = size_t(nPrimVarGrad) * nDim;
  if (symStage.extent(0) != ns) symStage = View3D<double>("NEMO::symStage", ns, nPrimVarGrad, nDim);
  const auto grad = reconstruction ? st.GradRec : st.GradPrim;
  const auto stage = symStage;
  const auto pts = symPoints;
  const int nvg = nPrimVarGrad, nd = nDim;
  Kokkos::parallel_for(
      "NEMO::GatherSymmetryRows", Kokkos::RangePolicy<exec_space>(0, ns), KOKKOS_LAMBDA(const size_t k) {
        const auto i = pts(k);
        for (int v = 0; v < nvg; ++v)
          for (int d = 0; d < nd; ++d) stage(k, v, d) = grad(i, v, d);
      });
  auto hs = Kokkos::create_mirror_view_and_copy(HostSpace(), symStage);
  for (size_t k = 0; k < ns; ++k) std::memcpy(hostGrad + h_symPoints[k] * row, &hs(k, 0, 0), row * sizeof(double));
  P.bytes_d2h += double(ns * row) * sizeof(double);
  P.d2h += timer.seconds();
}

bool CNEMOKokkosData::DeviceSymmetryEnabled() {
  static const bool on = [] {
    const char* env = std::getenv("SU2_KOKKOS_NEMO_DEVICE_SYM");
    return env != nullptr && std::atoi(env) != 0;
  }();
  return on;
}

void CNEMOKokkosData::UploadSymmetryRows(bool reconstruction, const double* hostGrad) {
  const size_t ns = h_symPoints.size();
  if (ns == 0) return;
  auto& P = stagePerf[ST_GRADIENT];
  Kokkos::Timer timer;
  const size_t row = size_t(nPrimVarGrad) * nDim;
  if (symStage.extent(0) != ns) symStage = View3D<double>("NEMO::symStage", ns, nPrimVarGrad, nDim);
  auto hs = Kokkos::create_mirror_view(symStage);
  for (size_t k = 0; k < ns; ++k) std::memcpy(&hs(k, 0, 0), hostGrad + h_symPoints[k] * row, row * sizeof(double));
  Kokkos::deep_copy(symStage, hs);
  P.bytes_h2d += double(ns * row) * sizeof(double);

  const auto grad = reconstruction ? st.GradRec : st.GradPrim;
  const auto stage = symStage;
  const auto pts = symPoints;
  const int nvg = nPrimVarGrad, nd = nDim;
  Kokkos::parallel_for(
      "NEMO::ScatterSymmetryRows", Kokkos::RangePolicy<exec_space>(0, ns), KOKKOS_LAMBDA(const size_t k) {
        const auto i = pts(k);
        for (int v = 0; v < nvg; ++v)
          for (int d = 0; d < nd; ++d) grad(i, v, d) = stage(k, v, d);
      });
  Kokkos::fence();
  P.h2d += timer.seconds();
}

void CNEMOKokkosData::UploadGradient(bool reconstruction, const double* hostGrad) {
  AllocateGradients();
  const auto& grad = reconstruction ? st.GradRec : st.GradPrim;
  stagePerf[ST_LIMITER].bytes_h2d += UploadRows(grad, hostGrad, nPoint);
  if (reconstruction || !separateRecGradient) gradRecValid = true;
  if (!reconstruction || !separateRecGradient) gradPrimValid = true;
}

void CNEMOKokkosData::ComputeLimiter(int kind, double umusclKappa, double epsVenkat, double* hostLim,
                                     double* hostMax, double* hostMin) {
  auto& P = stagePerf[ST_LIMITER];
  Kokkos::Timer timer;
  if (st.Limiter.size() == 0) {
    st.Limiter = View2D<double>("NEMO::Limiter_Primitive", nPoint, nPrimVarGrad);
    st.PrimMax = View2D<double>("NEMO::Solution_Max", nPoint, nPrimVarGrad);
    st.PrimMin = View2D<double>("NEMO::Solution_Min", nPoint, nPrimVarGrad);
  }
  if (!primDeviceValid) {
    UploadPrimitiveOnly();
    P.bytes_h2d += double(nPoint) * nPrimVar * sizeof(double);
  }
  P.h2d += timer.seconds();
  timer.reset();

  const int nvg = nPrimVarGrad;
  if (st.LimiterEps2.size() == 0) st.LimiterEps2 = View1D<double>("NEMO::limiter_eps2", nvg);
  const auto eps2 = st.LimiterEps2;
  if (kind == static_cast<int>(LIMITER::VENKATAKRISHNAN_WANG)) {
    /*--- CLimiterDetails<VENKATAKRISHNAN_WANG>::preprocess: global min/max of each variable over domain points. ---*/
    const double largeNum = 0.1 * std::numeric_limits<passivedouble>::max();
    const auto V = st.Primitive;
    std::vector<double> lmin(nvg), lmax(nvg), gmin(nvg), gmax(nvg);
    for (int v = 0; v < nvg; ++v) {
      double mn = largeNum, mx = -largeNum;
      Kokkos::parallel_reduce(
          "NEMO::LimiterRange", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
          KOKKOS_LAMBDA(const unsigned long i, double& a, double& b) {
            a = kmin(a, V(i, v));
            b = kmax(b, V(i, v));
          },
          Kokkos::Min<double>(mn), Kokkos::Max<double>(mx));
      lmin[v] = kmin(largeNum, mn);
      lmax[v] = kmax(-largeNum, mx);
    }
    SU2_MPI::Allreduce(lmin.data(), gmin.data(), nvg, MPI_DOUBLE, MPI_MIN, SU2_MPI::GetComm());
    SU2_MPI::Allreduce(lmax.data(), gmax.data(), nvg, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());
    auto he = Kokkos::create_mirror_view(eps2);
    const su2double K = epsVenkat;
    for (int v = 0; v < nvg; ++v) {
      su2double range = gmax[v] - gmin[v];
      he(v) = std::max(std::pow(K * range, 2), su2double(std::numeric_limits<passivedouble>::epsilon()));
    }
    Kokkos::deep_copy(eps2, he);
  } else {
    Kokkos::deep_copy(eps2, epsVenkat);
  }

  if (nDim == 2) KernelLimiter<2>(geo, st.Primitive, nvg, umusclKappa, eps2, st.GradRec, st.Limiter, st.PrimMax, st.PrimMin);
  else KernelLimiter<3>(geo, st.Primitive, nvg, umusclKappa, eps2, st.GradRec, st.Limiter, st.PrimMax, st.PrimMin);
  Kokkos::fence();
  P.kernel += timer.seconds();
  timer.reset();

  if (ResidentActive()) {
    MarkDeviceWritten(G_LIMITER);
  } else {
    P.bytes_d2h += DownloadRows(hostLim, st.Limiter, geo.nPointDomain);
    P.bytes_d2h += DownloadRows(hostMax, st.PrimMax, geo.nPointDomain);
    P.bytes_d2h += DownloadRows(hostMin, st.PrimMin, geo.nPointDomain);
  }
  P.d2h += timer.seconds();
  P.calls++;
  limValid = true;
}

void CNEMOKokkosData::ComputeTimeStep(bool viscous, double maxDeltaTime, double& minDt, double& maxDt) {
  auto& P = stagePerf[ST_TIMESTEP];
  Kokkos::Timer timer;
  if (st.DeltaTime.size() == 0) {
    st.LocalCFL = View1D<double>("NEMO::LocalCFL", nPoint);
    st.MaxLambdaInv = View1D<double>("NEMO::Max_Lambda_Inv", nPoint);
    st.DeltaTime = View1D<double>("NEMO::Delta_Time", nPoint);
  }
  if (viscous && st.MaxLambdaVisc.size() == 0) st.MaxLambdaVisc = View1D<double>("NEMO::Max_Lambda_Visc", nPoint);
  if (viscous && st.LamVisc.size() == 0) {
    st.LamVisc = View1D<double>("NEMO::LaminarViscosity", nPoint);
    st.ThermalCond = View1D<double>("NEMO::ThermalCond", nPoint);
    st.ThermalCondVe = View1D<double>("NEMO::ThermalCond_ve", nPoint);
  }
  if (!primDeviceValid) {
    UploadPrimitiveOnly();
    P.bytes_h2d += double(nPoint) * nPrimVar * sizeof(double);
  }
  /*--- LocalCFL is adapted on the host (AdaptCFLNumber); NS transport is computed on the host (Step 8).
   *    Device-resident mode: LocalCFL is uploaded only when the host changed it. ---*/
  if (ResidentActive()) EnsureDevice(G_LOCALCFL);
  else P.bytes_h2d += UploadRows(st.LocalCFL, nodes->LocalCFL.data(), nPoint);
  /*--- Step 8: mu, k, k_ve come from the device transport stage of SetPrimVar (uploaded only if host-computed). ---*/
  if (viscous && !transportDeviceValid) {
    auto* ns = static_cast<CNEMONSVariable*>(nodes);
    P.bytes_h2d += UploadRows(st.LamVisc, ns->LaminarViscosity.data(), nPoint);
    P.bytes_h2d += UploadRows(st.ThermalCond, ns->ThermalCond.data(), nPoint);
    P.bytes_h2d += UploadRows(st.ThermalCondVe, ns->ThermalCond_ve.data(), nPoint);
  }
  P.h2d += timer.seconds();
  timer.reset();

  DtArgs a;
  a.V = st.Primitive;
  a.cfl = st.LocalCFL;
  a.lamInv = st.MaxLambdaInv;
  a.lamVisc = st.MaxLambdaVisc;
  a.dt = st.DeltaTime;
  a.mu = st.LamVisc;
  a.k = st.ThermalCond;
  a.kve = st.ThermalCondVe;
  a.ns = nSpecies;
  a.eddyIdx = nSpecies + nDim + 9;
  a.viscous = viscous;
  a.maxDeltaTime = maxDeltaTime;
  if (nDim == 2) KernelTimeStep<2>(geo, a, minDt, maxDt);
  else KernelTimeStep<3>(geo, a, minDt, maxDt);
  P.kernel += timer.seconds();
  timer.reset();

  if (ResidentActive()) {
    MarkDeviceWritten(G_TIMESTEP);
  } else {
    P.bytes_d2h += DownloadRows(nodes->Max_Lambda_Inv.data(), st.MaxLambdaInv, geo.nPointDomain);
    if (viscous) P.bytes_d2h += DownloadRows(nodes->Max_Lambda_Visc.data(), st.MaxLambdaVisc, geo.nPointDomain);
    P.bytes_d2h += DownloadRows(nodes->Delta_Time.data(), st.DeltaTime, geo.nPointDomain);
  }
  P.d2h += timer.seconds();
  P.calls++;
}

void CNEMOKokkosData::ComputeMaxEigenvalue() {
  auto& P = stagePerf[ST_MAXEIG];
  Kokkos::Timer timer;
  if (st.Lambda.size() == 0) st.Lambda = View1D<double>("NEMO::Lambda", nPoint);
  if (!primDeviceValid) {
    UploadPrimitiveOnly();
    P.bytes_h2d += double(nPoint) * nPrimVar * sizeof(double);
  }
  P.h2d += timer.seconds();
  timer.reset();
  if (nDim == 2) KernelMaxEigenvalue<2>(geo, st.Primitive, nSpecies, st.Lambda);
  else KernelMaxEigenvalue<3>(geo, st.Primitive, nSpecies, st.Lambda);
  Kokkos::fence();
  P.kernel += timer.seconds();
  timer.reset();
  P.bytes_d2h += DownloadRows(nodes->Lambda.data(), st.Lambda, geo.nPointDomain);
  lambdaValid = true; /*--- domain rows; halo rows come from the host MAX_EIGENVALUE comms ---*/
  P.d2h += timer.seconds();
  P.calls++;
}

/*====================== Batch A: pressure sensor, axisymmetric auxiliary gradient ======================*/

void CNEMOKokkosData::ComputePressureSensor(double* hostOut) {
  auto& P = stagePerf[ST_SENSOR];
  Kokkos::Timer timer;
  if (st.Sensor.size() == 0) st.Sensor = View1D<double>("NEMO::Sensor", nPoint);
  if (!primDeviceValid) {
    UploadPrimitiveOnly();
    P.bytes_h2d += double(nPoint) * nPrimVar * sizeof(double);
  }
  P.h2d += timer.seconds();
  timer.reset();

  /*--- CNEMOEulerSolver::SetPressureDiffusionSensor: neighbours in CPoint::GetPoints order, std::min semantics. ---*/
  const auto ptr = geo.point_ptr, idx = geo.point_idx;
  const auto V = st.Primitive;
  const auto S = st.Sensor;
  const int P_INDEX = nSpecies + nDim + 2;
  Kokkos::parallel_for(
      "NEMO::PressureDiffusionSensor", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint) {
        double Sensor = 1.0;
        for (auto k = ptr(iPoint); k < ptr(iPoint + 1); ++k) {
          const auto jPoint = idx(k);
          const double P_k = V(jPoint, P_INDEX) / V(iPoint, P_INDEX);
          Sensor = kmin(Sensor, kmin(P_k, 1.0 / P_k));
        }
        S(iPoint) = Sensor;
      });
  Kokkos::fence();
  P.kernel += timer.seconds();
  timer.reset();
  if (hostOut != nullptr) {
    P.bytes_d2h += DownloadRows(hostOut, Kokkos::View<double**, Kokkos::LayoutRight, mem_space, Unmanaged>(
                                             st.Sensor.data(), nPoint, 1), geo.nPointDomain);
  } else {
    MarkDeviceWritten(G_SENSOR);
  }
  sensorValid = true; /*--- domain rows; halo rows from the caller's halo exchange ---*/
  P.d2h += timer.seconds();
  P.calls++;
}

double CNEMOKokkosData::DeviceSensor(unsigned long iPoint) {
  double v = 0.0;
  Kokkos::deep_copy(Kokkos::View<double, HostSpace, Unmanaged>(&v), Kokkos::subview(st.Sensor, iPoint));
  syncPerf[G_SENSOR].bytes_d2h += sizeof(double);
  return v;
}

void CNEMOKokkosData::AllocateAuxVar() {
  if (st.AuxVar.size()) return;
  st.AuxVar = View2D<double>("NEMO::AuxVar", nPoint, 3);
  st.AuxGrad = View3D<double>("NEMO::Grad_AuxVar", nPoint, 3, nDim);
}

void CNEMOKokkosData::UploadAuxVar() {
  AllocateAuxVar();
  auto& P = stagePerf[ST_AUXGRAD];
  P.bytes_h2d += UploadRows(st.AuxVar, nodes->AuxVar.data(), nPoint);
  P.bytes_h2d += UploadRows(st.AuxGrad, nodes->Grad_AuxVar.data(), nPoint);
  auxDeviceValid = true;
}

void CNEMOKokkosData::ComputeAxisymmetricAuxGradient(int method, bool deviceSymmetry) {
  auto& P = stagePerf[ST_AUXGRAD];
  Kokkos::Timer timer;
  if (nodes->GetnAuxVar() != 3 || nDim != 2)
    SU2_MPI::Error("CNEMOKokkosData: axisymmetric auxiliary variables not allocated.", CURRENT_FUNCTION);
  AllocateAuxVar();
  /*--- AuxVar rows with y <= EPS keep their previous value: the device copy must hold the host one once. ---*/
  if (ResidentActive()) EnsureDevice(G_AUXVAR);
  else if (!auxDeviceValid) UploadAuxVar();
  if (!primDeviceValid) {
    UploadPrimitiveOnly();
    P.bytes_h2d += double(nPoint) * nPrimVar * sizeof(double);
  }
  if (!transportDeviceValid) UploadTransport();
  P.h2d += timer.seconds();
  timer.reset();

  /*--- CFVMFlowSolverBase::ComputeAxisymmetricAuxGradients: all points (halos included). ---*/
  {
    const auto coord = geo.coord;
    const auto V = st.Primitive;
    const auto mu = st.LamVisc;
    const auto aux = st.AuxVar;
    const int iVel = nSpecies + 2, eddyIdx = nSpecies + nDim + 9;
    const double eps = EPS;
    Kokkos::parallel_for(
        "NEMO::AxisymmetricAuxVar", Kokkos::RangePolicy<exec_space>(0, nPoint), KOKKOS_LAMBDA(const unsigned long iPoint) {
          const double yCoord = coord(iPoint, 1);
          const double yVelocity = V(iPoint, iVel + 1);
          const double xVelocity = V(iPoint, iVel);
          const double Total_Viscosity = mu(iPoint) + V(iPoint, eddyIdx);
          if (yCoord > eps) {
            const double nu_v_on_y = Total_Viscosity * yVelocity / yCoord;
            aux(iPoint, 0) = nu_v_on_y;
            aux(iPoint, 1) = nu_v_on_y * yVelocity;
            aux(iPoint, 2) = nu_v_on_y * xVelocity;
          }
        });
  }

  /*--- SetAuxVar_Gradient_GG / _LS (weighted), varBegin 0, varEnd 3, idxVel -1. ---*/
  const PrimField f{st.AuxVar, 0, 0, false};
  if (method == 1) {
    KernelGreenGauss<2>(geo, f, 3, -1, deviceSymmetry, st.AuxGrad);
  } else if (method == 2) {
    const double eps = std::pow(std::numeric_limits<passivedouble>::epsilon(), 2);
    if (st.Rmatrix.size() == 0) st.Rmatrix = View3D<double>("NEMO::Rmatrix", nPoint, nDim, nDim);
    KernelLeastSquares<2>(geo, f, 3, -1, true, eps, deviceSymmetry, st.AuxGrad, st.Rmatrix);
  }
  Kokkos::fence();
  P.kernel += timer.seconds();
  timer.reset();
  if (ResidentActive()) MarkDeviceWritten(G_AUXVAR);
  P.calls++;
}

void CNEMOKokkosData::DownloadAuxVar(double* hostAux, double* hostGrad) {
  auto& P = stagePerf[ST_AUXGRAD];
  P.bytes_d2h += DownloadRows(hostAux, st.AuxVar, nPoint);
  P.bytes_d2h += DownloadRows(hostGrad, st.AuxGrad, geo.nPointDomain);
}

void CNEMOKokkosData::DownloadSymmetryRowsAux(double* hostGrad) {
  const size_t ns = h_symPoints.size();
  if (ns == 0) return;
  auto& P = stagePerf[ST_AUXGRAD];
  Kokkos::Timer timer;
  const size_t row = size_t(3) * nDim;
  if (symStageAux.extent(0) != ns) symStageAux = View3D<double>("NEMO::symStageAux", ns, 3, nDim);
  const auto grad = st.AuxGrad;
  const auto stage = symStageAux;
  const auto pts = symPoints;
  const int nd = nDim;
  Kokkos::parallel_for(
      "NEMO::GatherSymmetryRowsAux", Kokkos::RangePolicy<exec_space>(0, ns), KOKKOS_LAMBDA(const size_t k) {
        const auto i = pts(k);
        for (int v = 0; v < 3; ++v)
          for (int d = 0; d < nd; ++d) stage(k, v, d) = grad(i, v, d);
      });
  auto hs = Kokkos::create_mirror_view_and_copy(HostSpace(), symStageAux);
  for (size_t k = 0; k < ns; ++k) std::memcpy(hostGrad + h_symPoints[k] * row, &hs(k, 0, 0), row * sizeof(double));
  P.bytes_d2h += double(ns * row) * sizeof(double);
  P.d2h += timer.seconds();
}

void CNEMOKokkosData::UploadSymmetryRowsAux(const double* hostGrad) {
  const size_t ns = h_symPoints.size();
  if (ns == 0) return;
  auto& P = stagePerf[ST_AUXGRAD];
  Kokkos::Timer timer;
  const size_t row = size_t(3) * nDim;
  if (symStageAux.extent(0) != ns) symStageAux = View3D<double>("NEMO::symStageAux", ns, 3, nDim);
  auto hs = Kokkos::create_mirror_view(symStageAux);
  for (size_t k = 0; k < ns; ++k) std::memcpy(&hs(k, 0, 0), hostGrad + h_symPoints[k] * row, row * sizeof(double));
  Kokkos::deep_copy(symStageAux, hs);
  P.bytes_h2d += double(ns * row) * sizeof(double);
  const auto grad = st.AuxGrad;
  const auto stage = symStageAux;
  const auto pts = symPoints;
  const int nd = nDim;
  Kokkos::parallel_for(
      "NEMO::ScatterSymmetryRowsAux", Kokkos::RangePolicy<exec_space>(0, ns), KOKKOS_LAMBDA(const size_t k) {
        const auto i = pts(k);
        for (int v = 0; v < 3; ++v)
          for (int d = 0; d < nd; ++d) grad(i, v, d) = stage(k, v, d);
      });
  Kokkos::fence();
  P.h2d += timer.seconds();
}

bool CNEMOKokkosData::StageVerifyEnabled() {
  static const bool on = [] {
    const char* env = std::getenv("SU2_KOKKOS_STAGE_VERIFY");
    return env != nullptr && std::atoi(env) != 0;
  }();
  return on;
}

void CNEMOKokkosData::StageCompare(Stage s, const char* name, const double* cpu, const double* dev, size_t n,
                                   size_t ncol) {
  /*--- Relative difference normalised per column (variable/dimension), i.e. max_i |dev-cpu| / max_i |cpu|,
   *    so that trace species are not hidden behind the largest variable. ---*/
  auto& P = stagePerf[s];
  if (ncol == 0) ncol = 1;
  std::vector<double> amax(ncol, 0.0), dmax(ncol, 0.0);
  unsigned long nb = 0;
  for (size_t i = 0; i < n; ++i) {
    const size_t c = i % ncol;
    if (std::memcmp(&cpu[i], &dev[i], sizeof(double)) != 0) {
      ++nb;
      const double d = std::fabs(cpu[i] - dev[i]);
      dmax[c] = (d == d) ? std::max(dmax[c], d) : std::numeric_limits<double>::infinity();
    }
    amax[c] = std::max(amax[c], std::fabs(cpu[i]));
  }
  double rel = 0.0;
  size_t worst = 0;
  for (size_t c = 0; c < ncol; ++c) {
    const double r = dmax[c] / std::max(amax[c], 1e-300);
    if (r > rel) { rel = r; worst = c; }
  }
  if (nb && P.verifyCalls < 6 && SU2_MPI::GetRank() == MASTER_NODE)
    std::cout << "[KOKKOS_NEMO_VERIFY] MG " << iMesh << " " << name << ": " << nb << "/" << n
              << " non-bitwise, max per-column |diff|/max|cpu| " << rel << " (column " << worst << ", max|cpu| "
              << amax[worst] << ")" << std::endl;
  P.verifyNonBitwise += nb;
  P.verifyEntries += n;
  P.verifyMaxRel = std::max(P.verifyMaxRel, rel);
  P.verifyCalls++;
}

void CNEMOKokkosData::StageCompareScalar(Stage s, const char* name, double cpu, double dev) {
  StageCompare(s, name, &cpu, &dev, 1);
}

#endif  // HAVE_KOKKOS
