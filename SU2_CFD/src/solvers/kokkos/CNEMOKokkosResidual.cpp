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

#include "CNEMOKokkosResidualKernels.hpp"

/*--- The source / BC kernel templates (KernelSource, KernelIsothermal, ..., KernelBoundaryFlux) are defined in
 *    CNEMOKokkosResidualKernels.inl and explicitly instantiated in CNEMOKokkosResidual{Wall,Source,BCFlux2D,BCFlux3D}.cpp
 *    (split for nvcc compile time; the kernel code is unchanged). ---*/

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
  MarkDeviceWritten(G_LINSYS); /*--- resident mode: st.Res is the device LinSysRes ---*/
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
  if (needSensor && ResidentActive()) {
    EnsureDevice(G_SENSOR); /*--- device SetPressureDiffusionSensor (batch A) ---*/
  } else if (needSensor && (upload || !sensorValid)) {
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
  /*--- Batch A: the axisymmetric terms also read U (rho u, rho v, rho Eve) and, viscous, the primitive gradient, the
   *    transport properties and the AuxVar gradient. ---*/
  EnsureInputsOnDevice(uploadInputs, p.axisymmetric, false);
  if (p.axiViscous) {
    if (ResidentActive()) {
      EnsureDevice(G_GRADIENT);
      EnsureDevice(G_AUXVAR);
    } else {
      if (uploadInputs || !gradPrimValid) {
        AllocateGradients();
        Kokkos::deep_copy(st.GradPrim, Kokkos::View<const double***, Kokkos::LayoutRight, HostSpace, Unmanaged>(
                                           nodes->GetGradient_Primitive().data(), nPoint, nPrimVarGrad, nDim));
        P.bytes_h2d += double(st.GradPrim.size()) * sizeof(double);
        gradPrimValid = true;
        if (!separateRecGradient) gradRecValid = true;
      }
      if (uploadInputs || !transportDeviceValid) UploadTransport();
      if (uploadInputs) UploadAuxVar();
    }
  }
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
  r.axisym = p.axisymmetric;
  r.axiViscous = p.axiViscous;
  r.nHeavy = p.nHeavyNumerics;
  r.prandtlTurb = p.prandtlTurb;
  r.RuSI = p.RuSI;
  r.eddyIdx = nSpecies + nDim + 9;
  if (p.axisymmetric) {
    r.a.dPdU = st.dPdU;
    r.U = st.Solution;
    r.coord = geo.coord;
  }
  if (p.axiViscous) {
    r.grad = st.GradPrim;
    r.auxGrad = st.AuxGrad;
    r.Ds = st.DiffCoeff;
    r.mu = st.LamVisc;
    r.k = st.ThermalCond;
    r.kve = st.ThermalCondVe;
  }

  if (nDim == 2) KernelSource<2>(r, out.eChm, out.eVib, out.eAxi);
  else KernelSource<3>(r, out.eChm, out.eVib, out.eAxi);
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

  const bool smol = (p.kind == BCKind::SMOLUCHOWSKI_MAXWELL);
  const bool wall = (p.kind == BCKind::ISOTHERMAL || p.kind == BCKind::HEAT_FLUX || smol);
  const bool flux = (p.kind != BCKind::SYMMETRY) && !wall;
  const bool ausmPlusM = flux && (p.upw.scheme == UpwindScheme::AUSMPLUSM);
  /*--- Roe boundary numerics (CUpwRoe_NEMO: LAX or ROE convection) read the conservative states. ---*/
  const bool roe = flux && (p.upw.scheme == UpwindScheme::ROE || p.upw.scheme == UpwindScheme::MSW);
  const bool needState = (p.kind == BCKind::FAR_FIELD || p.kind == BCKind::OUTLET || p.kind == BCKind::SUPERSONIC_INLET);
  /*--- viscous part of BC_Far_Field (node_infty) or BC_Supersonic_Inlet (node_inlet): same CAvgGrad_NEMO call. ---*/
  const bool viscFar = (p.kind == BCKind::FAR_FIELD || p.kind == BCKind::SUPERSONIC_INLET) && p.viscous;
  EnsureInputsOnDevice(uploadInputs, p.kind == BCKind::SYMMETRY || p.kind == BCKind::OUTLET || roe, ausmPlusM);

  /*--- Step 8: NS inputs (gradient, transport), host-authoritative mode only (resident: always on the device). ---*/
  if (wall || viscFar) {
    if (!viscousVars) SU2_MPI::Error("CNEMOKokkosData: viscous BC without NS variables.", CURRENT_FUNCTION);
    const bool needGrad = viscFar || p.kind == BCKind::HEAT_FLUX || smol;
    if (needGrad && (uploadInputs || !gradPrimValid)) {
      if (ResidentActive()) SU2_MPI::Error("CNEMOKokkosData: device-resident gradient not valid.", CURRENT_FUNCTION);
      AllocateGradients();
      Kokkos::deep_copy(st.GradPrim, Kokkos::View<const double***, Kokkos::LayoutRight, HostSpace, Unmanaged>(
                                         nodes->GetGradient_Primitive().data(), nPoint, nPrimVarGrad, nDim));
      P.bytes_h2d += double(st.GradPrim.size()) * sizeof(double);
      gradPrimValid = true;
      if (!separateRecGradient) gradRecValid = true;
    }
    if (uploadInputs || !transportDeviceValid) {
      if (ResidentActive()) SU2_MPI::Error("CNEMOKokkosData: device-resident transport not valid.", CURRENT_FUNCTION);
      UploadTransport();
    }
    /*--- Host-authoritative mode: the caller updates the host Solution_Old / Res_TruncError rows itself. ---*/
    if (wall && ResidentActive()) {
      EnsureDevice(G_SOLUTION_OLD);
      EnsureDevice(G_TRUNCERR);
    }
  }
  if (viscFar) {
    /*--- node_infty Cvve | dTdU | dTvedU | gradient, uploaded when it changes (constant in practice). ---*/
    const size_t ng = size_t(nPrimVarGrad) * nDim;
    const size_t n = size_t(nSpecies) + 2 * size_t(nVar) + ng;
    std::vector<double> h(n, 0.0);
    std::copy(bs.Cvve, bs.Cvve + nSpecies, h.begin());
    if (p.upw.implicit) {
      std::copy(bs.dTdU, bs.dTdU + nVar, h.begin() + nSpecies);
      std::copy(bs.dTvedU, bs.dTvedU + nVar, h.begin() + nSpecies + nVar);
    }
    std::copy(bs.grad, bs.grad + ng, h.begin() + nSpecies + 2 * nVar);
    if (st.BStateVisc.size() != n) st.BStateVisc = View1D<double>("NEMO::BoundaryStateViscous", n);
    if (uploadInputs || bstateViscCache.size() != n ||
        std::memcmp(h.data(), bstateViscCache.data(), n * sizeof(double)) != 0) {
      Kokkos::deep_copy(st.BStateVisc, Kokkos::View<const double*, HostSpace, Unmanaged>(h.data(), n));
      bstateViscCache = h;
      P.bytes_h2d += double(n) * sizeof(double);
    }
  }

  if (needState) {
    /*--- V | eve | dPdU | Gamma (| U for Roe), uploaded only when it differs from the cached copy (constant in
     *    practice). ---*/
    const size_t n0 = size_t(nPrimVar) + nSpecies + nVar + 1;
    const size_t n = n0 + (roe ? size_t(nVar) : 0);
    std::vector<double> h(n, 0.0);
    std::copy(bs.V, bs.V + nPrimVar, h.begin());
    std::copy(bs.eve, bs.eve + nSpecies, h.begin() + nPrimVar);
    if (p.upw.implicit && bs.dPdU) std::copy(bs.dPdU, bs.dPdU + nVar, h.begin() + nPrimVar + nSpecies);
    h[n0 - 1] = bs.Gamma;
    if (roe) {
      if (!bs.U) SU2_MPI::Error("CNEMOKokkosData: Roe boundary numerics without the boundary Solution.", CURRENT_FUNCTION);
      std::copy(bs.U, bs.U + nVar, h.begin() + n0);
    }
    if (st.BState.size() != n) st.BState = View1D<double>("NEMO::BoundaryState", n);
    if (uploadInputs || bstateCache.size() != n || std::memcmp(h.data(), bstateCache.data(), n * sizeof(double)) != 0) {
      Kokkos::deep_copy(st.BState, Kokkos::View<const double*, HostSpace, Unmanaged>(h.data(), n));
      bstateCache = h;
      P.bytes_h2d += double(n) * sizeof(double);
    }
  }
  if (smol && !ResidentActive() && st.SlipVel.size() == 0) {
    const auto nv = Kokkos::create_mirror_view_and_copy(HostSpace(), geo.marker_ptr)(geo.nMarker);
    st.SlipVel = View2D<double>("NEMO::SlipVelocity", nv, nDim);
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
  r.a.mswAlpha = p.upw.mswAlpha;
  r.a.pi = PI_NUMBER;
  r.U = st.Solution;
  r.charac = st.Charac;
  r.vtxNormal = geo.vtx_normal;
  r.vtxNode = geo.vtx_node;
  r.bstate = needState ? st.BState : View1D<double>();
  r.nPointDomain = geo.nPointDomain;
  r.P_Exit = p.P_Exit;
  r.staleNumerics = p.staleNumerics;
  r.staleGammaI = p.staleGammaI;
  r.staleGammaJ = p.staleGammaJ;
  r.staleSensor = p.staleSensor;
  /*--- Step 8. ---*/
  r.viscous = viscFar;
  if (wall || viscFar) {
    r.vc.gas = gas;
    r.vc.ns = nSpecies;
    r.vc.nVar = nVar;
    r.vc.nPrimVar = nPrimVar;
    r.vc.nPrimVarGrad = nPrimVarGrad;
    r.vc.nHeavy = nSpecies - p.upw.nElNumerics;
    r.vc.implicit = p.upw.implicit;
    r.vc.Ru = 1000.0 * UNIVERSAL_GAS_CONSTANT;
    r.grad = st.GradPrim;
    r.coord = geo.coord;
    r.Ds = st.DiffCoeff;
    r.mu = st.LamVisc;
    r.k = st.ThermalCond;
    r.kve = st.ThermalCondVe;
    r.Cvves = st.Cvves;
    r.dTdU = st.dTdU;
    r.dTvedU = st.dTvedU;
    r.bstateVisc = viscFar ? st.BStateVisc : View1D<double>();
    r.vtxNormalNeighbor = geo.vtx_normal_neighbor;
    r.rowPtr = geo.row_ptr;
    r.colInd = geo.col_ind;
    r.eddyIdx = nSpecies + nDim + 9;
    r.catalytic = p.catalytic;
    r.supercat = p.supercatalytic;
    for (int s2 = 0; s2 < nSpecies; s2++) r.Yst[s2] = p.Yst[s2];
    r.Twall = p.Twall;
    r.catEff = p.catEff;
    r.heatFlux = p.heatFlux;
    r.gasConstant = p.gasConstant;
    r.prandtlLam = p.prandtlLam;
    r.slip = st.SlipVel;
    r.RuSI = UNIVERSAL_GAS_CONSTANT;
    r.writeWallState = wall && ResidentActive();
    if (r.writeWallState) {
      r.Uold = st.Solution_Old;
      r.trunc = st.TruncErr;
    }
  }

  /*--- Marker range (host copy of marker_ptr cached on first use). ---*/
  if (h_markerPtr.empty()) {
    auto hm = Kokkos::create_mirror_view_and_copy(HostSpace(), geo.marker_ptr);
    h_markerPtr.assign(hm.data(), hm.data() + geo.nMarker + 1);
  }
  const unsigned long v0 = h_markerPtr[p.marker], v1 = h_markerPtr[p.marker + 1];

  /*--- 6b.b: per-vertex wall temperature (python-custom isothermal marker): copy only this marker's values. ---*/
  if (p.kind == BCKind::ISOTHERMAL && p.customTwall != nullptr && v1 > v0) {
    if (customTwall_.extent(0) != geo.vtx_node.extent(0))
      customTwall_ = View1D<double>("NEMO::custom_Twall", geo.vtx_node.extent(0));
    Kokkos::View<const double*, HostSpace, Unmanaged> hT(p.customTwall, v1 - v0);
    Kokkos::deep_copy(Kokkos::subview(customTwall_, std::make_pair(v0, v1)), hT);
    r.customTwall = true;
    r.twallVtx = customTwall_;
  }

  long lastSub = -1;
  if (smol) {
    if (nDim == 2) KernelSmoluchowski<2>(r, v0, v1);
    else KernelSmoluchowski<3>(r, v0, v1);
  } else if (p.kind == BCKind::ISOTHERMAL) {
    if (nDim == 2) KernelIsothermal<2>(r, v0, v1);
    else KernelIsothermal<3>(r, v0, v1);
  } else if (p.kind == BCKind::HEAT_FLUX) {
    if (nDim == 2) KernelHeatFlux<2>(r, v0, v1);
    else KernelHeatFlux<3>(r, v0, v1);
  } else if (!flux) {
    if (nDim == 2) KernelSymmetry<2>(r, v0, v1);
    else KernelSymmetry<3>(r, v0, v1);
  } else {
    if (nDim == 2) KernelBoundaryFlux<2>(r, static_cast<int>(p.kind), v0, v1, lastSub);
    else KernelBoundaryFlux<3>(r, static_cast<int>(p.kind), v0, v1, lastSub);
  }
  Kokkos::fence();
  P.kernel += timer.seconds();
  timer.reset();
  if (r.writeWallState) {
    MarkDeviceWritten(G_SOLUTION_OLD);
    MarkDeviceWritten(G_TRUNCERR);
  }

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
  if (smol && p.downloadSlip && !ResidentActive() && v1 > v0) {
    out.slipVel.resize((v1 - v0) * nDim);
    Kokkos::deep_copy(Kokkos::View<double**, Kokkos::LayoutRight, HostSpace, Unmanaged>(out.slipVel.data(), v1 - v0, nDim),
                      Kokkos::subview(st.SlipVel, std::make_pair(v0, v1), Kokkos::ALL()));
    P.bytes_d2h += double(out.slipVel.size()) * sizeof(double);
  }
  P.d2h += timer.seconds();
  P.calls++;
  return out;
}


/*====================== Step 6: device-resident state, implicit / explicit solution update ======================*/

namespace {

using RR = CNEMOKokkosData::ResidualReduction;

KOKKOS_INLINE_FUNCTION void RRInit(RR& v) {
  for (int k = 0; k < su2kokkos::nemo::MAXNVAR; ++k) {
    v.rms[k] = 0.0;
    v.max[k] = 0.0;
    v.idx[k] = 0;
  }
}
/*--- Ordered chunks (Serial: a single chunk) give the CPU result; the index tie-break keeps the first maximum
 *    (CPU: strict ">") for any join order. ---*/
KOKKOS_INLINE_FUNCTION void RRJoin(RR& d, const RR& s, int nVar) {
  for (int k = 0; k < nVar; ++k) {
    d.rms[k] += s.rms[k];
    if (s.max[k] > d.max[k] || (s.max[k] == d.max[k] && s.idx[k] < d.idx[k])) {
      d.max[k] = s.max[k];
      d.idx[k] = s.idx[k];
    }
  }
}
/*--- CSolver::ResidualReductions_PerThread. ---*/
KOKKOS_INLINE_FUNCTION void RRAdd(RR& v, unsigned long iPoint, int iVar, double res) {
  res = Kokkos::fabs(res);
  v.rms[iVar] += res * res;
  if (res > v.max[iVar]) {
    v.max[iVar] = res;
    v.idx[iVar] = iPoint;
  }
}

/*--- PrepareImplicitIteration_impl (DummyPrec: AddVal2Diag / SetVal2Diag) fused with the RHS loop (per point
 *    independent, same statements per entry). ---*/
struct PrepareFunctor {
  using value_type = RR;
  View1D<double> vol, pvol, dt, jac;
  View1D<unsigned long> dia;
  View2D<double> res, sol, trunc;
  int nVar = 0;
  bool implicit = false;

  KOKKOS_INLINE_FUNCTION void init(value_type& v) const { RRInit(v); }
  KOKKOS_INLINE_FUNCTION void join(value_type& d, const value_type& s) const { RRJoin(d, s, nVar); }
  KOKKOS_INLINE_FUNCTION void operator()(const unsigned long iPoint, value_type& v) const {
    const unsigned long nv = nVar;
    const double dti = dt(iPoint);
    if (implicit) {
      const unsigned long base = dia(iPoint) * nv * nv;
      if (dti != 0.0) {
        const double Vol = vol(iPoint) + pvol(iPoint);
        const double Delta = Vol / dti;
        for (unsigned long iVar = 0; iVar < nv; iVar++) jac(base + iVar * (nv + 1)) += Delta;
      } else {
        for (unsigned long k = 0; k < nv * nv; k++) jac(base + k) = 0.0;
        for (unsigned long iVar = 0; iVar < nv; iVar++) jac(base + iVar * (nv + 1)) = 1.0;
      }
    }
    if (dti == 0.0) {
      for (int iVar = 0; iVar < nVar; iVar++) {
        res(iPoint, iVar) = 0.0;
        trunc(iPoint, iVar) = 0.0;
      }
    }
    for (int iVar = 0; iVar < nVar; iVar++) {
      res(iPoint, iVar) = -(res(iPoint, iVar) + trunc(iPoint, iVar));
      sol(iPoint, iVar) = 0.0;
      RRAdd(v, iPoint, iVar, res(iPoint, iVar));
    }
  }
};

/*--- Explicit_Iteration_impl<EULER_EXPLICIT> with the Identity residual "preconditioner". ---*/
struct ExplicitFunctor {
  using value_type = RR;
  View1D<double> vol, pvol, dt;
  View2D<double> res, trunc, U, Uold;
  int nVar = 0;

  KOKKOS_INLINE_FUNCTION void init(value_type& v) const { RRInit(v); }
  KOKKOS_INLINE_FUNCTION void join(value_type& d, const value_type& s) const { RRJoin(d, s, nVar); }
  KOKKOS_INLINE_FUNCTION void operator()(const unsigned long iPoint, value_type& v) const {
    const double Vol = vol(iPoint) + pvol(iPoint);
    const double Delta = dt(iPoint) / Vol;
    for (int iVar = 0; iVar < nVar; iVar++) {
      const double Res = res(iPoint, iVar) + trunc(iPoint, iVar);
      /*--- CVariable::AddSolution: Solution = Solution_Old + du (Solution_Old differs from Solution where the NS
       *    wall BCs zeroed its velocity, SetVelocity_Old). ---*/
      U(iPoint, iVar) = Uold(iPoint, iVar) + -Res * Delta;
      RRAdd(v, iPoint, iVar, Res);
    }
  }
};

/*--- Batch A: Explicit_Iteration_impl<RUNGE_KUTTA_EXPLICIT> / <CLASSICAL_RK4_EXPLICIT>, Identity preconditioner. ---*/
struct ExplicitRKFunctor {
  using value_type = RR;
  View1D<double> vol, pvol, dt;
  View2D<double> res, trunc, U, Uold, Unew;
  int nVar = 0, step = 0;
  bool rk4 = false;
  double alpha = 0.0, timeCoeff = 0.0, funcCoeff = 0.0;

  KOKKOS_INLINE_FUNCTION void init(value_type& v) const { RRInit(v); }
  KOKKOS_INLINE_FUNCTION void join(value_type& d, const value_type& s) const { RRJoin(d, s, nVar); }
  KOKKOS_INLINE_FUNCTION void operator()(const unsigned long iPoint, value_type& v) const {
    const double Vol = vol(iPoint) + pvol(iPoint);
    const double Delta = dt(iPoint) / Vol;
    for (int iVar = 0; iVar < nVar; iVar++) {
      const double Res = res(iPoint, iVar) + trunc(iPoint, iVar);
      if (!rk4) {
        /*--- AddSolution(-Res*Delta*RK_AlphaCoeff): Solution = Solution_Old + du. ---*/
        U(iPoint, iVar) = Uold(iPoint, iVar) + -Res * Delta * alpha;
      } else {
        const double tmp_time = -1.0 * timeCoeff * Delta;
        const double tmp_func = -1.0 * funcCoeff * Delta;
        if (step < 3) {
          U(iPoint, iVar) = Uold(iPoint, iVar) + tmp_time * Res;     /*--- AddSolution ---*/
          Unew(iPoint, iVar) = Unew(iPoint, iVar) + tmp_func * Res;  /*--- AddSolution_New ---*/
        } else {
          U(iPoint, iVar) = Unew(iPoint, iVar) + tmp_func * Res;
        }
      }
      RRAdd(v, iPoint, iVar, Res);
    }
  }
};

/*--- Batch A: point loop of CSolver::AdaptCFLNumber (no turbulence / species solver), std::min / std::max semantics. ---*/
struct CFLStats {
  double mn, mx, sum;
};
struct AdaptCFLFunctor {
  using value_type = CFLStats;
  View1D<double> cfl, ur;
  CNEMOKokkosData::CFLAdaptParams p;

  KOKKOS_INLINE_FUNCTION void init(value_type& v) const {
    v.mn = 1e30;
    v.mx = 0.0;
    v.sum = 0.0;
  }
  KOKKOS_INLINE_FUNCTION void join(value_type& d, const value_type& s) const {
    d.mn = (d.mn < s.mn) ? d.mn : s.mn;
    d.mx = (s.mx < d.mx) ? d.mx : s.mx;
    d.sum += s.sum;
  }
  KOKKOS_INLINE_FUNCTION void operator()(const unsigned long iPoint, value_type& v) const {
    double CFL = cfl(iPoint);
    const double underRelaxationFlow = ur(iPoint);
    const double underRelaxationTurb = 1.0;
    const double underRelaxation = (underRelaxationTurb < underRelaxationFlow) ? underRelaxationTurb : underRelaxationFlow;

    double CFLFactor = 1.0;
    if (underRelaxation < 0.1 || p.reduceCFL) {
      CFLFactor = p.factorDecrease;
    } else if ((underRelaxation >= 0.1 && underRelaxation < 1.0) || !p.canIncrease) {
      CFLFactor = 1.0;
    } else {
      CFLFactor = p.factorIncrease;
    }

    if (CFL * CFLFactor <= p.cflMin) {
      CFL = p.cflMin;
      CFLFactor = p.mgFactor;
    } else if (CFL * CFLFactor >= p.cflMax) {
      CFL = p.cflMax;
      CFLFactor = p.mgFactor;
    }

    if (p.resetCFL) {
      CFL = p.cflMin;
      CFLFactor = p.mgFactor;
    }

    CFL *= CFLFactor;
    cfl(iPoint) = CFL;

    if (p.stats) {
      v.mn = (v.mn < CFL) ? v.mn : CFL; /*--- min(CFL, myCFLMin) ---*/
      v.mx = (CFL < v.mx) ? v.mx : CFL; /*--- max(CFL, myCFLMax) ---*/
      v.sum += CFL;
    }
  }
};

template <class DView>
double GroupBytes(const DView& d) {
  return double(d.span()) * sizeof(typename DView::value_type);
}

}  // namespace

void CNEMOKokkosData::EnableResident(double* linSysRes, double* linSysSol, double* jacValues, double* hostRes,
                                     double* hostSol) {
  resident = true;
  hostLinSysRes = hostRes;
  hostLinSysSol = hostSol;
  /*--- The accumulator / linear-system views alias the CSysVector / CSysMatrix device storage (same row-major /
   *    CSysMatrix block layout), so the device solver reads them without any copy. ---*/
  st.Res = View2D<double>(linSysRes, nPoint, nVar); /*--- unmanaged (pointer constructor) ---*/
  st.Sol = View2D<double>(linSysSol, nPoint, nVar);
  if (jacValues != nullptr) st.JacVal = View1D<double>(jacValues, geo.nnz * size_t(nVar) * nVar);
  st.TruncErr = View2D<double>("NEMO::Res_TruncError", nPoint, nVar);
  st.UnderRelax = View1D<double>("NEMO::UnderRelaxation", nPoint);
  for (auto& r : residency) r = Residency::HOST;
  const char* env = std::getenv("SU2_KOKKOS_NEMO_POISON");
  poison = (env != nullptr && std::atoi(env) != 0);

  /*--- Unique marker-vertex points (force / traction computations read their Primitive). ---*/
  auto hm = Kokkos::create_mirror_view_and_copy(HostSpace(), geo.vtx_node);
  std::vector<char> seen(nPoint, 0);
  for (size_t k = 0; k < hm.size(); ++k)
    if (!seen[hm(k)]) {
      seen[hm(k)] = 1;
      h_bndPoints.push_back(hm(k));
    }
  std::sort(h_bndPoints.begin(), h_bndPoints.end());
  st.bndPoints = View1D<unsigned long>("NEMO::bndPoints", h_bndPoints.size());
  st.bndStage = View2D<double>("NEMO::bndStage", h_bndPoints.size(), nPrimVar);
  Kokkos::deep_copy(st.bndPoints, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(h_bndPoints.data(),
                                                                                           h_bndPoints.size()));
}

void CNEMOKokkosData::ActivateResident() {
  if (!resident || residentActive) return;
  residentActive = true;
  for (auto& r : residency) r = Residency::HOST;
  /*--- Device copies of the accumulator inputs that SetPrimVar's previous uploads may have left: none are trusted. ---*/
  primDeviceValid = secDeviceValid = consDeviceValid = false;
  gradPrimValid = gradRecValid = limValid = sensorValid = lambdaValid = false;
  transportDeviceValid = bndViscHost = false;
}

void CNEMOKokkosData::DownloadGroup(Group g) {
  auto& P = syncPerf[g];
  Kokkos::Timer timer;
  double b = 0.0;
  const auto nd = geo.nPointDomain;
  auto as2d = [&](const View1D<double>& v) {
    return Kokkos::View<double**, Kokkos::LayoutRight, mem_space, Unmanaged>(v.data(), v.size(), 1);
  };
  auto rows = [&](double* host, const auto& d, size_t n) {
    if (d.size() == 0) return;
    const size_t cnt = n * (d.size() / d.extent(0));
    Kokkos::deep_copy(Kokkos::View<double*, HostSpace, Unmanaged>(host, cnt),
                      Kokkos::View<const double*, mem_space, Unmanaged>(d.data(), cnt));
    b += double(cnt) * sizeof(double);
  };
  switch (g) {
    case G_SOLUTION: rows(nodes->Solution.data(), st.Solution, nPoint); break;
    case G_SOLUTION_OLD: rows(nodes->Solution_Old.data(), st.Solution_Old, nPoint); break;
    case G_PRIMITIVE:
      rows(nodes->Primitive.data(), st.Primitive, nPoint);
      if (implicit) {
        rows(nodes->dPdU.data(), st.dPdU, nPoint);
        rows(nodes->dTdU.data(), st.dTdU, nPoint);
        rows(nodes->dTvedU.data(), st.dTvedU, nPoint);
      }
      rows(nodes->eves.data(), st.eves, nPoint);
      rows(nodes->Cvves.data(), st.Cvves, nPoint);
      rows(nodes->Gamma.data(), Kokkos::View<double**, Kokkos::LayoutRight, mem_space, Unmanaged>(st.Gamma.data(),
                                                                                                    nPoint, 1), nPoint);
      if (viscousVars) {
        auto* ns = static_cast<CNEMONSVariable*>(nodes);
        rows(ns->DiffusionCoeff.data(), st.DiffCoeff, nPoint);
        rows(ns->Enthalpys.data(), st.Enthalpys, nPoint);
        rows(ns->LaminarViscosity.data(), as2d(st.LamVisc), nPoint);
        rows(ns->ThermalCond.data(), as2d(st.ThermalCond), nPoint);
        rows(ns->ThermalCond_ve.data(), as2d(st.ThermalCondVe), nPoint);
      }
      /*--- Host remainder of SetPrimVar (SetPrimitive_Variables): Velocity2 from the downloaded Primitive. ---*/
      for (unsigned long i = 0; i < nPoint; ++i) nodes->SetVelocity2(i);
      bndPrimHost = true;
      break;
    case G_GRADIENT:
      if (st.GradPrim.size()) rows(nodes->GetGradient_Primitive().data(), st.GradPrim, nd);
      if (separateRecGradient && st.GradRec.size()) rows(nodes->GetGradient_Reconstruction().data(), st.GradRec, nd);
      break;
    case G_LIMITER:
      rows(nodes->GetLimiter_Primitive().data(), st.Limiter, nd);
      rows(nodes->GetSolution_Max().data(), st.PrimMax, nd);
      rows(nodes->GetSolution_Min().data(), st.PrimMin, nd);
      break;
    case G_TIMESTEP: {
      auto as2 = [&](const View1D<double>& v) {
        return Kokkos::View<double**, Kokkos::LayoutRight, mem_space, Unmanaged>(v.data(), v.size(), 1);
      };
      if (st.DeltaTime.size()) rows(nodes->Delta_Time.data(), as2(st.DeltaTime), nd);
      if (st.MaxLambdaInv.size()) rows(nodes->Max_Lambda_Inv.data(), as2(st.MaxLambdaInv), nd);
      if (st.MaxLambdaVisc.size()) rows(nodes->Max_Lambda_Visc.data(), as2(st.MaxLambdaVisc), nd);
      break;
    }
    case G_LOCALCFL:
      if (st.LocalCFL.size())
        rows(nodes->LocalCFL.data(),
             Kokkos::View<double**, Kokkos::LayoutRight, mem_space, Unmanaged>(st.LocalCFL.data(), nPoint, 1), nPoint);
      break;
    case G_NONPHYS:
      if (st.NonPhys.size()) {
        auto hnp = Kokkos::create_mirror_view_and_copy(HostSpace(), st.NonPhys);
        auto hnc = Kokkos::create_mirror_view_and_copy(HostSpace(), st.NonPhysCount);
        for (unsigned long i = 0; i < nPoint; ++i) {
          nodes->Non_Physical(i) = (hnp(i) != 0);
          nodes->Non_Physical_Counter(i) = hnc(i);
        }
        b += double(nPoint) * 3;
      }
      break;
    case G_UNDERRELAX:
      rows(nodes->UnderRelaxation.data(),
           Kokkos::View<double**, Kokkos::LayoutRight, mem_space, Unmanaged>(st.UnderRelax.data(), nPoint, 1), nPoint);
      break;
    case G_TRUNCERR: rows(nodes->Res_TruncError.data(), st.TruncErr, nPoint); break;
    case G_LINSYS:
      rows(hostLinSysRes, st.Res, nPoint);
      rows(hostLinSysSol, st.Sol, nPoint);
      break;
    case G_SENSOR:
      if (st.Sensor.size()) rows(nodes->Sensor.data(), as2d(st.Sensor), nPoint);
      break;
    case G_AUXVAR:
      if (st.AuxVar.size()) {
        rows(nodes->AuxVar.data(), st.AuxVar, nPoint);
        rows(nodes->Grad_AuxVar.data(), st.AuxGrad, nPoint);
      }
      break;
    case G_TIMEN:
      if (st.SolTimeN.size()) {
        rows(nodes->Solution_time_n.data(), st.SolTimeN, nPoint);
        rows(nodes->Solution_time_n1.data(), st.SolTimeN1, nPoint);
      }
      break;
    default: break;
  }
  P.bytes_d2h += b;
  P.d2h += timer.seconds();
  P.calls++;
}

void CNEMOKokkosData::UploadGroup(Group g) {
  auto& P = syncPerf[g];
  Kokkos::Timer timer;
  double b = 0.0;
  auto rows = [&](const auto& d, const double* host, size_t n) {
    if (d.size() == 0) return;
    const size_t cnt = n * (d.size() / d.extent(0));
    Kokkos::deep_copy(Kokkos::View<double*, mem_space, Unmanaged>(d.data(), cnt),
                      Kokkos::View<const double*, HostSpace, Unmanaged>(host, cnt));
    b += double(cnt) * sizeof(double);
  };
  auto as2 = [&](const View1D<double>& v) {
    return Kokkos::View<double**, Kokkos::LayoutRight, mem_space, Unmanaged>(v.data(), v.size(), 1);
  };
  switch (g) {
    case G_SOLUTION: rows(st.Solution, nodes->Solution.data(), nPoint); break;
    case G_SOLUTION_OLD: rows(st.Solution_Old, nodes->Solution_Old.data(), nPoint); break;
    case G_PRIMITIVE:
      rows(st.Primitive, nodes->Primitive.data(), nPoint);
      if (implicit) {
        rows(st.dPdU, nodes->dPdU.data(), nPoint);
        rows(st.dTdU, nodes->dTdU.data(), nPoint);
        rows(st.dTvedU, nodes->dTvedU.data(), nPoint);
      }
      rows(st.eves, nodes->eves.data(), nPoint);
      rows(st.Cvves, nodes->Cvves.data(), nPoint);
      rows(as2(st.Gamma), nodes->Gamma.data(), nPoint);
      if (viscousVars) {
        auto* ns = static_cast<CNEMONSVariable*>(nodes);
        rows(st.DiffCoeff, ns->DiffusionCoeff.data(), nPoint);
        rows(st.Enthalpys, ns->Enthalpys.data(), nPoint);
        rows(as2(st.LamVisc), ns->LaminarViscosity.data(), nPoint);
        rows(as2(st.ThermalCond), ns->ThermalCond.data(), nPoint);
        rows(as2(st.ThermalCondVe), ns->ThermalCond_ve.data(), nPoint);
        transportDeviceValid = true;
      }
      break;
    case G_GRADIENT:
      AllocateGradients();
      rows(st.GradPrim, nodes->GetGradient_Primitive().data(), nPoint);
      if (separateRecGradient) rows(st.GradRec, nodes->GetGradient_Reconstruction().data(), nPoint);
      break;
    case G_LIMITER:
      rows(st.Limiter, nodes->GetLimiter_Primitive().data(), nPoint);
      rows(st.PrimMax, nodes->GetSolution_Max().data(), nPoint);
      rows(st.PrimMin, nodes->GetSolution_Min().data(), nPoint);
      break;
    case G_TIMESTEP:
      if (st.DeltaTime.size()) rows(as2(st.DeltaTime), nodes->Delta_Time.data(), nPoint);
      if (st.MaxLambdaInv.size()) rows(as2(st.MaxLambdaInv), nodes->Max_Lambda_Inv.data(), nPoint);
      if (st.MaxLambdaVisc.size()) rows(as2(st.MaxLambdaVisc), nodes->Max_Lambda_Visc.data(), nPoint);
      break;
    case G_LOCALCFL:
      if (st.LocalCFL.size() == 0) st.LocalCFL = View1D<double>("NEMO::LocalCFL", nPoint);
      rows(as2(st.LocalCFL), nodes->LocalCFL.data(), nPoint);
      break;
    case G_NONPHYS:
      if (st.NonPhys.size()) {
        auto hnp = Kokkos::create_mirror_view(st.NonPhys);
        auto hnc = Kokkos::create_mirror_view(st.NonPhysCount);
        for (unsigned long i = 0; i < nPoint; ++i) {
          hnp(i) = nodes->Non_Physical(i) ? 1 : 0;
          hnc(i) = nodes->Non_Physical_Counter(i);
        }
        Kokkos::deep_copy(st.NonPhys, hnp);
        Kokkos::deep_copy(st.NonPhysCount, hnc);
        b += double(nPoint) * 3;
      }
      break;
    case G_UNDERRELAX: rows(as2(st.UnderRelax), nodes->UnderRelaxation.data(), nPoint); break;
    case G_TRUNCERR: rows(st.TruncErr, nodes->Res_TruncError.data(), nPoint); break;
    case G_LINSYS:
      rows(st.Res, hostLinSysRes, nPoint);
      rows(st.Sol, hostLinSysSol, nPoint);
      break;
    case G_SENSOR:
      if (nodes->Sensor.size() < nPoint) break; /*--- not allocated (not AUSM+M) ---*/
      if (st.Sensor.size() == 0) st.Sensor = View1D<double>("NEMO::Sensor", nPoint);
      rows(as2(st.Sensor), nodes->Sensor.data(), nPoint);
      sensorValid = true;
      break;
    case G_AUXVAR:
      if (nodes->GetnAuxVar() != 3) break; /*--- not axisymmetric ---*/
      AllocateAuxVar();
      rows(st.AuxVar, nodes->AuxVar.data(), nPoint);
      rows(st.AuxGrad, nodes->Grad_AuxVar.data(), nPoint);
      auxDeviceValid = true;
      break;
    case G_TIMEN:
      if (size_t(nodes->Solution_time_n.size()) < size_t(nPoint) * nVar) break; /*--- not dual time ---*/
      if (st.SolTimeN.size() == 0) {
        st.SolTimeN = View2D<double>("NEMO::Solution_time_n", nPoint, nVar);
        st.SolTimeN1 = View2D<double>("NEMO::Solution_time_n1", nPoint, nVar);
      }
      rows(st.SolTimeN, nodes->Solution_time_n.data(), nPoint);
      rows(st.SolTimeN1, nodes->Solution_time_n1.data(), nPoint);
      break;
    default: break;
  }
  P.bytes_h2d += b;
  P.h2d += timer.seconds();
  P.calls++;
}

void CNEMOKokkosData::PoisonGroup(Group g) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  auto fill = [&](double* p, size_t n) { std::fill(p, p + n, nan); };
  const auto nd = geo.nPointDomain;
  switch (g) {
    case G_SOLUTION: fill(nodes->Solution.data(), nodes->Solution.size()); break;
    case G_SOLUTION_OLD: fill(nodes->Solution_Old.data(), nodes->Solution_Old.size()); break;
    case G_PRIMITIVE:
      fill(nodes->Primitive.data(), nodes->Primitive.size());
      if (implicit) {
        fill(nodes->dPdU.data(), nodes->dPdU.size());
        fill(nodes->dTdU.data(), nodes->dTdU.size());
        fill(nodes->dTvedU.data(), nodes->dTvedU.size());
      }
      fill(nodes->eves.data(), nodes->eves.size());
      fill(nodes->Cvves.data(), nodes->Cvves.size());
      fill(nodes->Gamma.data(), nodes->Gamma.size());
      fill(nodes->Velocity2.data(), nodes->Velocity2.size());
      if (viscousVars) {
        auto* ns = static_cast<CNEMONSVariable*>(nodes);
        fill(ns->DiffusionCoeff.data(), ns->DiffusionCoeff.size());
        fill(ns->Enthalpys.data(), ns->Enthalpys.size());
        fill(ns->LaminarViscosity.data(), ns->LaminarViscosity.size());
        fill(ns->ThermalCond.data(), ns->ThermalCond.size());
        fill(ns->ThermalCond_ve.data(), ns->ThermalCond_ve.size());
      }
      break;
    case G_GRADIENT:
      fill(nodes->GetGradient_Primitive().data(), nd * size_t(nPrimVarGrad) * nDim);
      if (separateRecGradient) fill(nodes->GetGradient_Reconstruction().data(), nd * size_t(nPrimVarGrad) * nDim);
      break;
    case G_LIMITER:
      fill(nodes->GetLimiter_Primitive().data(), nd * size_t(nPrimVarGrad));
      fill(nodes->GetSolution_Max().data(), nd * size_t(nPrimVarGrad));
      fill(nodes->GetSolution_Min().data(), nd * size_t(nPrimVarGrad));
      break;
    case G_TIMESTEP:
      fill(nodes->Delta_Time.data(), nd);
      fill(nodes->Max_Lambda_Inv.data(), nd);
      if (st.MaxLambdaVisc.size()) fill(nodes->Max_Lambda_Visc.data(), nd);
      break;
    case G_LOCALCFL: fill(nodes->LocalCFL.data(), nPoint); break;
    case G_NONPHYS:
      for (unsigned long i = 0; i < nPoint; ++i) {
        nodes->Non_Physical(i) = true;
        nodes->Non_Physical_Counter(i) = 0xFFFF;
      }
      break;
    case G_UNDERRELAX: fill(nodes->UnderRelaxation.data(), nPoint); break;
    case G_TRUNCERR: fill(nodes->Res_TruncError.data(), nodes->Res_TruncError.size()); break;
    case G_LINSYS:
      fill(hostLinSysRes, size_t(nPoint) * nVar);
      fill(hostLinSysSol, size_t(nPoint) * nVar);
      break;
    case G_SENSOR: fill(nodes->Sensor.data(), nodes->Sensor.size()); break;
    case G_AUXVAR:
      fill(nodes->AuxVar.data(), nodes->AuxVar.size());
      fill(nodes->Grad_AuxVar.data(), nodes->Grad_AuxVar.size());
      break;
    case G_TIMEN:
      fill(nodes->Solution_time_n.data(), nodes->Solution_time_n.size());
      fill(nodes->Solution_time_n1.data(), nodes->Solution_time_n1.size());
      break;
    default: break;
  }
}

void CNEMOKokkosData::EnsureHost(Group g) {
  if (!ResidentActive() || residency[g] != Residency::DEVICE) return;
  DownloadGroup(g);
  residency[g] = Residency::BOTH;
}

void CNEMOKokkosData::EnsureDevice(Group g) {
  if (!ResidentActive() || residency[g] != Residency::HOST) return;
  UploadGroup(g);
  residency[g] = Residency::BOTH;
}

void CNEMOKokkosData::MarkDeviceWritten(Group g) {
  if (!ResidentActive()) return;
  residency[g] = Residency::DEVICE;
  if (g == G_PRIMITIVE) bndPrimHost = false;
  if (g == G_PRIMITIVE || g == G_GRADIENT) bndViscHost = false;
  if (poison) PoisonGroup(g);
}

void CNEMOKokkosData::SetForceMarkers(const std::vector<int>& markers) {
  /*--- Zero-copy steady state 2(a) step 1: the per-iteration HostSyncScope::BOUNDARY sync (Pressure_Forces,
   *    Momentum_Forces, Friction_Forces, ComputeVertexTractions, SetAnalyzeSurface) reads the Primitive rows only at
   *    the points of these markers; the rows of the other marker points (far field, symmetry, supersonic outlet,
   *    periodic, send/receive) are not read on the host between outputs (EnsureHostAll downloads everything).
   *    Host arithmetic is unchanged (same values at the points that are read) -> bitwise. ---*/
  if (h_markerPtr.empty()) {
    auto hm = Kokkos::create_mirror_view_and_copy(HostSpace(), geo.marker_ptr);
    h_markerPtr.assign(hm.data(), hm.data() + geo.nMarker + 1);
  }
  auto hn = Kokkos::create_mirror_view_and_copy(HostSpace(), geo.vtx_node);
  std::vector<char> seen(nPoint, 0);
  nBndPointsAll = h_bndPoints.size();
  h_bndPoints.clear();
  for (int m : markers)
    for (auto k = h_markerPtr[m]; k < h_markerPtr[m + 1]; ++k)
      if (!seen[hn(k)]) {
        seen[hn(k)] = 1;
        h_bndPoints.push_back(hn(k));
      }
  std::sort(h_bndPoints.begin(), h_bndPoints.end());
  st.bndPoints = View1D<unsigned long>("NEMO::bndPoints", h_bndPoints.size());
  st.bndStage = View2D<double>("NEMO::bndStage", h_bndPoints.size(), nPrimVar);
  Kokkos::deep_copy(st.bndPoints, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(h_bndPoints.data(),
                                                                                           h_bndPoints.size()));
  bndPrimHost = false;
}

void CNEMOKokkosData::SetViscousWallMarkers(const std::vector<int>& markers) {
  if (!viscousVars) return;
  if (h_markerPtr.empty()) {
    auto hm = Kokkos::create_mirror_view_and_copy(HostSpace(), geo.marker_ptr);
    h_markerPtr.assign(hm.data(), hm.data() + geo.nMarker + 1);
  }
  auto hn = Kokkos::create_mirror_view_and_copy(HostSpace(), geo.vtx_node);
  std::vector<char> seen(nPoint, 0);
  h_viscWallPoints.clear();
  for (int m : markers)
    for (auto k = h_markerPtr[m]; k < h_markerPtr[m + 1]; ++k)
      if (!seen[hn(k)]) {
        seen[hn(k)] = 1;
        h_viscWallPoints.push_back(hn(k));
      }
  std::sort(h_viscWallPoints.begin(), h_viscWallPoints.end());
  const size_t rowLen = size_t(nPrimVarGrad) * nDim + 3 + 2 * size_t(nSpecies);
  st.viscWallPoints = View1D<unsigned long>("NEMO::viscWallPoints", h_viscWallPoints.size());
  st.viscWallStage = View2D<double>("NEMO::viscWallStage", h_viscWallPoints.size(), rowLen);
  Kokkos::deep_copy(st.viscWallPoints, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(
                                           h_viscWallPoints.data(), h_viscWallPoints.size()));
}

void CNEMOKokkosData::EnsureHostBoundaryPrimitive() {
  if (!ResidentActive()) return;
  /*--- NS (Friction_Forces): gradient + transport rows of the viscous-wall points. ---*/
  const bool viscStale = (residency[G_PRIMITIVE] == Residency::DEVICE || residency[G_GRADIENT] == Residency::DEVICE);
  if (viscousVars && !bndViscHost && viscStale && !h_viscWallPoints.empty()) {
    auto& P = syncPerf[N_GROUPS];
    Kokkos::Timer timer;
    const auto pts = st.viscWallPoints;
    const auto stage = st.viscWallStage;
    const auto G = st.GradPrim;
    const auto mu = st.LamVisc, k = st.ThermalCond, kve = st.ThermalCondVe;
    const auto Ds = st.DiffCoeff, hs = st.Enthalpys;
    const int ng = nPrimVarGrad * nDim, ns = nSpecies;
    const bool haveGrad = (G.size() != 0);
    Kokkos::parallel_for(
        "NEMO::GatherViscousWall", Kokkos::RangePolicy<exec_space>(0, h_viscWallPoints.size()),
        KOKKOS_LAMBDA(const size_t q) {
          const auto i = pts(q);
          if (haveGrad)
            for (int v = 0; v < ng; ++v) stage(q, v) = G.data()[i * ng + v];
          stage(q, ng) = mu(i);
          stage(q, ng + 1) = k(i);
          stage(q, ng + 2) = kve(i);
          for (int s = 0; s < ns; ++s) {
            stage(q, ng + 3 + s) = Ds(i, s);
            stage(q, ng + 3 + ns + s) = hs(i, s);
          }
        });
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), st.viscWallStage);
    auto* nsv = static_cast<CNEMONSVariable*>(nodes);
    double* hg = nodes->GetGradient_Primitive().data();
    for (size_t q = 0; q < h_viscWallPoints.size(); ++q) {
      const auto i = h_viscWallPoints[q];
      if (haveGrad && residency[G_GRADIENT] == Residency::DEVICE) std::memcpy(hg + i * ng, &h(q, 0), ng * sizeof(double));
      if (residency[G_PRIMITIVE] == Residency::DEVICE) {
        nsv->LaminarViscosity(i) = h(q, ng);
        nsv->ThermalCond(i) = h(q, ng + 1);
        nsv->ThermalCond_ve(i) = h(q, ng + 2);
        for (int s = 0; s < ns; ++s) {
          nsv->DiffusionCoeff(i, s) = h(q, ng + 3 + s);
          nsv->Enthalpys(i, s) = h(q, ng + 3 + ns + s);
        }
      }
    }
    P.bytes_d2h += double(h.size()) * sizeof(double);
    bndViscHost = true;
    P.d2h += timer.seconds();
  }
  if (residency[G_PRIMITIVE] != Residency::DEVICE || bndPrimHost) return;
  auto& P = syncPerf[N_GROUPS];
  Kokkos::Timer timer;
  const auto nb = h_bndPoints.size();
  if (nb > 0) {
    const auto pts = st.bndPoints;
    const auto stage = st.bndStage;
    const auto V = st.Primitive;
    const int npv = nPrimVar;
    Kokkos::parallel_for(
        "NEMO::GatherBoundaryPrimitive", Kokkos::RangePolicy<exec_space>(0, nb), KOKKOS_LAMBDA(const size_t k) {
          for (int v = 0; v < npv; ++v) stage(k, v) = V(pts(k), v);
        });
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), st.bndStage);
    for (size_t k = 0; k < nb; ++k)
      std::memcpy(&nodes->Primitive(h_bndPoints[k], 0), &h(k, 0), size_t(nPrimVar) * sizeof(double));
    P.bytes_d2h += double(nb) * nPrimVar * sizeof(double);
  }
  bndPrimHost = true;
  P.d2h += timer.seconds();
  P.calls++;
}

void CNEMOKokkosData::EnsureHostAll() {
  if (!ResidentActive()) return;
  outputThisIteration = true;
  for (int g = 0; g < N_GROUPS; ++g) EnsureHost(static_cast<Group>(g));
}

/*====================== Step 10: MPI halo exchange of device-resident arrays ======================*/

double* CNEMOKokkosData::HaloDevicePtr(HaloQuantity q, size_t& rowLen, Group& g) {
  switch (q) {
    case HaloQuantity::SOLUTION:
      rowLen = nVar;
      g = G_SOLUTION;
      return st.Solution.data();
    case HaloQuantity::GRADIENT_PRIMITIVE:
      rowLen = size_t(nPrimVarGrad) * nDim;
      g = G_GRADIENT;
      return st.GradPrim.data();
    case HaloQuantity::GRADIENT_RECONSTRUCTION:
      rowLen = size_t(nPrimVarGrad) * nDim;
      g = G_GRADIENT;
      return st.GradRec.data(); /*--- == GradPrim if Gradient_Reconstruction is not a separate container ---*/
    case HaloQuantity::LIMITER:
      rowLen = nPrimVarGrad;
      g = G_LIMITER;
      return st.Limiter.data();
    case HaloQuantity::SENSOR:
      rowLen = 1;
      g = G_SENSOR;
      return st.Sensor.data();
    case HaloQuantity::AUXVAR_GRADIENT:
      rowLen = size_t(3) * nDim;
      g = G_AUXVAR;
      return st.AuxGrad.data();
    case HaloQuantity::SOLUTION_OLD:
      rowLen = nVar;
      g = G_SOLUTION_OLD;
      return st.Solution_Old.data();
    case HaloQuantity::LAMBDA:
      rowLen = 1;
      g = N_GROUPS; /*--- not a residency group (domain rows downloaded by ComputeMaxEigenvalue) ---*/
      return st.Lambda.data();
    case HaloQuantity::DELTA_TIME:
      rowLen = 1;
      g = G_TIMESTEP;
      return st.DeltaTime.data();
    case HaloQuantity::PRIMITIVE:
      rowLen = nPrimVar;
      g = G_PRIMITIVE;
      return st.Primitive.data();
    case HaloQuantity::PRIM_MAX:
      rowLen = nPrimVarGrad;
      g = N_GROUPS; /*--- limiter scratch (written by LimiterPeriodicInit / ComputeLimiter), no residency group ---*/
      return st.PrimMax.data();
    case HaloQuantity::PRIM_MIN:
      rowLen = nPrimVarGrad;
      g = N_GROUPS;
      return st.PrimMin.data();
    default: break;
  }
  SU2_MPI::Error("CNEMOKokkosData: unknown halo quantity.", CURRENT_FUNCTION);
  return nullptr;
}

double* CNEMOKokkosData::HaloHostPtr(HaloQuantity q) {
  switch (q) {
    case HaloQuantity::SOLUTION: return nodes->Solution.data();
    case HaloQuantity::GRADIENT_PRIMITIVE: return nodes->GetGradient_Primitive().data();
    case HaloQuantity::GRADIENT_RECONSTRUCTION: return nodes->GetGradient_Reconstruction().data();
    case HaloQuantity::LIMITER: return nodes->GetLimiter_Primitive().data();
    case HaloQuantity::SENSOR: return nodes->Sensor.data();
    case HaloQuantity::AUXVAR_GRADIENT: return nodes->Grad_AuxVar.data();
    case HaloQuantity::SOLUTION_OLD: return nodes->Solution_Old.data();
    case HaloQuantity::LAMBDA: return nodes->Lambda.data();
    case HaloQuantity::DELTA_TIME: return nodes->Delta_Time.data();
    case HaloQuantity::PRIMITIVE: return nodes->Primitive.data();
    case HaloQuantity::PRIM_MAX: return nodes->GetSolution_Max().data();
    case HaloQuantity::PRIM_MIN: return nodes->GetSolution_Min().data();
    default: break;
  }
  return nullptr;
}

void CNEMOKokkosData::HaloPlan(CGeometry* geometry) {
  /*--- Geometry metadata, uploaded once per (solver, geometry). A different geometry object (e.g. after a mesh
   *    re-partition) or different halo counts rebuild the plan. ---*/
  const auto nSend = geometry->nP2PSend > 0 ? static_cast<unsigned long>(geometry->nPoint_P2PSend[geometry->nP2PSend]) : 0ul;
  const auto nRecv = geometry->nP2PRecv > 0 ? static_cast<unsigned long>(geometry->nPoint_P2PRecv[geometry->nP2PRecv]) : 0ul;
  if (haloGeometry == geometry && haloSend == nSend && haloRecv == nRecv) return;
  haloGeometry = geometry;
  haloSend = nSend;
  haloRecv = nRecv;
  haloSendIdx = View1D<unsigned long>("NEMO::haloSendIdx", nSend);
  haloRecvIdx = View1D<unsigned long>("NEMO::haloRecvIdx", nRecv);
  if (nSend) Kokkos::deep_copy(haloSendIdx, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(geometry->Local_Point_P2PSend, nSend));
  if (nRecv) Kokkos::deep_copy(haloRecvIdx, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(geometry->Local_Point_P2PRecv, nRecv));
  /*--- Buffers sized for the widest row (gradient). ---*/
  const size_t maxRow = std::max<size_t>(nVar, size_t(nPrimVarGrad) * nDim);
  haloSendBuf = View1D<double>("NEMO::haloSendBuf", nSend * maxRow);
  haloRecvBuf = View1D<double>("NEMO::haloRecvBuf", nRecv * maxRow);
  haloHostSend.assign(nSend * maxRow, 0.0);
  haloHostRecv.assign(nRecv * maxRow, 0.0);
}

void CNEMOKokkosData::HaloExchange(CGeometry* geometry, HaloQuantity q, bool gpuAware) {
  if (!ResidentActive()) return;
#ifdef HAVE_MPI
  Kokkos::Timer timer;
  HaloPlan(geometry);
  if (haloSend == 0 && haloRecv == 0) return;
  size_t rowLen = 0;
  Group g = G_SOLUTION;
  double* dev = HaloDevicePtr(q, rowLen, g);
  if (dev == nullptr) SU2_MPI::Error("CNEMOKokkosData: halo exchange of an unallocated device array.", CURRENT_FUNCTION);
  /*--- The device copy must be current (it is, after the stage that produced it). ---*/
  EnsureDevice(g);
  auto& P = syncPerf[SYNC_HALO];
  auto& H = haloPerf[static_cast<int>(q)];

  const int nMsgR = geometry->nP2PRecv, nMsgS = geometry->nP2PSend;
  std::vector<SU2_MPI::Request> reqR(nMsgR), reqS(nMsgS);
  double* recvBase = gpuAware ? haloRecvBuf.data() : haloHostRecv.data();
  double* sendBase = gpuAware ? haloSendBuf.data() : haloHostSend.data();

  /*--- Receives first (CGeometry::PostP2PRecvs: source = Neighbors_P2PRecv, tag = source + 1). ---*/
  for (int iMsg = 0; iMsg < nMsgR; ++iMsg) {
    const auto offset = static_cast<unsigned long>(geometry->nPoint_P2PRecv[iMsg]) * rowLen;
    const int count = static_cast<int>((geometry->nPoint_P2PRecv[iMsg + 1] - geometry->nPoint_P2PRecv[iMsg]) * rowLen);
    const int source = geometry->Neighbors_P2PRecv[iMsg];
    SU2_MPI::Irecv(recvBase + offset, count, MPI_DOUBLE, source, source + 1, SU2_MPI::GetComm(), &reqR[iMsg]);
  }

  /*--- Device gather of the send rows (point-major, the CPU packing order of the row). ---*/
  if (haloSend > 0) {
    const auto idx = haloSendIdx;
    const auto buf = haloSendBuf;
    const size_t rl = rowLen;
    Kokkos::parallel_for(
        "NEMO::HaloPack", Kokkos::RangePolicy<exec_space>(0, haloSend * rl), KOKKOS_LAMBDA(const unsigned long k) {
          buf(k) = dev[idx(k / rl) * rl + k % rl];
        });
    Kokkos::fence("NEMO::HaloPack");
    if (!gpuAware) {
      Kokkos::deep_copy(Kokkos::View<double*, HostSpace, Unmanaged>(haloHostSend.data(), haloSend * rl),
                        Kokkos::subview(haloSendBuf, std::make_pair(size_t(0), haloSend * rl)));
      P.bytes_d2h += double(haloSend) * rl * sizeof(double);
    }
  }
  /*--- Sends (CGeometry::PostP2PSends: dest = Neighbors_P2PSend, tag = rank + 1). ---*/
  const int rank = SU2_MPI::GetRank();
  for (int iMsg = 0; iMsg < nMsgS; ++iMsg) {
    const auto offset = static_cast<unsigned long>(geometry->nPoint_P2PSend[iMsg]) * rowLen;
    const int count = static_cast<int>((geometry->nPoint_P2PSend[iMsg + 1] - geometry->nPoint_P2PSend[iMsg]) * rowLen);
    SU2_MPI::Isend(sendBase + offset, count, MPI_DOUBLE, geometry->Neighbors_P2PSend[iMsg], rank + 1,
                   SU2_MPI::GetComm(), &reqS[iMsg]);
  }
  if (nMsgR > 0) SU2_MPI::Waitall(nMsgR, reqR.data(), MPI_STATUSES_IGNORE);

  /*--- Device scatter of the receive rows. ---*/
  if (haloRecv > 0) {
    const size_t rl = rowLen;
    if (!gpuAware) {
      Kokkos::deep_copy(Kokkos::subview(haloRecvBuf, std::make_pair(size_t(0), haloRecv * rl)),
                        Kokkos::View<const double*, HostSpace, Unmanaged>(haloHostRecv.data(), haloRecv * rl));
      P.bytes_h2d += double(haloRecv) * rl * sizeof(double);
    }
    const auto idx = haloRecvIdx;
    const auto buf = haloRecvBuf;
    Kokkos::parallel_for(
        "NEMO::HaloUnpack", Kokkos::RangePolicy<exec_space>(0, haloRecv * rl), KOKKOS_LAMBDA(const unsigned long k) {
          dev[idx(k / rl) * rl + k % rl] = buf(k);
        });
    Kokkos::fence("NEMO::HaloUnpack");
  }
  if (nMsgS > 0) SU2_MPI::Waitall(nMsgS, reqS.data(), MPI_STATUSES_IGNORE);

  /*--- The host copy (if it was current) no longer has the received rows. ---*/
  if (residency[g] != Residency::DEVICE) MarkDeviceWritten(g);
  P.calls++;
  H.calls++;
  H.bytesMsg += double(haloSend) * rowLen * sizeof(double);
  H.time += timer.seconds();
#else
  (void)geometry;
  (void)q;
  (void)gpuAware;
#endif
}

void CNEMOKokkosData::DownloadHaloRows(HaloQuantity q, const unsigned long* rows, unsigned long n) {
  if (!ResidentActive() || n == 0) return;
  /*--- Legacy path: send rows into the host container (device gather + one compact DtoH), host comms next. ---*/
  GatherRowsToHost(q, rows, n);
  size_t rowLen = 0;
  Group g = G_SOLUTION;
  (void)HaloDevicePtr(q, rowLen, g);
  haloPerf[static_cast<int>(q)].bytesMsg += double(n) * rowLen * sizeof(double);
}

void CNEMOKokkosData::UploadHaloRows(HaloQuantity q, const unsigned long* rows, unsigned long n) {
  if (!ResidentActive() || n == 0) return;
  ScatterRowsFromHost(q, rows, n);
  haloPerf[static_cast<int>(q)].calls++;
}

void CNEMOKokkosData::GatherRowsToHost(HaloQuantity q, const unsigned long* rows, unsigned long n) {
  if (n == 0) return;
  size_t rowLen = 0;
  Group g = G_SOLUTION;
  double* dev = HaloDevicePtr(q, rowLen, g);
  double* host = HaloHostPtr(q);
  if (g != N_GROUPS) EnsureDevice(g);
  auto& P = syncPerf[SYNC_HALO];
  View1D<unsigned long> idx("NEMO::haloIdx", n);
  View2D<double> stage("NEMO::haloStage", n, rowLen);
  Kokkos::deep_copy(idx, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(rows, n));
  Kokkos::parallel_for(
      "NEMO::GatherRows", Kokkos::RangePolicy<exec_space>(0, n), KOKKOS_LAMBDA(const unsigned long k) {
        for (size_t v = 0; v < rowLen; ++v) stage(k, v) = dev[idx(k) * rowLen + v];
      });
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), stage);
  for (unsigned long k = 0; k < n; ++k) std::memcpy(host + rows[k] * rowLen, &h(k, 0), rowLen * sizeof(double));
  P.bytes_d2h += double(n) * rowLen * sizeof(double);
}

void CNEMOKokkosData::ScatterRowsFromHost(HaloQuantity q, const unsigned long* rows, unsigned long n) {
  if (n == 0) return;
  size_t rowLen = 0;
  Group g = G_SOLUTION;
  double* dev = HaloDevicePtr(q, rowLen, g);
  const double* host = HaloHostPtr(q);
  auto& P = syncPerf[SYNC_HALO];
  View1D<unsigned long> idx("NEMO::haloIdx", n);
  View2D<double> stage("NEMO::haloStage", n, rowLen);
  auto h = Kokkos::create_mirror_view(stage);
  for (unsigned long k = 0; k < n; ++k) std::memcpy(&h(k, 0), host + rows[k] * rowLen, rowLen * sizeof(double));
  Kokkos::deep_copy(stage, h);
  Kokkos::deep_copy(idx, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(rows, n));
  Kokkos::parallel_for(
      "NEMO::ScatterRows", Kokkos::RangePolicy<exec_space>(0, n), KOKKOS_LAMBDA(const unsigned long k) {
        for (size_t v = 0; v < rowLen; ++v) dev[idx(k) * rowLen + v] = stage(k, v);
      });
  Kokkos::fence();
  P.bytes_h2d += double(n) * rowLen * sizeof(double);
  if (ResidentActive() && g != N_GROUPS && residency[g] != Residency::DEVICE) MarkDeviceWritten(g);
  P.calls++;
}

void CNEMOKokkosData::AccumulatorRows(bool toHost, const unsigned long* rows, unsigned long n, double* hostRes,
                                      double* hostJac, const unsigned long* hostRowPtr) {
  if (n == 0) return;
  auto& P = syncPerf[SYNC_HALO];
  const size_t nv = nVar, bs = size_t(nVar) * nVar;
  View1D<unsigned long> idx("NEMO::perIdx", n);
  Kokkos::deep_copy(idx, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(rows, n));
  double* res = st.Res.data();

  /*--- Residual rows (nVar each). ---*/
  View2D<double> rstage("NEMO::perResStage", n, nv);
  auto hr = Kokkos::create_mirror_view(rstage);
  if (toHost) {
    Kokkos::parallel_for(
        "NEMO::PerResGather", Kokkos::RangePolicy<exec_space>(0, n), KOKKOS_LAMBDA(const unsigned long k) {
          for (size_t v = 0; v < nv; ++v) rstage(k, v) = res[idx(k) * nv + v];
        });
    Kokkos::deep_copy(hr, rstage);
    for (unsigned long k = 0; k < n; ++k) std::memcpy(hostRes + rows[k] * nv, &hr(k, 0), nv * sizeof(double));
    P.bytes_d2h += double(n) * nv * sizeof(double);
  } else {
    for (unsigned long k = 0; k < n; ++k) std::memcpy(&hr(k, 0), hostRes + rows[k] * nv, nv * sizeof(double));
    Kokkos::deep_copy(rstage, hr);
    Kokkos::parallel_for(
        "NEMO::PerResScatter", Kokkos::RangePolicy<exec_space>(0, n), KOKKOS_LAMBDA(const unsigned long k) {
          for (size_t v = 0; v < nv; ++v) res[idx(k) * nv + v] = rstage(k, v);
        });
    P.bytes_h2d += double(n) * nv * sizeof(double);
  }

  /*--- Jacobian block rows (row_ptr[r] .. row_ptr[r+1], contiguous in the CSysMatrix layout). ---*/
  if (hostJac != nullptr) {
    std::vector<unsigned long> hoff(n + 1, 0), hstart(n);
    for (unsigned long k = 0; k < n; ++k) {
      hstart[k] = hostRowPtr[rows[k]];
      hoff[k + 1] = hoff[k] + (hostRowPtr[rows[k] + 1] - hostRowPtr[rows[k]]);
    }
    const unsigned long nb = hoff[n];
    View1D<unsigned long> off("NEMO::perJacOff", n + 1), start("NEMO::perJacStart", n);
    Kokkos::deep_copy(off, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(hoff.data(), n + 1));
    Kokkos::deep_copy(start, Kokkos::View<const unsigned long*, HostSpace, Unmanaged>(hstart.data(), n));
    View1D<double> jstage("NEMO::perJacStage", nb * bs);
    auto hj = Kokkos::create_mirror_view(jstage);
    double* jac = st.JacVal.data();
    if (toHost) {
      Kokkos::parallel_for(
          "NEMO::PerJacGather", Kokkos::RangePolicy<exec_space>(0, n), KOKKOS_LAMBDA(const unsigned long k) {
            const unsigned long len = (off(k + 1) - off(k)) * bs;
            for (unsigned long e = 0; e < len; ++e) jstage(off(k) * bs + e) = jac[start(k) * bs + e];
          });
      Kokkos::deep_copy(hj, jstage);
      for (unsigned long k = 0; k < n; ++k)
        std::memcpy(hostJac + hstart[k] * bs, &hj(hoff[k] * bs), (hoff[k + 1] - hoff[k]) * bs * sizeof(double));
      P.bytes_d2h += double(nb) * bs * sizeof(double);
    } else {
      for (unsigned long k = 0; k < n; ++k)
        std::memcpy(&hj(hoff[k] * bs), hostJac + hstart[k] * bs, (hoff[k + 1] - hoff[k]) * bs * sizeof(double));
      Kokkos::deep_copy(jstage, hj);
      Kokkos::parallel_for(
          "NEMO::PerJacScatter", Kokkos::RangePolicy<exec_space>(0, n), KOKKOS_LAMBDA(const unsigned long k) {
            const unsigned long len = (off(k + 1) - off(k)) * bs;
            for (unsigned long e = 0; e < len; ++e) jac[start(k) * bs + e] = jstage(off(k) * bs + e);
          });
      P.bytes_h2d += double(nb) * bs * sizeof(double);
    }
  }
  Kokkos::fence();
  P.calls++;
}

void CNEMOKokkosData::PrepareImplicit(bool impl, ResidualReduction& red) {
  Kokkos::Timer timer;
  EnsureDevice(G_TRUNCERR);
  if (impl && st.JacVal.size() == 0) SU2_MPI::Error("CNEMOKokkosData: no device Jacobian.", CURRENT_FUNCTION);
  PrepareFunctor f;
  f.vol = geo.volume;
  f.pvol = geo.periodic_volume;
  f.dt = st.DeltaTime;
  f.jac = st.JacVal;
  f.dia = geo.dia_ptr;
  f.res = st.Res;
  f.sol = st.Sol;
  f.trunc = st.TruncErr;
  f.nVar = nVar;
  f.implicit = impl;
  Kokkos::parallel_reduce("NEMO::PrepareImplicitIteration", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain), f,
                          red);
  /*--- ImplicitEuler_Iteration: halo rows of the RHS and of the initial guess are zero. ---*/
  if (geo.nPoint > geo.nPointDomain) {
    const auto res = st.Res, sol = st.Sol;
    const int nv = nVar;
    Kokkos::parallel_for(
        "NEMO::ZeroHaloRows", Kokkos::RangePolicy<exec_space>(geo.nPointDomain, geo.nPoint),
        KOKKOS_LAMBDA(const unsigned long i) {
          for (int v = 0; v < nv; ++v) res(i, v) = sol(i, v) = 0.0;
        });
  }
  Kokkos::fence();
  /*--- The accumulator has been consumed: st.Res now holds the RHS of the linear system. ---*/
  accOnDevice = false;
  hostAccZero = false;
  MarkDeviceWritten(G_LINSYS);
  MarkDeviceWritten(G_TRUNCERR);
  updatePerf.prepare += timer.seconds();
  updatePerf.nPrepare++;
}

void CNEMOKokkosData::ExplicitEulerUpdate(ResidualReduction& red) {
  Kokkos::Timer timer;
  EnsureDevice(G_TRUNCERR);
  EnsureDevice(G_SOLUTION);
  EnsureDevice(G_SOLUTION_OLD);
  ExplicitFunctor f;
  f.vol = geo.volume;
  f.pvol = geo.periodic_volume;
  f.dt = st.DeltaTime;
  f.res = st.Res;
  f.trunc = st.TruncErr;
  f.U = st.Solution;
  f.Uold = st.Solution_Old;
  f.nVar = nVar;
  Kokkos::parallel_reduce("NEMO::ExplicitEulerIteration", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain), f,
                          red);
  Kokkos::fence();
  accOnDevice = false;
  hostAccZero = false;
  MarkDeviceWritten(G_LINSYS); /*--- the residual stays in st.Res (volume output reads LinSysRes) ---*/
  MarkDeviceWritten(G_SOLUTION);
  updatePerf.explicitUpd += timer.seconds();
  updatePerf.nExplicit++;
}

void CNEMOKokkosData::CompleteImplicit(double allowableRatio) {
  Kokkos::Timer timer;
  EnsureDevice(G_SOLUTION);
  EnsureDevice(G_SOLUTION_OLD);
  const auto U = st.Solution, Uold = st.Solution_Old, sol = st.Sol;
  const auto UR = st.UnderRelax;
  const int nv = nVar, ns = nSpecies;
  const double eps = EPS;
  /*--- CNEMOEulerSolver::ComputeUnderRelaxationFactor then AddSolution(UR * LinSysSol), per point (the factor of a
   *    point reads only that point's Solution and LinSysSol, before its update). ---*/
  Kokkos::parallel_for(
      "NEMO::CompleteImplicitIteration", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint) {
        double localUnderRelaxation = 1.0;
        double num = 0.0;
        double denom = 0.0;
        for (int iVar = 0; iVar < nv; iVar++) {
          if (iVar < ns) {
            num += Kokkos::fabs(sol(iPoint, iVar));
            denom += Kokkos::fabs(U(iPoint, iVar));
            if (iVar == (ns - 1)) {
              const double ratio = (num / (denom + eps));
              if (ratio > allowableRatio) {
                const double r = allowableRatio / ratio;
                localUnderRelaxation = (localUnderRelaxation < r) ? localUnderRelaxation : r; /*--- std::min(r, ur) ---*/
              }
            }
            /*--- The CPU energy branch (iVar == nVar-2) is nested in iVar < nSpecies and never taken. ---*/
            if (iVar == (nv - 2)) {
              const double ratio = Kokkos::fabs(sol(iPoint, iVar)) / (Kokkos::fabs(U(iPoint, iVar)) + eps);
              if (ratio > allowableRatio) {
                const double r = allowableRatio / ratio;
                localUnderRelaxation = (localUnderRelaxation < r) ? localUnderRelaxation : r;
              }
            }
          }
        }
        if (localUnderRelaxation < 1e-10) localUnderRelaxation = 0.0;
        UR(iPoint) = localUnderRelaxation;
        /*--- CVariable::AddSolution: Solution = Solution_Old + UR*du (Solution_Old differs from Solution where the
         *    NS wall BCs zeroed its velocity, SetVelocity_Old). ---*/
        for (int iVar = 0; iVar < nv; iVar++) U(iPoint, iVar) = Uold(iPoint, iVar) + localUnderRelaxation * sol(iPoint, iVar);
      });
  Kokkos::fence();
  MarkDeviceWritten(G_SOLUTION);
  MarkDeviceWritten(G_UNDERRELAX);
  updatePerf.complete += timer.seconds();
  updatePerf.nComplete++;
}

void CNEMOKokkosData::SetOldSolution() {
  Kokkos::Timer timer;
  EnsureDevice(G_SOLUTION);
  Kokkos::deep_copy(st.Solution_Old, st.Solution);
  if (rk4) {
    /*--- CMultiGridIntegration calls Set_NewSolution right after Set_OldSolution (same Solution). ---*/
    if (st.SolutionNew.size() == 0) st.SolutionNew = View2D<double>("NEMO::Solution_New", nPoint, nVar);
    Kokkos::deep_copy(st.SolutionNew, st.Solution);
  }
  Kokkos::fence();
  MarkDeviceWritten(G_SOLUTION_OLD);
  updatePerf.oldSol += timer.seconds();
  updatePerf.nOldSol++;
}

double CNEMOKokkosData::ResidualSquaredNorm(unsigned long nElmDomain) const {
  Kokkos::Timer timer;
  const double* r = st.Res.data();
  double sum = 0.0;
  Kokkos::parallel_reduce(
      "NEMO::ResidualSquaredNorm", Kokkos::RangePolicy<exec_space>(0, nElmDomain),
      KOKKOS_LAMBDA(const unsigned long k, double& a) { a += r[k] * r[k]; }, sum);
  updatePerf.norm += timer.seconds();
  updatePerf.nNorm++;
  return sum;
}

void CNEMOKokkosData::TimeSteppingFill(double cfl, double dt) {
  const auto c = st.LocalCFL, d = st.DeltaTime;
  Kokkos::parallel_for(
      "NEMO::TimeSteppingFill", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long i) {
        c(i) = cfl;
        d(i) = dt;
      });
  Kokkos::fence();
  MarkDeviceWritten(G_LOCALCFL);
  MarkDeviceWritten(G_TIMESTEP);
}

void CNEMOKokkosData::ExplicitRKUpdate(bool rk4Update, unsigned short iRKStep, double alpha, double timeCoeff,
                                       double funcCoeff, ResidualReduction& red) {
  Kokkos::Timer timer;
  EnsureDevice(G_TRUNCERR);
  EnsureDevice(G_SOLUTION);
  EnsureDevice(G_SOLUTION_OLD);
  if (rk4Update && st.SolutionNew.size() == 0)
    SU2_MPI::Error("CNEMOKokkosData: classical RK4 without the device Solution_New.", CURRENT_FUNCTION);
  ExplicitRKFunctor f;
  f.vol = geo.volume;
  f.pvol = geo.periodic_volume;
  f.dt = st.DeltaTime;
  f.res = st.Res;
  f.trunc = st.TruncErr;
  f.U = st.Solution;
  f.Uold = st.Solution_Old;
  f.Unew = st.SolutionNew;
  f.nVar = nVar;
  f.step = iRKStep;
  f.rk4 = rk4Update;
  f.alpha = alpha;
  f.timeCoeff = timeCoeff;
  f.funcCoeff = funcCoeff;
  Kokkos::parallel_reduce("NEMO::ExplicitRKIteration", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain), f, red);
  Kokkos::fence();
  accOnDevice = false;
  hostAccZero = false;
  MarkDeviceWritten(G_LINSYS);
  MarkDeviceWritten(G_SOLUTION);
  updatePerf.explicitUpd += timer.seconds();
  updatePerf.nExplicit++;
}

void CNEMOKokkosData::AdaptCFL(const CFLAdaptParams& p, double& cflMin, double& cflMax, double& cflSum) {
  auto& P = stagePerf[ST_CFL];
  Kokkos::Timer timer;
  EnsureDevice(G_LOCALCFL);
  EnsureDevice(G_UNDERRELAX);
  if (st.LocalCFL.size() == 0) SU2_MPI::Error("CNEMOKokkosData: CFL adaptation without device LocalCFL.", CURRENT_FUNCTION);
  AdaptCFLFunctor f;
  f.cfl = st.LocalCFL;
  f.ur = st.UnderRelax;
  f.p = p;
  CFLStats v;
  Kokkos::parallel_reduce("NEMO::AdaptCFLNumber", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain), f, v);
  Kokkos::fence();
  if (p.stats) {
    cflMin = v.mn;
    cflMax = v.mx;
    cflSum = v.sum;
  }
  MarkDeviceWritten(G_LOCALCFL);
  P.kernel += timer.seconds();
  P.calls++;
}

void CNEMOKokkosData::DualTimeResidual(bool secondOrder, bool impl, double timeStep, bool uploadInputs) {
  auto& P = stagePerf[ST_DUALTIME];
  Kokkos::Timer timer;
  if (!accOnDevice) SU2_MPI::Error("CNEMOKokkosData: dual-time stage without a device accumulator.", CURRENT_FUNCTION);
  if (ResidentActive()) {
    EnsureDevice(G_SOLUTION);
    EnsureDevice(G_TIMEN);
  } else {
    EnsureInputsOnDevice(uploadInputs, true, false);
    /*--- Host-authoritative: Solution_time_n(1) change once per time step on the host, uploaded every call. ---*/
    UploadGroup(G_TIMEN);
  }
  if (st.SolTimeN.size() == 0) SU2_MPI::Error("CNEMOKokkosData: dual time without Solution_time_n.", CURRENT_FUNCTION);
  Kokkos::fence();
  P.h2d += timer.seconds();
  timer.reset();
  const auto U = st.Solution, Un = st.SolTimeN, Unm1 = st.SolTimeN1, res = st.Res;
  const auto vol = geo.volume;
  const auto dia = geo.dia_ptr;
  const auto jac = st.JacVal;
  const int nv = nVar;
  const double TimeStep = timeStep;
  if (geo.dynamic) {
    /*--- CFVMFlowSolverBase::SetResidual_DualTime, dynamic grid (GCL). The CPU runs three loops (edge GCL over the
     *    domain points, boundary GCL over every vertex of the non INTERNAL/NEARFIELD/PERIODIC markers incl. halo
     *    vertices, then the remainder over the domain points); each writes only row iPoint, so one point kernel
     *    over all points in that per-point order is the same arithmetic. ---*/
    const auto ptr = geo.point_ptr, idx = geo.point_idx;
    const auto pedge = geo.pedge_idx;
    const auto enrm = geo.edge_normal;
    const auto pvtx_ptr = geo.pvtx_ptr, pvtx_idx = geo.pvtx_idx;
    const auto vtx_marker = geo.vtx_marker;
    const auto mflags = geo.marker_flags;
    const auto vnrm = geo.vtx_normal;
    const auto gv = geo.grid_vel;
    const auto vnm1 = geo.volume_nm1;
    const int nd = geo.nDim;
    const unsigned long npd = geo.nPointDomain;
    Kokkos::parallel_for(
        "NEMO::DualTimeResidualGCL", Kokkos::RangePolicy<exec_space>(0, geo.nPoint),
        KOKKOS_LAMBDA(const unsigned long iPoint) {
          if (iPoint < npd) {
            for (auto k = ptr(iPoint); k < ptr(iPoint + 1); ++k) {
              const auto iEdge = static_cast<unsigned long>(pedge(k));
              const double* Normal = &enrm(iEdge, 0);
              const auto jPoint = idx(k);

              /*--- Determine whether to consider the normal outward or inward. ---*/
              const double dir = (iPoint < jPoint) ? 0.5 : -0.5;

              double Residual_GCL = 0.0;
              for (int iDim = 0; iDim < nd; iDim++)
                Residual_GCL += dir * (gv(iPoint, iDim) + gv(jPoint, iDim)) * Normal[iDim];

              for (int iVar = 0; iVar < nv; iVar++) res(iPoint, iVar) += Un(iPoint, iVar) * Residual_GCL;
            }
          }

          /*--- Boundary edges (no GetDomain check on the CPU). ---*/
          for (auto k = pvtx_ptr(iPoint); k < pvtx_ptr(iPoint + 1); ++k) {
            const auto kv = pvtx_idx(k);
            if (!(mflags(vtx_marker(kv)) & CNEMOKokkosData::MARKER_GG_BOUNDARY)) continue;
            const double* Normal = &vnrm(kv, 0);

            double Residual_GCL = 0.0;
            for (int iDim = 0; iDim < nd; iDim++)
              Residual_GCL -= 0.5 * (gv(iPoint, iDim) + gv(iPoint, iDim)) * Normal[iDim];

            for (int iVar = 0; iVar < nv; iVar++) res(iPoint, iVar) += Un(iPoint, iVar) * Residual_GCL;
          }

          if (iPoint >= npd) return;

          /*--- Remainder of the dual time source term (GCL form). ---*/
          const double Volume_nM1 = vnm1(iPoint);
          const double Volume_nP1 = vol(iPoint);
          for (int iVar = 0; iVar < nv; iVar++) {
            if (!secondOrder)
              res(iPoint, iVar) += (U(iPoint, iVar) - Un(iPoint, iVar)) * (Volume_nP1 / TimeStep);
            else
              res(iPoint, iVar) += (U(iPoint, iVar) - Un(iPoint, iVar)) * (3.0 * Volume_nP1 / (2.0 * TimeStep)) +
                                   (Unm1(iPoint, iVar) - Un(iPoint, iVar)) * (Volume_nM1 / (2.0 * TimeStep));
          }
          if (impl) {
            const double val = !secondOrder ? Volume_nP1 / TimeStep : (Volume_nP1 * 3.0) / (2.0 * TimeStep);
            const unsigned long base = dia(iPoint) * nv * nv;
            for (int iVar = 0; iVar < nv; iVar++) jac(base + iVar * (nv + 1)) += val;
          }
        });
    Kokkos::fence();
    P.kernel += timer.seconds();
    P.calls++;
    return;
  }
  /*--- CFVMFlowSolverBase::SetResidual_DualTime, static mesh: per point independent. ---*/
  Kokkos::parallel_for(
      "NEMO::DualTimeResidual", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long iPoint) {
        const double Volume_nP1 = vol(iPoint);
        for (int iVar = 0; iVar < nv; iVar++) {
          if (!secondOrder)
            res(iPoint, iVar) += (U(iPoint, iVar) - Un(iPoint, iVar)) * Volume_nP1 / TimeStep;
          else
            res(iPoint, iVar) += (3.0 * U(iPoint, iVar) - 4.0 * Un(iPoint, iVar) + 1.0 * Unm1(iPoint, iVar)) *
                                 Volume_nP1 / (2.0 * TimeStep);
        }
        if (impl) {
          const double val = !secondOrder ? Volume_nP1 / TimeStep : (Volume_nP1 * 3.0) / (2.0 * TimeStep);
          const unsigned long base = dia(iPoint) * nv * nv;
          for (int iVar = 0; iVar < nv; iVar++) jac(base + iVar * (nv + 1)) += val;
        }
      });
  Kokkos::fence();
  P.kernel += timer.seconds();
  P.calls++;
}

void CNEMOKokkosData::DualTimeShift(double cfl) {
  auto& P = stagePerf[ST_DUALTIME];
  Kokkos::Timer timer;
  EnsureDevice(G_SOLUTION);
  EnsureDevice(G_TIMEN);
  if (st.SolTimeN.size() == 0) SU2_MPI::Error("CNEMOKokkosData: dual time without Solution_time_n.", CURRENT_FUNCTION);
  if (st.LocalCFL.size() == 0) st.LocalCFL = View1D<double>("NEMO::LocalCFL", nPoint);
  /*--- CIntegration::SetDualTime_Solver: Set_Solution_time_n1, Set_Solution_time_n, UnderRelaxation, LocalCFL on all
   *    points (copies / constants: bitwise). ---*/
  Kokkos::deep_copy(st.SolTimeN1, st.SolTimeN);
  Kokkos::deep_copy(st.SolTimeN, st.Solution);
  Kokkos::deep_copy(st.UnderRelax, 1.0);
  Kokkos::deep_copy(st.LocalCFL, cfl);
  Kokkos::fence();
  MarkDeviceWritten(G_TIMEN);
  MarkDeviceWritten(G_UNDERRELAX);
  MarkDeviceWritten(G_LOCALCFL);
  P.kernel += timer.seconds();
}

double CNEMOKokkosData::DualTimeMinDt(double num, double init) {
  EnsureDevice(G_LOCALCFL);
  const auto c = st.LocalCFL;
  double m = init;
  Kokkos::parallel_reduce(
      "NEMO::DualTimeMinDt", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long i, double& v) {
        const double x = num / c(i);
        v = (x < v) ? x : v; /*--- min(v, x) ---*/
      },
      Kokkos::Min<double>(m));
  /*--- Kokkos::Min starts from +max: fold in the CPU initial value (min is order independent). ---*/
  return (m < init) ? m : init;
}

void CNEMOKokkosData::DualTimeExplicitDt(double dtMax) {
  EnsureDevice(G_TIMESTEP);
  const auto d = st.DeltaTime;
  Kokkos::parallel_for(
      "NEMO::DualTimeExplicitDt", Kokkos::RangePolicy<exec_space>(0, geo.nPointDomain),
      KOKKOS_LAMBDA(const unsigned long i) {
        d(i) = (d(i) < dtMax) ? d(i) : dtMax; /*--- min(dtMax, Delta_Time) ---*/
      });
  Kokkos::fence();
  MarkDeviceWritten(G_TIMESTEP);
}

double CNEMOKokkosData::DeviceGamma(unsigned long iPoint) {
  double g = 0.0;
  Kokkos::deep_copy(Kokkos::View<double, HostSpace, Unmanaged>(&g), Kokkos::subview(st.Gamma, iPoint));
  syncPerf[G_PRIMITIVE].bytes_d2h += sizeof(double);
  return g;
}

void CNEMOKokkosData::ReplayGasStateLastPoint(bool full) {
  if (nPoint == 0) return;
  auto& P = syncPerf[G_PRIMITIVE];
  const auto i = nPoint - 1;
  std::vector<double> V(nPrimVar), fdp(2);
  unsigned char status = 0;
  Kokkos::deep_copy(Kokkos::View<double*, HostSpace, Unmanaged>(V.data(), nPrimVar),
                    Kokkos::subview(st.Primitive, i, Kokkos::ALL()));
  Kokkos::deep_copy(Kokkos::View<double*, HostSpace, Unmanaged>(fdp.data(), 2), Kokkos::subview(st.FirstDP, i, Kokkos::ALL()));
  Kokkos::deep_copy(Kokkos::View<unsigned char, HostSpace, Unmanaged>(&status), Kokkos::subview(st.Status, i));
  P.bytes_d2h += double(nPrimVar + 2) * sizeof(double) + 1;
  su2kokkos::nemo::ReplayHostGasState(*gasModel, nDim, V.data(), status, fdp.data(), full);
}

#endif  // HAVE_KOKKOS
