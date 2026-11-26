/*!
 * \file CNEMOEulerSolver.cpp
 * \brief Headers of the CNEMOEulerSolver class
 * \author S. R. Copeland, F. Palacios, W. Maier, C. Garbacz, J. Needels
 * \version 8.5.0 "Harrier"
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2026, SU2 Contributors (cf. AUTHORS.md)
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * SU2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with SU2. If not, see <http://www.gnu.org/licenses/>.
 */

#include "../../include/solvers/CNEMOEulerSolver.hpp"
#include "../../include/gradients/correctGradientsSymmetry.hpp"
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include "../../include/variables/CNEMONSVariable.hpp"
#include "../../../Common/include/toolboxes/geometry_toolbox.hpp"
#include "../../../Common/include/toolboxes/printing_toolbox.hpp"
#include "../../include/fluid/CMutationTCLib.hpp"
#include "../../include/fluid/CSU2TCLib.hpp"
#ifdef HAVE_KOKKOS
#include "../../include/solvers/kokkos/CNEMOKokkosData.hpp"
#endif

CNEMOEulerSolver::CNEMOEulerSolver(CGeometry *geometry, CConfig *config,
                           unsigned short iMesh, const bool navier_stokes) :
  CFVMFlowSolverBase<CNEMOEulerVariable, ENUM_REGIME::COMPRESSIBLE>(*geometry, *config) {
  SU2_ZONE_SCOPED

  /*--- Based on the navier_stokes boolean, determine if this constructor is
   *    being called by itself, or by its derived class CNEMONSSolver. ---*/
  string description;
  if (navier_stokes) {
    description = "Navier-Stokes";
  }
  else {
    description = "Euler";
  }

  const auto nZone = geometry->GetnZone();
  const bool restart = (config->GetRestart() || config->GetRestart_Flow());
  const auto direct_diff = config->GetDirectDiff();
  const bool dual_time = (config->GetTime_Marching() == TIME_MARCHING::DT_STEPPING_1ST) ||
                         (config->GetTime_Marching() == TIME_MARCHING::DT_STEPPING_2ND);
  const bool time_stepping = (config->GetTime_Marching() == TIME_MARCHING::TIME_STEPPING);
  const bool adjoint = config->GetContinuous_Adjoint() || config->GetDiscrete_Adjoint();

  int Unst_RestartIter = 0;

  /*--- A grid is defined as dynamic if there's rigid grid movement or grid deformation AND the problem is time domain ---*/
  dynamic_grid = config->GetDynamic_Grid();

  /*--- Store the multigrid level. ---*/
  MGLevel = iMesh;

  /*--- Check for a restart file to evaluate if there is a change in the AoA
  before non-dimensionalizing ---*/
  if (restart && (iMesh == MESH_0) && nZone <=1 ) {

    /*--- Modify file name for a dual-time unsteady restart ---*/
    if (dual_time) {
      if (adjoint) Unst_RestartIter = SU2_TYPE::Int(config->GetUnst_AdjointIter())-1;
      else if (config->GetTime_Marching() == TIME_MARCHING::DT_STEPPING_1ST)
        Unst_RestartIter = SU2_TYPE::Int(config->GetRestart_Iter())-1;
      else Unst_RestartIter = SU2_TYPE::Int(config->GetRestart_Iter())-2;
    }

    /*--- Modify file name for a time stepping unsteady restart ---*/
    if (time_stepping) {
      if (adjoint) Unst_RestartIter = SU2_TYPE::Int(config->GetUnst_AdjointIter())-1;
      else Unst_RestartIter = SU2_TYPE::Int(config->GetRestart_Iter())-1;
    }

    string filename_ = "flow";
    filename_ = config->GetFilename(filename_, ".meta", Unst_RestartIter);

    /*--- Read and store the restart metadata ---*/
    Read_SU2_Restart_Metadata(geometry, config, false, filename_);

  }

  /*--- Define geometric constants in the solver structure ---*/
  nSpecies     = config->GetnSpecies();
  nMarker      = config->GetnMarker_All();
  nDim         = geometry->GetnDim();
  nPoint       = geometry->GetnPoint();
  nPointDomain = geometry->GetnPointDomain();

  /*--- Set size of the conserved and primitive vectors ---*/
  //     U: [rho1, ..., rhoNs, rhou, rhov, rhow, rhoe, rhoeve]^T
  //     V: [rho1, ..., rhoNs, T, Tve, u, v, w, P, rho, h, a, rhoCvtr, rhoCvve]^T
  // GradV: [rho1, ..., rhoNs, T, Tve, u, v, w, P, rho, h, a, rhoCvtr, rhoCvve]^T
  // Viscous: append [mu, mu_t]^T
  nVar         = nSpecies + nDim + 2;
  if (navier_stokes) { nPrimVar   = nSpecies + nDim + 10; }
  else {               nPrimVar   = nSpecies +nDim +8;    }
  nPrimVarGrad = nSpecies + nDim + 8;

  /*--- Initialize nVarGrad for deallocation ---*/
  nVarGrad     = nPrimVarGrad;

  /*--- Store the number of vertices on each marker for deallocation ---*/
  nVertex.resize(nMarker);
  for (auto iMarker = 0ul; iMarker < nMarker; iMarker++)
    nVertex[iMarker] = geometry->nVertex[iMarker];

  MassFrac_Inf = config->GetGas_Composition();

  /*--- Perform the non-dimensionalization for the flow equations using the
    specified reference values. ---*/
  SetNondimensionalization(config, iMesh);

  /// TODO: This type of variables will be replaced.

  AllocateTerribleLegacyTemporaryVariables();

  /*--- Allocate base class members. ---*/

  Allocate(*config);

  /*--- Allocate Jacobians for implicit time-stepping ---*/
  if (config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT) {

    /*--- Jacobians and vector  structures for implicit computations ---*/
    if (rank == MASTER_NODE) cout << "Initialize Jacobian structure (" << description << "). MG level: " << iMesh <<"." << endl;
    Jacobian.Initialize(nPoint, nPointDomain, nVar, nVar, true, geometry, config);
  }
  else {
    if (rank == MASTER_NODE)  cout<< "Explicit Scheme. No Jacobian structure (" << description << "). MG level: " << iMesh <<"."<<endl;
  }

  /*--- Read farfield conditions from the config file ---*/
  Mach_Inf            = config->GetMach();
  Density_Inf         = config->GetDensity_FreeStreamND();
  Pressure_Inf        = config->GetPressure_FreeStreamND();
  Velocity_Inf        = config->GetVelocity_FreeStreamND();
  Temperature_Inf     = config->GetTemperature_FreeStreamND();
  Temperature_ve_Inf  = config->GetTemperature_ve_FreeStreamND();

  /*--- Initialize the secondary values for direct derivative approximations ---*/
  switch(direct_diff) {
  case NO_DERIVATIVE:
    /*--- Default ---*/
    break;
  case D_DENSITY:
    SU2_TYPE::SetDerivative(Density_Inf, 1.0);
    break;
  case D_PRESSURE:
    SU2_TYPE::SetDerivative(Pressure_Inf, 1.0);
    break;
  case D_TEMPERATURE:
    SU2_TYPE::SetDerivative(Temperature_Inf, 1.0);
    break;
  case D_MACH: case D_AOA:
  case D_SIDESLIP: case D_REYNOLDS:
  case D_TURB2LAM: case D_DESIGN:
    /*--- Already done in postprocessing of config ---*/
    break;
  default:
    break;
  }

  SetReferenceValues(*config);

  /*--- Vectorize free stream Mach number based on AoA & AoS ---*/
  su2double *Mvec_Inf;
  Mvec_Inf = new su2double[nDim];
  const su2double Alpha = config->GetAoA()*PI_NUMBER/180.0;
  const su2double Beta = config->GetAoS()*PI_NUMBER/180.0;
  if (nDim == 2) {
    Mvec_Inf[0] = cos(Alpha)*Mach_Inf;
    Mvec_Inf[1] = sin(Alpha)*Mach_Inf;
  }
  if (nDim == 3) {
    Mvec_Inf[0] = cos(Alpha)*cos(Beta)*Mach_Inf;
    Mvec_Inf[1] = sin(Beta)*Mach_Inf;
    Mvec_Inf[2] = sin(Alpha)*cos(Beta)*Mach_Inf;
  }

  /*--- Initialize the solution to the far-field state everywhere. ---*/

  if (navier_stokes) {
    nodes      = new CNEMONSVariable    (Pressure_Inf, MassFrac_Inf, Mvec_Inf,
                                         Temperature_Inf, Temperature_ve_Inf,
                                         nPoint, nDim, nVar, nPrimVar, nPrimVarGrad,
                                         config, FluidModel);
    node_infty = new CNEMONSVariable    (Pressure_Inf, MassFrac_Inf, Mvec_Inf,
                                        Temperature_Inf, Temperature_ve_Inf,
                                        1, nDim, nVar, nPrimVar, nPrimVarGrad,
                                        config, FluidModel);
  } else {
    nodes      = new CNEMOEulerVariable(Pressure_Inf, MassFrac_Inf, Mvec_Inf,
                                        Temperature_Inf, Temperature_ve_Inf,
                                        nPoint, nDim, nVar, nPrimVar, nPrimVarGrad,
                                        config, FluidModel);
    node_infty = new CNEMOEulerVariable(Pressure_Inf, MassFrac_Inf, Mvec_Inf,
                                        Temperature_Inf, Temperature_ve_Inf,
                                        1, nDim, nVar, nPrimVar, nPrimVarGrad,
                                        config, FluidModel);
  }
  SetBaseClassPointerToNodes();

  node_infty->SetPrimVar(0, FluidModel);

  /*--- Initial comms. ---*/

  CommunicateInitialState(geometry, config);

  /*--- Add the solver name. ---*/
  SolverName = "NEMO.FLOW";

  /*--- Finally, check that the static arrays will be large enough (keep this
   *    check at the bottom to make sure we consider the "final" values). ---*/
  if((nDim > MAXNDIM) || (nPrimVar > MAXNVAR))
    SU2_MPI::Error("Oops! The CNEMOEulerSolver static array sizes are not large enough.",CURRENT_FUNCTION);

#ifdef HAVE_KOKKOS
  /*--- Device mirror and device Cons2Prim (Steps 1-2 of NEMO_KOKKOS_PORT_PLAN.md). ---*/
  if (config->GetKokkos()) {
    auto* su2tc = dynamic_cast<CSU2TCLib*>(FluidModel);
    if (su2tc != nullptr) {
      kokkosData = std::make_unique<CNEMOKokkosData>(geometry, config, nodes, su2tc, iMesh);
      kokkosConfig = config;
      /*--- Batch A: does a Gamma / Sensor setter precede each supersonic marker in the BC order (every rank)? ---*/
      {
        const auto nM = config->GetnMarker_All();
        std::vector<int> ok(nM, 1);
        bool gammaSet = false, sensorSet = false;
        for (unsigned short iMarker = 0; iMarker < nM; iMarker++) {
          const auto kb = config->GetMarker_All_KindBC(iMarker);
          bool hasDomain = false;
          for (auto iVertex = 0ul; iVertex < geometry->nVertex[iMarker]; iVertex++)
            if (geometry->nodes->GetDomain(geometry->vertex[iMarker][iVertex]->GetNode())) hasDomain = true;
          if (kb == SUPERSONIC_INLET && hasDomain) ok[iMarker] = gammaSet && sensorSet;
          if (kb == SUPERSONIC_OUTLET && hasDomain) ok[iMarker] = sensorSet;
          if (!hasDomain) continue;
          if (kb == FAR_FIELD || kb == OUTLET_FLOW) gammaSet = sensorSet = true;
          if (kb == SUPERSONIC_OUTLET) gammaSet = true;
        }
        /*--- Reduce over the config-file markers, matched by tag: nMarker_All also counts this rank's
         *    SEND_RECEIVE markers, which differ between ranks (a reduction of length nMarker_All mismatches,
         *    MPI_ERR_TRUNCATE). SEND_RECEIVE markers are not physical BCs, so they are always "ok". ---*/
        const auto nCfg = config->GetnMarker_CfgFile();
        std::vector<int> okCfg(nCfg, 1), okCfgAll(nCfg, 1);
        std::vector<int> cfgIndex(nM, -1);
        for (unsigned short iMarker = 0; iMarker < nM; iMarker++) {
          if (config->GetMarker_All_KindBC(iMarker) == SEND_RECEIVE) continue;
          const auto iCfg = config->GetMarker_CfgFile_TagBound(config->GetMarker_All_TagBound(iMarker));
          cfgIndex[iMarker] = iCfg;
          okCfg[iCfg] = std::min(okCfg[iCfg], ok[iMarker]);
        }
        if (nCfg > 0) SU2_MPI::Allreduce(okCfg.data(), okCfgAll.data(), nCfg, MPI_INT, MPI_MIN, SU2_MPI::GetComm());
        bndStaleOK.assign(nM, 1);
        for (unsigned short iMarker = 0; iMarker < nM; iMarker++)
          if (cfgIndex[iMarker] >= 0) bndStaleOK[iMarker] = static_cast<char>(okCfgAll[cfgIndex[iMarker]]);
      }
      deferResidualSync = Device_DeferralUnsupported(config).empty();
      deviceJacobianAllocated = (config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT);
      /*--- Step 6: device-resident state when every stage of the iteration is on the device. ---*/
      /*--- 4.6-MG2: coarse MG levels device-resident too (Device_MGUnsupported), transfers on the device. ---*/
      std::string whyNotResident;
      if (iMesh == MESH_0) {
        whyNotResident = Device_ResidentUnsupported(config);
      } else {
        whyNotResident = Device_MGUnsupported(config);
        if (whyNotResident.empty() && (iMesh > s_mgLevelDevice.size() || !s_mgLevelDevice[iMesh - 1]))
          whyNotResident = "finer level not on the device MG path";
        if (whyNotResident.empty()) whyNotResident = Device_ResidentUnsupported(config);
      }
      deviceResident = whyNotResident.empty();
      mgDevice = deviceResident && config->GetnMGLevels() > 0 && (iMesh > MESH_0 || Device_MGUnsupported(config).empty());
      if (config->GetnMGLevels() > 0) {
        if (s_mgLevelDevice.size() <= iMesh) s_mgLevelDevice.resize(iMesh + 1, 0);
        s_mgLevelDevice[iMesh] = mgDevice ? 1 : 0;
      }
      if (rank == MASTER_NODE && iMesh > MESH_0)
        cout << "[KOKKOS_NEMO] MG level " << iMesh << " device-resident state: "
             << (deviceResident ? std::string("ON (device MG transfers)") : "OFF (" + whyNotResident + ")") << endl;
      if (rank == MASTER_NODE && iMesh == MESH_0) PrintDeviceStages(config);
      if (deviceResident) {
        const bool impl = (config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT);
        kokkosData->EnableResident(LinSysRes.GetDevicePointer(), LinSysSol.GetDevicePointer(),
                                   impl ? Jacobian.GetKokkosDeviceValues() : nullptr, LinSysRes.GetBlock(0),
                                   LinSysSol.GetBlock(0));
        /*--- The linear solver reads / writes the device copies directly (no HtoD of b, x, matrix; no DtoH of x). ---*/
        LinSysRes.SetKokkosDeviceAuthoritative(true);
        LinSysSol.SetKokkosDeviceAuthoritative(true);
        if (impl) Jacobian.SetKokkosValuesOnDevice(true);
        /*--- Batch A: classical RK4 keeps Solution_New on the device (Set_OldSolution also copies it). ---*/
        kokkosData->SetRK4(config->GetKind_TimeIntScheme_Flow() == CLASSICAL_RK4_EXPLICIT);
        /*--- Step 8: Friction_Forces reads the gradient / transport rows of the viscous-wall points. ---*/
        if (config->GetViscous()) {
          std::vector<int> walls;
          for (unsigned short iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++)
            if (config->GetViscous_Wall(iMarker)) walls.push_back(iMarker);
          kokkosData->SetViscousWallMarkers(walls);
        }
      }
      if (rank == MASTER_NODE && iMesh == MESH_0 && config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT &&
          config->GetKind_Linear_Solver_Prec() == LU_SGS)
        cout << "[KOKKOS_NEMO] stage linear preconditioner (LU_SGS apply): " << (deviceResident ? "device" : "host")
             << endl;
      if (rank == MASTER_NODE && iMesh == MESH_0)
        cout << "[KOKKOS_NEMO] device-resident state (Step 6): "
             << (deviceResident ? std::string("ON (U, V, secondaries, gradient, limiter, dt, Non_Physical, residual, "
                                               "Jacobian, LinSysSol stay on the device; host copies on demand)")
                                : "OFF (" + whyNotResident + ")")
             << endl;
    } else if (rank == MASTER_NODE && iMesh == MESH_0) {
      cout << "[KOKKOS_NEMO] gas model is not SU2_NONEQ, device Cons2Prim disabled (host path)." << endl;
    }
  }
#endif

   /*--- Deallocate arrays ---*/
  delete [] Mvec_Inf;

}

CNEMOEulerSolver::~CNEMOEulerSolver() {
  SU2_ZONE_SCOPED

#ifdef HAVE_KOKKOS
  if (kokkosData && rank == MASTER_NODE && Jacobian.GetKokkosDeviceBuilds() > 0) {
    cout << "[KOKKOS_NEMO_PERF] device preconditioner builds="
         << Jacobian.GetKokkosDeviceBuilds() << " (" << Jacobian.GetKokkosBuildSeconds() << " s incl. verification)"
         << endl;
    if (Jacobian.GetKokkosVerifyBuilds() > 0)
      cout << "[KOKKOS_NEMO_VERIFY] preconditioner device build vs host build: compares="
           << Jacobian.GetKokkosVerifyBuilds() << " non-bitwise=" << Jacobian.GetKokkosVerifyNonBitwise() << endl;
  }
  if (kokkosData && rank == MASTER_NODE && mgHostTransfers > 0)
    cout << "[KOKKOS_NEMO_PERF] MG fine-level host transfers=" << mgHostTransfers << endl;
  if (kokkosData && rank == MASTER_NODE && kokkosData->mgDeviceOps > 0)
    cout << "[KOKKOS_NEMO_PERF] MG device transfers=" << kokkosData->mgDeviceOps << " (level " << MGLevel << ")" << endl;
  if (kokkosData && rank == MASTER_NODE && Jacobian.GetKokkosLU_SGSApplies() > 0) {
    cout << "[KOKKOS_NEMO_PERF] device LU_SGS applies=" << Jacobian.GetKokkosLU_SGSApplies() << " ("
         << Jacobian.GetKokkosLU_SGSSeconds() << " s incl. verification)" << endl;
    if (Jacobian.GetKokkosLU_SGSVerify() > 0)
      cout << "[KOKKOS_NEMO_VERIFY] LU_SGS device apply vs host apply: compares=" << Jacobian.GetKokkosLU_SGSVerify()
           << " non-bitwise=" << Jacobian.GetKokkosLU_SGSVerifyNonBitwise() << endl;
  }
  kokkosData.reset(); /*--- Prints [KOKKOS_NEMO_PERF]; must go before the gas model it references. ---*/
#endif

  delete node_infty;
  delete FluidModel;

}

void CNEMOEulerSolver::CommonPreprocessing(CGeometry *geometry, CSolver **solver_container, CConfig *config, unsigned short iMesh,
                                           unsigned short iRKStep, unsigned short RunTime_EqSystem, bool Output) {
  SU2_ZONE_SCOPED

  const bool implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);
  const bool center = (config->GetKind_ConvNumScheme_Flow() == SPACE_CENTERED);
  const bool center_jst = (config->GetKind_Centered_Flow() == CENTERED::JST) && (iMesh == MESH_0);
  const bool center_jst_ke = (config->GetKind_Centered_Flow() == CENTERED::JST_KE) && (iMesh == MESH_0);

  /*--- Set the primitive variables ---*/
  ompMasterAssignBarrier(ErrorCounter,0);
  SU2_OMP_ATOMIC
  ErrorCounter += SetPrimitive_Variables(solver_container, config, Output);

  BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS
  { /*--- Ops that are not OpenMP parallel go in this block. ---*/

    if ((iMesh == MESH_0) && (config->GetComm_Level() == COMM_FULL)) {
      unsigned long tmp = ErrorCounter;
      SU2_MPI::Allreduce(&tmp, &ErrorCounter, 1, MPI_UNSIGNED_LONG, MPI_SUM, SU2_MPI::GetComm());
      config->SetNonphysical_Points(ErrorCounter);

      if ((rank == MASTER_NODE) && (ErrorCounter != 0))
        cout << "Warning. The initial solution contains "<< ErrorCounter << " points that are not physical." << endl;
    }
  }
  END_SU2_OMP_SAFE_GLOBAL_ACCESS

  /*--- Artificial dissipation ---*/

  if (center && !Output) {
    SetMax_Eigenvalue(geometry, config);
    if (center_jst) SetUndivided_Laplacian(geometry, config);
    if (center_jst || center_jst_ke) SetCentered_Dissipation_Sensor(geometry, config);
  }

  /*--- Set Pressure diffusion sensor ---*/
  if (config->GetKind_Upwind_Flow() == UPWIND::AUSMPLUSM) {
    if (!Device_PressureSensor(geometry, config)) SetPressureDiffusionSensor(geometry, config);
  }

  /*--- Batch A: replay of the boundary numerics' Gamma / Sensor restarts with this space integration. ---*/
  if (!Output) bndNumerics = BoundaryNumericsState();

  /*--- Initialize the Jacobian matrix and residual, not needed for the reducer strategy
   *    as we set blocks (including diagonal ones) and completely overwrite. ---*/

  if(!ReducerStrategy && !Output) {
#ifdef HAVE_KOKKOS
    /*--- Device-resident: the device accumulator is zero-filled by the first device stage; the stale host copies
     *    are zeroed only if a host stage ever needs them (Device_BeginHostStage). ---*/
    if (Device_ResidentActive()) {
      hostAccNeedsZero = true;
    } else {
      LinSysRes.SetValZero();
      if (implicit) Jacobian.SetValZero();
    }
    if (kokkosData) kokkosData->MarkHostAccumulatorZero();
#else
    LinSysRes.SetValZero();
    if (implicit) Jacobian.SetValZero();
#endif
  }
}

void CNEMOEulerSolver::Preprocessing(CGeometry *geometry, CSolver **solver_container,
                                     CConfig *config, unsigned short iMesh,
                                     unsigned short iRKStep,
                                     unsigned short RunTime_EqSystem, bool Output) {
  SU2_ZONE_SCOPED
  const unsigned long InnerIter = config->GetInnerIter();
  const bool muscl       = config->GetMUSCL_Flow() && (iMesh == MESH_0);
  const bool limiter     = (config->GetKind_SlopeLimit_Flow() != LIMITER::NONE) && (InnerIter <= config->GetLimiterIter());
  const bool center      = config->GetKind_ConvNumScheme_Flow() == SPACE_CENTERED;
  const bool van_albada  = config->GetKind_SlopeLimit_Flow() == LIMITER::VAN_ALBADA_EDGE;

#ifdef HAVE_KOKKOS
  /*--- 6b.a: the host grid movement / deformation ran before this call: refresh the device metric arrays. ---*/
  if (dynamic_grid && iMesh == MESH_0) Device_RefreshDynamicGeometry(geometry, config);
#endif

  /*--- Common preprocessing steps ---*/
  CommonPreprocessing(geometry, solver_container, config, iMesh, iRKStep, RunTime_EqSystem, Output);

  /*--- Upwind second order reconstruction ---*/
  if (muscl && !center && !Output) {

    /*--- Calculate the gradients ---*/
    if (config->GetKind_Gradient_Method() == GREEN_GAUSS) {
      if (!Device_PrimitiveGradient(geometry, config, true, true, false))
        SetPrimitive_Gradient_GG(geometry, config, true);
    }
    if (config->GetKind_Gradient_Method() == WEIGHTED_LEAST_SQUARES) {
      if (!Device_PrimitiveGradient(geometry, config, true, false, false))
        SetPrimitive_Gradient_LS(geometry, config, true);
    }

    /*--- Limiter computation ---*/
    if (limiter && !van_albada) {
      if (!Device_PrimitiveLimiter(geometry, config))
        SetPrimitive_Limiter(geometry, config);
    }
  }
}

unsigned long CNEMOEulerSolver::SetPrimitive_Variables(CSolver **solver_container, CConfig *config, bool Output) {
  SU2_ZONE_SCOPED

  unsigned long nonPhysicalPoints = 0;
  bool nonphysical = true;

#ifdef HAVE_KOKKOS
  /*--- SU2_KOKKOS_NEMO_HOST=c2p: host Cons2Prim, the device stages then re-upload what they read. ---*/
  const bool deviceC2P = kokkosData && !CNEMOKokkosData::HostForced("c2p");
  if (kokkosData && !deviceC2P) kokkosData->InvalidatePrimitive();
  const bool verify = deviceC2P && CNEMOKokkosData::VerifyEnabled();
  unsigned long nonPhysDevice = 0;
  if (deviceC2P) {
    if (verify) kokkosData->Snapshot();

    /*--- Step 6: the first call activates the device-resident mode (the host state is current now, after restart /
     *    initialisation); every later non-output call closes one iteration of the transfer ledger. ---*/
    if (deviceResident) kokkosData->ActivateResident();
    if (!Output) kokkosData->LedgerIterationBoundary(false);

    if (Device_ResidentActive()) {
      /*--- Device Cons2Prim on the resident state, nothing downloaded; Velocity2 is recomputed with the host
       *    Primitive when it is downloaded; LinSysRes is the device accumulator (zero-filled by the first stage). ---*/
      nonPhysicalPoints = kokkosData->SetPrimVar();
      Kokkos::Timer timer;
      kokkosData->ReplayGasStateLastPoint(true);
      kokkosData->perf.host_post += timer.seconds();
      return nonPhysicalPoints;
    }

    /*--- Device Cons2Prim + rollback + Gamma (uploads U/V, downloads V, secondaries, U). ---*/
    nonPhysicalPoints = kokkosData->SetPrimVar();

    /*--- Host remainder of CNEMOEulerVariable::SetPrimVar, same per-point order. ---*/
    Kokkos::Timer timer;
    for (auto iPoint = 0ul; iPoint < nPoint; iPoint ++) {
      nodes->SetVelocity2(iPoint);
      if (!Output) LinSysRes.SetBlock_Zero(iPoint);
    }
    /*--- Leave the host gas model in the state the CPU loop leaves it (last point). ---*/
    if (nPoint > 0) kokkosData->ReplayGasState(nPoint-1, true);
    kokkosData->perf.host_post += timer.seconds();

    if (!verify) return nonPhysicalPoints;

    /*--- Debug A/B: restore the pre-call host state and fall through to the CPU loop. ---*/
    nonPhysDevice = nonPhysicalPoints;
    nonPhysicalPoints = 0;
    kokkosData->SwapSnapshot();
    kokkosData->InvalidatePrimitive(); /*--- The CPU loop below overwrites the host Primitive. ---*/
  }
#endif

  for (auto iPoint = 0ul; iPoint < nPoint; iPoint ++) {

    /*--- Incompressible flow, primitive variables ---*/

    nonphysical = nodes->SetPrimVar(iPoint,FluidModel);

    /* Check for non-realizable states for reporting. */

    if (nonphysical) nonPhysicalPoints++;

    /*--- Initialize the convective, source and viscous residual vector ---*/

    if (!Output) LinSysRes.SetBlock_Zero(iPoint);

  }

#ifdef HAVE_KOKKOS
  if (verify) kokkosData->CompareSnapshot(nonPhysDevice, nonPhysicalPoints);
#endif

  return nonPhysicalPoints;
}

void CNEMOEulerSolver::SetTime_Step(CGeometry *geometry, CSolver **solver_container, CConfig *config,
                                    unsigned short iMesh, unsigned long Iteration) {
  SU2_ZONE_SCOPED

#ifdef HAVE_KOKKOS
  if (kokkosData && Device_TimeStepUnsupported(config).empty()) {
    /*--- Device part of SetTime_Step_impl (edge + vertex spectral radii, local dt, rank-local min/max). ---*/
    const bool viscous = config->GetViscous();
    const bool verify = CNEMOKokkosData::StageVerifyEnabled();
    if (verify) kokkosData->UploadPrimitiveOnly(); /*--- Isolate this stage from the Cons2Prim A/B. ---*/
    /*--- A/B: the time-stepping branch below overwrites LocalCFL, an input of the host rerun. ---*/
    std::vector<su2double> cflIn;
    if (verify) {
      cflIn.resize(nPointDomain);
      for (auto i = 0ul; i < nPointDomain; ++i) cflIn[i] = nodes->GetLocalCFL(i);
    }

    su2double minDt = 1e30, maxDt = 0.0;
    kokkosData->ComputeTimeStep(viscous, config->GetMax_DeltaTime(), minDt, maxDt);

    /*--- Host remainder of SetTime_Step_impl, same statements (no OpenMP in Kokkos builds). ---*/
    Kokkos::Timer timer;
    const bool implicit      = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);
    const bool time_stepping = (config->GetTime_Marching() == TIME_MARCHING::TIME_STEPPING);
    const bool dual_time     = (config->GetTime_Marching() == TIME_MARCHING::DT_STEPPING_1ST) ||
                               (config->GetTime_Marching() == TIME_MARCHING::DT_STEPPING_2ND);

    Min_Delta_Time = 1e30; Max_Delta_Time = 0.0; Global_Delta_UnstTimeND = 1e30;
    Min_Delta_Time = min(Min_Delta_Time, minDt);
    Max_Delta_Time = max(Max_Delta_Time, maxDt);

    if (config->GetComm_Level() == COMM_FULL) {
      su2double rbuf_time;
      SU2_MPI::Allreduce(&Min_Delta_Time, &rbuf_time, 1, MPI_DOUBLE, MPI_MIN, SU2_MPI::GetComm());
      Min_Delta_Time = rbuf_time;
      SU2_MPI::Allreduce(&Max_Delta_Time, &rbuf_time, 1, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());
      Max_Delta_Time = rbuf_time;
    }
    Global_Delta_Time = Min_Delta_Time;

    if (time_stepping) {
      if (config->GetUnst_CFL() == 0.0) Global_Delta_Time = config->GetDelta_UnstTime();
      else Global_Delta_Time = Min_Delta_Time;
      Max_Delta_Time = Global_Delta_Time;
      config->SetDelta_UnstTimeND(Global_Delta_Time);
      if (Device_ResidentActive()) {
        kokkosData->TimeSteppingFill(config->GetUnst_CFL(), Global_Delta_Time);
      } else {
        for (auto iPoint = 0ul; iPoint < nPointDomain; iPoint++) {
          nodes->SetLocalCFL(iPoint, config->GetUnst_CFL());
          nodes->SetDelta_Time(iPoint, Global_Delta_Time);
        }
      }
    }

    if (dual_time && (Iteration == config->GetRestart_Iter()) && (config->GetUnst_CFL() != 0.0) && (iMesh == MESH_0)) {
      su2double glbDtND = 1e30;
      if (Device_ResidentActive()) {
        /*--- Batch A: same min over the device LocalCFL (order independent). ---*/
        glbDtND = kokkosData->DualTimeMinDt(config->GetUnst_CFL()*Global_Delta_Time, glbDtND);
      } else {
        for (auto iPoint = 0ul; iPoint < nPointDomain; iPoint++)
          glbDtND = min(glbDtND, config->GetUnst_CFL()*Global_Delta_Time / nodes->GetLocalCFL(iPoint));
      }
      Global_Delta_UnstTimeND = min(Global_Delta_UnstTimeND, glbDtND);
      SU2_MPI::Allreduce(&Global_Delta_UnstTimeND, &glbDtND, 1, MPI_DOUBLE, MPI_MIN, SU2_MPI::GetComm());
      Global_Delta_UnstTimeND = glbDtND;
      config->SetDelta_UnstTimeND(Global_Delta_UnstTimeND);
    }

    if (dual_time && !implicit) {
      if (Device_ResidentActive()) {
        kokkosData->DualTimeExplicitDt((2.0/3.0)*config->GetDelta_UnstTimeND());
      } else {
        for (auto iPoint = 0ul; iPoint < nPointDomain; iPoint++) {
          su2double dt = min((2.0/3.0)*config->GetDelta_UnstTimeND(), nodes->GetDelta_Time(iPoint));
          nodes->SetDelta_Time(iPoint, dt);
        }
      }
    }
    kokkosData->stagePerf[CNEMOKokkosData::ST_TIMESTEP].host += timer.seconds();

    if (!verify) return;

    /*--- A/B: keep the device results, rerun the unchanged host code, compare, continue with the host results. ---*/
    const auto nd = nPointDomain;
    auto grab = [&](auto getter) {
      std::vector<su2double> v(nd);
      for (auto i = 0ul; i < nd; ++i) v[i] = getter(i);
      return v;
    };
    auto lamInv = [&](unsigned long i) { return nodes->GetMax_Lambda_Inv(i); };
    auto lamVisc = [&](unsigned long i) { return nodes->GetMax_Lambda_Visc(i); };
    auto delta = [&](unsigned long i) { return nodes->GetDelta_Time(i); };
    const auto dLamInv = grab(lamInv), dDt = grab(delta);
    const auto dLamVisc = viscous ? grab(lamVisc) : std::vector<su2double>();
    const su2double dMin = Min_Delta_Time, dMax = Max_Delta_Time, dGlb = Global_Delta_Time;

    for (auto i = 0ul; i < nPointDomain; ++i) nodes->SetLocalCFL(i, cflIn[i]);
    SetTime_Step_Host(geometry, solver_container, config, iMesh, Iteration);

    using K = CNEMOKokkosData;
    kokkosData->StageCompare(K::ST_TIMESTEP, "Max_Lambda_Inv", grab(lamInv).data(), dLamInv.data(), nd);
    if (viscous) kokkosData->StageCompare(K::ST_TIMESTEP, "Max_Lambda_Visc", grab(lamVisc).data(), dLamVisc.data(), nd);
    kokkosData->StageCompare(K::ST_TIMESTEP, "Delta_Time", grab(delta).data(), dDt.data(), nd);
    kokkosData->StageCompareScalar(K::ST_TIMESTEP, "Min_Delta_Time", Min_Delta_Time, dMin);
    kokkosData->StageCompareScalar(K::ST_TIMESTEP, "Max_Delta_Time", Max_Delta_Time, dMax);
    kokkosData->StageCompareScalar(K::ST_TIMESTEP, "Global_Delta_Time", Global_Delta_Time, dGlb);
    return;
  }
#endif

  SetTime_Step_Host(geometry, solver_container, config, iMesh, Iteration);
}

void CNEMOEulerSolver::SetTime_Step_Host(CGeometry *geometry, CSolver **solver_container, CConfig *config,
                                         unsigned short iMesh, unsigned long Iteration) {

  /*--- Define an object to compute the speed of sound. ---*/
  struct SoundSpeed {
    FORCEINLINE su2double operator() (const CNEMOEulerVariable& nodes, unsigned long iPoint, unsigned long jPoint) const {
      return 0.5 * (nodes.GetSoundSpeed(iPoint) + nodes.GetSoundSpeed(jPoint));
    }

    FORCEINLINE su2double operator() (const CNEMOEulerVariable& nodes, unsigned long iPoint) const {
      return nodes.GetSoundSpeed(iPoint);
    }

  } soundSpeed;

  /*--- Define an object to compute the viscous eigenvalue. ---*/
  struct LambdaVisc {
    FORCEINLINE su2double lambda(su2double lamVisc, su2double eddyVisc, su2double rho, su2double k, su2double cv) const {
      /*--- Determine the viscous spectral radius and apply it to the control volume ---*/
      const su2double Lambda_1 = (4.0/3.0)*(lamVisc + eddyVisc);
      return (Lambda_1 + k/cv)/rho;
    }

    FORCEINLINE su2double operator() (const CNEMOEulerVariable& nodes, unsigned long iPoint, unsigned long jPoint) const {
      const su2double lamVisc = 0.5*(nodes.GetLaminarViscosity(iPoint) + nodes.GetLaminarViscosity(jPoint));
      const su2double eddyVisc = 0.5*(nodes.GetEddyViscosity(iPoint) + nodes.GetEddyViscosity(jPoint));
      const su2double thermalCond = 0.5*(nodes.GetThermalConductivity(iPoint) + nodes.GetThermalConductivity(jPoint) +
                                   nodes.GetThermalConductivity_ve(iPoint) + nodes.GetThermalConductivity_ve(jPoint));
      const su2double density = 0.5*(nodes.GetDensity(iPoint) + nodes.GetDensity(jPoint));
      const su2double cv = 0.5*(nodes.GetRhoCv_tr(iPoint) + nodes.GetRhoCv_ve(iPoint) +
                          nodes.GetRhoCv_tr(jPoint) + nodes.GetRhoCv_ve(jPoint))/ density;
      return lambda(lamVisc, eddyVisc, density, thermalCond, cv);
    }

    FORCEINLINE su2double operator() (const CNEMOEulerVariable& nodes, unsigned long iPoint) const {
      const su2double lamVisc = nodes.GetLaminarViscosity(iPoint);
      const su2double eddyVisc = nodes.GetEddyViscosity(iPoint);
      const su2double thermalCond = nodes.GetThermalConductivity(iPoint) + nodes.GetThermalConductivity_ve(iPoint);
      const su2double density = nodes.GetDensity(iPoint);
      const su2double cv = (nodes.GetRhoCv_tr(iPoint) + nodes.GetRhoCv_ve(iPoint))/ density;
      return lambda(lamVisc, eddyVisc, density, thermalCond, cv);
    }

  } lambdaVisc;

  /*--- Now instantiate the generic implementation with the two functors above. ---*/

  SetTime_Step_impl(soundSpeed, lambdaVisc, geometry, solver_container, config, iMesh, Iteration);

}

void CNEMOEulerSolver::SetMax_Eigenvalue(CGeometry *geometry, CConfig *config) {
  SU2_ZONE_SCOPED

#ifdef HAVE_KOKKOS
  if (kokkosData && Device_TimeStepUnsupported(config).empty()) {
    const bool verify = CNEMOKokkosData::StageVerifyEnabled();
    if (verify) kokkosData->UploadPrimitiveOnly();
    kokkosData->ComputeMaxEigenvalue();

    if (!verify) {
      Kokkos::Timer timer;
      /*--- Same comms as SetMax_Eigenvalue_impl. ---*/
      for (unsigned short iPeriodic = 1; iPeriodic <= config->GetnMarker_Periodic()/2; iPeriodic++) {
        InitiatePeriodicComms(geometry, config, iPeriodic, PERIODIC_MAX_EIG);
        CompletePeriodicComms(geometry, config, iPeriodic, PERIODIC_MAX_EIG);
      }
      InitiateComms(geometry, config, MPI_QUANTITIES::MAX_EIGENVALUE);
      CompleteComms(geometry, config, MPI_QUANTITIES::MAX_EIGENVALUE);
      kokkosData->stagePerf[CNEMOKokkosData::ST_MAXEIG].host += timer.seconds();
      return;
    }
    std::vector<su2double> dLam(nPointDomain), hLam(nPointDomain);
    for (auto i = 0ul; i < nPointDomain; ++i) dLam[i] = nodes->GetLambda(i);
    SetMax_Eigenvalue_Host(geometry, config);
    for (auto i = 0ul; i < nPointDomain; ++i) hLam[i] = nodes->GetLambda(i);
    kokkosData->StageCompare(CNEMOKokkosData::ST_MAXEIG, "Lambda", hLam.data(), dLam.data(), nPointDomain);
    return;
  }
#endif

  SetMax_Eigenvalue_Host(geometry, config);
}

void CNEMOEulerSolver::SetMax_Eigenvalue_Host(CGeometry *geometry, CConfig *config) {

  /*--- Define an object to compute the speed of sound. ---*/
  struct SoundSpeed {
    FORCEINLINE su2double operator() (const CNEMOEulerVariable& nodes, unsigned long iPoint, unsigned long jPoint) const {
      return 0.5 * (nodes.GetSoundSpeed(iPoint) + nodes.GetSoundSpeed(jPoint));
    }

    FORCEINLINE su2double operator() (const CNEMOEulerVariable& nodes, unsigned long iPoint) const {
      return nodes.GetSoundSpeed(iPoint);
    }

  } soundSpeed;

  /*--- Instantiate generic implementation. ---*/

  SetMax_Eigenvalue_impl(soundSpeed, geometry, config);

}

void CNEMOEulerSolver::Centered_Residual(CGeometry *geometry, CSolver **solver_container, CNumerics **numerics_container,
                                         CConfig *config, unsigned short iMesh, unsigned short iRKStep) {
  SU2_ZONE_SCOPED

#ifdef HAVE_KOKKOS
  if (!centeredHostOnly &&
      Device_CenteredResidual(geometry, solver_container, numerics_container, config, iMesh, iRKStep)) return;
  Device_BeginHostStage(geometry);
#endif

  CNumerics* numerics = numerics_container[CONV_TERM];

  /*--- Set booleans based on config settings ---*/
  const bool implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);

  for (auto iEdge = 0ul; iEdge < geometry->GetnEdge(); iEdge++) {

    /*--- Points in edge, set normal vectors, and number of neighbors ---*/
    const auto iPoint = geometry->edges->GetNode(iEdge, 0);
    const auto jPoint = geometry->edges->GetNode(iEdge, 1);
    numerics->SetNormal(geometry->edges->GetNormal(iEdge));
    numerics->SetNeighbor(geometry->nodes->GetnNeighbor(iPoint),
                          geometry->nodes->GetnNeighbor(jPoint));

    /*--- Pass conservative & primitive variables w/o reconstruction to CNumerics ---*/
    numerics->SetConservative(nodes->GetSolution(iPoint),  nodes->GetSolution(jPoint));
    numerics->SetPrimitive   (nodes->GetPrimitive(iPoint), nodes->GetPrimitive(jPoint));

    /*--- Pass supplementary information to CNumerics ---*/
    numerics->SetdPdU  ( nodes->GetdPdU(iPoint),   nodes->GetdPdU(jPoint));
    numerics->SetdTdU  ( nodes->GetdTdU(iPoint),   nodes->GetdTdU(jPoint));
    numerics->SetdTvedU( nodes->GetdTvedU(iPoint), nodes->GetdTvedU(jPoint));
    numerics->SetEve   ( nodes->GetEve(iPoint),    nodes->GetEve(jPoint));
    numerics->SetCvve  ( nodes->GetCvve(iPoint),   nodes->GetCvve(jPoint));
    numerics->SetGamma ( nodes->GetGamma(iPoint),  nodes->GetGamma(jPoint));

    /*--- Set the largest convective eigenvalue ---*/
    numerics->SetLambda(nodes->GetLambda(iPoint), nodes->GetLambda(jPoint));

    /*--- Compute residuals, and Jacobians ---*/
    auto residual = numerics->ComputeResidual(config);

    /*--- Check residuals/Jacobians --*/
    bool err = CNumerics::CheckResidualNaNs(implicit, nVar, residual);

    /*--- Update the residual and Jacobian ---*/
    if (!err) {
      LinSysRes.AddBlock(iPoint, residual);
      LinSysRes.SubtractBlock(jPoint, residual);
      if (implicit) {
        Jacobian.UpdateBlocks(iEdge, iPoint, jPoint, residual.jacobian_i, residual.jacobian_j);
      }
    }
  }
}

void CNEMOEulerSolver::Upwind_Residual(CGeometry *geometry, CSolver **solver_container, CNumerics **numerics_container,
                                       CConfig *config, unsigned short iMesh) {
  SU2_ZONE_SCOPED

#ifdef HAVE_KOKKOS
  if (!upwindHostOnly && Device_UpwindResidual(geometry, solver_container, numerics_container, config, iMesh)) return;
  Device_BeginHostStage(geometry);
#endif

  /*--- Set booleans based on config settings ---*/
  const bool implicit         = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);
  const bool muscl            = (config->GetMUSCL_Flow() && (iMesh == MESH_0));
  const bool limiter          = (config->GetKind_SlopeLimit_Flow() != LIMITER::NONE);
  const bool van_albada       = (config->GetKind_SlopeLimit_Flow() == LIMITER::VAN_ALBADA_EDGE);

  const su2double kappa       = config->GetMUSCL_Kappa_Flow();
  const su2double musclRamp   = config->GetMUSCLRampValue() * config->GetNewtonKrylovRelaxation();

  /*--- Non-physical counter. ---*/
  unsigned long counter_local = 0;
  SU2_OMP_MASTER
  ErrorCounter = 0;
  END_SU2_OMP_MASTER

  /*--- Pick one numerics object per thread. ---*/
  CNumerics* numerics = numerics_container[CONV_TERM];

  /*--- Static arrays for MUSCL reconstructed variables ---*/
  su2double     Primitive_i[MAXNVAR] = {0.0},    Primitive_j[MAXNVAR] = {0.0};
  su2double     Conserved_i[MAXNVAR] = {0.0},    Conserved_j[MAXNVAR] = {0.0};
  su2double          dPdU_i[MAXNVAR] = {0.0},         dPdU_j[MAXNVAR] = {0.0};
  su2double          dTdU_i[MAXNVAR] = {0.0},         dTdU_j[MAXNVAR] = {0.0};
  su2double        dTvedU_i[MAXNVAR] = {0.0},       dTvedU_j[MAXNVAR] = {0.0};
  su2double           Eve_i[MAXNVAR] = {0.0},          Eve_j[MAXNVAR] = {0.0};
  su2double          Cvve_i[MAXNVAR] = {0.0},         Cvve_j[MAXNVAR] = {0.0};
  su2double  Project_Grad_i[MAXNVAR] = {0.0}, Project_Grad_j[MAXNVAR] = {0.0};
  su2double Gamma_i = 0.0, Gamma_j = 0.0;

  /*--- Loop over edges and calculate convective fluxes ---*/
  for(auto iEdge = 0ul; iEdge < geometry->GetnEdge(); iEdge++) {

    /*--- Retrieve node numbers and pass edge normal to CNumerics ---*/
    const auto iPoint = geometry->edges->GetNode(iEdge, 0);
    const auto jPoint = geometry->edges->GetNode(iEdge, 1);

    numerics->SetNormal(geometry->edges->GetNormal(iEdge));

    const auto Coord_i = geometry->nodes->GetCoord(iPoint);
    const auto Coord_j = geometry->nodes->GetCoord(jPoint);

    /*--- Get conserved & primitive variables from CVariable ---*/
    auto U_i = nodes->GetSolution(iPoint);   auto U_j = nodes->GetSolution(jPoint);
    auto V_i = nodes->GetPrimitive(iPoint);  auto V_j = nodes->GetPrimitive(jPoint);

    /*--- Set them with or without high order reconstruction using MUSCL strategy. ---*/
    if (!muscl) {

      numerics->SetPrimitive   (V_i, V_j);
      numerics->SetConservative(U_i, U_j);
      numerics->SetdPdU  (nodes->GetdPdU(iPoint),   nodes->GetdPdU(jPoint));
      numerics->SetdTdU  (nodes->GetdTdU(iPoint),   nodes->GetdTdU(jPoint));
      numerics->SetdTvedU(nodes->GetdTvedU(iPoint), nodes->GetdTvedU(jPoint));
      numerics->SetEve   (nodes->GetEve(iPoint),    nodes->GetEve(jPoint));
      numerics->SetCvve  (nodes->GetCvve(iPoint),   nodes->GetCvve(jPoint));
      numerics->SetGamma (nodes->GetGamma(iPoint),  nodes->GetGamma(jPoint));

    } else {

      /*--- High order reconstruction using MUSCL strategy ---*/
      su2double Vector_ij[MAXNDIM] = {0.0};
      GeometryToolbox::Distance(nDim, Coord_j, Coord_i, Vector_ij);

      /*--- Retrieve gradient information ---*/
      auto Gradient_i = nodes->GetGradient_Reconstruction(iPoint);
      auto Gradient_j = nodes->GetGradient_Reconstruction(jPoint);

      /*--- Set and extract limiters ---*/
      su2double *Limiter_i = nullptr, *Limiter_j = nullptr;

      if (limiter && !van_albada){
        Limiter_i = nodes->GetLimiter_Primitive(iPoint);
        Limiter_j = nodes->GetLimiter_Primitive(jPoint);
      }

      su2double lim_i = 2.0;
      su2double lim_j = 2.0;

      for (auto iVar = 0ul; iVar < nPrimVarGrad; iVar++) {
        const su2double V_ij = V_j[iVar] - V_i[iVar];

        Project_Grad_i[iVar] = MUSCL_Reconstruction(Gradient_i[iVar], Vector_ij, V_ij, kappa, musclRamp);
        Project_Grad_j[iVar] = MUSCL_Reconstruction(Gradient_j[iVar], Vector_ij, V_ij, kappa, musclRamp);

        if (limiter) {
          if (van_albada) {
            su2double va_lim_i = LimiterHelpers<>::vanAlbadaFunction(Project_Grad_i[iVar], V_ij, EPS);
            su2double va_lim_j = LimiterHelpers<>::vanAlbadaFunction(Project_Grad_j[iVar], V_ij, EPS);
            lim_i = min(lim_i, va_lim_i);
            lim_j = min(lim_j, va_lim_j);
          } else {
            lim_i = min(lim_i, Limiter_i[iVar]);
            lim_j = min(lim_j, Limiter_j[iVar]);
          }
        } else {
          lim_i = lim_j = 1.0;
        }
      }
      su2double lim_ij = min(lim_i, lim_j);

      for (auto iVar = 0ul; iVar < nPrimVarGrad; iVar++) {
        Primitive_i[iVar] = V_i[iVar] + 0.5 * lim_ij * Project_Grad_i[iVar];
        Primitive_j[iVar] = V_j[iVar] - 0.5 * lim_ij * Project_Grad_j[iVar];
      }

      /*--- Check for non-physical solutions after reconstruction. If found, use the
       cell-average value of the solution. This is a locally 1st order approximation,
       which is typically only active during the start-up of a calculation. ---*/
      bool chk_err_i = CheckNonPhys(Primitive_i);
      bool chk_err_j = CheckNonPhys(Primitive_j);

      nodes->SetNon_Physical(iPoint, chk_err_i);
      nodes->SetNon_Physical(jPoint, chk_err_j);

      /*--- Get updated state, in case the point recovered after the set. ---*/
      chk_err_i = nodes->GetNon_Physical(iPoint);
      chk_err_j = nodes->GetNon_Physical(jPoint);

      counter_local += chk_err_i + chk_err_j;

      /*--- Compute Secondary variables in a thermaodynamically consistent way. ---*/
      if (!chk_err_i) Gamma_i = ComputeConsistentExtrapolation(GetFluidModel(), nSpecies, Primitive_i, dPdU_i, dTdU_i, dTvedU_i, Eve_i, Cvve_i);
      if (!chk_err_j) Gamma_j = ComputeConsistentExtrapolation(GetFluidModel(), nSpecies, Primitive_j, dPdU_j, dTdU_j, dTvedU_j, Eve_j, Cvve_j);

      /*--- Recompute Conserved variables if Roe or MSW scheme ---*/
      if ((config->GetKind_Upwind_Flow() == UPWIND::ROE) || (config->GetKind_Upwind_Flow() == UPWIND::MSW)){
        if (!chk_err_i) RecomputeConservativeVector(Conserved_i, Primitive_i);
        if (!chk_err_j) RecomputeConservativeVector(Conserved_j, Primitive_j);
      }

      /*--- If non-physical, revert to first order ---*/
      numerics->SetConservative(chk_err_i ? U_i : Conserved_i, chk_err_j ? U_j : Conserved_j);
      numerics->SetPrimitive   (chk_err_i ? V_i : Primitive_i, chk_err_j ? V_j : Primitive_j);
      numerics->SetdPdU  (chk_err_i ? nodes->GetdPdU  (iPoint) : dPdU_i,    chk_err_j ? nodes->GetdPdU  (jPoint) : dPdU_j);
      numerics->SetdTdU  (chk_err_i ? nodes->GetdTdU  (iPoint) : dTdU_i,    chk_err_j ? nodes->GetdTdU  (jPoint) : dTdU_j);
      numerics->SetdTvedU(chk_err_i ? nodes->GetdTvedU(iPoint) : dTvedU_i,  chk_err_j ? nodes->GetdTvedU(jPoint) : dTvedU_j);
      numerics->SetEve   (chk_err_i ? nodes->GetEve   (iPoint) : Eve_i,     chk_err_j ? nodes->GetEve   (jPoint) : Eve_j);
      numerics->SetCvve  (chk_err_i ? nodes->GetCvve  (iPoint) : Cvve_i,    chk_err_j ? nodes->GetCvve  (jPoint) : Cvve_j);
      numerics->SetGamma (chk_err_i ? nodes->GetGamma (iPoint) : Gamma_i,   chk_err_j ? nodes->GetGamma (jPoint) : Gamma_j);

    }
    if (config->GetKind_Upwind_Flow() == UPWIND::AUSMPLUSM)
      numerics->SetSensor (nodes->GetSensor(iPoint), nodes->GetSensor(jPoint));

    /*--- Compute the residual ---*/
    auto residual = numerics->ComputeResidual(config);

    /*--- Check for NaNs before applying the residual to the linear system ---*/
    bool err = CNumerics::CheckResidualNaNs(implicit, nVar, residual);

    /*--- Update the residual and Jacobian ---*/
    if (!err) {
      LinSysRes.AddBlock(iPoint, residual);
      LinSysRes.SubtractBlock(jPoint, residual);
      if (implicit) {
        Jacobian.UpdateBlocks(iEdge, iPoint, jPoint, residual.jacobian_i, residual.jacobian_j);
      }
    }
  }

  /*--- Warning message about non-physical reconstructions. ---*/
  if ((iMesh == MESH_0) && (config->GetComm_Level() == COMM_FULL)) {
    /*--- Add counter results for all threads. ---*/
    atomicAdd(counter_local, ErrorCounter);

    /*--- Add counter results for all ranks. ---*/
    BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS
    {
      counter_local = ErrorCounter;
      SU2_MPI::Reduce(&counter_local, &ErrorCounter, 1, MPI_UNSIGNED_LONG, MPI_SUM, MASTER_NODE, SU2_MPI::GetComm());
      config->SetNonphysical_Reconstr(ErrorCounter);
    }
    END_SU2_OMP_SAFE_GLOBAL_ACCESS
  }
}

su2double CNEMOEulerSolver::ComputeConsistentExtrapolation(CNEMOGas *fluidmodel, unsigned short nSpecies, su2double *V,
                                                           su2double* dPdU, su2double* dTdU, su2double* dTvedU,
                                                           su2double* val_eves, su2double *val_Cvves) {
  SU2_ZONE_SCOPED

  //NOTE: TODO - this doesnt compute Cvves/ dPdU,etc.yet

  /*--- Rename index information ---*/
  const unsigned short T_INDEX   = nSpecies;
  const unsigned short TVE_INDEX = nSpecies+1;

  /*--- Rename density vector ---*/
  vector<su2double> rhos;
  rhos.resize(nSpecies,0.0);
  for (auto iSpecies = 0ul; iSpecies < nSpecies; iSpecies++ ){
    rhos[iSpecies] = V[iSpecies];
  }

  /*--- Set new fluid state ---*/
  fluidmodel->SetTDStateRhosTTv(rhos, V[T_INDEX], V[TVE_INDEX]);

  /*---Compute the secondary values ---*/
  // This block of code copies a vector to corresponding pointer.
  auto it = val_eves;
  auto& ref = fluidmodel->ComputeSpeciesEve(V[TVE_INDEX]);
  for (const auto& v : ref) {
    *it = v;  ++it;
  }

  const su2double val_gamma = fluidmodel->ComputeGamma();

  return val_gamma;
}

void CNEMOEulerSolver::RecomputeConservativeVector(su2double *U, const su2double *V) const {
  SU2_ZONE_SCOPED

  /*---Useful variables ---*/
  vector<su2double> rhos;
  rhos.resize(nSpecies,0.0);

  /*--- Set Indices ---*/
  //TODO: Move these to a general location
  const unsigned short RHO_INDEX = nodes->GetRhoIndex();
  const unsigned short T_INDEX   = nodes->GetTIndex();
  const unsigned short TVE_INDEX = nodes->GetTveIndex();
  const unsigned short VEL_INDEX = nodes->GetVelIndex();

  /*--- Set densities and mass fraction ---*/
  for (auto iSpecies = 0ul; iSpecies < nSpecies; iSpecies++){
    U[iSpecies]    = V[iSpecies];
    rhos[iSpecies] = V[iSpecies];
  }

  /*--- Set momentum and compute v^2 ---*/
  su2double sqvel = 0.0;
  for (auto iDim = 0ul; iDim < nDim; iDim++){
    U[nSpecies+iDim] = V[RHO_INDEX]*V[VEL_INDEX+iDim];
    sqvel += V[VEL_INDEX+iDim]*V[VEL_INDEX+iDim];
  }

  /*--- Set the fluidmodel and recompute energies ---*/
  FluidModel->SetTDStateRhosTTv( rhos, V[T_INDEX], V[TVE_INDEX]);
  const auto& Energies = FluidModel->ComputeMixtureEnergies();

  /*--- Set conservative energies ---*/
  U[nSpecies+nDim]   = V[RHO_INDEX]*(Energies[0]+0.5*sqvel);
  U[nSpecies+nDim+1] = V[RHO_INDEX]*(Energies[1]);

}

bool CNEMOEulerSolver::CheckNonPhys(const su2double *V) const {
  SU2_ZONE_SCOPED

  /*--- Set booleans ---*/
  bool nonPhys = false;

  /*--- Set Indices ---*/
  //TODO: Move these to a general location
  const unsigned short RHOS_INDEX = nodes->GetRhosIndex();
  const unsigned short T_INDEX    = nodes->GetTIndex();
  const unsigned short TVE_INDEX  = nodes->GetTveIndex();
  const unsigned short P_INDEX    = nodes->GetPIndex();
  const unsigned short A_INDEX    = nodes->GetAIndex();

  /*--- Set temperature clipping values ---*/
  const su2double Tmin   = 50.0; const su2double Tmax   = 8E4;
  const su2double Tvemin = 50.0; const su2double Tvemax = 8E4;

  /*--- Check whether state makes sense ---*/
  for (auto iSpecies = 0ul; iSpecies < nSpecies; iSpecies++)
    if (V[RHOS_INDEX+iSpecies] < 0.0) nonPhys = true;

  if (V[P_INDEX] < 0.0) nonPhys = true;

  if (V[T_INDEX] < Tmin || V[T_INDEX] > Tmax) nonPhys = true;

  if (V[TVE_INDEX] < Tvemin || V[TVE_INDEX] > Tvemax) nonPhys = true;

  if (V[A_INDEX] < 0.0 ) nonPhys = true;

  return nonPhys;

}

void CNEMOEulerSolver::Source_Residual(CGeometry *geometry, CSolver **solver_container, CNumerics **numerics_container, CConfig *config, unsigned short iMesh) {
  SU2_ZONE_SCOPED

#ifdef HAVE_KOKKOS
  if (!sourceHostOnly && Device_SourceResidual(geometry, solver_container, numerics_container, config, iMesh)) return;
  Device_BeginHostStage(geometry);
#endif

  /*--- Assign booleans ---*/
  bool err        = false;
  const bool implicit   = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);
  const bool frozen     = config->GetFrozen();
  const bool monoatomic = config->GetMonoatomic();
  const bool axisymm    = config->GetAxisymmetric();
  const bool viscous    = config->GetViscous();
  const bool rans       = (config->GetKind_Turb_Model() != TURB_MODEL::NONE);

  CNumerics* numerics = numerics_container[SOURCE_FIRST_TERM];

  /*--- Initialize the error counter ---*/
  unsigned long eAxi_local = 0;
  unsigned long eChm_local = 0;
  unsigned long eVib_local = 0;

  /*--- Initialize the source residual to zero ---*/
  for (auto iVar = 0ul; iVar < nVar; iVar++) Residual[iVar] = 0.0;

  /*--- Preprocess viscous axisymm variables (if necessary) ---*/
  if (axisymm && viscous) {
    ComputeAxisymmetricAuxGradients(geometry,config);
  }

  AD::StartNoSharedReading();

  /*--- loop over interior points ---*/
  SU2_OMP_FOR_DYN(omp_chunk_size)
  for (auto iPoint = 0ul; iPoint < nPointDomain; iPoint++) {

    /*--- Set conserved & primitive variables  ---*/
    numerics->SetConservative(nodes->GetSolution(iPoint),  nullptr);
    numerics->SetPrimitive   (nodes->GetPrimitive(iPoint), nullptr);

    /*--- Pass supplementary information to CNumerics ---*/
    numerics->SetdPdU(nodes->GetdPdU(iPoint),     nullptr);
    numerics->SetdTdU(nodes->GetdTdU(iPoint),     nullptr);
    numerics->SetdTvedU(nodes->GetdTvedU(iPoint), nullptr);
    numerics->SetEve(nodes->GetEve(iPoint),       nullptr);
    numerics->SetCvve(nodes->GetCvve(iPoint),     nullptr);

    /*--- Set volume of the dual grid cell ---*/
    numerics->SetVolume(geometry->nodes->GetVolume(iPoint));
    numerics->SetCoord(geometry->nodes->GetCoord(iPoint), nullptr);

    /*--- Compute finite rate chemistry ---*/

    if(!monoatomic){
      if(!frozen){
        /*--- Compute the non-equilibrium chemistry ---*/
        auto residual = numerics->ComputeChemistry(config);

        /*--- Check for errors before applying source to the linear system ---*/
        err = CNumerics::CheckResidualNaNs(implicit, nVar, residual);

        /*--- Apply the chemical sources to the linear system ---*/
        if (!err) {
          LinSysRes.SubtractBlock(iPoint, residual);
          if (implicit)
            Jacobian.SubtractBlock2Diag(iPoint, residual.jacobian_i);
        } else
          eChm_local++;
      }
    }

    /*--- Compute vibrational energy relaxation ---*/
    /// NOTE: Jacobians don't account for relaxation time derivatives

    if (!monoatomic){
      auto residual = numerics->ComputeVibRelaxation(config);

      /*--- Check for errors before applying source to the linear system ---*/
      err = CNumerics::CheckResidualNaNs(implicit, nVar, residual);

      /*--- Apply the vibrational relaxation terms to the linear system ---*/
      if (!err) {
        LinSysRes.SubtractBlock(iPoint, residual);
        if (implicit)
          Jacobian.SubtractBlock2Diag(iPoint, residual.jacobian_i);
      } else
        eVib_local++;
    }

    /*--- Compute axisymmetric source terms (if needed) ---*/
    if (axisymm) {

      /*--- If necessary, set variables needed for viscous computation ---*/
      if (viscous) {

        /*--- Set gradient of primitive variables ---*/
        numerics->SetPrimVarGradient(nodes->GetGradient_Primitive(iPoint), nullptr);

        /*--- Set gradient of auxillary variables ---*/
        numerics->SetAuxVarGrad(nodes->GetAuxVarGradient(iPoint), nullptr);

        /*--- Set diffusion coefficient ---*/
        numerics->SetDiffusionCoeff(nodes->GetDiffusionCoeff(iPoint), nullptr);

        /*--- Laminar viscosity ---*/
        numerics->SetLaminarViscosity(nodes->GetLaminarViscosity(iPoint), 0.0);

        /*--- Eddy viscosity ---*/
        numerics->SetEddyViscosity(nodes->GetEddyViscosity(iPoint), 0.0);

        /*--- Thermal conductivity ---*/
        numerics->SetThermalConductivity(nodes->GetThermalConductivity(iPoint), 0.0);

        /*--- Vib-el. thermal conductivity ---*/
        numerics->SetThermalConductivity_ve(nodes->GetThermalConductivity_ve(iPoint), 0.0);

        /*--- Set turbulence kinetic energy ---*/
        if (rans){
          CVariable* turbNodes = solver_container[TURB_SOL]->GetNodes();
          numerics->SetTurbKineticEnergy(turbNodes->GetSolution(iPoint,0), 0.0);
        }
      }

      auto residual = numerics->ComputeAxisymmetric(config);

      /*--- Check for errors before applying source to the linear system ---*/
      err = CNumerics::CheckResidualNaNs(implicit, nVar, residual);

      /*--- Apply the update to the linear system ---*/
      if (!err) {
        LinSysRes.AddBlock(iPoint, residual);
        if (implicit)
          Jacobian.AddBlock2Diag(iPoint, residual.jacobian_i);
      } else
        eAxi_local++;
    }
  }
  END_SU2_OMP_FOR

  AD::EndNoSharedReading();

  /*--- Checking for NaN ---*/
  unsigned long eAxi_global = eAxi_local;
  unsigned long eChm_global = eChm_local;
  unsigned long eVib_global = eVib_local;

  //THIS IS NO FUN
  if ((eAxi_global != 0) ||
      (eChm_global != 0) ||
      (eVib_global != 0)) {
    cout << "Warning!! Instances of NaN in the following source terms: " << endl;
    cout << "Axisymmetry: " << eAxi_global << endl;
    cout << "Chemical:    " << eChm_global << endl;
    cout << "Vib. Relax:  " << eVib_global << endl;
  }
}

void CNEMOEulerSolver::ExplicitRK_Iteration(CGeometry *geometry, CSolver **solver_container,
                                            CConfig *config, unsigned short iRKStep) {
  SU2_ZONE_SCOPED

  if (Device_ExplicitRK(geometry, config, iRKStep, false)) return;
  Device_ResidualToHost(geometry); /*--- no-op unless a device residual is pending ---*/

  Explicit_Iteration<RUNGE_KUTTA_EXPLICIT>(geometry, solver_container, config, iRKStep);
}

void CNEMOEulerSolver::ClassicalRK4_Iteration(CGeometry *geometry, CSolver **solver_container,
                                              CConfig *config, unsigned short iRKStep) {
  SU2_ZONE_SCOPED

  if (Device_ExplicitRK(geometry, config, iRKStep, true)) return;
  Device_ResidualToHost(geometry); /*--- no-op unless a device residual is pending ---*/

  Explicit_Iteration<CLASSICAL_RK4_EXPLICIT>(geometry, solver_container, config, iRKStep);
}

void CNEMOEulerSolver::ExplicitEuler_Iteration(CGeometry *geometry, CSolver **solver_container, CConfig *config) {
  SU2_ZONE_SCOPED

#ifdef HAVE_KOKKOS
  if (Device_ResidentActive()) {
    /*--- Device Explicit_Iteration_impl<EULER_EXPLICIT>: U -= (R + truncation error) dt/Vol, residual reductions
     *    (scalars) into the CPU SetResidual_RMS / MPI logic, staged SOLUTION comms. ---*/
    CNEMOKokkosData::ResidualReduction red;
    kokkosData->ExplicitEulerUpdate(red);
    ResidualReductions_FromAllThreads(geometry, config, red.rms, red.max, red.idx);
    Device_StagedComms(geometry, config, MPI_QUANTITIES::SOLUTION, CNEMOKokkosData::G_SOLUTION);
    ComputeVerificationError(geometry, config);
    return;
  }
#endif

  Device_ResidualToHost(geometry); /*--- no-op unless a device residual is pending ---*/

  Explicit_Iteration<EULER_EXPLICIT>(geometry, solver_container, config, 0);
}

void CNEMOEulerSolver::PrepareImplicitIteration(CGeometry *geometry, CSolver**, CConfig *config) {
  SU2_ZONE_SCOPED

#ifdef HAVE_KOKKOS
  if (Device_ResidentActive()) {
    /*--- Device PrepareImplicitIteration_impl: diagonal += Vol/dt into the device Jacobian (= CSysMatrix d_matrix),
     *    RHS / initial guess in the device LinSysRes / LinSysSol, residual reductions -> host scalars. The linear
     *    solver then uses the device values directly (CSysMatrix::SetKokkosValuesOnDevice). ---*/
    CNEMOKokkosData::ResidualReduction red;
    kokkosData->PrepareImplicit(config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT, red);
    ResidualReductions_FromAllThreads(geometry, config, red.rms, red.max, red.idx);
    return;
  }
#endif

  Device_ResidualToHost(geometry); /*--- no-op unless a device residual is pending ---*/

  struct DummyPrec {
    const bool active = false;
    FORCEINLINE su2double** operator() (const CConfig*, unsigned long, su2double) const { return nullptr; }
  } precond;

  PrepareImplicitIteration_impl(precond, geometry, config);
}

void CNEMOEulerSolver::CompleteImplicitIteration(CGeometry *geometry, CSolver**, CConfig *config) {
  SU2_ZONE_SCOPED

#ifdef HAVE_KOKKOS
  if (Device_ResidentActive()) {
    /*--- Device ComputeUnderRelaxationFactor + AddSolution into the resident U, staged SOLUTION comms. ---*/
    kokkosData->CompleteImplicit(config->GetMaxUpdateFractionFlow());
    Device_StagedComms(geometry, config, MPI_QUANTITIES::SOLUTION, CNEMOKokkosData::G_SOLUTION);
    ComputeVerificationError(geometry, config);
    return;
  }
#endif

  CompleteImplicitIteration_impl<true>(geometry, config);
}

void CNEMOEulerSolver::ComputeUnderRelaxationFactor(const CConfig *config) {
  SU2_ZONE_SCOPED

  /* Loop over the solution update given by relaxing the linear
   system for this nonlinear iteration. */

  const su2double allowableRatio = config->GetMaxUpdateFractionFlow();

  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (auto iPoint = 0ul; iPoint < nPointDomain; iPoint++) {
    su2double localUnderRelaxation = 1.0;

    su2double num = 0.0;
    su2double denom = 0.0;

    for (auto iVar = 0; iVar < nVar; iVar++) {
      /* We impose a limit on the maximum percentage that the
       density (sum of all species) and energy can change over a nonlinear iteration. */

      const unsigned long index = iPoint * nVar + iVar;
      if (iVar < config->GetnSpecies()) {
        num   += fabs(LinSysSol[index]);
        denom += fabs(nodes->GetSolution(iPoint, iVar));

        /*--- If final density/species, compute Under-relaxation ---*/
        if (iVar == (config ->GetnSpecies()-1)){
          su2double ratio = (num/(denom+EPS));
          if (ratio > allowableRatio) {
            localUnderRelaxation = min(allowableRatio / ratio, localUnderRelaxation);
          }
        }

        /*--- Energy ---*/
        if (iVar == (nVar-2)){
          su2double ratio = fabs(LinSysSol[index]) / (fabs(nodes->GetSolution(iPoint, iVar)) + EPS);
          if (ratio > allowableRatio) {
            localUnderRelaxation = min(allowableRatio / ratio, localUnderRelaxation);
          }
        }
      }
    }

    /* Threshold the relaxation factor in the event that there is
     a very small value. This helps avoid catastrophic crashes due
     to non-realizable states by canceling the update. */

    if (localUnderRelaxation < 1e-10) localUnderRelaxation = 0.0;

    /* Store the under-relaxation factor for this point. */

    nodes->SetUnderRelaxation(iPoint, localUnderRelaxation);
  }
  END_SU2_OMP_FOR
}

void CNEMOEulerSolver::SetNondimensionalization(CConfig *config, unsigned short iMesh) {
  SU2_ZONE_SCOPED

  su2double
  Temperature_FreeStream = 0.0, Temperature_ve_FreeStream = 0.0, Mach2Vel_FreeStream         = 0.0,
  ModVel_FreeStream      = 0.0, Energy_FreeStream         = 0.0, ModVel_FreeStreamND         = 0.0,
  Velocity_Reynolds      = 0.0, Omega_FreeStream          = 0.0, Omega_FreeStreamND          = 0.0,
  Viscosity_FreeStream   = 0.0, Density_FreeStream        = 0.0, Pressure_FreeStream         = 0.0,
  Tke_FreeStream         = 0.0, Length_Ref                = 0.0, Density_Ref                 = 0.0,
  Pressure_Ref           = 0.0, Velocity_Ref              = 0.0, Temperature_Ref             = 0.0,
  Temperature_ve_Ref     = 0.0, Time_Ref                  = 0.0, Omega_Ref                   = 0.0,
  Force_Ref              = 0.0, Gas_Constant_Ref          = 0.0, Viscosity_Ref               = 0.0,
  Conductivity_Ref       = 0.0, Energy_Ref                = 0.0, Pressure_FreeStreamND       = 0.0,
  Energy_FreeStreamND    = 0.0, Temperature_FreeStreamND  = 0.0, Temperature_ve_FreeStreamND = 0.0,
  Gas_ConstantND         = 0.0, Viscosity_FreeStreamND    = 0.0, sqvel                       = 0.0,
  Tke_FreeStreamND       = 0.0, Total_UnstTimeND          = 0.0, Delta_UnstTimeND            = 0.0,
  soundspeed             = 0.0, GasConstant_Inf           = 0.0, Froude                      = 0.0,
  Density_FreeStreamND   = 0.0, Heat_Flux_Ref             = 0.0;

  su2double Velocity_FreeStreamND[3] = {0.0, 0.0, 0.0};

  /*--- Local variables ---*/
  const su2double Alpha= config->GetAoA()*PI_NUMBER/180.0;
  const su2double Beta = config->GetAoS()*PI_NUMBER/180.0;
  const su2double Mach = config->GetMach();

  const bool unsteady = (config->GetTime_Marching() != TIME_MARCHING::STEADY);
  const bool viscous  = config->GetViscous();
  const bool gravity  = config->GetGravityForce();
  const bool turbulent = false;
  const bool tkeNeeded = ((turbulent) && (config->GetKind_Turb_Model() == TURB_MODEL::SST));
  const bool reynolds_init = (config->GetKind_InitOption() == REYNOLDS);

  /*--- The dimensional viscosity is needed to determine the free-stream conditions.
        To accomplish this, simply set the non-dimensional coefficients to the
        dimensional ones. This will be overruled later.---*/

  config->SetTemperature_Ref(1.0);
  config->SetViscosity_Ref(1.0);
  config->SetConductivity_Ref(1.0);

  /*--- Instatiate the fluid model ---*/
  switch (config->GetKind_FluidModel()) {
  case MUTATIONPP:
   #if defined(HAVE_MPP) && !defined(CODI_REVERSE_TYPE) && !defined(CODI_FORWARD_TYPE)
     FluidModel = new CMutationTCLib(config, nDim);
   #else
     SU2_MPI::Error(string("Either 1) Mutation++ has not been configured/compiled (add '-Denable-mpp=true' to your meson string) or 2) CODI must be deactivated since it is not compatible with Mutation++."),
     CURRENT_FUNCTION);
   #endif
   break;
  case SU2_NONEQ:
   FluidModel = new CSU2TCLib(config, nDim, viscous);
   break;
  }

  /*--- Compute the Free Stream Pressure, Temperatrue, and Density ---*/
  Pressure_FreeStream        = config->GetPressure_FreeStream();
  Temperature_FreeStream     = config->GetTemperature_FreeStream();
  Temperature_ve_FreeStream  = config->GetTemperature_ve_FreeStream();

  /*---                                     ---*/
  /*--- Compressible non dimensionalization ---*/
  /*---                                     ---*/

  /*--- Set mixture state based on pressure, mass fractions and temperatures ---*/
  FluidModel->SetTDStatePTTv(Pressure_FreeStream, MassFrac_Inf,
                             Temperature_FreeStream, Temperature_ve_FreeStream);

  /*--- Compute Gas Constant ---*/
  GasConstant_Inf = FluidModel->ComputeGasConstant();
  config->SetGas_Constant(GasConstant_Inf);

  /*--- Compute the freestream density, soundspeed ---*/
  Density_FreeStream = FluidModel->GetDensity();
  soundspeed         = FluidModel->ComputeSoundSpeed();
  Gamma              = FluidModel->ComputeGamma();

  /*--- Compute the Free Stream velocity, using the Mach number ---*/
  if (nDim == 2) {
    config->GetVelocity_FreeStream()[0] = cos(Alpha)*Mach*soundspeed;
    config->GetVelocity_FreeStream()[1] = sin(Alpha)*Mach*soundspeed;
  }
  if (nDim == 3) {
    config->GetVelocity_FreeStream()[0] = cos(Alpha)*cos(Beta)*Mach*soundspeed;
    config->GetVelocity_FreeStream()[1] = sin(Beta)*Mach*soundspeed;
    config->GetVelocity_FreeStream()[2] = sin(Alpha)*cos(Beta)*Mach*soundspeed;
  }

  /*--- Compute the modulus of the free stream velocity ---*/
  ModVel_FreeStream = 0.0;
  for (auto iDim = 0ul; iDim < nDim; iDim++){
    ModVel_FreeStream += config->GetVelocity_FreeStream()[iDim]*config->GetVelocity_FreeStream()[iDim];
  }
  sqvel = ModVel_FreeStream;
  ModVel_FreeStream = sqrt(ModVel_FreeStream); config->SetModVel_FreeStream(ModVel_FreeStream);

  /*--- Calculate energies ---*/
  const auto& energies = FluidModel->ComputeMixtureEnergies();

  /*--- Viscous initialization ---*/
  if (viscous) {

    /*--- First, check if there is mesh motion. If yes, use the Mach
         number relative to the body to initialize the flow. ---*/

    if (dynamic_grid) Velocity_Reynolds = config->GetMach_Motion()*Mach2Vel_FreeStream;
    else Velocity_Reynolds = ModVel_FreeStream;

    if (!reynolds_init) {

      /*--- Thermodynamics quantities based initialization ---*/
      Viscosity_FreeStream = FluidModel->GetViscosity();
      Energy_FreeStream    = energies[0] + 0.5*sqvel;

    } else {

      /*--- Reynolds based initialization not present in NEMO ---*/
      SU2_MPI::Error("Only thermodynamics quantities based initialization: set pressure, temperatures and flag INIT_OPTION= TD_CONDITIONS." , CURRENT_FUNCTION);
    }

    config->SetViscosity_FreeStream(Viscosity_FreeStream);

    /*--- Compute Reynolds number ---*/
    const su2double Reynolds = (Density_FreeStream*Velocity_Reynolds*config->GetLength_Reynolds())/Viscosity_FreeStream;
    config->SetReynolds(Reynolds);

    /*--- Turbulence kinetic energy ---*/
    Tke_FreeStream  = 3.0/2.0*(ModVel_FreeStream*ModVel_FreeStream*config->GetTurbulenceIntensity_FreeStream()*config->GetTurbulenceIntensity_FreeStream());

  } else {

    /*--- For inviscid flow, energy is calculated from the specified
       FreeStream quantities using the proper gas law. ---*/
    Energy_FreeStream = energies[0] + 0.5*sqvel;

  }

  config->SetDensity_FreeStream(Density_FreeStream);

  /*-- Compute the freestream energy. ---*/
  if (tkeNeeded) { Energy_FreeStream += Tke_FreeStream; }; config->SetEnergy_FreeStream(Energy_FreeStream);

  /*--- Compute non dimensional quantities. By definition,
     Lref is one because we have converted the grid to meters. ---*/

  if (config->GetRef_NonDim() == DIMENSIONAL) {
    Pressure_Ref       = 1.0;
    Density_Ref        = 1.0;
    Temperature_Ref    = 1.0;
    Temperature_ve_Ref = 1.0;
  }
  else if (config->GetRef_NonDim() == FREESTREAM_PRESS_EQ_ONE) {
    Pressure_Ref       = Pressure_FreeStream;       // Pressure_FreeStream = 1.0
    Density_Ref        = Density_FreeStream;        // Density_FreeStream = 1.0
    Temperature_Ref    = Temperature_FreeStream;    // Temperature_FreeStream = 1.0
    Temperature_ve_Ref = Temperature_ve_FreeStream; // Temperature_ve_FreeStream = 1.0
  }
  else if (config->GetRef_NonDim() == FREESTREAM_VEL_EQ_MACH) {
    Pressure_Ref       = Gamma*Pressure_FreeStream; // Pressure_FreeStream = 1.0/Gamma
    Density_Ref        = Density_FreeStream;        // Density_FreeStream = 1.0
    Temperature_Ref    = Temperature_FreeStream;    // Temp_FreeStream = 1.0
    Temperature_ve_Ref = Temperature_ve_FreeStream; // Temp_ve_FreeStream = 1.0
  }
  else if (config->GetRef_NonDim() == FREESTREAM_VEL_EQ_ONE) {
    Pressure_Ref       = Mach*Mach*Gamma*Pressure_FreeStream; // Pressure_FreeStream = 1.0/(Gamma*(M_inf)^2)
    Density_Ref        = Density_FreeStream;                  // Density_FreeStream = 1.0
    Temperature_Ref    = Temperature_FreeStream;              // Temp_FreeStream = 1.0
    Temperature_ve_Ref = Temperature_ve_FreeStream;           // Temp_ve_FreeStream = 1.0
  }
  config->SetPressure_Ref(Pressure_Ref);
  config->SetDensity_Ref(Density_Ref);
  config->SetTemperature_Ref(Temperature_Ref);
  config->SetTemperature_ve_Ref(Temperature_ve_Ref);

  Length_Ref        = 1.0;                                                         config->SetLength_Ref(Length_Ref);
  Velocity_Ref      = sqrt(config->GetPressure_Ref()/config->GetDensity_Ref());    config->SetVelocity_Ref(Velocity_Ref);
  Time_Ref          = Length_Ref/Velocity_Ref;                                     config->SetTime_Ref(Time_Ref);
  Omega_Ref         = Velocity_Ref/Length_Ref;                                     config->SetOmega_Ref(Omega_Ref);
  Force_Ref         = config->GetDensity_Ref()*Velocity_Ref*Velocity_Ref*Length_Ref*Length_Ref; config->SetForce_Ref(Force_Ref);
  Heat_Flux_Ref     = Density_Ref*Velocity_Ref*Velocity_Ref*Velocity_Ref;           config->SetHeat_Flux_Ref(Heat_Flux_Ref);
  Gas_Constant_Ref  = Velocity_Ref*Velocity_Ref/config->GetTemperature_Ref();      config->SetGas_Constant_Ref(Gas_Constant_Ref);
  Viscosity_Ref     = config->GetDensity_Ref()*Velocity_Ref*Length_Ref;            config->SetViscosity_Ref(Viscosity_Ref);
  Conductivity_Ref  = Viscosity_Ref*Gas_Constant_Ref;                              config->SetConductivity_Ref(Conductivity_Ref);
  Froude            = ModVel_FreeStream/sqrt(STANDARD_GRAVITY*Length_Ref);         config->SetFroude(Froude);

  /*--- Divide by reference values, to compute the non-dimensional free-stream values ---*/

  Pressure_FreeStreamND = Pressure_FreeStream/config->GetPressure_Ref();  config->SetPressure_FreeStreamND(Pressure_FreeStreamND);
  Density_FreeStreamND  = Density_FreeStream/config->GetDensity_Ref();    config->SetDensity_FreeStreamND(Density_FreeStreamND);

  for (auto iDim = 0ul; iDim < nDim; iDim++) {
    Velocity_FreeStreamND[iDim] = config->GetVelocity_FreeStream()[iDim]/Velocity_Ref; config->SetVelocity_FreeStreamND(Velocity_FreeStreamND[iDim], iDim);
  }

  Temperature_FreeStreamND    = Temperature_FreeStream/config->GetTemperature_Ref();       config->SetTemperature_FreeStreamND(Temperature_FreeStreamND);
  Temperature_ve_FreeStreamND = Temperature_ve_FreeStream/config->GetTemperature_ve_Ref(); config->SetTemperature_ve_FreeStreamND(Temperature_ve_FreeStreamND);
  Gas_ConstantND              = config->GetGas_Constant()/Gas_Constant_Ref;                config->SetGas_ConstantND(Gas_ConstantND);

  ModVel_FreeStreamND = 0.0;
  for (auto iDim = 0ul; iDim < nDim; iDim++) ModVel_FreeStreamND += Velocity_FreeStreamND[iDim]*Velocity_FreeStreamND[iDim];
  ModVel_FreeStreamND    = sqrt(ModVel_FreeStreamND); config->SetModVel_FreeStreamND(ModVel_FreeStreamND);

  Viscosity_FreeStreamND = Viscosity_FreeStream / Viscosity_Ref;   config->SetViscosity_FreeStreamND(Viscosity_FreeStreamND);

  Tke_FreeStream  = 3.0/2.0*(ModVel_FreeStream*ModVel_FreeStream*config->GetTurbulenceIntensity_FreeStream()*config->GetTurbulenceIntensity_FreeStream());
  config->SetTke_FreeStream(Tke_FreeStream);

  Tke_FreeStreamND  = 3.0/2.0*(ModVel_FreeStreamND*ModVel_FreeStreamND*config->GetTurbulenceIntensity_FreeStream()*config->GetTurbulenceIntensity_FreeStream());
  config->SetTke_FreeStreamND(Tke_FreeStreamND);

  Omega_FreeStream = Density_FreeStream*Tke_FreeStream/(Viscosity_FreeStream*config->GetTurb2LamViscRatio_FreeStream());
  config->SetOmega_FreeStream(Omega_FreeStream);

  Omega_FreeStreamND = Density_FreeStreamND*Tke_FreeStreamND/(Viscosity_FreeStreamND*config->GetTurb2LamViscRatio_FreeStream());
  config->SetOmega_FreeStreamND(Omega_FreeStreamND);

  /*--- Initialize the dimensionless Fluid Model that will be used to solve the dimensionless problem ---*/

  Energy_FreeStreamND = energies[0] + 0.5 * ModVel_FreeStreamND * ModVel_FreeStreamND;

  if (tkeNeeded) { Energy_FreeStreamND += Tke_FreeStreamND; };  config->SetEnergy_FreeStreamND(Energy_FreeStreamND);

  Energy_Ref = Energy_FreeStream/Energy_FreeStreamND; config->SetEnergy_Ref(Energy_Ref);

  Total_UnstTimeND = config->GetTotal_UnstTime() / Time_Ref;    config->SetTotal_UnstTimeND(Total_UnstTimeND);
  Delta_UnstTimeND = config->GetDelta_UnstTime() / Time_Ref;    config->SetDelta_UnstTimeND(Delta_UnstTimeND);

  /*--- Write output to the console if this is the master node and first domain ---*/

  if ((rank == MASTER_NODE) && (iMesh == MESH_0)) {

    cout.precision(6);

    if (viscous) {
      if (reynolds_init){
        cout << "Viscous flow: Computing pressure using the equation of state for multi-species and multi-temperatures" << endl;
        cout << "based on the free-stream temperatures and a density computed" << endl;
        cout << "from the Reynolds number." << endl;
      } else {
        cout << "Viscous flow: Computing density using the equation of state for multi-species and multi-temperatures" << endl;
        cout << "based on the free-stream temperatures and pressure." << endl;
      }
    } else {
      cout << "Inviscid flow: Computing density based on free-stream" << endl;
      cout << "and pressure using the the equation of state for multi-species and multi-temperatures." << endl;
    }

    if (dynamic_grid) cout << "Force coefficients computed using MACH_MOTION." << endl;
    else cout << "Force coefficients computed using free-stream values." << endl;

    stringstream NonDimTableOut, ModelTableOut;
    stringstream Unit;

    cout << endl;
    PrintingToolbox::CTablePrinter ModelTable(&ModelTableOut);
    ModelTableOut <<"-- Models:"<< endl;

    ModelTable.AddColumn("Mixture", 25);
    ModelTable.AddColumn("Fluid Model", 25);
    ModelTable.AddColumn("Transport Model", 25);
    ModelTable.SetAlign(PrintingToolbox::CTablePrinter::RIGHT);
    ModelTable.PrintHeader();

    PrintingToolbox::CTablePrinter NonDimTable(&NonDimTableOut);
    NonDimTable.AddColumn("Name", 22);
    NonDimTable.AddColumn("Dim. value", 14);
    NonDimTable.AddColumn("Ref. value", 14);
    NonDimTable.AddColumn("Unit", 10);
    NonDimTable.AddColumn("Non-dim. value", 14);
    NonDimTable.SetAlign(PrintingToolbox::CTablePrinter::RIGHT);

    NonDimTableOut <<"-- Fluid properties:"<< endl;

    NonDimTable.PrintHeader();

    if      (config->GetSystemMeasurements() == SI) Unit << "N.m/kg.K";
    else if (config->GetSystemMeasurements() == US) Unit << "lbf.ft/slug.R";
    NonDimTable << "Gas Constant" << config->GetGas_Constant() << config->GetGas_Constant_Ref() << Unit.str() << config->GetGas_ConstantND();
    Unit.str("");
    if      (config->GetSystemMeasurements() == SI) Unit << "N.m/kg.K";
    else if (config->GetSystemMeasurements() == US) Unit << "lbf.ft/slug.R";
    NonDimTable << "Spec. Heat Ratio" << "-" << "-" << "-" << Gamma;
    Unit.str("");

    ModelTable << config->GetGasModel();

    switch(config->GetKind_FluidModel()){
    case SU2_NONEQ:
      ModelTable << "SU2 NonEq";
      break;
    case MUTATIONPP:
      ModelTable << "Mutation++ NonEq";
      break;
    }

    if (viscous) {

      switch(config->GetKind_TransCoeffModel()){
      case TRANSCOEFFMODEL::SUTHERLAND:
        ModelTable << "Sutherland + WBE";
        NonDimTable.PrintFooter();
        break;

      case TRANSCOEFFMODEL::WILKE:
        ModelTable << "Wilke-Blottner-Eucken";
        NonDimTable.PrintFooter();
        break;

      case TRANSCOEFFMODEL::GUPTAYOS:
        ModelTable << "Gupta-Yos";
        NonDimTable.PrintFooter();
        break;

      case TRANSCOEFFMODEL::CHAPMANN_ENSKOG:
        ModelTable << "CHAPMANN-ENSKOG_LDLT";
        NonDimTable.PrintFooter();
        break;

      default:
        break;
      }
    } else {
      ModelTable << "-" ;
    }

    NonDimTable.PrintFooter();
    NonDimTableOut <<"-- Initial and free-stream conditions:"<< endl;
    NonDimTable.PrintHeader();

    if      (config->GetSystemMeasurements() == SI) Unit << "Pa";
    else if (config->GetSystemMeasurements() == US) Unit << "psf";
    NonDimTable << "Static Pressure" << config->GetPressure_FreeStream() << config->GetPressure_Ref() << Unit.str() << config->GetPressure_FreeStreamND();
    Unit.str("");
    if      (config->GetSystemMeasurements() == SI) Unit << "kg/m^3";
    else if (config->GetSystemMeasurements() == US) Unit << "slug/ft^3";
    NonDimTable << "Density" << config->GetDensity_FreeStream() << config->GetDensity_Ref() << Unit.str() << config->GetDensity_FreeStreamND();
    Unit.str("");
    if      (config->GetSystemMeasurements() == SI) Unit << "K";
    else if (config->GetSystemMeasurements() == US) Unit << "R";
    NonDimTable << " T-R Temperature" << config->GetTemperature_FreeStream() << config->GetTemperature_Ref() << Unit.str() << config->GetTemperature_FreeStreamND();
    Unit.str("");
    if      (config->GetSystemMeasurements() == SI) Unit << "K";
    else if (config->GetSystemMeasurements() == US) Unit << "R";
    NonDimTable << " V-E Temperature" << config->GetTemperature_ve_FreeStream() << config->GetTemperature_ve_Ref() << Unit.str() << config->GetTemperature_ve_FreeStreamND();
    Unit.str("");
    if      (config->GetSystemMeasurements() == SI) Unit << "m^2/s^2";
    else if (config->GetSystemMeasurements() == US) Unit << "ft^2/s^2";
    NonDimTable << "Total Energy" << config->GetEnergy_FreeStream() << config->GetEnergy_Ref() << Unit.str() << config->GetEnergy_FreeStreamND();
    Unit.str("");
    if      (config->GetSystemMeasurements() == SI) Unit << "m/s";
    else if (config->GetSystemMeasurements() == US) Unit << "ft/s";
    NonDimTable << "Velocity-X" << config->GetVelocity_FreeStream()[0] << config->GetVelocity_Ref() << Unit.str() << config->GetVelocity_FreeStreamND()[0];
    NonDimTable << "Velocity-Y" << config->GetVelocity_FreeStream()[1] << config->GetVelocity_Ref() << Unit.str() << config->GetVelocity_FreeStreamND()[1];
    if (nDim == 3){
      NonDimTable << "Velocity-Z" << config->GetVelocity_FreeStream()[2] << config->GetVelocity_Ref() << Unit.str() << config->GetVelocity_FreeStreamND()[2];
    }
    NonDimTable << "Velocity Magnitude" << config->GetModVel_FreeStream() << config->GetVelocity_Ref() << Unit.str() << config->GetModVel_FreeStreamND();
    Unit.str("");

    if (viscous){
      NonDimTable.PrintFooter();
      if      (config->GetSystemMeasurements() == SI) Unit << "N.s/m^2";
      else if (config->GetSystemMeasurements() == US) Unit << "lbf.s/ft^2";
      NonDimTable << "Viscosity" << config->GetViscosity_FreeStream() << config->GetViscosity_Ref() << Unit.str() << config->GetViscosity_FreeStreamND();
      Unit.str("");
      if (turbulent){
        if      (config->GetSystemMeasurements() == SI) Unit << "m^2/s^2";
        else if (config->GetSystemMeasurements() == US) Unit << "ft^2/s^2";
        NonDimTable << "Turb. Kin. Energy" << config->GetTke_FreeStream() << config->GetTke_FreeStream()/config->GetTke_FreeStreamND() << Unit.str() << config->GetTke_FreeStreamND();
        Unit.str("");
        if      (config->GetSystemMeasurements() == SI) Unit << "1/s";
        else if (config->GetSystemMeasurements() == US) Unit << "1/s";
        NonDimTable << "Spec. Dissipation" << config->GetOmega_FreeStream() << config->GetOmega_FreeStream()/config->GetOmega_FreeStreamND() << Unit.str() << config->GetOmega_FreeStreamND();
        Unit.str("");
      }
    }

    NonDimTable.PrintFooter();
    NonDimTable << "Mach Number" << "-" << "-" << "-" << config->GetMach();
    if (viscous){
      NonDimTable << "Reynolds Number" << "-" << "-" << "-" << config->GetReynolds();
    }
    if (gravity) {
      NonDimTable << "Froude Number" << "-" << "-" << "-" << Froude;
      NonDimTable << "Wave Length"   << "-" << "-" << "-" << 2.0*PI_NUMBER*Froude*Froude;
    }
    NonDimTable.PrintFooter();
    ModelTable.PrintFooter();

    if (unsteady){
      NonDimTableOut << "-- Unsteady conditions" << endl;
      NonDimTable.PrintHeader();
      NonDimTable << "Total Time" << config->GetMax_Time() << config->GetTime_Ref() << "s" << config->GetMax_Time()/config->GetTime_Ref();
      Unit.str("");
      NonDimTable << "Time Step" << config->GetTime_Step() << config->GetTime_Ref() << "s" << config->GetDelta_UnstTimeND();
      Unit.str("");
      NonDimTable.PrintFooter();
    }

    cout << ModelTableOut.str();
    cout << NonDimTableOut.str();

  }
}

void CNEMOEulerSolver::SetReferenceValues(const CConfig& config) {
  SU2_ZONE_SCOPED

  DynamicPressureRef = 0.5 * Density_Inf * GeometryToolbox::SquaredNorm(nDim, Velocity_Inf);

  if (DynamicPressureRef < EPS) {
    DynamicPressureRef = 1.0;
  }

  AeroCoeffForceRef =  DynamicPressureRef * config.GetRefArea();

}

void CNEMOEulerSolver::BC_Sym_Plane(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics,
                                    CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {
  SU2_ZONE_SCOPED

#ifdef HAVE_KOKKOS
  if (!bcHostOnly && Device_BCResidual(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker,
                                       static_cast<int>(CNEMOKokkosData::BCKind::SYMMETRY)))
    return;
  Device_BeginHostStage(geometry);
#endif

  /*--- Allocate the necessary vector structures ---*/
  su2double Normal[MAXNDIM] = {0.0}, UnitNormal[MAXNDIM] = {0.0};

  const bool implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);

  /*--- Loop over all the vertices on this boundary (val_marker) ---*/
  for (auto iVertex = 0ul; iVertex < geometry->nVertex[val_marker]; iVertex++) {
    const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

    /*--- Check if the node belongs to the domain (i.e, not a halo node) ---*/
    if (geometry->nodes->GetDomain(iPoint)) {

      /*--- Normal vector for this vertex (negative for outward convention) ---*/
      geometry->vertex[val_marker][iVertex]->GetNormal(Normal);

      /*--- Calculate parameters from the geometry ---*/
      const su2double Area = GeometryToolbox::Norm(nDim, Normal);

      for (auto iDim = 0ul; iDim < nDim; iDim++){
        UnitNormal[iDim] = -Normal[iDim]/Area;
      }

      /*--- Retrieve the pressure on the vertex ---*/
      const su2double P = nodes->GetPressure(iPoint);

      /*--- Apply the flow-tangency b.c. to the convective flux ---*/
      for (auto iSpecies = 0ul; iSpecies < nSpecies; iSpecies++)
        Residual[iSpecies] = 0.0;
      for (auto iDim = 0ul; iDim < nDim; iDim++){
        Residual[nSpecies+iDim] = P * UnitNormal[iDim] * Area;
      }
      Residual[nSpecies+nDim]   = 0.0;
      Residual[nSpecies+nDim+1] = 0.0;

      /*--- Add value to the residual ---*/
      LinSysRes.AddBlock(iPoint, Residual);

      /*--- If using implicit time-stepping, calculate b.c. contribution to Jacobian ---*/
      if (implicit) {

        /*--- Allocate arrays needed for implicit solver ---*/
        su2double Velocity[MAXNDIM] = {0.0};

        /*--- Get species molar mass ---*/
        auto& Ms = FluidModel->GetSpeciesMolarMass();

        /*--- Initialize Jacobian ---*/
        for (auto iVar = 0ul; iVar < nVar; iVar++)
          for (auto jVar = 0ul; jVar < nVar; jVar++)
            Jacobian_i[iVar][jVar] = 0.0;

        /*--- Calculate state i ---*/
        const su2double rho    = nodes->GetDensity(iPoint);
        const su2double rhoE   = nodes->GetSolution(iPoint,nSpecies+nDim);
        const su2double rhoEve = nodes->GetSolution(iPoint,nSpecies+nDim+1);
        const auto dPdU    = nodes->GetdPdU(iPoint);
        for (auto iDim = 0ul; iDim < nDim; iDim++)
          Velocity[iDim] = nodes->GetVelocity(iPoint,iDim);

        su2double conc = 0.0;
        for (auto iSpecies = 0ul; iSpecies < nSpecies; iSpecies++) {
          const su2double cs = nodes->GetMassFraction(iPoint,iSpecies);
          conc += cs * rho/Ms[iSpecies];

          for (auto iDim = 0ul; iDim < nDim; iDim++) {
            Jacobian_i[nSpecies+iDim][iSpecies] = dPdU[iSpecies] * UnitNormal[iDim];
            Jacobian_i[iSpecies][nSpecies+iDim] = cs * UnitNormal[iDim];
          }
        }

        for (auto iDim = 0ul; iDim < nDim; iDim++) {
          for (auto jDim = 0ul; jDim < nDim; jDim++) {
            Jacobian_i[nSpecies+iDim][nSpecies+jDim] = Velocity[iDim]*UnitNormal[jDim]
                + dPdU[nSpecies+jDim]*UnitNormal[iDim];
          }
          Jacobian_i[nSpecies+iDim][nSpecies+nDim]   = dPdU[nSpecies+nDim]  *UnitNormal[iDim];
          Jacobian_i[nSpecies+iDim][nSpecies+nDim+1] = dPdU[nSpecies+nDim+1]*UnitNormal[iDim];

          Jacobian_i[nSpecies+nDim][nSpecies+iDim]   = (rhoE+P)/rho * UnitNormal[iDim];
          Jacobian_i[nSpecies+nDim+1][nSpecies+iDim] = rhoEve/rho   * UnitNormal[iDim];
        }

        /*--- Integrate over the dual-grid area ---*/
        for (auto iVar = 0ul; iVar < nVar; iVar++)
          for (auto jVar = 0ul; jVar < nVar; jVar++)
            Jacobian_i[iVar][jVar] = Jacobian_i[iVar][jVar] * Area;

        /*--- Apply the contribution to the system ---*/
        Jacobian.AddBlock2Diag(iPoint, Jacobian_i);

      }
    }
  }
}

void CNEMOEulerSolver::BC_Far_Field(CGeometry *geometry,
                                    CSolver **solver_container,
                                    CNumerics *conv_numerics,
                                    CNumerics *visc_numerics, CConfig *config,
                                    unsigned short val_marker) {
  SU2_ZONE_SCOPED

  if (!bcHostOnly) RecordBoundaryNumerics(geometry, config, val_marker, 1); /*--- BCKind::FAR_FIELD ---*/
#ifdef HAVE_KOKKOS
  if (!bcHostOnly && Device_BCResidual(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker,
                                       static_cast<int>(CNEMOKokkosData::BCKind::FAR_FIELD)))
    return;
  Device_BeginHostStage(geometry);
#endif

  /*--- Allocate arrays ---*/
  su2double Normal[MAXNDIM] = {0.0};

  /*--- Loop over all the vertices on this boundary (val_marker) ---*/
  for (auto iVertex = 0ul; iVertex < geometry->nVertex[val_marker];
       iVertex++) {
    const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

    /*--- Check if the node belongs to the domain (i.e, not a halo node) ---*/
    if (geometry->nodes->GetDomain(iPoint)) {

      /*--- Retrieve index of the closest interior node ---*/
      const auto Point_Normal = geometry->vertex[val_marker][iVertex]
              ->GetNormal_Neighbor(); // only used for implicit

      /*--- Pass boundary node normal to CNumerics ---*/
      geometry->vertex[val_marker][iVertex]->GetNormal(Normal);
      for (auto iDim = 0ul; iDim < nDim; iDim++)
        Normal[iDim] = -Normal[iDim];
      conv_numerics->SetNormal(Normal);

      /*--- Pass conserved & primitive variables to CNumerics ---*/
      conv_numerics->SetConservative(nodes->GetSolution(iPoint),
                                     node_infty->GetSolution(0));
      conv_numerics->SetPrimitive(nodes->GetPrimitive(iPoint),
                                  node_infty->GetPrimitive(0));

      /*--- Pass supplementary information to CNumerics ---*/
      conv_numerics->SetdPdU  (nodes->GetdPdU(iPoint),   node_infty->GetdPdU(0));
      conv_numerics->SetdTdU  (nodes->GetdTdU(iPoint),   node_infty->GetdTdU(0));
      conv_numerics->SetdTvedU(nodes->GetdTvedU(iPoint), node_infty->GetdTvedU(0));
      conv_numerics->SetEve   (nodes->GetEve(iPoint),    node_infty->GetEve(0));
      conv_numerics->SetCvve  (nodes->GetCvve(iPoint),   node_infty->GetCvve(0));
      conv_numerics->SetGamma (nodes->GetGamma(iPoint),  node_infty->GetGamma(0));
      if(config->GetKind_Upwind_Flow() == UPWIND::AUSMPLUSM)
        conv_numerics->SetSensor (nodes->GetSensor(iPoint), nodes->GetSensor(iPoint));

      /*--- Compute the convective residual (and Jacobian) ---*/
      // Note: This uses the specified boundary num. method specified in
      // driver_structure.cpp
      auto residual = conv_numerics->ComputeResidual(config);

      /*--- Apply contribution to the linear system ---*/
      LinSysRes.AddBlock(iPoint, residual);

      const bool implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);
      if (implicit)
        Jacobian.AddBlock2Diag(iPoint, residual.jacobian_i);

      /*--- Viscous contribution ---*/
      const bool viscous = config->GetViscous();
      if (viscous) {
        su2double Coord_Reflected[MAXNDIM];
        GeometryToolbox::PointPointReflect(
            nDim, geometry->nodes->GetCoord(Point_Normal),
            geometry->nodes->GetCoord(iPoint), Coord_Reflected);
        visc_numerics->SetCoord(geometry->nodes->GetCoord(iPoint),
                                Coord_Reflected);
        visc_numerics->SetNormal(Normal);

        /*--- Primitive variables, and gradient ---*/
        visc_numerics->SetConservative(nodes->GetSolution(iPoint),
                                       node_infty->GetSolution(0));
        visc_numerics->SetPrimitive(nodes->GetPrimitive(iPoint),
                                    node_infty->GetPrimitive(0));
        visc_numerics->SetPrimVarGradient(nodes->GetGradient_Primitive(iPoint),
                                          node_infty->GetGradient_Primitive(0));

        /*--- Pass supplementary information to CNumerics ---*/
        visc_numerics->SetdPdU(nodes->GetdPdU(iPoint), node_infty->GetdPdU(0));
        visc_numerics->SetdTdU(nodes->GetdTdU(iPoint), node_infty->GetdTdU(0));
        visc_numerics->SetdTvedU(nodes->GetdTvedU(iPoint),
                                 node_infty->GetdTvedU(0));
        visc_numerics->SetEve(nodes->GetEve(iPoint), node_infty->GetEve(0));
        visc_numerics->SetCvve(nodes->GetCvve(iPoint), node_infty->GetCvve(0));

        /*--- Species diffusion coefficients ---*/
        visc_numerics->SetDiffusionCoeff(nodes->GetDiffusionCoeff(iPoint),
                                         nodes->GetDiffusionCoeff(iPoint));

        /*--- Laminar viscosity ---*/
        visc_numerics->SetLaminarViscosity(nodes->GetLaminarViscosity(iPoint),
                                           nodes->GetLaminarViscosity(iPoint));

        /*--- Eddy viscosity ---*/
        visc_numerics->SetEddyViscosity(nodes->GetEddyViscosity(iPoint),
                                        nodes->GetEddyViscosity(iPoint));

        /*--- Thermal conductivity ---*/
        visc_numerics->SetThermalConductivity(
            nodes->GetThermalConductivity(iPoint),
            nodes->GetThermalConductivity(iPoint));

        /*--- Vib-el. thermal conductivity ---*/
        visc_numerics->SetThermalConductivity_ve(
            nodes->GetThermalConductivity_ve(iPoint),
            nodes->GetThermalConductivity_ve(iPoint));

        /*--- Compute and update residual ---*/
        auto residual = visc_numerics->ComputeResidual(config);

        LinSysRes.SubtractBlock(iPoint, residual);
        if (implicit) {
          Jacobian.SubtractBlock2Diag(iPoint, residual.jacobian_i);
        }
      }
    }
  }
}

void CNEMOEulerSolver::BC_Inlet(CGeometry *geometry, CSolver **solver_container,
                                CNumerics *conv_numerics, CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {
  SU2_ZONE_SCOPED

#ifdef HAVE_KOKKOS
  Device_BeginHostStage(geometry);
#endif
  SU2_MPI::Error("BC_INLET: Not operational in NEMO.", CURRENT_FUNCTION);

  unsigned short RHO_INDEX, nSpecies;

  su2double  T_Total, P_Total, Velocity[3], Velocity2, H_Total, Temperature, Riemann,
  Pressure, Density, Energy, Mach2, SoundSpeed2, SoundSpeed_Total2, Vel_Mag,
  alpha, aa, bb, cc, dd, Area, UnitNormal[3] = {0.0};

  su2double Flow_Dir[MAXNDIM] = {};
  bool implicit             = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);

  bool dynamic_grid         = config->GetGrid_Movement();
  su2double Two_Gamma_M1    = 2.0/Gamma_Minus_One;
  su2double Gas_Constant    = config->GetGas_ConstantND();
  INLET_TYPE Kind_Inlet     = config->GetKind_Inlet();

  auto *U_domain = new su2double[nVar];      auto *U_inlet = new su2double[nVar];
  auto *V_domain = new su2double[nPrimVar];  auto *V_inlet = new su2double[nPrimVar];
  auto *Normal   = new su2double[nDim];

  nSpecies = config->GetnSpecies();
  auto *Spec_Density = new su2double[nSpecies];
  for (auto iSpecies = 0ul; iSpecies<nSpecies; iSpecies++)
    Spec_Density[iSpecies] = 0.0;               /*--- To avoid a compiler warning. ---*/

  RHO_INDEX = nodes->GetRhoIndex();

  /*--- Loop over all the vertices on this boundary marker ---*/
  for (auto iVertex = 0ul; iVertex < geometry->nVertex[val_marker]; iVertex++) {
    const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

    /*--- Check if the node belongs to the domain (i.e., not a halo node) ---*/
    if (geometry->nodes->GetDomain(iPoint)) {

      /*--- Normal vector for this vertex (negate for outward convention) ---*/
      geometry->vertex[val_marker][iVertex]->GetNormal(Normal);
      for (auto iDim = 0ul; iDim < nDim; iDim++) Normal[iDim] = -Normal[iDim];
      conv_numerics->SetNormal(Normal);

      Area = GeometryToolbox::Norm(nDim, Normal);

      for (auto iDim = 0ul; iDim < nDim; iDim++)
        UnitNormal[iDim] = Normal[iDim]/Area;

      /*--- Retrieve solution at this boundary node ---*/
      for (auto iVar = 0ul; iVar < nVar; iVar++)     U_domain[iVar] = nodes->GetSolution(iPoint, iVar);
      for (auto iVar = 0ul; iVar < nPrimVar; iVar++) V_domain[iVar] = nodes->GetPrimitive(iPoint,iVar);

      const su2double* dir = Inlet_FlowDir[val_marker][iVertex];
      const su2double mag = GeometryToolbox::Norm(nDim, dir);
      if (config->GetInletUseNormal()) {
        for (auto iDim = 0u; iDim < nDim; iDim++)
          Flow_Dir[iDim] = -UnitNormal[iDim];
      } else {
        for (auto iDim = 0u; iDim < nDim; iDim++)
          Flow_Dir[iDim] = dir[iDim]/mag;
      }

      /*--- Build the fictitious intlet state based on characteristics ---*/

      /*--- Subsonic inflow: there is one outgoing characteristic (u-c),
       therefore we can specify all but one state variable at the inlet.
       The outgoing Riemann invariant provides the final piece of info.
       Adapted from an original implementation in the Stanford University
       multi-block (SUmb) solver in the routine bcSubsonicInflow.f90
       written by Edwin van der Weide, last modified 04-20-2009. ---*/

      switch (Kind_Inlet) {

      /*--- Total properties have been specified at the inlet. ---*/
      case INLET_TYPE::TOTAL_CONDITIONS:

        /*--- Retrieve the specified total conditions for this inlet. ---*/
        P_Total  = Inlet_Ptotal[val_marker][iVertex];
        T_Total  = Inlet_Ttotal[val_marker][iVertex];

        /*--- Non-dim. the inputs if necessary. ---*/
        P_Total /= config->GetPressure_Ref();
        T_Total /= config->GetTemperature_Ref();

        /*--- Store primitives and set some variables for clarity. ---*/
        Density = V_domain[RHO_INDEX];
        for (auto iDim = 0ul; iDim < nDim; iDim++)
          Velocity[iDim] = U_domain[nSpecies+iDim]/Density;

        Velocity2   = GeometryToolbox::SquaredNorm(nDim, Velocity);
        Energy      = U_domain[nVar-2]/Density;
        Pressure    = Gamma_Minus_One*Density*(Energy-0.5*Velocity2);
        H_Total     = (Gamma*Gas_Constant/Gamma_Minus_One)*T_Total;
        SoundSpeed2 = Gamma*Pressure/Density;

        /*--- Compute the acoustic Riemann invariant that is extrapolated
           from the domain interior. ---*/
        Riemann   = 2.0*sqrt(SoundSpeed2)/Gamma_Minus_One;
        for (auto iDim = 0ul; iDim < nDim; iDim++)
          Riemann += Velocity[iDim]*UnitNormal[iDim];

        /*--- Total speed of sound ---*/
        SoundSpeed_Total2 = Gamma_Minus_One*(H_Total - (Energy + Pressure/Density)+0.5*Velocity2) + SoundSpeed2;

        /*--- Dot product of normal and flow direction. This should
           be negative due to outward facing boundary normal convention. ---*/
        alpha = GeometryToolbox::DotProduct(nDim, UnitNormal, Flow_Dir);

        /*--- Coefficients in the quadratic equation for the velocity ---*/
        aa =  1.0 + 0.5*Gamma_Minus_One*alpha*alpha;
        bb = -1.0*Gamma_Minus_One*alpha*Riemann;
        cc =  0.5*Gamma_Minus_One*Riemann*Riemann
            -2.0*SoundSpeed_Total2/Gamma_Minus_One;

        /*--- Solve quadratic equation for velocity magnitude. Value must
           be positive, so the choice of root is clear. ---*/
        dd = bb*bb - 4.0*aa*cc;
        dd = sqrt(max(0.0,dd));
        Vel_Mag   = (-bb + dd)/(2.0*aa);
        Vel_Mag   = max(0.0,Vel_Mag);
        Velocity2 = Vel_Mag*Vel_Mag;

        /*--- Compute speed of sound from total speed of sound eqn. ---*/
        SoundSpeed2 = SoundSpeed_Total2 - 0.5*Gamma_Minus_One*Velocity2;

        /*--- Mach squared (cut between 0-1), use to adapt velocity ---*/
        Mach2 = Velocity2/SoundSpeed2;
        Mach2 = min(1.0,Mach2);
        Velocity2   = Mach2*SoundSpeed2;
        Vel_Mag     = sqrt(Velocity2);
        SoundSpeed2 = SoundSpeed_Total2 - 0.5*Gamma_Minus_One*Velocity2;

        /*--- Compute new velocity vector at the inlet ---*/
        for (auto iDim = 0ul; iDim < nDim; iDim++)
          Velocity[iDim] = Vel_Mag*Flow_Dir[iDim];

        /*--- Static temperature from the speed of sound relation ---*/
        Temperature = SoundSpeed2/(Gamma*Gas_Constant);
        //NEED TVE AS WELL

        /*--- Static pressure using isentropic relation at a point ---*/
        Pressure = P_Total*pow((Temperature/T_Total),Gamma/Gamma_Minus_One);

        /*--- Density at the inlet from the gas law ---*/
        Density = Pressure/(Gas_Constant*Temperature);
        //NEED SPECIES DENSITIES

        /*--- Using pressure, density, & velocity, compute the energy ---*/
        Energy = Pressure/(Density*Gamma_Minus_One)+0.5*Velocity2;
        //NEED EVE AS WELL

        /*--- Conservative variables, using the derived quantities ---*/
        for (auto iSpecies = 0ul; iSpecies<nSpecies; iSpecies++)
          U_inlet[iSpecies] = Spec_Density[iSpecies];
        for (auto iDim = 0ul; iDim < nDim; iDim++)
          U_inlet[nSpecies+iDim] = Velocity[iDim]*Density;
        U_inlet[nVar-2] = Energy*Density;
        //U_inlet[nVar-1]=Eve

        /*--- Primitive variables, using the derived quantities ---*/
        for (auto iSpecies = 0ul; iSpecies<nSpecies; iSpecies++)
          V_inlet[iSpecies] = Spec_Density[iSpecies];
        V_inlet[nSpecies] = Temperature;
        //V_inlet[nSpecies+1] = Tve
        for (auto iDim = 0ul; iDim < nDim; iDim++)
          V_inlet[nSpecies+2] = Velocity[iDim];
        V_inlet[nSpecies+nDim+2] = Pressure;
        V_inlet[RHO_INDEX] = Density;
        //V_inlet[H_INDEX] = H;
        //V_inlet[A_INDEX] = A;
        //V_inlet[RHO_CVTR_INDEX] = rcvtr;
        //V_inlet[RHO_CVVE_INDEX] = rcvve;

        break;

      /*--- Mass flow has been specified at the inlet. ---*/
      case INLET_TYPE::MASS_FLOW:

        /*--- Retrieve the specified mass flow for the inlet. ---*/
        Vel_Mag  = Inlet_Ptotal[val_marker][iVertex];
        Density  = Inlet_Ttotal[val_marker][iVertex];

        /*--- Non-dim. the inputs if necessary. ---*/
        Density /= config->GetDensity_Ref();
        Vel_Mag /= config->GetVelocity_Ref();

        /*--- Get primitives from current inlet state. ---*/
        for (auto iDim = 0ul; iDim < nDim; iDim++)
          Velocity[iDim] = nodes->GetVelocity(iPoint, iDim);
        Pressure    = nodes->GetPressure(iPoint);
        SoundSpeed2 = Gamma*Pressure/U_domain[0];

        /*--- Compute the acoustic Riemann invariant that is extrapolated
           from the domain interior. ---*/
        Riemann = Two_Gamma_M1*sqrt(SoundSpeed2);
        for (auto iDim = 0ul; iDim < nDim; iDim++)
          Riemann += Velocity[iDim]*UnitNormal[iDim];

        /*--- Speed of sound squared for fictitious inlet state ---*/
        SoundSpeed2 = Riemann;
        for (auto iDim = 0ul; iDim < nDim; iDim++)
          SoundSpeed2 -= Vel_Mag*Flow_Dir[iDim]*UnitNormal[iDim];

        SoundSpeed2 = max(0.0,0.5*Gamma_Minus_One*SoundSpeed2);
        SoundSpeed2 = SoundSpeed2*SoundSpeed2;

        /*--- Pressure for the fictitious inlet state ---*/
        Pressure = SoundSpeed2*Density/Gamma;

        /*--- Energy for the fictitious inlet state ---*/
        Energy = Pressure/(Density*Gamma_Minus_One)+0.5*Vel_Mag*Vel_Mag;

        /*--- Conservative variables, using the derived quantities ---*/
        U_inlet[0] = Density;
        for (auto iDim = 0ul; iDim < nDim; iDim++)
          U_inlet[iDim+1] = Vel_Mag*Flow_Dir[iDim]*Density;
        U_inlet[nDim+1] = Energy*Density;

        /*--- Primitive variables, using the derived quantities ---*/
        V_inlet[0] = Pressure / ( Gas_Constant * Density);
        for (auto iDim = 0ul; iDim < nDim; iDim++)
          V_inlet[iDim+1] = Vel_Mag*Flow_Dir[iDim];
        V_inlet[nDim+1] = Pressure;
        V_inlet[nDim+2] = Density;

        break;

      default:
        SU2_MPI::Error("Unsupported INLET_TYPE.", CURRENT_FUNCTION);
        break;
      }

      /*--- Set various quantities in the solver class ---*/
      conv_numerics->SetConservative(U_domain, U_inlet);

      if (dynamic_grid)
        conv_numerics->SetGridVel(geometry->nodes->GetGridVel(iPoint), geometry->nodes->GetGridVel(iPoint));

      /*--- Compute the residual using an upwind scheme ---*/
      auto residual = conv_numerics->ComputeResidual(config);

      LinSysRes.AddBlock(iPoint, residual);

      /*--- Jacobian contribution for implicit integration ---*/
      if (implicit) Jacobian.AddBlock2Diag(iPoint, residual.jacobian_i);

//      /*--- Viscous contribution ---*/
//      if (viscous) {

//        /*--- Set the normal vector and the coordinates ---*/
//        visc_numerics->SetNormal(Normal);
//        su2double Coord_Reflected[MAXNDIM];
//        GeometryToolbox::PointPointReflect(nDim, geometry->nodes->GetCoord(Point_Normal),
//                                                 geometry->nodes->GetCoord(iPoint), Coord_Reflected);
//        visc_numerics->SetCoord(geometry->nodes->GetCoord(), Coord_Reflected);

//        /*--- Primitive variables, and gradient ---*/
//        visc_numerics->SetPrimitive(V_domain, V_inlet);
//        visc_numerics->SetPrimVarGradient(nodes->GetGradient_Primitive(), nodes->GetGradient_Primitive());

//        /*--- Laminar viscosity ---*/
//        visc_numerics->SetLaminarViscosity(nodes->GetLaminarViscosity(), nodes->GetLaminarViscosity());

//        /*--- Compute and update residual ---*/
//        visc_numerics->ComputeResidual(Residual, Jacobian_i, Jacobian_j, config);
//        LinSysRes.SubtractBlock(iPoint, Residual);

//        /*--- Jacobian contribution for implicit integration ---*/
//        if (implicit)
//          Jacobian.SubtractBlock(iPoint, iPoint, Jacobian_i);
//      }
    }
  }

  /*--- Free locally allocated memory ---*/
  delete [] U_domain;
  delete [] U_inlet;
  delete [] V_domain;
  delete [] V_inlet;
  delete [] Normal;
  delete [] Spec_Density;
}

void CNEMOEulerSolver::BC_Outlet(CGeometry *geometry, CSolver **solver_container,
                                 CNumerics *conv_numerics, CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {
  SU2_ZONE_SCOPED

  if (!bcHostOnly) RecordBoundaryNumerics(geometry, config, val_marker, 2); /*--- BCKind::OUTLET ---*/
#ifdef HAVE_KOKKOS
  if (!bcHostOnly && Device_BCResidual(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker,
                                       static_cast<int>(CNEMOKokkosData::BCKind::OUTLET)))
    return;
  Device_BeginHostStage(geometry);
#endif

  su2double UnitNormal[MAXNDIM] = {0.0};
  vector<su2double> rhos;
  rhos.resize(nSpecies,0.0);

  string Marker_Tag = config->GetMarker_All_TagBound(val_marker);
  const bool dynamic_grid = config->GetGrid_Movement();
  const bool gravity = config->GetGravityForce();
  const bool implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);

  su2double *U_domain;
  auto *U_outlet = new su2double[nVar];
  su2double *V_domain, *V_outlet;
  auto *Normal   = new su2double[nDim];
  auto *Ys       = new su2double[nSpecies];

  const unsigned short T_INDEX       = nodes->GetTIndex();
  const unsigned short TVE_INDEX     = nodes->GetTveIndex();
  const unsigned short VEL_INDEX     = nodes->GetVelIndex();
  const unsigned short P_INDEX       = nodes->GetPIndex();
  const unsigned short RHO_INDEX     = nodes->GetRhoIndex();
  const unsigned short H_INDEX       = nodes->GetHIndex();
  const unsigned short A_INDEX       = nodes->GetAIndex();
  const unsigned short RHOCVTR_INDEX = nodes->GetRhoCvtrIndex();
  const unsigned short RHOCVVE_INDEX = nodes->GetRhoCvveIndex();

  /*--- Loop over all the vertices on this boundary marker ---*/
  for (auto iVertex = 0ul; iVertex < geometry->nVertex[val_marker]; iVertex++) {
    const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

    /*--- Allocate the value at the outlet ---*/
    V_outlet = GetCharacPrimVar(val_marker, iVertex);

    /*--- Check if the node belongs to the domain (i.e., not a halo node) ---*/
    if (geometry->nodes->GetDomain(iPoint)) {

      /*--- Normal vector for this vertex (negate for outward convention) ---*/
      geometry->vertex[val_marker][iVertex]->GetNormal(Normal);
      for (auto iDim = 0ul; iDim < nDim; iDim++) Normal[iDim] = -Normal[iDim];
      conv_numerics->SetNormal(Normal);

      const su2double Area = GeometryToolbox::Norm(nDim, Normal);

      for (auto iDim = 0ul; iDim < nDim; iDim++)
        UnitNormal[iDim] = Normal[iDim]/Area;

      /*--- Current solution at this boundary node ---*/
      U_domain = nodes->GetSolution(iPoint);
      V_domain = nodes->GetPrimitive(iPoint);

      /*--- Initialize solution at outlet ---*/
      for (auto iVar = 0ul; iVar < nVar; iVar++)     U_outlet[iVar] = 0.0;
      for (auto iVar = 0ul; iVar < nPrimVar; iVar++) V_outlet[iVar] = 0.0;

      /*--- Build the fictitious intlet state based on characteristics ---*/

      /*--- Compute Gamma using domain state ---*/
      Gamma = nodes->GetGamma(iPoint);
      Gamma_Minus_One = Gamma - 1.0;

      /*--- Retrieve the specified back pressure for this outlet. ---*/
      su2double P_Exit;
      if (gravity) P_Exit = config->GetOutlet_Pressure(Marker_Tag) - geometry->nodes->GetCoord(iPoint, nDim-1)*STANDARD_GRAVITY;
      else         P_Exit = config->GetOutlet_Pressure(Marker_Tag);

      /*--- Non-dim. the inputs if necessary. ---*/
      P_Exit = P_Exit/config->GetPressure_Ref();

      /*--- Check whether the flow is supersonic at the exit. The type
       of boundary update depends on this. ---*/
      su2double Density = V_domain[RHO_INDEX];
      su2double Velocity2 = 0.0;
      su2double Vn = 0.0;
      su2double Velocity[MAXNDIM] = {0.0};
      for (auto iDim = 0ul; iDim < nDim; iDim++) {
        Velocity[iDim] = V_domain[VEL_INDEX+iDim];
        Velocity2 += Velocity[iDim]*Velocity[iDim];
        Vn += Velocity[iDim]*UnitNormal[iDim];
      }
      su2double Temperature = V_domain[T_INDEX];
      su2double Tve         = V_domain[TVE_INDEX];
      su2double Pressure    = V_domain[P_INDEX];
      su2double SoundSpeed  = V_domain[A_INDEX];
      su2double Mach_Exit   = sqrt(Velocity2)/SoundSpeed;

      /*--- Compute Species Concentrations ---*/
      for (auto iSpecies = 0ul; iSpecies<nSpecies;iSpecies++){
        Ys[iSpecies] = V_domain[iSpecies]/Density;
      }

      /*--- Recompute boundary state depending Mach number ---*/
      if (Mach_Exit >= 1.0) {

        /*--- Supersonic exit flow: there are no incoming characteristics,
         so no boundary condition is necessary. Set outlet state to current
         state so that upwinding handles the direction of propagation. ---*/
        for (auto iVar = 0ul; iVar < nVar; iVar++)     U_outlet[iVar] = U_domain[iVar];
        for (auto iVar = 0ul; iVar < nPrimVar; iVar++) V_outlet[iVar] = V_domain[iVar];

      } else {

        /*--- Subsonic exit flow: there is one incoming characteristic,
         therefore one variable can be specified (back pressure) and is used
         to update the conservative variables. Compute the entropy and the
         acoustic Riemann variable. These invariants, as well as the
         tangential velocity components, are extrapolated. The Temperatures
         (T and Tve) and species concentraition are also assumed to be extrapolated.
         ---*/

        const su2double Entropy = Pressure*pow(1.0/Density,Gamma);
        const su2double Riemann = Vn + 2.0*SoundSpeed/Gamma_Minus_One;

        /*--- Compute the new fictious state at the outlet ---*/
        //     U: [rho1, ..., rhoNs, rhou, rhov, rhow, rhoe, rhoeve]^T
        //     V: [rho1, ..., rhoNs, T, Tve, u, v, w, P, rho, h, a, rhoCvtr, rhoCvve]^T
        Density    = pow(P_Exit/Entropy,1.0/Gamma);
        Pressure   = P_Exit;
        SoundSpeed = sqrt(Gamma*P_Exit/Density);
        const su2double Vn_Exit    = Riemann - 2.0*SoundSpeed/Gamma_Minus_One;
        Velocity2  = 0.0;
        for (auto iDim = 0ul; iDim < nDim; iDim++) {
          Velocity[iDim] = Velocity[iDim] + (Vn_Exit-Vn)*UnitNormal[iDim];
          Velocity2 += Velocity[iDim]*Velocity[iDim];
        }

        /*--- Primitive variables, using the derived quantities ---*/
        for (auto iSpecies = 0ul; iSpecies < nSpecies; iSpecies ++) {
          V_outlet[iSpecies] = Ys[iSpecies]*Density;
          rhos[iSpecies] = V_outlet[iSpecies];
        }

        V_outlet[T_INDEX] = V_domain[T_INDEX];
        V_outlet[TVE_INDEX] = V_domain[TVE_INDEX];

        for (auto iDim = 0ul; iDim < nDim; iDim++){
          V_outlet[VEL_INDEX+iDim] = Velocity[iDim];
        }

        V_outlet[P_INDEX]     = Pressure;
        V_outlet[RHO_INDEX]   = Density;
        V_outlet[A_INDEX]     = SoundSpeed;

        /*--- Set mixture state and compute quantities ---*/
        FluidModel->SetTDStateRhosTTv(rhos, Temperature, Tve);
        V_outlet[RHOCVTR_INDEX] = FluidModel->ComputerhoCvtr();
        V_outlet[RHOCVVE_INDEX] = FluidModel->ComputerhoCvve();

        const auto& energies = FluidModel->ComputeMixtureEnergies();

        /*--- Conservative variables, using the derived quantities ---*/
        for (auto iSpecies = 0ul; iSpecies < nSpecies; iSpecies ++){
          U_outlet[iSpecies] = V_outlet[iSpecies];
        }

        for (auto iDim = 0ul; iDim < nDim; iDim++)
          U_outlet[nSpecies+iDim] = Velocity[iDim]*Density;

        U_outlet[nVar-2] = (energies[0] + 0.5*Velocity2) * Density;
        U_outlet[nVar-1] = energies[1] * Density;

      }

      /*--- Setting Last remaining variables ---*/
      V_outlet[H_INDEX]= (U_outlet[nVar-2]+Pressure)/Density;

      /*--- Set various quantities in the solver class ---*/
      conv_numerics->SetConservative(U_domain, U_outlet);
      conv_numerics->SetPrimitive(V_domain,V_outlet);

      if (dynamic_grid)
        conv_numerics->SetGridVel(geometry->nodes->GetGridVel(iPoint), geometry->nodes->GetGridVel(iPoint));

      /*--- Passing supplementary information to CNumerics ---*/
      conv_numerics->SetdPdU  (nodes->GetdPdU(iPoint),   node_infty->GetdPdU(0));
      conv_numerics->SetdTdU  (nodes->GetdTdU(iPoint),   node_infty->GetdTdU(0));
      conv_numerics->SetdTvedU(nodes->GetdTvedU(iPoint), node_infty->GetdTvedU(0));
      conv_numerics->SetEve   (nodes->GetEve(iPoint),    node_infty->GetEve(0));
      conv_numerics->SetCvve  (nodes->GetCvve(iPoint),   node_infty->GetCvve(0));
      conv_numerics->SetGamma (nodes->GetGamma(iPoint),  node_infty->GetGamma(0));
      if(config->GetKind_Upwind_Flow() == UPWIND::AUSMPLUSM)
        conv_numerics->SetSensor (nodes->GetSensor(iPoint), nodes->GetSensor(iPoint));

      /*--- Compute the residual using an upwind scheme ---*/
      auto residual = conv_numerics->ComputeResidual(config);
      LinSysRes.AddBlock(iPoint, residual);

      /*--- Jacobian contribution for implicit integration ---*/
      if (implicit)
        Jacobian.AddBlock2Diag(iPoint, residual.jacobian_i);

      /*--- Viscous contribution ---*/
//      if (viscous) {

//        /*--- Set the normal vector and the coordinates ---*/
//        visc_numerics->SetNormal(Normal);
//        su2double Coord_Reflected[MAXNDIM];
//        GeometryToolbox::PointPointReflect(nDim, geometry->nodes->GetCoord(Point_Normal),
//                                                 geometry->nodes->GetCoord(iPoint), Coord_Reflected);
//        visc_numerics->SetCoord(geometry->nodes->GetCoord(), Coord_Reflected);

//        /*--- Primitive variables, and gradient ---*/
//        visc_numerics->SetPrimitive(V_domain, V_outlet);
//        visc_numerics->SetPrimVarGradient(nodes->GetGradient_Primitive(), nodes->GetGradient_Primitive());

//        /*--- Conservative variables, and gradient ---*/
//        visc_numerics->SetConservative(U_domain, U_outlet);

//        /*--- Pass supplementary information to CNumerics ---*/
//        visc_numerics->SetdPdU(nodes->GetdPdU(), node_infty->GetdPdU());
//        visc_numerics->SetdTdU(nodes->GetdTdU(), node_infty->GetdTdU());
//        visc_numerics->SetdTvedU(nodes->GetdTvedU(), node_infty->GetdTvedU());

//        /*--- Species diffusion coefficients ---*/
//        visc_numerics->SetDiffusionCoeff(nodes->GetDiffusionCoeff(),
//                                         node_infty->GetDiffusionCoeff() );

//        /*--- Laminar viscosity ---*/
//        visc_numerics->SetLaminarViscosity(nodes->GetLaminarViscosity(),
//                                           node_infty->GetLaminarViscosity() );

//        /*--- Thermal conductivity ---*/
//        visc_numerics->SetThermalConductivity(nodes->GetThermalConductivity(),
//                                              node_infty->GetThermalConductivity());

//        /*--- Vib-el. thermal conductivity ---*/
//        visc_numerics->SetThermalConductivity_ve(nodes->GetThermalConductivity_ve(),
//                                                 node_infty->GetThermalConductivity_ve() );

//        /*--- Laminar viscosity ---*/
//        visc_numerics->SetLaminarViscosity(nodes->GetLaminarViscosity(), nodes->GetLaminarViscosity());

//        /*--- Compute and update residual ---*/
//        visc_numerics->ComputeResidual(Residual, Jacobian_i, Jacobian_j, config);
//        LinSysRes.SubtractBlock(iPoint, Residual);

//        /*--- Jacobian contribution for implicit integration ---*/
//        if (implicit)
//          Jacobian.SubtractBlock2Diag(iPoint, residual.jacobian_i);
//      }
    }
  }

  /*--- Free locally allocated memory ---*/
  delete [] U_outlet;
  delete [] Normal;
  delete [] Ys;

}

void CNEMOEulerSolver::BC_Supersonic_Inlet(
    CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics,
    CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {
  SU2_ZONE_SCOPED

#ifdef HAVE_KOKKOS
  if (!bcHostOnly && Device_BCResidual(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker,
                                       static_cast<int>(CNEMOKokkosData::BCKind::SUPERSONIC_INLET)))
    return;
  Device_BeginHostStage(geometry);
#endif

  /*--- Supersonic inlet flow: there are no outgoing characteristics,
   so all flow variables can be imposed at the inlet.
   First, retrieve the specified values for the primitive variables. ---*/

  const string Marker_Tag = config->GetMarker_All_TagBound(val_marker);

  const su2double Temperature = config->GetInlet_Temperature(Marker_Tag);
  const su2double Pressure = config->GetInlet_Pressure(Marker_Tag);
  const su2double *Velocity = config->GetInlet_Velocity(Marker_Tag);
  const su2double *Mass_Frac = config->GetInlet_MassFrac();
  su2double Temperature_ve = config->GetInlet_Temperature_ve();

  /*--- If no Tve value specified (left as 0 K default), set to Ttr value ---*/
  if (Temperature_ve == 0.0) {
    Temperature_ve = Temperature;
  }

  /*--- Set mixture state ---*/
  FluidModel->SetTDStatePTTv(Pressure, Mass_Frac, Temperature, Temperature_ve);

  /*--- Compute Ma vector for flow direction ---*/
  const su2double soundspeed = FluidModel->ComputeSoundSpeed();

  su2double Mvec[MAXNDIM] = {0.0};

  for (auto iDim = 0ul; iDim < nDim; iDim++)
    Mvec[iDim] = Velocity[iDim] / soundspeed;

  /*--- Allocate inlet node to compute gradients for numerics ---*/
  CNEMOEulerVariable node_inlet(Pressure, Mass_Frac, Mvec, Temperature,
                                Temperature_ve, 1, nDim, nVar, nPrimVar,
                                nPrimVarGrad, config, FluidModel);
  node_inlet.SetPrimVar(0, FluidModel);

  su2double Normal[MAXNDIM] = {0.0};

  /*--- Loop over all the vertices on this boundary marker ---*/
  for (auto iVertex = 0ul; iVertex < geometry->nVertex[val_marker];
       iVertex++) {
    const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

    /*--- Check if the node belongs to the domain (i.e, not a halo node) ---*/
    if (!geometry->nodes->GetDomain(iPoint))
      continue;

    /*--- Index of the closest interior node ---*/
    const auto Point_Normal =
        geometry->vertex[val_marker][iVertex]->GetNormal_Neighbor();

    /*--- Normal vector for this vertex (negate for outward convention) ---*/
    geometry->vertex[val_marker][iVertex]->GetNormal(Normal);
    for (auto iDim = 0ul; iDim < nDim; iDim++)
      Normal[iDim] = -Normal[iDim];

    /*--- Set various quantities in the solver class ---*/
    conv_numerics->SetNormal(Normal);
    conv_numerics->SetConservative(nodes->GetSolution(iPoint),
                                   node_inlet.GetSolution(0));
    conv_numerics->SetPrimitive(nodes->GetPrimitive(iPoint),
                                node_inlet.GetPrimitive(0));

    /*--- Pass supplementary info to CNumerics ---*/
    conv_numerics->SetdPdU(nodes->GetdPdU(iPoint), node_inlet.GetdPdU(0));
    conv_numerics->SetdTdU(nodes->GetdTdU(iPoint), node_inlet.GetdTdU(0));
    conv_numerics->SetdTvedU(nodes->GetdTvedU(iPoint), node_inlet.GetdTvedU(0));
    conv_numerics->SetEve(nodes->GetEve(iPoint), node_inlet.GetEve(0));
    conv_numerics->SetCvve(nodes->GetCvve(iPoint), node_inlet.GetCvve(0));

    const bool dynamic_grid = config->GetGrid_Movement();
    if (dynamic_grid)
      conv_numerics->SetGridVel(geometry->nodes->GetGridVel(iPoint),
                                geometry->nodes->GetGridVel(iPoint));

    /*--- Compute the residual using an upwind scheme ---*/
    auto residual = conv_numerics->ComputeResidual(config);
    LinSysRes.AddBlock(iPoint, residual);

    /*--- Jacobian contribution for implicit integration ---*/
    const bool implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);
    if (implicit)
      Jacobian.AddBlock2Diag(iPoint, residual.jacobian_i);

    /*--- Viscous contribution ---*/
    const bool viscous = config->GetViscous();
    if (viscous) {

      /*--- Set the normal vector and the coordinates ---*/
      su2double Coord_Reflected[MAXNDIM];
      GeometryToolbox::PointPointReflect(
          nDim, geometry->nodes->GetCoord(Point_Normal),
          geometry->nodes->GetCoord(iPoint), Coord_Reflected);
      visc_numerics->SetCoord(geometry->nodes->GetCoord(iPoint),
                              Coord_Reflected);
      visc_numerics->SetNormal(Normal);

      /*--- Primitive variables, and gradient ---*/
      visc_numerics->SetConservative(nodes->GetSolution(iPoint),
                                     node_inlet.GetSolution(0));
      visc_numerics->SetPrimitive(nodes->GetPrimitive(iPoint),
                                  node_inlet.GetPrimitive(0));
      visc_numerics->SetPrimVarGradient(nodes->GetGradient_Primitive(iPoint),
                                        node_inlet.GetGradient_Primitive(0));

      /*--- Pass supplementary information to CNumerics ---*/
      visc_numerics->SetdPdU(nodes->GetdPdU(iPoint), node_inlet.GetdPdU(0));
      visc_numerics->SetdTdU(nodes->GetdTdU(iPoint), node_inlet.GetdTdU(0));
      visc_numerics->SetdTvedU(nodes->GetdTvedU(iPoint),
                               node_inlet.GetdTvedU(0));
      visc_numerics->SetEve(nodes->GetEve(iPoint), node_inlet.GetEve(0));
      visc_numerics->SetCvve(nodes->GetCvve(iPoint), node_inlet.GetCvve(0));

      /*--- Species diffusion coefficients ---*/
      visc_numerics->SetDiffusionCoeff(nodes->GetDiffusionCoeff(iPoint),
                                       nodes->GetDiffusionCoeff(iPoint));
      /*--- Laminar viscosity ---*/
      visc_numerics->SetLaminarViscosity(nodes->GetLaminarViscosity(iPoint),
                                         nodes->GetLaminarViscosity(iPoint));
      /*--- Eddy viscosity ---*/
      visc_numerics->SetEddyViscosity(nodes->GetEddyViscosity(iPoint),
                                      nodes->GetEddyViscosity(iPoint));
      /*--- Thermal conductivity ---*/
      visc_numerics->SetThermalConductivity(
          nodes->GetThermalConductivity(iPoint),
          nodes->GetThermalConductivity(iPoint));
      /*--- Vib-el. thermal conductivity ---*/
      visc_numerics->SetThermalConductivity_ve(
          nodes->GetThermalConductivity_ve(iPoint),
          nodes->GetThermalConductivity_ve(iPoint));

      /*--- Compute and update residual ---*/
      auto residual = visc_numerics->ComputeResidual(config);
      LinSysRes.SubtractBlock(iPoint, residual);

      /*--- Jacobian contribution for implicit integration ---*/
      if (implicit)
        Jacobian.SubtractBlock2Diag(iPoint, residual.jacobian_i);
    }
  }
}

void CNEMOEulerSolver::BC_Supersonic_Outlet(
    CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics,
    CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {
  SU2_ZONE_SCOPED

  if (!bcHostOnly) RecordBoundaryNumerics(geometry, config, val_marker, 4); /*--- BCKind::SUPERSONIC_OUTLET ---*/
#ifdef HAVE_KOKKOS
  if (!bcHostOnly && Device_BCResidual(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker,
                                       static_cast<int>(CNEMOKokkosData::BCKind::SUPERSONIC_OUTLET)))
    return;
  Device_BeginHostStage(geometry);
#endif

  /*--- Supersonic outlet flow: there are no ingoing characteristics,
   so all flow variables can should be interpolated from the domain. ---*/

  su2double Normal[MAXNDIM] = {0.0};

  const string Marker_Tag = config->GetMarker_All_TagBound(val_marker);

  /*--- Loop over all the vertices on this boundary marker ---*/
  for (auto iVertex = 0ul; iVertex < geometry->nVertex[val_marker];
       iVertex++) {

    const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

    /*--- Check if the node belongs to the domain (i.e, not a halo node) ---*/
    if (!geometry->nodes->GetDomain(iPoint))
      continue;

    /*--- Normal vector for this vertex (negate for outward convention) ---*/
    geometry->vertex[val_marker][iVertex]->GetNormal(Normal);
    for (auto iDim = 0ul; iDim < nDim; iDim++)
      Normal[iDim] = -Normal[iDim];

    /*--- Set various quantities in the solver class, outlet properties are
     * equal to domain properties ---*/
    conv_numerics->SetNormal(Normal);
    conv_numerics->SetPrimitive(nodes->GetPrimitive(iPoint),
                                nodes->GetPrimitive(iPoint));
    conv_numerics->SetConservative(nodes->GetSolution(iPoint),
                                   nodes->GetSolution(iPoint));

    /*--- Pass supplementary information to CNumerics ---*/
    conv_numerics->SetdPdU(nodes->GetdPdU(iPoint), nodes->GetdPdU(iPoint));
    conv_numerics->SetdTdU(nodes->GetdTdU(iPoint), nodes->GetdTdU(iPoint));
    conv_numerics->SetdTvedU(nodes->GetdTvedU(iPoint),
                             nodes->GetdTvedU(iPoint));
    conv_numerics->SetEve(nodes->GetEve(iPoint), nodes->GetEve(iPoint));
    conv_numerics->SetCvve(nodes->GetCvve(iPoint), nodes->GetCvve(iPoint));
    conv_numerics->SetGamma(nodes->GetGamma(iPoint), nodes->GetGamma(iPoint));

    const bool dynamic_grid = config->GetGrid_Movement();
    if (dynamic_grid)
      conv_numerics->SetGridVel(geometry->nodes->GetGridVel(iPoint),
                                geometry->nodes->GetGridVel(iPoint));

    /*--- Compute the residual using an upwind scheme ---*/
    auto residual = conv_numerics->ComputeResidual(config);
    LinSysRes.AddBlock(iPoint, residual);

    /*--- Jacobian contribution for implicit integration ---*/
    const bool implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);
    if (implicit)
      Jacobian.AddBlock2Diag(iPoint, residual.jacobian_i);
  }
}

void CNEMOEulerSolver::SetPressureDiffusionSensor(CGeometry *geometry, CConfig *config) {
  SU2_ZONE_SCOPED

  const auto P_INDEX = nodes->GetPIndex();

  for (auto iPoint = 0ul; iPoint < nPointDomain; iPoint++) {

    su2double Sensor = 1.0;

    for (auto jPoint : geometry->nodes->GetPoints(iPoint)) {
      const su2double P_k = nodes->GetPrimitive(jPoint, P_INDEX) / nodes->GetPrimitive(iPoint, P_INDEX);
      Sensor = min(Sensor, min(P_k, 1.0/P_k));
    }

    nodes->SetSensor(iPoint,Sensor);
  }

  /*--- MPI parallelization ---*/

  InitiateComms(geometry, config, MPI_QUANTITIES::SENSOR);
  CompleteComms(geometry, config, MPI_QUANTITIES::SENSOR);

}

namespace {
/*! \brief SU2_KOKKOS_NEMO_MIXED=1: device stages may upload a non-zero host accumulator (after a host stage). */
bool MixedAllowed() {
  static const bool on = [] {
    const char* e = std::getenv("SU2_KOKKOS_NEMO_MIXED");
    return e != nullptr && std::atoi(e) != 0;
  }();
  return on;
}
/*! \brief SU2_KOKKOS_NEMO_EAGER=1: download the accumulator after every device stage (debug / test of that path). */
bool EagerForced() {
  static const bool on = [] {
    const char* e = std::getenv("SU2_KOKKOS_NEMO_EAGER");
    return e != nullptr && std::atoi(e) != 0;
  }();
  return on;
}
#ifdef HAVE_KOKKOS
/*! \brief Base of the CSysMatrix values (row 0 starts at non-zero 0; its diagonal block is at dia_ptr[0]). */
su2mixedfloat* JacobianValues(CGeometry* geometry, CSysMatrix<su2mixedfloat>& jac, unsigned short nVar) {
  const auto& pattern = geometry->GetSparsePattern(ConnectivityType::FiniteVolume, 0);
  return jac.GetBlock(0, 0) - pattern.diagPtr()[0] * nVar * nVar;
}
const char* BCKindName(int kind) {
  using K = CNEMOKokkosData::BCKind;
  switch (static_cast<K>(kind)) {
    case K::SYMMETRY: return "SYMMETRY_PLANE/EULER_WALL";
    case K::FAR_FIELD: return "FAR_FIELD";
    case K::OUTLET: return "OUTLET_FLOW";
    case K::SUPERSONIC_INLET: return "SUPERSONIC_INLET";
    case K::SUPERSONIC_OUTLET: return "SUPERSONIC_OUTLET";
    case K::ISOTHERMAL: return "ISOTHERMAL";
    case K::HEAT_FLUX: return "HEAT_FLUX";
    case K::SMOLUCHOWSKI_MAXWELL: return "SMOLUCHOWSKI_MAXWELL";
  }
  return "?";
}
#endif
}  // namespace

/*============================ Device (Kokkos) Steps 3-4: dispatch ============================*/

namespace {
/*! \brief Debug: SU2_KOKKOS_NEMO_HOST=grad,lim,dt forces individual stages back to the host. */
bool ForcedHost(const char* stage) {
#ifdef HAVE_KOKKOS
  return CNEMOKokkosData::HostForced(stage);
#else
  (void)stage;
  return false;
#endif
}
}  // namespace

std::string CNEMOEulerSolver::Device_GradientUnsupported(const CConfig *config, bool reconstruction,
                                                         bool greenGauss) const {
#ifdef HAVE_KOKKOS
  if (!kokkosData) return "no device data";
  if (!kokkosData->geo.domain_first) return "domain points not first";
  if (config->GetDiscrete_Adjoint()) return "discrete adjoint";
  if (!greenGauss && config->GetnMarker_Periodic() > 0) return "periodic least squares";
  if (ForcedHost("grad")) return "SU2_KOKKOS_NEMO_HOST";
  const auto& grad = reconstruction ? nodes->GetGradient_Reconstruction() : nodes->GetGradient_Primitive();
  if (grad.size() == 0) return "gradient not allocated";
  return "";
#else
  return "no Kokkos";
#endif
}

std::string CNEMOEulerSolver::Device_LimiterUnsupported(const CConfig *config) const {
#ifdef HAVE_KOKKOS
  if (!kokkosData) return "no device data";
  if (!kokkosData->geo.domain_first) return "domain points not first";
  if (config->GetDiscrete_Adjoint()) return "discrete adjoint";
  if (config->GetnMarker_Periodic() > 0) return "periodic";
  if (ForcedHost("lim")) return "SU2_KOKKOS_NEMO_HOST";
  const auto kind = config->GetKind_SlopeLimit_Flow();
  if (kind != LIMITER::BARTH_JESPERSEN && kind != LIMITER::VENKATAKRISHNAN && kind != LIMITER::VENKATAKRISHNAN_WANG)
    return "limiter kind not ported";
  return "";
#else
  return "no Kokkos";
#endif
}

std::string CNEMOEulerSolver::Device_TimeStepUnsupported(const CConfig *config) const {
#ifdef HAVE_KOKKOS
  if (!kokkosData) return "no device data";
  if (!kokkosData->geo.domain_first) return "domain points not first";
  if (dynamic_grid) { auto whyDyn = Device_DynamicGridUnsupported(config); if (!whyDyn.empty()) return whyDyn; }
  if (ForcedHost("dt")) return "SU2_KOKKOS_NEMO_HOST";
  if (config->GetViscous() && dynamic_cast<const CNEMONSVariable*>(nodes) == nullptr) return "viscous without NS variables";
  return "";
#else
  return "no Kokkos";
#endif
}

void CNEMOEulerSolver::PrintDeviceStages(const CConfig *config) const {
#ifdef HAVE_KOKKOS
  const bool center = config->GetKind_ConvNumScheme_Flow() == SPACE_CENTERED;
  const bool muscl = config->GetMUSCL_Flow();
  const bool viscous = config->GetViscous();
  auto gname = [](unsigned short m) {
    return m == GREEN_GAUSS ? "GREEN_GAUSS" : m == WEIGHTED_LEAST_SQUARES ? "WEIGHTED_LEAST_SQUARES"
         : m == LEAST_SQUARES ? "LEAST_SQUARES" : "NONE";
  };
  auto where = [](const std::string& why) { return why.empty() ? std::string("device") : "host (" + why + ")"; };

  cout << "[KOKKOS_NEMO] stage Cons2Prim (SetPrimVar + non-physical reduce): "
       << where(CNEMOKokkosData::HostForced("c2p") ? "SU2_KOKKOS_NEMO_HOST" : "") << endl;

  /*--- Mirror the Preprocessing logic of CNEMOEulerSolver / CNEMONSSolver. ---*/
  const auto gm = config->GetKind_Gradient_Method();
  if (viscous) {
    if (config->GetReconstructionGradientRequired() && muscl && !center) {
      const auto rm = config->GetKind_Gradient_Method_Recon();
      cout << "[KOKKOS_NEMO] stage gradient reconstruction (" << gname(rm) << (rm == GREEN_GAUSS ? ", Y_s" : "")
           << "): " << where(Device_GradientUnsupported(config, true, rm == GREEN_GAUSS)) << endl;
    }
    if (gm == GREEN_GAUSS || gm == WEIGHTED_LEAST_SQUARES)
      cout << "[KOKKOS_NEMO] stage gradient primitive (" << gname(gm) << (gm == GREEN_GAUSS ? ", Y_s" : "")
           << "): " << where(Device_GradientUnsupported(config, false, gm == GREEN_GAUSS)) << endl;
  } else if (muscl && !center && (gm == GREEN_GAUSS || gm == WEIGHTED_LEAST_SQUARES)) {
    cout << "[KOKKOS_NEMO] stage gradient (" << gname(gm) << ", into Gradient_Reconstruction): "
         << where(Device_GradientUnsupported(config, true, gm == GREEN_GAUSS)) << endl;
  } else {
    cout << "[KOKKOS_NEMO] stage gradient: not used (first order / centered)" << endl;
  }
  const auto lim = config->GetKind_SlopeLimit_Flow();
  if (muscl && !center && lim != LIMITER::NONE && lim != LIMITER::VAN_ALBADA_EDGE)
    cout << "[KOKKOS_NEMO] stage limiter (" << (lim == LIMITER::VENKATAKRISHNAN ? "VENKATAKRISHNAN"
            : lim == LIMITER::BARTH_JESPERSEN ? "BARTH_JESPERSEN" : lim == LIMITER::VENKATAKRISHNAN_WANG ? "VENKATAKRISHNAN_WANG"
            : "other") << "): " << where(Device_LimiterUnsupported(config)) << endl;
  else
    cout << "[KOKKOS_NEMO] stage limiter: not used" << endl;
  cout << "[KOKKOS_NEMO] stage time step (spectral radii + local dt + min/max reduce): "
       << where(Device_TimeStepUnsupported(config)) << endl;
  cout << "[KOKKOS_NEMO] stage max eigenvalue (centered only): "
       << (center ? where(Device_TimeStepUnsupported(config)) : std::string("not used")) << endl;
  if (dynamic_grid)
    cout << "[KOKKOS_NEMO] stage dynamic grid (grid-velocity dt/eigenvalue terms + GCL, metric re-upload per time step): "
         << where(Device_DynamicGridUnsupported(config)) << endl;
  if (!center) {
    const auto up = config->GetKind_Upwind_Flow();
    const char* uname = up == UPWIND::AUSM ? "AUSM" : up == UPWIND::AUSMPLUSM ? "AUSM+M"
                      : up == UPWIND::AUSMPLUSUP2 ? "AUSM+-up2" : up == UPWIND::ROE ? "ROE"
                      : up == UPWIND::MSW ? "MSW" : "other";
    cout << "[KOKKOS_NEMO] stage upwind (" << uname << (muscl ? " + MUSCL" : "") << ", residual"
         << (config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT ? " + Jacobian" : "") << "): "
         << where(Device_UpwindUnsupported(config, MESH_0)) << endl;
  }
  /*--- Step 7: residual pipeline in the CIntegration::Space_Integration order, predicting where each stage runs
   *    (the dispatch decides the same way at run time: a device stage needs the accumulator on the device, or a
   *    zero host accumulator, or permission to upload it). ---*/
  {
    using K = CNEMOKokkosData;
    const bool mixed = MixedAllowed() || K::StageVerifyEnabled();
    bool acc = false, hostZero = true;
    auto stage = [&](const std::string& name, const std::string& why) {
      std::string where;
      if (why.empty() && (acc || hostZero || mixed)) {
        where = "device";
        acc = deferResidualSync;
        hostZero = false;
        if (!deferResidualSync) where += " (eager download after the stage)";
      } else {
        where = "host (" + (why.empty() ? std::string("accumulator on the host after a host stage; "
                                                       "SU2_KOKKOS_NEMO_MIXED=1 uploads it") : why) + ")";
        acc = false;
        hostZero = false;
      }
      cout << "[KOKKOS_NEMO] stage " << name << ": " << where << endl;
    };
    auto hostStage = [&](const std::string& name) {
      cout << "[KOKKOS_NEMO] stage " << name << ": host" << endl;
      acc = false;
      hostZero = false;
    };
    if (!center) {
      /*--- Upwind line printed above; replay its outcome here. ---*/
      if (Device_UpwindUnsupported(config, MESH_0).empty()) {
        acc = deferResidualSync;
        hostZero = false;
      } else {
        acc = hostZero = false;
      }
    } else {
      stage("centered residual (LAX-FRIEDRICH)", Device_CenteredUnsupported(config));
    }
    if (viscous) stage("viscous residual (CAvgGradCorrected_NEMO" + std::string(
                             config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT ? ", residual + Jacobian)" : ")"),
                       Device_ViscousUnsupported(config, MESH_0));
    stage("source (finite-rate chemistry + V-T relaxation, residual" +
              std::string(config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT ? " + Jacobian" : "") + ")",
          Device_SourceUnsupported(config));
    if (config->GetTime_Marching() == TIME_MARCHING::DT_STEPPING_1ST ||
        config->GetTime_Marching() == TIME_MARCHING::DT_STEPPING_2ND)
      stage(std::string("dual-time source (") +
                (config->GetTime_Marching() == TIME_MARCHING::DT_STEPPING_2ND ? "2nd" : "1st") + " order)",
            Device_DualTimeUnsupported(config));
    auto bcKind = [&](unsigned short kindBC) -> int {
      switch (kindBC) {
        case FAR_FIELD: return static_cast<int>(K::BCKind::FAR_FIELD);
        case OUTLET_FLOW: return static_cast<int>(K::BCKind::OUTLET);
        case SUPERSONIC_INLET: return static_cast<int>(K::BCKind::SUPERSONIC_INLET);
        case SUPERSONIC_OUTLET: return static_cast<int>(K::BCKind::SUPERSONIC_OUTLET);
        case SYMMETRY_PLANE: case EULER_WALL: return static_cast<int>(K::BCKind::SYMMETRY);
        case ISOTHERMAL: return static_cast<int>(K::BCKind::ISOTHERMAL);
        case HEAT_FLUX: return static_cast<int>(K::BCKind::HEAT_FLUX);
        case SMOLUCHOWSKI_MAXWELL: return static_cast<int>(K::BCKind::SMOLUCHOWSKI_MAXWELL);
        default: return -1;
      }
    };
    auto kname = [](unsigned short kindBC) -> std::string {
      switch (kindBC) {
        case FAR_FIELD: return "FAR_FIELD"; case OUTLET_FLOW: return "OUTLET_FLOW";
        case SUPERSONIC_INLET: return "SUPERSONIC_INLET"; case SUPERSONIC_OUTLET: return "SUPERSONIC_OUTLET";
        case SYMMETRY_PLANE: return "SYMMETRY_PLANE"; case EULER_WALL: return "EULER_WALL";
        case INLET_FLOW: return "INLET_FLOW"; case ISOTHERMAL: return "ISOTHERMAL"; case HEAT_FLUX: return "HEAT_FLUX";
        case SMOLUCHOWSKI_MAXWELL: return "SMOLUCHOWSKI_MAXWELL"; case SEND_RECEIVE: return "SEND_RECEIVE";
        default: return "kind " + std::to_string(kindBC);
      }
    };
    /*--- Weak BCs, strong BCs, then symmetry / Euler walls (CIntegration order). ---*/
    for (int pass = 0; pass < 3; pass++) {
      for (unsigned short iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
        const auto kb = config->GetMarker_All_KindBC(iMarker);
        const bool weak = (kb == FAR_FIELD || kb == OUTLET_FLOW || kb == SUPERSONIC_INLET || kb == SUPERSONIC_OUTLET ||
                           kb == INLET_FLOW);
        const bool strong = (kb == ISOTHERMAL || kb == HEAT_FLUX || kb == SMOLUCHOWSKI_MAXWELL);
        const bool sym = (kb == SYMMETRY_PLANE || kb == EULER_WALL);
        if ((pass == 0 && !weak) || (pass == 1 && !strong) || (pass == 2 && !sym)) continue;
        const std::string name = "bc:" + kname(kb) + " (" + config->GetMarker_All_TagBound(iMarker) + ")";
        const int k = bcKind(kb);
        if (k < 0) hostStage(name + " [not ported]");
        else stage(name + ((kb == ISOTHERMAL || kb == HEAT_FLUX)
                               ? std::string(config->GetCatalytic_Wall(iMarker)
                                                 ? (config->GetSupercatalytic_Wall() ? " [supercatalytic]" : " [catalytic]")
                                                 : " [non-catalytic]")
                               : std::string(kb == FAR_FIELD && viscous ? " [+ viscous]" : "")),
                   Device_BCUnsupported(config, k, iMarker));
      }
    }
    const auto why = Device_DeferralUnsupported(config);
    if (why.empty())
      cout << "[KOKKOS_NEMO] residual sync: deferred, the device residual + Jacobian are downloaded once after the BCs"
              " (FinalizeSpaceIntegration), or before the first host stage" << endl;
    else
      cout << "[KOKKOS_NEMO] residual sync: eager (" << why << "), download after every device stage" << endl;
  }
  if (viscous)
    cout << "[KOKKOS_NEMO] stage NS transport (" << (config->GetKind_TransCoeffModel() == TRANSCOEFFMODEL::WILKE ? "WILKE"
             : config->GetKind_TransCoeffModel() == TRANSCOEFFMODEL::GUPTAYOS ? "GUPTAYOS"
             : config->GetKind_TransCoeffModel() == TRANSCOEFFMODEL::SUTHERLAND ? "SUTHERLAND" : "other")
         << ", in SetPrimVar): " << (kokkosData->DeviceTransport() ? "device" : "host") << endl;
  if (config->GetKind_Upwind_Flow() == UPWIND::AUSMPLUSM && !center)
    cout << "[KOKKOS_NEMO] stage pressure diffusion sensor (AUSM+M): "
         << (deviceResident ? "device" : "host (device only in device-resident mode; A/B with SU2_KOKKOS_STAGE_VERIFY)")
         << endl;
  if (config->GetCFL_Adapt())
    cout << "[KOKKOS_NEMO] stage CFL adaptation (point loop): " << (deviceResident ? "device" : "host") << endl;
  if (config->GetAxisymmetric() && viscous)
    cout << "[KOKKOS_NEMO] stage axisymmetric auxiliary gradients: "
         << where(Device_SourceUnsupported(config)) << endl;
  {
    const auto ti = config->GetKind_TimeIntScheme_Flow();
    cout << "[KOKKOS_NEMO] stage update ("
         << (ti == EULER_IMPLICIT ? "EULER_IMPLICIT" : ti == EULER_EXPLICIT ? "EULER_EXPLICIT"
             : ti == RUNGE_KUTTA_EXPLICIT ? "RUNGE_KUTTA_EXPLICIT" : ti == CLASSICAL_RK4_EXPLICIT ? "CLASSICAL_RK4_EXPLICIT"
             : "other")
         << "): " << (deviceResident ? "device" : "host") << endl;
  }
  cout << "[KOKKOS_NEMO] host: MPI/periodic comms, Velocity2, linear solve"
       << (viscous && !kokkosData->DeviceTransport() ? ", NS transport" : "") << endl;
  cout << "[KOKKOS_NEMO] sync: SetPrimVar HtoD U,U_old,V,secondaries + DtoH all; stages reuse device V, DtoH domain rows"
          " of gradient/limiter/min/max/dt/lambda; dt HtoD LocalCFL (+mu,k,k_ve for NS); upwind reuses device"
          " V/secondaries/gradient/limiter (HtoD halo rows, Non_Physical, sensor); source/BCs reuse device U/V/"
          "secondaries (HtoD node_infty once); residual accumulator DtoH (see residual sync)"
       << endl;
  if (CNEMOKokkosData::StageVerifyEnabled())
    cout << "[KOKKOS_NEMO] SU2_KOKKOS_STAGE_VERIFY=1: gradient/limiter/time step/upwind/source/BC A/B against the host"
            " code" << endl;
#endif
}

void CNEMOEulerSolver::CorrectGradientsSymmetryHost(CGeometry *geometry, const CConfig *config,
                                                    CVectorOfMatrix& gradient) const {
  /*--- Called through a volatile pointer so that it is not inlined here: the program then runs the same
   *    (COMDAT) instantiation as computeGradientsLeastSquares/GreenGauss, i.e. the same machine code. ---*/
  using Fn = void (*)(CGeometry&, const CConfig&, const size_t, const size_t, const int, CVectorOfMatrix&);
  static Fn volatile fn2 = &correctGradientsSymmetry<2, CVectorOfMatrix>;
  static Fn volatile fn3 = &correctGradientsSymmetry<3, CVectorOfMatrix>;
  (nDim == 2 ? fn2 : fn3)(*geometry, *config, 0, nPrimVarGrad, prim_idx.Velocity(), gradient);
}

void CNEMOEulerSolver::CorrectGradientsSymmetryHost(CGeometry *geometry, const CConfig *config,
                                                    CVectorOfMatrix& gradient, size_t varEnd, int idxVel) const {
  using Fn = void (*)(CGeometry&, const CConfig&, const size_t, const size_t, const int, CVectorOfMatrix&);
  static Fn volatile fn2 = &correctGradientsSymmetry<2, CVectorOfMatrix>;
  static Fn volatile fn3 = &correctGradientsSymmetry<3, CVectorOfMatrix>;
  (nDim == 2 ? fn2 : fn3)(*geometry, *config, 0, varEnd, idxVel, gradient);
}

bool CNEMOEulerSolver::Device_PrimitiveGradient(CGeometry *geometry, const CConfig *config, bool reconstruction,
                                                bool greenGauss, bool massFractions) {
#ifdef HAVE_KOKKOS
  if (!kokkosData || !Device_GradientUnsupported(config, reconstruction, greenGauss).empty()) return false;
  SU2_ZONE_SCOPED

  using K = CNEMOKokkosData;
  auto& gradient = reconstruction ? nodes->GetGradient_Reconstruction() : nodes->GetGradient_Primitive();
  const auto comm = reconstruction? MPI_QUANTITIES::PRIMITIVE_GRAD_REC : MPI_QUANTITIES::PRIMITIVE_GRADIENT;

  /*--- Same LS weighting / periodic comm selection as CFVMFlowSolverBase::SetPrimitive_Gradient_LS/GG. ---*/
  K::GradMethod method = K::GradMethod::GREEN_GAUSS;
  if (!greenGauss) {
    const bool weighted = reconstruction ? (config->GetKind_Gradient_Method_Recon() == WEIGHTED_LEAST_SQUARES)
                                         : (config->GetKind_Gradient_Method() == WEIGHTED_LEAST_SQUARES);
    method = weighted ? K::GradMethod::WEIGHTED_LEAST_SQUARES : K::GradMethod::LEAST_SQUARES;
  }

  const bool verify = K::StageVerifyEnabled();
  if (verify) kokkosData->UploadPrimitiveOnly(); /*--- Isolate this stage from the Cons2Prim A/B. ---*/

  const bool deviceSym = K::DeviceSymmetryEnabled();
  kokkosData->ComputeGradient(method, massFractions, reconstruction, prim_idx.Velocity(), deviceSym, gradient.data());

  if (!deviceSym) {
    /*--- Symmetry / Euler-wall correction with the CPU routine itself (boundary-local work, bitwise by
     *    construction), then refresh those rows of the device gradient (read by the device limiter).
     *    Device-resident: only the symmetry rows are downloaded (the correction reads and writes only them). ---*/
    if (Device_ResidentActive()) kokkosData->DownloadSymmetryRows(reconstruction, gradient.data());
    Kokkos::Timer timer;
    CorrectGradientsSymmetryHost(geometry, config, gradient);
    kokkosData->stagePerf[K::ST_GRADIENT].host += timer.seconds();
    kokkosData->UploadSymmetryRows(reconstruction, gradient.data());
  }

  if (verify) {
    const size_t n = size_t(nPointDomain) * nPrimVarGrad * nDim;
    std::vector<su2double> dev(gradient.data(), gradient.data() + n);
    if (greenGauss) SetPrimitive_Gradient_GG(geometry, config, reconstruction); /*--- virtual: NS Y_s override ---*/
    else SetPrimitive_Gradient_LS(geometry, config, reconstruction);
    kokkosData->StageCompare(K::ST_GRADIENT, reconstruction ? "Gradient_Reconstruction" : "Gradient_Primitive",
                             gradient.data(), dev.data(), n, size_t(nPrimVarGrad) * nDim);
    return true;
  }

  Kokkos::Timer timer;
  if (greenGauss) {
    const auto commPer = reconstruction? PERIODIC_PRIM_GG_R : PERIODIC_PRIM_GG;
    for (size_t iPeriodic = 1; iPeriodic <= config->GetnMarker_Periodic() / 2; ++iPeriodic) {
      InitiatePeriodicComms(geometry, config, iPeriodic, commPer);
      CompletePeriodicComms(geometry, config, iPeriodic, commPer);
    }
  }
  if (Device_ResidentActive()) {
    Device_StagedComms(geometry, config, comm, K::G_GRADIENT);
  } else {
    InitiateComms(geometry, config, comm);
    CompleteComms(geometry, config, comm);
  }
  kokkosData->stagePerf[K::ST_GRADIENT].host += timer.seconds();
  return true;
#else
  return false;
#endif
}

bool CNEMOEulerSolver::Device_PrimitiveLimiter(CGeometry *geometry, const CConfig *config) {
#ifdef HAVE_KOKKOS
  if (!kokkosData || !Device_LimiterUnsupported(config).empty()) return false;
  SU2_ZONE_SCOPED

  using K = CNEMOKokkosData;
  const auto kind = config->GetKind_SlopeLimit_Flow();
  const bool verify = K::StageVerifyEnabled();

  /*--- eps2 as CLimiterDetails<kind>::preprocess (Venkatakrishnan-Wang: K, eps2 from the device range). ---*/
  su2double eps = LimiterHelpers<>::epsilon();
  if (kind == LIMITER::VENKATAKRISHNAN) {
    su2double L = config->GetRefElemLength();
    su2double Kc = config->GetVenkat_LimiterCoeff();
    su2double eps1 = fabs(L*Kc);
    eps = max(eps1*eps1*eps1, LimiterHelpers<>::epsilon());
  } else if (kind == LIMITER::VENKATAKRISHNAN_WANG) {
    eps = config->GetVenkat_LimiterCoeff();
  }

  auto& grad = nodes->GetGradient_Reconstruction();
  if (verify || !kokkosData->GradientOnDevice(true)) kokkosData->UploadGradient(true, grad.data());
  if (verify) kokkosData->UploadPrimitiveOnly();

  auto& lim = nodes->GetLimiter_Primitive();
  auto& pmax = nodes->GetSolution_Max();
  auto& pmin = nodes->GetSolution_Min();
  kokkosData->ComputeLimiter(static_cast<int>(kind), config->GetMUSCL_Kappa_Flow(), eps, lim.data(), pmax.data(),
                             pmin.data());

  if (verify) {
    const size_t n = size_t(nPointDomain) * nPrimVarGrad;
    std::vector<su2double> dl(lim.data(), lim.data() + n), dx(pmax.data(), pmax.data() + n),
        dn(pmin.data(), pmin.data() + n);
    SetPrimitive_Limiter(geometry, config);
    kokkosData->StageCompare(K::ST_LIMITER, "Limiter_Primitive", lim.data(), dl.data(), n, nPrimVarGrad);
    kokkosData->StageCompare(K::ST_LIMITER, "Solution_Max", pmax.data(), dx.data(), n, nPrimVarGrad);
    kokkosData->StageCompare(K::ST_LIMITER, "Solution_Min", pmin.data(), dn.data(), n, nPrimVarGrad);
    return true;
  }

  Kokkos::Timer timer;
  if (Device_ResidentActive()) {
    Device_StagedComms(geometry, config, MPI_QUANTITIES::PRIMITIVE_LIMITER, K::G_LIMITER);
  } else {
    InitiateComms(geometry, config, MPI_QUANTITIES::PRIMITIVE_LIMITER);
    CompleteComms(geometry, config, MPI_QUANTITIES::PRIMITIVE_LIMITER);
  }
  kokkosData->stagePerf[K::ST_LIMITER].host += timer.seconds();
  return true;
#else
  return false;
#endif
}


/*============================ Device (Kokkos) Step 5: upwind residual ============================*/

std::string CNEMOEulerSolver::Device_UpwindUnsupported(const CConfig *config, unsigned short iMesh) const {
#ifdef HAVE_KOKKOS
  if (!kokkosData) return "no device data";
  if (!kokkosData->geo.domain_first) return "domain points not first";
  if (config->GetDiscrete_Adjoint()) return "discrete adjoint";
  if (dynamic_grid) { auto whyDyn = Device_DynamicGridUnsupported(config); if (!whyDyn.empty()) return whyDyn; }
  if (ReducerStrategy) return "reducer strategy";
  if (ForcedHost("upwind")) return "SU2_KOKKOS_NEMO_HOST";
  const auto kind = config->GetKind_Upwind_Flow();
  if (kind != UPWIND::AUSM && kind != UPWIND::AUSMPLUSM && kind != UPWIND::ROE && kind != UPWIND::MSW &&
      kind != UPWIND::AUSMPLUSUP2)
    return "upwind scheme not ported";
  /*--- CUpwAUSMPLUSUP2_NEMO shadows M_L / M_R with locals: its ComputeJacobian reads members that are never set. ---*/
  if (kind == UPWIND::AUSMPLUSUP2 && config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT)
    return "implicit AUSM+-up2 (CPU Jacobian reads uninitialised M_L/M_R members)";
  /*--- CPU RoedPdU is a persistent member: ComputedPdU with ionization reads the previous edge's state. ---*/
  if (kind == UPWIND::ROE && config->GetIonization()) return "ROE + ionization (CPU RoedPdU depends on the previous edge)";
  if (kind == UPWIND::ROE && kokkosData->gas.nSpecies != nSpecies) return "gas tables";
  /*--- CPU dPdUst_i / dPdUst_j are persistent members: same edge-order dependence as RoedPdU with ionization. ---*/
  if (kind == UPWIND::MSW && config->GetIonization()) return "MSW + ionization (CPU dPdUst depends on the previous edge)";
  if (kind == UPWIND::MSW && kokkosData->gas.nSpecies != nSpecies) return "gas tables";
  if (config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT && !kokkosData->geo.has_pattern) return "no sparse pattern";
  const bool muscl = config->GetMUSCL_Flow() && (iMesh == MESH_0);
  if (muscl && nodes->GetGradient_Reconstruction().size() == 0) return "gradient not allocated";
  return "";
#else
  return "no Kokkos";
#endif
}

bool CNEMOEulerSolver::Device_UpwindResidual(CGeometry *geometry, CSolver **solver_container,
                                             CNumerics **numerics_container, CConfig *config, unsigned short iMesh) {
#ifdef HAVE_KOKKOS
  if (!Device_UpwindUnsupported(config, iMesh).empty()) return false;
  SU2_ZONE_SCOPED

  using K = CNEMOKokkosData;
  K::UpwindParams p;
  p.scheme = (config->GetKind_Upwind_Flow() == UPWIND::AUSMPLUSM)   ? K::UpwindScheme::AUSMPLUSM
             : (config->GetKind_Upwind_Flow() == UPWIND::ROE)       ? K::UpwindScheme::ROE
             : (config->GetKind_Upwind_Flow() == UPWIND::MSW)       ? K::UpwindScheme::MSW
             : (config->GetKind_Upwind_Flow() == UPWIND::AUSMPLUSUP2) ? K::UpwindScheme::AUSMPLUSUP2
                                                                    : K::UpwindScheme::AUSM;
  p.mswAlpha = config->GetMSW_Alpha();
  p.implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);
  p.muscl = (config->GetMUSCL_Flow() && (iMesh == MESH_0));
  p.limiter = (config->GetKind_SlopeLimit_Flow() != LIMITER::NONE);
  p.vanAlbada = (config->GetKind_SlopeLimit_Flow() == LIMITER::VAN_ALBADA_EDGE);
  p.kappa = config->GetMUSCL_Kappa_Flow();
  p.musclRamp = config->GetMUSCLRampValue() * config->GetNewtonKrylovRelaxation();
  p.Minf = config->GetMach();
  p.nElNumerics = config->GetIonization() ? 1 : 0; /*--- CNEMONumerics::nEl ---*/

  const bool verify = K::StageVerifyEnabled();

  /*--- Base of the CSysMatrix values (row 0 starts at non-zero 0; its diagonal block is at dia_ptr[0]). ---*/
  su2mixedfloat* jac = nullptr;
  size_t nJac = 0;
  if (p.implicit) {
    const auto& pattern = geometry->GetSparsePattern(ConnectivityType::FiniteVolume, 0);
    jac = JacobianValues(geometry, Jacobian, nVar);
    nJac = size_t(pattern.getNumNonZeros()) * nVar * nVar;
  }
  const size_t nRes = size_t(nPoint) * nVar;

  /*--- A/B: inputs of the host rerun that the device call modifies. ---*/
  std::vector<su2double> res0, jac0, np0, cnt0;
  if (verify) {
    res0.assign(LinSysRes.GetBlock(0), LinSysRes.GetBlock(0) + nRes);
    if (p.implicit) jac0.assign(jac, jac + nJac);
    kokkosData->GetNonPhysical(np0, cnt0);
  }

  /*--- Device accumulator: zero-filled (host zeroed in CommonPreprocessing), or the host copy uploaded (mixed). ---*/
  if (!Device_ResidualToDevice(geometry, config)) return false;
  auto r = kokkosData->ComputeUpwindResidual(p, verify);
  if (!deferResidualSync) Device_ResidualToHost(geometry);

  /*--- Host remainder: gas-state replay (the CPU loop leaves FluidModel in the state of its last
   *    ComputeConsistentExtrapolation call) and the non-physical reconstruction counter. ---*/
  Kokkos::Timer timer;
  if (r.haveLastRec) {
    su2double scratch[4][MAXNVAR] = {{0.0}}, eve[MAXNVAR] = {0.0};
    ComputeConsistentExtrapolation(GetFluidModel(), nSpecies, r.lastRec.data(), scratch[0], scratch[1], scratch[2],
                                   eve, scratch[3]);
    /*--- Roe / MSW: the CPU loop ends with RecomputeConservativeVector of the same state (FluidModel). ---*/
    if (p.scheme == K::UpwindScheme::ROE || p.scheme == K::UpwindScheme::MSW) RecomputeConservativeVector(scratch[0], r.lastRec.data());
  }
  ErrorCounter = 0;
  if ((iMesh == MESH_0) && (config->GetComm_Level() == COMM_FULL)) {
    unsigned long counter_local = r.nonPhysical;
    ErrorCounter += counter_local;
    counter_local = ErrorCounter;
    SU2_MPI::Reduce(&counter_local, &ErrorCounter, 1, MPI_UNSIGNED_LONG, MPI_SUM, MASTER_NODE, SU2_MPI::GetComm());
    config->SetNonphysical_Reconstr(ErrorCounter);
  }
  kokkosData->stagePerf[K::ST_UPWIND].host += timer.seconds();

  if (!verify) return true;

  /*--- A/B: keep the device results, restore the inputs, rerun the unchanged host code, compare. ---*/
  std::vector<su2double> resD(LinSysRes.GetBlock(0), LinSysRes.GetBlock(0) + nRes), jacD, npD, cntD;
  if (p.implicit) jacD.assign(jac, jac + nJac);
  kokkosData->GetNonPhysical(npD, cntD);
  const su2double errD = ErrorCounter;

  std::copy(res0.begin(), res0.end(), LinSysRes.GetBlock(0));
  if (p.implicit) std::copy(jac0.begin(), jac0.end(), jac);
  kokkosData->SetNonPhysical(np0, cnt0);

  upwindHostOnly = true;
  timer.reset();
  Upwind_Residual(geometry, solver_container, numerics_container, config, iMesh);
  kokkosData->stagePerf[K::ST_UPWIND].hostRef += timer.seconds();
  kokkosData->stagePerf[K::ST_UPWIND].hostRefCalls++;
  upwindHostOnly = false;

  std::vector<su2double> npH, cntH;
  kokkosData->GetNonPhysical(npH, cntH);
  kokkosData->StageCompare(K::ST_UPWIND, "LinSysRes", LinSysRes.GetBlock(0), resD.data(), nRes, nVar);
  if (p.implicit) {
    std::vector<su2double> jacH(jac, jac + nJac);
    kokkosData->StageCompare(K::ST_UPWIND, "Jacobian", jacH.data(), jacD.data(), nJac, size_t(nVar) * nVar);
  }
  if (p.muscl) {
    kokkosData->StageCompare(K::ST_UPWIND, "Non_Physical", npH.data(), npD.data(), nPoint);
    kokkosData->StageCompare(K::ST_UPWIND, "Non_Physical_Counter", cntH.data(), cntD.data(), nPoint);
  }
  kokkosData->StageCompareScalar(K::ST_UPWIND, "Nonphysical_Reconstr", su2double(ErrorCounter), errD);
  return true;
#else
  return false;
#endif
}

std::string CNEMOEulerSolver::Device_CenteredUnsupported(const CConfig *config) const {
#ifdef HAVE_KOKKOS
  if (!kokkosData) return "no device data";
  if (!kokkosData->geo.domain_first) return "domain points not first";
  if (config->GetDiscrete_Adjoint()) return "discrete adjoint";
  if (dynamic_grid) { auto whyDyn = Device_DynamicGridUnsupported(config); if (!whyDyn.empty()) return whyDyn; }
  if (ReducerStrategy) return "reducer strategy";
  if (ForcedHost("centered")) return "SU2_KOKKOS_NEMO_HOST";
  if (config->GetKind_Centered_Flow() != CENTERED::LAX) return "centered scheme not ported";
  /*--- CPU MeandPdU is a persistent member updated with "-=" under ionization (edge-order dependent). ---*/
  if (config->GetIonization()) return "LAX + ionization (CPU MeandPdU depends on the previous edge)";
  if (config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT && !kokkosData->geo.has_pattern) return "no sparse pattern";
  if (kokkosData->gas.nSpecies != nSpecies) return "gas tables";
  return "";
#else
  (void)config;
  return "no Kokkos";
#endif
}

bool CNEMOEulerSolver::Device_CenteredResidual(CGeometry *geometry, CSolver **solver_container,
                                               CNumerics **numerics_container, CConfig *config, unsigned short iMesh,
                                               unsigned short iRKStep) {
#ifdef HAVE_KOKKOS
  if (!Device_CenteredUnsupported(config).empty()) return false;
  SU2_ZONE_SCOPED

  using K = CNEMOKokkosData;
  const bool implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);
  const bool verify = K::StageVerifyEnabled();
  /*--- Lambda is resident only if this iteration's SetMax_Eigenvalue ran on the device. ---*/
  const bool uploadAll = verify || !Device_TimeStepUnsupported(config).empty();

  su2mixedfloat* jac = nullptr;
  size_t nJac = 0;
  if (implicit) {
    const auto& pattern = geometry->GetSparsePattern(ConnectivityType::FiniteVolume, 0);
    jac = JacobianValues(geometry, Jacobian, nVar);
    nJac = size_t(pattern.getNumNonZeros()) * nVar * nVar;
  }
  const size_t nRes = size_t(nPoint) * nVar;

  std::vector<su2double> res0, jac0;
  if (verify) {
    res0.assign(LinSysRes.GetBlock(0), LinSysRes.GetBlock(0) + nRes);
    if (implicit) jac0.assign(jac, jac + nJac);
  }

  if (!Device_ResidualToDevice(geometry, config)) return false;
  const auto lastMeanV = kokkosData->ComputeCenteredResidual(implicit, config->GetKappa_1st_Flow(), uploadAll);
  if (!deferResidualSync) Device_ResidualToHost(geometry);

  /*--- Host gas-state replay: the CPU loop calls ComputeSpeciesEve + ComputedPdU at every edge's mean state. ---*/
  Kokkos::Timer timer;
  if (!lastMeanV.empty()) {
    su2double scratch[MAXNVAR] = {0.0};
    const auto& eves = GetFluidModel()->ComputeSpeciesEve(lastMeanV[nSpecies + 1]); /*--- TVE_INDEX ---*/
    GetFluidModel()->ComputedPdU(lastMeanV.data(), eves, scratch);
  }
  kokkosData->stagePerf[K::ST_CENTERED].host += timer.seconds();

  if (!verify) return true;

  /*--- A/B: keep the device results, restore the inputs, rerun the unchanged host code, compare. ---*/
  std::vector<su2double> resD(LinSysRes.GetBlock(0), LinSysRes.GetBlock(0) + nRes), jacD;
  if (implicit) jacD.assign(jac, jac + nJac);
  std::copy(res0.begin(), res0.end(), LinSysRes.GetBlock(0));
  if (implicit) std::copy(jac0.begin(), jac0.end(), jac);

  centeredHostOnly = true;
  timer.reset();
  Centered_Residual(geometry, solver_container, numerics_container, config, iMesh, iRKStep);
  kokkosData->stagePerf[K::ST_CENTERED].hostRef += timer.seconds();
  kokkosData->stagePerf[K::ST_CENTERED].hostRefCalls++;
  centeredHostOnly = false;

  kokkosData->StageCompare(K::ST_CENTERED, "LinSysRes", LinSysRes.GetBlock(0), resD.data(), nRes, nVar);
  if (implicit) {
    std::vector<su2double> jacH(jac, jac + nJac);
    kokkosData->StageCompare(K::ST_CENTERED, "Jacobian", jacH.data(), jacD.data(), nJac, size_t(nVar) * nVar);
  }
  return true;
#else
  (void)geometry; (void)solver_container; (void)numerics_container; (void)config; (void)iMesh; (void)iRKStep;
  return false;
#endif
}

/*============================ Device (Kokkos) Step 7: source terms, BCs, residual sync ============================*/


std::string CNEMOEulerSolver::Device_DeferralUnsupported(const CConfig *config) const {
#ifdef HAVE_KOKKOS
  if (!kokkosData) return "no device data";
  if (CNEMOKokkosData::StageVerifyEnabled()) return "SU2_KOKKOS_STAGE_VERIFY";
  if (EagerForced()) return "SU2_KOKKOS_NEMO_EAGER";
  /*--- Batch A: SetResidual_DualTime is a NEMO override with the host-stage hook (device or host). ---*/
  if (config->GetnMarker_Periodic() > 0) return "periodic markers";
  if (config->GetBoolTurbomachinery() || config->GetBoolGiles()) return "turbomachinery";
  if (ReducerStrategy) return "reducer strategy";
  /*--- Every residual writer must be a NEMO function with the host-stage hook (Device_BeginHostStage). ---*/
  for (unsigned short iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    switch (config->GetMarker_All_KindBC(iMarker)) {
      case SYMMETRY_PLANE: case EULER_WALL: case FAR_FIELD: case OUTLET_FLOW: case INLET_FLOW:
      case SUPERSONIC_INLET: case SUPERSONIC_OUTLET: case ISOTHERMAL: case HEAT_FLUX: case SMOLUCHOWSKI_MAXWELL:
      case SEND_RECEIVE:
        break;
      default:
        return "marker " + config->GetMarker_All_TagBound(iMarker) + " of a kind without host-stage hook";
    }
  }
  return "";
#else
  (void)config;
  return "no Kokkos";
#endif
}

std::string CNEMOEulerSolver::Device_SourceUnsupported(const CConfig *config) const {
#ifdef HAVE_KOKKOS
  if (!kokkosData) return "no device data";
  if (!kokkosData->geo.domain_first) return "domain points not first";
  if (config->GetDiscrete_Adjoint()) return "discrete adjoint";
  if (ForcedHost("source")) return "SU2_KOKKOS_NEMO_HOST";
  if (config->GetAxisymmetric()) {
    /*--- Batch A: CSource_NEMO::ComputeAxisymmetric (2D) + device auxiliary gradients (viscous). ---*/
    if (nDim != 2) return "3D axisymmetric (the CPU source assumes 2D)";
    if (nodes->GetnAuxVar() != 3) return "axisymmetric auxiliary variables not allocated";
    if (config->GetViscous()) {
      if (dynamic_cast<const CNEMONSVariable*>(nodes) == nullptr) return "axisymmetric viscous terms without NS variables";
      if (config->GetKind_Turb_Model() != TURB_MODEL::NONE) return "axisymmetric viscous terms with turbulence";
      if (nodes->GetGradient_Primitive().size() == 0) return "gradient not allocated";
      if (config->GetnMarker_Periodic() > 0) return "axisymmetric auxiliary gradient with periodic markers";
    }
  }
  if (config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT && !kokkosData->geo.has_pattern) return "no sparse pattern";
  if (kokkosData->gas.nSpecies != nSpecies) return "gas tables";
  return "";
#else
  (void)config;
  return "no Kokkos";
#endif
}

std::string CNEMOEulerSolver::Device_BCUnsupported(const CConfig *config, int kind, int marker) const {
#ifdef HAVE_KOKKOS
  using K = CNEMOKokkosData::BCKind;
  if (!kokkosData) return "no device data";
  if (!kokkosData->geo.domain_first) return "domain points not first";
  if (config->GetDiscrete_Adjoint()) return "discrete adjoint";
  if (dynamic_grid) { auto whyDyn = Device_DynamicGridUnsupported(config); if (!whyDyn.empty()) return whyDyn; }
  if (ForcedHost("bc")) return "SU2_KOKKOS_NEMO_HOST";
  if (config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT && !kokkosData->geo.has_pattern) return "no sparse pattern";
  const auto k = static_cast<K>(kind);
  if (k == K::SYMMETRY) return "";
  /*--- Step 8: NS walls (no boundary numerics). ---*/
  if (k == K::ISOTHERMAL || k == K::HEAT_FLUX) {
    if (!config->GetViscous() || dynamic_cast<const CNEMONSVariable*>(nodes) == nullptr) return "no NS variables";
    if (config->GetIonization()) return "ionization (the CPU wall BCs stop with an error)";
    if (k == K::HEAT_FLUX && (marker >= 0 && config->GetCatalytic_Wall(marker)))
      return "catalytic heat-flux wall (not operational on the CPU either)";
    if (k == K::ISOTHERMAL && (marker >= 0 && config->GetCatalytic_Wall(marker)) && !config->GetSupercatalytic_Wall() &&
        config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT)
      return "implicit partially catalytic wall (not available on the CPU either)";
    return "";
  }
  /*--- 4.5: BC_Smoluchowski_Maxwell (no boundary numerics, no Jacobian contribution). ---*/
  if (k == K::SMOLUCHOWSKI_MAXWELL) {
    if (!config->GetViscous() || dynamic_cast<const CNEMONSVariable*>(nodes) == nullptr) return "no NS variables";
    if (config->GetIonization()) return "ionization (the CPU slip wall stops with an error)";
    if (FluidModel == nullptr) return "no host gas model (ComputeGasConstant)";
    return "";
  }
  /*--- Flux BCs use the boundary numerics, of the same type as the convective scheme (CDriver). ---*/
  /*--- CDriver: LAX (centered) and ROE use CUpwRoe_NEMO as boundary numerics. ---*/
  const bool centered = (config->GetKind_ConvNumScheme_Flow() == SPACE_CENTERED);
  if (!centered && config->GetKind_ConvNumScheme_Flow() != SPACE_UPWIND) return "convective scheme";
  const auto up = config->GetKind_Upwind_Flow();
  const bool bndRoe = centered || (up == UPWIND::ROE);
  /*--- CDriver: MSW uses CUpwMSW_NEMO as boundary numerics (conservative states, as Roe). ---*/
  const bool bndMSW = !centered && (up == UPWIND::MSW);
  if (!bndRoe && !bndMSW && up != UPWIND::AUSM && up != UPWIND::AUSMPLUSM && up != UPWIND::AUSMPLUSUP2)
    return "boundary numerics not AUSM / AUSM+M / AUSM+-up2 / Roe / MSW";
  if (!centered && up == UPWIND::AUSMPLUSUP2 && config->GetKind_TimeIntScheme_Flow() == EULER_IMPLICIT)
    return "implicit AUSM+-up2 boundary numerics (CPU Jacobian reads uninitialised M_L/M_R members)";
  /*--- BC_Supersonic_Inlet does not SetGamma: AUSM+-up2 (C*) reads the Gamma an earlier BC left (not replayed). ---*/
  if (!centered && up == UPWIND::AUSMPLUSUP2 && k == K::SUPERSONIC_INLET)
    return "AUSM+-up2: supersonic inlet reads the boundary numerics' stale Gamma (replay not ported)";
  if (bndMSW && config->GetIonization()) return "MSW boundary numerics + ionization (CPU dPdUst depends on the previous call)";
  if (bndMSW && kokkosData->gas.nSpecies != nSpecies) return "gas tables";
  if (bndRoe && config->GetIonization()) return "Roe boundary numerics + ionization (CPU RoedPdU depends on the previous call)";
  if (bndRoe && kokkosData->gas.nSpecies != nSpecies) return "gas tables";
  /*--- Batch A: AUSM+M reads the Gamma / Sensor the boundary numerics kept from an earlier BC (replayed). ---*/
  if ((k == K::SUPERSONIC_INLET || k == K::SUPERSONIC_OUTLET) && up == UPWIND::AUSMPLUSM && marker >= 0 &&
      (size_t(marker) >= bndStaleOK.size() || !bndStaleOK[marker]))
    return "AUSM+M: no earlier FAR_FIELD / OUTLET (/ SUPERSONIC_OUTLET) BC sets the boundary numerics' Gamma / Sensor "
           "(the CPU reads uninitialised values in the first iteration)";
  /*--- 4.5: the viscous part of BC_Supersonic_Inlet is ported (same CAvgGrad_NEMO pattern as BC_Far_Field). ---*/
  if ((k == K::FAR_FIELD || k == K::SUPERSONIC_INLET) && config->GetViscous() &&
      dynamic_cast<const CNEMONSVariable*>(nodes) == nullptr)
    return "viscous contribution without NS variables";
  if (k == K::OUTLET && config->GetGravityForce()) return "gravity";
  return "";
#else
  (void)config;
  (void)kind;
  (void)marker;
  return "no Kokkos";
#endif
}

bool CNEMOEulerSolver::Device_ResidualToDevice(CGeometry *geometry, const CConfig *config) {
#ifdef HAVE_KOKKOS
  using K = CNEMOKokkosData;
  if (!kokkosData) return false;
  if (!(kokkosData->AccumulatorOnDevice() || kokkosData->HostAccumulatorZero() || MixedAllowed() ||
        K::StageVerifyEnabled()))
    return false;
  const bool implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);
  su2mixedfloat* jac = implicit ? JacobianValues(geometry, Jacobian, nVar) : nullptr;
  kokkosData->AccumulatorToDevice(implicit, LinSysRes.GetBlock(0), jac);
  return true;
#else
  (void)geometry;
  (void)config;
  return false;
#endif
}

void CNEMOEulerSolver::Device_ResidualToHost(CGeometry *geometry) {
#ifdef HAVE_KOKKOS
  if (!kokkosData || !kokkosData->AccumulatorOnDevice()) return;
  /*--- The accumulator knows whether it holds a Jacobian (only then is the pointer dereferenced). ---*/
  su2mixedfloat* jac = deviceJacobianAllocated ? JacobianValues(geometry, Jacobian, nVar) : nullptr;
  kokkosData->AccumulatorToHost(LinSysRes.GetBlock(0), jac);
#else
  (void)geometry;
#endif
}

void CNEMOEulerSolver::Device_BeginHostStage(CGeometry *geometry) {
#ifdef HAVE_KOKKOS
  if (!kokkosData) return;
  /*--- Device-resident mode is only enabled when no host residual stage exists (Device_ResidentUnsupported). ---*/
  if (Device_ResidentActive())
    SU2_MPI::Error("Kokkos NEMO device-resident mode: unexpected host residual stage.", CURRENT_FUNCTION);
  Device_ResidualToHost(geometry);
  kokkosData->HostAccumulatorTouched();
#else
  (void)geometry;
#endif
}

void CNEMOEulerSolver::FinalizeSpaceIntegration(CGeometry *geometry, CConfig*, unsigned short) {
  /*--- Device-resident: the accumulator stays on the device (consumed by the device implicit / explicit update). ---*/
  if (Device_ResidentActive()) return;
  Device_ResidualToHost(geometry);
}

bool CNEMOEulerSolver::Device_SourceResidual(CGeometry *geometry, CSolver **solver_container,
                                             CNumerics **numerics_container, CConfig *config, unsigned short iMesh) {
#ifdef HAVE_KOKKOS
  if (!Device_SourceUnsupported(config).empty()) return false;
  using K = CNEMOKokkosData;
  const bool verify = K::StageVerifyEnabled();
  const bool implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);

  /*--- A/B: host inputs of the rerun (the accumulator is on the host in A/B mode: eager sync). ---*/
  su2mixedfloat* jac = implicit ? JacobianValues(geometry, Jacobian, nVar) : nullptr;
  const size_t nRes = size_t(nPoint) * nVar, nJac = implicit ? size_t(kokkosData->geo.nnz) * nVar * nVar : 0;
  std::vector<su2double> res0, jac0;
  if (verify) {
    res0.assign(LinSysRes.GetBlock(0), LinSysRes.GetBlock(0) + nRes);
    if (implicit) jac0.assign(jac, jac + nJac);
  }

  if (!Device_ResidualToDevice(geometry, config)) return false;
  SU2_ZONE_SCOPED

  K::SourceParams p;
  p.implicit = implicit;
  p.frozen = config->GetFrozen();
  p.monoatomic = config->GetMonoatomic();
  p.vtLimiting = config->GetVTTransferResidualLimiting();
  /*--- Batch A: axisymmetric source (CSource_NEMO::ComputeAxisymmetric), viscous part with device aux. gradients. ---*/
  p.axisymmetric = config->GetAxisymmetric();
  p.axiViscous = p.axisymmetric && config->GetViscous();
  p.nHeavyNumerics = nSpecies - (config->GetIonization() ? 1 : 0); /*--- CNEMONumerics::nHeavy ---*/
  p.prandtlTurb = config->GetPrandtl_Turb();
  p.RuSI = UNIVERSAL_GAS_CONSTANT;
  if (p.axiViscous) Device_AxisymAuxGradients(geometry, config, verify);
  const auto r = kokkosData->ComputeSourceResidual(p, verify);
  if (!deferResidualSync) Device_ResidualToHost(geometry);

  /*--- Same report as the CPU loop. ---*/
  if ((r.eAxi != 0) || (r.eChm != 0) || (r.eVib != 0)) {
    cout << "Warning!! Instances of NaN in the following source terms: " << endl;
    cout << "Axisymmetry: " << r.eAxi << endl;
    cout << "Chemical:    " << r.eChm << endl;
    cout << "Vib. Relax:  " << r.eVib << endl;
  }
  if (!verify) return true;

  std::vector<su2double> resD(LinSysRes.GetBlock(0), LinSysRes.GetBlock(0) + nRes), jacD;
  if (implicit) jacD.assign(jac, jac + nJac);
  std::copy(res0.begin(), res0.end(), LinSysRes.GetBlock(0));
  if (implicit) std::copy(jac0.begin(), jac0.end(), jac);

  sourceHostOnly = true;
  Kokkos::Timer timer;
  Source_Residual(geometry, solver_container, numerics_container, config, iMesh);
  kokkosData->stagePerf[K::ST_SOURCE].hostRef += timer.seconds();
  kokkosData->stagePerf[K::ST_SOURCE].hostRefCalls++;
  sourceHostOnly = false;

  kokkosData->StageCompare(K::ST_SOURCE, "LinSysRes", LinSysRes.GetBlock(0), resD.data(), nRes, nVar);
  if (implicit) {
    std::vector<su2double> jacH(jac, jac + nJac);
    kokkosData->StageCompare(K::ST_SOURCE, "Jacobian", jacH.data(), jacD.data(), nJac, size_t(nVar) * nVar);
  }
  return true;
#else
  return false;
#endif
}

bool CNEMOEulerSolver::Device_BCResidual(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics,
                                         CNumerics *visc_numerics, CConfig *config, unsigned short val_marker,
                                         int kind) {
#ifdef HAVE_KOKKOS
  if (!Device_BCUnsupported(config, kind, val_marker).empty()) return false;
  using K = CNEMOKokkosData;
  const auto bk = static_cast<K::BCKind>(kind);
  const bool verify = K::StageVerifyEnabled();
  const bool implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);

  su2mixedfloat* jac = implicit ? JacobianValues(geometry, Jacobian, nVar) : nullptr;
  const size_t nRes = size_t(nPoint) * nVar, nJac = implicit ? size_t(kokkosData->geo.nnz) * nVar * nVar : 0;

  /*--- Batch A: AUSM+M supersonic BCs read the boundary numerics' Gamma / Sensor left by an earlier BC (replay). ---*/
  const bool staleNeeded = (bk == K::BCKind::SUPERSONIC_INLET || bk == K::BCKind::SUPERSONIC_OUTLET) &&
                           (config->GetKind_Upwind_Flow() == UPWIND::AUSMPLUSM);
  K::BCParams p;
  if (staleNeeded) {
    const bool have = (bndNumerics.sensorPoint >= 0) &&
                      (bk == K::BCKind::SUPERSONIC_OUTLET || bndNumerics.gammaPoint >= 0);
    bool hasDomain = false;
    for (auto iVertex = 0ul; iVertex < geometry->nVertex[val_marker] && !hasDomain; iVertex++)
      hasDomain = geometry->nodes->GetDomain(geometry->vertex[val_marker][iVertex]->GetNode());
    if (!have && hasDomain) {
      if (Device_ResidentActive())
        SU2_MPI::Error("Kokkos NEMO device-resident mode: boundary numerics state not replayable.", CURRENT_FUNCTION);
      return false;
    }
    p.staleNumerics = true;
    if (have) {
      const bool res = Device_ResidentActive();
      auto gammaOf = [&](unsigned long i) { return res ? kokkosData->DeviceGamma(i) : nodes->GetGamma(i); };
      if (bndNumerics.gammaPoint >= 0) {
        p.staleGammaI = gammaOf(bndNumerics.gammaPoint);
        p.staleGammaJ = bndNumerics.gammaJInfty ? node_infty->GetGamma(0) : p.staleGammaI;
      }
      p.staleSensor = res ? kokkosData->DeviceSensor(bndNumerics.sensorPoint) : nodes->GetSensor(bndNumerics.sensorPoint);
    }
  }

  std::vector<su2double> res0, jac0;
  if (verify) {
    res0.assign(LinSysRes.GetBlock(0), LinSysRes.GetBlock(0) + nRes);
    if (implicit) jac0.assign(jac, jac + nJac);
  }

  if (!Device_ResidualToDevice(geometry, config)) return false;
  SU2_ZONE_SCOPED

  p.kind = bk;
  p.marker = val_marker;
  /*--- Boundary numerics (CDriver): CUpwRoe_NEMO for LAX and ROE, else the upwind scheme. ---*/
  const bool bndRoe = (config->GetKind_ConvNumScheme_Flow() == SPACE_CENTERED) ||
                      (config->GetKind_Upwind_Flow() == UPWIND::ROE);
  p.upw.scheme = bndRoe ? K::UpwindScheme::ROE
                 : (config->GetKind_Upwind_Flow() == UPWIND::MSW)       ? K::UpwindScheme::MSW
                 : (config->GetKind_Upwind_Flow() == UPWIND::AUSMPLUSM) ? K::UpwindScheme::AUSMPLUSM
                 : (config->GetKind_Upwind_Flow() == UPWIND::AUSMPLUSUP2) ? K::UpwindScheme::AUSMPLUSUP2
                                                                        : K::UpwindScheme::AUSM;
  p.upw.implicit = implicit;
  p.upw.Minf = config->GetMach();
  p.upw.mswAlpha = config->GetMSW_Alpha();
  p.upw.nElNumerics = config->GetIonization() ? 1 : 0; /*--- CNEMONumerics::nEl ---*/
  p.downloadCharac = verify || (config->GetKind_Turb_Model() != TURB_MODEL::NONE);

  K::BoundaryState bs;
  std::unique_ptr<CNEMOEulerVariable> node_inlet;
  if (bk == K::BCKind::FAR_FIELD || bk == K::BCKind::OUTLET) {
    bs.V = node_infty->GetPrimitive(0);
    bs.eve = node_infty->GetEve(0);
    bs.dPdU = implicit ? node_infty->GetdPdU(0) : nullptr;
    bs.Gamma = node_infty->GetGamma(0);
    bs.U = node_infty->GetSolution(0);
  }
  if (bk == K::BCKind::OUTLET) {
    const string Marker_Tag = config->GetMarker_All_TagBound(val_marker);
    su2double P_Exit = config->GetOutlet_Pressure(Marker_Tag);
    P_Exit = P_Exit/config->GetPressure_Ref();
    p.P_Exit = P_Exit;
  }
  if (bk == K::BCKind::SUPERSONIC_INLET) {
    /*--- Host part of BC_Supersonic_Inlet, unchanged (also leaves FluidModel in the CPU state). ---*/
    const string Marker_Tag = config->GetMarker_All_TagBound(val_marker);
    const su2double Temperature = config->GetInlet_Temperature(Marker_Tag);
    const su2double Pressure = config->GetInlet_Pressure(Marker_Tag);
    const su2double *Velocity = config->GetInlet_Velocity(Marker_Tag);
    const su2double *Mass_Frac = config->GetInlet_MassFrac();
    su2double Temperature_ve = config->GetInlet_Temperature_ve();
    if (Temperature_ve == 0.0) Temperature_ve = Temperature;
    FluidModel->SetTDStatePTTv(Pressure, Mass_Frac, Temperature, Temperature_ve);
    const su2double soundspeed = FluidModel->ComputeSoundSpeed();
    su2double Mvec[MAXNDIM] = {0.0};
    for (auto iDim = 0ul; iDim < nDim; iDim++) Mvec[iDim] = Velocity[iDim] / soundspeed;
    node_inlet = std::make_unique<CNEMOEulerVariable>(Pressure, Mass_Frac, Mvec, Temperature, Temperature_ve, 1, nDim,
                                                      nVar, nPrimVar, nPrimVarGrad, config, FluidModel);
    node_inlet->SetPrimVar(0, FluidModel);
    bs.V = node_inlet->GetPrimitive(0);
    bs.eve = node_inlet->GetEve(0);
    bs.dPdU = implicit ? node_inlet->GetdPdU(0) : nullptr;
    bs.Gamma = node_inlet->GetGamma(0);
    bs.U = node_inlet->GetSolution(0);
  }

  /*--- Step 8: viscous far-field / supersonic-inlet part (node_infty / node_inlet secondaries + gradient), NS walls. ---*/
  std::vector<su2double> inftyGrad;
  if ((bk == K::BCKind::FAR_FIELD || bk == K::BCKind::SUPERSONIC_INLET) && config->GetViscous()) {
    CNEMOEulerVariable* nb = (bk == K::BCKind::FAR_FIELD) ? node_infty : node_inlet.get();
    p.viscous = true;
    bs.Cvve = nb->GetCvve(0);
    bs.dTdU = nb->GetdTdU(0);
    bs.dTvedU = nb->GetdTvedU(0);
    const size_t ng = size_t(nPrimVarGrad) * nDim;
    auto& g = nb->GetGradient_Primitive();
    inftyGrad.assign(ng, 0.0);
    if (size_t(g.size()) >= ng) std::copy_n(g.data(), ng, inftyGrad.begin());
    bs.grad = inftyGrad.data();
  }
  const bool wallBC = (bk == K::BCKind::ISOTHERMAL || bk == K::BCKind::HEAT_FLUX);
  if (wallBC) {
    const string Marker_Tag = config->GetMarker_All_TagBound(val_marker);
    if (bk == K::BCKind::ISOTHERMAL) {
      p.Twall = config->GetIsothermal_Temperature(Marker_Tag);
      p.catalytic = config->GetCatalytic_Wall(val_marker);
      p.supercatalytic = config->GetSupercatalytic_Wall();
      if (p.catalytic && p.supercatalytic) {
        const auto* Yst = config->GetSupercatalytic_Wall_Composition();
        for (unsigned short iSpecies = 0; iSpecies < nSpecies; iSpecies++) p.Yst[iSpecies] = Yst[iSpecies];
      }
      p.catEff = config->GetCatalytic_Efficiency();
    } else {
      su2double Wall_HeatFlux = config->GetWall_HeatFlux(Marker_Tag) / config->GetHeat_Flux_Ref();
      if (config->GetIntegrated_HeatFlux()) Wall_HeatFlux /= geometry->GetSurfaceArea(config, val_marker);
      p.heatFlux = Wall_HeatFlux;
    }
  }
  const bool smolBC = (bk == K::BCKind::SMOLUCHOWSKI_MAXWELL);
  if (smolBC) {
    const string Marker_Tag = config->GetMarker_All_TagBound(val_marker);
    p.Twall = config->GetIsothermal_Temperature(Marker_Tag);
    /*--- CPU: FluidModel->ComputeGasConstant() per vertex; the gas-model state is constant over the marker loop (the
     *    host gas model is kept in the CPU state: SetPrimVar / BC_Outlet replays), so one host evaluation. ---*/
    p.gasConstant = FluidModel->ComputeGasConstant();
    p.prandtlLam = Prandtl_Lam;
    p.downloadSlip = !Device_ResidentActive();
  }

  const auto r = kokkosData->ComputeBCResidual(p, bs, verify);

  /*--- Host-authoritative NS walls: SetVelocity_Old(0) and the momentum Res_TruncError rows (resident: device). ---*/
  if (wallBC && !Device_ResidentActive()) {
    Kokkos::Timer timer;
    su2double zero[MAXNDIM] = {0.0};
    for (auto iVertex = 0ul; iVertex < geometry->nVertex[val_marker]; iVertex++) {
      const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();
      if (!geometry->nodes->GetDomain(iPoint)) continue;
      nodes->SetVelocity_Old(iPoint, zero);
      nodes->SetVel_ResTruncError_Zero(iPoint);
    }
    kokkosData->stagePerf[K::ST_BC].host += timer.seconds();
  }
  /*--- Host-authoritative slip wall: SetVelocity_Old(slip velocity), momentum Res_TruncError rows. ---*/
  if (smolBC && !Device_ResidentActive()) {
    Kokkos::Timer timer;
    for (auto iVertex = 0ul; iVertex < geometry->nVertex[val_marker]; iVertex++) {
      const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();
      if (!geometry->nodes->GetDomain(iPoint)) continue;
      nodes->SetVelocity_Old(iPoint, &r.slipVel[iVertex * nDim]);
      for (auto iDim = 0ul; iDim < nDim; iDim++) nodes->SetVal_ResTruncError_Zero(iPoint, nSpecies + iDim);
    }
    kokkosData->stagePerf[K::ST_BC].host += timer.seconds();
  }

  /*--- Host state the CPU BC_Outlet leaves behind: FluidModel (last subsonic vertex), Gamma / Gamma_Minus_One
   *    (last domain vertex), CharacPrimVar (domain vertices; only if something reads it). ---*/
  std::vector<su2double> characD;
  if (bk == K::BCKind::OUTLET) {
    Kokkos::Timer timer;
    if (r.lastSubsonic >= 0) {
      vector<su2double> rhos(r.lastVOutlet.begin(), r.lastVOutlet.begin() + nSpecies);
      FluidModel->SetTDStateRhosTTv(rhos, r.lastVOutlet[nodes->GetTIndex()], r.lastVOutlet[nodes->GetTveIndex()]);
      FluidModel->ComputerhoCvtr();
      FluidModel->ComputerhoCvve();
      FluidModel->ComputeMixtureEnergies();
    }
    for (long iVertex = long(geometry->nVertex[val_marker]) - 1; iVertex >= 0; iVertex--) {
      const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();
      if (geometry->nodes->GetDomain(iPoint)) {
        /*--- Device-resident: the host Gamma is stale, read the one value from the device. ---*/
        Gamma = Device_ResidentActive() ? kokkosData->DeviceGamma(iPoint) : nodes->GetGamma(iPoint);
        Gamma_Minus_One = Gamma - 1.0;
        break;
      }
    }
    if (p.downloadCharac) {
      for (auto iVertex = 0ul; iVertex < geometry->nVertex[val_marker]; iVertex++) {
        if (!geometry->nodes->GetDomain(geometry->vertex[val_marker][iVertex]->GetNode())) continue;
        std::copy_n(&r.charac[iVertex * nPrimVar], nPrimVar, GetCharacPrimVar(val_marker, iVertex));
      }
      if (verify) characD = r.charac;
    }
    kokkosData->stagePerf[K::ST_BC].host += timer.seconds();
  }
  if (!deferResidualSync) Device_ResidualToHost(geometry);
  if (!verify) return true;

  std::vector<su2double> resD(LinSysRes.GetBlock(0), LinSysRes.GetBlock(0) + nRes), jacD;
  if (implicit) jacD.assign(jac, jac + nJac);
  std::copy(res0.begin(), res0.end(), LinSysRes.GetBlock(0));
  if (implicit) std::copy(jac0.begin(), jac0.end(), jac);

  bcHostOnly = true;
  Kokkos::Timer timer;
  switch (bk) {
    case K::BCKind::SYMMETRY:
      BC_Sym_Plane(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
      break;
    case K::BCKind::FAR_FIELD:
      BC_Far_Field(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
      break;
    case K::BCKind::OUTLET:
      BC_Outlet(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
      break;
    case K::BCKind::SUPERSONIC_INLET:
      BC_Supersonic_Inlet(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
      break;
    case K::BCKind::SUPERSONIC_OUTLET:
      BC_Supersonic_Outlet(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
      break;
    case K::BCKind::ISOTHERMAL:
      BC_Isothermal_Wall(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
      break;
    case K::BCKind::HEAT_FLUX:
      BC_HeatFlux_Wall(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
      break;
    case K::BCKind::SMOLUCHOWSKI_MAXWELL:
      BC_Smoluchowski_Maxwell(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
      break;
  }
  kokkosData->stagePerf[K::ST_BC].hostRef += timer.seconds();
  kokkosData->stagePerf[K::ST_BC].hostRefCalls++;
  bcHostOnly = false;

  const std::string tag = std::string("LinSysRes[") + BCKindName(kind) + "]";
  kokkosData->StageCompare(K::ST_BC, tag.c_str(), LinSysRes.GetBlock(0), resD.data(), nRes, nVar);
  if (implicit) {
    std::vector<su2double> jacH(jac, jac + nJac);
    kokkosData->StageCompare(K::ST_BC, "Jacobian", jacH.data(), jacD.data(), nJac, size_t(nVar) * nVar);
  }
  if (!characD.empty()) {
    std::vector<su2double> characH;
    for (auto iVertex = 0ul; iVertex < geometry->nVertex[val_marker]; iVertex++) {
      const bool dom = geometry->nodes->GetDomain(geometry->vertex[val_marker][iVertex]->GetNode());
      for (unsigned short iVar = 0; iVar < nPrimVar; iVar++)
        characH.push_back(dom ? GetCharacPrimVar(val_marker, iVertex)[iVar] : characD[iVertex * nPrimVar + iVar]);
    }
    kokkosData->StageCompare(K::ST_BC, "CharacPrimVar", characH.data(), characD.data(), characH.size(), nPrimVar);
  }
  return true;
#else
  (void)geometry;
  (void)solver_container;
  (void)conv_numerics;
  (void)visc_numerics;
  (void)config;
  (void)val_marker;
  (void)kind;
  return false;
#endif
}

/*============================ Device (Kokkos) Step 6: device-resident state ============================*/

/*============================ Device (Kokkos) Step 8: viscous residual ============================*/

std::string CNEMOEulerSolver::Device_ViscousUnsupported(const CConfig *config, unsigned short iMesh) const {
#ifdef HAVE_KOKKOS
  if (!kokkosData) return "no device data";
  if (!config->GetViscous()) return "inviscid";
  if (dynamic_cast<const CNEMONSVariable*>(nodes) == nullptr) return "no NS variables";
  if (!kokkosData->geo.domain_first) return "domain points not first";
  if (config->GetDiscrete_Adjoint()) return "discrete adjoint";
  if (ReducerStrategy) return "reducer strategy";
  if (iMesh != MESH_0) return "coarse multigrid level (CAvgGrad_NEMO)";
  if (ForcedHost("viscous")) return "SU2_KOKKOS_NEMO_HOST";
  if (config->GetKind_TimeIntScheme() == EULER_IMPLICIT && !kokkosData->geo.has_pattern) return "no sparse pattern";
  if (nodes->GetGradient_Primitive().size() == 0) return "gradient not allocated";
  return "";
#else
  (void)config;
  (void)iMesh;
  return "no Kokkos";
#endif
}

bool CNEMOEulerSolver::Device_ViscousResidual(CGeometry *geometry, CSolver **solver_container,
                                              CNumerics **numerics_container, CConfig *config, unsigned short iMesh) {
#ifdef HAVE_KOKKOS
  if (!Device_ViscousUnsupported(config, iMesh).empty()) return false;
  using K = CNEMOKokkosData;
  const bool verify = K::StageVerifyEnabled();
  const bool implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);

  su2mixedfloat* jac = implicit ? JacobianValues(geometry, Jacobian, nVar) : nullptr;
  const size_t nRes = size_t(nPoint) * nVar, nJac = implicit ? size_t(kokkosData->geo.nnz) * nVar * nVar : 0;
  std::vector<su2double> res0, jac0;
  if (verify) {
    res0.assign(LinSysRes.GetBlock(0), LinSysRes.GetBlock(0) + nRes);
    if (implicit) jac0.assign(jac, jac + nJac);
  }

  if (!Device_ResidualToDevice(geometry, config)) return false;
  SU2_ZONE_SCOPED

  K::ViscousParams p;
  p.implicit = implicit;
  p.nElNumerics = config->GetIonization() ? 1 : 0; /*--- CNEMONumerics::nEl ---*/
  kokkosData->ComputeViscousResidual(p, verify);
  if (!deferResidualSync) Device_ResidualToHost(geometry);
  if (!verify) return true;

  std::vector<su2double> resD(LinSysRes.GetBlock(0), LinSysRes.GetBlock(0) + nRes), jacD;
  if (implicit) jacD.assign(jac, jac + nJac);
  std::copy(res0.begin(), res0.end(), LinSysRes.GetBlock(0));
  if (implicit) std::copy(jac0.begin(), jac0.end(), jac);

  viscousHostOnly = true;
  Kokkos::Timer timer;
  Viscous_Residual(geometry, solver_container, numerics_container, config, iMesh, 0);
  kokkosData->stagePerf[K::ST_VISCOUS].hostRef += timer.seconds();
  kokkosData->stagePerf[K::ST_VISCOUS].hostRefCalls++;
  viscousHostOnly = false;

  kokkosData->StageCompare(K::ST_VISCOUS, "LinSysRes", LinSysRes.GetBlock(0), resD.data(), nRes, nVar);
  if (implicit) {
    std::vector<su2double> jacH(jac, jac + nJac);
    kokkosData->StageCompare(K::ST_VISCOUS, "Jacobian", jacH.data(), jacD.data(), nJac, size_t(nVar) * nVar);
  }
  return true;
#else
  (void)geometry;
  (void)solver_container;
  (void)numerics_container;
  (void)config;
  (void)iMesh;
  return false;
#endif
}

bool CNEMOEulerSolver::Device_ResidentActive() const {
#ifdef HAVE_KOKKOS
  return kokkosData && kokkosData->ResidentActive();
#else
  return false;
#endif
}

std::vector<char> CNEMOEulerSolver::s_mgLevelDevice;

std::string CNEMOEulerSolver::Device_MGUnsupported(const CConfig *config) const {
#ifdef HAVE_KOKKOS
  /*--- 4.6-MG2: coarse levels device-resident + device transfer operators (CNEMOKokkosMG.cpp). Not covered (step-1
   *    path: fine level resident, coarse levels host): MPI (the SOLUTION / SOLUTION_OLD comms of the transfer
   *    operators are host comms), viscous (coarse CAvgGrad_NEMO residual, viscous-wall corrections), full MG. ---*/
  if (config->GetnMGLevels() == 0) return "no multigrid";
  if (const char* e = std::getenv("SU2_KOKKOS_NEMO_MG_DEVICE")) {
    if (std::atoi(e) == 0) return "SU2_KOKKOS_NEMO_MG_DEVICE=0";
  }
  if (const char* e = std::getenv("SU2_KOKKOS_NEMO_MG_RESIDENT")) {
    if (std::atoi(e) == 0) return "SU2_KOKKOS_NEMO_MG_RESIDENT=0";
  }
  if (SU2_MPI::GetSize() > 1) return "MPI (host SOLUTION / SOLUTION_OLD comms in the transfer operators)";
  if (config->GetViscous()) return "viscous (coarse-level CAvgGrad_NEMO residual not on the device)";
  for (unsigned short iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++)
    if (config->GetViscous_Wall(iMarker)) return "viscous wall marker (transfer-operator wall corrections)";
  if (config->GetMGCycle() == MG_CYCLE::FULL) return "full multigrid cycle";
  if (config->GetGrid_Movement() || dynamic_grid) return "grid movement";
  return "";
#else
  (void)config;
  return "no Kokkos";
#endif
}

bool CNEMOEulerSolver::MG_DeviceTransfer(const CSolver *coarse) const {
#ifdef HAVE_KOKKOS
  if (coarse == nullptr) return mgDevice && Device_ResidentActive();
  const auto* c = dynamic_cast<const CNEMOEulerSolver*>(coarse);
  if (c == nullptr) return false;
  if (c->deviceResident && !mgDevice)
    SU2_MPI::Error("4.6-MG2: coarse MG level device-resident but its fine level is not on the device MG path.",
                   CURRENT_FUNCTION);
  return mgDevice && c->mgDevice;
#else
  (void)coarse;
  return false;
#endif
}

void CNEMOEulerSolver::MG_DeviceOp(MGDeviceOp op, CSolver *coarse, CGeometry *geometry, CGeometry *geoCoarse,
                                   const CConfig *config, su2double value) {
#ifdef HAVE_KOKKOS
  auto* c = dynamic_cast<CNEMOEulerSolver*>(coarse);
  auto& K = *kokkosData;
  /*--- MultigridProjectEulerWall uses prim_idx.Velocity() on the conservative Solution (CPU index, kept). ---*/
  const int iVel = static_cast<int>(prim_idx.Velocity());
  if (c != nullptr) c->kokkosData->MGSetupCoarse(geoCoarse, config);
  switch (op) {
    case MGDeviceOp::RESIDUAL_TERM: K.MGResidualTerm(); break;
    case MGDeviceOp::RESTRICT_SOLUTION: K.MGRestrictSolution(*c->kokkosData, iVel); break;
    case MGDeviceOp::FORCING_TERM: K.MGForcingTerm(*c->kokkosData, value); break;
    case MGDeviceOp::PROLONG_CORRECTION: K.MGProlongCorrection(*c->kokkosData, iVel); break;
    case MGDeviceOp::PROLONG_SCATTER: K.MGProlongScatter(*c->kokkosData); break;
    case MGDeviceOp::SMOOTH_SAVE:
      K.MGSetupSmoothing(geometry, config);
      K.MGSmoothSave();
      break;
    case MGDeviceOp::SMOOTH_ITERATION: K.MGSmoothIteration(value); break;
    case MGDeviceOp::SET_CORRECTION: K.MGSetCorrection(value); break;
    case MGDeviceOp::LOCAL_CFL: K.MGFillLocalCFL(value); break;
  }
#else
  (void)op; (void)coarse; (void)geometry; (void)geoCoarse; (void)config; (void)value;
#endif
}

std::string CNEMOEulerSolver::Device_ResidentUnsupported(const CConfig *config) const {
#ifdef HAVE_KOKKOS
  using K = CNEMOKokkosData;
  if (!kokkosData) return "no device data";
  if (!std::is_same<su2mixedfloat, su2double>::value) return "mixed-precision linear solver";
  if (const char* e = std::getenv("SU2_KOKKOS_NEMO_RESIDENT")) {
    if (std::atoi(e) == 0) return "SU2_KOKKOS_NEMO_RESIDENT=0";
  }
  if (K::StageVerifyEnabled()) return "SU2_KOKKOS_STAGE_VERIFY";
  if (K::VerifyEnabled()) return "SU2_KOKKOS_C2P_VERIFY";
  if (std::getenv("SU2_KOKKOS_NEMO_HOST") && std::string(std::getenv("SU2_KOKKOS_NEMO_HOST")).size())
    return "SU2_KOKKOS_NEMO_HOST";
  if (MixedAllowed()) return "SU2_KOKKOS_NEMO_MIXED";
  const auto defer = Device_DeferralUnsupported(config);
  if (!defer.empty()) return "residual sync not deferred: " + defer;
  if (config->GetnMGLevels() > 0) {
    /*--- 4.6-MG step 1: fine level device-resident, coarse levels on the host (their own non-resident pipeline);
     *    the MG transfer operators sync through SyncHostData(MG_TRANSFER). Full MG cycle (fine-level
     *    SetProlongated_Solution, moving FinestMesh) not covered. SU2_KOKKOS_NEMO_MG_RESIDENT=0 = previous behaviour. ---*/
    if (config->GetMGCycle() == MG_CYCLE::FULL) return "full multigrid cycle";
    if (const char* e = std::getenv("SU2_KOKKOS_NEMO_MG_RESIDENT")) {
      if (std::atoi(e) == 0) return "multigrid (SU2_KOKKOS_NEMO_MG_RESIDENT=0)";
    }
  }
  if (config->GetDiscrete_Adjoint() || config->GetContinuous_Adjoint()) return "adjoint";
  if (dynamic_grid) { auto whyDyn = Device_DynamicGridUnsupported(config); if (!whyDyn.empty()) return whyDyn; }
  if (config->GetViscous()) {
    /*--- Step 8: laminar NEMO NS (device transport, viscous residual, walls, viscous far field). ---*/
    if (config->GetKind_Turb_Model() != TURB_MODEL::NONE) return "turbulence model (eddy viscosity from the host turbulence solver)";
    if (config->GetWall_Functions()) return "wall functions";
    if (!kokkosData->DeviceTransport()) return "NS transport properties on the host (model / SU2_KOKKOS_NEMO_HOST)";
    auto whyV = Device_ViscousUnsupported(config, MESH_0);
    if (!whyV.empty()) return "viscous residual: " + whyV;
    const auto gmv = config->GetKind_Gradient_Method();
    if (gmv != GREEN_GAUSS && gmv != WEIGHTED_LEAST_SQUARES) return "NS gradient method";
    whyV = Device_GradientUnsupported(config, false, gmv == GREEN_GAUSS);
    if (!whyV.empty()) return "NS gradient: " + whyV;
  }
  if (VerificationSolution) return "verification solution";
  /*--- Batch A: CFL adaptation on the device (AdaptCFLNumber_Device; no turbulence / species solvers here). ---*/
  if (config->GetCFL_Adapt() && config->GetKind_Species_Model() != SPECIES_MODEL::NONE) return "CFL adaptation with species";
  if (config->GetnMarker_Analyze() > 0) return "MARKER_ANALYZE (host surface analysis each iteration)";
  if (config->GetInvDesign_Cp()) return "inverse design";
  if (config->GetWind_Gust()) return "wind gust";
  const auto tm = config->GetTime_Marching();
  if (tm == TIME_MARCHING::DT_STEPPING_1ST || tm == TIME_MARCHING::DT_STEPPING_2ND) {
    const auto whyDT = Device_DualTimeUnsupported(config);
    if (!whyDT.empty()) return "dual time stepping: " + whyDT;
  } else if (tm != TIME_MARCHING::STEADY && tm != TIME_MARCHING::TIME_STEPPING) {
    return "time marching not ported";
  }
  const auto ti = config->GetKind_TimeIntScheme_Flow();
  if (ti != EULER_IMPLICIT && ti != EULER_EXPLICIT && ti != RUNGE_KUTTA_EXPLICIT && ti != CLASSICAL_RK4_EXPLICIT)
    return "time integration scheme not ported";
  if (ti == EULER_IMPLICIT) {
    const auto ls = config->GetKind_Linear_Solver();
    if (ls != BCGSTAB && ls != FGMRES && ls != RESTARTED_FGMRES && ls != CONJUGATE_GRADIENT)
      return "linear solver without a Kokkos device-resident branch";
    const auto pc = config->GetKind_Linear_Solver_Prec();
    /*--- 4.6: LU_SGS applied on the device (KokkosComputeLU_SGSPreconditioner, nothing to build). ---*/
    if (pc != ILU && pc != JACOBI && pc != LU_SGS) return "preconditioner without a device build (ILU / JACOBI / LU_SGS only)";
    if (config->GetKind_Linear_Solver_Inner() != LINEAR_SOLVER_INNER::NONE) return "nested linear solver";
    if (Jacobian.GetNumNonZeros() != kokkosData->geo.nnz) return "Jacobian pattern != FV pattern";
  }

  /*--- Every stage of the iteration on the device. ---*/
  if (CNEMOKokkosData::HostForced("c2p")) return "Cons2Prim on the host";
  std::string why;
  const bool centered = (config->GetKind_ConvNumScheme_Flow() == SPACE_CENTERED);
  if (!centered && config->GetKind_ConvNumScheme_Flow() != SPACE_UPWIND) return "convective scheme (host)";
  if (centered) {
    /*--- 4.4-LAX: device centered residual; Lambda halo rows come from the host MAX_EIGENVALUE comms. ---*/
    why = Device_CenteredUnsupported(config);
    if (!why.empty()) return "centered: " + why;
    if (kokkosData->geo.nPointDomain != nPoint) return "centered: halo Lambda rows (no device MAX_EIGENVALUE exchange)";
  } else {
    /*--- Batch A: AUSM+M pressure sensor on the device (Device_PressureSensor). ---*/
    why = Device_UpwindUnsupported(config, MESH_0);
    if (!why.empty()) return "upwind: " + why;
  }
  const bool muscl = !centered && config->GetMUSCL_Flow();
  if (muscl) {
    const auto gm = config->GetKind_Gradient_Method();
    if (gm != GREEN_GAUSS && gm != WEIGHTED_LEAST_SQUARES) return "gradient method";
    why = Device_GradientUnsupported(config, true, gm == GREEN_GAUSS);
    if (!why.empty()) return "gradient: " + why;
    const auto lim = config->GetKind_SlopeLimit_Flow();
    if (lim != LIMITER::NONE && lim != LIMITER::VAN_ALBADA_EDGE) {
      why = Device_LimiterUnsupported(config);
      if (!why.empty()) return "limiter: " + why;
    }
  }
  why = Device_TimeStepUnsupported(config);
  if (!why.empty()) return "time step: " + why;
  why = Device_SourceUnsupported(config);
  if (!why.empty()) return "source: " + why;
  for (unsigned short iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    int k = -1;
    switch (config->GetMarker_All_KindBC(iMarker)) {
      case FAR_FIELD: k = static_cast<int>(K::BCKind::FAR_FIELD); break;
      case OUTLET_FLOW: k = static_cast<int>(K::BCKind::OUTLET); break;
      case SUPERSONIC_INLET: k = static_cast<int>(K::BCKind::SUPERSONIC_INLET); break;
      case SUPERSONIC_OUTLET: k = static_cast<int>(K::BCKind::SUPERSONIC_OUTLET); break;
      case SYMMETRY_PLANE: case EULER_WALL: k = static_cast<int>(K::BCKind::SYMMETRY); break;
      case ISOTHERMAL: k = static_cast<int>(K::BCKind::ISOTHERMAL); break;
      case HEAT_FLUX: k = static_cast<int>(K::BCKind::HEAT_FLUX); break;
      case SMOLUCHOWSKI_MAXWELL: k = static_cast<int>(K::BCKind::SMOLUCHOWSKI_MAXWELL); break;
      case SEND_RECEIVE: continue;
      default: return "marker " + config->GetMarker_All_TagBound(iMarker) + ": BC not ported";
    }
    why = Device_BCUnsupported(config, k, iMarker);
    if (!why.empty()) return "marker " + config->GetMarker_All_TagBound(iMarker) + ": " + why;
  }
  return "";
#else
  (void)config;
  return "no Kokkos";
#endif
}

void CNEMOEulerSolver::Device_StagedComms(CGeometry *geometry, const CConfig *config, MPI_QUANTITIES comm,
                                          int group) {
#ifdef HAVE_KOKKOS
  /*--- Step 10: halo exchange of a device-resident array. Only halo rows move: default = device gather/scatter +
   *    MPI on compact send / receive buffers (host-staged, or GPU-aware with KOKKOS_GPU_AWARE_MPI= YES);
   *    SU2_KOKKOS_NEMO_HALO=host = legacy A/B path (send rows -> host container, host InitiateComms / CompleteComms,
   *    receive rows -> device). Both give the host comms' values (copies). Without MPI halos: no-op. ---*/
  (void)group;
  const auto nSend = geometry->nP2PSend > 0 ? static_cast<unsigned long>(geometry->nPoint_P2PSend[geometry->nP2PSend]) : 0ul;
  const auto nRecv = geometry->nP2PRecv > 0 ? static_cast<unsigned long>(geometry->nPoint_P2PRecv[geometry->nP2PRecv]) : 0ul;
  if (nSend == 0 && nRecv == 0) return;
  using HQ = CNEMOKokkosData::HaloQuantity;
  HQ q;
  switch (comm) {
    case MPI_QUANTITIES::SOLUTION: q = HQ::SOLUTION; break;
    case MPI_QUANTITIES::PRIMITIVE_GRADIENT: q = HQ::GRADIENT_PRIMITIVE; break;
    case MPI_QUANTITIES::PRIMITIVE_GRAD_REC: q = HQ::GRADIENT_RECONSTRUCTION; break;
    case MPI_QUANTITIES::PRIMITIVE_LIMITER: q = HQ::LIMITER; break;
    case MPI_QUANTITIES::SENSOR: q = HQ::SENSOR; break;
    case MPI_QUANTITIES::AUXVAR_GRADIENT: q = HQ::AUXVAR_GRADIENT; break;
    default:
      SU2_MPI::Error("Device-resident NEMO: no device halo exchange for this MPI quantity.", CURRENT_FUNCTION);
      return;
  }
  static const std::string haloMode = [] {
    const char* e = std::getenv("SU2_KOKKOS_NEMO_HALO");
    return std::string(e ? e : "");
  }();
  if (haloMode == "none") return; /*--- debug negative control only: stale halos must change the results ---*/
  if (haloMode != "host") {
    kokkosData->HaloExchange(geometry, q, config->GetKokkosGPUAwareMPI());
    return;
  }
  kokkosData->DownloadHaloRows(q, geometry->Local_Point_P2PSend, nSend);
  InitiateComms(geometry, config, comm);
  CompleteComms(geometry, config, comm);
  kokkosData->UploadHaloRows(q, geometry->Local_Point_P2PRecv, nRecv);
#else
  (void)geometry; (void)config; (void)comm; (void)group;
#endif
}

void CNEMOEulerSolver::Set_OldSolution() {
#ifdef HAVE_KOKKOS
  if (Device_ResidentActive()) {
    kokkosData->SetOldSolution();
    return;
  }
#endif
  CSolver::Set_OldSolution();
}

void CNEMOEulerSolver::SyncHostData(CGeometry *geometry, HostSyncScope scope) {
#ifdef HAVE_KOKKOS
  if (!Device_ResidentActive()) return;
  if (scope == HostSyncScope::BOUNDARY) {
    kokkosData->EnsureHostBoundaryPrimitive();
  } else if (scope == HostSyncScope::MG_TRANSFER) {
    /*--- 4.6-MG step 1: the fine level stays device-resident, the MG transfer operators (SetResidual_Term,
     *    SetRestricted_Solution, SetForcing_Term, Get/Smooth/SetProlongated_Correction) run on the host between the
     *    fine level's Space_Integration and its post-smoothing. They read Solution, LinSysRes, Res_TruncError and
     *    write LinSysRes and Solution: download once, hand ownership to the host; Cons2Prim re-uploads Solution, the
     *    next residual zero-fills the device LinSysRes (the CPU zeroes it too). ---*/
    using K = CNEMOKokkosData;
    kokkosData->EnsureHost(K::G_SOLUTION);
    kokkosData->EnsureHost(K::G_LINSYS);
    kokkosData->EnsureHost(K::G_TRUNCERR);
    kokkosData->MarkHostWritten(K::G_SOLUTION);
    kokkosData->MarkHostWritten(K::G_LINSYS);
    ++mgHostTransfers;
  } else {
    kokkosData->EnsureHostAll();
    /*--- Step 8: NS vorticity / strain magnitude (output only) from the downloaded gradient (CNEMONSSolver::
     *    Preprocessing skips them in device-resident mode). ---*/
    if (kokkosConfig && kokkosConfig->GetViscous()) ComputeVorticityAndStrainMag(*kokkosConfig, geometry, MESH_0);
  }
#else
  (void)geometry;
  (void)scope;
#endif
}

su2double CNEMOEulerSolver::LinSysResSquaredNorm() const {
#ifdef HAVE_KOKKOS
  if (Device_ResidentActive() && kokkosData->GetResidency(CNEMOKokkosData::G_LINSYS) == CNEMOKokkosData::Residency::DEVICE) {
    /*--- Same reduction as CSysVector::dot (ordered on Serial) + the MPI sum. ---*/
    su2double sum = kokkosData->ResidualSquaredNorm(LinSysRes.GetNElmDomain()), res = 0.0;
    SU2_MPI::Allreduce(&sum, &res, 1, MPI_DOUBLE, MPI_SUM, SU2_MPI::GetComm());
    return res;
  }
#endif
  return LinSysRes.squaredNorm();
}

/*============================ Device (Kokkos) batch A: sensor, CFL, axisymmetry, RK / dual time ============================*/

bool CNEMOEulerSolver::Device_PressureSensor(CGeometry *geometry, const CConfig *config) {
#ifdef HAVE_KOKKOS
  if (!kokkosData) return false;
  using K = CNEMOKokkosData;
  if (Device_ResidentActive()) {
    /*--- Device SetPressureDiffusionSensor on the resident Primitive, device halo exchange of the sensor. ---*/
    kokkosData->ComputePressureSensor(nullptr);
    Device_StagedComms(geometry, config, MPI_QUANTITIES::SENSOR, K::G_SENSOR);
    return true;
  }
  if (K::StageVerifyEnabled() && kokkosData->geo.domain_first) {
    /*--- A/B: device sensor into a copy, then the unchanged host code (the run continues with its result). ---*/
    kokkosData->UploadPrimitiveOnly();
    std::vector<su2double> dev(nPointDomain);
    kokkosData->ComputePressureSensor(dev.data());
    Kokkos::Timer timer;
    SetPressureDiffusionSensor(geometry, const_cast<CConfig*>(config));
    kokkosData->stagePerf[K::ST_SENSOR].hostRef += timer.seconds();
    kokkosData->stagePerf[K::ST_SENSOR].hostRefCalls++;
    kokkosData->StageCompare(K::ST_SENSOR, "Sensor", nodes->GetSensor().data(), dev.data(), nPointDomain);
    return true;
  }
#else
  (void)geometry;
  (void)config;
#endif
  return false;
}

void CNEMOEulerSolver::RecordBoundaryNumerics(CGeometry *geometry, const CConfig *config, unsigned short val_marker,
                                              int kind) {
#ifdef HAVE_KOKKOS
  if (!kokkosData) return;
  /*--- The BC sets Gamma (and, AUSM+M, Sensor) of the shared boundary numerics at every domain vertex: the last one
   *    remains. kind: CNEMOKokkosData::BCKind (FAR_FIELD, OUTLET: Gamma_j = node_infty; SUPERSONIC_OUTLET: Gamma_j =
   *    Gamma_i, no sensor). ---*/
  const bool ausmPlusM = (config->GetKind_Upwind_Flow() == UPWIND::AUSMPLUSM);
  for (long iVertex = long(geometry->nVertex[val_marker]) - 1; iVertex >= 0; iVertex--) {
    const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();
    if (!geometry->nodes->GetDomain(iPoint)) continue;
    bndNumerics.gammaPoint = long(iPoint);
    if (kind == static_cast<int>(CNEMOKokkosData::BCKind::SUPERSONIC_OUTLET)) {
      bndNumerics.gammaJInfty = false;
    } else {
      bndNumerics.gammaJInfty = true;
      if (ausmPlusM) bndNumerics.sensorPoint = long(iPoint);
    }
    return;
  }
#else
  (void)geometry;
  (void)config;
  (void)val_marker;
  (void)kind;
#endif
}

void CNEMOEulerSolver::Device_AxisymAuxGradients(CGeometry *geometry, const CConfig *config, bool verify) {
#ifdef HAVE_KOKKOS
  using K = CNEMOKokkosData;
  const auto gm = config->GetKind_Gradient_Method();
  /*--- ComputeAxisymmetricAuxGradients computes the gradient only for these two methods. ---*/
  const int method = (gm == GREEN_GAUSS) ? 1 : (gm == WEIGHTED_LEAST_SQUARES) ? 2 : 0;
  auto& grad = nodes->GetAuxVarGradient();
  if (verify) {
    kokkosData->UploadPrimitiveOnly();
    kokkosData->UploadTransport();
    kokkosData->UploadAuxVar();
  }
  const bool deviceSym = K::DeviceSymmetryEnabled();
  kokkosData->ComputeAxisymmetricAuxGradient(method, deviceSym);
  if (method != 0 && !deviceSym) {
    /*--- Symmetry / Euler-wall correction with the CPU routine on the downloaded symmetry rows (as the primitives). ---*/
    kokkosData->DownloadSymmetryRowsAux(grad.data());
    Kokkos::Timer timer;
    CorrectGradientsSymmetryHost(geometry, config, grad, 3, -1);
    kokkosData->stagePerf[K::ST_AUXGRAD].host += timer.seconds();
    kokkosData->UploadSymmetryRowsAux(grad.data());
  }
  if (verify) {
    std::vector<su2double> dA(size_t(nPoint) * 3), dG(size_t(nPointDomain) * 3 * nDim);
    kokkosData->DownloadAuxVar(dA.data(), dG.data());
    Kokkos::Timer timer;
    ComputeAxisymmetricAuxGradients(geometry, config);
    kokkosData->stagePerf[K::ST_AUXGRAD].hostRef += timer.seconds();
    kokkosData->stagePerf[K::ST_AUXGRAD].hostRefCalls++;
    kokkosData->StageCompare(K::ST_AUXGRAD, "AuxVar", nodes->GetAuxVar().data(), dA.data(), dA.size(), 3);
    kokkosData->StageCompare(K::ST_AUXGRAD, "Grad_AuxVar", grad.data(), dG.data(), dG.size(), size_t(3) * nDim);
    return;
  }
  /*--- Halo rows: read by nothing in the device source (domain points), exchanged to keep the CPU state. ---*/
  if (Device_ResidentActive() && method != 0)
    Device_StagedComms(geometry, config, MPI_QUANTITIES::AUXVAR_GRADIENT, K::G_AUXVAR);
#else
  (void)geometry;
  (void)config;
  (void)verify;
#endif
}

std::string CNEMOEulerSolver::Device_DynamicGridUnsupported(const CConfig *config) const {
#ifdef HAVE_KOKKOS
  if (!dynamic_grid) return "";
  if (!kokkosData) return "no device data";
  if (const char* e = std::getenv("SU2_KOKKOS_NEMO_DYNGRID")) {
    if (std::atoi(e) == 0) return "dynamic grid (SU2_KOKKOS_NEMO_DYNGRID=0)";
  }
  if (!kokkosData->geo.dynamic) return "dynamic grid (device geometry not uploaded)";
  if (config->GetnMGLevels() > 0) return "dynamic grid with multigrid";
  if (config->GetnMarker_Periodic() > 0) return "dynamic grid with periodic markers";
  if (config->GetDiscrete_Adjoint()) return "dynamic grid with discrete adjoint";
  return "";
#else
  (void)config;
  return "no Kokkos";
#endif
}

void CNEMOEulerSolver::Device_RefreshDynamicGeometry(CGeometry *geometry, const CConfig *config) {
#ifdef HAVE_KOKKOS
  if (!kokkosData || !kokkosData->geo.dynamic) return;
  const unsigned long t = config->GetTimeIter(), o = config->GetOuterIter();
  if (t == dynGeoTimeIter && o == dynGeoOuterIter) return;
  dynGeoTimeIter = t;
  dynGeoOuterIter = o;
  kokkosData->RefreshDynamicGeometry(geometry);
#else
  (void)geometry;
  (void)config;
#endif
}

std::string CNEMOEulerSolver::Device_DualTimeUnsupported(const CConfig *config) const {
#ifdef HAVE_KOKKOS
  if (!kokkosData) return "no device data";
  if (!kokkosData->geo.domain_first) return "domain points not first";
  if (config->GetDiscrete_Adjoint()) return "discrete adjoint";
  if (dynamic_grid) { auto whyDyn = Device_DynamicGridUnsupported(config); if (!whyDyn.empty()) return whyDyn; }
  if (ForcedHost("dualtime")) return "SU2_KOKKOS_NEMO_HOST";
  if (config->GetKind_TimeIntScheme() == EULER_IMPLICIT && !kokkosData->geo.has_pattern) return "no sparse pattern";
  if (size_t(nodes->GetSolution_time_n().size()) < size_t(nPoint) * nVar) return "Solution_time_n not allocated";
  return "";
#else
  (void)config;
  return "no Kokkos";
#endif
}

void CNEMOEulerSolver::SetResidual_DualTime(CGeometry *geometry, CSolver **solver_container, CConfig *config,
                                            unsigned short iRKStep, unsigned short iMesh,
                                            unsigned short RunTime_EqSystem) {
#ifdef HAVE_KOKKOS
  if (!dualTimeHostOnly &&
      Device_DualTimeResidual(geometry, solver_container, config, iRKStep, iMesh, RunTime_EqSystem))
    return;
  Device_BeginHostStage(geometry);
#endif
  CFVMFlowSolverBase<CNEMOEulerVariable, ENUM_REGIME::COMPRESSIBLE>::SetResidual_DualTime(
      geometry, solver_container, config, iRKStep, iMesh, RunTime_EqSystem);
}

bool CNEMOEulerSolver::Device_DualTimeResidual(CGeometry *geometry, CSolver **solver_container, CConfig *config,
                                               unsigned short iRKStep, unsigned short iMesh,
                                               unsigned short RunTime_EqSystem) {
#ifdef HAVE_KOKKOS
  if (!Device_DualTimeUnsupported(config).empty()) return false;
  using K = CNEMOKokkosData;
  const bool verify = K::StageVerifyEnabled();
  const bool implicit = (config->GetKind_TimeIntScheme() == EULER_IMPLICIT);
  su2mixedfloat* jac = implicit ? JacobianValues(geometry, Jacobian, nVar) : nullptr;
  const size_t nRes = size_t(nPoint) * nVar, nJac = implicit ? size_t(kokkosData->geo.nnz) * nVar * nVar : 0;
  std::vector<su2double> res0, jac0;
  if (verify) {
    res0.assign(LinSysRes.GetBlock(0), LinSysRes.GetBlock(0) + nRes);
    if (implicit) jac0.assign(jac, jac + nJac);
  }
  if (!Device_ResidualToDevice(geometry, config)) return false;
  SU2_ZONE_SCOPED

  kokkosData->DualTimeResidual(config->GetTime_Marching() == TIME_MARCHING::DT_STEPPING_2ND, implicit,
                               config->GetDelta_UnstTimeND(), verify);
  if (!deferResidualSync) Device_ResidualToHost(geometry);
  if (!verify) return true;

  std::vector<su2double> resD(LinSysRes.GetBlock(0), LinSysRes.GetBlock(0) + nRes), jacD;
  if (implicit) jacD.assign(jac, jac + nJac);
  std::copy(res0.begin(), res0.end(), LinSysRes.GetBlock(0));
  if (implicit) std::copy(jac0.begin(), jac0.end(), jac);
  dualTimeHostOnly = true;
  Kokkos::Timer timer;
  SetResidual_DualTime(geometry, solver_container, config, iRKStep, iMesh, RunTime_EqSystem);
  kokkosData->stagePerf[K::ST_DUALTIME].hostRef += timer.seconds();
  kokkosData->stagePerf[K::ST_DUALTIME].hostRefCalls++;
  dualTimeHostOnly = false;
  kokkosData->StageCompare(K::ST_DUALTIME, "LinSysRes", LinSysRes.GetBlock(0), resD.data(), nRes, nVar);
  if (implicit) {
    std::vector<su2double> jacH(jac, jac + nJac);
    kokkosData->StageCompare(K::ST_DUALTIME, "Jacobian", jacH.data(), jacD.data(), nJac, size_t(nVar) * nVar);
  }
  return true;
#else
  (void)geometry;
  (void)solver_container;
  (void)config;
  (void)iRKStep;
  (void)iMesh;
  (void)RunTime_EqSystem;
  return false;
#endif
}

bool CNEMOEulerSolver::SetDualTime_Device(su2double cfl) {
#ifdef HAVE_KOKKOS
  if (!Device_ResidentActive()) return false;
  kokkosData->DualTimeShift(cfl);
  return true;
#else
  (void)cfl;
  return false;
#endif
}

bool CNEMOEulerSolver::AdaptCFLNumber_Device(unsigned short iMesh, bool reduceCFL, bool canIncrease, bool resetCFL,
                                             su2double CFLFactorDecrease, su2double CFLFactorIncrease,
                                             su2double CFLMin, su2double CFLMax, su2double MGFactor, bool stats,
                                             su2double& cflMin, su2double& cflMax, su2double& cflSum) {
#ifdef HAVE_KOKKOS
  /*--- 4.6-MG2: coarse levels too when they are device-resident (their own LocalCFL / UnderRelaxation). ---*/
  (void)iMesh;
  if (!Device_ResidentActive()) return false;
  CNEMOKokkosData::CFLAdaptParams p;
  p.reduceCFL = reduceCFL;
  p.canIncrease = canIncrease;
  p.resetCFL = resetCFL;
  p.stats = stats;
  p.factorDecrease = CFLFactorDecrease;
  p.factorIncrease = CFLFactorIncrease;
  p.cflMin = CFLMin;
  p.cflMax = CFLMax;
  p.mgFactor = MGFactor;
  kokkosData->AdaptCFL(p, cflMin, cflMax, cflSum);
  return true;
#else
  (void)iMesh; (void)reduceCFL; (void)canIncrease; (void)resetCFL; (void)CFLFactorDecrease; (void)CFLFactorIncrease;
  (void)CFLMin; (void)CFLMax; (void)MGFactor; (void)stats; (void)cflMin; (void)cflMax; (void)cflSum;
  return false;
#endif
}

bool CNEMOEulerSolver::Device_ExplicitRK(CGeometry *geometry, CConfig *config, unsigned short iRKStep, bool rk4) {
#ifdef HAVE_KOKKOS
  if (!Device_ResidentActive()) return false;
  /*--- Explicit_Iteration_impl<RUNGE_KUTTA_EXPLICIT / CLASSICAL_RK4_EXPLICIT> with the Identity "preconditioner",
   *    same coefficients (the hard-coded classical RK4 tables), staged SOLUTION comms. ---*/
  const su2double RK_AlphaCoeff = config->Get_Alpha_RKStep(iRKStep);
  const su2double RK_FuncCoeff[] = {1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0};
  const su2double RK_TimeCoeff[] = {0.5, 0.5, 1.0, 1.0};
  CNEMOKokkosData::ResidualReduction red;
  kokkosData->ExplicitRKUpdate(rk4, iRKStep, RK_AlphaCoeff, rk4 ? RK_TimeCoeff[iRKStep] : 0.0,
                               rk4 ? RK_FuncCoeff[iRKStep] : 0.0, red);
  ResidualReductions_FromAllThreads(geometry, config, red.rms, red.max, red.idx);
  Device_StagedComms(geometry, config, MPI_QUANTITIES::SOLUTION, CNEMOKokkosData::G_SOLUTION);
  ComputeVerificationError(geometry, config);
  return true;
#else
  (void)geometry;
  (void)config;
  (void)iRKStep;
  (void)rk4;
  return false;
#endif
}
