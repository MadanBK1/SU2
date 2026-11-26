/*!
 * \file CNEMOEulerSolver.hpp
 * \brief Headers of the CNEMOEulerSolver class
 * \author S. R. Copeland, F. Palacios, W. Maier.
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

#pragma once

#include "../variables/CNEMOEulerVariable.hpp"
#include "../fluid/CNEMOGas.hpp"
#include "CFVMFlowSolverBase.hpp"

#ifdef HAVE_KOKKOS
#include <memory>
class CNEMOKokkosData;
#endif

/*!
 * \class CNEMOEulerSolver
 * \brief Main class for defining the NEMO Euler's flow solver.
 * \ingroup Euler_Equations
 * \author S. R. Copeland, F. Palacios, W. Maier.
 * \version 8.5.0 "Harrier"
 */
class CNEMOEulerSolver : public CFVMFlowSolverBase<CNEMOEulerVariable, ENUM_REGIME::COMPRESSIBLE> {
protected:

  su2double
  Prandtl_Lam = 0.0,              /*!< \brief Laminar Prandtl number. */
  Prandtl_Turb = 0.0;             /*!< \brief Turbulent Prandtl number. */

  unsigned short nSpecies;        /*!< \brief Number of species in the gas mixture. */

  su2double
  Energy_ve_Inf,                  /*!< \brief Vib.-el. free stream energy. */
  Temperature_ve_Inf;             /*!< \brief Vib.-el. free stream temperature. */
  const su2double *MassFrac_Inf;  /*!< \brief Free stream species mass fraction. */

  su2double *Source;              /*!< \brief Auxiliary vector to store source terms. */

  unsigned long ErrorCounter = 0; /*!< \brief Counter for number of un-physical states. */

  CNEMOGas  *FluidModel;          /*!< \brief fluid model used in the solver */

  CNEMOEulerVariable* node_infty = nullptr;

#ifdef HAVE_KOKKOS
  /*--- Device mirror + device Cons2Prim, only created when ENABLE_KOKKOS= YES (and the gas model is SU2TCLib). ---*/
  std::unique_ptr<CNEMOKokkosData> kokkosData;
#endif

  /*!
   * \brief Device (Kokkos) Steps 3-4. Each returns false when the host path must run instead
   *        (no ENABLE_KOKKOS, unsupported option); always false without HAVE_KOKKOS.
   *        With SU2_KOKKOS_STAGE_VERIFY=1 they also run the host path and compare (A/B).
   * \param[in] greenGauss - GREEN_GAUSS (true) or the least-squares family (weighting from config, as the base class).
   * \param[in] massFractions - NEMO-NS Green-Gauss override (gradient of Y_s instead of rho_s).
   */
  bool Device_PrimitiveGradient(CGeometry *geometry, const CConfig *config, bool reconstruction, bool greenGauss,
                                bool massFractions);
  bool Device_PrimitiveLimiter(CGeometry *geometry, const CConfig *config);

  /*!
   * \brief Device (Kokkos) Step 5: Upwind_Residual for AUSM / AUSM+M into the device residual accumulator (Step 7:
   *        no download here, see Device_ResidualToHost). False -> run the host Upwind_Residual. A/B with
   *        SU2_KOKKOS_STAGE_VERIFY=1.
   */
  bool Device_UpwindResidual(CGeometry *geometry, CSolver **solver_container, CNumerics **numerics_container,
                             CConfig *config, unsigned short iMesh);
  std::string Device_UpwindUnsupported(const CConfig *config, unsigned short iMesh) const;
  bool upwindHostOnly = false; /*!< \brief Set while Upwind_Residual must run the host code (A/B rerun). */

  /*!
   * \brief Device (Kokkos) 4.4-LAX: Centered_Residual for LAX-FRIEDRICH into the device residual accumulator.
   *        False -> run the host Centered_Residual. A/B with SU2_KOKKOS_STAGE_VERIFY=1.
   */
  bool Device_CenteredResidual(CGeometry *geometry, CSolver **solver_container, CNumerics **numerics_container,
                               CConfig *config, unsigned short iMesh, unsigned short iRKStep);
  std::string Device_CenteredUnsupported(const CConfig *config) const;
  bool centeredHostOnly = false; /*!< \brief Set while Centered_Residual must run the host code (A/B rerun). */

  /*!
   * \brief Device (Kokkos) Step 7: source terms (chemistry + V-T relaxation) and Euler boundary conditions added into
   *        the device residual accumulator, which is downloaded once at the end of the space integration
   *        (FinalizeSpaceIntegration) when the whole residual pipeline can stay on the device ("deferred" sync), or
   *        after every device stage otherwise ("eager"). A host stage first moves the accumulator to the host
   *        (Device_BeginHostStage). A device stage that finds the accumulator on the host (non-zero) uploads it only
   *        with SU2_KOKKOS_NEMO_MIXED=1 or in A/B mode, otherwise it runs on the host (no extra transfers).
   */
  bool Device_SourceResidual(CGeometry *geometry, CSolver **solver_container, CNumerics **numerics_container,
                             CConfig *config, unsigned short iMesh);
  std::string Device_SourceUnsupported(const CConfig *config) const;
  /*! \brief kind: CNEMOKokkosData::BCKind. */
  bool Device_BCResidual(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics,
                         CNumerics *visc_numerics, CConfig *config, unsigned short val_marker, int kind);
  std::string Device_BCUnsupported(const CConfig *config, int kind, int marker = -1) const;
  /*! \brief Make the device accumulator authoritative if allowed (see above); false -> run the stage on the host. */
  bool Device_ResidualToDevice(CGeometry *geometry, const CConfig *config);
  /*! \brief Download the device accumulator into LinSysRes / Jacobian if it is authoritative. */
  void Device_ResidualToHost(CGeometry *geometry);
  /*! \brief Called at the top of every host residual stage (no-op without device data). */
  void Device_BeginHostStage(CGeometry *geometry);
  /*! \brief Static analysis: "" if the accumulator may stay on the device until FinalizeSpaceIntegration. */
  std::string Device_DeferralUnsupported(const CConfig *config) const;
  bool deferResidualSync = false; /*!< \brief Deferred (single) residual download, set at construction. */
  bool deviceJacobianAllocated = false; /*!< \brief The Jacobian was initialized (implicit at construction). */
  bool sourceHostOnly = false, bcHostOnly = false; /*!< \brief A/B reruns of the host code. */

  /*! \brief Reason a device stage is not used ("" if it is), shared by the dispatch and the startup report. */
  std::string Device_GradientUnsupported(const CConfig *config, bool reconstruction, bool greenGauss) const;
  std::string Device_LimiterUnsupported(const CConfig *config) const;
  std::string Device_TimeStepUnsupported(const CConfig *config) const;
  void PrintDeviceStages(const CConfig *config) const;

  /*!
   * \brief Device (Kokkos) Step 6: device-resident mode. When every stage of the iteration runs on the device
   *        (Device_ResidentUnsupported() == ""), the device state is authoritative: no per-iteration state transfers,
   *        device PrepareImplicitIteration / CompleteImplicitIteration / ExplicitEuler_Iteration / Set_OldSolution,
   *        LinSysRes / LinSysSol / Jacobian handed to the Kokkos linear solver on the device, host copies refreshed
   *        lazily (SyncHostData, LinSysResSquaredNorm).
   */
  std::string Device_ResidentUnsupported(const CConfig *config) const;
  bool deviceResident = false;   /*!< \brief Device-resident mode (decided at construction). */
  unsigned long mgHostTransfers = 0; /*!< \brief 4.6-MG step 1: fine-level MG transfer syncs (perf line). */
  /*! \brief 4.6-MG2: why the device MG path (coarse levels resident, device transfer operators) is off ("" = on). */
  std::string Device_MGUnsupported(const CConfig *config) const;
  bool mgDevice = false;   /*!< \brief 4.6-MG2: this level is on the device MG path (decided at construction). */
  static std::vector<char> s_mgLevelDevice; /*!< \brief 4.6-MG2: per MG level, mgDevice (levels are built in order). */
  bool hostAccNeedsZero = false; /*!< \brief Resident: host LinSysRes / Jacobian were not zeroed (stale). */
  bool Device_ResidentActive() const;
  /*! \brief Device-resident MPI comms: download the send rows of the group, host comms, upload the receive rows. */
  void Device_StagedComms(CGeometry *geometry, const CConfig *config, MPI_QUANTITIES comm, int group);

  /*!
   * \brief Device (Kokkos) Step 8: NS viscous residual (CAvgGradCorrected_NEMO edges) into the device accumulator.
   *        False -> run the host Viscous_Residual. A/B with SU2_KOKKOS_STAGE_VERIFY=1.
   */
  bool Device_ViscousResidual(CGeometry *geometry, CSolver **solver_container, CNumerics **numerics_container,
                              CConfig *config, unsigned short iMesh);
  std::string Device_ViscousUnsupported(const CConfig *config, unsigned short iMesh) const;
  bool viscousHostOnly = false; /*!< \brief A/B rerun of the host viscous residual. */
  const CConfig *kokkosConfig = nullptr; /*!< \brief Config of the run (deferred host work in SyncHostData). */

  /*! \brief Host implementations (unchanged CPU code), used by the fallback and the A/B verification. */
  void SetTime_Step_Host(CGeometry *geometry, CSolver **solver_container, CConfig *config,
                         unsigned short iMesh, unsigned long Iteration);
  void SetMax_Eigenvalue_Host(CGeometry *geometry, CConfig *config);
  void CorrectGradientsSymmetryHost(CGeometry *geometry, const CConfig *config, CVectorOfMatrix& gradient) const;
  void CorrectGradientsSymmetryHost(CGeometry *geometry, const CConfig *config, CVectorOfMatrix& gradient,
                                    size_t varEnd, int idxVel) const;

  /*!
   * \brief Device (Kokkos) batch A. Each returns false when the host code must run instead.
   *        Device_PressureSensor: SetPressureDiffusionSensor on the device (device-resident mode; A/B otherwise).
   *        Device_AxisymAuxGradients: ComputeAxisymmetricAuxGradients on the device (inside the device source stage).
   *        Device_DualTimeResidual: SetResidual_DualTime into the device accumulator.
   */
  bool Device_PressureSensor(CGeometry *geometry, const CConfig *config);
  void Device_AxisymAuxGradients(CGeometry *geometry, const CConfig *config, bool verify);
  bool Device_DualTimeResidual(CGeometry *geometry, CSolver **solver_container, CConfig *config,
                               unsigned short iRKStep, unsigned short iMesh, unsigned short RunTime_EqSystem);
  std::string Device_DualTimeUnsupported(const CConfig *config) const;
  bool dualTimeHostOnly = false; /*!< \brief A/B rerun of the host SetResidual_DualTime. */
  /*!
   * \brief 6b.a dynamic grid on the device: "" if the device stages may run with dynamic_grid (grid velocity /
   *        GCL terms in the device time step, max eigenvalue and dual-time kernels; metric arrays re-uploaded once per
   *        time / outer iteration by Device_RefreshDynamicGeometry), else the host-fallback reason.
   *        SU2_KOKKOS_NEMO_DYNGRID=0: previous behaviour (dynamic grid -> host).
   */
  std::string Device_DynamicGridUnsupported(const CConfig *config) const;
  void Device_RefreshDynamicGeometry(CGeometry *geometry, const CConfig *config);
  unsigned long dynGeoTimeIter = ~0ul, dynGeoOuterIter = ~0ul; /*!< \brief last refreshed (time, outer) iteration */
  bool Device_ExplicitRK(CGeometry *geometry, CConfig *config, unsigned short iRKStep, bool rk4);

  /*!
   * \brief Batch A: the boundary numerics (CONV_BOUND_TERM, shared by every marker) keep Gamma / Sensor from the last
   *        BC that set them; BC_Supersonic_Inlet (Gamma, Sensor) and BC_Supersonic_Outlet (Sensor) read those with
   *        AUSM+M. Replay: the last domain vertex of the last such BC (FAR_FIELD / OUTLET: Gamma_i = Gamma(point),
   *        Gamma_j = Gamma_infty, Sensor = Sensor(point); SUPERSONIC_OUTLET: Gamma_i = Gamma_j = Gamma(point)) in this
   *        space integration. bndStaleOK[marker]: a setter precedes the marker in the BC order on every rank (else the
   *        first iteration reads uninitialised values on the CPU: host BC).
   */
  struct BoundaryNumericsState {
    long gammaPoint = -1, sensorPoint = -1;
    bool gammaJInfty = false;
  } bndNumerics;
  std::vector<char> bndStaleOK;
  void RecordBoundaryNumerics(CGeometry *geometry, const CConfig *config, unsigned short val_marker, int kind);

  /*!
   * \brief Set the maximum value of the eigenvalue.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] config - Definition of the particular problem.
   */
  void SetMax_Eigenvalue(CGeometry *geometry, CConfig *config);

  /*!
   * \brief Compute a pressure sensor switch.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] config - Definition of the particular problem.
   */
  inline void SetCentered_Dissipation_Sensor(CGeometry *geometry, CConfig *config) { }

  /*!
   * \brief Set the value of undivided laplacian.
   * \param[in] val_und_lapl_i Undivided laplacian at point i.
   * \param[in] val_und_lapl_j Undivided laplacian at point j.
   */
  inline void SetUndivided_Laplacian(CGeometry *geometry, CConfig *config) { }

  /*!
   * \brief Set reference values for pressure, forces, etc.
   */
  void SetReferenceValues(const CConfig& config) final;

public:
  CNEMOEulerSolver() = delete;

  /*!
   * \brief Contructor of the class.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] config - Definition of the particular problem.
   */
  CNEMOEulerSolver(CGeometry *geometry, CConfig *config, unsigned short iMesh, const bool navier_stokes = false);

  /*!
   * \brief Destructor of the class.
   */
  ~CNEMOEulerSolver(void) override;

  /*!
   * \brief Compute the time step for solving the Euler equations.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] config - Definition of the particular problem.
   * \param[in] iMesh - Index of the mesh in multigrid computations.
   * \param[in] Iteration - Value of the current iteration.
   */
  void SetTime_Step(CGeometry *geometry, CSolver **solver_container,
                    CConfig *config, unsigned short iMesh, unsigned long Iteration) final;

  /*!
   * \brief Compute the spatial integration using a centered scheme.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] numerics - Description of the numerical method.
   * \param[in] config - Definition of the particular problem.
   * \param[in] iMesh - Index of the mesh in multigrid computations.
   */
  void Centered_Residual(CGeometry *geometry, CSolver **solver_container, CNumerics **numerics,
                         CConfig *config, unsigned short iMesh, unsigned short iRKStep) final;

  /*!
   * \brief Compute the spatial integration using a upwind scheme.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] solver - Description of the numerical method.
   * \param[in] config - Definition of the particular problem.
   * \param[in] iMesh - Index of the mesh in multigrid computations.
   */
  void Upwind_Residual(CGeometry *geometry, CSolver **solver_container, CNumerics **numerics_container,
                       CConfig *config, unsigned short iMesh) final;

  /*!
   * \brief Recompute the extrapolated quantities, after MUSCL reconstruction,
   *        in a more thermodynamically consistent way.
   * \param[in] V - primitve variables.
   * \param[out] d*dU - reconstructed secondaryvariables.
   * \param[out] val_eves - reconstructed eve per species.
   * \param[out] val_cvves - reconstructed cvve per species.
   * \param[out] Gamma - reconstructed gamma.
   */
  static su2double ComputeConsistentExtrapolation(CNEMOGas *fluidmodel, unsigned short nSpecies, su2double *V,
                                                  su2double* dPdU, su2double* dTdU, su2double* dTvedU,
                                                  su2double* val_eves, su2double* val_cvves);
  /*!
   * \brief Source term integration.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] numerics - Description of the numerical method.
   * \param[in] second_numerics - Description of the second numerical method.
   * \param[in] config - Definition of the particular problem.
   * \param[in] iMesh - Index of the mesh in multigrid computations.
   */
  void Source_Residual(CGeometry *geometry, CSolver **solver_container, CNumerics **numerics_container,
                       CConfig *config, unsigned short iMesh) final;

  /*!
   * \brief Preprocessing actions common to the Euler and NS solvers.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] config - Definition of the particular problem.
   * \param[in] iRKStep - Current step of the Runge-Kutta iteration.
   * \param[in] RunTime_EqSystem - System of equations which is going to be solved.
   * \param[in] Output - boolean to determine whether to print output.
   */
  void CommonPreprocessing(CGeometry *geometry, CSolver **solver_container, CConfig *config, unsigned short iMesh,
                           unsigned short iRKStep, unsigned short RunTime_EqSystem, bool Output);

  /*!
   * \brief Compute the velocity^2, SoundSpeed, Pressure, Enthalpy, Viscosity.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] config - Definition of the particular problem.
   * \param[in] iRKStep - Current step of the Runge-Kutta iteration.
   * \param[in] RunTime_EqSystem - System of equations which is going to be solved.
   */
  void Preprocessing(CGeometry *geometry, CSolver **solver_container, CConfig *config, unsigned short iMesh,
                     unsigned short iRKStep, unsigned short RunTime_EqSystem, bool Output) override;

  /*!
   * \brief Computes primitive variables.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] config - Definition of the particular problem.
   * \param[in] Output - boolean to determine whether to print output.
   * \return - The number of non-physical points.
   */
  virtual unsigned long SetPrimitive_Variables(CSolver **solver_container,
                                               CConfig *config, bool Output);

  /*!
   * \brief Compute a suitable under-relaxation parameter to limit the change in the solution variables over
   * a nonlinear iteration for stability.
   * \param[in] config - Definition of the particular problem.
   */
  void ComputeUnderRelaxationFactor(const CConfig *config) final;

  /*!
   * \brief Set the fluid solver nondimensionalization.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] config - Definition of the particular problem.
   */
  void SetNondimensionalization(CConfig *config, unsigned short iMesh);

  /*!
   * \brief Set all the conserved variables from the primitive vector..
   */
  void RecomputeConservativeVector(su2double *U, const su2double *V) const;

  /*!
   * \brief Check for unphysical points.
   * \return Boolean value of physical point
   */
  bool CheckNonPhys(const su2double *V) const;

  /*!
   * \brief Compute the pressure at the infinity.
   * \return Value of the pressure at the infinity.
   */
  inline CNEMOGas* GetFluidModel(void) const final { return FluidModel;}

  /*!
   * \brief Impose the far-field boundary condition using characteristics.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] conv_numerics - Description of the numerical method for convective terms.
   * \param[in] visc_numerics - Description of the numerical method for viscous terms.
   * \param[in] config - Definition of the particular problem.
   * \param[in] val_marker - Surface marker where the boundary condition is applied.
   */
  void BC_Far_Field(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics,
                    CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) override;

  /*!
   * \brief Impose the symmetry boundary condition using the residual.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] conv_numerics - Description of the numerical method for convective terms.
   * \param[in] visc_numerics - Description of the numerical method for viscous terms.
   * \param[in] config - Definition of the particular problem.
   * \param[in] val_marker - Surface marker where the boundary condition is applied.
   */
  void BC_Sym_Plane(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics,
                    CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) final;

  /*!
   * \brief Impose a subsonic inlet boundary condition.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] conv_numerics - Description of the numerical method for convective terms.
   * \param[in] visc_numerics - Description of the numerical method for viscous terms.
   * \param[in] config - Definition of the particular problem.
   * \param[in] val_marker - Surface marker where the boundary condition is applied.
   */
  void BC_Inlet(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics,
                CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) override;

  /*!
   * \brief Impose a supersonic inlet boundary condition.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] conv_numerics - Description of the numerical method for convective terms.
   * \param[in] visc_numerics - Description of the numerical method for viscous terms.
   * \param[in] config - Definition of the particular problem.
   * \param[in] val_marker - Surface marker where the boundary condition is applied.
   */
  void BC_Supersonic_Inlet(CGeometry *geometry, CSolver **solver_container,
                           CNumerics *conv_numerics, CNumerics *visc_numerics,
                           CConfig *config, unsigned short val_marker) override;
  /*!
   * \brief Impose the supersonic outlet boundary condition.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] solver - Description of the numerical method.
   * \param[in] config - Definition of the particular problem.
   * \param[in] val_marker - Surface marker where the boundary condition is applied.
   */
  void BC_Supersonic_Outlet(CGeometry *geometry, CSolver **solver_container,
                            CNumerics *conv_numerics, CNumerics *visc_numerics,
                            CConfig *config, unsigned short val_marker) override;
  /*!
   * \brief Impose the outlet boundary condition.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] conv_numerics - Description of the numerical method for convective terms.
   * \param[in] visc_numerics - Description of the numerical method for viscous terms.
   * \param[in] config - Definition of the particular problem.
   * \param[in] val_marker - Surface marker where the boundary condition is applied.
   */
  void BC_Outlet(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics,
                 CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) override;

  /*!
   * \brief Update the solution using an explicit Euler scheme.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] config - Definition of the particular problem.
   */
  void ExplicitEuler_Iteration(CGeometry *geometry, CSolver **solver_container, CConfig *config) final;

  /*!
   * \brief Update the solution using a general explicit RK scheme.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] config - Definition of the particular problem.
   * \param[in] iRKStep - Runge-Kutta step.
   */
  void ExplicitRK_Iteration(CGeometry *geometry, CSolver **solver_container,
                            CConfig *config, unsigned short iRKStep) final;

  /*!
   * \brief Update the solution using the classical RK4 scheme.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - Container vector with all the solutions.
   * \param[in] config - Definition of the particular problem.
   * \param[in] iRKStep - Runge-Kutta step.
   */
  void ClassicalRK4_Iteration(CGeometry *geometry, CSolver **solver_container,
                              CConfig *config, unsigned short iRKStep) final;

  /*!
   * \brief Dual time-stepping source term (device accumulator in Kokkos runs, else the base class).
   */
  void SetResidual_DualTime(CGeometry *geometry, CSolver **solver_container, CConfig *config, unsigned short iRKStep,
                            unsigned short iMesh, unsigned short RunTime_EqSystem) override;

  /*!
   * \brief Device-resident mode: point loop of AdaptCFLNumber on the device (Kokkos batch A).
   */
  bool AdaptCFLNumber_Device(unsigned short iMesh, bool reduceCFL, bool canIncrease, bool resetCFL,
                             su2double CFLFactorDecrease, su2double CFLFactorIncrease, su2double CFLMin,
                             su2double CFLMax, su2double MGFactor, bool stats, su2double& cflMin, su2double& cflMax,
                             su2double& cflSum) override;

  /*!
   * \brief Device-resident mode: SetDualTime_Solver state update on the device (Kokkos batch A).
   */
  bool SetDualTime_Device(su2double cfl) override;

  /*!
   * \brief Prepare an implicit iteration.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] config - Definition of the particular problem.
   */
  void PrepareImplicitIteration(CGeometry *geometry, CSolver**, CConfig *config) final;

  /*!
   * \brief End of CIntegration::Space_Integration: download the device residual accumulator (Kokkos Step 7).
   */
  void FinalizeSpaceIntegration(CGeometry *geometry, CConfig *config, unsigned short iMesh) override;

  /*!
   * \brief Solution_Old = Solution (on the device in device-resident mode).
   */
  void Set_OldSolution() override;

  /*!
   * \brief Device-resident mode: refresh the host copies read by host code (Kokkos Step 6).
   */
  void SyncHostData(CGeometry *geometry, HostSyncScope scope) override;

  /*!
   * \brief Squared norm of LinSysRes, reduced on the device in device-resident mode.
   */
  su2double LinSysResSquaredNorm() const override;

  /*!
   * \brief 4.6-MG2: the MG transfer operators between this level and coarse run on the device.
   */
  bool MG_DeviceTransfer(const CSolver *coarse) const override;

  /*!
   * \brief 4.6-MG2: one device MG transfer operator (CNEMOKokkosMG.cpp kernels).
   */
  void MG_DeviceOp(MGDeviceOp op, CSolver *coarse, CGeometry *geometry, CGeometry *geoCoarse, const CConfig *config,
                   su2double value) override;

  /*!
   * \brief Complete an implicit iteration.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] config - Definition of the particular problem.
   */
  void CompleteImplicitIteration(CGeometry *geometry, CSolver**, CConfig *config) final;

  /*!
   * \brief Print verification error to screen.
   * \param[in] config - Definition of the particular problem.
   */
  void PrintVerificationError(const CConfig* config) const final { }

  /*!
   * \brief Compute the Pressure sensor for NEMO schemes.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] config - Definition of the particular problem.
   */
  void SetPressureDiffusionSensor(CGeometry *geometry, CConfig *config);

};
