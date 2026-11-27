/*!
 * \file CNEMOKokkosData.hpp
 * \brief Device (Kokkos) mirror of the geometry and state used by the NEMO solvers.
 *
 * Ownership model (Steps 2-4 of NEMO_KOKKOS_PORT_PLAN.md):
 *  - Geometry group: uploaded once at construction, read-only on the device afterwards.
 *    The host CGeometry stays authoritative. TODO(mesh deformation / dynamic grids): call
 *    UploadGeometry() again after the host geometry changes.
 *  - State group (Solution, Solution_Old, Primitive, dPdU/dTdU/dTvedU, eves, Cvves, Gamma):
 *    the HOST CNEMOEulerVariable is authoritative between solver calls. The device copies are
 *    transient: every device kernel uploads what it reads and downloads what it writes, so the
 *    rest of the CPU pipeline (restart, BCs, output, host SetPrimVar callers) is untouched.
 *    Later steps will move authority to the device and shrink the transfers.
 *    Steps 3-4 (CNEMOKokkosStages.cpp) read the Primitive that SetPrimVar left on the device (no re-upload),
 *    keep gradients device-resident for the limiter, and download only domain rows of their outputs.
 *  - Flags / scratch (Cons2Prim status bits, first-attempt gas Density/Pressure): device-written,
 *    downloaded to host mirrors owned by this class.
 *  - Residual accumulator (Step 7: st.Res + st.JacVal): upwind, source and BC stages add into it on the device;
 *    exactly one of {host LinSysRes/Jacobian, device accumulator} is authoritative (AccumulatorToDevice/ToHost),
 *    downloaded once per space integration when the whole residual pipeline is on the device.
 *  - Device-resident mode (Step 6, EnableResident): when every stage of the pipeline is on the device, the device
 *    State becomes authoritative. Each array group (Group) carries a Residency {HOST, DEVICE, BOTH}: device kernels
 *    EnsureDevice() their inputs (upload only if the host copy is newer) and MarkDeviceWritten() their outputs (no
 *    download); host consumers call EnsureHost() (output / restart: everything, force and traction computations:
 *    EnsureHostBoundaryPrimitive()). LinSysRes / LinSysSol / Jacobian values are the device accumulator itself
 *    (st.Res and st.JacVal alias the CSysVector / CSysMatrix device storage) and are handed to the Kokkos linear
 *    solver without transfers. SU2_KOKKOS_NEMO_POISON=1 fills stale host copies with NaN (debug: any host read
 *    that misses an EnsureHost shows up in the results).
 *
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#pragma once

#ifdef HAVE_KOKKOS

#include <Kokkos_Core.hpp>

#include <string>
#include <vector>

#include "../../fluid/kokkos/NEMOGasKokkos.hpp"

class CGeometry;
class CConfig;
class CNEMOEulerVariable;
class CSU2TCLib;

/*!
 * \brief Step 10 (Task B): accumulation of the edge kernels (upwind, viscous) into the residual accumulator.
 *        ATOMIC: Kokkos atomics (order of the edge loop on Serial = CPU order; nondeterministic on GPUs).
 *        COLOR: edge colouring (colorSparsePattern on the edge -> node pattern, balanced): one launch per colour, the
 *        edges of a colour share no point, plain += / -= (deterministic on every backend, different summation order
 *        than the CPU: noise-floor exact). LEVEL: order-preserving colouring (colour of an edge = 1 + the highest colour
 *        of the earlier edges of its two points): plain writes AND the per-entry CPU (edge) order -> bitwise vs ATOMIC on
 *        Serial / vs the CPU (validation of the colour path; many colours, poor GPU occupancy).
 */
namespace nemo_edge {
enum class EdgeMode : int { ATOMIC = 0, COLOR = 1, LEVEL = 2 };
template <bool ATOMIC>
KOKKOS_FORCEINLINE_FUNCTION void Add(double* p, double v) {
  if (ATOMIC) Kokkos::atomic_add(p, v);
  else *p += v;
}
template <bool ATOMIC>
KOKKOS_FORCEINLINE_FUNCTION void Sub(double* p, double v) {
  if (ATOMIC) Kokkos::atomic_sub(p, v);
  else *p -= v;
}
}  // namespace nemo_edge

class CNEMOKokkosData {
 public:
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space = exec_space::memory_space;
  template <class T>
  using View1D = Kokkos::View<T*, mem_space>;
  template <class T>
  using View2D = Kokkos::View<T**, Kokkos::LayoutRight, mem_space>;
  template <class T>
  using View3D = Kokkos::View<T***, Kokkos::LayoutRight, mem_space>;

  /*--- Per-marker flags for the boundary-vertex loops (Steps 3-4). ---*/
  enum MarkerFlag : unsigned char {
    MARKER_GG_BOUNDARY = 1, /*!< Green-Gauss boundary face (not INTERNAL/NEARFIELD/PERIODIC). */
    MARKER_DT_BOUNDARY = 2, /*!< Time step / max eigenvalue boundary face (also not SYMMETRY/EULER_WALL). */
    MARKER_SYMMETRY = 4,    /*!< SYMMETRY_PLANE or EULER_WALL: gradient symmetry correction. */
  };

  /*!
   * \brief Read-only (after upload) geometry. Index types follow the host containers.
   */
  struct Geometry {
    unsigned long nPoint = 0, nPointDomain = 0, nEdge = 0;
    int nDim = 0, nMarker = 0;

    View2D<unsigned long> edge_nodes; /*!< (nEdge, 2) */
    View2D<double> edge_normal;       /*!< (nEdge, nDim) */
    View2D<double> coord;             /*!< (nPoint, nDim) */
    View1D<double> volume;            /*!< (nPoint) */
    View1D<unsigned short> nneighbor; /*!< (nPoint) geometry->nodes->GetnNeighbor */
    View1D<double> periodic_volume;   /*!< (nPoint) */

    View1D<unsigned long> point_ptr;  /*!< point -> neighbour CSR (nPoint+1) */
    View1D<unsigned long> point_idx;
    View1D<unsigned long> pedge_ptr;  /*!< point -> edge CSR (nPoint+1), same ordering as point_idx */
    View1D<long> pedge_idx;

    View1D<unsigned long> marker_ptr; /*!< marker -> vertex CSR (nMarker+1) */
    View1D<unsigned long> vtx_node;
    View2D<double> vtx_normal;        /*!< (nVertexTotal, nDim) */
    View1D<long> vtx_normal_neighbor;

    /*--- FV sparse pattern (fill 0), all nPoint rows (not only nPointDomain as in CSysMatrixKokkos). ---*/
    View1D<unsigned long> row_ptr;    /*!< (nPoint+1) */
    View1D<unsigned long> col_ind;    /*!< (nnz) */
    View1D<unsigned long> dia_ptr;    /*!< (nPoint) */
    View2D<unsigned long> edge_nnz;   /*!< (nEdge, 2): non-zero index of (i,j) and (j,i) */
    bool has_pattern = false;
    unsigned long nnz = 0;

    /*--- Step 5: point -> (edge << 1 | side) CSR, edges in increasing order (CPU edge-loop order per point;
     *    side 0: the point is node 0 (iPoint) of the edge, side 1: node 1 (jPoint)). ---*/
    View1D<unsigned long> pside_ptr;  /*!< (nPoint+1) */
    View1D<unsigned long> pside_idx;  /*!< (2*nEdge) */

    /*--- Boundary info for gather-based point kernels (UploadBoundaryInfo). ---*/
    View1D<unsigned long> pvtx_ptr;   /*!< point -> flattened vertex CSR (nPoint+1), marker-major order */
    View1D<unsigned long> pvtx_idx;
    View1D<int> vtx_marker;           /*!< (nVertexTotal) marker of each flattened vertex */
    View1D<unsigned char> marker_flags; /*!< (nMarker) MarkerFlag bits */
    View2D<double> vtx_sym_normal;    /*!< (nVertexTotal, nDim) unit normal used by correctGradientsSymmetry */
    bool domain_first = true;         /*!< GetDomain(i) == (i < nPointDomain), required by the device stages */

    /*--- 6b.a dynamic grid (rigid motion / deformation, config->GetDynamic_Grid()): allocated and filled by
     *    RefreshDynamicGeometry; the time step, max eigenvalue and dual-time kernels add the grid-velocity / GCL
     *    terms of CFVMFlowSolverBase when `dynamic` is set. ---*/
    View2D<double> grid_vel;          /*!< (nPoint, nDim) geometry->nodes->GetGridVel */
    View1D<double> volume_nm1;        /*!< (nPoint) geometry->nodes->GetVolume_nM1 */
    bool dynamic = false;
  };

  /*!
   * \brief Point state, same row-major layout as the host C2DContainers.
   */
  struct State {
    View2D<double> Solution, Solution_Old; /*!< (nPoint, nVar) */
    View2D<double> Primitive;              /*!< (nPoint, nPrimVar) */
    View2D<double> dPdU, dTdU, dTvedU;     /*!< (nPoint, nVar), allocated only if implicit */
    View2D<double> eves, Cvves;            /*!< (nPoint, nSpecies) */
    View1D<double> Gamma;                  /*!< (nPoint) */
    View1D<unsigned char> Status;          /*!< (nPoint) su2kokkos::nemo::NEMOCons2PrimStatus bits */
    View2D<double> FirstDP;                /*!< (nPoint, 2) gas Density/Pressure of a full-path first attempt */

    /*--- Steps 3-4 (allocated on first use). ---*/
    View3D<double> GradPrim;               /*!< (nPoint, nPrimVarGrad, nDim) Gradient_Primitive */
    View3D<double> GradRec;                /*!< Gradient_Reconstruction if it is a separate container, else == GradPrim */
    View3D<double> Rmatrix;                /*!< (nPoint, nDim, nDim) least-squares R (device only, not downloaded) */
    View2D<double> Limiter, PrimMax, PrimMin; /*!< (nPoint, nPrimVarGrad) */
    View1D<double> LimiterEps2;               /*!< (nPrimVarGrad) limiter eps^2 per variable */
    View1D<double> LimiterGeo;                /*!< (nPointDomain) SHARP_EDGES / WALL_DISTANCE geometric factor */
    View1D<double> LocalCFL, MaxLambdaInv, MaxLambdaVisc, DeltaTime, Lambda; /*!< (nPoint) */
    View1D<double> LamVisc, ThermalCond, ThermalCondVe;                      /*!< (nPoint), NS only */

    /*--- Step 5 (allocated on first use). ---*/
    View2D<double> Res;                    /*!< (nPoint, nVar) upwind residual (LinSysRes contribution) */
    View1D<double> JacVal;                 /*!< (nnz*nVar*nVar) upwind Jacobian contribution, CSysMatrix layout */
    View1D<unsigned char> NonPhys;         /*!< (nPoint) CVariable::Non_Physical (sticky MUSCL first-order flag) */
    View1D<unsigned short> NonPhysCount;   /*!< (nPoint) CVariable::Non_Physical_Counter */
    View2D<unsigned char> EdgeNonPhys;     /*!< (nEdge, 2) CheckNonPhys of the reconstructed states, then the
                                                sticky flag the CPU reads back (GetNon_Physical) for that edge */
    View1D<double> Sensor;                 /*!< (nPoint) pressure diffusion sensor (AUSM+M) */
    View1D<double> LastRec;                /*!< (nPrimVar) last reconstructed state passed to ComputeConsistentExtrapolation */

    /*--- Step 7 (allocated on first use). Res / JacVal above are the device residual accumulator. ---*/
    View2D<double> Charac;                 /*!< (nVertexTotal, nPrimVar) CharacPrimVar written by the device BC_Outlet */
    View2D<double> SlipVel;                /*!< (nVertexTotal, nDim) BC_Smoluchowski_Maxwell slip velocity (host mode) */
    View1D<double> BState;                 /*!< far-field / supersonic-inlet state: V | eve | dPdU | Gamma */

    /*--- Step 6 (device-resident mode). Res / JacVal alias LinSysRes / Jacobian device storage, Sol LinSysSol. ---*/
    View2D<double> Sol;                    /*!< (nPoint, nVar) LinSysSol (device storage of the CSysVector) */
    View2D<double> TruncErr;               /*!< (nPoint, nVar) Res_TruncError */
    View1D<double> UnderRelax;             /*!< (nPoint) UnderRelaxation */
    View1D<unsigned long> bndPoints;       /*!< unique marker-vertex points (EnsureHostBoundaryPrimitive) */
    View2D<double> bndStage;               /*!< (nBndPoints, nPrimVar) staging */

    /*--- Step 8 (NS). LamVisc / ThermalCond / ThermalCondVe above hold the device transport properties. ---*/
    View2D<double> DiffCoeff, Enthalpys;   /*!< (nPoint, nSpecies) CNEMONSVariable DiffusionCoeff / Enthalpys */
    View2D<double> TransDP;                /*!< (nPoint, 2) gas Density / Pressure the transport of the point reads */
    View1D<double> BStateVisc;             /*!< far-field viscous state: Cvve | dTdU | dTvedU | gradient (node_infty) */
    View1D<unsigned long> viscWallPoints;  /*!< unique points of the viscous-wall markers (forces / heat flux) */
    View2D<double> viscWallStage;          /*!< (n, nPrimVarGrad*nDim + 3 + 2*nSpecies) staging */

    /*--- Batch A (allocated on first use). Sensor above: device SetPressureDiffusionSensor (AUSM+M). ---*/
    View2D<double> AuxVar;                 /*!< (nPoint, 3) axisymmetric auxiliary variables (CVariable::AuxVar) */
    View3D<double> AuxGrad;                /*!< (nPoint, 3, nDim) their gradient (CVariable::Grad_AuxVar) */
    View2D<double> SolTimeN, SolTimeN1;    /*!< (nPoint, nVar) Solution_time_n / Solution_time_n1 (dual time) */
    View2D<double> SolutionNew;            /*!< (nPoint, nVar) CFlowVariable::Solution_New (classical RK4) */
  };

  /*! \brief Edge colouring of the edge kernels (Step 10, Task B; env SU2_KOKKOS_NEMO_EDGE=atomic|color|level,
   *         group size env SU2_KOKKOS_NEMO_EDGE_GROUP, default 1 on device backends, EDGE_COLORING_GROUP_SIZE on host
   *         backends). Colour c = edges[colorPtr[c], colorPtr[c+1]), increasing edge index; consecutive chunks of
   *         groupSize edges are one work item (processed sequentially, in edge order). */
  struct EdgeColoring {
    nemo_edge::EdgeMode mode = nemo_edge::EdgeMode::ATOMIC;
    unsigned long groupSize = 1;
    std::vector<unsigned long> colorPtr;
    View1D<unsigned long> edges;
    unsigned long minSize = 0, maxSize = 0;
    double efficiency = 1.0, efficiencyGPU = 1.0; /*!< coloringEfficiency at exec concurrency / at 2^16 threads */
    double buildSeconds = 0.0;
    bool Atomic() const { return mode == nemo_edge::EdgeMode::ATOMIC; }
    unsigned long NumColors() const { return colorPtr.empty() ? 0 : colorPtr.size() - 1; }
  };
  EdgeColoring edgeColoring;
  /*! \brief Build the edge colouring (constructor, if the mode is not ATOMIC). */
  void BuildEdgeColoring(CGeometry* geometry, const CConfig* config);
  std::string EdgeColoringReport() const;

  Geometry geo;
  State st;
  su2kokkos::nemo::NEMOGasTables gas;

  /*--- Host mirrors of the device-only flag/scratch arrays (valid after DownloadState()). ---*/
  std::vector<unsigned char> h_Status;
  std::vector<double> h_FirstDP;

  CNEMOKokkosData(CGeometry* geometry, const CConfig* config, CNEMOEulerVariable* nodes, CSU2TCLib* gasModel,
                  unsigned short iMesh);
  ~CNEMOKokkosData();
  CNEMOKokkosData(const CNEMOKokkosData&) = delete;
  CNEMOKokkosData& operator=(const CNEMOKokkosData&) = delete;

  /*! \brief (Re)upload all geometry arrays (constructor; TODO: call after mesh deformation). */
  void UploadGeometry(CGeometry* geometry);

  /*!
   * \brief 6b.a: re-upload the metric arrays that move with the grid (edge / vertex normals, coordinates, volumes,
   *        periodic volumes, symmetry unit normals, grid velocity, Volume_nM1) from the host geometry, which the
   *        host grid movement / deformation updated. Topology (CSR, pattern, colouring) is unchanged.
   *        Called once per (time iteration, outer iteration) by the solver; sets geo.dynamic.
   */
  void RefreshDynamicGeometry(CGeometry* geometry);
  unsigned long dynGeoRefreshes = 0; /*!< number of RefreshDynamicGeometry calls (perf report) */

  /*! \brief HtoD: Solution, Solution_Old (inputs of Cons2Prim). */
  void UploadConservative();
  /*! \brief HtoD: Primitive and secondaries (Cons2Prim reads Tve_old and has in/out quirks). */
  void UploadPrimitive();
  /*! \brief DtoH: Solution (clipping + rollback), Primitive, secondaries, Gamma, flags/scratch. */
  void DownloadState();

  /*!
   * \brief Upload state, run the device SetPrimVar (Cons2Prim + rollback + Gamma), download.
   *        Velocity2, transport (NS) and gas-state replay are left to the caller (host).
   * \return Number of non-physical points (first attempt), as the CPU loop counts them.
   */
  unsigned long SetPrimVar();

  /*! \brief Host gas-state replay for one point after SetPrimVar() (see ReplayHostGasState). */
  void ReplayGasState(unsigned long iPoint, bool full) const;

  /*!
   * \brief Step 8: NS transport properties are computed by the device SetPrimVar (CNEMONSVariable::SetTransportProperties
   *        on the device: diffusion coefficients, viscosity, k_tr, k_ve, species enthalpies, Primitive[LAM_VISC]), for
   *        the WILKE / GUPTAYOS / SUTHERLAND models of a viscous CSU2TCLib (SU2_KOKKOS_NEMO_HOST=transport: host).
   */
  bool DeviceTransport() const { return deviceTransport; }
  /*! \brief Device transport arrays == host ones (computed by the last SetPrimVar), read by dt / viscous / walls. */
  bool TransportOnDevice() const { return transportDeviceValid; }
  /*! \brief Host replay of the gas state + transport members after the last point (CPU loop end state). */
  void ReplayTransportLastPoint();

  /*! \brief Round-trip check: device copies == host data (bitwise). Returns number of mismatches. */
  unsigned long SelfCheck(CGeometry* geometry) const;

  /*!
   * \brief Debug A/B verification (env SU2_KOKKOS_C2P_VERIFY=1): the solver snapshots the host state,
   *        runs the device path, swaps the device results into the snapshot (restoring the pre-call host
   *        state), runs the unchanged CPU loop, then compares. The run continues with the CPU results.
   */
  static bool VerifyEnabled();
  void Snapshot();
  void SwapSnapshot();
  void CompareSnapshot(unsigned long nonPhysDevice, unsigned long nonPhysHost);

  /*====================== Steps 3-4: gradients, limiter, time step ======================*/

  /*! \brief Per-marker flags, per-point vertex lists and symmetry unit normals (needs config). */
  void UploadBoundaryInfo(CGeometry* geometry, const CConfig* config);

  /*! \brief The device Primitive is what the last SetPrimVar() produced (== host Primitive for the first
   *         nPrimVarGrad columns). Cleared when the host overwrites Primitive behind our back (A/B verify). */
  bool PrimitiveOnDevice() const { return primDeviceValid; }
  void InvalidatePrimitive() {
    primDeviceValid = false;
    secDeviceValid = false;
    consDeviceValid = false;
    transportDeviceValid = false;
  }
  /*! \brief HtoD of the host Primitive only (no secondaries), marks it valid. */
  void UploadPrimitiveOnly();

  enum class GradMethod { GREEN_GAUSS, LEAST_SQUARES, WEIGHTED_LEAST_SQUARES };
  /*!
   * \brief Device gradient of the primitives [0, nPrimVarGrad) (computeGradientsGreenGauss incl. boundary faces /
   *        computeGradientsLeastSquares without periodicity; symmetry corrections only if deviceSymmetry),
   *        downloaded into rows [0, nPointDomain) of hostGrad. Symmetry (if not on device) and comms: caller.
   * \param[in] massFractions - NEMO-NS GG: differentiate Y_s = rho_s/rho instead of rho_s (Primitive_Aux).
   * \param[in] reconstruction - Target the device view mirroring Gradient_Reconstruction.
   */
  void ComputeGradient(GradMethod method, bool massFractions, bool reconstruction, int idxVel, bool deviceSymmetry,
                       double* hostGrad);

  /*!
   * \brief Symmetry-plane / Euler-wall gradient correction placement. Default: host (the CPU
   *        correctGradientsSymmetry itself, after the download, then UploadSymmetryRows): its fast-math
   *        realisation is vectorisation-dependent, so a device copy is only ULP-close. SU2_KOKKOS_NEMO_DEVICE_SYM=1
   *        runs the device pass instead (kept for GPU builds / Step 10).
   */
  static bool DeviceSymmetryEnabled();
  bool HasSymmetryPoints() const { return !h_symPoints.empty(); }
  /*! \brief HtoD of the gradient rows of the (domain) points on symmetry/Euler markers. */
  void UploadSymmetryRows(bool reconstruction, const double* hostGrad);
  /*! \brief DtoH of the same rows (device-resident mode: the host correction needs only them). */
  void DownloadSymmetryRows(bool reconstruction, double* hostGrad);

  /*!
   * \brief Device slope limiter (computeLimiters_impl without periodicity) for kinds BARTH_JESPERSEN,
   *        VENKATAKRISHNAN (eps2 given) and VENKATAKRISHNAN_WANG (eps2 per variable from a device min/max
   *        reduction + host MPI reduction, computed here). Reads the device reconstruction gradient.
   *        Downloads rows [0, nPointDomain) of limiter, primMax, primMin.
   * \param[in] epsVenkat - eps2 (BJ/Venkat), or K (Venkat-Wang).
   */
  void ComputeLimiter(int kind, double umusclKappa, double epsVenkat, double* hostLim, double* hostMax,
                      double* hostMin, bool periodic = false);
  /*!
   * \brief 4.6-PeriodicLim: computeLimiters_impl "if (periodic)" initialisation, fieldMax = fieldMin = field for all
   *        nPoint rows (device), before the host PERIODIC_LIM_PRIM_1 comms on staged rows. ComputeLimiter(periodic=true)
   *        then skips its per-point initialisation, as the CPU main loop does.
   */
  void LimiterPeriodicInit();
  /*!
   * \brief 4.6-Limiters: per-point geometric factor of SHARP_EDGES / WALL_DISTANCE (CLimiterDetails::geometricFactor,
   *        evaluated on the host from the geometry distances, uploaded once: static mesh). Used by ComputeLimiter
   *        when kind is SHARP_EDGES or WALL_DISTANCE.
   */
  void SetLimiterGeoFactor(const double* hostFactor);
  bool LimiterGeoFactorValid() const { return limGeoValid; }
  /*! \brief Re-upload the host reconstruction gradient (when it was computed on the host or for A/B isolation). */
  void UploadGradient(bool reconstruction, const double* hostGrad);
  bool GradientOnDevice(bool reconstruction) const { return reconstruction ? gradRecValid : gradPrimValid; }

  /*!
   * \brief Device part of SetTime_Step_impl: edge + boundary-vertex spectral radii (gather), local dt,
   *        min/max dt reduction (domain points with Vol != 0, rank-local). Downloads Max_Lambda_Inv,
   *        Max_Lambda_Visc (viscous), Delta_Time for [0, nPointDomain). Uploads LocalCFL (+ NS transport).
   */
  void ComputeTimeStep(bool viscous, double maxDeltaTime, double& minDt, double& maxDt);
  /*! \brief Device part of SetMax_Eigenvalue_impl, downloads Lambda for [0, nPointDomain). */
  void ComputeMaxEigenvalue();

  /*!
   * \brief Stage A/B verification (env SU2_KOKKOS_STAGE_VERIFY=1): each device stage re-uploads its host inputs
   *        (isolation; also checks that the resident device Primitive equals the host one), runs, keeps a copy of
   *        its results, reruns the unchanged host code and compares. The run continues with the host results.
   */
  static bool StageVerifyEnabled();
  enum Stage { ST_GRADIENT = 0, ST_LIMITER, ST_TIMESTEP, ST_MAXEIG, ST_UPWIND, ST_SOURCE, ST_BC, ST_RESSYNC, ST_VISCOUS,
               ST_SENSOR, ST_AUXGRAD, ST_DUALTIME, ST_CFL, ST_CENTERED, N_STAGES };
  /*! \brief Compare host (CPU) result vs a copy of the device result over n entries (row-major, ncol columns,
   *         relative difference normalised per column); accumulates per stage. */
  void StageCompare(Stage s, const char* name, const double* cpu, const double* dev, size_t n, size_t ncol = 1);
  void StageCompareScalar(Stage s, const char* name, double cpu, double dev);

  /*====================== Step 5: upwind residual (AUSM family) ======================*/

  enum class UpwindScheme { AUSM = 0, AUSMPLUSM = 1, ROE = 2, MSW = 3, AUSMPLUSUP2 = 4 };
  struct UpwindParams {
    UpwindScheme scheme = UpwindScheme::AUSM;
    bool implicit = false, muscl = false, limiter = false, vanAlbada = false;
    double kappa = 0.0, musclRamp = 1.0, Minf = 0.0;
    double mswAlpha = 5.0; /*!< config MSW_ALPHA (MSW only) */
    int nElNumerics = 0; /*!< CNEMONumerics::nEl = config IONIZATION (may differ from the gas model's, e.g. AIR-7) */
  };
  /*! \brief Host results of ComputeUpwindResidual (besides the accumulated residual/Jacobian). */
  struct UpwindResult {
    unsigned long nonPhysical = 0;    /*!< counter_local of the CPU loop (sum of the sticky flags read back) */
    bool haveLastRec = false;         /*!< a reconstructed state was passed to ComputeConsistentExtrapolation */
    std::vector<double> lastRec;      /*!< the last such state (CPU order), for the host gas-state replay */
  };

  /*!
   * \brief Device CNEMOEulerSolver::Upwind_Residual for AUSM / AUSM+M (CUpwAUSM_SLAU_Base_NEMO::ComputeResidual +
   *        ComputeJacobian): MUSCL reconstruction with the device gradient/limiter (or van Albada), CheckNonPhys, the
   *        sticky Non_Physical/Non_Physical_Counter update in CPU edge order, consistent extrapolation (Eve, Gamma),
   *        flux, approximate Jacobian, NaN check and atomic accumulation into the device residual accumulator
   *        (Res, JacVal: nnz*nVar^2, CSysMatrix layout), which the caller must have made authoritative with
   *        AccumulatorToDevice(). Non_Physical(+counter) DtoH.
   *        Uploads Non_Physical(+counter) (host authoritative), the AUSM+M sensor, halo rows of gradient/limiter.
   * \param[in] uploadAll - Re-upload Primitive+secondaries, gradient and limiter from the host (A/B isolation).
   */
  UpwindResult ComputeUpwindResidual(const UpwindParams& p, bool uploadAll);

  /*====================== 4.4-LAX: centered residual (LAX-FRIEDRICH) ======================*/

  /*!
   * \brief Device CNEMOEulerSolver::Centered_Residual for LAX-FRIEDRICH (CCentLax_NEMO::ComputeResidual incl. the
   *        implicit Jacobian), NaN check, accumulation into the device residual accumulator (caller: AccumulatorToDevice).
   *        Uploads only the halo rows of Lambda (domain rows resident from ComputeMaxEigenvalue).
   * \return Mean primitive state of the last edge (nPrimVar), for the host gas-state replay (empty if no edges).
   */
  std::vector<double> ComputeCenteredResidual(bool implicit, double kappa0, bool uploadAll);

  /*====================== Step 7: residual accumulator, source terms, boundary conditions ======================*/

  /*!
   * \brief Device residual accumulator (st.Res + st.JacVal). Exactly one copy is authoritative at any time:
   *        - host LinSysRes / Jacobian (initially, after every host stage, after AccumulatorToHost), or
   *        - the device accumulator (after AccumulatorToDevice, until AccumulatorToHost).
   *        Device stages add into the device copy statement by statement in the CPU order, so moving the whole
   *        accumulator between host and device (exact copies) keeps every entry bitwise equal to the CPU sums.
   * \param[in] hostRes/hostJac - host LinSysRes / CSysMatrix values (hostJac: nnz*nVar^2, nullptr if explicit).
   *        Uploaded unless the host is known to be zero (MarkHostAccumulatorZero), then the device copy is zeroed.
   */
  void AccumulatorToDevice(bool implicit, const double* hostRes, const double* hostJac);
  /*! \brief DtoH (overwrite) of the device accumulator into the host arrays, if the device copy is authoritative. */
  void AccumulatorToHost(double* hostRes, double* hostJac);
  bool AccumulatorOnDevice() const { return accOnDevice; }
  /*! \brief The host LinSysRes / Jacobian were just zeroed (CommonPreprocessing): the next device stage zero-fills
   *         instead of uploading. A pending device accumulator (should not happen) is discarded and counted. */
  void MarkHostAccumulatorZero();
  bool HostAccumulatorZero() const { return !accOnDevice && hostAccZero; }
  /*! \brief A host stage is going to modify the host accumulator. */
  void HostAccumulatorTouched() { hostAccZero = false; }

  struct SourceParams {
    bool implicit = false, frozen = false, monoatomic = false, vtLimiting = false;
    /*--- Batch A: CSource_NEMO::ComputeAxisymmetric (2D), viscous part with the device AuxVar gradient. ---*/
    bool axisymmetric = false, axiViscous = false;
    int nHeavyNumerics = 0;   /*!< CNEMONumerics::nHeavy (config IONIZATION) */
    double prandtlTurb = 0.0; /*!< CNumerics::Prandtl_Turb */
    double RuSI = 0.0;        /*!< UNIVERSAL_GAS_CONSTANT */
  };
  struct SourceResult {
    unsigned long eChm = 0, eVib = 0, eAxi = 0; /*!< NaN counters of the CPU loop (chemistry, vib. relaxation, axisym.) */
  };
  /*!
   * \brief Device CNEMOEulerSolver::Source_Residual without axisymmetry: per domain point, fused
   *        CSource_NEMO::ComputeChemistry (ComputeNetProductionRates + ChemistryJacobian) and ComputeVibRelaxation
   *        (ComputeEveSourceTerm + GetEveSourceTermJacobian, VT residual limiting), CheckResidualNaNs, subtraction
   *        from the device accumulator (residual rows and diagonal Jacobian blocks, point-unique, no atomics).
   * \param[in] uploadInputs - Re-upload Primitive + secondaries from the host (A/B isolation).
   */
  SourceResult ComputeSourceResidual(const SourceParams& p, bool uploadInputs);

  enum class BCKind { SYMMETRY = 0, FAR_FIELD, OUTLET, SUPERSONIC_INLET, SUPERSONIC_OUTLET, ISOTHERMAL, HEAT_FLUX,
                      SMOLUCHOWSKI_MAXWELL };
  struct BCParams {
    BCKind kind = BCKind::SYMMETRY;
    int marker = 0;
    UpwindParams upw;        /*!< scheme, implicit, Minf, nElNumerics of the boundary numerics (same type as CONV_TERM) */
    double P_Exit = 0.0;     /*!< OUTLET: non-dimensional back pressure */
    bool downloadCharac = false; /*!< OUTLET: copy the marker's CharacPrimVar rows to the host */
    /*--- Step 8. ---*/
    bool viscous = false;    /*!< FAR_FIELD: viscous contribution (CAvgGrad_NEMO with node_infty, reflected point) */
    double Twall = 0.0;      /*!< ISOTHERMAL: config Isothermal_Temperature (as the CPU, not non-dimensionalised) */
    bool catalytic = false, supercatalytic = false; /*!< ISOTHERMAL: CATALYTIC_WALL marker, SUPERCATALYTIC_WALL */
    /*--- 6b.b coupling: ISOTHERMAL with MARKER_PYTHON_CUSTOM: host pointer to the marker's nVertex per-vertex wall
     *    temperatures (geometry CustomBoundaryTemperature, iVertex order); nullptr -> uniform Twall. Only these
     *    nVertex doubles move host->device per BC call. ---*/
    const double* customTwall = nullptr;
    double Yst[su2kokkos::nemo::MAXNS] = {0.0}; /*!< ISOTHERMAL supercatalytic: wall composition */
    double catEff = 0.0;     /*!< ISOTHERMAL partially catalytic: CATALYTIC_EFFICIENCY */
    double heatFlux = 0.0;   /*!< HEAT_FLUX: non-dimensional (integrated: per area) wall heat flux */
    /*--- 4.5 SMOLUCHOWSKI_MAXWELL (Twall = Isothermal_Temperature): FluidModel->ComputeGasConstant() and Prandtl_Lam,
     *    evaluated on the host (the gas model's state does not change inside the CPU marker loop). ---*/
    double gasConstant = 0.0, prandtlLam = 0.0;
    bool downloadSlip = false; /*!< SMOLUCHOWSKI_MAXWELL, host-authoritative: copy the slip velocities to the host */
    /*--- Batch A: SUPERSONIC_INLET / SUPERSONIC_OUTLET with AUSM+M read the Gamma / Sensor the shared boundary
     *    numerics kept from the last BC that set them (CPU semantics): SUPERSONIC_INLET Gamma_i, Gamma_j and Sensor,
     *    SUPERSONIC_OUTLET Sensor only (it sets Gamma itself). ---*/
    bool staleNumerics = false;
    double staleGammaI = 0.0, staleGammaJ = 0.0, staleSensor = 0.0;
  };
  /*! \brief Far-field (node_infty) or supersonic-inlet state, uploaded only when it changes (host cache). */
  struct BoundaryState {
    const double *V = nullptr, *eve = nullptr, *dPdU = nullptr;
    double Gamma = 0.0;
    /*--- Roe boundary numerics (LAX / ROE): node_infty / node_inlet Solution (nVar). ---*/
    const double* U = nullptr;
    /*--- FAR_FIELD viscous part: node_infty Cvve, dTdU, dTvedU (implicit), gradient (nPrimVarGrad x nDim). ---*/
    const double *Cvve = nullptr, *dTdU = nullptr, *dTvedU = nullptr, *grad = nullptr;
  };
  struct BCResult {
    long lastSubsonic = -1;          /*!< OUTLET: last domain vertex (CPU order) on the subsonic branch, -1 if none */
    std::vector<double> lastVOutlet; /*!< OUTLET: its V_outlet (host gas-state replay) */
    std::vector<double> charac;      /*!< OUTLET with downloadCharac: (nVertex, nPrimVar) */
    std::vector<double> slipVel;     /*!< SMOLUCHOWSKI_MAXWELL with downloadSlip: (nVertex, nDim) relaxed slip velocity */
  };
  /*!
   * \brief Device boundary conditions of one marker (vertex kernel over the flattened vertex arrays, domain vertices
   *        only, accumulation into the device accumulator in the CPU statement order):
   *        SYMMETRY (CNEMOEulerSolver::BC_Sym_Plane, also EULER_WALL), FAR_FIELD (inviscid part of BC_Far_Field),
   *        OUTLET (BC_Outlet, no gravity / dynamic grid), SUPERSONIC_INLET / SUPERSONIC_OUTLET (AUSM only: the CPU
   *        leaves Gamma/Sensor of the boundary numerics stale there, which AUSM+M would read).
   *        The AUSM flux/Jacobian is the Step-5 device function.
   * \param[in] bstate - FAR_FIELD / OUTLET: node_infty; SUPERSONIC_INLET: node_inlet; else unused.
   * \param[in] uploadInputs - Re-upload Solution, Primitive + secondaries, sensor from the host (A/B isolation).
   */
  BCResult ComputeBCResidual(const BCParams& p, const BoundaryState& bstate, bool uploadInputs);

  /*====================== Step 8: viscous residual (NS) ======================*/

  struct ViscousParams {
    bool implicit = false;
    int nElNumerics = 0; /*!< CNEMONumerics::nEl (config IONIZATION) */
  };
  /*!
   * \brief Device CNEMONSSolver::Viscous_Residual (CAvgGradCorrected_NEMO on every edge, CheckResidualNaNs, atomic
   *        LinSysRes Subtract/AddBlock + Jacobian UpdateBlocks into the device accumulator, edge order on Serial).
   *        Reads the device Primitive, secondaries, Gradient_Primitive, transport properties.
   * \param[in] uploadInputs - Re-upload Primitive + secondaries, gradient and transport from the host (A/B isolation).
   */
  void ComputeViscousResidual(const ViscousParams& p, bool uploadInputs);
  /*! \brief Re-upload the host NS transport arrays (host transport / A/B isolation). */
  void UploadTransport();
  /*! \brief Points of the viscous-wall markers whose gradient / transport rows Friction_Forces reads (resident). */
  void SetViscousWallMarkers(const std::vector<int>& markers);
  /*! \brief The AUSM+M sensor on the device is stale (set false by SetPrimVar, true when uploaded). */
  bool SensorOnDevice() const { return sensorValid; }
  /*! \brief Copy of the host Non_Physical / Non_Physical_Counter (A/B snapshot), and restore. */
  void GetNonPhysical(std::vector<double>& flag, std::vector<double>& count) const;
  void SetNonPhysical(const std::vector<double>& flag, const std::vector<double>& count);

  /*====================== Step 6: device-resident state, implicit / explicit update ======================*/

  /*! \brief Host-array groups with a residency state (device-resident mode only). */
  enum Group : int {
    G_SOLUTION = 0, /*!< Solution */
    G_SOLUTION_OLD, /*!< Solution_Old */
    G_PRIMITIVE,    /*!< Primitive, dPdU/dTdU/dTvedU, eves, Cvves, Gamma (+ host Velocity2 recomputed on download) */
    G_GRADIENT,     /*!< Gradient_Primitive (and Gradient_Reconstruction if separate), domain rows */
    G_LIMITER,      /*!< Limiter_Primitive, Solution_Max, Solution_Min, domain rows */
    G_TIMESTEP,     /*!< Delta_Time, Max_Lambda_Inv (+ Max_Lambda_Visc), domain rows */
    G_LOCALCFL,     /*!< LocalCFL */
    G_NONPHYS,      /*!< Non_Physical, Non_Physical_Counter */
    G_UNDERRELAX,   /*!< UnderRelaxation */
    G_TRUNCERR,     /*!< Res_TruncError */
    G_LINSYS,       /*!< LinSysRes and LinSysSol (CSysVector host copies) */
    G_SENSOR,       /*!< Sensor (AUSM+M pressure diffusion sensor), batch A */
    G_AUXVAR,       /*!< AuxVar + Grad_AuxVar (axisymmetric viscous source), batch A */
    G_TIMEN,        /*!< Solution_time_n, Solution_time_n1 (dual time stepping), batch A */
    N_GROUPS
  };
  enum class Residency : unsigned char { HOST, DEVICE, BOTH };

  /*!
   * \brief Switch to device-resident mode: st.Res / st.Sol / st.JacVal alias the device storage of LinSysRes /
   *        LinSysSol / the Jacobian values (jacValues may be nullptr: explicit). Called once, at construction.
   */
  void EnableResident(double* linSysRes, double* linSysSol, double* jacValues, double* hostRes, double* hostSol);
  bool Resident() const { return resident; }
  /*! \brief First device call of the run: every group is current on the host (restart / initialisation were done
   *         there after construction), the device copies are refreshed on first use. */
  void ActivateResident();
  bool ResidentActive() const { return resident && residentActive; }
  Residency GetResidency(Group g) const { return residency[g]; }
  void EnsureHost(Group g);
  void EnsureDevice(Group g);
  void MarkDeviceWritten(Group g);
  void MarkHostWritten(Group g) { if (resident) residency[g] = Residency::HOST; }
  /*! \brief DtoH of the Primitive rows of every marker-vertex point (force / traction computations), and (NS) of the
   *         gradient + transport rows of the viscous-wall points (Friction_Forces: shear stress, heat flux). */
  void EnsureHostBoundaryPrimitive();
  void EnsureHostAll();
  /*! \brief One device Gamma value (host replay of BC_Outlet's solver Gamma in device-resident mode). */
  double DeviceGamma(unsigned long iPoint);

  /*! \brief Per-variable residual RMS (sum of squares) / max / argmax, CPU ResidualReductions_PerThread semantics. */
  struct ResidualReduction {
    double rms[su2kokkos::nemo::MAXNVAR];
    double max[su2kokkos::nemo::MAXNVAR];
    unsigned long idx[su2kokkos::nemo::MAXNVAR];
  };
  /*!
   * \brief Device PrepareImplicitIteration_impl (no diagonal preconditioner): diagonal += Vol/dt (or SetVal2Diag 1
   *        if dt == 0), RHS = -(R + truncation error), LinSysSol = 0, halo rows of both zeroed (the host loop of
   *        ImplicitEuler_Iteration), residual reduction (ordered loop on Serial: bitwise with the CPU).
   */
  void PrepareImplicit(bool implicit, ResidualReduction& red);
  /*! \brief Device Explicit_Iteration_impl<EULER_EXPLICIT>: U -= (R + truncErr) dt/Vol, residual reduction. */
  void ExplicitEulerUpdate(ResidualReduction& red);
  /*! \brief Device ComputeUnderRelaxationFactor (NEMO) + AddSolution(UR * LinSysSol). */
  void CompleteImplicit(double allowableRatio);
  /*! \brief Solution_Old = Solution on the device (classical RK4: also Solution_New = Solution). */
  void SetOldSolution();
  /*! \brief Squared norm of the first nElmDomain entries of the device residual (CSysVector::squaredNorm order). */
  double ResidualSquaredNorm(unsigned long nElmDomain) const;
  /*! \brief Time-stepping branch of SetTime_Step: LocalCFL = cfl, Delta_Time = dt for the domain points. */
  void TimeSteppingFill(double cfl, double dt);
  /*! \brief Last point's Primitive row + Cons2Prim status / first-attempt Density, Pressure (host gas-state replay). */
  void ReplayGasStateLastPoint(bool full);

  /*====================== Batch A: sensor, CFL adaptation, axisymmetry, RK / dual time ======================*/

  /*!
   * \brief Device CNEMOEulerSolver::SetPressureDiffusionSensor for the domain points (neighbour order of the point CSR,
   *        std::min semantics) on the device Primitive. Halo rows: HaloExchange(SENSOR) by the caller.
   * \param[out] hostOut - If not null: the domain rows are copied there (A/B), else (resident) G_SENSOR is marked.
   */
  void ComputePressureSensor(double* hostOut);
  /*! \brief One device Sensor value (stale boundary-numerics replay in device-resident mode). */
  double DeviceSensor(unsigned long iPoint);

  /*! \brief Inputs of the point loop of CSolver::AdaptCFLNumber (host flags and parameters). */
  struct CFLAdaptParams {
    bool reduceCFL = false, canIncrease = false, resetCFL = false, stats = false;
    double factorDecrease = 0.0, factorIncrease = 0.0, cflMin = 0.0, cflMax = 0.0, mgFactor = 1.0;
  };
  /*!
   * \brief Device point loop of CSolver::AdaptCFLNumber (flow solver only: no turbulence / species): LocalCFL from
   *        UnderRelaxation and the flags; min / max / sum of the new values (ordered on Serial = the CPU loop; the
   *        sum is order-dependent on GPUs: average CFL monitoring only). Outputs start from the CPU initial values.
   */
  void AdaptCFL(const CFLAdaptParams& p, double& cflMin, double& cflMax, double& cflSum);

  /*!
   * \brief Device CFVMFlowSolverBase::ComputeAxisymmetricAuxGradients: AuxVar (all points with y > EPS: nu*v/y,
   *        nu*v^2/y, nu*u*v/y; other rows keep their value) and its Green-Gauss / weighted least-squares gradient over
   *        the domain points (no symmetry pass if !deviceSymmetry: caller, see Download/UploadSymmetryRowsAux).
   *        Halo rows: HaloExchange(AUXVAR_GRADIENT) by the caller (device-resident mode; otherwise unread).
   * \param[in] method - 0: no gradient (the CPU computes it only for GREEN_GAUSS / WEIGHTED_LEAST_SQUARES), 1: Green-Gauss,
   *                     2: weighted least squares.
   */
  void ComputeAxisymmetricAuxGradient(int method, bool deviceSymmetry);
  /*! \brief A/B: DtoH of the device AuxVar (all rows) and of the domain rows of its gradient. */
  void DownloadAuxVar(double* hostAux, double* hostGrad);
  void DownloadSymmetryRowsAux(double* hostGrad);
  void UploadSymmetryRowsAux(const double* hostGrad);
  /*! \brief Re-upload the host AuxVar / Grad_AuxVar (A/B isolation). */
  void UploadAuxVar();

  /*!
   * \brief Device CFVMFlowSolverBase::SetResidual_DualTime (static mesh): residual += (U - U_n) Vol / dt (1st order) or
   *        (3U - 4U_n + U_n1) Vol / (2 dt) (2nd order), Jacobian diagonal += Vol/dt or 3 Vol / (2 dt), into the device
   *        accumulator (domain points, per entry after the source terms: CPU order).
   * \param[in] uploadInputs - Re-upload Solution, Solution_time_n(1) from the host (A/B isolation).
   */
  void DualTimeResidual(bool secondOrder, bool implicit, double timeStep, bool uploadInputs);
  /*! \brief CIntegration::SetDualTime_Solver on the device: U_n1 = U_n, U_n = U, UnderRelaxation = 1, LocalCFL = cfl. */
  void DualTimeShift(double cfl);
  /*! \brief SetTime_Step dual-time start: min over the domain points of num / LocalCFL (std::min from init). */
  double DualTimeMinDt(double num, double init);
  /*! \brief SetTime_Step explicit dual time: Delta_Time = min(dtMax, Delta_Time) for the domain points. */
  void DualTimeExplicitDt(double dtMax);

  /*!
   * \brief Device Explicit_Iteration_impl<RUNGE_KUTTA_EXPLICIT> (alpha = RK alpha of the stage) and
   *        <CLASSICAL_RK4_EXPLICIT> (timeCoeff / funcCoeff of the stage, Solution_New on the device), residual reduction.
   * \param[in] rk4 - Classical RK4 (else general RK).
   */
  void ExplicitRKUpdate(bool rk4, unsigned short iRKStep, double alpha, double timeCoeff, double funcCoeff,
                        ResidualReduction& red);
  /*! \brief Classical RK4: Set_OldSolution also does the Set_NewSolution copy on the device (both are called back to
   *         back by CMultiGridIntegration, from the same Solution). */
  void SetRK4(bool on) { rk4 = on; }
  /*====================== Step 10: MPI halo exchange of device-resident arrays ======================*/

  /*! \brief The device arrays the NEMO host comms (InitiateComms / CompleteComms) exchange in device-resident mode:
   *         SOLUTION (Solution, nVar), PRIMITIVE_GRADIENT (Gradient_Primitive, nPrimVarGrad*nDim),
   *         PRIMITIVE_GRAD_REC (Gradient_Reconstruction, == Gradient_Primitive if not separate), PRIMITIVE_LIMITER
   *         (Limiter_Primitive only: Solution_Max / Solution_Min are not communicated by the CPU either). */
  enum class HaloQuantity : int { SOLUTION = 0, GRADIENT_PRIMITIVE, GRADIENT_RECONSTRUCTION, LIMITER,
                                  SENSOR,           /*!< batch A: Sensor (1 per point) */
                                  AUXVAR_GRADIENT,  /*!< batch A: Grad_AuxVar (3*nDim per point) */
                                  SOLUTION_OLD,     /*!< 4.6-Periodic: Solution_Old (PERIODIC_IMPLICIT), periodic rows only */
                                  LAMBDA,           /*!< 4.6-Periodic: Lambda (PERIODIC_MAX_EIG), no residency group */
                                  DELTA_TIME,       /*!< 4.6-Periodic: Delta_Time (PERIODIC_RESIDUAL), periodic rows only */
                                  PRIMITIVE,        /*!< 4.6-PeriodicLim: Primitive rows read by PERIODIC_LIM_PRIM_1 (gather only) */
                                  PRIM_MAX,         /*!< 4.6-PeriodicLim: Solution_Max (PERIODIC_LIM_PRIM_1), periodic rows only */
                                  PRIM_MIN,         /*!< 4.6-PeriodicLim: Solution_Min (PERIODIC_LIM_PRIM_1), periodic rows only */
                                  N };
  /*!
   * \brief Device halo exchange with the SU2 point-to-point layout (CGeometry nP2PSend / nPoint_P2PSend /
   *        Neighbors_P2PSend / Local_Point_P2PSend, same message order, offsets and tags as CGeometry::PostP2PSends /
   *        PostP2PRecvs): device gather of the send rows into a persistent buffer, MPI, device scatter of the receive
   *        rows. Only the halo rows move: host-staged (default: send / receive buffers DtoH / HtoD) or GPU-aware
   *        (gpuAware: MPI on the device buffers, KOKKOS_GPU_AWARE_MPI= YES; on host backends the same memory).
   *        Receive rows are copies of the owner's values -> bitwise the same as the host comms.
   */
  void HaloExchange(CGeometry* geometry, HaloQuantity q, bool gpuAware);
  /*! \brief Legacy / A/B path (SU2_KOKKOS_NEMO_HALO=host): download the send rows into the host container, host
   *         comms by the caller, upload the receive rows. */
  void DownloadHaloRows(HaloQuantity q, const unsigned long* rows, unsigned long n);
  void UploadHaloRows(HaloQuantity q, const unsigned long* rows, unsigned long n);
  /*! \brief 4.6-Periodic: the same row gather (device -> host container) / scatter (host container -> device) for an
   *         arbitrary point list, in resident and non-resident mode (residency bookkeeping only when resident).
   *         Used around the host periodic comms (InitiatePeriodicComms / CompletePeriodicComms), which read and write
   *         only the rows of the periodic-marker points. */
  void GatherRowsToHost(HaloQuantity q, const unsigned long* rows, unsigned long n);
  void ScatterRowsFromHost(HaloQuantity q, const unsigned long* rows, unsigned long n);
  /*! \brief 4.6-Periodic, resident mode: rows of the device residual accumulator (st.Res, and the whole block rows of
   *         st.JacVal if hostJac != nullptr; CSysMatrix layout, hostRowPtr = FV pattern outerPtr) to the host LinSysRes /
   *         Jacobian values (toHost) or back. Called around the host PERIODIC_RESIDUAL comms in BC_Periodic. */
  void AccumulatorRows(bool toHost, const unsigned long* rows, unsigned long n, double* hostRes, double* hostJac,
                       const unsigned long* hostRowPtr);
  struct HaloPerf {
    unsigned long calls = 0;
    double time = 0, bytesMsg = 0; /*!< wall time, MPI message bytes sent (both paths) */
  };
  mutable HaloPerf haloPerf[static_cast<int>(HaloQuantity::N)];

  /*! \brief Transfer ledger: totals (bytes) including the linear-algebra backend, per-iteration bookkeeping. */
  void LedgerIterationBoundary(bool outputIteration);
  double TotalH2D() const;
  double TotalD2H() const;
  struct SyncPerf {
    double h2d = 0, d2h = 0, bytes_h2d = 0, bytes_d2h = 0;
    unsigned long calls = 0;
  };
  /*! \brief Per group, + [N_GROUPS] boundary primitive (+ viscous-wall gradient / transport), + [SYNC_HALO] halo rows
   *         (Step 10, MPI: host-staged send / receive buffers). */
  static constexpr int SYNC_HALO = N_GROUPS + 1;
  static constexpr int N_SYNC = N_GROUPS + 2;
  mutable SyncPerf syncPerf[N_SYNC];
  struct UpdatePerf {
    double prepare = 0, complete = 0, explicitUpd = 0, oldSol = 0, norm = 0;
    unsigned long nPrepare = 0, nComplete = 0, nExplicit = 0, nOldSol = 0, nNorm = 0;
  };
  mutable UpdatePerf updatePerf;

  /*! \brief Debug: env SU2_KOKKOS_NEMO_HOST contains stage ("c2p", "grad", "lim", "dt", "upwind", "all"). */
  static bool HostForced(const char* stage);
  bool LimiterOnDevice() const { return limValid; }

  struct StagePerf {
    double h2d = 0, kernel = 0, d2h = 0, host = 0;
    unsigned long calls = 0;
    double bytes_h2d = 0, bytes_d2h = 0;
    double verifyMaxRel = 0;
    unsigned long verifyCalls = 0, verifyNonBitwise = 0, verifyEntries = 0;
    double hostRef = 0;          /*!< time of the host (CPU) code rerun by the A/B verification */
    unsigned long hostRefCalls = 0;
  };
  mutable StagePerf stagePerf[N_STAGES];

  /*! \brief Accumulated timers (seconds) and transfer volume. */
  struct Perf {
    double h2d = 0, kernel = 0, d2h = 0, host_post = 0;
    unsigned long calls = 0;
    double bytes_h2d = 0, bytes_d2h = 0;
  };
  mutable Perf perf;

  void PrintPerf() const;

  /*====================== 4.6-MG2: multigrid transfer operators on the device ======================*/
  /*--- Verbatim CMultiGridIntegration loops; `this` is the fine level, `c` the coarse level (both device-resident,
   *    serial). Each op activates the resident mode of both levels (a coarse level's first device access may
   *    precede its first Preprocessing), EnsureDevice()s its inputs and MarkDeviceWritten()s its outputs.
   *    Halo comms: caller. Public: they enclose KOKKOS_LAMBDAs. ---*/
  /*! \brief Static MG data of a coarse level: children CSR over its domain points (GetChildren_CV order), EULER_WALL
   *         projection list (marker / vertex order, grouped per point keeping that order), once. */
  void MGSetupCoarse(CGeometry* geoCoarse, const CConfig* config);
  /*! \brief Static smoothing data of a level: unique points of the markers SmoothProlongated_Correction resets. */
  void MGSetupSmoothing(CGeometry* geometry, const CConfig* config);
  /*! \brief SetResidual_Term: LinSysRes += Res_TruncError (domain points). */
  void MGResidualTerm();
  /*! \brief SetRestricted_Solution (no viscous walls, static grid): MultigridRestriction of Solution +
   *         MultigridProjectEulerWall(coarse, Solution). iVel = prim_idx.Velocity() of the solver (CPU index). */
  void MGRestrictSolution(CNEMOKokkosData& c, int iVel);
  /*! \brief SetForcing_Term (no viscous walls): coarse TruncErr = sum_children factor * fine LinSysRes - coarse LinSysRes. */
  void MGForcingTerm(CNEMOKokkosData& c, double factor);
  /*! \brief GetProlongated_Correction, first part: coarse Solution_Old = Solution - restricted fine Solution,
   *         MultigridProjectEulerWall(coarse, Solution_Old). */
  void MGProlongCorrection(CNEMOKokkosData& c, int iVel);
  /*! \brief GetProlongated_Correction, second part (after the SOLUTION_OLD comms): fine LinSysRes(child) = coarse Solution_Old. */
  void MGProlongScatter(CNEMOKokkosData& c);
  /*! \brief SmoothProlongated_Correction: Residual_Old = LinSysRes (all points). */
  void MGSmoothSave();
  /*! \brief One Jacobi pass of SmoothProlongated_Correction (all points) + boundary reset to Residual_Old. */
  void MGSmoothIteration(double coeff);
  /*! \brief SetProlongated_Correction: NaN entries of LinSysRes -> 0, Solution += factor * LinSysRes (domain points). */
  void MGSetCorrection(double factor);
  /*! \brief MultiGrid_Iteration: LocalCFL = cfl on every point of a coarse level. */
  void MGFillLocalCFL(double cfl);
  unsigned long mgDeviceOps = 0; /*!< number of device MG transfer ops (perf line) */
  CNEMOEulerVariable* nodes;
  CSU2TCLib* gasModel;
  unsigned short iMesh;
  int nDim, nSpecies, nVar, nPrimVar, nPrimVarGrad;
  unsigned long nPoint;
  bool viscousVars = false; /*!< nodes is a CNEMONSVariable */
  bool primDeviceValid = false, gradPrimValid = false, gradRecValid = false;
  bool secDeviceValid = false; /*!< device dPdU/dTdU/dTvedU/eves/Cvves/Gamma == host (set by SetPrimVar/UploadPrimitive) */
  bool limValid = false;       /*!< device Limiter == host domain rows (set by ComputeLimiter) */
  bool limGeoValid = false;    /*!< st.LimiterGeo uploaded (SetLimiterGeoFactor) */
  bool consDeviceValid = false; /*!< device Solution == host (set by SetPrimVar, cleared by InvalidatePrimitive) */
  bool sensorValid = false;     /*!< device Sensor == host (uploaded by the upwind/BC stages, cleared by SetPrimVar) */
  bool lambdaValid = false;     /*!< device Lambda domain rows == host (set by ComputeMaxEigenvalue, cleared with Primitive) */
  bool accOnDevice = false;     /*!< the device accumulator (Res, JacVal) is authoritative */
  bool hostAccZero = false;     /*!< host LinSysRes / Jacobian are known to be zero */
  bool accImplicit = false;     /*!< JacVal is part of the accumulator */
  unsigned long accDropped = 0; /*!< pending device accumulators discarded by MarkHostAccumulatorZero (expect 0) */
  unsigned long outletCalls = 0, outletSubsonicCalls = 0; /*!< device BC_Outlet calls / with a subsonic vertex */
  std::vector<double> bstateCache; /*!< host copy of st.BState */
  std::vector<unsigned long> h_markerPtr; /*!< host copy of geo.marker_ptr */
  View1D<double> customTwall_;            /*!< 6b.b: per-vertex wall temperature (packed marker layout), lazy */
  void EnsureAccumulator(bool implicit);
  /*--- Step 6. ---*/
  bool resident = false, residentActive = false, poison = false;
  Residency residency[N_GROUPS];
  bool bndPrimHost = false;             /*!< host Primitive rows of the marker-vertex points are current */
  double* hostLinSysRes = nullptr;      /*!< host storage of LinSysRes / LinSysSol (resident mode) */
  double* hostLinSysSol = nullptr;
  std::vector<unsigned long> h_bndPoints;
  /*--- 4.6-MG2 (MGSetupCoarse / MGSetupSmoothing). ---*/
  bool mgCoarseReady = false, mgSmoothReady = false;
  View1D<unsigned long> mgChildPtr, mgChildIdx;  /*!< coarse domain point -> fine children */
  View1D<unsigned long> mgWallPts, mgWallPtr;    /*!< unique EULER_WALL domain points -> projection entries */
  View2D<double> mgWallNormal;                   /*!< (nEntries, nDim) unit normal of each entry */
  View1D<unsigned long> mgSmoothBnd;             /*!< unique points reset by the correction smoothing */
  View2D<double> mgResOld, mgResSum;             /*!< Residual_Old / Residual_Sum of the correction smoothing */
  /*--- Step 8. ---*/
  bool deviceTransport = false, transportDeviceValid = false, bndViscHost = false;
  /*--- Batch A. ---*/
  bool rk4 = false;
  bool auxDeviceValid = false; /*!< device AuxVar == host (host-authoritative mode; resident: G_AUXVAR) */
  View3D<double> symStageAux;  /*!< (nSymPoints, 3, nDim) staging of the AuxVar gradient symmetry rows */
  void AllocateAuxVar();
  std::vector<unsigned long> h_viscWallPoints;
  std::vector<double> bstateViscCache;
 public:
  /*--- Public only because it encloses a KOKKOS_LAMBDA: nvcc rejects extended __host__ __device__ lambdas in
   *    private / protected member functions. Internal: called from SetPrimVar. ---*/
  void ComputeTransportStage(); /*!< DP carry + transport kernel (end of SetPrimVar) */

 private:
  /*--- Step 10: halo plan (device send / receive point indices, persistent buffers). ---*/
  const CGeometry* haloGeometry = nullptr;
  unsigned long haloSend = 0, haloRecv = 0;
  View1D<unsigned long> haloSendIdx, haloRecvIdx;
  View1D<double> haloSendBuf, haloRecvBuf;
  std::vector<double> haloHostSend, haloHostRecv;
  void HaloPlan(CGeometry* geometry);
  double* HaloDevicePtr(HaloQuantity q, size_t& rowLen, Group& g);
  double* HaloHostPtr(HaloQuantity q);
  void DownloadGroup(Group g);
  void UploadGroup(Group g);
  void PoisonGroup(Group g);
  /*--- Ledger: totals at the start of the current iteration, per-iteration deltas (non-output iterations). ---*/
  std::vector<double> ledgerLast;
  std::vector<bool> ledgerOutput;
  std::vector<std::vector<double>> ledgerCat; /*!< per iteration: per-category bytes (see PrintPerf) */
  std::vector<double> LedgerCategories() const;
  bool outputThisIteration = false;
  void EnsureInputsOnDevice(bool upload, bool needSolution, bool needSensor);
  void UploadUpwindStatic();   /*!< pside CSR (once) */
  bool separateRecGradient = false;
  unsigned long residencyChecks = 0, residencyMismatch = 0;
  std::vector<unsigned long> h_symPoints; /*!< domain points with a SYMMETRY_PLANE/EULER_WALL vertex */
  View1D<unsigned long> symPoints;
  View3D<double> symStage;                /*!< (nSymPoints, nPrimVarGrad, nDim) staging for UploadSymmetryRows */
  void AllocateGradients();
  bool implicit, monoatomic;
  double Tve_Freestream;

  /*--- Verification buffers: (name, host pointer, size) of every array written by SetPrimVar. ---*/
  struct HostArray {
    const char* name;
    double* ptr;
    size_t size;
  };
  std::vector<HostArray> verifyArrays;
  std::vector<std::vector<double>> verifyCopies;
  double verifyMaxRel = 0.0;
  unsigned long verifyMismatch = 0, verifyCalls = 0;
};

#endif  // HAVE_KOKKOS
