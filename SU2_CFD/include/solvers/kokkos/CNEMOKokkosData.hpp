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

    /*--- Boundary info for gather-based point kernels (UploadBoundaryInfo). ---*/
    View1D<unsigned long> pvtx_ptr;   /*!< point -> flattened vertex CSR (nPoint+1), marker-major order */
    View1D<unsigned long> pvtx_idx;
    View1D<int> vtx_marker;           /*!< (nVertexTotal) marker of each flattened vertex */
    View1D<unsigned char> marker_flags; /*!< (nMarker) MarkerFlag bits */
    View2D<double> vtx_sym_normal;    /*!< (nVertexTotal, nDim) unit normal used by correctGradientsSymmetry */
    bool domain_first = true;         /*!< GetDomain(i) == (i < nPointDomain), required by the device stages */
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
    View1D<double> LocalCFL, MaxLambdaInv, MaxLambdaVisc, DeltaTime, Lambda; /*!< (nPoint) */
    View1D<double> LamVisc, ThermalCond, ThermalCondVe;                      /*!< (nPoint), NS only */
  };

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
  void InvalidatePrimitive() { primDeviceValid = false; }
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

  /*!
   * \brief Device slope limiter (computeLimiters_impl without periodicity) for kinds BARTH_JESPERSEN,
   *        VENKATAKRISHNAN (eps2 given) and VENKATAKRISHNAN_WANG (eps2 per variable from a device min/max
   *        reduction + host MPI reduction, computed here). Reads the device reconstruction gradient.
   *        Downloads rows [0, nPointDomain) of limiter, primMax, primMin.
   * \param[in] epsVenkat - eps2 (BJ/Venkat), or K (Venkat-Wang).
   */
  void ComputeLimiter(int kind, double umusclKappa, double epsVenkat, double* hostLim, double* hostMax,
                      double* hostMin);
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
  enum Stage { ST_GRADIENT = 0, ST_LIMITER, ST_TIMESTEP, ST_MAXEIG, N_STAGES };
  /*! \brief Compare host (CPU) result vs a copy of the device result over n entries (row-major, ncol columns,
   *         relative difference normalised per column); accumulates per stage. */
  void StageCompare(Stage s, const char* name, const double* cpu, const double* dev, size_t n, size_t ncol = 1);
  void StageCompareScalar(Stage s, const char* name, double cpu, double dev);

  struct StagePerf {
    double h2d = 0, kernel = 0, d2h = 0, host = 0;
    unsigned long calls = 0;
    double bytes_h2d = 0, bytes_d2h = 0;
    double verifyMaxRel = 0;
    unsigned long verifyCalls = 0, verifyNonBitwise = 0, verifyEntries = 0;
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

 private:
  CNEMOEulerVariable* nodes;
  CSU2TCLib* gasModel;
  unsigned short iMesh;
  int nDim, nSpecies, nVar, nPrimVar, nPrimVarGrad;
  unsigned long nPoint;
  bool viscousVars = false; /*!< nodes is a CNEMONSVariable */
  bool primDeviceValid = false, gradPrimValid = false, gradRecValid = false;
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
