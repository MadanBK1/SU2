/*!
 * \file CNEMOKokkosData.hpp
 * \brief Device (Kokkos) mirror of the geometry and state used by the NEMO solvers.
 *
 * Ownership model (Step 2 of NEMO_KOKKOS_PORT_PLAN.md):
 *  - Geometry group: uploaded once at construction, read-only on the device afterwards.
 *    The host CGeometry stays authoritative. TODO(mesh deformation / dynamic grids): call
 *    UploadGeometry() again after the host geometry changes.
 *  - State group (Solution, Solution_Old, Primitive, dPdU/dTdU/dTvedU, eves, Cvves, Gamma):
 *    the HOST CNEMOEulerVariable is authoritative between solver calls. The device copies are
 *    transient: every device kernel uploads what it reads and downloads what it writes, so the
 *    rest of the CPU pipeline (restart, BCs, output, host SetPrimVar callers) is untouched.
 *    Later steps will move authority to the device and shrink the transfers.
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
  int nDim, nSpecies, nVar, nPrimVar;
  unsigned long nPoint;
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
