/*!
 * \file CSysMatrixKokkos.cpp
 * \brief Portable Kokkos implementation of the block-CSR matrix-vector product.
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include <limits>
#include <type_traits>
#include <unordered_map>
#ifdef HAVE_MPI
#include <mutex>
#endif

#include "../../include/linear_algebra/CSysMatrix.hpp"
#include "../../include/geometry/CGeometry.hpp"
#include "../../include/toolboxes/allocation_toolbox.hpp"

namespace {
template <class T>
using DeviceView = Kokkos::View<T*, typename Kokkos::DefaultExecutionSpace::memory_space,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <class T>
using HostView = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

#ifdef HAVE_MPI
/*--- Only MPI builds exchange halos: keeping this out of the non-MPI unit leaves its code generation unchanged
 *    (fast-math builds: inlining decisions depend on the unit's size). ---*/
/*!
 * \brief Device halo point indices already uploaded into a CSysVector's persistent index buffers.
 *
 * Keyed by the device index buffer itself (not by the vector address: vectors are swapped / reallocated, and a freed
 * vector's address can be reused by a new one). An entry records which geometry halo map (geometry object, host index
 * array, count) the buffer currently holds; CSysVector invalidates its entries when it frees or reallocates the
 * buffers (KokkosHaloIndexCacheInvalidate), so a recycled device address can never be mistaken for uploaded indices.
 * Shared by the GPU-aware and host-staged exchanges.
 */
struct HaloIndexEntry {
  const CGeometry* geometry = nullptr;
  const unsigned long* host_indices = nullptr;
  unsigned long points = 0;
};

/*--- Never destroyed (vectors may be freed during static destruction). ---*/
std::mutex& HaloIndexMutex() {
  static auto* m = new std::mutex;
  return *m;
}
std::unordered_map<const void*, HaloIndexEntry>& HaloIndexCache() {
  static auto* cache = new std::unordered_map<const void*, HaloIndexEntry>;
  return *cache;
}

/*! \brief Upload host indices into the device buffer unless it already holds exactly this halo map. */
void EnsureHaloIndices(unsigned long* device_indices, const unsigned long* host_indices, unsigned long points,
                       const CGeometry* geometry) {
  if (points == 0 || device_indices == nullptr) return;
  std::lock_guard<std::mutex> lock(HaloIndexMutex());
  auto& entry = HaloIndexCache()[static_cast<const void*>(device_indices)];
  if (entry.geometry == geometry && entry.host_indices == host_indices && entry.points == points) return;
  Kokkos::deep_copy(DeviceView<unsigned long>(device_indices, points), HostView<const unsigned long>(host_indices, points));
  entry.geometry = geometry;
  entry.host_indices = host_indices;
  entry.points = points;
}
#endif  // HAVE_MPI

/*--- Per-thread execution mode used only by an explicitly device-resident Krylov call.
 * Default values preserve the original synchronized SpMV behavior. ---*/
struct KokkosSpMVExecutionMode {
  bool active = false;
  bool matrix_on_device = false;
  bool input_on_device = false;
  bool output_to_host = true;
};

thread_local KokkosSpMVExecutionMode kokkos_spmv_mode;

/*!
 * \brief Exchange CSysVector halo entries through device-resident buffers.
 *
 * The implementation intentionally uses only Kokkos memory and execution-space
 * abstractions plus standard MPI calls. Consequently, the same source supports
 * CUDA-aware MPI and SYCL/Level-Zero-aware MPI. The user must explicitly enable
 * the path and provide an MPI implementation capable of handling device USM.
 */
template <class ScalarType>
void KokkosGPUAwareHaloExchange(CSysVector<ScalarType>& vector, CGeometry* geometry, bool output_to_host) {
#ifdef HAVE_MPI
  static_assert(std::is_same_v<ScalarType, float> || std::is_same_v<ScalarType, double>,
                "Kokkos GPU-aware MPI currently supports float and double vectors.");

  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space = typename execution_space::memory_space;
  using index_view = Kokkos::View<unsigned long*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using value_view = Kokkos::View<ScalarType*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using range_policy = Kokkos::RangePolicy<execution_space, Kokkos::IndexType<unsigned long>>;

  const auto n_var = vector.GetNVar();
  const auto send_points = geometry->nP2PSend > 0
                               ? static_cast<unsigned long>(geometry->nPoint_P2PSend[geometry->nP2PSend])
                               : 0ul;
  const auto recv_points = geometry->nP2PRecv > 0
                               ? static_cast<unsigned long>(geometry->nPoint_P2PRecv[geometry->nP2PRecv])
                               : 0ul;
  const auto send_value_count = send_points * n_var;
  const auto recv_value_count = recv_points * n_var;

  vector.PrepareKokkosMPIBuffers(send_points, recv_points);
  index_view send_indices(vector.GetKokkosMPISendIndices(), send_points);
  index_view recv_indices(vector.GetKokkosMPIRecvIndices(), recv_points);
  value_view send_values(vector.GetKokkosMPISendBuffer(), send_value_count);
  value_view recv_values(vector.GetKokkosMPIRecvBuffer(), recv_value_count);

  /*--- Halo point maps are geometry metadata: uploaded once per index buffer and geometry (EnsureHaloIndices). ---*/
  EnsureHaloIndices(vector.GetKokkosMPISendIndices(), geometry->Local_Point_P2PSend, send_points, geometry);
  EnsureHaloIndices(vector.GetKokkosMPIRecvIndices(), geometry->Local_Point_P2PRecv, recv_points, geometry);

  const auto mpi_datatype = std::is_same_v<ScalarType, float> ? MPI_FLOAT : MPI_DOUBLE;

  /*--- Receives can be posted before packing the send buffer. ---*/
  for (int i_recv = 0; i_recv < geometry->nP2PRecv; ++i_recv) {
    const auto offset = static_cast<unsigned long>(geometry->nPoint_P2PRecv[i_recv]) * n_var;
    const auto count = static_cast<unsigned long>(geometry->nPoint_P2PRecv[i_recv + 1] -
                                                  geometry->nPoint_P2PRecv[i_recv]) *
                       n_var;
    if (count > static_cast<unsigned long>(std::numeric_limits<int>::max()))
      SU2_MPI::Error("Kokkos GPU-aware MPI receive count exceeds MPI's int limit.", CURRENT_FUNCTION);

    const auto source = geometry->Neighbors_P2PRecv[i_recv];
    SU2_MPI::Irecv(recv_values.data() + offset, static_cast<int>(count), mpi_datatype, source, source + 1,
                   SU2_MPI::GetComm(), &geometry->GetP2PRecvReq<ScalarType>()[i_recv]);
  }

  if (send_value_count > 0) {
    const auto device_vector = vector.GetDevicePointer();
    Kokkos::parallel_for(
        "SU2::PackKokkosMPIHalo", range_policy(0ul, send_value_count), KOKKOS_LAMBDA(const unsigned long i) {
          const auto point = i / n_var;
          const auto variable = i % n_var;
          send_values(i) = device_vector[send_indices(point) * n_var + variable];
        });
    /*--- This fence also completes the preceding SpMV because both kernels use
     * the default execution space and are ordered on the same execution stream. ---*/
    Kokkos::fence("SU2::Kokkos MPI send buffer ready");
  }

  for (int i_send = 0; i_send < geometry->nP2PSend; ++i_send) {
    const auto offset = static_cast<unsigned long>(geometry->nPoint_P2PSend[i_send]) * n_var;
    const auto count = static_cast<unsigned long>(geometry->nPoint_P2PSend[i_send + 1] -
                                                  geometry->nPoint_P2PSend[i_send]) *
                       n_var;
    if (count > static_cast<unsigned long>(std::numeric_limits<int>::max()))
      SU2_MPI::Error("Kokkos GPU-aware MPI send count exceeds MPI's int limit.", CURRENT_FUNCTION);

    SU2_MPI::Isend(send_values.data() + offset, static_cast<int>(count), mpi_datatype,
                   geometry->Neighbors_P2PSend[i_send], SU2_MPI::GetRank() + 1, SU2_MPI::GetComm(),
                   &geometry->GetP2PSendReq<ScalarType>()[i_send]);
  }

  if (geometry->nP2PRecv > 0) {
    SU2_MPI::Waitall(geometry->nP2PRecv, geometry->GetP2PRecvReq<ScalarType>(), MPI_STATUSES_IGNORE);
  }

  if (recv_value_count > 0) {
    const auto device_vector = vector.GetDevicePointer();
    Kokkos::parallel_for(
        "SU2::UnpackKokkosMPIHalo", range_policy(0ul, recv_value_count), KOKKOS_LAMBDA(const unsigned long i) {
          const auto point = i / n_var;
          const auto variable = i % n_var;
          device_vector[recv_indices(point) * n_var + variable] = recv_values(i);
        });
    Kokkos::fence("SU2::Kokkos MPI receive buffer consumed");
  }

  if (geometry->nP2PSend > 0) {
    SU2_MPI::Waitall(geometry->nP2PSend, geometry->GetP2PSendReq<ScalarType>(), MPI_STATUSES_IGNORE);
  }
#endif

  if (output_to_host) vector.DtHTransfer();
}

/*!
 * \brief Exchange only halo entries through host-staged MPI buffers.
 *
 * The full CSysVector remains device resident. Device gather/scatter kernels
 * operate on the persistent Kokkos halo buffers already owned by CSysVector.
 * Only the compact send/receive halo buffers cross the device-host boundary.
 */
template <class ScalarType>
void KokkosHostStagedHaloExchange(CSysVector<ScalarType>& vector,
                                  CGeometry* geometry,
                                  bool output_to_host) {
#ifdef HAVE_MPI
  static_assert(std::is_same_v<ScalarType, float> ||
                    std::is_same_v<ScalarType, double>,
                "Kokkos host-staged halo exchange supports float and double vectors.");

  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space = typename execution_space::memory_space;

  using index_view =
      Kokkos::View<unsigned long*, memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using value_view =
      Kokkos::View<ScalarType*, memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using host_value_view =
      Kokkos::View<ScalarType*, Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using range_policy =
      Kokkos::RangePolicy<execution_space,
                          Kokkos::IndexType<unsigned long>>;

  const auto n_var = vector.GetNVar();

  const auto send_points =
      geometry->nP2PSend > 0
          ? static_cast<unsigned long>(
                geometry->nPoint_P2PSend[geometry->nP2PSend])
          : 0ul;

  const auto recv_points =
      geometry->nP2PRecv > 0
          ? static_cast<unsigned long>(
                geometry->nPoint_P2PRecv[geometry->nP2PRecv])
          : 0ul;

  const auto send_value_count = send_points * n_var;
  const auto recv_value_count = recv_points * n_var;

  vector.PrepareKokkosMPIBuffers(send_points, recv_points);

  index_view send_indices(vector.GetKokkosMPISendIndices(), send_points);
  index_view recv_indices(vector.GetKokkosMPIRecvIndices(), recv_points);

  value_view device_send(vector.GetKokkosMPISendBuffer(), send_value_count);
  value_view device_recv(vector.GetKokkosMPIRecvBuffer(), recv_value_count);

  /* Same index cache as the direct device-MPI path. */
  EnsureHaloIndices(vector.GetKokkosMPISendIndices(), geometry->Local_Point_P2PSend, send_points, geometry);
  EnsureHaloIndices(vector.GetKokkosMPIRecvIndices(), geometry->Local_Point_P2PRecv, recv_points, geometry);

  /*
   * Persistent host buffers per execution thread. They grow only when the halo
   * grows, avoiding allocation in every Krylov/preconditioner application.
   */
  static thread_local std::vector<ScalarType> host_send;
  static thread_local std::vector<ScalarType> host_recv;

  if (host_send.size() < send_value_count)
    host_send.resize(send_value_count);

  if (host_recv.size() < recv_value_count)
    host_recv.resize(recv_value_count);

  const auto mpi_datatype =
      std::is_same_v<ScalarType, float> ? MPI_FLOAT : MPI_DOUBLE;

  /* Post host receives first so communication can begin immediately. */
  for (int i_recv = 0; i_recv < geometry->nP2PRecv; ++i_recv) {
    const auto offset =
        static_cast<unsigned long>(geometry->nPoint_P2PRecv[i_recv]) * n_var;

    const auto count =
        static_cast<unsigned long>(
            geometry->nPoint_P2PRecv[i_recv + 1] -
            geometry->nPoint_P2PRecv[i_recv]) *
        n_var;

    if (count > static_cast<unsigned long>(std::numeric_limits<int>::max()))
      SU2_MPI::Error(
          "Kokkos host-staged MPI receive count exceeds MPI int limit.",
          CURRENT_FUNCTION);

    const auto source = geometry->Neighbors_P2PRecv[i_recv];

    SU2_MPI::Irecv(
        host_recv.data() + offset,
        static_cast<int>(count),
        mpi_datatype,
        source,
        source + 1,
        SU2_MPI::GetComm(),
        &geometry->GetP2PRecvReq<ScalarType>()[i_recv]);
  }

  /* Gather only send-halo entries on the GPU. */
  if (send_value_count > 0) {
    const auto device_vector = vector.GetDevicePointer();

    Kokkos::parallel_for(
        "SU2::PackKokkosHostStagedHalo",
        range_policy(0ul, send_value_count),
        KOKKOS_LAMBDA(const unsigned long i) {
          const auto point = i / n_var;
          const auto variable = i % n_var;

          device_send(i) =
              device_vector[send_indices(point) * n_var + variable];
        });

    /* Copies only the compact halo send buffer. */
    Kokkos::deep_copy(
        host_value_view(host_send.data(), send_value_count),
        device_send);
    KokkosTransferLedger::Add(KokkosTransferLedger::HALO_D2H, double(send_value_count) * sizeof(ScalarType));
  }

  for (int i_send = 0; i_send < geometry->nP2PSend; ++i_send) {
    const auto offset =
        static_cast<unsigned long>(geometry->nPoint_P2PSend[i_send]) * n_var;

    const auto count =
        static_cast<unsigned long>(
            geometry->nPoint_P2PSend[i_send + 1] -
            geometry->nPoint_P2PSend[i_send]) *
        n_var;

    if (count > static_cast<unsigned long>(std::numeric_limits<int>::max()))
      SU2_MPI::Error(
          "Kokkos host-staged MPI send count exceeds MPI int limit.",
          CURRENT_FUNCTION);

    SU2_MPI::Isend(
        host_send.data() + offset,
        static_cast<int>(count),
        mpi_datatype,
        geometry->Neighbors_P2PSend[i_send],
        SU2_MPI::GetRank() + 1,
        SU2_MPI::GetComm(),
        &geometry->GetP2PSendReq<ScalarType>()[i_send]);
  }

  if (geometry->nP2PRecv > 0) {
    SU2_MPI::Waitall(
        geometry->nP2PRecv,
        geometry->GetP2PRecvReq<ScalarType>(),
        MPI_STATUSES_IGNORE);
  }

  if (recv_value_count > 0) {
    /* Copies only the compact halo receive buffer. */
    Kokkos::deep_copy(
        device_recv,
        host_value_view(host_recv.data(), recv_value_count));
    KokkosTransferLedger::Add(KokkosTransferLedger::HALO_H2D, double(recv_value_count) * sizeof(ScalarType));

    const auto device_vector = vector.GetDevicePointer();

    Kokkos::parallel_for(
        "SU2::UnpackKokkosHostStagedHalo",
        range_policy(0ul, recv_value_count),
        KOKKOS_LAMBDA(const unsigned long i) {
          const auto point = i / n_var;
          const auto variable = i % n_var;

          device_vector[recv_indices(point) * n_var + variable] =
              device_recv(i);
        });

    Kokkos::fence("SU2::Kokkos host-staged halo unpack complete");
  }

  if (geometry->nP2PSend > 0) {
    SU2_MPI::Waitall(
        geometry->nP2PSend,
        geometry->GetP2PSendReq<ScalarType>(),
        MPI_STATUSES_IGNORE);
  }
#endif

  if (output_to_host) vector.DtHTransfer();
}

}  // namespace

#ifdef HAVE_MPI
/*! \brief Called by CSysVector before it frees / reallocates a device halo index buffer. */
void KokkosHaloIndexCacheInvalidate(const void* device_indices) {
  if (device_indices == nullptr) return;
  std::lock_guard<std::mutex> lock(HaloIndexMutex());
  HaloIndexCache().erase(device_indices);
}
#endif

namespace KokkosSpMVControl {
void Begin(bool matrix_on_device, bool input_on_device, bool output_to_host) {
  kokkos_spmv_mode.active = true;
  kokkos_spmv_mode.matrix_on_device = matrix_on_device;
  kokkos_spmv_mode.input_on_device = input_on_device;
  kokkos_spmv_mode.output_to_host = output_to_host;
}

void End() { kokkos_spmv_mode = KokkosSpMVExecutionMode{}; }
}  // namespace KokkosSpMVControl

/*--- KOKKOS JACOBI DEVICE STORAGE ---------------------------------------------*/
template <class ScalarType>
void CSysMatrix<ScalarType>::SyncKokkosJacobiPreconditioner() {
  if (invM == nullptr || nPointDomain == 0) return;

  const auto count = nPointDomain * nVar * nVar;

  if (d_invM == nullptr)
    d_invM = GPUMemoryAllocation::gpu_alloc<ScalarType>(
        count * sizeof(ScalarType));

  Kokkos::deep_copy(
      DeviceView<ScalarType>(d_invM, count),
      HostView<const ScalarType>(invM, count));
  KokkosTransferLedger::Add(KokkosTransferLedger::PRECOND_H2D, double(count) * sizeof(ScalarType));
}

template <class ScalarType>
void CSysMatrix<ScalarType>::KokkosComputeJacobiPreconditioner(
    const CSysVector<ScalarType>& vec,
    CSysVector<ScalarType>& prod,
    CGeometry* geometry,
    const CConfig* config) const {

  if (d_invM == nullptr)
    SU2_MPI::Error(
        "Kokkos Jacobi device storage is not initialized.",
        CURRENT_FUNCTION);

  using execution_space = Kokkos::DefaultExecutionSpace;
  using range_policy =
      Kokkos::RangePolicy<execution_space,
                          Kokkos::IndexType<unsigned long>>;

  execution_space exec;

  const auto inv_diag = d_invM;
  const auto input = vec.GetDevicePointer();
  const auto output = prod.GetDevicePointer();

  const auto block_size = nVar;
  const auto block_entries = nVar * nVar;
  const auto domain_points = nPointDomain;

  Kokkos::parallel_for(
      "SU2::KokkosJacobiApply",
      range_policy(exec, 0ul, domain_points),
      KOKKOS_LAMBDA(const unsigned long i) {
        const auto offset = i * block_size;
        const auto block = inv_diag + i * block_entries;

        for (unsigned long i_var = 0;
             i_var < block_size; ++i_var) {
          ScalarType value = 0;

          for (unsigned long j_var = 0;
               j_var < block_size; ++j_var)
            value +=
                block[i_var * block_size + j_var] *
                input[offset + j_var];

          output[offset + i_var] = value;
        }
      });

  exec.fence("SU2::Kokkos Jacobi apply complete");

  if (config->GetKokkosGPUAwareMPI()) {
    KokkosGPUAwareHaloExchange(prod, geometry, false);
  } else {
    /*--- Step 6: only the compact halo rows cross the host boundary (was a full DtoH + host comms + HtoD;
     *    the exchanged halo values and every other entry are the same). ---*/
    KokkosHostStagedHaloExchange(prod, geometry, false);
  }
}

/*--- KOKKOS ILU DEVICE STORAGE ------------------------------------------------
 * Keep CPU factorization semantics unchanged. The completed factors are
 * mirrored after each BuildILUPreconditioner(); sparse and level metadata are
 * allocated/copied only once because the matrix graph is immutable. ---*/
template <class ScalarType>
void CSysMatrix<ScalarType>::SyncKokkosILUPreconditioner() {
  if (ILU_matrix == nullptr || nnz_ilu == 0 || levels_ilu.empty()) return;

  EnsureKokkosILUStructure();
  const auto factor_count = nnz_ilu * nVar * nEqn;
  Kokkos::deep_copy(DeviceView<ScalarType>(d_ILU_matrix, factor_count),
                    HostView<const ScalarType>(ILU_matrix, factor_count));
  KokkosTransferLedger::Add(KokkosTransferLedger::PRECOND_H2D, double(factor_count) * sizeof(ScalarType));
}

template <class ScalarType>
void CSysMatrix<ScalarType>::EnsureKokkosILUStructure() {
  const auto block_size = nVar * nEqn;
  const auto factor_count = nnz_ilu * block_size;
  const auto level_count = levels_ilu.getOuterSize();
  const auto level_rows = levels_ilu.getNumNonZeros();

  if (d_ILU_matrix == nullptr)
    d_ILU_matrix = GPUMemoryAllocation::gpu_alloc<ScalarType>(factor_count * sizeof(ScalarType));

  if (d_row_ptr_ilu == nullptr)
    d_row_ptr_ilu = GPUMemoryAllocation::gpu_alloc_cpy<const unsigned long>(
        row_ptr_ilu, (nPoint + 1) * sizeof(unsigned long));
  if (d_dia_ptr_ilu == nullptr)
    d_dia_ptr_ilu = GPUMemoryAllocation::gpu_alloc_cpy<const unsigned long>(
        dia_ptr_ilu, nPoint * sizeof(unsigned long));
  if (d_col_ind_ilu == nullptr)
    d_col_ind_ilu = GPUMemoryAllocation::gpu_alloc_cpy<const unsigned long>(
        col_ind_ilu, nnz_ilu * sizeof(unsigned long));
  if (d_ilu_level_ptr == nullptr)
    d_ilu_level_ptr = GPUMemoryAllocation::gpu_alloc_cpy<const unsigned long>(
        levels_ilu.outerPtr(), (level_count + 1) * sizeof(unsigned long));
  if (d_ilu_level_rows == nullptr)
    d_ilu_level_rows = GPUMemoryAllocation::gpu_alloc_cpy<const unsigned long>(
        levels_ilu.innerIdx(), level_rows * sizeof(unsigned long));
}

/*--- KOKKOS DEVICE ILU APPLY --------------------------------------------------
 * CPU BuildILUPreconditioner() remains unchanged.  Its completed factors are
 * mirrored by SyncKokkosILUPreconditioner(); this routine applies those factors
 * using dependency levels. Rows in one level are independent, while levels are
 * fenced to preserve the exact triangular dependency ordering portably across
 * CUDA, HIP, and SYCL execution spaces. ---*/
template <class ScalarType>
void CSysMatrix<ScalarType>::KokkosComputeILUPreconditioner(const CSysVector<ScalarType>& vec,
                                                            CSysVector<ScalarType>& prod,
                                                            CGeometry* geometry,
                                                            const CConfig* config) const {
  if (d_ILU_matrix == nullptr || d_row_ptr_ilu == nullptr || d_dia_ptr_ilu == nullptr ||
      d_col_ind_ilu == nullptr || d_ilu_level_ptr == nullptr || d_ilu_level_rows == nullptr ||
      levels_ilu.empty()) {
    SU2_MPI::Error("Kokkos ILU device storage is not initialized.", CURRENT_FUNCTION);
  }

#ifndef NDEBUG
  if (vec.GetNVar() != nVar || prod.GetNVar() != nVar)
    SU2_MPI::Error("Incompatible vector dimensions in Kokkos ILU apply.", CURRENT_FUNCTION);
#endif

  using execution_space = Kokkos::DefaultExecutionSpace;
  using team_policy = Kokkos::TeamPolicy<execution_space>;
  using member_type = typename team_policy::member_type;

  execution_space exec;

  /*--- 64 threads was tuned for V100. Cap by backend concurrency so host
   * backends (Serial: max team size 1) remain valid. ---*/
  const int ilu_team_size = static_cast<int>(
      std::min<long>(64, static_cast<long>(exec.concurrency())));

  const auto factors = d_ILU_matrix;
  const auto rows = d_row_ptr_ilu;
  const auto diagonal = d_dia_ptr_ilu;
  const auto columns = d_col_ind_ilu;
  const auto level_ptr = d_ilu_level_ptr;
  const auto level_rows = d_ilu_level_rows;

  const auto input = vec.GetDevicePointer();
  const auto output = prod.GetDevicePointer();

  const auto block_size = nVar;
  const auto block_entries = nVar * nVar;
  const auto n_levels = levels_ilu.getOuterSize();
  const auto domain_points = nPointDomain;

  /*--- Persistent level-scheduled forward substitution.
   *
   * There is deliberately one league team.  Rows within a dependency level
   * execute in parallel through TeamThreadRange, then the complete team
   * synchronizes before advancing to the next level.  This preserves the
   * triangular level ordering while avoiding one kernel launch per level.
   * ---*/
  Kokkos::parallel_for(
      "SU2::KokkosILUForwardPersistent",
      team_policy(exec, 1, ilu_team_size),
      KOKKOS_LAMBDA(const member_type& team) {
        for (unsigned long level = 0; level < n_levels; ++level) {
          const auto begin = level_ptr[level];
          const auto end = level_ptr[level + 1];

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, begin, end),
              [&](const unsigned long k) {
                const auto i = level_rows[k];
                const auto out_i = i * block_size;

                for (unsigned long i_var = 0; i_var < block_size; ++i_var)
                  output[out_i + i_var] = input[out_i + i_var];

                for (auto index = rows[i]; index < diagonal[i]; ++index) {
                  const auto j = columns[index];
                  const auto out_j = j * block_size;
                  const auto block = factors + index * block_entries;

                  for (unsigned long i_var = 0; i_var < block_size; ++i_var) {
                    ScalarType value = 0;
                    for (unsigned long j_var = 0; j_var < block_size; ++j_var)
                      value += block[i_var * block_size + j_var] *
                               output[out_j + j_var];

                    output[out_i + i_var] -= value;
                  }
                }
              });

          /*--- All rows in this dependency level must finish before any row
           * in the following level starts.  Since league_size == 1 this team
           * barrier synchronizes the complete persistent solve kernel. ---*/
          team.team_barrier();
        }
      });

  /*--- Persistent level-scheduled backward substitution.
   *
   * Same dependency levels are traversed in reverse.  Halo columns are
   * excluded exactly as in the validated CPU/device implementation.
   * ---*/
  Kokkos::parallel_for(
      "SU2::KokkosILUBackwardPersistent",
      team_policy(exec, 1, ilu_team_size),
      KOKKOS_LAMBDA(const member_type& team) {
        for (unsigned long level_plus_one = n_levels;
             level_plus_one > 0; --level_plus_one) {
          const auto level = level_plus_one - 1;
          const auto begin = level_ptr[level];
          const auto end = level_ptr[level + 1];

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, begin, end),
              [&](const unsigned long k) {
                const auto i = level_rows[k];
                const auto out_i = i * block_size;

                ScalarType rhs[20];

                for (unsigned long i_var = 0; i_var < block_size; ++i_var)
                  rhs[i_var] = output[out_i + i_var];

                for (auto index = diagonal[i] + 1;
                     index < rows[i + 1]; ++index) {
                  const auto j = columns[index];

                  /*--- Match CPU BackwardSolve(iPoint,nPointDomain).
                   * Halo columns are not part of the local triangular solve. ---*/
                  if (j >= domain_points) break;

                  const auto out_j = j * block_size;
                  const auto block = factors + index * block_entries;

                  for (unsigned long i_var = 0; i_var < block_size; ++i_var) {
                    ScalarType value = 0;

                    for (unsigned long j_var = 0; j_var < block_size; ++j_var)
                      value += block[i_var * block_size + j_var] *
                               output[out_j + j_var];

                    rhs[i_var] -= value;
                  }
                }

                const auto inv_diag =
                    factors + diagonal[i] * block_entries;

                for (unsigned long i_var = 0; i_var < block_size; ++i_var) {
                  ScalarType value = 0;

                  for (unsigned long j_var = 0; j_var < block_size; ++j_var)
                    value +=
                        inv_diag[i_var * block_size + j_var] * rhs[j_var];

                  output[out_i + i_var] = value;
                }
              });

          team.team_barrier();
        }
      });

  /*--- Forward and backward kernels use the same execution-space instance and
   * are therefore ordered on the same backend queue.  Synchronize once before
   * MPI or host communication consumes the completed triangular solve. ---*/
  exec.fence("SU2::Kokkos persistent ILU triangular solve complete");

  /*--- Preserve existing ILU halo semantics. ---*/
  if (config->GetKokkosGPUAwareMPI()) {
    KokkosGPUAwareHaloExchange(prod, geometry, false);
  } else {
    /* Keep the complete preconditioned vector resident. Only compact halo
     * buffers are staged through host memory for non-GPU-aware MPI. */
    KokkosHostStagedHaloExchange(prod, geometry, false);
  }
}

template <class ScalarType>
void CSysMatrix<ScalarType>::HtDTransfer(bool trigger) const {
  if (!trigger || nnz == 0) return;
  const auto count = nnz * nVar * nEqn;
  Kokkos::deep_copy(DeviceView<ScalarType>(d_matrix, count), HostView<const ScalarType>(matrix, count));
  KokkosTransferLedger::Add(KokkosTransferLedger::MATRIX_H2D, double(count) * sizeof(ScalarType));
}

template <class ScalarType>
void CSysMatrix<ScalarType>::KokkosValuesToHost() {
  if (nnz == 0) return;
  const auto count = nnz * nVar * nEqn;
  Kokkos::deep_copy(HostView<ScalarType>(matrix, count), DeviceView<const ScalarType>(d_matrix, count));
  KokkosTransferLedger::Add(KokkosTransferLedger::MATRIX_D2H, double(count) * sizeof(ScalarType));
}

template <class ScalarType>
void CSysMatrix<ScalarType>::KokkosMatrixVectorProduct(const CSysVector<ScalarType>& vec,
                                                       CSysVector<ScalarType>& prod, CGeometry* geometry,
                                                       const CConfig* config) const {
#ifndef NDEBUG
  if ((nEqn != vec.GetNVar()) || (nVar != prod.GetNVar()) || (nPoint != prod.GetNBlk()))
    SU2_MPI::Error("Incompatible matrix and vector dimensions in Kokkos SpMV.", CURRENT_FUNCTION);
#endif

  const bool matrix_on_device = kokkos_spmv_mode.active && kokkos_spmv_mode.matrix_on_device;
  const bool input_on_device = kokkos_spmv_mode.active && kokkos_spmv_mode.input_on_device;
  const bool output_to_host = !kokkos_spmv_mode.active || kokkos_spmv_mode.output_to_host;

  /*--- Device-authoritative values (SetKokkosValuesOnDevice) are never overwritten from the stale host copy. ---*/
  if (!matrix_on_device && !kokkos_values_on_device) HtDTransfer();
  if (!input_on_device) vec.HtDTransfer();

  const auto values = d_matrix;
  const auto row_offsets = d_row_ptr;
  const auto columns = d_col_ind;
  const auto input = vec.GetDevicePointer();
  const auto output = prod.GetDevicePointer();
  const auto block_rows = nPointDomain;
  const auto block_size_out = nVar;
  const auto block_size_in = nEqn;

  using execution_space = Kokkos::DefaultExecutionSpace;
  using team_policy = Kokkos::TeamPolicy<execution_space>;
  using member_type = typename team_policy::member_type;
  Kokkos::parallel_for(
      "SU2::BlockCrsSpMV", team_policy(block_rows * block_size_out, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& team) {
        const unsigned long flat_row = team.league_rank();
        const unsigned long row = flat_row / block_size_out;
        const unsigned long block_row = flat_row % block_size_out;
        const unsigned long first_block = row_offsets[row];
        const unsigned long work_count = (row_offsets[row + 1] - first_block) * block_size_in;
        ScalarType sum = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, work_count),
            [=](const unsigned long work, ScalarType& local_sum) {
              const unsigned long block = first_block + work / block_size_in;
              const unsigned long block_col = work % block_size_in;
              const unsigned long value_offset =
                  block * block_size_out * block_size_in + block_row * block_size_in + block_col;
              const unsigned long vector_offset = columns[block] * block_size_in + block_col;
              local_sum += values[value_offset] * input[vector_offset];
            },
            sum);
        Kokkos::single(Kokkos::PerTeam(team), [=]() { output[flat_row] = sum; });
      });

  if (config->GetKokkosGPUAwareMPI()) {
    KokkosGPUAwareHaloExchange(prod, geometry, output_to_host);
  } else {
    /* Keep the full SpMV result device resident. Only the compact halo
     * send/receive buffers are staged through host memory. */
    KokkosHostStagedHaloExchange(prod, geometry, output_to_host);
  }
}


/*============================== Step 6: device preconditioner builds ==============================*/
namespace {

/*--- RegularizePivot of CSysMatrix.cpp: "const float eps = 1e-12", compared and copysign'ed in the promoted type. ---*/
template <class T>
KOKKOS_INLINE_FUNCTION void DeviceRegularizePivot(T& pivot) {
  using P = std::common_type_t<float, T>;
  const float eps = 1e-12;
  if (Kokkos::abs(pivot) < eps) pivot = Kokkos::copysign(static_cast<P>(eps), static_cast<P>(pivot));
}

/*--- CSysMatrix::MatrixInverse (non-MKL path), same statement order. A is destroyed, M = inv(A). ---*/
template <class T>
KOKKOS_INLINE_FUNCTION void DeviceMatrixInverse(unsigned long nVar, T* A_, T* M_) {
#define A(I, J) A_[(I)*nVar + (J)]
#define M(I, J) M_[(I)*nVar + (J)]
  for (unsigned long iVar = 0; iVar < nVar; iVar++)
    for (unsigned long jVar = 0; jVar < nVar; jVar++) M(iVar, jVar) = T(iVar == jVar);

  for (unsigned long iVar = 1; iVar < nVar; iVar++) {
    for (unsigned long jVar = 0; jVar < iVar; jVar++) {
      DeviceRegularizePivot(A(jVar, jVar));
      T weight = A(iVar, jVar) / A(jVar, jVar);
      for (auto kVar = jVar; kVar < nVar; kVar++) A(iVar, kVar) -= weight * A(jVar, kVar);
      for (unsigned long kVar = 0; kVar <= jVar; kVar++) M(iVar, kVar) -= weight * M(jVar, kVar);
    }
  }
  for (auto iVar = nVar; iVar > 0ul;) {
    iVar--;
    for (auto jVar = iVar + 1; jVar < nVar; jVar++)
      for (unsigned long kVar = 0; kVar < nVar; kVar++) M(iVar, kVar) -= A(iVar, jVar) * M(jVar, kVar);
    DeviceRegularizePivot(A(iVar, iVar));
    for (unsigned long kVar = 0; kVar < nVar; kVar++) M(iVar, kVar) /= A(iVar, iVar);
  }
#undef A
#undef M
}

/*--- gemm_impl of CSysMatrix.inl. ---*/
template <class T>
KOKKOS_INLINE_FUNCTION void DeviceGemm(unsigned long n, const T* a, const T* b, T* c) {
  for (unsigned long i = 0; i < n; i++) {
    for (unsigned long j = 0; j < n; j++) {
      c[i * n + j] = 0.0;
      for (unsigned long k = 0; k < n; k++) c[i * n + j] += a[i * n + k] * b[k * n + j];
    }
  }
}

bool PrecondVerifyEnabled() {
  static const bool on = [] {
    const char* env = std::getenv("SU2_KOKKOS_PRECOND_VERIFY");
    return env != nullptr && std::atoi(env) != 0;
  }();
  return on;
}

template <class T>
unsigned long CountNonBitwise(const T* a, const T* b, size_t n) {
  unsigned long bad = 0;
  for (size_t k = 0; k < n; ++k) bad += (std::memcmp(&a[k], &b[k], sizeof(T)) != 0);
  return bad;
}

}  // namespace

template <class ScalarType>
void CSysMatrix<ScalarType>::KokkosBuildJacobiPreconditioner() {
  if (invM == nullptr || nPointDomain == 0) return;
  if (nVar != nEqn) SU2_MPI::Error("Kokkos Jacobi build needs square blocks.", CURRENT_FUNCTION);
  Kokkos::Timer timer;
  const auto count = nPointDomain * nVar * nVar;
  if (d_invM == nullptr) d_invM = GPUMemoryAllocation::gpu_alloc<ScalarType>(count * sizeof(ScalarType));
  if (d_dia_ptr == nullptr)
    d_dia_ptr = GPUMemoryAllocation::gpu_alloc_cpy<const unsigned long>(dia_ptr, nPointDomain * sizeof(unsigned long));

  using range_policy = Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<unsigned long>>;
  const auto values = d_matrix;
  const auto diag = d_dia_ptr;
  const auto inv = d_invM;
  const unsigned long nv = nVar, bs = nVar * nVar;
  /*--- InverseDiagonalBlock: copy the block (MatrixInverse destroys its input), invert. ---*/
  Kokkos::parallel_for(
      "SU2::KokkosJacobiBuild", range_policy(0ul, nPointDomain), KOKKOS_LAMBDA(const unsigned long i) {
        ScalarType block[MAXNVAR * MAXNVAR];
        const auto* src = values + diag[i] * bs;
        for (unsigned long k = 0; k < bs; ++k) block[k] = src[k];
        DeviceMatrixInverse(nv, block, inv + i * bs);
      });
  Kokkos::fence("SU2::Kokkos Jacobi build complete");
  kokkos_device_builds++;

  if (PrecondVerifyEnabled()) {
    /*--- A/B: host build from the downloaded values (also re-syncs d_invM from the host result). ---*/
    std::vector<ScalarType> dev(count);
    Kokkos::deep_copy(HostView<ScalarType>(dev.data(), count), DeviceView<const ScalarType>(d_invM, count));
    KokkosValuesToHost();
    kokkos_force_host_build = true;
    BuildJacobiPreconditioner();
    kokkos_force_host_build = false;
    kokkos_verify_builds++;
    kokkos_verify_nonbitwise += CountNonBitwise(invM, dev.data(), count);
  }
  kokkos_build_seconds += timer.seconds();
}

template <class ScalarType>
void CSysMatrix<ScalarType>::KokkosBuildILUPreconditioner() {
  if (ILU_matrix == nullptr || nnz_ilu == 0 || levels_ilu.empty()) return;
  if (ilu_fill_in != 0 || nVar != nEqn)
    SU2_MPI::Error("Kokkos ILU build supports ILU(0) with square blocks only.", CURRENT_FUNCTION);
  Kokkos::Timer timer;
  EnsureKokkosILUStructure();

  using range_policy = Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<unsigned long>>;
  const auto values = d_matrix;
  const auto ilu = d_ILU_matrix;
  const auto rows = d_row_ptr_ilu;
  const auto diagonal = d_dia_ptr_ilu;
  const auto columns = d_col_ind_ilu;
  const auto level_rows = d_ilu_level_rows;
  const unsigned long nv = nVar, bs = nVar * nVar, end = nPointDomain;

  /*--- Same per-row operations as BuildILUPreconditioner (level-scheduled branch). Rows of one level only read
   *    rows of earlier levels and write their own blocks, so the row order within a level does not matter. ---*/
  for (auto level = 0ul; level < levels_ilu.getOuterSize(); ++level) {
    const auto first = levels_ilu.outerPtr()[level];
    const auto last = levels_ilu.outerPtr()[level + 1];
    const bool firstLevel = (level == 0);
    Kokkos::parallel_for(
        "SU2::KokkosILU0Build", range_policy(first, last), KOKKOS_LAMBDA(const unsigned long k) {
          const auto iPoint = level_rows[k];

          /*--- InitIluRow (ILU0: direct copy of the row's blocks). ---*/
          for (auto idx = rows[iPoint] * bs; idx < rows[iPoint + 1] * bs; ++idx) ilu[idx] = values[idx];

          if (!firstLevel) {
            /*--- BuildIluRow(iPoint, 0, nPointDomain). ---*/
            ScalarType Lij[MAXNVAR * MAXNVAR], Lij_Ujk[MAXNVAR * MAXNVAR];
            for (auto index = rows[iPoint]; index < diagonal[iPoint]; ++index) {
              const auto jPoint = columns[index];
              auto* Block_ij = ilu + index * bs;
              const auto* invUjj = ilu + diagonal[jPoint] * bs;
              DeviceGemm(nv, Block_ij, invUjj, Lij);

              for (auto index_ = diagonal[jPoint] + 1; index_ < rows[jPoint + 1]; ++index_) {
                const auto kPoint = columns[index_];
                if (kPoint >= end) break;

                /*--- GetBlock_ILUMatrix(iPoint, kPoint). ---*/
                ScalarType* Block_ik = nullptr;
                const auto searchEnd = (kPoint < iPoint) ? diagonal[iPoint] : rows[iPoint + 1];
                for (auto s = (kPoint < iPoint) ? rows[iPoint] : diagonal[iPoint]; s < searchEnd; ++s) {
                  if (columns[s] == kPoint) {
                    Block_ik = ilu + s * bs;
                    break;
                  }
                }
                if (Block_ik == nullptr) continue;

                const auto* Ujk = ilu + index_ * bs;
                DeviceGemm(nv, Lij, Ujk, Lij_Ujk);
                for (unsigned long v = 0; v < bs; ++v) Block_ik[v] = Block_ik[v] - Lij_Ujk[v];
              }
              for (unsigned long v = 0; v < bs; ++v) Block_ij[v] = Lij[v];
            }
          }

          /*--- InvertDiagonalBlockILUMatrix(iPoint). ---*/
          ScalarType block[MAXNVAR * MAXNVAR];
          auto* Uii = ilu + diagonal[iPoint] * bs;
          for (unsigned long v = 0; v < bs; ++v) block[v] = Uii[v];
          DeviceMatrixInverse(nv, block, Uii);
        });
  }
  Kokkos::fence("SU2::Kokkos ILU(0) build complete");
  kokkos_device_builds++;

  if (PrecondVerifyEnabled()) {
    /*--- A/B: host build from the downloaded values; the host factors are compared and then synced (run continues
     *    with the host result, identical if bitwise). ---*/
    const auto count = nnz_ilu * bs;
    std::vector<ScalarType> dev(count);
    Kokkos::deep_copy(HostView<ScalarType>(dev.data(), count), DeviceView<const ScalarType>(d_ILU_matrix, count));
    KokkosValuesToHost();
    kokkos_force_host_build = true;
    BuildILUPreconditioner();
    kokkos_force_host_build = false;
    kokkos_verify_builds++;
    /*--- Only the factorized rows (those in levels_ilu) are defined. ---*/
    unsigned long bad = 0;
    for (auto k = 0ul; k < levels_ilu.getNumNonZeros(); ++k) {
      const auto i = levels_ilu.innerIdx()[k];
      const auto b = row_ptr_ilu[i] * bs, e = row_ptr_ilu[i + 1] * bs;
      bad += CountNonBitwise(ILU_matrix + b, dev.data() + b, e - b);
    }
    kokkos_verify_nonbitwise += bad;
  }
  kokkos_build_seconds += timer.seconds();
}


/*============================== 4.6: device LU_SGS apply ==============================*/
namespace {

/*--- CSysMatrix::Gauss_Elimination(ScalarType*, ScalarType*) (non-MKL path), same statement order. A is destroyed. ---*/
template <class T>
KOKKOS_INLINE_FUNCTION void DeviceGaussElimination(unsigned long nVar, T* A_, T* vec) {
#define A(I, J) A_[(I)*nVar + (J)]
  for (unsigned long iVar = 1; iVar < nVar; iVar++) {
    for (unsigned long jVar = 0; jVar < iVar; jVar++) {
      DeviceRegularizePivot(A(jVar, jVar));
      T weight = A(iVar, jVar) / A(jVar, jVar);
      for (unsigned long kVar = jVar; kVar < nVar; kVar++) A(iVar, kVar) -= weight * A(jVar, kVar);
      vec[iVar] -= weight * vec[jVar];
    }
  }
  for (unsigned long iVar = nVar; iVar > 0ul;) {
    iVar--;
    for (unsigned long jVar = iVar + 1; jVar < nVar; jVar++) vec[iVar] -= A(iVar, jVar) * vec[jVar];
    DeviceRegularizePivot(A(iVar, iVar));
    vec[iVar] /= A(iVar, iVar);
  }
#undef A
}

/*--- Rows grouped by level (CSR), ascending row index inside a level. ---*/
void GroupByLevel(const std::vector<unsigned long>& lev, unsigned long nLev, std::vector<unsigned long>& ptr,
                  std::vector<unsigned long>& rows) {
  ptr.assign(nLev + 1, 0);
  for (auto l : lev) ptr[l + 1]++;
  for (unsigned long l = 0; l < nLev; ++l) ptr[l + 1] += ptr[l];
  rows.resize(lev.size());
  std::vector<unsigned long> pos(ptr.begin(), ptr.end() - 1);
  for (unsigned long i = 0; i < lev.size(); ++i) rows[pos[lev[i]]++] = i;
}

}  // namespace

template <class ScalarType>
void CSysMatrix<ScalarType>::EnsureKokkosLU_SGSStructure() const {
  if (d_lusgs_fwd_ptr != nullptr) return;
  if (d_dia_ptr == nullptr)
    d_dia_ptr = GPUMemoryAllocation::gpu_alloc_cpy<const unsigned long>(dia_ptr, nPointDomain * sizeof(unsigned long));

  /*--- Partition bounds of every row, exactly as the omp loop of ComputeLU_SGSPreconditioner sees them
   *    (redundant partitions with begin == end own no rows). ---*/
  std::vector<unsigned long> lb(nPointDomain, 0), ub(nPointDomain, nPointDomain);
  for (unsigned long t = 0; t < omp_num_parts; ++t) {
    const auto begin = omp_partitions[t], end = omp_partitions[t + 1];
    for (auto i = begin; i < end; ++i) { lb[i] = begin; ub[i] = end; }
  }

  /*--- Forward: row i needs the lower columns j in [lb, i). Backward: the upper domain columns j in (i, ub);
   *    halo columns are inputs (exchanged between the sweeps). ---*/
  std::vector<unsigned long> lf(nPointDomain, 0), lbk(nPointDomain, 0);
  unsigned long nf = 0, nb = 0;
  for (unsigned long i = 0; i < nPointDomain; ++i) {
    unsigned long l = 0;
    for (auto k = row_ptr[i]; k < dia_ptr[i]; ++k) {
      const auto j = col_ind[k];
      if (j >= lb[i]) l = std::max(l, lf[j] + 1);
    }
    lf[i] = l; nf = std::max(nf, l + 1);
  }
  for (unsigned long i = nPointDomain; i > 0;) {
    --i;
    unsigned long l = 0;
    for (auto k = dia_ptr[i] + 1; k < row_ptr[i + 1]; ++k) {
      const auto j = col_ind[k];
      if (j < ub[i]) l = std::max(l, lbk[j] + 1);
    }
    lbk[i] = l; nb = std::max(nb, l + 1);
  }
  if (nPointDomain == 0) nf = nb = 0;
  std::vector<unsigned long> fp, fr, bp, br;
  GroupByLevel(lf, nf, fp, fr);
  GroupByLevel(lbk, nb, bp, br);
  lusgs_n_fwd = nf;
  lusgs_n_bwd = nb;

  auto up = [](const std::vector<unsigned long>& v) {
    return GPUMemoryAllocation::gpu_alloc_cpy<const unsigned long>(v.data(), std::max<size_t>(v.size(), 1) * sizeof(unsigned long));
  };
  d_lusgs_fwd_ptr = up(fp);
  d_lusgs_fwd_rows = up(fr);
  d_lusgs_bwd_ptr = up(bp);
  d_lusgs_bwd_rows = up(br);
  d_lusgs_lb = up(lb);
  d_lusgs_ub = up(ub);
}

/*--- Transcription of ComputeLU_SGSPreconditioner: LowerProduct / VectorSubtraction / Gauss_Elimination per row
 *    (forward), halo exchange, DiagonalProduct / UpperProduct / VectorSubtraction / Gauss_Elimination per row
 *    (backward), halo exchange. Every row reads only rows of earlier levels (same values as the sequential sweep)
 *    and accumulates in the CPU order (gemv_impl: c[i] += a[i*m+j]*b[j], column order of the row). ---*/
template <class ScalarType>
void CSysMatrix<ScalarType>::KokkosComputeLU_SGSPreconditioner(const CSysVector<ScalarType>& vec,
                                                               CSysVector<ScalarType>& prod, CGeometry* geometry,
                                                               const CConfig* config) const {
  if (nVar != nEqn || nVar > MAXNVAR) SU2_MPI::Error("Kokkos LU_SGS needs square blocks <= MAXNVAR.", CURRENT_FUNCTION);
  Kokkos::Timer timer;
  EnsureKokkosLU_SGSStructure();

  const bool matrix_on_device = kokkos_spmv_mode.active && kokkos_spmv_mode.matrix_on_device;
  if (!matrix_on_device && !kokkos_values_on_device) HtDTransfer();

  using execution_space = Kokkos::DefaultExecutionSpace;
  using team_policy = Kokkos::TeamPolicy<execution_space>;
  using member_type = typename team_policy::member_type;
  execution_space exec;
  const int team_size = static_cast<int>(std::min<long>(64, static_cast<long>(exec.concurrency())));

  const auto values = d_matrix;
  const auto rows = d_row_ptr;
  const auto diag = d_dia_ptr;
  const auto cols = d_col_ind;
  const auto lbound = d_lusgs_lb;
  const auto ubound = d_lusgs_ub;
  const auto fptr = d_lusgs_fwd_ptr;
  const auto frows = d_lusgs_fwd_rows;
  const auto bptr = d_lusgs_bwd_ptr;
  const auto brows = d_lusgs_bwd_rows;
  const auto n_fwd = lusgs_n_fwd;
  const auto n_bwd = lusgs_n_bwd;
  const auto input = vec.GetDevicePointer();
  const auto output = prod.GetDevicePointer();
  const unsigned long nv = nVar, ne = nEqn, bs = nVar * nEqn, nPD = nPointDomain;

  /*--- First part of the symmetric iteration: (D+L).x* = b ---*/
  Kokkos::parallel_for(
      "SU2::KokkosLUSGSForward", team_policy(exec, 1, team_size), KOKKOS_LAMBDA(const member_type& team) {
        for (unsigned long level = 0; level < n_fwd; ++level) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, fptr[level], fptr[level + 1]), [&](const unsigned long k) {
            const auto iPoint = frows[k];
            const auto idx = iPoint * nv;
            ScalarType low_prod[MAXNVAR];
            /*--- LowerProduct(prod, iPoint, begin, low_prod) ---*/
            for (unsigned long iVar = 0; iVar < nv; iVar++) low_prod[iVar] = 0.0;
            for (auto index = rows[iPoint]; index < diag[iPoint]; index++) {
              const auto col_j = cols[index];
              if (col_j >= lbound[iPoint]) {
                const auto* a = values + index * bs;
                const auto* b = output + col_j * ne;
                for (unsigned long i = 0; i < nv; i++)
                  for (unsigned long j = 0; j < ne; j++) low_prod[i] += a[i * ne + j] * b[j];
              }
            }
            /*--- VectorSubtraction(&vec[idx], low_prod, &prod[idx]) ---*/
            for (unsigned long iVar = 0; iVar < nv; iVar++) output[idx + iVar] = input[idx + iVar] - low_prod[iVar];
            /*--- Gauss_Elimination(iPoint, &prod[idx]) ---*/
            ScalarType block[MAXNVAR * MAXNVAR];
            const auto* src = values + diag[iPoint] * bs;
            for (unsigned long m = 0; m < nv * nv; ++m) block[m] = src[m];
            DeviceGaussElimination(nv, block, output + idx);
          });
          team.team_barrier();
        }
      });
  exec.fence("SU2::Kokkos LU_SGS forward sweep complete");

  if (config->GetKokkosGPUAwareMPI()) KokkosGPUAwareHaloExchange(prod, geometry, false);
  else KokkosHostStagedHaloExchange(prod, geometry, false);

  /*--- Second part of the symmetric iteration: (D+U).x_(1) = D.x* ---*/
  Kokkos::parallel_for(
      "SU2::KokkosLUSGSBackward", team_policy(exec, 1, team_size), KOKKOS_LAMBDA(const member_type& team) {
        for (unsigned long level = 0; level < n_bwd; ++level) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, bptr[level], bptr[level + 1]), [&](const unsigned long k) {
            const auto iPoint = brows[k];
            const auto idx = iPoint * nv;
            ScalarType up_prod[MAXNVAR], dia_prod[MAXNVAR];
            /*--- DiagonalProduct(prod, iPoint, dia_prod) ---*/
            {
              const auto* a = values + diag[iPoint] * bs;
              const auto* b = output + iPoint * ne;
              for (unsigned long i = 0; i < nv; i++) {
                dia_prod[i] = 0.0;
                for (unsigned long j = 0; j < ne; j++) dia_prod[i] += a[i * ne + j] * b[j];
              }
            }
            /*--- UpperProduct(prod, iPoint, row_end, up_prod): always include halos. ---*/
            for (unsigned long iVar = 0; iVar < nv; iVar++) up_prod[iVar] = 0.0;
            for (auto index = diag[iPoint] + 1; index < rows[iPoint + 1]; index++) {
              const auto col_j = cols[index];
              if (col_j < ubound[iPoint] || col_j >= nPD) {
                const auto* a = values + index * bs;
                const auto* b = output + col_j * ne;
                for (unsigned long i = 0; i < nv; i++)
                  for (unsigned long j = 0; j < ne; j++) up_prod[i] += a[i * ne + j] * b[j];
              }
            }
            /*--- VectorSubtraction(dia_prod, up_prod, &prod[idx]); Gauss_Elimination(iPoint, &prod[idx]) ---*/
            for (unsigned long iVar = 0; iVar < nv; iVar++) output[idx + iVar] = dia_prod[iVar] - up_prod[iVar];
            ScalarType block[MAXNVAR * MAXNVAR];
            const auto* src = values + diag[iPoint] * bs;
            for (unsigned long m = 0; m < nv * nv; ++m) block[m] = src[m];
            DeviceGaussElimination(nv, block, output + idx);
          });
          team.team_barrier();
        }
      });
  exec.fence("SU2::Kokkos LU_SGS backward sweep complete");

  if (config->GetKokkosGPUAwareMPI()) KokkosGPUAwareHaloExchange(prod, geometry, false);
  else KokkosHostStagedHaloExchange(prod, geometry, false);

  kokkos_lusgs_applies++;

  if (PrecondVerifyEnabled()) {
    /*--- A/B: host apply on host copies of the device input and matrix values (debug only, full transfers). ---*/
    const auto n = prod.GetLocSize();
    std::vector<ScalarType> dev(n);
    Kokkos::deep_copy(HostView<ScalarType>(dev.data(), n), DeviceView<const ScalarType>(output, n));
    if (nnz > 0) {
      const auto count = nnz * nVar * nEqn;
      Kokkos::deep_copy(HostView<ScalarType>(matrix, count), DeviceView<const ScalarType>(d_matrix, count));
    }
    vec.DtHTransfer();
    CSysVector<ScalarType> hin(vec), hout(prod);
    ComputeLU_SGSPreconditioner(hin, hout, geometry, config);
    kokkos_lusgs_verify++;
    kokkos_lusgs_verify_nonbitwise += CountNonBitwise(&hout[0], dev.data(), n);
  }
  kokkos_lusgs_seconds += timer.seconds();
}

template class CSysMatrix<su2mixedfloat>;
