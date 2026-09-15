/*!
 * \file CSysMatrixKokkos.cpp
 * \brief Portable Kokkos implementation of the block-CSR matrix-vector product.
 */

#include <Kokkos_Core.hpp>

#include <limits>
#include <type_traits>
#include <unordered_map>

#include "../../include/linear_algebra/CSysMatrix.hpp"
#include "../../include/geometry/CGeometry.hpp"
#include "../../include/toolboxes/allocation_toolbox.hpp"

namespace {

#if defined(KOKKOS_ARCH_HOPPER90)
constexpr int KOKKOS_ILU_TEAM_SIZE = 512;
#elif defined(KOKKOS_ARCH_VOLTA70) || defined(KOKKOS_ARCH_VOLTA72)
constexpr int KOKKOS_ILU_TEAM_SIZE = 64;
#else
constexpr int KOKKOS_ILU_TEAM_SIZE = 64;
#endif

template <class T>
using DeviceView = Kokkos::View<T*, typename Kokkos::DefaultExecutionSpace::memory_space,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <class T>
using HostView = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

struct HaloIndexCache {
  const CGeometry* geometry = nullptr;
  const unsigned long* host_send_indices = nullptr;
  const unsigned long* host_recv_indices = nullptr;
  unsigned long* device_send_indices = nullptr;
  unsigned long* device_recv_indices = nullptr;
  unsigned long send_points = 0;
  unsigned long recv_points = 0;
};

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

  /*--- Halo point maps are geometry metadata and do not change during Krylov iterations.
   * Upload them only when a vector first participates in the exchange, when its
   * device index allocation changes, or when the geometry/layout changes. ---*/
  static std::unordered_map<const void*, HaloIndexCache> halo_index_cache;
  auto& cache = halo_index_cache[static_cast<const void*>(&vector)];
  const bool refresh_send =
      cache.geometry != geometry || cache.host_send_indices != geometry->Local_Point_P2PSend ||
      cache.device_send_indices != vector.GetKokkosMPISendIndices() || cache.send_points != send_points;
  const bool refresh_recv =
      cache.geometry != geometry || cache.host_recv_indices != geometry->Local_Point_P2PRecv ||
      cache.device_recv_indices != vector.GetKokkosMPIRecvIndices() || cache.recv_points != recv_points;

  if (refresh_send && send_points > 0) {
    Kokkos::deep_copy(send_indices, HostView<const unsigned long>(geometry->Local_Point_P2PSend, send_points));
  }
  if (refresh_recv && recv_points > 0) {
    Kokkos::deep_copy(recv_indices, HostView<const unsigned long>(geometry->Local_Point_P2PRecv, recv_points));
  }

  if (refresh_send || refresh_recv) {
    cache.geometry = geometry;
    cache.host_send_indices = geometry->Local_Point_P2PSend;
    cache.host_recv_indices = geometry->Local_Point_P2PRecv;
    cache.device_send_indices = vector.GetKokkosMPISendIndices();
    cache.device_recv_indices = vector.GetKokkosMPIRecvIndices();
    cache.send_points = send_points;
    cache.recv_points = recv_points;
  }

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

  /* Reuse the same geometry/index cache as the direct device-MPI path. */
  static std::unordered_map<const void*, HaloIndexCache> halo_index_cache;
  auto& cache = halo_index_cache[static_cast<const void*>(&vector)];

  const bool refresh_send =
      cache.geometry != geometry ||
      cache.host_send_indices != geometry->Local_Point_P2PSend ||
      cache.device_send_indices != vector.GetKokkosMPISendIndices() ||
      cache.send_points != send_points;

  const bool refresh_recv =
      cache.geometry != geometry ||
      cache.host_recv_indices != geometry->Local_Point_P2PRecv ||
      cache.device_recv_indices != vector.GetKokkosMPIRecvIndices() ||
      cache.recv_points != recv_points;

  if (refresh_send && send_points > 0) {
    Kokkos::deep_copy(
        send_indices,
        HostView<const unsigned long>(
            geometry->Local_Point_P2PSend, send_points));
  }

  if (refresh_recv && recv_points > 0) {
    Kokkos::deep_copy(
        recv_indices,
        HostView<const unsigned long>(
            geometry->Local_Point_P2PRecv, recv_points));
  }

  if (refresh_send || refresh_recv) {
    cache.geometry = geometry;
    cache.host_send_indices = geometry->Local_Point_P2PSend;
    cache.host_recv_indices = geometry->Local_Point_P2PRecv;
    cache.device_send_indices = vector.GetKokkosMPISendIndices();
    cache.device_recv_indices = vector.GetKokkosMPIRecvIndices();
    cache.send_points = send_points;
    cache.recv_points = recv_points;
  }

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
    prod.DtHTransfer();
    CSysMatrixComms::Initiate(prod, geometry, config);
    CSysMatrixComms::Complete(prod, geometry, config);
    prod.HtDTransfer();
  }
}

/*--- KOKKOS ILU DEVICE STORAGE ------------------------------------------------
 * Keep CPU factorization semantics unchanged. The completed factors are
 * mirrored after each BuildILUPreconditioner(); sparse and level metadata are
 * allocated/copied only once because the matrix graph is immutable. ---*/
template <class ScalarType>
void CSysMatrix<ScalarType>::SyncKokkosILUPreconditioner() {
  if (ILU_matrix == nullptr || nnz_ilu == 0 || levels_ilu.empty()) return;

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

  Kokkos::deep_copy(DeviceView<ScalarType>(d_ILU_matrix, factor_count),
                    HostView<const ScalarType>(ILU_matrix, factor_count));
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
      team_policy(exec, 1, KOKKOS_ILU_TEAM_SIZE),
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
      team_policy(exec, 1, KOKKOS_ILU_TEAM_SIZE),
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

  if (!matrix_on_device) HtDTransfer();
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

template class CSysMatrix<su2mixedfloat>;
