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

namespace {
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

/*!
 * \brief Exchange CSysVector halo entries through device-resident buffers.
 *
 * The implementation intentionally uses only Kokkos memory and execution-space
 * abstractions plus standard MPI calls. Consequently, the same source supports
 * CUDA-aware MPI and SYCL/Level-Zero-aware MPI. The user must explicitly enable
 * the path and provide an MPI implementation capable of handling device USM.
 */
template <class ScalarType>
void KokkosGPUAwareHaloExchange(CSysVector<ScalarType>& vector, CGeometry* geometry) {
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

  /*--- Host-resident Krylov and preconditioner operations still consume the result. ---*/
  vector.DtHTransfer();
}
}  // namespace

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

  HtDTransfer();
  vec.HtDTransfer();

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
    KokkosGPUAwareHaloExchange(prod, geometry);
  } else {
    prod.DtHTransfer();
    CSysMatrixComms::Initiate(prod, geometry, config);
    CSysMatrixComms::Complete(prod, geometry, config);
  }
}

template class CSysMatrix<su2mixedfloat>;
