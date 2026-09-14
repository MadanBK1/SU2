/*!
 * \file CSysVectorKokkos.cpp
 * \brief Kokkos host/device transfers for the distributed linear-system vector.
 */

#include <Kokkos_Core.hpp>

#include "../../include/linear_algebra/CSysVector.hpp"

namespace {
template <class ScalarType>
using DeviceView = Kokkos::View<ScalarType*, typename Kokkos::DefaultExecutionSpace::memory_space,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <class ScalarType>
using HostView =
    Kokkos::View<ScalarType*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
}  // namespace

template <class ScalarType>
void CSysVector<ScalarType>::HtDTransfer(bool trigger) const {
  if (trigger && nElm > 0) Kokkos::deep_copy(DeviceView<ScalarType>(d_vec_val, nElm),
                                              HostView<const ScalarType>(vec_val, nElm));
}

template <class ScalarType>
void CSysVector<ScalarType>::DtHTransfer(bool trigger) const {
  if (trigger && nElm > 0) Kokkos::deep_copy(HostView<ScalarType>(vec_val, nElm),
                                              DeviceView<const ScalarType>(d_vec_val, nElm));
}

template <class ScalarType>
void CSysVector<ScalarType>::GPUSetVal(ScalarType val, bool trigger) const {
  if (trigger && nElm > 0) Kokkos::deep_copy(DeviceView<ScalarType>(d_vec_val, nElm), val);
}

template <class ScalarType>
void CSysVector<ScalarType>::PrepareKokkosMPIBuffers(unsigned long sendPoints, unsigned long recvPoints) {
  using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
  const auto sendValues = sendPoints * nVar;
  const auto recvValues = recvPoints * nVar;

  if (sendValues > kokkos_mpi_send_capacity) {
    if (d_kokkos_mpi_send != nullptr) Kokkos::kokkos_free(d_kokkos_mpi_send);
    d_kokkos_mpi_send = static_cast<ScalarType*>(
        Kokkos::kokkos_malloc<memory_space>("SU2::KokkosMPISendBuffer", sendValues * sizeof(ScalarType)));
    kokkos_mpi_send_capacity = sendValues;
  }
  if (recvValues > kokkos_mpi_recv_capacity) {
    if (d_kokkos_mpi_recv != nullptr) Kokkos::kokkos_free(d_kokkos_mpi_recv);
    d_kokkos_mpi_recv = static_cast<ScalarType*>(
        Kokkos::kokkos_malloc<memory_space>("SU2::KokkosMPIRecvBuffer", recvValues * sizeof(ScalarType)));
    kokkos_mpi_recv_capacity = recvValues;
  }
  if (sendPoints > kokkos_mpi_send_index_capacity) {
    if (d_kokkos_mpi_send_indices != nullptr) Kokkos::kokkos_free(d_kokkos_mpi_send_indices);
    d_kokkos_mpi_send_indices = static_cast<unsigned long*>(
        Kokkos::kokkos_malloc<memory_space>("SU2::KokkosMPISendIndices", sendPoints * sizeof(unsigned long)));
    kokkos_mpi_send_index_capacity = sendPoints;
  }
  if (recvPoints > kokkos_mpi_recv_index_capacity) {
    if (d_kokkos_mpi_recv_indices != nullptr) Kokkos::kokkos_free(d_kokkos_mpi_recv_indices);
    d_kokkos_mpi_recv_indices = static_cast<unsigned long*>(
        Kokkos::kokkos_malloc<memory_space>("SU2::KokkosMPIRecvIndices", recvPoints * sizeof(unsigned long)));
    kokkos_mpi_recv_index_capacity = recvPoints;
  }
}

template <class ScalarType>
void CSysVector<ScalarType>::ReleaseKokkosMPIBuffers() {
  if (d_kokkos_mpi_send != nullptr) Kokkos::kokkos_free(d_kokkos_mpi_send);
  if (d_kokkos_mpi_recv != nullptr) Kokkos::kokkos_free(d_kokkos_mpi_recv);
  if (d_kokkos_mpi_send_indices != nullptr) Kokkos::kokkos_free(d_kokkos_mpi_send_indices);
  if (d_kokkos_mpi_recv_indices != nullptr) Kokkos::kokkos_free(d_kokkos_mpi_recv_indices);

  d_kokkos_mpi_send = nullptr;
  d_kokkos_mpi_recv = nullptr;
  d_kokkos_mpi_send_indices = nullptr;
  d_kokkos_mpi_recv_indices = nullptr;
  kokkos_mpi_send_capacity = 0;
  kokkos_mpi_recv_capacity = 0;
  kokkos_mpi_send_index_capacity = 0;
  kokkos_mpi_recv_index_capacity = 0;
}

template class CSysVector<su2mixedfloat>;
