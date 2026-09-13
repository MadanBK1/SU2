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

template class CSysVector<su2mixedfloat>;
