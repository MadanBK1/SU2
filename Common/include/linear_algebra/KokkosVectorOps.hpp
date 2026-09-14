/*!
 * \file KokkosVectorOps.hpp
 * \brief Device-resident vector primitives used by Kokkos linear solvers.
 */

#pragma once

#ifdef HAVE_KOKKOS

#include <Kokkos_Core.hpp>
#include <cmath>
#include <type_traits>

#include "CSysVector.hpp"
#include "../parallelization/mpi_structure.hpp"

namespace KokkosLinearAlgebra {

template <class ScalarType>
using ExecSpace = Kokkos::DefaultExecutionSpace;

template <class ScalarType>
using RangePolicy = Kokkos::RangePolicy<ExecSpace<ScalarType>, Kokkos::IndexType<unsigned long>>;

template <class ScalarType>
inline void AssertSameSize(const CSysVector<ScalarType>& x, const CSysVector<ScalarType>& y, const char* where) {
#ifndef NDEBUG
  if (x.GetLocSize() != y.GetLocSize()) SU2_MPI::Error("Kokkos vector size mismatch.", where);
#else
  (void)x;
  (void)y;
  (void)where;
#endif
}

template <class ScalarType>
inline auto MPIDatatype() {
  static_assert(std::is_same_v<ScalarType, float> || std::is_same_v<ScalarType, double>,
                "Kokkos linear vector reductions currently support float and double.");
  if constexpr (std::is_same_v<ScalarType, float>)
    return MPI_FLOAT;
  else
    return MPI_DOUBLE;
}

template <class ScalarType>
inline void Fill(CSysVector<ScalarType>& x, ScalarType value, bool fence = false) {
  const auto n = x.GetLocSize();
  if (n == 0) return;
  auto* px = x.GetDevicePointer();
  Kokkos::parallel_for("SU2::KokkosVectorFill", RangePolicy<ScalarType>(0ul, n),
                       KOKKOS_LAMBDA(const unsigned long i) { px[i] = value; });
  if (fence) Kokkos::fence("SU2::KokkosVectorFill complete");
}

template <class ScalarType>
inline void Copy(const CSysVector<ScalarType>& x, CSysVector<ScalarType>& y, bool fence = false) {
  AssertSameSize(x, y, CURRENT_FUNCTION);
  const auto n = x.GetLocSize();
  if (n == 0) return;
  const auto* px = x.GetDevicePointer();
  auto* py = y.GetDevicePointer();
  Kokkos::parallel_for("SU2::KokkosVectorCopy", RangePolicy<ScalarType>(0ul, n),
                       KOKKOS_LAMBDA(const unsigned long i) { py[i] = px[i]; });
  if (fence) Kokkos::fence("SU2::KokkosVectorCopy complete");
}

template <class ScalarType>
inline void Scale(CSysVector<ScalarType>& x, ScalarType alpha, bool fence = false) {
  const auto n = x.GetLocSize();
  if (n == 0) return;
  auto* px = x.GetDevicePointer();
  Kokkos::parallel_for("SU2::KokkosVectorScale", RangePolicy<ScalarType>(0ul, n),
                       KOKKOS_LAMBDA(const unsigned long i) { px[i] *= alpha; });
  if (fence) Kokkos::fence("SU2::KokkosVectorScale complete");
}

template <class ScalarType>
inline void AXPY(CSysVector<ScalarType>& y, ScalarType alpha, const CSysVector<ScalarType>& x,
                 bool fence = false) {
  AssertSameSize(x, y, CURRENT_FUNCTION);
  const auto n = y.GetLocSize();
  if (n == 0) return;
  const auto* px = x.GetDevicePointer();
  auto* py = y.GetDevicePointer();
  Kokkos::parallel_for("SU2::KokkosVectorAXPY", RangePolicy<ScalarType>(0ul, n),
                       KOKKOS_LAMBDA(const unsigned long i) { py[i] += alpha * px[i]; });
  if (fence) Kokkos::fence("SU2::KokkosVectorAXPY complete");
}

template <class ScalarType>
inline void AXPBY(CSysVector<ScalarType>& y, ScalarType alpha, const CSysVector<ScalarType>& x, ScalarType beta,
                  bool fence = false) {
  AssertSameSize(x, y, CURRENT_FUNCTION);
  const auto n = y.GetLocSize();
  if (n == 0) return;
  const auto* px = x.GetDevicePointer();
  auto* py = y.GetDevicePointer();
  Kokkos::parallel_for("SU2::KokkosVectorAXPBY", RangePolicy<ScalarType>(0ul, n),
                       KOKKOS_LAMBDA(const unsigned long i) { py[i] = alpha * px[i] + beta * py[i]; });
  if (fence) Kokkos::fence("SU2::KokkosVectorAXPBY complete");
}

template <class ScalarType>
inline void LinearCombination(CSysVector<ScalarType>& y, ScalarType alpha, const CSysVector<ScalarType>& x,
                              ScalarType beta, const CSysVector<ScalarType>& z, bool fence = false) {
  AssertSameSize(x, y, CURRENT_FUNCTION);
  AssertSameSize(z, y, CURRENT_FUNCTION);
  const auto n = y.GetLocSize();
  if (n == 0) return;
  const auto* px = x.GetDevicePointer();
  const auto* pz = z.GetDevicePointer();
  auto* py = y.GetDevicePointer();
  Kokkos::parallel_for("SU2::KokkosVectorLinearCombination", RangePolicy<ScalarType>(0ul, n),
                       KOKKOS_LAMBDA(const unsigned long i) { py[i] = alpha * px[i] + beta * pz[i]; });
  if (fence) Kokkos::fence("SU2::KokkosVectorLinearCombination complete");
}

template <class ScalarType>
inline ScalarType LocalDot(const CSysVector<ScalarType>& x, const CSysVector<ScalarType>& y) {
  AssertSameSize(x, y, CURRENT_FUNCTION);
  const auto n = x.GetNElmDomain();
  if (n == 0) return ScalarType(0);
  const auto* px = x.GetDevicePointer();
  const auto* py = y.GetDevicePointer();
  ScalarType local = 0;
  Kokkos::parallel_reduce(
      "SU2::KokkosVectorDot", RangePolicy<ScalarType>(0ul, n),
      KOKKOS_LAMBDA(const unsigned long i, ScalarType& update) { update += px[i] * py[i]; }, local);
  return local;
}

template <class ScalarType>
inline ScalarType Dot(const CSysVector<ScalarType>& x, const CSysVector<ScalarType>& y, bool global_reduce = true) {
  ScalarType local = LocalDot(x, y);
  if (!global_reduce || SU2_MPI::GetSize() == 1) return local;

  ScalarType global = 0;
  using MPIWrapper = typename SelectMPIWrapper<ScalarType>::W;
  MPIWrapper::Allreduce(&local, &global, 1, MPIDatatype<ScalarType>(), MPI_SUM, SU2_MPI::GetComm());
  return global;
}

template <class ScalarType>
inline ScalarType Norm(const CSysVector<ScalarType>& x, bool global_reduce = true) {
  const ScalarType value = Dot(x, x, global_reduce);
  return value > ScalarType(0) ? std::sqrt(value) : ScalarType(0);
}

/*--- Synchronization helpers. These are deliberately explicit so solver code can
 * keep vectors device-resident and synchronize only at CPU preconditioner or
 * monitoring boundaries. ---*/
template <class ScalarType>
inline void HostToDevice(const CSysVector<ScalarType>& x) {
  x.HtDTransfer();
}

template <class ScalarType>
inline void DeviceToHost(const CSysVector<ScalarType>& x) {
  x.DtHTransfer();
}

}  // namespace KokkosLinearAlgebra

#endif  // HAVE_KOKKOS
