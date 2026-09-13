/*!
 * \file CSysMatrixKokkos.cpp
 * \brief Portable Kokkos implementation of the block-CSR matrix-vector product.
 */

#include <Kokkos_Core.hpp>

#include "../../include/linear_algebra/CSysMatrix.hpp"

namespace {
template <class T>
using DeviceView = Kokkos::View<T*, typename Kokkos::DefaultExecutionSpace::memory_space,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <class T>
using HostView = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
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
  Kokkos::fence("SU2::BlockCrsSpMV complete");

  prod.DtHTransfer();
  CSysMatrixComms::Initiate(prod, geometry, config);
  CSysMatrixComms::Complete(prod, geometry, config);
}

template class CSysMatrix<su2mixedfloat>;
