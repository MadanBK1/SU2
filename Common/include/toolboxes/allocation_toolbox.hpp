/*!
 * \file allocation_toolbox.hpp
 * \brief Helper function and classes for memory allocation.
 *        Focus on portability across platforms.
 * \note  These are "kernel" functions, only to be used with good reason,
 *        always try to use higher level container classes.
 * \author P. Gomes, D. Kavolis
 * \version 8.5.0 "Harrier"
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2026, SU2 Contributors (cf. AUTHORS.md)
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * SU2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with SU2. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#if defined(_WIN32)
#include <malloc.h>
#else
#include <stdlib.h>
#endif

#ifdef HAVE_CUDA
#include "../linear_algebra/GPUComms.cuh"
#endif

#ifdef HAVE_KOKKOS
#include <Kokkos_Core.hpp>
#endif

#include <cstring>
#include <type_traits>

#include <cassert>

namespace MemoryAllocation {

inline constexpr bool is_power_of_two(size_t x) { return x && !(x & (x - 1)); }

inline constexpr size_t round_up(size_t multiple, size_t x) { return ((x + multiple - 1) / multiple) * multiple; }

/*!
 * \brief Aligned memory allocation compatible across platforms.
 * \param[in] alignment, in bytes, of the memory being allocated.
 * \param[in] size, also in bytes.
 * \tparam ZeroInit, initialize memory to 0.
 * \return Pointer to memory, always use su2::aligned_free to deallocate.
 */
template <class T, bool ZeroInit = false>
inline T* aligned_alloc(size_t alignment, size_t size) noexcept {
  assert(is_power_of_two(alignment));

  if (alignment < alignof(void*)) alignment = alignof(void*);

  size = round_up(alignment, size);

  void* ptr = nullptr;

  if (size > 0) {
#if defined(__APPLE__)
    if (::posix_memalign(&ptr, alignment, size) != 0) {
      ptr = nullptr;
    }
#elif defined(_WIN32)
    ptr = _aligned_malloc(size, alignment);
#else
    ptr = ::aligned_alloc(alignment, size);
#endif
    if (ZeroInit) memset(ptr, 0, size);
  }
  return static_cast<T*>(ptr);
}

/*!
 * \brief Free memory allocated with su2::aligned_alloc.
 * \param[in] ptr, pointer to memory we want to release.
 */
template <class T>
inline void aligned_free(T* ptr) noexcept {
#if defined(_WIN32)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

}  // namespace MemoryAllocation

namespace GPUMemoryAllocation {
#if defined(HAVE_CUDA) && defined(HAVE_KOKKOS)
#error "CUDA and Kokkos device allocation backends cannot be enabled together"
#endif

#ifdef HAVE_KOKKOS
namespace detail {
class KokkosRuntime {
 private:
  bool owns_runtime = false;

 public:
  KokkosRuntime() {
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
      owns_runtime = true;
    }
  }
  ~KokkosRuntime() {
    if (owns_runtime && Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
  }
};

inline void ensure_kokkos_initialized() {
  static KokkosRuntime runtime;
  (void)runtime;
}
}  // namespace detail
#endif

/*!
 * \brief Memory allocation for variables on the GPU.
 * \param[in] size in bytes.
 * \tparam ZeroInit, initialize memory to 0.
 * \return Pointer to memory, always use gpu_free to deallocate.
 */
template <class T, bool ZeroInit = false>
inline T* gpu_alloc(size_t size) noexcept {
  void* ptr = nullptr;

#if defined(HAVE_CUDA)
  gpuErrChk(cudaMalloc((void**)(&ptr), size));
  if (ZeroInit) gpuErrChk(cudaMemset((void*)(ptr), 0.0, size));
#elif defined(HAVE_KOKKOS)
  detail::ensure_kokkos_initialized();
  if (size > 0) {
    using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
    ptr = Kokkos::kokkos_malloc<memory_space>("SU2 device allocation", size);
    if constexpr (ZeroInit) {
      using view_type = Kokkos::View<unsigned char*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      Kokkos::deep_copy(view_type(static_cast<unsigned char*>(ptr), size), static_cast<unsigned char>(0));
    }
  }
#else
  return 0;
#endif

  return static_cast<T*>(ptr);
}

/*!
 * \brief Free memory allocated on the GPU with gpu_alloc.
 * \param[in] ptr, pointer to memory we want to release.
 */
template <class T>
inline void gpu_free(T* ptr) noexcept {
#ifdef HAVE_CUDA
  gpuErrChk(cudaFree((void*)ptr));
#elif defined(HAVE_KOKKOS)
  using value_type = std::remove_const_t<T>;
  if (ptr != nullptr) Kokkos::kokkos_free(const_cast<value_type*>(ptr));
#endif
}
/*!
 * \brief Memory allocation for variables on the GPU along with initialization from a source host array.
 * \param[in] size in bytes.
 * \return Pointer to memory, always use gpu_free to deallocate.
 */
template <class T>
inline T* gpu_alloc_cpy(const T* src_ptr, size_t size) noexcept {
  void* ptr = nullptr;

#ifdef HAVE_CUDA
  gpuErrChk(cudaMalloc((void**)(&ptr), size));
  gpuErrChk(cudaMemcpy((void*)(ptr), (void*)src_ptr, size, cudaMemcpyHostToDevice));
#elif defined(HAVE_KOKKOS)
  detail::ensure_kokkos_initialized();
  if (size > 0) {
    using value_type = std::remove_const_t<T>;
    using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
    ptr = Kokkos::kokkos_malloc<memory_space>("SU2 device copy", size);
    const size_t count = size / sizeof(value_type);
    using device_view = Kokkos::View<value_type*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using host_view = Kokkos::View<const value_type*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::deep_copy(device_view(static_cast<value_type*>(ptr), count), host_view(src_ptr, count));
  }
#endif

  return static_cast<T*>(ptr);
}
}  // namespace GPUMemoryAllocation
