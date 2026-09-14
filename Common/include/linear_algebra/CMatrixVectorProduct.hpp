/*!
 * \file CMatrixVectorProduct.hpp
 * \brief Headers for the classes related to sparse matrix-vector product wrappers.
 *        The actual operations are currently implemented mostly by CSysMatrix.
 * \author F. Palacios, J. Hicken, T. Economon
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

#include "../CConfig.hpp"
#include "../geometry/CGeometry.hpp"
#include "CSysVector.hpp"
#include "CSysMatrix.hpp"

#ifdef HAVE_KOKKOS
/*!
 * \brief Controls synchronization performed by the Kokkos SpMV for one call.
 *
 * Normal matrix-vector products do not touch this state and retain the legacy
 * host-synchronized behavior. Kokkos Krylov solvers may opt into the device-
 * resident path when both the matrix and vectors are already current on device.
 */
namespace KokkosSpMVControl {
void Begin(bool matrix_on_device, bool input_on_device, bool output_to_host);
void End();
}  // namespace KokkosSpMVControl
#endif

/*!
 * \class CMatrixVectorProduct
 * \ingroup SpLinSys
 * \brief Abstract base class for defining matrix-vector products
 * \author J. Hicken.
 *
 * The Krylov-subspace solvers require only matrix-vector products and
 * not the actual matrix/Jacobian.  We need some way to indicate which
 * function will perform the product.  However, sometimes the
 * functions that define the product will require different numbers
 * and types of inputs.  For example, the forward-difference
 * approximation to a Jacobian-vector product requires the vector that
 * defines the Jacobian and a perturbation parameter.  The
 * CMatrixVectorProduct class is used to derive child classes that can
 * handle the different types of matrix-vector products and still be
 * passed to a single implementation of the Krylov solvers.
 * This abstraction may also be used to define matrix-free products.
 */

template <class ScalarType>
class CMatrixVectorProduct {
 public:
  virtual ~CMatrixVectorProduct() = 0;
  virtual void operator()(const CSysVector<ScalarType>& u, CSysVector<ScalarType>& v) const = 0;

  /*!
   * \brief Whether this product supports a Kokkos device-resident application.
   * \note Matrix-free and other generic products return false by default.
   */
  virtual bool SupportsKokkosDeviceResident() const { return false; }

  /*!
   * \brief Apply the product while allowing a Kokkos solver to suppress redundant synchronization.
   * \param[in] u - Input vector.
   * \param[out] v - Output vector.
   * \param[in] matrix_on_device - Matrix values are already current on device.
   * \param[in] input_on_device - Input vector values are already current on device.
   * \param[in] output_to_host - Copy the completed output back to host before returning.
   * \note The generic fallback preserves existing behavior.
   */
  virtual void KokkosDeviceResident(const CSysVector<ScalarType>& u, CSysVector<ScalarType>& v,
                                    bool matrix_on_device, bool input_on_device, bool output_to_host) const {
    (void)matrix_on_device;
    (void)input_on_device;
    (void)output_to_host;
    (*this)(u, v);
  }
};
template <class ScalarType>
CMatrixVectorProduct<ScalarType>::~CMatrixVectorProduct() {}

/*!
 * \class CSysMatrixVectorProduct
 * \ingroup SpLinSys
 * \brief Specialization of matrix-vector product that uses CSysMatrix class
 */
template <class ScalarType>
class CSysMatrixVectorProduct final : public CMatrixVectorProduct<ScalarType> {
 private:
  const CSysMatrix<ScalarType>& matrix; /*!< \brief pointer to matrix that defines the product. */
  CGeometry* geometry;                  /*!< \brief geometry associated with the matrix. */
  const CConfig* config;                /*!< \brief config of the problem. */

 public:
  /*!
   * \brief constructor of the class
   * \param[in] matrix_ref - matrix reference that will be used to define the products
   * \param[in] geometry_ref - geometry associated with the problem
   * \param[in] config_ref - config of the problem
   */
  inline CSysMatrixVectorProduct(const CSysMatrix<ScalarType>& matrix_ref, CGeometry* geometry_ref,
                                 const CConfig* config_ref)
      : matrix(matrix_ref), geometry(geometry_ref), config(config_ref) {}

  /*!
   * \note This class cannot be default constructed as that would leave us with invalid pointers.
   */
  CSysMatrixVectorProduct() = delete;

  /*!
   * \brief operator that defines the CSysMatrix-CSysVector product
   * \param[in] u - CSysVector that is being multiplied by the sparse matrix
   * \param[out] v - CSysVector that is the result of the product
   */
  inline void operator()(const CSysVector<ScalarType>& u, CSysVector<ScalarType>& v) const override {
    if (config->GetKokkos()) {
#ifdef HAVE_KOKKOS
      matrix.KokkosMatrixVectorProduct(u, v, geometry, config);
#else
      SU2_MPI::Error(
          "\nError launching the Kokkos matrix-vector product.\nENABLE_KOKKOS is YES in the configuration, "
          "but SU2 was not compiled with -Denable-kokkos=true.",
          CURRENT_FUNCTION);
#endif
    } else if (config->GetCUDA()) {
#ifdef HAVE_CUDA
      matrix.GPUMatrixVectorProduct(u, v, geometry, config);
#else
      SU2_MPI::Error(
          "\nError in launching Matrix-Vector Product Function\nENABLE_CUDA is set to YES\nPlease compile with CUDA "
          "options enabled in Meson to access GPU Functions",
          CURRENT_FUNCTION);
#endif
    } else {
      matrix.MatrixVectorProduct(u, v, geometry, config);
    }
  }

  inline bool SupportsKokkosDeviceResident() const override {
#ifdef HAVE_KOKKOS
    /*--- Direct device residency currently requires the device-buffer halo path.
     * Host-staged MPI still needs the SpMV result on host. ---*/
    return config->GetKokkos() && config->GetKokkosGPUAwareMPI();
#else
    return false;
#endif
  }

  inline void KokkosDeviceResident(const CSysVector<ScalarType>& u, CSysVector<ScalarType>& v,
                                   bool matrix_on_device, bool input_on_device,
                                   bool output_to_host) const override {
#ifdef HAVE_KOKKOS
    if (!SupportsKokkosDeviceResident()) {
      (*this)(u, v);
      return;
    }
    KokkosSpMVControl::Begin(matrix_on_device, input_on_device, output_to_host);
    matrix.KokkosMatrixVectorProduct(u, v, geometry, config);
    KokkosSpMVControl::End();
#else
    (void)matrix_on_device;
    (void)input_on_device;
    (void)output_to_host;
    (*this)(u, v);
#endif
  }
};
