/*!
 * \file CNEMOKokkosResidualBCFlux3D.cpp
 * \brief Device NEMO convective boundary flux kernel (AUSM family / Roe / LAX / MSW), ND = 3.
 *        Split out of CNEMOKokkosResidual.cpp to cut nvcc compile time; code verbatim (see that file for the
 *        exactness notes).
 *
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */


#ifdef HAVE_KOKKOS

#include "CNEMOKokkosResidualKernels.hpp"
#include "CNEMOKokkosResidualKernels.inl"

namespace nemo_kokkos_res {
template void KernelBoundaryFlux<3>(const ResArgs&, int, unsigned long, unsigned long, long&);
}  // namespace nemo_kokkos_res

#endif  // HAVE_KOKKOS
