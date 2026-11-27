/*!
 * \file CNEMOKokkosResidualWall.cpp
 * \brief Device NEMO wall / symmetry BC kernels (isothermal, heat flux, Smoluchowski-Maxwell, symmetry).
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
template void KernelIsothermal<2>(const ResArgs&, unsigned long, unsigned long);
template void KernelIsothermal<3>(const ResArgs&, unsigned long, unsigned long);
template void KernelHeatFlux<2>(const ResArgs&, unsigned long, unsigned long);
template void KernelHeatFlux<3>(const ResArgs&, unsigned long, unsigned long);
template void KernelSmoluchowski<2>(const ResArgs&, unsigned long, unsigned long);
template void KernelSmoluchowski<3>(const ResArgs&, unsigned long, unsigned long);
template void KernelSymmetry<2>(const ResArgs&, unsigned long, unsigned long);
template void KernelSymmetry<3>(const ResArgs&, unsigned long, unsigned long);
}  // namespace nemo_kokkos_res

#endif  // HAVE_KOKKOS
