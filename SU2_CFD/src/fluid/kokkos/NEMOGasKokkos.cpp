/*!
 * \file NEMOGasKokkos.cpp
 * \brief Host-side builder of su2kokkos::nemo::NEMOGasTables from a CSU2TCLib object.
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#ifdef HAVE_KOKKOS

#include "../../../include/fluid/kokkos/NEMOGasKokkos.hpp"
#include "../../../include/fluid/CSU2TCLib.hpp"
#include "../../../../Common/include/toolboxes/allocation_toolbox.hpp"

#include <string>
#include <type_traits>

static_assert(std::is_same_v<su2double, double>, "NEMOGasKokkos requires a passive (double) build.");
static_assert(std::is_trivially_copyable_v<su2kokkos::nemo::NEMOPrimIndices>);

namespace su2kokkos {
namespace nemo {

/*! \brief Friend of CSU2TCLib, reads its private/protected constants. */
struct NEMOGasTablesAccess {
  static NEMOGasTables Build(const CSU2TCLib& gas) {
    const int ns = gas.nSpecies;

    if (ns < 1 || ns > MAXNS)
      SU2_MPI::Error("NEMOGasKokkos: nSpecies = " + std::to_string(ns) + " exceeds MAXNS = " + std::to_string(MAXNS),
                     CURRENT_FUNCTION);
    if (gas.nDim > MAXNDIM) SU2_MPI::Error("NEMOGasKokkos: nDim exceeds MAXNDIM.", CURRENT_FUNCTION);
    if (gas.nReactions > MAXNR)
      SU2_MPI::Error("NEMOGasKokkos: nReactions = " + std::to_string(gas.nReactions) + " exceeds MAXNR.",
                     CURRENT_FUNCTION);
    if (gas.ionization && gas.nEl != 1)
      SU2_MPI::Error("NEMOGasKokkos: ionized mixture must have exactly one electron (species 0).", CURRENT_FUNCTION);

    int maxEl = 0;
    for (int s = 0; s < ns; s++) maxEl = std::max<int>(maxEl, gas.nElStates[s]);
    if (maxEl > MAXEL)
      SU2_MPI::Error("NEMOGasKokkos: electronic levels = " + std::to_string(maxEl) + " exceed MAXEL.", CURRENT_FUNCTION);
    if (static_cast<int>(gas.CharElTemp.rows()) != ns || static_cast<int>(gas.CharElTemp.cols()) < maxEl)
      SU2_MPI::Error("NEMOGasKokkos: unexpected CharElTemp shape.", CURRENT_FUNCTION);
    /*--- CPU reads level 0 even when nElStates == 0, so a 0-column table is not supported. ---*/
    if (maxEl == 0) SU2_MPI::Error("NEMOGasKokkos: no electronic levels.", CURRENT_FUNCTION);

    /*--- Same lazy runtime init as SU2's device allocations (no-op if already initialized). ---*/
    GPUMemoryAllocation::detail::ensure_kokkos_initialized();

    using mem = NEMOGasTables::memory_space;
    Kokkos::View<double*, mem> M("NEMO::MolarMass", ns), thv("NEMO::CharVibTemp", ns), rot("NEMO::RotationModes", ns),
        hf("NEMO::Enthalpy_Formation", ns), tref("NEMO::Ref_Temperature", ns), cvtr("NEMO::Cvtrs", ns);
    Kokkos::View<int*, mem> nel("NEMO::nElStates", ns);
    Kokkos::View<double**, Kokkos::LayoutRight, mem> the("NEMO::CharElTemp", ns, maxEl),
        gdeg("NEMO::ElDegeneracy", ns, maxEl);

    auto hM = Kokkos::create_mirror_view(M);
    auto hthv = Kokkos::create_mirror_view(thv);
    auto hrot = Kokkos::create_mirror_view(rot);
    auto hhf = Kokkos::create_mirror_view(hf);
    auto htref = Kokkos::create_mirror_view(tref);
    auto hcvtr = Kokkos::create_mirror_view(cvtr);
    auto hnel = Kokkos::create_mirror_view(nel);
    auto hthe = Kokkos::create_mirror_view(the);
    auto hg = Kokkos::create_mirror_view(gdeg);

    const double Ru = gas.Ru;

    for (int s = 0; s < ns; s++) {
      hM(s) = gas.MolarMass[s];
      hthv(s) = gas.CharVibTemp[s];
      hrot(s) = gas.RotationModes[s];
      hhf(s) = gas.Enthalpy_Formation[s];
      htref(s) = gas.Ref_Temperature[s];
      hnel(s) = gas.nElStates[s];
      for (int l = 0; l < maxEl; l++) {
        hthe(s, l) = gas.CharElTemp(s, l);
        hg(s, l) = gas.ElDegeneracy(s, l);
      }
      /*--- Effective GetSpeciesCvTraRot(): only [nEl, nHeavy) is filled, the rest stays 0. ---*/
      hcvtr(s) = 0.0;
    }
    for (int s = gas.nEl; s < gas.nHeavy; s++)
      hcvtr(s) = (3.0 / 2.0 + gas.RotationModes[s] / 2.0) * Ru / gas.MolarMass[s];

    Kokkos::deep_copy(M, hM);
    Kokkos::deep_copy(thv, hthv);
    Kokkos::deep_copy(rot, hrot);
    Kokkos::deep_copy(hf, hhf);
    Kokkos::deep_copy(tref, htref);
    Kokkos::deep_copy(cvtr, hcvtr);
    Kokkos::deep_copy(nel, hnel);
    Kokkos::deep_copy(the, hthe);
    Kokkos::deep_copy(gdeg, hg);

    NEMOGasTables t;
    t.nSpecies = ns;
    t.nHeavy = gas.nHeavy;
    t.nEl = gas.nEl;
    t.nDim = gas.nDim;
    t.nReactions = gas.nReactions;
    t.maxEl = maxEl;
    t.ionization = gas.ionization;
    t.Ru = Ru;
    t.MolarMass = M;
    t.CharVibTemp = thv;
    t.RotationModes = rot;
    t.Enthalpy_Formation = hf;
    t.Ref_Temperature = tref;
    t.Cvtrs = cvtr;
    t.nElStates = nel;
    t.CharElTemp = the;
    t.ElDegeneracy = gdeg;
    return t;
  }
};

NEMOGasTables BuildNEMOGasTables(const CSU2TCLib& gas) { return NEMOGasTablesAccess::Build(gas); }

}  // namespace nemo
}  // namespace su2kokkos

#endif  // HAVE_KOKKOS
