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

#include <cmath>
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

    /*--- Step 7: reaction data and V-T relaxation constants. ---*/
    const int nr = gas.nReactions;
    Kokkos::View<int**, Kokkos::LayoutRight, mem> rx("NEMO::Reactions", nr, 6);
    Kokkos::View<double*, mem> arrA("NEMO::ArrheniusCoefficient", nr), arrEta("NEMO::ArrheniusEta", nr),
        arrTh("NEMO::ArrheniusTheta", nr), tfa("NEMO::Tcf_a", nr), tfb("NEMO::Tcf_b", nr), tba("NEMO::Tcb_a", nr),
        tbb("NEMO::Tcb_b", nr), diss("NEMO::Diss", ns);
    Kokkos::View<double**, Kokkos::LayoutRight, mem> rxt("NEMO::RxnConstantTable", nr, 30), mwA("NEMO::MW_A", ns, ns),
        mwB("NEMO::MW_B", ns, ns);
    {
      auto hrx = Kokkos::create_mirror_view(rx);
      auto hA = Kokkos::create_mirror_view(arrA);
      auto hEta = Kokkos::create_mirror_view(arrEta);
      auto hTh = Kokkos::create_mirror_view(arrTh);
      auto hfa = Kokkos::create_mirror_view(tfa);
      auto hfb = Kokkos::create_mirror_view(tfb);
      auto hba = Kokkos::create_mirror_view(tba);
      auto hbb = Kokkos::create_mirror_view(tbb);
      auto hdiss = Kokkos::create_mirror_view(diss);
      auto hrxt = Kokkos::create_mirror_view(rxt);
      auto hmwA = Kokkos::create_mirror_view(mwA);
      auto hmwB = Kokkos::create_mirror_view(mwB);

      /*--- GetChemistryEquilConstants(r) fills the member RxnConstantTable from std::string compares of the gas
       *    model name. Call it per reaction in the CPU order (0..nr-1, as the reaction loop does) on the gas object
       *    and keep each table; the member table is restored afterwards so the host object is left unchanged.
       *    (A reaction index not covered by any branch keeps the previous reaction's table, exactly as in the CPU
       *    loop, except for reaction 0, where the CPU would see the table left by the previous point.) ---*/
      auto& g = const_cast<CSU2TCLib&>(gas);
      const su2activematrix saved = g.RxnConstantTable;
      for (int r = 0; r < nr; r++) {
        for (int k = 0; k < 3; k++) {
          hrx(r, k) = static_cast<int>(gas.Reactions(r, 0, k));
          hrx(r, 3 + k) = static_cast<int>(gas.Reactions(r, 1, k));
          if (hrx(r, k) < 0 || hrx(r, k) > ns || hrx(r, 3 + k) < 0 || hrx(r, 3 + k) > ns)
            SU2_MPI::Error("NEMOGasKokkos: reaction species index out of range.", CURRENT_FUNCTION);
        }
        hA(r) = gas.ArrheniusCoefficient[r];
        hEta(r) = gas.ArrheniusEta[r];
        hTh(r) = gas.ArrheniusTheta[r];
        hfa(r) = gas.Tcf_a[r];
        hfb(r) = gas.Tcf_b[r];
        hba(r) = gas.Tcb_a[r];
        hbb(r) = gas.Tcb_b[r];
        g.GetChemistryEquilConstants(r);
        for (int row = 0; row < 6; row++)
          for (int c = 0; c < 5; c++) hrxt(r, row * 5 + c) = gas.RxnConstantTable(row, c);
      }
      g.RxnConstantTable = saved;

      /*--- Millikan-White constants, same expressions as CSU2TCLib::ComputeEveSourceTerm. ---*/
      for (int i = 0; i < ns; i++) {
        hdiss(i) = gas.Diss[i];
        for (int j = 0; j < ns; j++) {
          const su2double mu = gas.MolarMass[i] * gas.MolarMass[j] / (gas.MolarMass[i] + gas.MolarMass[j]);
          hmwA(i, j) = 1.16 * 1E-3 * sqrt(mu) * pow(gas.CharVibTemp[i], 4.0 / 3.0);
          hmwB(i, j) = 0.015 * pow(mu, 0.25);
        }
      }
      Kokkos::deep_copy(rx, hrx);
      Kokkos::deep_copy(arrA, hA);
      Kokkos::deep_copy(arrEta, hEta);
      Kokkos::deep_copy(arrTh, hTh);
      Kokkos::deep_copy(tfa, hfa);
      Kokkos::deep_copy(tfb, hfb);
      Kokkos::deep_copy(tba, hba);
      Kokkos::deep_copy(tbb, hbb);
      Kokkos::deep_copy(diss, hdiss);
      Kokkos::deep_copy(rxt, hrxt);
      Kokkos::deep_copy(mwA, hmwA);
      Kokkos::deep_copy(mwB, hmwB);
    }

    NEMOGasTables t;
    t.frozen = gas.frozen;
    t.Avogadro = AVOGAD_CONSTANT;
    t.Pi = PI_NUMBER;
    t.Reactions = rx;
    t.ArrheniusCoefficient = arrA;
    t.ArrheniusEta = arrEta;
    t.ArrheniusTheta = arrTh;
    t.Tcf_a = tfa;
    t.Tcf_b = tfb;
    t.Tcb_a = tba;
    t.Tcb_b = tbb;
    t.RxnConstantTable = rxt;
    t.MW_A = mwA;
    t.MW_B = mwB;
    t.Diss = diss;
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

  static void Replay(CSU2TCLib& gas, int nDim, const double* V, unsigned char status, const double* firstDP,
                     bool full) {
    const int ns = gas.nSpecies;
    const NEMOPrimIndices idx(ns, nDim);
    vector<su2double> rhos(V + idx.RHOS(), V + idx.RHOS() + ns);
    const double T = V[idx.T()], Tve = V[idx.TVE()];

    if (status & C2P_FINAL_FULL) {
      /*--- Same calls, same order, as the full path of CNEMOEulerVariable::Cons2PrimVar. ---*/
      gas.SetTDStateRhosTTv(rhos, T, Tve);
      if (full) {
        gas.ComputeSpeciesCvVibEle(Tve);
        gas.ComputeSpeciesEve(Tve);
        gas.ComputerhoCvtr();
        gas.ComputerhoCvve();
        gas.ComputePressure();
        gas.ComputeSoundSpeed();
      }
    } else {
      /*--- Early return: only ComputeTemperatures ran (sets rhos, T (clipped), Tve). Density and
       *    Pressure keep the values of the last SetTDStateRhosTTv, which is the first attempt of
       *    this point if that one took the full path, otherwise the previous point (sequential replay). ---*/
      gas.rhos = rhos;
      gas.T = T;
      gas.Tve = Tve;
      if (status & C2P_FIRST_FULL) {
        gas.Density = firstDP[0];
        gas.Pressure = firstDP[1];
      }
    }
    if (full) gas.ComputeGamma();
  }
};

NEMOGasTables BuildNEMOGasTables(const CSU2TCLib& gas) { return NEMOGasTablesAccess::Build(gas); }

void ReplayHostGasState(CSU2TCLib& gas, int nDim, const double* V, unsigned char status, const double* firstDP,
                        bool full) {
  NEMOGasTablesAccess::Replay(gas, nDim, V, status, firstDP, full);
}

}  // namespace nemo
}  // namespace su2kokkos

#endif  // HAVE_KOKKOS
