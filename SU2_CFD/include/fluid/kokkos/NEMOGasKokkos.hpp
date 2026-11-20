/*!
 * \file NEMOGasKokkos.hpp
 * \brief Stateless, device-callable re-implementation of the CSU2TCLib (SU2_NONEQ)
 *        two-temperature thermodynamics needed by NEMO Cons2Prim.
 *
 * Every function is a KOKKOS_INLINE_FUNCTION that takes the read-only gas tables
 * (NEMOGasTables, captured by value) plus per-point inputs, and writes outputs to
 * caller-provided (stack) arrays. The arithmetic deliberately mirrors the expression
 * order of CSU2TCLib.cpp / CNEMOGas.cpp / CNEMOEulerVariable.cpp so that results are
 * bitwise (or near-bitwise, see notes) identical to the CPU reference.
 *
 * Reproduced CPU quirks (intentionally, the CPU path is the ground truth):
 *  - Effective species Cv_tr is the table filled by CSU2TCLib::GetSpeciesCvTraRot,
 *    which only fills [nEl, nHeavy). With ionization (AIR-7) the last species (NO+)
 *    keeps Cvtr = 0 and the electron has Cvtr = 0.
 *  - ComputerhoCvtr sums over [0, nHeavy) while ComputeTemperatures / ComputeSoundSpeed
 *    sum over [nEl, nSpecies).
 *  - ComputedPdU with ionization does "dPdU[s] -= ..." for s in [nEl,nSpecies) but only
 *    assigns dPdU[s] for s in [nEl,nHeavy): dPdU[nSpecies-1] accumulates on top of the
 *    value previously stored in the array. ComputedTdU never writes dTdU[0..nEl).
 *    Those arrays are therefore in/out here, as on the CPU.
 *  - ComputeSoundSpeed uses the unclipped pressure (a = NaN if P < 0) while V[P] and h
 *    use the clipped value 1e-20.
 *  - On a non-physical T/Tve, Cons2Prim returns early: eves, Cvves, dPdU, dTdU, dTvedU
 *    and V[P..rhoCvve] are left untouched.
 *
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#pragma once

#ifdef HAVE_KOKKOS

#include <Kokkos_Core.hpp>

class CSU2TCLib;

namespace su2kokkos {
namespace nemo {

/*--- Compile-time capacities (checked at table build time). ---*/
constexpr int MAXNS = 8;                          /*!< AIR-7 uses 7 species. */
constexpr int MAXEL = 16;                         /*!< NO uses 16 electronic levels (max over N2/AIR-5/AIR-7/ARGON). */
constexpr int MAXNR = 24;                         /*!< AIR-7 uses 22 reactions (unused by thermo, for later steps). */
constexpr int MAXNDIM = 3;
constexpr int MAXNVAR = MAXNS + MAXNDIM + 2;      /*!< rho_s, rho*u_i, rho*E, rho*Eve */
constexpr int MAXNPRIMVAR = MAXNS + MAXNDIM + 10; /*!< NS layout (Euler uses +8). */

/*!
 * \brief Index helpers for the NEMO primitive vector layout:
 *        rho_s | T | Tve | u_i | P | rho | h | a | rhoCvtr | rhoCvve | [mu, mu_t]
 */
struct NEMOPrimIndices {
  int nSpecies, nDim;
  KOKKOS_INLINE_FUNCTION NEMOPrimIndices(int ns, int nd) : nSpecies(ns), nDim(nd) {}
  KOKKOS_INLINE_FUNCTION int RHOS() const { return 0; }
  KOKKOS_INLINE_FUNCTION int T() const { return nSpecies; }
  KOKKOS_INLINE_FUNCTION int TVE() const { return nSpecies + 1; }
  KOKKOS_INLINE_FUNCTION int VEL() const { return nSpecies + 2; }
  KOKKOS_INLINE_FUNCTION int P() const { return nSpecies + nDim + 2; }
  KOKKOS_INLINE_FUNCTION int RHO() const { return nSpecies + nDim + 3; }
  KOKKOS_INLINE_FUNCTION int H() const { return nSpecies + nDim + 4; }
  KOKKOS_INLINE_FUNCTION int A() const { return nSpecies + nDim + 5; }
  KOKKOS_INLINE_FUNCTION int RHOCVTR() const { return nSpecies + nDim + 6; }
  KOKKOS_INLINE_FUNCTION int RHOCVVE() const { return nSpecies + nDim + 7; }
};

/*!
 * \brief Read-only gas data. Views live in the memory space of the default execution space;
 *        the struct is cheap to copy (a handful of scalars and View handles).
 */
struct NEMOGasTables {
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
  using RealView1D = Kokkos::View<const double*, memory_space>;
  using IntView1D = Kokkos::View<const int*, memory_space>;
  using RealView2D = Kokkos::View<const double**, Kokkos::LayoutRight, memory_space>;

  int nSpecies = 0;   /*!< Number of species. */
  int nHeavy = 0;     /*!< Number of heavy species (nSpecies - nEl). */
  int nEl = 0;        /*!< Number of electrons (0 or 1, electron is species 0). */
  int nDim = 0;       /*!< nDim the CPU gas was built with. */
  int nReactions = 0; /*!< Number of reactions. */
  int maxEl = 0;      /*!< Max number of electronic levels (2nd extent of CharElTemp). */
  bool ionization = false;
  double Ru = 0.0;    /*!< Universal gas constant [J/(kmol K)] = 1000*UNIVERSAL_GAS_CONSTANT. */

  RealView1D MolarMass;          /*!< [kg/kmol] */
  RealView1D CharVibTemp;        /*!< [K] */
  RealView1D RotationModes;
  RealView1D Enthalpy_Formation; /*!< [J/kg] */
  RealView1D Ref_Temperature;    /*!< [K] */
  RealView1D Cvtrs;              /*!< Effective GetSpeciesCvTraRot() table [J/(kg K)]. */
  IntView1D nElStates;
  RealView2D CharElTemp;         /*!< (nSpecies, maxEl) */
  RealView2D ElDegeneracy;       /*!< (nSpecies, maxEl) */

  /*--- Step 7: finite-rate chemistry and vibrational-electronic relaxation (CSU2TCLib). ---*/
  using IntView2D = Kokkos::View<const int**, Kokkos::LayoutRight, memory_space>;
  bool frozen = false;           /*!< CNEMOGas::frozen (config FROZEN_MIXTURE): no omega_CV term in the Eve source. */
  double Avogadro = 0.0;         /*!< AVOGAD_CONSTANT [1/kmol] */
  double Pi = 0.0;               /*!< PI_NUMBER */
  IntView2D Reactions;           /*!< (nReactions, 6): [0,3) reactants, [3,6) products, value nSpecies = none
                                      (CSU2TCLib::Reactions(r, 0|1, 0..2)). */
  RealView1D ArrheniusCoefficient, ArrheniusEta, ArrheniusTheta; /*!< (nReactions) */
  RealView1D Tcf_a, Tcf_b, Tcb_a, Tcb_b;                           /*!< (nReactions) rate-controlling T exponents */
  RealView2D RxnConstantTable;   /*!< (nReactions, 6*5): CSU2TCLib::GetChemistryEquilConstants(r) pre-baked at build
                                      time (the CPU re-fills its member table with std::string compares per call). */
  RealView2D MW_A, MW_B;         /*!< (nSpecies, nSpecies) Millikan-White A_sr, B_sr of ComputeEveSourceTerm. */
  RealView1D Diss;               /*!< (nSpecies) dissociation potential (not used by the source terms, kept for Step 8). */

  /*--- Step 8: transport properties (CSU2TCLib, viscous gas objects only). ---*/
  enum TransportModel : int { TRANSPORT_NONE = -1, TRANSPORT_WILKE = 0, TRANSPORT_GUPTAYOS = 1, TRANSPORT_SUTHERLAND = 2 };
  int transModel = TRANSPORT_NONE; /*!< CSU2TCLib::Kind_TransCoeffModel (NONE: inviscid gas object, no tables). */
  RealView2D Blottner;           /*!< (nSpecies, 3) Blottner viscosity coefficients. */
  RealView2D Omega11, Omega22;   /*!< (nSpecies*nSpecies, 4) collision-integral fits, row i*nSpecies+j. */
  double muRef = 0.0, kRef = 0.0, SmRef = 0.0, SkRef = 0.0, TRefSuth = 0.0; /*!< Sutherland constants */
  double coul[12] = {0.0};       /*!< D1_a, C1_a, c1_a, D2_a, C2_a, c2_a, D1_r, C1_r, c1_r, D2_r, C2_r, c2_r */
  double eCGS = 0.0, kB = 0.0, eps = 0.0; /*!< FUND_ELEC_CHARGE_CGS, BOLTZMANN_CONSTANT, EPS */
  IntView2D CatRecomb;           /*!< (nSpecies, 2) CNEMOGas::GetCatalyticRecombination() */
};

/*!
 * \brief Host-side builder: copies the constants out of an existing CSU2TCLib object.
 * \note Defined in SU2_CFD/src/fluid/kokkos/NEMOGasKokkos.cpp.
 */
NEMOGasTables BuildNEMOGasTables(const CSU2TCLib& gas);

/*!
 * \brief Per-point status bits written by the solver-level device Cons2Prim (SetPrimVar semantics).
 *        "FULL" means the call got past the temperature checks, i.e. the CPU would have called
 *        CSU2TCLib::SetTDStateRhosTTv (which sets the gas members Density and Pressure).
 */
enum NEMOCons2PrimStatus : unsigned char {
  C2P_FIRST_NONPHYS = 1, /*!< First attempt non-physical (counted, triggers rollback to Solution_Old). */
  C2P_FIRST_FULL = 2,    /*!< First attempt reached the full path. */
  C2P_FINAL_FULL = 4,    /*!< Final (possibly rolled back) attempt reached the full path. */
};

/*!
 * \brief Host-side replay of the CSU2TCLib member state that CNEMOEulerVariable::SetPrimVar leaves
 *        behind for one point, given the device results (final V, status bits, and the gas
 *        Density/Pressure of the first attempt when it reached the full path but the rollback did not).
 * \param[in] full - false: only rhos, T, Tve, Density, Pressure (what transport properties read);
 *                   true: also the members touched by Cons2PrimVar + ComputeGamma (rhoCvtr, rhoCvve,
 *                   Cvves, SoundSpeed2, gamma, ...), used for the last point so the gas object
 *                   ends in the same state as after the CPU loop.
 * \note Defined in SU2_CFD/src/fluid/kokkos/NEMOGasKokkos.cpp.
 */
void ReplayHostGasState(CSU2TCLib& gas, int nDim, const double* V, unsigned char status, const double* firstDP,
                        bool full);

/*!
 * \brief Step 8: the gas Density / Pressure members at the moment the CPU transport routines of a point run
 *        (CNEMONSVariable::SetTransportProperties after Cons2PrimVar), i.e. the values of the LAST SetTDStateRhosTTv:
 *        this point (final attempt), its first attempt (FirstDP), or an earlier point (carry, also across calls).
 * \param[in,out] dp - Effective {Density, Pressure} of this point; set by the caller for non-full states (carry).
 * \note Defined in SU2_CFD/src/fluid/kokkos/NEMOGasKokkos.cpp: host replay of the members CSU2TCLib transport reads and
 *       of what SetTransportProperties leaves (mus, phis, MolarFracWBE, Cvves, Mu, ...), for the last point.
 */
void ReplayHostTransportState(CSU2TCLib& gas, int nDim, const double* V, unsigned char status, const double* dp,
                              const double* eves);
/*! \brief Host gas Density / Pressure members (initial carry of the device transport stage). */
void HostGasDensityPressure(const CSU2TCLib& gas, double* dp);

/*! \brief True if V (after Cons2Prim) passed the temperature checks, i.e. the full path was taken. */
KOKKOS_INLINE_FUNCTION
bool Cons2PrimReachedFullPath(int nSpecies, bool monoatomic, const double* V) {
  const double Tmin = 50.0, Tmax = 8E4, Tvemin = 50.0, Tvemax = 8E4;
  const double T = V[nSpecies], Tve = V[nSpecies + 1];
  if (T <= Tmin || T >= Tmax || T != T) return false;
  if (!monoatomic && (Tve <= Tvemin || Tve >= Tvemax || Tve != Tve)) return false;
  return true;
}

/*--------------------------------------------------------------------------------------*/
/*--- Species-level thermodynamics (CSU2TCLib).                                      ---*/
/*--------------------------------------------------------------------------------------*/

/*! \brief CSU2TCLib::GetSpeciesCvTraRot (effective values). */
KOKKOS_INLINE_FUNCTION
void GetSpeciesCvTraRot(const NEMOGasTables& g, double* cvtrs) {
  for (int s = 0; s < g.nSpecies; s++) cvtrs[s] = g.Cvtrs(s);
}

/*! \brief CSU2TCLib::ComputeSpeciesCvVibEle. */
KOKKOS_INLINE_FUNCTION
void ComputeSpeciesCvVibEle(const NEMOGasTables& g, double val_T, double* cvves) {
  const double Ru = g.Ru;
  const int iElectron = 0;

  for (int s = 0; s < g.nSpecies; s++) {
    double Cvvs, Cves;
    const double Ms = g.MolarMass(s);

    if (g.ionization && s == iElectron) {
      Cvvs = 0.0;
      Cves = 3.0 / 2.0 * Ru / Ms;
    } else {
      /*--- Vibrational ---*/
      const double thv = g.CharVibTemp(s);
      if (thv != 0.0) {
        const double thoTve = thv / val_T;
        const double exptv = Kokkos::exp(thv / val_T);
        Cvvs = Ru / Ms * thoTve * thoTve * exptv / ((exptv - 1.0) * (exptv - 1.0));
      } else {
        Cvvs = 0.0;
      }

      /*--- Electronic ---*/
      const int nEls = g.nElStates(s);
      if (nEls != 0) {
        double num = 0.0, num2 = 0.0;
        const double th0 = g.CharElTemp(s, 0);
        const double g0 = g.ElDegeneracy(s, 0);
        double denom = g0 * Kokkos::exp(-th0 / val_T);
        double num3 = g0 * (th0 / (val_T * val_T)) * Kokkos::exp(-th0 / val_T);
        for (int iEl = 1; iEl < nEls; iEl++) {
          const double the = g.CharElTemp(s, iEl);
          const double gl = g.ElDegeneracy(s, iEl);
          const double thoTve = the / val_T;
          const double exptv = Kokkos::exp(-the / val_T);
          num += gl * the * exptv;
          denom += gl * exptv;
          num2 += gl * (thoTve * thoTve) * exptv;
          num3 += gl * thoTve / val_T * exptv;
        }
        Cves = Ru / Ms * (num2 / denom - num * num3 / (denom * denom));
      } else {
        Cves = 0.0;
      }
    }
    cvves[s] = Cvvs + Cves;
  }
}

/*! \brief CSU2TCLib::ComputeSpeciesEve. */
KOKKOS_INLINE_FUNCTION
void ComputeSpeciesEve(const NEMOGasTables& g, double val_T, double* eves, bool vibe_only = false) {
  const double Ru = g.Ru;
  const int iElectron = 0;

  for (int s = 0; s < g.nSpecies; s++) {
    double Ev, Eel;
    const double Ms = g.MolarMass(s);

    if (g.ionization && s == iElectron) {
      const double Ef = g.Enthalpy_Formation(s) - Ru / Ms * g.Ref_Temperature(s);
      Eel = (3.0 / 2.0) * Ru / Ms * (val_T - g.Ref_Temperature(s)) + Ef;
      Ev = 0.0;
    } else {
      const double thv = g.CharVibTemp(s);
      if (thv != 0.0)
        Ev = Ru / Ms * thv / (Kokkos::exp(thv / val_T) - 1.0);
      else
        Ev = 0.0;

      double num = 0.0;
      double denom = g.ElDegeneracy(s, 0) * Kokkos::exp(-g.CharElTemp(s, 0) / val_T);
      for (int iEl = 1; iEl < g.nElStates(s); iEl++) {
        num += g.ElDegeneracy(s, iEl) * g.CharElTemp(s, iEl) * Kokkos::exp(-g.CharElTemp(s, iEl) / val_T);
        denom += g.ElDegeneracy(s, iEl) * Kokkos::exp(-g.CharElTemp(s, iEl) / val_T);
      }
      Eel = Ru / Ms * (num / denom);
    }
    if (vibe_only) eves[s] = Ev;
    else eves[s] = Ev + Eel;
  }
}

/*!
 * \brief CSU2TCLib::ComputeMixtureEnergies after SetTDStateRhosTTv(rhos, T, Tve).
 * \param[out] energies - {e_mix, e_ve} per unit mass.
 */
KOKKOS_INLINE_FUNCTION
void ComputeMixtureEnergies(const NEMOGasTables& g, const double* rhos, double T, double Tve, double* energies) {
  const double Ru = g.Ru;

  double Density = 0.0;
  for (int s = 0; s < g.nSpecies; s++) Density += rhos[s];

  double rhoEmix = 0.0;
  double rhoEve = 0.0;

  for (int s = 0; s < g.nEl; s++) {
    rhoEve += rhos[s] * ((3.0 / 2.0) * Ru / g.MolarMass(s) * (Tve - g.Ref_Temperature(s)));
  }

  for (int s = g.nEl; s < g.nSpecies; s++) {
    const double Ms = g.MolarMass(s);
    const double Ef = g.Enthalpy_Formation(s) - Ru / Ms * g.Ref_Temperature(s);

    double Ev;
    const double thv = g.CharVibTemp(s);
    if (thv != 0.0)
      Ev = Ru / Ms * thv / (Kokkos::exp(thv / Tve) - 1.0);
    else
      Ev = 0.0;

    double num = 0.0;
    double denom = g.ElDegeneracy(s, 0) * Kokkos::exp(-g.CharElTemp(s, 0) / Tve);
    for (int iEl = 1; iEl < g.nElStates(s); iEl++) {
      num += g.ElDegeneracy(s, iEl) * g.CharElTemp(s, iEl) * Kokkos::exp(-g.CharElTemp(s, iEl) / Tve);
      denom += g.ElDegeneracy(s, iEl) * Kokkos::exp(-g.CharElTemp(s, iEl) / Tve);
    }
    const double Ee = Ru / Ms * (num / denom);

    rhoEmix += rhos[s] * ((3.0 / 2.0 + g.RotationModes(s) / 2.0) * Ru / Ms * (T - g.Ref_Temperature(s)) + Ev + Ee + Ef);
    rhoEve += rhos[s] * (Ev + Ee);
  }

  energies[0] = rhoEmix / Density;
  energies[1] = rhoEve / Density;
}

/*! \brief CSU2TCLib::ComputeSpeciesEnthalpy (val_Tve unused, as on CPU). */
KOKKOS_INLINE_FUNCTION
void ComputeSpeciesEnthalpy(const NEMOGasTables& g, double val_T, double /*val_Tve*/, const double* val_eves,
                            double* hs) {
  for (int s = 0; s < g.nSpecies; s++) {
    hs[s] = g.Ru / g.MolarMass(s) * val_T + g.Cvtrs(s) * val_T + g.Enthalpy_Formation(s) + val_eves[s];
  }
}

/*!
 * \brief CSU2TCLib::ComputeTemperatures: closed-form T, damped Newton for Tve with
 *        bisection fallback, Tve = T if neither converges.
 */
KOKKOS_INLINE_FUNCTION
void ComputeTemperatures(const NEMOGasTables& g, const double* rhos, double rhoE, double rhoEve, double rhoEvel,
                         double Tve_old, double& T_out, double& Tve_out) {
  const double Ru = g.Ru;
  const int ns = g.nSpecies;

  /*--- Translational temperature ---*/
  double rhoE_f = 0.0;
  double rhoE_ref = 0.0;
  double rhoCvtr = 0.0;
  for (int s = g.nEl; s < ns; s++) {
    rhoCvtr += rhos[s] * g.Cvtrs(s);
    rhoE_ref += rhos[s] * g.Cvtrs(s) * g.Ref_Temperature(s);
    rhoE_f += rhos[s] * (g.Enthalpy_Formation(s) - Ru / g.MolarMass(s) * g.Ref_Temperature(s));
  }

  double T = (rhoE - rhoEve - rhoE_f + rhoE_ref - rhoEvel) / rhoCvtr;

  const double Tmin = 50.0, Tmax = 8E4;
  const double Tvemin = 50.0, Tvemax = 8E4;
  double Tve_o = 50.0;
  double Tve2 = 8E4;

  if (Tve_old < 1) Tve_old = T;
  if (T < Tmin) T = Tmin;
  else if (T > Tmax) T = Tmax;
  if (Tve_old < Tvemin) Tve_old = Tvemin;
  else if (Tve_old > Tvemax) Tve_old = Tvemax;

  const double NRtol = 1.0E-6;
  const double Btol = 1.0E-6;
  const int maxBIter = 100;
  const int maxNIter = 100;
  const double scale = 0.9;

  double Tve = Tve_old;
  bool Bconvg = false;
  bool NRconvg = false;
  double rhoEve_t = 0.0, rhoCvve = 0.0;
  double val_eves[MAXNS], val_cvves[MAXNS];

  for (int iIter = 0; iIter < maxNIter; iIter++) {
    rhoEve_t = rhoCvve = 0.0;
    ComputeSpeciesEve(g, Tve, val_eves);
    ComputeSpeciesCvVibEle(g, Tve, val_cvves);

    for (int s = 0; s < ns; s++) {
      rhoEve_t += rhos[s] * val_eves[s];
      rhoCvve += rhos[s] * val_cvves[s];
    }

    const double f = rhoEve - rhoEve_t;
    const double df = -rhoCvve;
    Tve2 = Tve - (f / df) * scale;

    if ((Kokkos::fabs(Tve2 - Tve) < NRtol) && (Tve > Tvemin) && (Tve < Tvemax)) {
      NRconvg = true;
      Tve = Tve2;
      break;
    }
    Tve = Tve2;
  }

  Tve_o = Tvemin;
  Tve2 = Tvemax;
  if (!NRconvg) {
    for (int iIter = 0; iIter < maxBIter; iIter++) {
      Tve = (Tve_o + Tve2) / 2.0;
      ComputeSpeciesEve(g, Tve, val_eves);
      rhoEve_t = 0.0;
      for (int s = 0; s < ns; s++) rhoEve_t += rhos[s] * val_eves[s];
      if (Kokkos::fabs(rhoEve_t - rhoEve) < Btol) {
        Bconvg = true;
        break;
      }
      if (rhoEve_t > rhoEve) Tve2 = Tve;
      else Tve_o = Tve;
    }
  }

  if (!NRconvg && !Bconvg) Tve = T;

  T_out = T;
  Tve_out = Tve;
}

/*--------------------------------------------------------------------------------------*/
/*--- Mixture-level quantities (CNEMOGas).                                           ---*/
/*--------------------------------------------------------------------------------------*/

/*! \brief CNEMOGas::ComputePressure (unclipped). */
KOKKOS_INLINE_FUNCTION
double ComputePressure(const NEMOGasTables& g, const double* rhos, double T, double Tve) {
  double P = 0.0;
  for (int s = 0; s < g.nEl; s++) P += rhos[s] * g.Ru / g.MolarMass(s) * Tve;
  for (int s = g.nEl; s < g.nSpecies; s++) P += rhos[s] * g.Ru / g.MolarMass(s) * T;
  return P;
}

/*! \brief CNEMOGas::ComputeSoundSpeed; Pressure is the (unclipped) value of ComputePressure. */
KOKKOS_INLINE_FUNCTION
double ComputeSoundSpeed(const NEMOGasTables& g, const double* rhos, double Pressure) {
  double conc = 0.0;
  double rhoCvtr = 0.0;
  double Density = 0.0;
  for (int s = 0; s < g.nSpecies; s++) Density += rhos[s];
  for (int s = g.nEl; s < g.nSpecies; s++) {
    conc += rhos[s] / g.MolarMass(s);
    rhoCvtr += rhos[s] * g.Cvtrs(s);
  }
  const double SoundSpeed2 = (1.0 + g.Ru / rhoCvtr * conc) * Pressure / Density;
  return Kokkos::sqrt(SoundSpeed2);
}

/*! \brief CNEMOGas::ComputerhoCvtr (sums over [0, nHeavy) as on CPU). */
KOKKOS_INLINE_FUNCTION
double ComputerhoCvtr(const NEMOGasTables& g, const double* rhos) {
  double rhoCvtr = 0.0;
  for (int s = 0; s < g.nHeavy; s++) rhoCvtr += rhos[s] * g.Cvtrs(s);
  return rhoCvtr;
}

/*! \brief CNEMOGas::ComputerhoCvve; optionally returns species Cvve at Tve. */
KOKKOS_INLINE_FUNCTION
double ComputerhoCvve(const NEMOGasTables& g, const double* rhos, double Tve, double* cvves_out = nullptr) {
  double cvves[MAXNS];
  ComputeSpeciesCvVibEle(g, Tve, cvves);
  double rhoCvve = 0.0;
  for (int s = 0; s < g.nSpecies; s++) rhoCvve += rhos[s] * cvves[s];
  if (cvves_out != nullptr)
    for (int s = 0; s < g.nSpecies; s++) cvves_out[s] = cvves[s];
  return rhoCvve;
}

/*! \brief CNEMOGas::ComputeGamma. */
KOKKOS_INLINE_FUNCTION
double ComputeGamma(const NEMOGasTables& g, const double* rhos) {
  const double rhoCvtr = ComputerhoCvtr(g, rhos);
  double rhoR = 0.0;
  for (int s = 0; s < g.nSpecies; s++) rhoR += rhos[s] * g.Ru / g.MolarMass(s);
  return rhoR / rhoCvtr + 1;
}

/*!
 * \brief CNEMOGas::ComputedPdU. dPdU is in/out: with ionization, dPdU[nSpecies-1] is
 *        updated with "-=" on its previous content (CPU behaviour).
 */
KOKKOS_INLINE_FUNCTION
void ComputedPdU(const NEMOGasTables& g, int nDim, const double* V, const double* val_eves, double* val_dPdU) {
  const double Ru = g.Ru;
  const int ns = g.nSpecies;
  const NEMOPrimIndices idx(ns, nDim);

  double rhos[MAXNS];
  for (int s = 0; s < ns; s++) rhos[s] = V[idx.RHOS() + s];

  double rho_el = 0.0;
  if (g.ionization) rho_el = rhos[0];

  const double rhoCvtr = V[idx.RHOCVTR()];
  const double rhoCvve = V[idx.RHOCVVE()];
  const double T = V[idx.T()];

  double sqvel = 0.0;
  double conc = 0.0;
  for (int iDim = 0; iDim < nDim; iDim++) sqvel += V[idx.VEL() + iDim] * V[idx.VEL() + iDim];
  for (int s = 0; s < ns; s++) conc += rhos[s] / g.MolarMass(s);

  double ef = 0.0;
  for (int s = g.nEl; s < g.nHeavy; s++) {
    const double Ms = g.MolarMass(s);
    const double Tref = g.Ref_Temperature(s);
    ef = g.Enthalpy_Formation(s) - Ru / Ms * Tref;
    val_dPdU[s] = T * Ru / Ms + Ru * conc / rhoCvtr * (-g.Cvtrs(s) * (T - Tref) - ef + 0.5 * sqvel);
  }

  if (g.ionization) {
    for (int s = g.nEl; s < ns; s++) {
      val_dPdU[s] -= rho_el * Ru / g.MolarMass(0) * (val_eves[s]) / rhoCvve;
    }
    ef = g.Enthalpy_Formation(0) - Ru / g.MolarMass(0) * g.Ref_Temperature(0);
    val_dPdU[0] = Ru * conc / rhoCvtr * (-ef + 0.5 * sqvel) + Ru / g.MolarMass(0) * V[idx.TVE()] -
                  rho_el * Ru / g.MolarMass(0) * (-3.0 / 2.0 * Ru / g.MolarMass(0) * V[idx.TVE()]) / rhoCvve;
  }

  for (int iDim = 0; iDim < nDim; iDim++) val_dPdU[ns + iDim] = -conc * Ru * V[idx.VEL() + iDim] / rhoCvtr;

  val_dPdU[ns + nDim] = conc * Ru / rhoCvtr;
  val_dPdU[ns + nDim + 1] = -val_dPdU[ns + nDim] + rho_el * Ru / g.MolarMass(0) * 1.0 / rhoCvve;
}

/*! \brief CNEMOGas::ComputedTdU. dTdU[0..nEl) is left untouched (CPU behaviour). */
KOKKOS_INLINE_FUNCTION
void ComputedTdU(const NEMOGasTables& g, int nDim, const double* V, double* val_dTdU) {
  const double Ru = g.Ru;
  const int ns = g.nSpecies;
  const NEMOPrimIndices idx(ns, nDim);

  const double T = V[idx.T()];
  const double rhoCvtr = V[idx.RHOCVTR()];

  double v2 = 0.0;
  for (int iDim = 0; iDim < nDim; ++iDim) v2 += V[idx.VEL() + iDim] * V[idx.VEL() + iDim];

  for (int s = g.nEl; s < ns; s++) {
    const double ef = g.Enthalpy_Formation(s) - Ru / g.MolarMass(s) * g.Ref_Temperature(s);
    val_dTdU[s] = (-ef + 0.5 * v2 + g.Cvtrs(s) * (g.Ref_Temperature(s) - T)) / rhoCvtr;
  }

  for (int iDim = 0; iDim < nDim; iDim++) val_dTdU[ns + iDim] = -V[idx.VEL() + iDim] / V[idx.RHOCVTR()];

  val_dTdU[ns + nDim] = 1.0 / V[idx.RHOCVTR()];
  val_dTdU[ns + nDim + 1] = -1.0 / V[idx.RHOCVTR()];
}

/*! \brief CNEMOGas::ComputedTvedU. */
KOKKOS_INLINE_FUNCTION
void ComputedTvedU(const NEMOGasTables& g, int nDim, const double* V, const double* val_eves, double* val_dTvedU) {
  const int ns = g.nSpecies;
  const NEMOPrimIndices idx(ns, nDim);
  const double rhoCvve = V[idx.RHOCVVE()];

  for (int s = 0; s < ns; s++) val_dTvedU[s] = -val_eves[s] / rhoCvve;
  for (int iDim = 0; iDim < nDim; iDim++) val_dTvedU[ns + iDim] = 0.0;
  val_dTvedU[ns + nDim] = 0.0;
  val_dTvedU[ns + nDim + 1] = 1.0 / rhoCvve;
}

/*--------------------------------------------------------------------------------------*/
/*--- Step 7: finite-rate chemistry and V-T relaxation (CSU2TCLib).                  ---*/
/*--------------------------------------------------------------------------------------*/

/*!
 * \brief CSU2TCLib::ComputeKeqConstants (with the pre-baked GetChemistryEquilConstants table).
 * \note Reproduces the CPU index arithmetic exactly, including its quirk: "unsigned short pwr = floor(log10(N))"
 *       and "unsigned short iIndex = int(pwr) - 14" wrap for N < 1e14 cm^-3 (iIndex = 65535 - ..., i.e. >= 5), which
 *       selects the HIGHEST-density row 5 instead of row 0 (low-density / high-altitude cases). The double ->
 *       unsigned short conversion is emulated as (unsigned short)(int)x, i.e. what GCC/x86-64 emits (cvttsd2si into a
 *       32-bit register, keep the low 16 bits), also for N < 1 and NaN. A config-guarded fix is a TODO (not done, the
 *       CPU path is the reference).
 */
KOKKOS_INLINE_FUNCTION
void ComputeKeqConstants(const NEMOGasTables& g, int iReaction, const double* rhos, double* A) {
  /*--- Calculate mixture number density ---*/
  double N = 0.0;
  for (int s = 0; s < g.nSpecies; s++) {
    N += rhos[s] / g.MolarMass(s) * g.Avogadro;
  }

  /*--- Convert number density from 1/m^3 to 1/cm^3 for table look-up ---*/
  N = N * (1E-6);

  /*--- Determine table index based on mixture N ---*/
  const unsigned short tbl_offset = 14;
  const unsigned short pwr = static_cast<unsigned short>(static_cast<int>(Kokkos::floor(Kokkos::log10(N))));

  /*--- Bound the interpolation to table limit values ---*/
  const unsigned short iIndex = static_cast<unsigned short>(int(pwr) - tbl_offset);
  auto tbl = [&](int row, int col) { return g.RxnConstantTable(iReaction, row * 5 + col); };
  if (iIndex <= 0) {
    for (int ii = 0; ii < 5; ii++) A[ii] = tbl(0, ii);
    return;
  }
  if (iIndex >= 5) {
    for (int ii = 0; ii < 5; ii++) A[ii] = tbl(5, ii);
    return;
  }

  /*--- Calculate interpolation denominator terms avoiding pow() ---*/
  double tmp1 = 1.0;
  double tmp2 = 1.0;
  for (unsigned short ii = 0; ii < pwr; ii++) {
    tmp1 *= 10.0;
    tmp2 *= 10.0;
  }
  tmp2 *= 10.0;

  /*--- Interpolate ---*/
  for (int ii = 0; ii < 5; ii++) {
    A[ii] = (tbl(iIndex + 1, ii) - tbl(iIndex, ii)) / (tmp2 - tmp1) * (N - tmp1) + tbl(iIndex, ii);
  }
}

/*!
 * \brief CSU2TCLib::ComputeNetProductionRates (+ ChemistryJacobian per reaction if implicit) after
 *        SetTDStateRhosTTv(V[rho_s], V[T], V[Tve]).
 * \param[out] ws - Species net production rates (nSpecies), what the CPU keeps in its "ws" member (read again by
 *                  ComputeEveSourceTerm).
 * \param[in,out] jac - nVar x nVar row-major, accumulated into ("+=", the caller zeroes it), not scaled by Volume.
 *                      Untouched if !implicit. dTdU/dTvedU may be nullptr if !implicit.
 */
KOKKOS_INLINE_FUNCTION
void ComputeNetProductionRates(const NEMOGasTables& g, int nDim, bool implicit, const double* V, const double* eve,
                               const double* cvve, const double* dTdU, const double* dTvedU, double* ws,
                               double* jac) {
  const int nSpecies = g.nSpecies;
  const int nEve = nSpecies + nDim + 1;
  const int nVar = nSpecies + nDim + 2;
  const double* rhos = V;
  const double T = V[nSpecies], Tve = V[nSpecies + 1];

  for (int s = 0; s < nSpecies; s++) ws[s] = 0.0;

  /*--- Define artificial chemistry parameters ---*/
  const double T_min = 800.0;
  const double epsilon = 80;

  for (int iReaction = 0; iReaction < g.nReactions; iReaction++) {
    /*--- Determine the rate-controlling temperature ---*/
    const double af = g.Tcf_a(iReaction);
    const double bf = g.Tcf_b(iReaction);
    const double ab = g.Tcb_a(iReaction);
    const double bb = g.Tcb_b(iReaction);
    const double Trxnf = Kokkos::pow(T, af) * Kokkos::pow(Tve, bf);
    const double Trxnb = Kokkos::pow(T, ab) * Kokkos::pow(Tve, bb);

    /*--- Calculate the modified temperature ---*/
    const double Thf = 0.5 * (Trxnf + T_min + Kokkos::sqrt((Trxnf - T_min) * (Trxnf - T_min) + epsilon * epsilon));
    const double Thb = 0.5 * (Trxnb + T_min + Kokkos::sqrt((Trxnb - T_min) * (Trxnb - T_min) + epsilon * epsilon));

    /*--- Get the Keq & Arrhenius coefficients ---*/
    double A[5];
    ComputeKeqConstants(g, iReaction, rhos, A);

    /*--- Calculate Keq ---*/
    const double Keq = Kokkos::exp(A[0] * (Thb / 1E4) + A[1] + A[2] * Kokkos::log(1E4 / Thb) + A[3] * (1E4 / Thb) +
                                   A[4] * (1E4 / Thb) * (1E4 / Thb));

    /*--- Calculate rate coefficients ---*/
    const double kf = g.ArrheniusCoefficient(iReaction) * Kokkos::exp(g.ArrheniusEta(iReaction) * Kokkos::log(Thf)) *
                      Kokkos::exp(-g.ArrheniusTheta(iReaction) / Thf);
    const double kfb = g.ArrheniusCoefficient(iReaction) * Kokkos::exp(g.ArrheniusEta(iReaction) * Kokkos::log(Thb)) *
                       Kokkos::exp(-g.ArrheniusTheta(iReaction) / Thb);
    const double kb = kfb / Keq;

    /*--- Determine production & destruction of each species ---*/
    double fwdRxn = 1.0;
    double bkwRxn = 1.0;
    for (int ii = 0; ii < 3; ii++) {
      /*--- Reactants ---*/
      const int iSpecies = g.Reactions(iReaction, ii);
      if (iSpecies != nSpecies) fwdRxn *= 0.001 * rhos[iSpecies] / g.MolarMass(iSpecies);

      /*--- Products ---*/
      const int jSpecies = g.Reactions(iReaction, 3 + ii);
      if (jSpecies != nSpecies) {
        bkwRxn *= 0.001 * rhos[jSpecies] / g.MolarMass(jSpecies);
      }
    }

    fwdRxn = 1000.0 * kf * fwdRxn;
    bkwRxn = 1000.0 * kb * bkwRxn;

    for (int ii = 0; ii < 3; ii++) {
      /*--- Products ---*/
      int iSpecies = g.Reactions(iReaction, 3 + ii);
      if (iSpecies != nSpecies) ws[iSpecies] += g.MolarMass(iSpecies) * (fwdRxn - bkwRxn);

      /*--- Reactants ---*/
      iSpecies = g.Reactions(iReaction, ii);
      if (iSpecies != nSpecies) ws[iSpecies] -= g.MolarMass(iSpecies) * (fwdRxn - bkwRxn);
    }

    if (!implicit) continue;

    /*====================== CSU2TCLib::ChemistryJacobian(iReaction) ======================*/
    double dkf[MAXNVAR], dkb[MAXNVAR], dRfok[MAXNVAR], dRbok[MAXNVAR];
    int alphak[MAXNS], betak[MAXNS];
    for (int iVar = 0; iVar < nVar; iVar++) {
      dkf[iVar] = 0.0;
      dRfok[iVar] = 0.0;
      dkb[iVar] = 0.0;
      dRbok[iVar] = 0.0;
    }
    for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
      alphak[iSpecies] = 0;
      betak[iSpecies] = 0;
    }

    /*--- Extract additional Arrhenius information ---*/
    const double eta = g.ArrheniusEta(iReaction);
    const double theta = g.ArrheniusTheta(iReaction);

    /*--- Derivative of modified temperature wrt Trxnf ---*/
    const double dThf = 0.5 * (1.0 + (Trxnf - T_min) / Kokkos::sqrt((Trxnf - T_min) * (Trxnf - T_min) + epsilon * epsilon));
    const double dThb = 0.5 * (1.0 + (Trxnb - T_min) / Kokkos::sqrt((Trxnb - T_min) * (Trxnb - T_min) + epsilon * epsilon));

    /*--- Fwd rate coefficient derivatives ---*/
    double coeff = kf * (eta / Thf + theta / (Thf * Thf)) * dThf;
    for (int iVar = 0; iVar < nVar; iVar++) {
      dkf[iVar] = coeff * (af * Trxnf / T * dTdU[iVar] + bf * Trxnf / Tve * dTvedU[iVar]);
    }

    /*--- Bkwd rate coefficient derivatives ---*/
    coeff = kb * (eta / Thb + theta / (Thb * Thb)) * dThb;
    for (int iVar = 0; iVar < nVar; iVar++) {
      dkb[iVar] = coeff * (ab * Trxnb / T * dTdU[iVar] + bb * Trxnb / Tve * dTvedU[iVar]) -
                  kb * ((A[0] * Thb / 1E4 - A[2] - A[3] * 1E4 / Thb - 2 * A[4] * (1E4 / Thb) * (1E4 / Thb)) / Thb) *
                      dThb * (ab * Trxnb / T * dTdU[iVar] + bb * Trxnb / Tve * dTvedU[iVar]);
    }

    /*--- Rxn rate derivatives ---*/
    for (int ii = 0; ii < 3; ii++) {
      /*--- Products ---*/
      int iSpecies = g.Reactions(iReaction, 3 + ii);
      if (iSpecies != nSpecies) betak[iSpecies]++;

      /*--- Reactants ---*/
      iSpecies = g.Reactions(iReaction, ii);
      if (iSpecies != nSpecies) alphak[iSpecies]++;
    }

    /*--- pow(double, int) is std::pow(double, double(int)) (C++11 promotion), written out explicitly. ---*/
    for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
      // Fwd
      dRfok[iSpecies] = 0.001 * alphak[iSpecies] / g.MolarMass(iSpecies) *
                        Kokkos::pow(0.001 * rhos[iSpecies] / g.MolarMass(iSpecies),
                                    double(alphak[iSpecies] - 1 > 0 ? alphak[iSpecies] - 1 : 0));

      for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++)
        if (jSpecies != iSpecies)
          dRfok[iSpecies] *= Kokkos::pow(0.001 * rhos[jSpecies] / g.MolarMass(jSpecies), double(alphak[jSpecies]));
      dRfok[iSpecies] *= 1000.0;

      // Bkw
      dRbok[iSpecies] = 0.001 * betak[iSpecies] / g.MolarMass(iSpecies) *
                        Kokkos::pow(0.001 * rhos[iSpecies] / g.MolarMass(iSpecies),
                                    double(betak[iSpecies] - 1 > 0 ? betak[iSpecies] - 1 : 0));

      for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++)
        if (jSpecies != iSpecies)
          dRbok[iSpecies] *= Kokkos::pow(0.001 * rhos[jSpecies] / g.MolarMass(jSpecies), double(betak[jSpecies]));
      dRbok[iSpecies] *= 1000.0;
    }

#define NEMO_JAC(r, c) jac[(r) * nVar + (c)]
    for (int ii = 0; ii < 3; ii++) {
      /*--- Products ---*/
      int iSpecies = g.Reactions(iReaction, 3 + ii);
      if (iSpecies != nSpecies) {
        const double Ms = g.MolarMass(iSpecies);
        for (int iVar = 0; iVar < nVar; iVar++) {
          NEMO_JAC(iSpecies, iVar) +=
              Ms * (dkf[iVar] * (fwdRxn / kf) + kf * dRfok[iVar] - dkb[iVar] * (bkwRxn / kb) - kb * dRbok[iVar]);
          NEMO_JAC(nEve, iVar) +=
              Ms * (dkf[iVar] * (fwdRxn / kf) + kf * dRfok[iVar] - dkb[iVar] * (bkwRxn / kb) - kb * dRbok[iVar]) *
              eve[iSpecies];
        }

        for (int jVar = 0; jVar < nVar; jVar++) {
          NEMO_JAC(nEve, jVar) += Ms * (fwdRxn - bkwRxn) * cvve[iSpecies] * dTvedU[jVar];
        }
      }

      /*--- Reactants ---*/
      iSpecies = g.Reactions(iReaction, ii);
      if (iSpecies != nSpecies) {
        const double Ms = g.MolarMass(iSpecies);
        for (int iVar = 0; iVar < nVar; iVar++) {
          NEMO_JAC(iSpecies, iVar) -=
              Ms * (dkf[iVar] * (fwdRxn / kf) + kf * dRfok[iVar] - dkb[iVar] * (bkwRxn / kb) - kb * dRbok[iVar]);
          NEMO_JAC(nEve, iVar) -=
              Ms * (dkf[iVar] * (fwdRxn / kf) + kf * dRfok[iVar] - dkb[iVar] * (bkwRxn / kb) - kb * dRbok[iVar]) *
              eve[iSpecies];
        }

        for (int jVar = 0; jVar < nVar; jVar++) {
          NEMO_JAC(nEve, jVar) -= Ms * (fwdRxn - bkwRxn) * cvve[iSpecies] * dTvedU[jVar];
        }
      }
    }
#undef NEMO_JAC
  }
}

/*!
 * \brief CSU2TCLib::ComputeEveSourceTerm after SetTDStateRhosTTv(rhos, T, Tve) (Landau-Teller, Millikan-White with
 *        Park's limiting cross section, + sum ws*Ev if !frozen).
 * \param[in] ws - Net production rates of the PRECEDING ComputeNetProductionRates call (the CPU's hidden "ws" member
 *                 dependency made explicit); unused if frozen.
 * \param[out] taus, eve_eq, eve_v - Members the CPU leaves for GetEveSourceTermJacobian: relaxation times,
 *                 vibrational energies at T and at Tve (ComputeSpeciesEve(.., true)).
 */
KOKKOS_INLINE_FUNCTION
double ComputeEveSourceTerm(const NEMOGasTables& g, const double* rhos, double T, double Tve, const double* ws,
                            double* taus, double* eve_eq, double* eve_v) {
  const int nSpecies = g.nSpecies;
  const double Pressure = ComputePressure(g, rhos, T, Tve);
  double MolarFrac[MAXNS];

  double omegaVT = 0.0;
  double omegaCV = 0.0;

  /*--- Calculate mole fractions ---*/
  double N = 0.0;
  double conc = 0.0;
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    conc += rhos[iSpecies] / g.MolarMass(iSpecies);
    N += rhos[iSpecies] / g.MolarMass(iSpecies) * g.Avogadro;
  }

  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++)
    MolarFrac[iSpecies] = (rhos[iSpecies] / g.MolarMass(iSpecies)) / conc;

  /*--- Compute Eve and Eve* ---*/
  ComputeSpeciesEve(g, T, eve_eq, true);
  ComputeSpeciesEve(g, Tve, eve_v, true);

  /*--- Loop over species to calculate source term --*/
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    /*--- Millikan & White relaxation time (A_sr, B_sr pre-baked, same expressions) ---*/
    double num = 0.0;
    double denom = 0.0;
    for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++) {
      const double A_sr = g.MW_A(iSpecies, jSpecies);
      const double B_sr = g.MW_B(iSpecies, jSpecies);
      const double tau_sr = 101325.0 / Pressure * Kokkos::exp(A_sr * (Kokkos::pow(T, -1.0 / 3.0) - B_sr) - 18.42);

      num += MolarFrac[jSpecies];
      denom += MolarFrac[jSpecies] / tau_sr;
    }

    const double tauMW = num / denom;

    /*--- Park limiting cross section ---*/
    const double Cs = Kokkos::sqrt((8.0 * g.Ru * T) / (g.Pi * g.MolarMass(iSpecies)));
    const double sig_s = 3E-21 * (2.5E9) / (T * T);

    const double tauP = 1 / (sig_s * Cs * N);

    /*--- Species relaxation time ---*/
    taus[iSpecies] = tauMW + tauP;

    /*--- Add species contribution to residual ---*/
    omegaVT += rhos[iSpecies] * (eve_eq[iSpecies] - eve_v[iSpecies]) / taus[iSpecies];
  }

  /*--- Vibrational energy change due to chemical reactions ---*/
  if (!g.frozen) {
    for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) omegaCV += ws[iSpecies] * eve_v[iSpecies];
  }

  return omegaVT + omegaCV;
}

/*!
 * \brief CSU2TCLib::GetEveSourceTermJacobian (after ComputeEveSourceTerm, whose taus / eve_eq it reads).
 * \note As on the CPU the parameter "eve" (the node's Eve, vib + el) shadows the member of the same name in the
 *       last loop, so d(omega)/d(rho_s) uses (Ev_eq(T) - Eve_node)/tau_s.
 * \param[in,out] jac - nVar x nVar row-major, accumulated into, not scaled by Volume.
 */
KOKKOS_INLINE_FUNCTION
void GetEveSourceTermJacobian(const NEMOGasTables& g, int nDim, const double* rhos, double T, const double* eve,
                              const double* cvve, const double* dTdU, const double* dTvedU, const double* taus,
                              const double* eve_eq, double* jac) {
  const int nSpecies = g.nSpecies;
  const int nEv = nSpecies + nDim + 1;
  const int nVar = nSpecies + nDim + 2;

  /*--- Compute Cvvs ---*/
  double cvve_eq[MAXNS];
  ComputeSpeciesCvVibEle(g, T, cvve_eq);

  /*--- Loop through species ---*/
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    for (int iVar = 0; iVar < nVar; iVar++) {
      jac[nEv * nVar + iVar] +=
          rhos[iSpecies] / taus[iSpecies] * (cvve_eq[iSpecies] * dTdU[iVar] - cvve[iSpecies] * dTvedU[iVar]);
    }
  }

  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++)
    jac[nEv * nVar + iSpecies] += (eve_eq[iSpecies] - eve[iSpecies]) / taus[iSpecies];
}

/*--------------------------------------------------------------------------------------*/
/*--- Step 8: transport properties (CSU2TCLib, CPU statement order).                 ---*/
/*--- The gas "state" (members rhos, T, Tve, Density, Pressure of the CPU object) is  ---*/
/*--- passed explicitly; the hidden ViscosityWBE -> ThermalConductivitiesWBE          ---*/
/*--- dependency (mus, phis, MolarFracWBE members) is made explicit by computing both ---*/
/*--- in ComputeTransport.                                                            ---*/
/*--------------------------------------------------------------------------------------*/

/*! \brief The gas object state the CPU transport routines read. */
struct GasState {
  const double* rhos;
  double T, Tve, Density, Pressure;
};

/*! \brief CSU2TCLib::ComputeCollisionCrossSection. */
KOKKOS_INLINE_FUNCTION
double ComputeCollisionCrossSection(const NEMOGasTables& g, const GasState& s, int iSpecies, int jSpecies, double T,
                                    bool d1, bool coulomb) {
  const double pi = g.Pi;
  const double Na = g.Avogadro;
  const int ij = iSpecies * g.nSpecies + jSpecies;

  if (coulomb) {
    const double e_cgs = g.eCGS;
    const double kb_cgs = g.kB * 1E7;
    const double ne_cgs = Na * s.rhos[0] / g.MolarMass(0) * 1E-6;

    /*--- pow(x, 2) is x*x (exactly rounded, what GCC emits for std::pow(double, int 2)). ---*/
    const double debyeLength = Kokkos::sqrt(kb_cgs * T / 4 / pi / ne_cgs / (e_cgs * e_cgs));
    const double T_star = debyeLength / ((e_cgs * e_cgs) / (kb_cgs * T));
    const double dl = debyeLength / T;
    const double* c = g.coul;

    if (g.Omega11(ij, 0) == 1.0 && d1) {
      return 1E-20 * 5E15 * pi * (dl * dl) * Kokkos::log(c[0] * T_star * (1 - c[1] * Kokkos::exp(-c[2] * T_star)) + 1);
    }
    if (g.Omega11(ij, 0) == -1.0 && d1) {
      return 1E-20 * 5E15 * pi * (dl * dl) * Kokkos::log(c[6] * T_star * (1 - c[7] * Kokkos::exp(-c[8] * T_star)) + 1);
    } else if (g.Omega22(ij, 0) == 1.0 && !d1) {
      return 1E-20 * 5E15 * pi * (dl * dl) * Kokkos::log(c[3] * T_star * (1 - c[4] * Kokkos::exp(-c[5] * T_star)) + 1);
    } else {
      return 1E-20 * 5E15 * pi * (dl * dl) * Kokkos::log(c[9] * T_star * (1 - c[10] * Kokkos::exp(-c[11] * T_star)) + 1);
    }
  }
  if (d1) {
    return 1E-20 * g.Omega11(ij, 3) *
           Kokkos::pow(T, g.Omega11(ij, 0) * Kokkos::log(T) * Kokkos::log(T) + g.Omega11(ij, 1) * Kokkos::log(T) +
                              g.Omega11(ij, 2));
  }
  return 1E-20 * g.Omega22(ij, 3) *
         Kokkos::pow(T, g.Omega22(ij, 0) * Kokkos::log(T) * Kokkos::log(T) + g.Omega22(ij, 1) * Kokkos::log(T) +
                            g.Omega22(ij, 2));
}

/*! \brief CSU2TCLib::ComputeCollisionDelta. */
KOKKOS_INLINE_FUNCTION
double ComputeCollisionDelta(const NEMOGasTables& g, const GasState& s, int iSpecies, int jSpecies, double Mi,
                             double Mj, double T, bool d1) {
  bool coulomb = false;
  if (Kokkos::fabs(g.Omega11(iSpecies * g.nSpecies + jSpecies, 0)) == 1.0 && g.ionization) coulomb = true;

  const double Omega_ij = ComputeCollisionCrossSection(g, s, iSpecies, jSpecies, T, d1, coulomb);
  const double pi = g.Pi;
  const double Ru = g.Ru;
  double delta = 0.0;

  if (d1) {
    delta = 8.0 / 3.0 * Kokkos::sqrt((2.0 * Mi * Mj) / (pi * Ru * T * (Mi + Mj))) * Omega_ij;
  } else {
    delta = 16.0 / 5.0 * Kokkos::sqrt((2.0 * Mi * Mj) / (pi * Ru * T * (Mi + Mj))) * Omega_ij;
  }
  return Kokkos::fmin(delta, 1E16);
}

/*! \brief Mole fractions as MolarFracWBE (DiffusionCoeffWBE / ViscosityWBE recompute them identically). */
KOKKOS_INLINE_FUNCTION
void MolarFractionsWBE(const NEMOGasTables& g, const double* rhos, double* X) {
  double conc = 0.0;
  for (int s = 0; s < g.nSpecies; s++) {
    X[s] = rhos[s] / g.MolarMass(s);
    conc += X[s];
  }
  for (int s = 0; s < g.nSpecies; s++) X[s] = X[s] / conc;
}

/*! \brief CSU2TCLib::DiffusionCoeffWBE. */
KOKKOS_INLINE_FUNCTION
void DiffusionCoeffWBE(const NEMOGasTables& g, const GasState& st, double* Ds) {
  const int nSpecies = g.nSpecies;
  double X[MAXNS];
  MolarFractionsWBE(g, st.rhos, X);

  double M = 0.0;
  for (int s = 0; s < nSpecies; s++) M += g.MolarMass(s) * X[s];
  M = M * 1E-3;

  double Dij[MAXNS * MAXNS];
  for (int k = 0; k < nSpecies * nSpecies; k++) Dij[k] = 0.0;

  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    const double Mi = g.MolarMass(iSpecies) * 1E-3;
    for (int jSpecies = iSpecies; jSpecies < nSpecies; jSpecies++) {
      const double Mj = g.MolarMass(jSpecies) * 1E-3;

      bool coulomb = false;
      if (Kokkos::fabs(g.Omega11(iSpecies * nSpecies + jSpecies, 0)) == 1.0 && g.ionization) coulomb = true;

      const double T_col = (iSpecies == 0 && g.ionization) ? st.Tve : st.T;
      const double Omega_ij = ComputeCollisionCrossSection(g, st, iSpecies, jSpecies, T_col, true, coulomb) / g.Pi;

      Dij[iSpecies * nSpecies + jSpecies] = 7.1613E-25 * M * Kokkos::sqrt(st.T * (1 / Mi + 1 / Mj)) / (st.Density * Omega_ij);
      Dij[jSpecies * nSpecies + iSpecies] = 7.1613E-25 * M * Kokkos::sqrt(st.T * (1 / Mi + 1 / Mj)) / (st.Density * Omega_ij);
    }
  }

  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    Ds[iSpecies] = 0.0;
    double denom = 0.0;
    for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++) {
      if (jSpecies != iSpecies) denom += X[jSpecies] / Dij[iSpecies * nSpecies + jSpecies];
    }
    if (nSpecies == 1) Ds[0] = 0;
    else Ds[iSpecies] = (1 - X[iSpecies]) / denom;
  }
}

/*! \brief CSU2TCLib::ViscosityWBE (+ the mus, phis, MolarFracWBE members ThermalConductivitiesWBE reads). */
KOKKOS_INLINE_FUNCTION
double ViscosityWBE(const NEMOGasTables& g, const GasState& st, double* X, double* mus, double* phis) {
  const int nSpecies = g.nSpecies;
  MolarFractionsWBE(g, st.rhos, X);

  for (int s = 0; s < nSpecies; s++)
    mus[s] = 0.1 * Kokkos::exp((g.Blottner(s, 0) * Kokkos::log(st.T) + g.Blottner(s, 1)) * Kokkos::log(st.T) +
                               g.Blottner(s, 2));

  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    phis[iSpecies] = 0.0;
    for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++) {
      const double tmp1 = 1.0 + Kokkos::sqrt(mus[iSpecies] / mus[jSpecies]) *
                                    Kokkos::pow(g.MolarMass(jSpecies) / g.MolarMass(iSpecies), 0.25);
      const double tmp2 = Kokkos::sqrt(8.0 * (1.0 + g.MolarMass(iSpecies) / g.MolarMass(jSpecies)));
      phis[iSpecies] += X[jSpecies] * tmp1 * tmp1 / tmp2;
    }
  }

  double Mu = 0.0;
  for (int s = 0; s < nSpecies; s++) Mu += X[s] * mus[s] / phis[s];
  return Mu;
}

/*! \brief CSU2TCLib::ThermalConductivitiesWBE (reads the mus / phis / MolarFracWBE of the preceding ViscosityWBE). */
KOKKOS_INLINE_FUNCTION
void ThermalConductivitiesWBE(const NEMOGasTables& g, const GasState& st, const double* X, const double* mus,
                              const double* phis, double* k) {
  const int nSpecies = g.nSpecies;
  double cvves[MAXNS], ks[MAXNS], kves[MAXNS];
  ComputeSpeciesCvVibEle(g, st.Tve, cvves);

  for (int s = 0; s < nSpecies; s++) {
    ks[s] = mus[s] * (15.0 / 4.0 + g.RotationModes(s) / 2.0) * g.Ru / g.MolarMass(s);
    kves[s] = mus[s] * cvves[s];
  }
  double ktr = 0.0, kve = 0.0;
  for (int s = 0; s < nSpecies; s++) {
    ktr += X[s] * ks[s] / phis[s];
    kve += X[s] * kves[s] / phis[s];
  }
  k[0] = ktr;
  k[1] = kve;
}

/*! \brief CSU2TCLib::DiffusionCoeffGY. */
KOKKOS_INLINE_FUNCTION
void DiffusionCoeffGY(const NEMOGasTables& g, const GasState& st, double* Ds) {
  const int nSpecies = g.nSpecies;
  double gam_t = 0.0;
  for (int s = 0; s < nSpecies; s++) gam_t += st.rhos[s] / (st.Density * g.MolarMass(s));

  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    Ds[iSpecies] = 0.0;
    const double Mi = (g.MolarMass(iSpecies) + g.eps);
    const double gam_i = st.rhos[iSpecies] / (st.Density * Mi);
    double denom = 0.0;

    for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++) {
      if (jSpecies != iSpecies) {
        const double Mj = (g.MolarMass(jSpecies) + g.eps);
        const double gam_j = st.rhos[jSpecies] / (st.Density * Mj);
        const double kb = g.kB;
        const double T_col = (iSpecies == 0 && g.ionization) ? st.Tve : st.T;
        const double d1_ij = ComputeCollisionDelta(g, st, iSpecies, jSpecies, Mi, Mj, T_col, true);
        const double D_ij = kb * T_col / (st.Pressure * d1_ij);
        denom += gam_j / D_ij;
      }
    }
    Ds[iSpecies] = (gam_t * gam_t * Mi * (1 - Mi * gam_i) / denom);
  }
}

/*! \brief CSU2TCLib::ViscosityGY. */
KOKKOS_INLINE_FUNCTION
double ViscosityGY(const NEMOGasTables& g, const GasState& st) {
  const int nSpecies = g.nSpecies;
  const double Na = g.Avogadro;
  double Mu = 0.0;
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    double denom = 0.0;
    const double Mi = (g.MolarMass(iSpecies) + g.eps);
    const double gam_i = st.rhos[iSpecies] / (st.Density * Mi);
    for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++) {
      const double Mj = (g.MolarMass(jSpecies) + g.eps);
      const double gam_j = st.rhos[jSpecies] / (st.Density * Mj);
      const double T_col = (iSpecies == 0 && g.ionization) ? st.Tve : st.T;
      const double d2_ij = ComputeCollisionDelta(g, st, iSpecies, jSpecies, Mi, Mj, T_col, false);
      denom += gam_j * d2_ij;
    }
    Mu += (Mi / Na * gam_i) / denom;
  }
  return Mu;
}

/*! \brief CSU2TCLib::ThermalConductivitiesGY (gam_j uses rhos[iSpecies], as on the CPU). */
KOKKOS_INLINE_FUNCTION
void ThermalConductivitiesGY(const NEMOGasTables& g, const GasState& st, double* k) {
  const int nSpecies = g.nSpecies;
  const double Na = g.Avogadro;
  const double kb = g.kB;
  const bool ionization = g.ionization;

  double Cvves[MAXNS];
  ComputeSpeciesCvVibEle(g, st.Tve, Cvves);
  double rhoCvve = 0.0;
  for (int s = 0; s < nSpecies; s++) rhoCvve += st.rhos[s] * Cvves[s];
  const double Cvve = rhoCvve / st.Density;

  double R = 0.0;
  for (int s = 0; s < nSpecies; s++) R += g.Ru / g.MolarMass(s) * st.rhos[s] / st.Density;

  double ThermalCond_tr = 0.0;
  double ThermalCond_ve = 0.0;
  for (int iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    const double Mi = (g.MolarMass(iSpecies) + g.eps);
    const double mi = Mi / Na;
    const double gam_i = st.rhos[iSpecies] / (st.Density * Mi);
    double denom_t = 0.0;
    double denom_r = 0.0;
    double denom_re = 0.0;

    for (int jSpecies = 0; jSpecies < nSpecies; jSpecies++) {
      const double Mj = (g.MolarMass(jSpecies) + g.eps);
      const double mj = Mj / Na;
      const double gam_j = st.rhos[iSpecies] / (st.Density * Mj);
      const double a_ij = 1.0 + (1.0 - mi / mj) * (0.45 - 2.54 * mi / mj) / ((1.0 + mi / mj) * (1.0 + mi / mj));

      const double T_col = ((iSpecies == 0 && ionization) || (jSpecies == 0 && ionization)) ? st.Tve : st.T;

      const double d1_ij = ComputeCollisionDelta(g, st, iSpecies, jSpecies, Mi, Mj, T_col, true);
      const double d2_ij = ComputeCollisionDelta(g, st, iSpecies, jSpecies, Mi, Mj, T_col, false);

      if (jSpecies == 0 && ionization) {
        denom_t += 3.54 * gam_j * d2_ij;
      } else {
        denom_t += a_ij * gam_j * d2_ij;
      }
      denom_r += gam_j * d1_ij;
      denom_re += gam_j * d2_ij;
    }

    if (denom_t <= 0.0) denom_t = g.eps;
    if (denom_r <= 0.0) denom_r = g.eps;
    if (denom_re <= 0.0) denom_re = g.eps;

    if (!ionization || iSpecies != 0) ThermalCond_tr += ((15.0 / 4.0) * kb * gam_i / denom_t);
    if (g.RotationModes(iSpecies) != 0.0) ThermalCond_tr += (kb * gam_i / denom_r);
    if ((!ionization || iSpecies != 0) && g.RotationModes(iSpecies) != 0.0)
      ThermalCond_ve += (kb * Cvve / R * gam_i / denom_r);
    if (ionization && iSpecies == 0) ThermalCond_ve += ((15.0 / 4.0) * kb * gam_i / (1.45 * denom_re));
  }
  k[0] = ThermalCond_tr;
  k[1] = ThermalCond_ve;
}

/*! \brief CSU2TCLib::ViscositySuth. */
KOKKOS_INLINE_FUNCTION
double ViscositySuth(const NEMOGasTables& g, const GasState& st) {
  const double T_nd = st.T / g.TRefSuth;
  return g.muRef * T_nd * Kokkos::sqrt(T_nd) * ((g.TRefSuth + g.SmRef) / (st.T + g.SmRef));
}

/*! \brief CSU2TCLib::ThermalConductivitiesSuth. */
KOKKOS_INLINE_FUNCTION
void ThermalConductivitiesSuth(const NEMOGasTables& g, const GasState& st, double* k) {
  double rho = 0.0;
  for (int ii = 0; ii < g.nSpecies; ii++) rho += st.rhos[ii];
  const double Cvtr = ComputerhoCvtr(g, st.rhos) / rho;
  const double Cvve = ComputerhoCvve(g, st.rhos, st.Tve) / rho;
  const double scl = Cvve / Cvtr;
  const double T_nd = st.T / g.TRefSuth;
  const double kk = g.kRef * T_nd * Kokkos::sqrt(T_nd) * ((g.TRefSuth + g.SkRef) / (st.T + g.SkRef));
  k[0] = kk;
  k[1] = scl * kk;
}

/*!
 * \brief CNEMONSVariable::SetTransportProperties (gas part): GetDiffusionCoeff, GetViscosity,
 *        GetThermalConductivities for the configured model, on the given gas state.
 * \param[out] Ds - species diffusion coefficients; Mu - mixture viscosity; k - {k_tr, k_ve}.
 */
KOKKOS_INLINE_FUNCTION
void ComputeTransport(const NEMOGasTables& g, const GasState& st, double* Ds, double& Mu, double* k) {
  if (g.transModel == NEMOGasTables::TRANSPORT_GUPTAYOS) {
    DiffusionCoeffGY(g, st, Ds);
    Mu = ViscosityGY(g, st);
    ThermalConductivitiesGY(g, st, k);
  } else if (g.transModel == NEMOGasTables::TRANSPORT_SUTHERLAND) {
    DiffusionCoeffWBE(g, st, Ds);
    Mu = ViscositySuth(g, st);
    ThermalConductivitiesSuth(g, st, k);
  } else {
    double X[MAXNS], mus[MAXNS], phis[MAXNS];
    DiffusionCoeffWBE(g, st, Ds);
    Mu = ViscosityWBE(g, st, X, mus, phis);
    ThermalConductivitiesWBE(g, st, X, mus, phis, k);
  }
}

/*--------------------------------------------------------------------------------------*/
/*--- Composite conservative -> primitive conversion.                                ---*/
/*--------------------------------------------------------------------------------------*/

/*!
 * \brief Reproduces CNEMOEulerVariable::Cons2PrimVar (Euler layout) followed by the
 *        ComputeGamma() call of CNEMOEulerVariable::SetPrimVar.
 * \param[in,out] U - Conservative vector; negative rho_s are reset to 1e-20 (as CPU).
 * \param[in,out] V - Primitive vector; V[TVE] on input is the previous Tve (Newton guess).
 * \param[in,out] dPdU, dTdU, dTvedU, eves, Cvves - Secondary arrays (see quirks above);
 *                written only when the state is physical (and the derivatives only if implicit).
 * \param[out] Gamma - Always written (depends only on rho_s).
 * \return Non-physical flag, exactly as CPU.
 */
KOKKOS_INLINE_FUNCTION
bool Cons2Prim(const NEMOGasTables& g, int nDim, bool implicit, bool monoatomic, double Tve_Freestream, double* U,
               double* V, double* val_dPdU, double* val_dTdU, double* val_dTvedU, double* val_eves,
               double* val_Cvves, double& Gamma) {
  const int ns = g.nSpecies;
  const NEMOPrimIndices idx(ns, nDim);
  double rhos[MAXNS];

  bool nonPhys = false;
  const double Tmin = 50.0, Tmax = 8E4;
  const double Tvemin = 50.0, Tvemax = 8E4;

  const double rhoE = U[ns + nDim];
  const double rhoEve = U[ns + nDim + 1];

  V[idx.RHO()] = 0.0;
  for (int s = 0; s < ns; s++) {
    if (U[s] < 0.0) {
      U[s] = 1E-20;
      V[idx.RHOS() + s] = 1E-20;
      rhos[s] = 1E-20;
    } else {
      V[idx.RHOS() + s] = U[s];
      rhos[s] = U[s];
    }
    V[idx.RHO()] += U[s];
  }

  const double rho = V[idx.RHO()];

  double sqvel = 0.0;
  for (int iDim = 0; iDim < nDim; iDim++) {
    V[idx.VEL() + iDim] = U[ns + iDim] / V[idx.RHO()];
    sqvel += V[idx.VEL() + iDim] * V[idx.VEL() + iDim];
  }

  /*--- Gamma only depends on rho_s (CPU computes it after Cons2PrimVar from the gas state). ---*/
  Gamma = ComputeGamma(g, rhos);

  const double Tve_old = V[idx.TVE()];
  double T, Tve;
  ComputeTemperatures(g, rhos, rhoE, rhoEve, 0.5 * rho * sqvel, Tve_old, T, Tve);

  V[idx.T()] = T;
  V[idx.TVE()] = Tve;

  if (V[idx.T()] <= Tmin) { return true; }
  if (V[idx.T()] >= Tmax) { return true; }
  else if (V[idx.T()] != V[idx.T()]) { return true; }

  if (!monoatomic) {
    if (V[idx.TVE()] <= Tvemin) { return true; }
    if (V[idx.TVE()] >= Tvemax) { return true; }
    else if (V[idx.TVE()] != V[idx.TVE()]) { return true; }
  } else {
    V[idx.TVE()] = Tve_Freestream;
  }

  const double Tp = V[idx.T()];
  const double Tvep = V[idx.TVE()];

  double eves[MAXNS];
  ComputeSpeciesCvVibEle(g, Tvep, val_Cvves);
  ComputeSpeciesEve(g, Tvep, eves);
  for (int s = 0; s < ns; s++) val_eves[s] = eves[s];

  V[idx.RHOCVTR()] = ComputerhoCvtr(g, rhos);
  V[idx.RHOCVVE()] = ComputerhoCvve(g, rhos, Tvep);

  const double Pressure = ComputePressure(g, rhos, Tp, Tvep);
  V[idx.P()] = Pressure;
  if (V[idx.P()] < 0.0) {
    V[idx.P()] = 1E-20;
    nonPhys = true;
  }

  if (implicit) {
    ComputedPdU(g, nDim, V, eves, val_dPdU);
    ComputedTdU(g, nDim, V, val_dTdU);
    ComputedTvedU(g, nDim, V, eves, val_dTvedU);
  }

  V[idx.A()] = ComputeSoundSpeed(g, rhos, Pressure);
  V[idx.H()] = (U[ns + nDim] + V[idx.P()]) / V[idx.RHO()];

  return nonPhys;
}

}  // namespace nemo
}  // namespace su2kokkos

#endif  // HAVE_KOKKOS
