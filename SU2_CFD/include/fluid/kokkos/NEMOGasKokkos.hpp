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
  int nReactions = 0; /*!< Number of reactions (informational for now). */
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
