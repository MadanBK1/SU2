/*!
 * \file NEMOGasKokkos_validation.cpp
 * \brief Standalone validation of su2kokkos::nemo (stateless device thermo) against
 *        CSU2TCLib / CNEMOGas / CNEMOEulerVariable::Cons2PrimVar on random states.
 *
 * Build (Kokkos build dir):  ninja -C build-kokkos-serial UnitTests/nemo_gas_kokkos_validation
 * Run:                       build-kokkos-serial/UnitTests/nemo_gas_kokkos_validation [nStates] [tol]
 * Exit status 0 iff every compared quantity has max relative difference <= tol (default 1e-12)
 * and all non-physical flags agree.
 *
 * \version 8.5.0 "Harrier"
 *
 * SU2 is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../../../Common/include/CConfig.hpp"
#include "../../../Common/include/parallelization/mpi_structure.hpp"
#include "../../../SU2_CFD/include/fluid/CSU2TCLib.hpp"
#include "../../../SU2_CFD/include/fluid/kokkos/NEMOGasKokkos.hpp"
#include "../../../SU2_CFD/include/variables/CNEMOEulerVariable.hpp"

namespace nk = su2kokkos::nemo;

using DView2D = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>;
using IView1D = Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space>;

/*--- Max relative difference bookkeeping, per named quantity. ---*/
struct DiffStats {
  double maxrel = 0.0;
  double maxabs = 0.0;
  long n = 0, nbitwise = 0, nNaNmismatch = 0;
  double worst_cpu = 0.0, worst_dev = 0.0;
  int worst_k = -1, worst_i = -1;
};

static double RelDiff(double a, double b) {
  if (a == b) return 0.0;
  if (std::isnan(a) && std::isnan(b)) return 0.0;
  if (std::isnan(a) || std::isnan(b)) return INFINITY;
  const double den = std::max(std::fabs(a), std::fabs(b));
  return std::fabs(a - b) / (den > 0 ? den : 1.0);
}

struct Report {
  std::map<std::string, DiffStats> stats;
  std::vector<std::string> order;
  void add(const std::string& name, double cpu, double dev, int k = -1, int i = -1) {
    if (!stats.count(name)) order.push_back(name);
    auto& s = stats[name];
    s.n++;
    const bool bw = (cpu == dev) || (std::isnan(cpu) && std::isnan(dev));
    s.nbitwise += bw;
    if (std::isnan(cpu) != std::isnan(dev)) s.nNaNmismatch++;
    const double r = RelDiff(cpu, dev);
    if (r > s.maxrel) { s.maxrel = r; s.worst_cpu = cpu; s.worst_dev = dev; s.worst_k = k; s.worst_i = i; }
    if (!std::isnan(cpu) && !std::isnan(dev)) s.maxabs = std::max(s.maxabs, std::fabs(cpu - dev));
  }
  bool print(const std::string& title, double tol) const {
    bool ok = true;
    std::printf("\n== %s ==\n%-14s %10s %10s %12s %12s  %s\n", title.c_str(), "quantity", "n", "bitwise%", "max_rel",
                "max_abs", "status");
    for (const auto& k : order) {
      const auto& s = stats.at(k);
      const bool pass = s.maxrel <= tol && s.nNaNmismatch == 0;
      ok = ok && pass;
      std::printf("%-14s %10ld %9.3f%% %12.3e %12.3e  %s", k.c_str(), s.n, 100.0 * s.nbitwise / std::max(1L, s.n),
                  s.maxrel, s.maxabs, pass ? "ok" : "FAIL");
      if (!pass || s.maxrel > 0) std::printf("  [k=%d i=%d cpu=%.17g dev=%.17g]", s.worst_k, s.worst_i, s.worst_cpu, s.worst_dev);
      std::printf("\n");
    }
    return ok;
  }
};

static std::unique_ptr<CConfig> MakeConfig(const std::string& gas_model, const std::string& composition) {
  std::stringstream cfg;
  cfg << "SOLVER= NEMO_EULER\n"
      << "FLUID_MODEL= SU2_NONEQ\n"
      << "GAS_MODEL= " << gas_model << "\n"
      << "GAS_COMPOSITION= " << composition << "\n"
      << "TIME_DISCRE_FLOW= EULER_IMPLICIT\n";
  return std::make_unique<CConfig>(cfg, SU2_COMPONENT::SU2_CFD, false);
}

struct RandomState {
  std::vector<double> rhos, vel;
  double T, Tve, Tve_guess;
};

/*--- Random states: rho 1e-6..1 (log), Y with trace species, T/Tve 200..30000 K (log). ---*/
static RandomState MakeState(std::mt19937_64& rng, int ns, int nDim, bool ionization, const std::vector<double>& M) {
  std::uniform_real_distribution<double> U01(0.0, 1.0);
  auto loguni = [&](double a, double b) { return a * std::pow(b / a, U01(rng)); };

  RandomState st;
  const double rho = loguni(1e-6, 1.0);
  std::vector<double> Y(ns, 0.0);
  double sum = 0.0;
  for (int s = 0; s < ns; s++) {
    const double r = U01(rng);
    if (r < 0.25) Y[s] = 1e-12 * loguni(0.1, 10.0);  // trace species
    else if (r < 0.30) Y[s] = 0.0;                   // absent species
    else Y[s] = -std::log(std::max(U01(rng), 1e-300));
    sum += Y[s];
  }
  if (sum <= 0.0) { Y[ns - 1] = 1.0; sum = 1.0; }
  for (auto& y : Y) y /= sum;
  if (ionization) {
    /*--- Electron (species 0) roughly charge-balanced with NO+ (last species), or trace. ---*/
    Y[0] = (U01(rng) < 0.5) ? Y[ns - 1] * M[0] / M[ns - 1] : 1e-12 * Y[ns - 1];
  }
  st.rhos.resize(ns);
  for (int s = 0; s < ns; s++) st.rhos[s] = rho * Y[s];

  st.vel.resize(nDim);
  for (int d = 0; d < nDim; d++) st.vel[d] = -5000.0 + 10000.0 * U01(rng);
  st.T = loguni(200.0, 30000.0);
  st.Tve = loguni(200.0, 30000.0);

  /*--- Newton initial guess: previous Tve perturbed, exact, far off, or "first iteration" (0). ---*/
  const double r = U01(rng);
  if (r < 0.1) st.Tve_guess = 0.0;
  else if (r < 0.2) st.Tve_guess = st.Tve;
  else if (r < 0.3) st.Tve_guess = loguni(10.0, 1e5);
  else st.Tve_guess = st.Tve * (0.8 + 0.4 * U01(rng));
  return st;
}

/*--- Unit tests of the individual functions. Packed output layout per state. ---*/
struct IndivLayout {
  int ns, nVar;
  int cvtr() const { return 0; }
  int cvve() const { return ns; }
  int eve() const { return 2 * ns; }
  int ev() const { return 3 * ns; }
  int hs() const { return 4 * ns; }
  int ener() const { return 5 * ns; }                // 2
  int scal() const { return 5 * ns + 2; }            // P, a, gamma, rhoCvtr, rhoCvve, T, Tve, rhoCvve_cvves...
  static constexpr int nScal = 7;
  int dPdU() const { return scal() + nScal; }
  int dTdU() const { return dPdU() + nVar; }
  int dTvedU() const { return dTdU() + nVar; }
  int size() const { return dTvedU() + nVar; }
};
static const char* kScalNames[IndivLayout::nScal] = {"P", "a", "Gamma", "rhoCvtr", "rhoCvve", "T(CompT)", "Tve(CompT)"};

static bool RunGas(const std::string& gas_model, const std::string& composition, int nDim, int nStates, double tol,
                   unsigned seed) {
  auto config = MakeConfig(gas_model, composition);
  CSU2TCLib gas(config.get(), nDim, false);
  const auto tables = nk::BuildNEMOGasTables(gas);

  const int ns = tables.nSpecies;
  const int nVar = ns + nDim + 2;
  const int nPrim = ns + nDim + 8;
  const nk::NEMOPrimIndices idx(ns, nDim);
  const bool ionization = tables.ionization;
  const bool monoatomic = config->GetMonoatomic();
  const double Tve_fs = config->GetTemperature_ve_FreeStream();

  std::printf("\n######## %s  nSpecies=%d nHeavy=%d nEl=%d nDim=%d maxEl=%d nReactions=%d states=%d\n",
              gas_model.c_str(), ns, tables.nHeavy, tables.nEl, nDim, tables.maxEl, tables.nReactions, nStates);

  /*--- As in the solver (CNEMOEulerVariable ctor -> ComputeSoundSpeed), Cvtrs gets filled once. ---*/
  const std::vector<double> cvtrs_cpu = gas.GetSpeciesCvTraRot();
  const std::vector<double> M = gas.GetSpeciesMolarMass();

  /*--- CPU variable object (1 point) so the real Cons2PrimVar is the reference. ---*/
  std::vector<double> mf0(ns, 0.0);
  {
    std::stringstream ss(composition);
    std::string tok;
    int s = 0;
    while (std::getline(ss, tok, ',') && s < ns) mf0[s++] = std::stod(tok);
  }
  std::vector<double> mach0(nDim, 0.1);
  CNEMOEulerVariable var(101325.0, mf0.data(), mach0.data(), 300.0, 300.0, 1, nDim, nVar, nPrim, nPrim, config.get(),
                         &gas);
  var.SetPrimVar(0, &gas);  // sets the variable's fluidmodel pointer

  std::mt19937_64 rng(seed);
  std::vector<RandomState> states(nStates);
  for (auto& st : states) st = MakeState(rng, ns, nDim, ionization, M);

  bool all_ok = true;

  /*=================== 1) Individual functions ===================*/
  const IndivLayout L{ns, nVar};
  {
    const int nIn = ns + 3 + nPrim;  // rhos, T, Tve, Tve_guess, V (for derivative tests)
    DView2D din("in", nStates, nIn), dout("out", nStates, L.size());
    auto hin = Kokkos::create_mirror_view(din);
    std::vector<std::vector<double>> cpu(nStates, std::vector<double>(L.size(), 0.0));

    for (int i = 0; i < nStates; i++) {
      auto& st = states[i];
      auto& o = cpu[i];
      std::vector<double> rhos = st.rhos;

      const auto& cvtr = gas.GetSpeciesCvTraRot();
      for (int s = 0; s < ns; s++) o[L.cvtr() + s] = cvtr[s];
      const auto& cvve = gas.ComputeSpeciesCvVibEle(st.T);
      for (int s = 0; s < ns; s++) o[L.cvve() + s] = cvve[s];
      const auto evo = gas.ComputeSpeciesEve(st.T, true);
      for (int s = 0; s < ns; s++) o[L.ev() + s] = evo[s];
      std::vector<double> eve = gas.ComputeSpeciesEve(st.T);
      for (int s = 0; s < ns; s++) o[L.eve() + s] = eve[s];
      const auto& hs = gas.ComputeSpeciesEnthalpy(st.T, st.Tve, eve.data());
      for (int s = 0; s < ns; s++) o[L.hs() + s] = hs[s];

      gas.SetTDStateRhosTTv(rhos, st.T, st.Tve);
      const auto& en = gas.ComputeMixtureEnergies();
      o[L.ener() + 0] = en[0];
      o[L.ener() + 1] = en[1];
      const double e_mix = en[0], e_ve = en[1];
      o[L.scal() + 0] = gas.ComputePressure();
      o[L.scal() + 1] = gas.ComputeSoundSpeed();
      o[L.scal() + 2] = gas.ComputeGamma();
      o[L.scal() + 3] = gas.ComputerhoCvtr();
      o[L.scal() + 4] = gas.ComputerhoCvve();

      /*--- ComputeTemperatures on the exact conservative energies of this state. ---*/
      double rho = 0.0, v2 = 0.0;
      for (int s = 0; s < ns; s++) rho += rhos[s];
      for (int d = 0; d < nDim; d++) v2 += st.vel[d] * st.vel[d];
      const double rhoE = rho * (e_mix + 0.5 * v2), rhoEve = rho * e_ve, rhoEvel = 0.5 * rho * v2;
      const auto& TT = gas.ComputeTemperatures(rhos, rhoE, rhoEve, rhoEvel, st.Tve_guess);
      o[L.scal() + 5] = TT[0];
      o[L.scal() + 6] = TT[1];

      /*--- Derivatives at a synthesized primitive state (gas state set consistently). ---*/
      std::vector<double> V(nPrim, 0.0);
      for (int s = 0; s < ns; s++) V[s] = rhos[s];
      V[idx.T()] = st.T;
      V[idx.TVE()] = st.Tve;
      for (int d = 0; d < nDim; d++) V[idx.VEL() + d] = st.vel[d];
      gas.SetTDStateRhosTTv(rhos, st.T, st.Tve);
      V[idx.P()] = gas.ComputePressure();
      V[idx.RHO()] = rho;
      V[idx.RHOCVTR()] = gas.ComputerhoCvtr();
      V[idx.RHOCVVE()] = gas.ComputerhoCvve();
      std::vector<double> eveTve = gas.ComputeSpeciesEve(st.Tve);
      gas.ComputedPdU(V.data(), eveTve, &o[L.dPdU()]);
      gas.ComputedTdU(V.data(), &o[L.dTdU()]);
      gas.ComputedTvedU(V.data(), eveTve, &o[L.dTvedU()]);

      for (int s = 0; s < ns; s++) hin(i, s) = rhos[s];
      hin(i, ns) = st.T;
      hin(i, ns + 1) = st.Tve;
      hin(i, ns + 2) = st.Tve_guess;
      for (int k = 0; k < nPrim; k++) hin(i, ns + 3 + k) = V[k];
      /*--- extra inputs needed: velocity^2 is in V; energies recomputed on device ---*/
    }
    Kokkos::deep_copy(din, hin);

    const auto g = tables;
    Kokkos::parallel_for(
        "NEMO::indiv", Kokkos::RangePolicy<>(0, nStates), KOKKOS_LAMBDA(const int i) {
          const int ns = g.nSpecies;
          double rhos[nk::MAXNS], tmp[nk::MAXNS], eve[nk::MAXNS], V[nk::MAXNPRIMVAR], out[nk::MAXNVAR];
          for (int s = 0; s < ns; s++) rhos[s] = din(i, s);
          const double T = din(i, ns), Tve = din(i, ns + 1), Tg = din(i, ns + 2);
          for (int k = 0; k < nPrim; k++) V[k] = din(i, ns + 3 + k);

          nk::GetSpeciesCvTraRot(g, tmp);
          for (int s = 0; s < ns; s++) dout(i, L.cvtr() + s) = tmp[s];
          nk::ComputeSpeciesCvVibEle(g, T, tmp);
          for (int s = 0; s < ns; s++) dout(i, L.cvve() + s) = tmp[s];
          nk::ComputeSpeciesEve(g, T, tmp, true);
          for (int s = 0; s < ns; s++) dout(i, L.ev() + s) = tmp[s];
          nk::ComputeSpeciesEve(g, T, eve);
          for (int s = 0; s < ns; s++) dout(i, L.eve() + s) = eve[s];
          nk::ComputeSpeciesEnthalpy(g, T, Tve, eve, tmp);
          for (int s = 0; s < ns; s++) dout(i, L.hs() + s) = tmp[s];

          double en[2];
          nk::ComputeMixtureEnergies(g, rhos, T, Tve, en);
          dout(i, L.ener() + 0) = en[0];
          dout(i, L.ener() + 1) = en[1];
          const double P = nk::ComputePressure(g, rhos, T, Tve);
          dout(i, L.scal() + 0) = P;
          dout(i, L.scal() + 1) = nk::ComputeSoundSpeed(g, rhos, P);
          dout(i, L.scal() + 2) = nk::ComputeGamma(g, rhos);
          dout(i, L.scal() + 3) = nk::ComputerhoCvtr(g, rhos);
          dout(i, L.scal() + 4) = nk::ComputerhoCvve(g, rhos, Tve);

          double rho = 0.0, v2 = 0.0;
          for (int s = 0; s < ns; s++) rho += rhos[s];
          for (int d = 0; d < nDim; d++) v2 += V[ns + 2 + d] * V[ns + 2 + d];
          double Tc, Tvec;
          nk::ComputeTemperatures(g, rhos, rho * (en[0] + 0.5 * v2), rho * en[1], 0.5 * rho * v2, Tg, Tc, Tvec);
          dout(i, L.scal() + 5) = Tc;
          dout(i, L.scal() + 6) = Tvec;

          nk::ComputeSpeciesEve(g, Tve, eve);
          for (int k = 0; k < nVar; k++) out[k] = 0.0;
          nk::ComputedPdU(g, nDim, V, eve, out);
          for (int k = 0; k < nVar; k++) dout(i, L.dPdU() + k) = out[k];
          for (int k = 0; k < nVar; k++) out[k] = 0.0;
          nk::ComputedTdU(g, nDim, V, out);
          for (int k = 0; k < nVar; k++) dout(i, L.dTdU() + k) = out[k];
          nk::ComputedTvedU(g, nDim, V, eve, out);
          for (int k = 0; k < nVar; k++) dout(i, L.dTvedU() + k) = out[k];
        });
    Kokkos::fence();
    auto hout = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dout);

    Report rep;
    for (int i = 0; i < nStates; i++) {
      const auto& o = cpu[i];
      for (int s = 0; s < ns; s++) {
        rep.add("Cvtr_s", o[L.cvtr() + s], hout(i, L.cvtr() + s));
        rep.add("Cvve_s", o[L.cvve() + s], hout(i, L.cvve() + s));
        rep.add("Eve_s", o[L.eve() + s], hout(i, L.eve() + s));
        rep.add("Ev_s(vibonly)", o[L.ev() + s], hout(i, L.ev() + s));
        rep.add("h_s", o[L.hs() + s], hout(i, L.hs() + s));
      }
      rep.add("e_mix", o[L.ener()], hout(i, L.ener()));
      rep.add("e_ve", o[L.ener() + 1], hout(i, L.ener() + 1));
      for (int k = 0; k < IndivLayout::nScal; k++) rep.add(kScalNames[k], o[L.scal() + k], hout(i, L.scal() + k));
      for (int k = 0; k < nVar; k++) {
        rep.add("dPdU", o[L.dPdU() + k], hout(i, L.dPdU() + k), k, i);
        rep.add("dTdU", o[L.dTdU() + k], hout(i, L.dTdU() + k), k, i);
        rep.add("dTvedU", o[L.dTvedU() + k], hout(i, L.dTvedU() + k), k, i);
      }
    }
    all_ok = rep.print(gas_model + ": individual functions", tol) && all_ok;
  }

  /*=================== 2) Composite Cons2Prim ===================*/
  {
    /*--- Columns: U[nVar] | V[nPrim] | dPdU | dTdU | dTvedU | eves | Cvves | Gamma | nonPhys ---*/
    const int oU = 0, oV = nVar, oP = oV + nPrim, oT = oP + nVar, oTv = oT + nVar, oE = oTv + nVar, oC = oE + ns,
              oG = oC + ns, nCol = oG + 1;
    const int nPasses = 2;  // second pass re-uses the arrays (exposes the dPdU "-=" accumulation)

    std::vector<std::vector<double>> cpu(nStates, std::vector<double>(nCol, 0.0));
    std::vector<int> cpu_np(nStates * nPasses, 0);
    DView2D dbuf("c2p", nStates, nCol);
    IView1D dnp("np", nStates * nPasses);
    auto hbuf = Kokkos::create_mirror_view(dbuf);

    long nInjected = 0;
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    for (int i = 0; i < nStates; i++) {
      auto& st = states[i];
      auto& b = cpu[i];
      std::vector<double> rhos = st.rhos;
      gas.SetTDStateRhosTTv(rhos, st.T, st.Tve);
      const auto& en = gas.ComputeMixtureEnergies();
      double rho = 0.0, v2 = 0.0;
      for (int s = 0; s < ns; s++) rho += rhos[s];
      for (int d = 0; d < nDim; d++) v2 += st.vel[d] * st.vel[d];
      for (int s = 0; s < ns; s++) b[oU + s] = rhos[s];
      for (int d = 0; d < nDim; d++) b[oU + ns + d] = rho * st.vel[d];
      b[oU + ns + nDim] = rho * (en[0] + 0.5 * v2);
      b[oU + ns + nDim + 1] = rho * en[1];

      /*--- Inject some non-physical / edge cases. ---*/
      const double r = U01(rng);
      if (r < 0.02) { b[oU + (i % ns)] = -1e-8; nInjected++; }               // negative partial density
      else if (r < 0.04) { b[oU + ns + nDim] *= 1e-3; nInjected++; }         // too little energy -> T<50
      else if (r < 0.05) { b[oU + ns + nDim + 1] *= -1.0; nInjected++; }     // negative Eve
      else if (r < 0.06) { b[oU + ns + nDim] *= 1e3; nInjected++; }          // T > 8e4

      b[oV + idx.TVE()] = st.Tve_guess;
      for (int k = 0; k < nCol; k++) hbuf(i, k) = b[k];
    }
    Kokkos::deep_copy(dbuf, hbuf);

    for (int pass = 0; pass < nPasses; pass++) {
      for (int i = 0; i < nStates; i++) {
        auto& b = cpu[i];
        cpu_np[pass * nStates + i] =
            var.Cons2PrimVar(&b[oU], &b[oV], &b[oP], &b[oT], &b[oTv], &b[oE], &b[oC]) ? 1 : 0;
        b[oG] = gas.ComputeGamma();
      }
    }

    const auto g = tables;
    for (int pass = 0; pass < nPasses; pass++) {
      Kokkos::parallel_for(
          "NEMO::Cons2Prim", Kokkos::RangePolicy<>(0, nStates), KOKKOS_LAMBDA(const int i) {
            double U[nk::MAXNVAR], V[nk::MAXNPRIMVAR], dP[nk::MAXNVAR], dT[nk::MAXNVAR], dTv[nk::MAXNVAR],
                eves[nk::MAXNS], cv[nk::MAXNS], Gamma;
            for (int k = 0; k < nVar; k++) U[k] = dbuf(i, oU + k);
            for (int k = 0; k < nPrim; k++) V[k] = dbuf(i, oV + k);
            for (int k = 0; k < nVar; k++) { dP[k] = dbuf(i, oP + k); dT[k] = dbuf(i, oT + k); dTv[k] = dbuf(i, oTv + k); }
            for (int s = 0; s < g.nSpecies; s++) { eves[s] = dbuf(i, oE + s); cv[s] = dbuf(i, oC + s); }
            const bool np =
                nk::Cons2Prim(g, nDim, true, monoatomic, Tve_fs, U, V, dP, dT, dTv, eves, cv, Gamma);
            for (int k = 0; k < nVar; k++) dbuf(i, oU + k) = U[k];
            for (int k = 0; k < nPrim; k++) dbuf(i, oV + k) = V[k];
            for (int k = 0; k < nVar; k++) { dbuf(i, oP + k) = dP[k]; dbuf(i, oT + k) = dT[k]; dbuf(i, oTv + k) = dTv[k]; }
            for (int s = 0; s < g.nSpecies; s++) { dbuf(i, oE + s) = eves[s]; dbuf(i, oC + s) = cv[s]; }
            dbuf(i, oG) = Gamma;
            dnp(pass * nStates + i) = np ? 1 : 0;
          });
      Kokkos::fence();
    }
    Kokkos::deep_copy(hbuf, dbuf);
    auto hnp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dnp);

    static const char* primNames[] = {"T", "Tve", "u_i", "P", "rho", "h", "a", "rhoCvtr", "rhoCvve"};
    Report rep;
    long nNonPhys = 0, nFlagMismatch = 0;
    for (int i = 0; i < nStates * nPasses; i++) {
      nNonPhys += cpu_np[i];
      nFlagMismatch += (cpu_np[i] != hnp(i));
    }
    for (int i = 0; i < nStates; i++) {
      const auto& b = cpu[i];
      for (int k = 0; k < nVar; k++) rep.add("U(clipped)", b[oU + k], hbuf(i, oU + k));
      for (int s = 0; s < ns; s++) rep.add("V:rho_s", b[oV + s], hbuf(i, oV + s));
      for (int k = ns; k < nPrim; k++) {
        int j = k - ns;
        const char* nm = (j < 2) ? primNames[j] : (j < 2 + nDim ? primNames[2] : primNames[j - nDim + 1]);
        rep.add(std::string("V:") + nm, b[oV + k], hbuf(i, oV + k));
      }
      for (int k = 0; k < nVar; k++) {
        rep.add("dPdU", b[oP + k], hbuf(i, oP + k), k, i);
        rep.add("dTdU", b[oT + k], hbuf(i, oT + k), k, i);
        rep.add("dTvedU", b[oTv + k], hbuf(i, oTv + k));
      }
      for (int s = 0; s < ns; s++) {
        rep.add("eves", b[oE + s], hbuf(i, oE + s));
        rep.add("Cvves", b[oC + s], hbuf(i, oC + s));
      }
      rep.add("Gamma", b[oG], hbuf(i, oG));
    }
    const bool ok = rep.print(gas_model + ": Cons2Prim (2 passes, implicit)", tol);
    std::printf("nonPhys: %ld of %d calls (injected edge cases: %ld), flag mismatches: %ld\n", nNonPhys,
                nStates * nPasses, nInjected, nFlagMismatch);
    all_ok = all_ok && ok && nFlagMismatch == 0;
  }
  return all_ok;
}

int main(int argc, char* argv[]) {
  SU2_MPI::Init(&argc, &argv);
  bool ok = true;
  {
    Kokkos::ScopeGuard guard(argc, argv);
    const int nStates = (argc > 1) ? std::atoi(argv[1]) : 20000;
    const double tol = (argc > 2) ? std::atof(argv[2]) : 1e-12;
    std::printf("Kokkos execution space: %s\n", Kokkos::DefaultExecutionSpace::name());

    ok = RunGas("N2", "0.75, 0.25", 3, nStates, tol, 1234u) && ok;
    ok = RunGas("N2", "0.75, 0.25", 2, nStates / 4, tol, 99u) && ok;
    ok = RunGas("AIR-5", "0.75, 0.25, 0.0, 0.0, 0.0", 3, nStates, tol, 5678u) && ok;
    ok = RunGas("AIR-7", "0.0, 0.75, 0.25, 0.0, 0.0, 0.0, 0.0", 3, nStates, tol, 4321u) && ok;
    ok = RunGas("ARGON", "1.0", 3, nStates / 4, tol, 777u) && ok;
  }
  std::printf("\nRESULT: %s\n", ok ? "PASS" : "FAIL");
  SU2_MPI::Finalize();
  return ok ? 0 : 1;
}
