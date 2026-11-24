#!/usr/bin/env python3
"""
===============================================================================================
 TomoCosmo  --  Gravitational-wave tomography of the pre-BBN Universe          (prototype v0.1)
===============================================================================================
 Author : Satyabrata Mahapatra, School of Physical Sciences, IIT Goa
 Licence: MIT (see LICENSE)

 WHAT THE PROGRAM DOES
 ---------------------
 TomoCosmo computes ONE expansion history of the early Universe and uses that same history for
 two things:

   1. the inflationary gravitational-wave (GW) background: each tensor mode re-enters the horizon
      at a moment fixed by the history, which shapes today's spectrum h^2 Omega_GW(f) and its
      signal-to-noise ratio (SNR) in future detectors;
   2. dark-matter (DM) production: the freeze-out cross section and the freeze-in decay width that
      give the observed relic abundance Omega h^2 = 0.12 depend on the same expansion rate H(T).

 Comparing both with standard radiation domination (RD) relates a GW observable (the SNR) to a
 laboratory observable (the DM coupling). A non-standard era therefore leaves two correlated
 imprints, and the code makes that correlation quantitative.

 THIS FILE (the core module)
 ---------------------------
   * constants and the Standard-Model degrees of freedom g_*(T), g_*s(T);
   * the class Background, which builds a(N), T(N), H(N) and the time t(N) on a grid in N = ln a,
     from a starting temperature T_ini down to today (a = 1 today), for
        kind="std"    standard history: RD -> matter -> Lambda,
        kind="stiff"  a matter-like phase followed by a stiff era of constant w, ending at T_tr,
        kind="w_of_T" a GENERIC equation of state w(T) of an extra component, supplied by the user,
        kind="emd"    an early matter-dominated era from a decaying particle, with entropy injection;
   * DM freeze-out (relic_density, sigv_required) and DM freeze-in from decays (freeze_in_gamma).
     Both work on any Background, so any history above is automatically included.

 OTHER FILES AND FOLDERS
 -----------------------
   gwsources.py                inflationary tensor spectrum on a Background
   detectors.py                detector noise curves, sensitivity curves and SNR
   validation.py               checks of the RD results against analytic formulae
   compute_snr_vs_w.py         scans the equation of state w: SNRs and DM rate ratios -> data/
   make_fig1.py                draws Fig. 1 (spectra and the SNR-coupling relation) -> figures/
   make_sensitivity_curves.py  regenerates data/sensitivity_curves.npz
   tests/test_tomocosmo.py     automatic checks (run: python3 -m pytest tests  or  python3 tests/test_tomocosmo.py)
   data/                       lal_tables.npz (ground-based detector noise), sensitivity_curves.npz,
                               snr_vs_w.json (output of compute_snr_vs_w.py)
   figures/                    output figures

 USING A GENERIC w(T)
 --------------------
   bg = Background("w_of_T", w_of_T=my_w, T_norm=0.01, ratio_norm=1.0)
 my_w(T) returns the equation of state of the extra component at plasma temperature T (array in,
 array out). The component is normalised by rho_X / rho_r = ratio_norm at T = T_norm, and evolves as
        d ln rho_X / dN = -3 (1 + w(T(N))).
 The component must not exchange energy with the plasma (otherwise use kind="emd"-type equations),
 so that T(a) still follows from entropy conservation.

 UNITS AND CONVENTIONS
 ---------------------
 Natural units (GeV) everywhere; a = 1 today; N = ln a. Seconds and Hz appear only in the GW part.

 PROTOTYPE STATUS
 ----------------
 This is a research prototype. Its simplifications are listed in README.md.
===============================================================================================
"""
import warnings

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import brentq
from scipy.special import k1 as bessel_k1

# =============================================================================================
# 1. Constants
# =============================================================================================
MP = 2.435e18                  # reduced Planck mass [GeV]
GEV_INV_S = 6.582119569e-25    # 1 GeV^-1 in seconds
T0 = 2.7255 * 8.617333262e-14  # CMB temperature today [GeV]
H100_S = 3.2407793e-18         # 100 km/s/Mpc in s^-1
H100_GEV = H100_S * GEV_INV_S  # 100 km/s/Mpc in GeV
RHOC_H2 = 3.0 * MP**2 * H100_GEV**2   # critical density / h^2 [GeV^4]
h_HUB = 0.674                  # Hubble parameter h
OMEGA_M = 0.315                # matter density parameter today
GS0, G0 = 3.909, 3.363         # g_*s and g_* today (photons + three neutrinos)
CM3S_PER_GEVM2 = 1.1663e-17    # 1 GeV^-2 = 1.1663e-17 cm^3/s
OMEGA_DM_H2 = 0.120            # observed DM abundance

# =============================================================================================
# 2. Standard-Model degrees of freedom and the radiation bath
# =============================================================================================
# columns: T [GeV], g_*(T) (energy), g_*s(T) (entropy); interpolated in ln T
_TAB = np.array([
    [1e-18, 3.363, 3.909], [1e-5, 3.363, 3.909], [3e-5, 3.40, 3.94], [6e-5, 3.75, 4.25],
    [1e-4, 4.80, 5.20], [2e-4, 7.30, 7.40], [3e-4, 8.90, 8.95], [5e-4, 10.10, 10.13],
    [1e-3, 10.60, 10.62], [2e-3, 10.73, 10.74], [5e-3, 10.76, 10.76], [1e-2, 10.90, 10.90],
    [2e-2, 11.40, 11.40], [3e-2, 12.20, 12.20], [5e-2, 13.90, 13.90], [8e-2, 15.80, 15.80],
    [0.10, 17.50, 17.50], [0.12, 20.0, 20.0], [0.15, 28.0, 28.0], [0.20, 43.0, 43.0],
    [0.25, 52.0, 52.0], [0.30, 57.0, 57.0], [0.50, 63.0, 63.0], [1.0, 70.0, 70.0],
    [2.0, 76.0, 76.0], [5.0, 83.0, 83.0], [10.0, 86.25, 86.25], [20.0, 88.5, 88.5],
    [50.0, 95.0, 95.0], [100.0, 101.5, 101.5], [200.0, 105.3, 105.3], [500.0, 106.6, 106.6],
    [1e3, 106.75, 106.75], [1e25, 106.75, 106.75]])
_LT, _G, _GS = np.log(_TAB[:, 0]), _TAB[:, 1], _TAB[:, 2]


def gstar(T):
    """energy degrees of freedom g_*(T)"""
    return np.interp(np.log(T), _LT, _G)


def gstar_s(T):
    """entropy degrees of freedom g_*s(T)"""
    return np.interp(np.log(T), _LT, _GS)


def rho_rad(T):
    """radiation energy density rho_r = (pi^2/30) g_* T^4"""
    return np.pi**2 / 30.0 * gstar(T) * T**4


def s_ent(T):
    """entropy density s = (2 pi^2/45) g_*s T^3"""
    return 2.0 * np.pi**2 / 45.0 * gstar_s(T) * T**3


_LT_FINE = np.linspace(np.log(1e-17), np.log(1e24), 60000)
_LS_FINE = np.log(s_ent(np.exp(_LT_FINE)))


def T_of_s(s):
    """temperature for a given entropy density (inverse of s_ent)"""
    return np.exp(np.interp(np.log(s), _LS_FINE, _LT_FINE))


def H_rad(T):
    """Hubble rate of pure radiation domination at temperature T"""
    return np.sqrt(rho_rad(T) / 3.0) / MP


def a_adiabatic(T):
    """scale factor at temperature T when entropy is conserved (a = 1 today)"""
    return (GS0 / gstar_s(T)) ** (1.0 / 3.0) * T0 / T


S0 = s_ent(T0)                                  # comoving entropy today (a = 1)
OMEGA_R = rho_rad(T0) / (RHOC_H2 * h_HUB**2)    # radiation density parameter today
OMEGA_L = 1.0 - OMEGA_M - OMEGA_R               # flat Universe
H0_GEV = H100_GEV * h_HUB                       # Hubble rate today [GeV]


# =============================================================================================
# 3. Expansion histories
# =============================================================================================
_ALIASES = {"kin": "stiff"}                     # old name kept for backward compatibility


class Background:
    """
    Expansion history on a grid in N = ln a, from T_ini down to today.

    Background("std")                                   standard RD -> matter -> Lambda
    Background("stiff", T_tr=..., T_K=..., w=...)       matter-like phase above T_K, stiff era of
                                                        constant w between T_K and T_tr, RD below T_tr
                                                        (rho_phi = rho_r at T_tr)
    Background("w_of_T", w_of_T=f, T_norm=..., ratio_norm=1.0)
                                                        extra component with a generic w(T)
    Background("emd", T_dom=..., T_RH=...)              decaying matter X with entropy injection:
                                                        rho_X = rho_r at T_dom without decay,
                                                        width Gamma_X = H_rad(T_RH)

    Common options: T_ini [GeV] (start of the grid), npts (grid points), check_start (warn if the
    extra component already dominates at T_ini; switch off when only the late part is needed).
    Old names "kin" and T_kin are accepted for "stiff" and T_tr.

    Attributes after construction:
      N, a, T, H [GeV], rho_X [GeV^4]  arrays on the grid (T decreases along the grid)
      t [s], Hs [1/s]                  cosmic time and Hubble rate in seconds
      D                                entropy dilution factor (1 unless kind="emd")
    """

    def __init__(self, kind="std", T_ini=1e10, npts=9000, check_start=True, **pars):
        kind = _ALIASES.get(kind, kind)
        if "T_kin" in pars:                                  # old parameter name
            pars["T_tr"] = pars.pop("T_kin")
        if kind not in ("std", "stiff", "w_of_T", "emd"):
            raise ValueError(f"unknown kind '{kind}'")
        self.kind, self.pars, self.T_ini = kind, pars, T_ini
        self.D = 1.0
        if kind == "emd":
            self._solve_emd(npts, D_guess=1.0)
            self._solve_emd(npts, D_guess=self.D)            # second pass: a = 1 today exactly
        else:
            self._solve_adiabatic(npts)
            if check_start:
                self._check_start()
        self._finish()

    # ---------------------------------------------------------------- extra component rho_X(a)
    def _rho_X(self, N, a, T):
        if self.kind == "std":
            return np.zeros_like(a)
        if self.kind == "stiff":
            return rho_stiff(a, self.pars["T_tr"], self.pars["T_K"], self.pars.get("w", 1.0))
        # kind == "w_of_T": d ln rho_X / dN = -3 (1 + w), normalised at T_norm
        w = np.asarray(self.pars["w_of_T"](T), dtype=float) * np.ones_like(T)
        I = cumulative_trapezoid(3.0 * (1.0 + w), N, initial=0.0)
        Tn = self.pars["T_norm"]
        Nn = np.interp(-np.log(Tn), -np.log(T), N)          # N at which T = T_norm
        In = np.interp(Nn, N, I)
        rho_n = self.pars.get("ratio_norm", 1.0) * rho_rad(Tn)
        return rho_n * np.exp(-(I - In))

    def _late(self, a):
        """late-time matter and Lambda"""
        return 3.0 * MP**2 * H0_GEV**2 * (OMEGA_M * a**-3 + OMEGA_L)

    # ---------------------------------------------------------------- histories without entropy injection
    def _solve_adiabatic(self, npts):
        N = np.linspace(np.log(a_adiabatic(self.T_ini)), 0.0, npts)
        a = np.exp(N)
        T = T_of_s(S0 / a**3)                                # entropy conservation
        rho_X = self._rho_X(N, a, T)
        self.N, self.a, self.T, self.rho_X = N, a, T, rho_X
        self.H = np.sqrt((rho_rad(T) + rho_X + self._late(a)) / 3.0) / MP   # Friedmann equation

    def _check_start(self):
        """warn if the extra component already dominates at T_ini (the earliest part is then cut)"""
        if self.kind != "std" and self.rho_X[0] > rho_rad(self.T[0]):
            warnings.warn(f"the extra component dominates already at T_ini = {self.T_ini:.3g} GeV; "
                          "the earliest phase and the highest frequencies are cut. Raise T_ini "
                          "(for kind='stiff' use T_ini_stiff(...)).", stacklevel=3)

    # ---------------------------------------------------------------- early matter domination with entropy injection
    def _solve_emd(self, npts, D_guess):
        Td, Trh, Ti = self.pars["T_dom"], self.pars["T_RH"], self.T_ini
        Gam = H_rad(Trh)                                     # decay width
        a_ini = a_adiabatic(Ti) * D_guess ** (-1.0 / 3.0)
        S_ini = s_ent(Ti) * a_ini**3
        rhoX_ini = rho_rad(Td) * s_ent(Ti) / s_ent(Td)       # rho_X ~ a^-3 ~ s

        def rhs(N, y):                                       # y = (ln rho_X, ln S)
            a = np.exp(N)
            rhoX, S = np.exp(y[0]), np.exp(y[1])
            T = T_of_s(S / a**3)
            H = np.sqrt((rho_rad(T) + rhoX) / 3.0) / MP
            return [-3.0 - Gam / H, Gam * rhoX * a**3 / (H * T * S)]

        def done(N, y):                                      # stop when rho_X / rho_r < 1e-12
            T = T_of_s(np.exp(y[1]) / np.exp(3 * N))
            return y[0] - np.log(rho_rad(T)) + np.log(1e12)
        done.terminal, done.direction = True, -1

        N0 = np.log(a_ini)
        sol = solve_ivp(rhs, [N0, 0.0], [np.log(rhoX_ini), np.log(S_ini)], method="LSODA",
                        events=done, dense_output=True, rtol=1e-9, atol=1e-11, max_step=0.05)
        N_stop, S_fin = sol.t[-1], np.exp(sol.y[1, -1])
        self.D = S_fin / S_ini                               # dilution factor
        N = np.linspace(N0, 0.0, npts)
        a = np.exp(N)
        early = N <= N_stop
        y = sol.sol(N[early])
        S = np.full_like(a, S_fin)
        S[early] = np.exp(y[1])
        rhoX = np.zeros_like(a)
        rhoX[early] = np.exp(y[0])
        T = T_of_s(S / a**3)
        self.N, self.a, self.T, self.rho_X, self.S = N, a, T, rhoX, S
        self.H = np.sqrt((rho_rad(T) + rhoX + self._late(a)) / 3.0) / MP

    # ---------------------------------------------------------------- cosmic time and look-ups
    def _finish(self):
        Hs = self.H / GEV_INV_S                              # Hubble rate in s^-1
        self.t = 0.5 / Hs[0] + cumulative_trapezoid(1.0 / Hs, self.N, initial=0.0)   # t = int dN/H
        self.lnt, self.lna, self.Hs, self.t0 = np.log(self.t), np.log(self.a), Hs, self.t[-1]

    def a_of_t(self, t):
        return np.exp(np.interp(np.log(t), self.lnt, self.lna))

    def T_of_t(self, t):
        return np.exp(np.interp(np.log(t), self.lnt, np.log(self.T)))


def rho_stiff(a, T_tr, T_K, w=1.0):
    """energy density of the stiff-era fluid: ~a^-3 above T_K, ~a^-3(1+w) below, rho = rho_r at T_tr"""
    a_tr, a_K = a_adiabatic(T_tr), a_adiabatic(T_K)
    n = 3.0 * (1.0 + w)
    rho_tr = rho_rad(T_tr)
    rho_K = rho_tr * (a_tr / a_K) ** n
    return np.where(a >= a_K, rho_tr * (a_tr / a) ** n, rho_K * (a_K / a) ** 3)


def T_ini_stiff(T_tr, T_K, w, margin=10.0, T_min=1e13):
    """starting temperature for kind='stiff' such that the matter-like phase begins inside the grid:
    margin times above the temperature at which the fluid first dominates (and at least T_min)"""
    R_K = float(rho_stiff(a_adiabatic(T_K), T_tr, T_K, w) / rho_rad(T_K))   # largest rho_phi/rho_r
    T_start = T_K * R_K                                      # rho_phi/rho_r ~ 1/T in the matter-like phase
    return max(T_min, margin * T_start) if R_K > 1 else T_min


# =============================================================================================
# 4. Dark matter: thermal freeze-out
# =============================================================================================
def relic_density(bg, m, sigv_cm3s, g=2.0, x_start=8.0):
    """
    Omega h^2 of a thermal relic of mass m [GeV] with a constant s-wave <sigma v> [cm^3/s] on bg.
    Solves dY/dN = -<sigma v>/(H a^3) (Y^2 - Y_eq^2) for the comoving number Y = n a^3.
    """
    sig = sigv_cm3s / CM3S_PER_GEVM2                      # <sigma v> in GeV^-2
    i0 = np.searchsorted(-bg.T, -m / x_start)             # start at x = m/T = x_start (equilibrium)
    N, T, H = bg.N[i0:], bg.T[i0:], bg.H[i0:]
    lnT, lnH = np.log(T), np.log(H)

    def Yeq(Nv):                                          # equilibrium comoving number (Maxwell-Boltzmann)
        Tv = np.exp(np.interp(Nv, N, lnT))
        return g * (m * Tv / (2 * np.pi)) ** 1.5 * np.exp(-m / Tv) * np.exp(3 * Nv)

    def rhs(Nv, y):                                       # evolve ln Y
        Hv = np.exp(np.interp(Nv, N, lnH))
        Y, Ye = np.exp(y[0]), Yeq(Nv)
        return [-sig / (Hv * np.exp(3 * Nv)) * (Y * Y - Ye * Ye) / Y]

    # integrate down to T = m/3000, long after freeze-out; Y is constant afterwards
    i1 = min(len(N) - 1, np.searchsorted(-T, -m / 3000.0))
    sol = solve_ivp(rhs, [N[0], N[i1]], [np.log(Yeq(N[0]))], method="Radau", rtol=1e-7, atol=1e-9)
    n0 = np.exp(sol.y[0, -1])                             # comoving number = number density today
    return m * n0 / RHOC_H2


def sigv_required(bg, m, target=OMEGA_DM_H2, g=2.0):
    """<sigma v> [cm^3/s] that gives Omega h^2 = target by freeze-out on bg"""
    f = lambda ls: np.log(relic_density(bg, m, 10**ls, g) / target)
    return 10 ** brentq(f, -30.5, -23.5, xtol=1e-4)


# =============================================================================================
# 5. Dark matter: freeze-in from decays B -> DM + SM
# =============================================================================================
def freeze_in_gamma(bg, m_dm, m_B, g_B=1.0, target=OMEGA_DM_H2):
    """
    Decay width Gamma [GeV] of a parent B (mass m_B, in equilibrium) into one DM particle (mass m_dm)
    that gives Omega h^2 = target on bg. DM is never in equilibrium (no inverse decays).
      dY/dN = (a^3/H) g_B Gamma m_B^2 T K_1(m_B/T) / (2 pi^2)
    The yield is linear in Gamma, so it is integrated once per unit width.
    """
    T, H, a = bg.T, bg.H, bg.a
    x = m_B / T
    ok = x < 300                                          # K_1 ~ e^-x is negligible beyond x = 300
    integrand = np.zeros_like(T)
    integrand[ok] = a[ok] ** 3 * g_B * m_B**2 * T[ok] * bessel_k1(x[ok]) / (2 * np.pi**2 * H[ok])
    n0_per_gamma = np.trapezoid(integrand, bg.N)          # number density today per unit Gamma
    return target * RHOC_H2 / (m_dm * n0_per_gamma)
