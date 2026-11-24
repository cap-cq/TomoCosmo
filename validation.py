#!/usr/bin/env python3
"""
TomoCosmo -- checks of the standard radiation-dominated (RD) results
====================================================================

Every result of TomoCosmo is quoted RELATIVE to standard RD (a GW spectrum relative to the RD
plateau, a cross section or a decay width relative to its RD value). Before forming those ratios we
check that the RD numbers themselves agree with independent analytic results:

  1. entropy conservation  s(T) a^3 = s(T_0) along the RD background;
  2. the inflationary tensor plateau, numerical background vs the analytic RD formula;
  3. the tensor tilt in a stiff era, numerical vs 2(3w-1)/(1+3w);
  4. the freeze-out cross section for m = 10 GeV vs the standard value 2.2e-26 cm^3/s
     (precise numerical treatments with the SM g_*(T)) and vs the semi-analytic freeze-out estimate;
  5. the freeze-in decay width vs the analytic decay freeze-in yield for a constant g_*.

Run on its own:   python3 validation.py
compute_snr_vs_w.py runs the same checks first and stops if one of them fails.
"""
import numpy as np

import tomocosmo_proto as tc
import gwsources as gs

M_PL = np.sqrt(8 * np.pi) * tc.MP          # (non-reduced) Planck mass [GeV]
S0_OVER_RHOC = tc.S0 / tc.RHOC_H2          # s_0 / (rho_c/h^2) [GeV^-1]


# --------------------------------------------------------------------------- analytic estimates
def sigv_semianalytic(m, g=2.0, target=tc.OMEGA_DM_H2):
    """<sigma v> [cm^3/s] from the standard semi-analytic freeze-out estimate (s-wave):
       x_f = ln[0.038 g M_Pl m <sigma v> / sqrt(g_* x_f)],   Omega h^2 = 1.07e9 x_f sqrt(g_*) / (g_*s M_Pl <sigma v>)"""
    sig = 2e-26 / tc.CM3S_PER_GEVM2
    for _ in range(50):                                          # fixed-point iteration
        xf = 20.0
        for _ in range(30):
            gs_, g_ = tc.gstar_s(m / xf), tc.gstar(m / xf)
            xf = np.log(0.038 * g * M_PL * m * sig / np.sqrt(g_ * xf))
        oh2 = 1.07e9 * xf * np.sqrt(g_) / (gs_ * M_PL * sig)
        sig *= oh2 / target
    return sig * tc.CM3S_PER_GEVM2


def gamma_freeze_in_analytic(m_dm, m_B, g_B=1.0, target=tc.OMEGA_DM_H2):
    """decay width from the analytic freeze-in yield for constant g_* (evaluated at T = m_B/3):
       Y_inf = n/s = 135 g_B M_Pl Gamma / (8 pi^3 1.66 g_*s sqrt(g_*) m_B^2)"""
    T = m_B / 3.0
    Y_per_gamma = 135 * g_B * M_PL / (8 * np.pi**3 * 1.66 * tc.gstar_s(T) * np.sqrt(tc.gstar(T)) * m_B**2)
    return target / (m_dm * S0_OVER_RHOC * Y_per_gamma)


def tilt(bg, f1, f2, r=0.035):
    """logarithmic slope of the spectrum between f1 and f2"""
    o1, o2 = gs.omega_inflation(bg, np.array([f1, f2]), r=r)
    return np.log(o2 / o1) / np.log(f2 / f1)


# --------------------------------------------------------------------------- the checks
def run_rd_checks(verbose=True, m_fo=10.0, m_fi=0.1, m_parent=10.0):
    """returns a list of (name, numerical, reference, relative difference, tolerance, passed)"""
    rows = []

    def add(name, num, ref, tol):
        d = abs(num / ref - 1.0)
        rows.append((name, num, ref, d, tol, d < tol))

    bg = tc.Background("std", T_ini=1e10, npts=12000)
    sa3 = tc.s_ent(bg.T) * bg.a**3 / tc.S0
    add("entropy s a^3 / s_0 (worst point)", sa3[np.argmax(abs(sa3 - 1))], 1.0, 1e-3)

    f = np.logspace(-2, 0, 5)                                    # modes that entered at T >> 1 TeV
    add("tensor RD plateau, r = 0.035", gs.omega_inflation(bg, f).mean(), gs.rd_plateau(0.035), 0.01)

    bg_s = tc.Background("stiff", T_ini=1e13, npts=16000, T_tr=0.01, T_K=1e3, w=0.7)
    add("tensor tilt in a w = 0.7 era", tilt(bg_s, 1e-6, 1e-4), 2 * (3 * 0.7 - 1) / (1 + 3 * 0.7), 0.02)

    bg_dm = tc.Background("std", T_ini=1e6, npts=12000)
    sv = tc.sigv_required(bg_dm, m_fo)
    add(f"freeze-out <sigma v>, m = {m_fo:g} GeV, vs 2.2e-26", sv, 2.2e-26, 0.05)
    add(f"freeze-out <sigma v>, m = {m_fo:g} GeV, vs semi-analytic", sv, sigv_semianalytic(m_fo), 0.15)

    gam = tc.freeze_in_gamma(bg_dm, m_fi, m_parent)
    add(f"freeze-in Gamma, {m_parent:g} GeV -> {m_fi:g} GeV DM, vs analytic", gam,
        gamma_freeze_in_analytic(m_fi, m_parent), 0.15)

    if verbose:
        print("Checks of the RD results")
        print(f"  {'quantity':52s} {'TomoCosmo':>11s} {'reference':>11s} {'diff':>7s}  {'tol':>6s}")
        for name, num, ref, d, tol, ok in rows:
            print(f"  {name:52s} {num:11.4g} {ref:11.4g} {100*d:6.2f}%  {100*tol:5.1f}%  {'ok' if ok else 'FAILED'}")
    return rows


if __name__ == "__main__":
    run_rd_checks()
