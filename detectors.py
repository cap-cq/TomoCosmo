#!/usr/bin/env python3
"""
TomoCosmo -- detector sensitivities for a stochastic GW background
==================================================================

Every detector is described by an effective noise h^2 Omega_n(f) in the same units as the signal,
defined so that the signal-to-noise ratio of a background h^2 Omega_GW(f) observed for a time T_obs is

        SNR^2 = T_obs * Integral df [ h^2 Omega_GW(f) / h^2 Omega_n(f) ]^2 .

Main functions
--------------
    snr(name, omega_fn)    SNR of a spectrum omega_fn(f) [a function of f in Hz] in detector `name`
    pls(name, snr=10)      power-law-integrated sensitivity curve (the lines drawn in Fig. 1a)
    grid(name)             (f, h^2 Omega_n(f), T_obs) on the detector band
    DETECTORS              registry: name -> (noise function, f_min, f_max, T_obs); add new detectors here

Where the sensitivity projections come from
-------------------------------------------
  LISA    Instrument noise and Galactic confusion noise (4-year mission) from the analytic fit of
          Robson, Cornish and Liu, "The construction and use of LISA sensitivity curves",
          https://arxiv.org/abs/1803.01944
  DECIGO  Analytic noise fit of Yagi and Seto, Phys. Rev. D 83, 044011 (2011),
          https://arxiv.org/abs/1101.3940 ; two units cross-correlated.
  ET      ET-D design noise, as distributed in LALSimulation (SimNoisePSDEinsteinTelescopeP1600143).
          Curve: LIGO-P1600143, https://dcc.ligo.org/LIGO-P1600143/public ; ET-D design:
          https://arxiv.org/abs/1012.0908 . Triangle of three co-located interferometers.
  CE      Cosmic Explorer (40 km) noise, LALSimulation SimNoisePSDCosmicExplorerP1600143.
          Curve: LIGO-P1600143, https://dcc.ligo.org/LIGO-P1600143/public (https://arxiv.org/abs/1607.08697).
          Two detectors are placed at the Hanford and Livingston sites (an assumption of this code).
  A+      LIGO A+ design noise, LALSimulation SimNoisePSDaLIGOAPlusDesignSensitivityT1800042.
          Curve: LIGO-T1800042, https://dcc.ligo.org/LIGO-T1800042/public .
          Network of Hanford, Livingston and LIGO-India (Aundha), all at A+ sensitivity.
  SKA     Simple SKA-like pulsar timing array: 100 pulsars, white timing noise of 100 ns,
          14-day cadence, 15 years. This is an illustrative model, not an official SKA curve.
  Pairs of separated detectors are combined with the overlap reduction function computed from the
  detector geometry (Allen and Romano, https://arxiv.org/abs/gr-qc/9710117). The power-law-integrated
  curves follow Thrane and Romano, https://arxiv.org/abs/1310.5300 .
  LALSuite (LAL and LALSimulation): https://git.ligo.org/lscsoft/lalsuite ,
  documentation https://lscsoft.docs.ligo.org/lalsuite/ .

This file use .npz files
------------------------------
  Yes, one: data/lal_tables.npz. It holds the ET, CE and A+ noise curves and the geometry of the
  Hanford, Livingston and LIGO-India detectors, read once from LALSuite and stored, so that LALSuite
  does not need to be installed. If LALSuite IS installed, this file reads the same quantities
  directly from it and ignores the table. LISA, DECIGO and SKA are analytic and need no file.
  (data/sensitivity_curves.npz is NOT read here: it stores the output of pls(); see
  make_sensitivity_curves.py.)

All curves are projections of design sensitivities and should be taken as indicative. Published
LISA and DECIGO conventions differ by factors of about sqrt(2), which change the SNR by the same factor.
"""
import os

import numpy as np
from scipy.special import spherical_jn

H100_S = 3.2407793e-18                                   # 100 km/s/Mpc in s^-1
YR = 3.15576e7                                           # one year in seconds
C_LIGHT = 299792458.0                                    # speed of light [m/s]
_PREF_STRAIN = 2 * np.pi**2 / (3 * H100_S**2)            # h^2 Omega = _PREF_STRAIN f^3 S_h
_PREF_XC = 10 * np.pi**2 / (3 * H100_S**2)               # cross-correlated pair of interferometers
_TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lal_tables.npz")


# ============================================================================== LISA
def lisa_Sn(f):
    """LISA sky-averaged strain noise S_n(f) [1/Hz] including the Galactic confusion noise (4 yr)"""
    L, fs = 2.5e9, 19.09e-3                              # arm length [m], transfer frequency c/(2 pi L)
    poms = (1.5e-11) ** 2 * (1 + (2e-3 / f) ** 4)        # optical metrology noise
    pacc = (3e-15) ** 2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)   # test-mass acceleration noise
    sn = 10 / (3 * L**2) * (poms + 2 * (1 + np.cos(f / fs) ** 2) * pacc / (2 * np.pi * f) ** 4) \
        * (1 + 0.6 * (f / fs) ** 2)
    A, al, be, ka, ga, fk = 9e-45, 0.138, -221.0, 521.0, 1680.0, 0.00113   # confusion noise, 4 yr
    sc = A * f ** (-7 / 3) * np.exp(-f**al + be * f * np.sin(ka * f)) * (1 + np.tanh(ga * (fk - f)))
    return sn + sc


def lisa(f):
    """effective noise h^2 Omega_n of LISA (factor sqrt(2): two channels folded into S_n)"""
    return np.sqrt(2.0) * _PREF_STRAIN * f**3 * lisa_Sn(f)


# ============================================================================== DECIGO
def decigo(f):
    """effective noise h^2 Omega_n of DECIGO, two units cross-correlated"""
    fp = 7.36
    Sh = 7.05e-48 * (1 + (f / fp) ** 2) + 4.8e-51 * f**-4 / (1 + (f / fp) ** 2) + 5.33e-52 * f**-4
    return _PREF_STRAIN * f**3 * Sh / np.sqrt(2.0)


# ============================================================================== ground-based detectors
_cache = {}


def _have_lal():
    try:
        import lal, lalsimulation  # noqa: F401
        return True
    except ImportError:
        return False


def _psd_table(name):
    """(frequencies, PSD) of a LALSimulation noise curve: from LALSuite if installed, else from the table"""
    if name in _cache:
        return _cache[name]
    if _have_lal():
        import lal
        import lalsimulation as ls
        df, n = 0.125, int(4096 / 0.125) + 1                  # 0 to 4096 Hz in steps of 0.125 Hz
        s = lal.CreateREAL8FrequencySeries("", lal.LIGOTimeGPS(0), 0.0, df, lal.SecondUnit, n)
        getattr(ls, name)(s, 1.0)
        _cache[name] = (np.arange(n) * df, s.data.data.copy())
    else:
        t = np.load(_TABLES)
        _cache[name] = (t["psd_f_" + name], t["psd_p_" + name])
    return _cache[name]


def psd(name, f):
    """noise power spectral density S_n(f) [1/Hz], log-log interpolation of the table"""
    ff, pp = _psd_table(name)
    ok = pp > 0
    return np.exp(np.interp(np.log(f), np.log(ff[ok][1:]), np.log(pp[ok][1:])))


def _det(name):
    """detector response tensor d_ab (3x3) and Earth-centred position x [m]"""
    if _have_lal():
        import lal
        d = lal.CachedDetectors[getattr(lal, name)]
        return np.array(d.response, dtype=float), np.array(d.location, dtype=float)
    t = np.load(_TABLES)
    return t["det_r_" + name], t["det_x_" + name]


def orf(f, det1, det2):
    """normalised overlap reduction function gamma(f); = 1 for coincident, aligned L-shaped detectors"""
    d1, x1 = _det(det1)
    d2, x2 = _det(det2)
    dx = x1 - x2
    dist = np.linalg.norm(dx)
    s = dx / dist
    al = 2 * np.pi * f * dist / C_LIGHT
    j0, j1, j2 = spherical_jn(0, al), spherical_jn(1, al), spherical_jn(2, al)
    r1 = 5 * j0 - 10 * j1 / al + 5 * j2 / al**2
    r2 = -10 * j0 + 40 * j1 / al - 50 * j2 / al**2
    r3 = 2.5 * j0 - 25 * j1 / al + 87.5 * j2 / al**2
    c1 = np.sum(d1 * d2)
    c2 = (d1 @ s) @ (d2 @ s)
    c3 = (s @ d1 @ s) * (s @ d2 @ s)
    return r1 * c1 + r2 * c2 + r3 * c3


def et(f):
    """ET triangle: three co-located pairs with gamma = -3/8"""
    P = psd("SimNoisePSDEinsteinTelescopeP1600143", f)
    return _PREF_XC * f**3 * P / (3.0 / 8.0) / np.sqrt(2.0 * 3.0)


def ce(f):
    """two Cosmic Explorers at the Hanford and Livingston sites"""
    P = psd("SimNoisePSDCosmicExplorerP1600143", f)
    g = np.abs(orf(f, "LHO_4K_DETECTOR", "LLO_4K_DETECTOR"))
    return _PREF_XC * f**3 * P / np.clip(g, 1e-6, None) / np.sqrt(2.0)


def aplus_network(f, sites=("LHO_4K_DETECTOR", "LLO_4K_DETECTOR", "LIO_4K_DETECTOR")):
    """LIGO A+ network; 1/Omega_n^2 = sum over pairs of 2/Omega_eff^2"""
    P = psd("SimNoisePSDaLIGOAPlusDesignSensitivityT1800042", f)
    inv2 = np.zeros_like(f)
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            g = orf(f, sites[i], sites[j])
            inv2 += 2.0 * g**2 / (_PREF_XC * f**3 * P) ** 2
    return 1.0 / np.sqrt(inv2)


# ============================================================================== pulsar timing array
def pta(f, n_psr=100, sigma=100e-9, cadence=14 * 86400.0):
    """SKA-like PTA with white timing noise; all N(N-1)/2 pulsar pairs cross-correlated"""
    Sn = 12 * np.pi**2 * f**2 * 2 * sigma**2 * cadence
    zeta_rms = 0.1443                                     # rms of the Hellings-Downs curve
    Seff = Sn / (zeta_rms * np.sqrt(n_psr * (n_psr - 1) / 2.0))
    return _PREF_STRAIN * f**3 * Seff / np.sqrt(2.0)


# ============================================================================== registry
DETECTORS = {
    #  name                  (noise function, f_min [Hz], f_max [Hz], T_obs [s])
    "SKA":                  (pta, 1.0 / (15 * YR), 1.0 / (2 * 14 * 86400.0), 15 * YR),
    "LISA":                 (lisa, 3e-5, 0.5, 3 * YR),
    "DECIGO":               (decigo, 1e-3, 1e2, 3 * YR),
    "ET":                   (et, 1.5, 2.0e3, 3 * YR),
    "CE":                   (ce, 5.0, 2.0e3, 3 * YR),
    "A+ (HL)":              (lambda f: aplus_network(f, ("LHO_4K_DETECTOR", "LLO_4K_DETECTOR")), 10.0, 1.0e3, 3 * YR),
    "A+ (HL+LIGO-India)":   (aplus_network, 10.0, 1.0e3, 3 * YR),
}


def grid(name, n=3000):
    """log-spaced frequencies across the band, the effective noise there, and T_obs"""
    fn, f1, f2, T = DETECTORS[name]
    f = np.logspace(np.log10(f1), np.log10(f2), n)
    return f, fn(f), T


def pls(name, snr=10.0, betas=np.linspace(-8, 8, 161), n=3000):
    """power-law-integrated sensitivity curve: envelope of all power laws f^beta that reach the SNR"""
    f, On, T = grid(name, n)
    fref = np.sqrt(f[0] * f[-1])
    out = np.zeros_like(f)
    for b in betas:
        amp = snr / np.sqrt(T * np.trapezoid((f / fref) ** (2 * b) / On**2, f))
        out = np.maximum(out, amp * (f / fref) ** b)
    return f, out


def snr(name, omega_fn):
    """SNR of the spectrum omega_fn(f) [h^2 Omega_GW as a function of f in Hz] in detector `name`"""
    f, On, T = grid(name)
    return np.sqrt(T * np.trapezoid((omega_fn(f) / On) ** 2, f))
