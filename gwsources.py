#!/usr/bin/env python3
"""
TomoCosmo -- gravitational-wave spectra on a given expansion history
===================================================================

This file computes today's spectrum of the INFLATIONARY gravitational-wave background,
h^2 Omega_GW(f), on any expansion history built with tomocosmo_proto.Background
(standard RD, a stiff era, a generic w(T), or an early matter era with entropy injection).

How it works
------------
A tensor mode of comoving wavenumber k (frequency today f = k / 2 pi, with a = 1 today) stays frozen
while it is outside the horizon and starts to oscillate and redshift like radiation once it
re-enters, at a_k H_k = k. Its energy today is proportional to

        Omega_GW(k)  ~  P_T  a_k^4 H_k^2 ,         P_T = r A_s  (scale-invariant, n_T = 0).

In RD with constant g_*, a^4 H^2 is constant, which gives the flat RD plateau

        h^2 Omega_GW^RD = (1/24) r A_s Omega_r h^2 (g_k/g_0) (g_s0/g_sk)^(4/3)   (~4.9e-17 for r = 0.035).

The code finds a_k and H_k on the numerical background and multiplies the plateau by
(a_k^4 H_k^2) / (a_k^4 H_k^2)_RD, so any departure from RD (the tilt 2(3w-1)/(1+3w) of an era with
equation of state w, the suppression by entropy injection, the effect of g_*(T)) follows directly
from the background. Re-entry is treated as instantaneous.

Main function
-------------
    omega_inflation(bg, f, r=0.035, As=2.1e-9)   ->  h^2 Omega_GW at the frequencies f [Hz]

It can be used directly for plots, e.g.

    import numpy as np, tomocosmo_proto as tc, gwsources as gs
    bg = tc.Background("stiff", T_ini=1e13, npts=16000, T_tr=0.01, T_K=1e3, w=0.7)
    f  = np.logspace(-12, 4, 700)
    spectrum = gs.omega_inflation(bg, f, r=0.035)

"""
import numpy as np
import tomocosmo_proto as tc

HZ_PER_GEV = 1.0 / tc.GEV_INV_S        # 1 GeV = 1.519e24 Hz (converts a H [GeV] into Hz)
G_HIGH = 106.75                        # SM degrees of freedom at high temperature


def rd_plateau(r=0.035, As=2.1e-9, g_k=G_HIGH):
    """h^2 Omega_GW of the flat RD plateau for modes that entered when g_* = g_*s = g_k"""
    omega_r_over_g0 = (np.pi**2 / 30.0) * tc.T0**4 / tc.RHOC_H2    # Omega_r h^2 / g_0
    return (1.0 / 24.0) * r * As * omega_r_over_g0 * g_k * (tc.GS0 / g_k) ** (4.0 / 3.0)


def reentry(bg, f):
    """scale factor a_k and Hubble rate H_k [GeV] at the (last) horizon re-entry of frequency f [Hz]"""
    k = 2 * np.pi * np.asarray(f) / HZ_PER_GEV         # comoving wavenumber in GeV (a = 1 today)
    lnaH = np.log(bg.a * bg.H)                         # ln(aH) decreases in decelerating eras
    order = np.argsort(lnaH)                           # np.interp needs increasing x
    Nk = np.interp(np.log(k), lnaH[order], bg.N[order])
    return np.exp(Nk), np.exp(np.interp(Nk, bg.N, np.log(bg.H)))


def omega_inflation(bg, f, r=0.035, As=2.1e-9):
    """h^2 Omega_GW(f) today of scale-invariant inflationary tensors (n_T = 0) on the background bg"""
    a_k, H_k = reentry(bg, f)
    X = H_k**2 * a_k**4                                # what the history sets for each mode
    # the same product in RD with g_* = g_*s = G_HIGH:  H^2 a^4 = (pi^2 g / (90 M_P^2)) (g_s0/g)^(4/3) T0^4
    X_rd = (np.pi**2 * G_HIGH / (90 * tc.MP**2)) * (tc.GS0 / G_HIGH) ** (4.0 / 3.0) * tc.T0**4
    return rd_plateau(r, As) * X / X_rd


def f_of_T(bg, T):
    """frequency today [Hz] of the mode that re-entered when the plasma had temperature T (useful
    for marking an era on a plot, e.g. the end of a stiff era)"""
    N = np.interp(-np.log(T), -np.log(bg.T), bg.N)
    aH = np.exp(N) * np.exp(np.interp(N, bg.N, np.log(bg.H)))
    return aH * HZ_PER_GEV / (2 * np.pi)
