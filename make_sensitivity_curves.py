#!/usr/bin/env python3
"""
TomoCosmo -- regenerate the detector sensitivity curves used in Fig. 1(a)
=========================================================================

Computes the power-law-integrated sensitivity curve (SNR = 10) of every detector in
detectors.DETECTORS and stores them in data/sensitivity_curves.npz as arrays
    pls_f_<name>   frequencies [Hz]
    pls_o_<name>   h^2 Omega_GW of the curve
make_fig1.py reads this file, so the figure does not have to recompute the curves each time.
Re-run this script after changing a noise model, a band, an observation time or the SNR threshold.

Run:  python3 make_sensitivity_curves.py      (a few seconds)
"""
import numpy as np
import detectors as dt

SNR_THRESHOLD = 10.0

curves = {}
for name in dt.DETECTORS:
    f, om = dt.pls(name, snr=SNR_THRESHOLD)
    curves["pls_f_" + name], curves["pls_o_" + name] = f, om
    print(f"{name:22s} best sensitivity h^2 Omega = {om.min():.2e} at f = {f[om.argmin()]:.3g} Hz")
curves["snr_threshold"] = np.array(SNR_THRESHOLD)
np.savez("data/sensitivity_curves.npz", **curves)
print("saved data/sensitivity_curves.npz")
