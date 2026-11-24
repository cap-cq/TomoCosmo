#!/usr/bin/env python3
"""
Automatic checks of TomoCosmo.
Run from the top folder:   python3 -m pytest tests      or      python3 tests/test_tomocosmo.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import tomocosmo_proto as tc        # noqa: E402
import gwsources as gs              # noqa: E402
import detectors as dt              # noqa: E402
from validation import run_rd_checks  # noqa: E402


def test_rd_checks():
    """entropy conservation, tensor plateau and tilt, freeze-out and freeze-in in RD"""
    for name, num, ref, d, tol, ok in run_rd_checks(verbose=False):
        assert ok, f"{name}: {num:.4g} vs {ref:.4g}"


def test_tilt_values():
    """tilt 2(3w-1)/(1+3w) for several w"""
    for w in (0.5, 0.9):
        bg = tc.Background("stiff", T_ini=tc.T_ini_stiff(0.01, 1e3, w), npts=16000, T_tr=0.01, T_K=1e3, w=w)
        o1, o2 = gs.omega_inflation(bg, np.array([1e-7, 1e-6]))
        assert abs(np.log(o2 / o1) / np.log(10.0) - 2 * (3 * w - 1) / (1 + 3 * w)) < 0.02


def test_generic_w_reproduces_stiff():
    """a step-function w(T) given to kind='w_of_T' reproduces the analytic stiff history"""
    w0, T_tr, T_K = 0.7, 0.01, 1e3
    step = lambda T: np.where(T > T_K, 0.0, w0)
    bg_g = tc.Background("w_of_T", T_ini=1e6, npts=40000, check_start=False, w_of_T=step, T_norm=T_tr)
    bg_s = tc.Background("stiff", T_ini=1e6, npts=40000, check_start=False, T_tr=T_tr, T_K=T_K, w=w0)
    assert np.max(abs(bg_g.H / bg_s.H - 1)) < 0.02
    assert abs(tc.sigv_required(bg_g, 10.0) / tc.sigv_required(bg_s, 10.0) - 1) < 0.01


def test_old_names_still_work():
    """kind='kin' and T_kin are accepted as aliases of kind='stiff' and T_tr"""
    a = tc.Background("kin", T_ini=1e6, npts=4000, check_start=False, T_kin=0.01, T_K=1e3, w=0.6)
    b = tc.Background("stiff", T_ini=1e6, npts=4000, check_start=False, T_tr=0.01, T_K=1e3, w=0.6)
    assert np.allclose(a.H, b.H)


def test_emd_dilution():
    """early matter domination example of Fig. 1: T_dom = 100 GeV, T_RH = 10 GeV gives D ~ 11"""
    bg = tc.Background("emd", T_ini=1e6, npts=9000, T_dom=100.0, T_RH=10.0)
    assert 10.0 < bg.D < 12.5
    assert abs(bg.T[-1] / tc.T0 - 1) < 1e-3          # a = 1 today after the normalisation pass


def test_sensitivity_file_up_to_date():
    """data/sensitivity_curves.npz agrees with detectors.pls"""
    stored = np.load(os.path.join("data", "sensitivity_curves.npz"))
    for name in ("LISA", "ET"):
        f, om = dt.pls(name)
        assert np.allclose(om, stored["pls_o_" + name], rtol=1e-6)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"passed  {t.__name__}")
    print(f"all {len(tests)} tests passed")
