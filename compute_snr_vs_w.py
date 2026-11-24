#!/usr/bin/env python3
"""
TomoCosmo -- scan over the equation of state w of a stiff era
=============================================================

For each w this script computes, on ONE expansion history,
  * the SNR of the inflationary tensor background in LISA, DECIGO and ET;
  * the freeze-out cross section <sigma v> that gives Omega h^2 = 0.12 (DM masses 1 and 10 GeV);
  * the freeze-in decay width Gamma that gives Omega h^2 = 0.12 (0.1 GeV DM from a 10 GeV parent);
and divides the DM rates by their RD values. The result is written to data/snr_vs_w.json,
which make_fig1.py reads.

History assumed for this demonstration
--------------------------------------
A fluid that behaves like matter above T_K and like a stiff fluid of CONSTANT w between T_K and
T_tr, after which radiation dominates again:
      RD -> matter-like phase -> stiff era (w) -> RD           (kind="stiff" in tomocosmo_proto)
Any other history can be scanned in the same way. For a generic equation of state w(T), replace
the line marked "HISTORY" below by, for example,
      bg = tc.Background("w_of_T", T_ini=T_ini, npts=NPTS, w_of_T=my_w, T_norm=T_TR, ratio_norm=1.0)
with my_w(T) a function returning w at temperature T.

Before any ratio is formed, the RD results are checked against analytic formulae (validation.py);
the script stops if a check fails.

Run:  python3 compute_snr_vs_w.py      (about 30 s)
"""
import json
import sys

import numpy as np

import tomocosmo_proto as tc
import gwsources as gs
import detectors as dt
from validation import run_rd_checks

# =============================================================================
# 1. Parameters (use the same values in make_fig1.py)
# =============================================================================
T_INI = 1e13      # GeV, lowest starting temperature of the GW backgrounds (raised automatically
                  # for w close to 1 so that the matter-like phase starts inside the grid)
NPTS = 16000      # grid points in N = ln a
T_TR = 0.01       # GeV, end of the stiff era (10 MeV, safely above BBN)
T_K = 1e3         # GeV, start of the stiff era (end of the matter-like phase)
R_TENSOR = 0.035  # tensor-to-scalar ratio; the SNR scales linearly with r
W_GRID = np.round(np.arange(0.40, 0.961, 0.02), 3)   # w = 0.40, 0.42, ..., 0.96

M_DM_FO = (1.0, 10.0)            # GeV, DM masses for freeze-out (both stored; pick one in make_fig1.py, FO_KEY)
M_DM_FI, M_PARENT = 0.1, 10.0    # GeV, DM and parent masses for freeze-in
DETECTORS = ("LISA", "DECIGO", "ET")   # SNRs stored for all three; any can be plotted (SNR_DETECTOR in make_fig1.py)

T_INI_DM, NPTS_DM = 1e6, 12000   # GeV-scale DM only needs the history below 10^6 GeV

# =============================================================================
# 2. Check the RD results, then compute the RD reference rates
# =============================================================================
if not all(row[-1] for row in run_rd_checks()):
    sys.exit("An RD check failed: fix this before computing ratios to RD.")

bg_rd = tc.Background("std", T_ini=T_INI_DM, npts=NPTS_DM)
sigv_rd = {m: tc.sigv_required(bg_rd, m) for m in M_DM_FO}
gamma_rd = tc.freeze_in_gamma(bg_rd, M_DM_FI, M_PARENT)
print("\nRD reference values")
for m in M_DM_FO:
    print(f"  freeze-out <sigma v> (m = {m:g} GeV) = {sigv_rd[m]:.3e} cm^3/s")
print(f"  freeze-in  Gamma ({M_PARENT:g} GeV -> {M_DM_FI:g} GeV DM) = {gamma_rd:.3e} GeV\n")

# =============================================================================
# 3. Scan over w
# =============================================================================
out = {"w": [], "fo1": [], "fo10": [], "fi": [], **{d: [] for d in DETECTORS},
       "params": dict(T_TR=T_TR, T_K=T_K, r=R_TENSOR, M_DM_FO=M_DM_FO, M_DM_FI=M_DM_FI, M_PARENT=M_PARENT),
       "rd_reference": dict(sigv_1=sigv_rd[1.0], sigv_10=sigv_rd[10.0], gamma=gamma_rd)}

for w in W_GRID:
    # (a) GW side: tensor spectrum and SNRs                                  # HISTORY
    T_ini = tc.T_ini_stiff(T_TR, T_K, w, T_min=T_INI)
    bg = tc.Background("stiff", T_ini=T_ini, npts=NPTS, T_tr=T_TR, T_K=T_K, w=w)
    spectrum = lambda f, b=bg: gs.omega_inflation(b, f, r=R_TENSOR)
    for d in DETECTORS:
        out[d].append(float(dt.snr(d, spectrum)))

    # (b) DM side: the same history below 10^6 GeV (rho_phi is analytic, so it is identical there;
    #     the fluid may already dominate at 10^6 GeV, which is fine for GeV-scale DM)
    bg_dm = tc.Background("stiff", T_ini=T_INI_DM, npts=NPTS_DM, check_start=False,
                          T_tr=T_TR, T_K=T_K, w=w)
    out["fo1"].append(tc.sigv_required(bg_dm, 1.0) / sigv_rd[1.0])
    out["fo10"].append(tc.sigv_required(bg_dm, 10.0) / sigv_rd[10.0])
    out["fi"].append(tc.freeze_in_gamma(bg_dm, M_DM_FI, M_PARENT) / gamma_rd)
    out["w"].append(float(w))
    print(f"w = {w:.2f}   SNR(LISA) = {out['LISA'][-1]:9.3g}   "
          f"<sv>/RD (10 GeV) = {out['fo10'][-1]:6.3g}   Gamma/RD = {out['fi'][-1]:7.3g}")

# =============================================================================
# 4. Save (rate ratios; make_fig1.py converts them into coupling ratios)
# =============================================================================
with open("data/snr_vs_w.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("saved data/snr_vs_w.json")
