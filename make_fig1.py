#!/usr/bin/env python3
"""
TomoCosmo -- Figure 1: inflationary GWs through a stiff era and the DM coupling
================================================================================

Panel (a): today's inflationary GW spectrum h^2 Omega_GW(f) for standard radiation domination (RD),
           for stiff eras of several w, for w = 1 (kination, excluded) and for an early
           matter-dominated era (EMD), with existing bounds (shaded) and projected detector
           sensitivities (lines).
Panel (b): the DM coupling needed for Omega h^2 = 0.12, relative to RD, against the SNR of the
           tensor signal in a chosen detector. Each point of a curve is one value of w.

Run (after compute_snr_vs_w.py):   python3 make_fig1.py        (about 30 s)
Output: figures/fig1.pdf and figures/fig1.png

-------------------------------------------------------------------------------------------------
HOW TO CHANGE THE FIGURE  (all settings are in SECTIONS 1 and 2 below; nothing else needs editing)
-------------------------------------------------------------------------------------------------
 What you want to change                          Where
 ------------------------------------------------ -----------------------------------------------
 the w values drawn in panel (a)                  W_LIST                                (Sec. 1)
 r, T_tr, T_K of the stiff era                    R_TENSOR, T_TR, T_K  (Sec. 1)  -> also change
                                                  them in compute_snr_vs_w.py and re-run it
 the EMD example (or remove it)                   EMD, SHOW_EMD                         (Sec. 1)
 a generic w(T) history instead of constant w     function history() at the end of Sec. 1
 the second, dashed r in panel (b)                R_LOW (None removes the dashed curves)
 freeze-out DM mass in panel (b)                  FO_KEY: "fo10" (10 GeV) or "fo1" (1 GeV)
 detector on the x-axis of panel (b)              SNR_DETECTOR: "LISA", "DECIGO" or "ET"
 detectability threshold in panel (b)             SNR_THRESHOLD
 bounds (Planck, Simons Obs., LVK)                BOUNDS                                (Sec. 1)
 which detector curves are drawn                  SHOW_DETECTORS                        (Sec. 1)
 colours, line widths, line styles                COLORS, STYLE                         (Sec. 2)
 axis labels and panel titles                     LABELS                                (Sec. 2)
 legend entries (key text)                        LEGEND                                (Sec. 2)
 axis ranges                                      LIMITS                                (Sec. 2)
 position of the detector names                   DETECTOR_LABELS                       (Sec. 2)
 position of other text on the plot               TEXT                                  (Sec. 2)
 font sizes, figure size                          FONT, FIGSIZE                         (Sec. 2)
 file names and formats                           OUTPUT                                (Sec. 2)
 a new detector                                   add it to DETECTORS in detectors.py, run
                                                  make_sensitivity_curves.py, add its name here
-------------------------------------------------------------------------------------------------
"""
import json
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

import tomocosmo_proto as tc
import gwsources as gs

# =================================================================================================
# SECTION 1. Physics settings
# =================================================================================================
T_INI, NPTS = 1e13, 16000        # lowest starting temperature [GeV] and grid points (as in the scan)
T_TR = 0.01                      # end of the stiff era [GeV]
T_K = 1e3                        # start of the stiff era [GeV]
R_TENSOR = 0.035                 # tensor-to-scalar ratio (panel a, solid curves in panel b)
R_LOW = 0.005                    # dashed curves in panel (b); None to leave them out
W_LIST = (0.5, 0.6, 0.7, 0.8, 0.9)    # stiff eras drawn in panel (a) and marked in panel (b)
SHOW_KINATION = True             # draw w = 1 (excluded) in panel (a)
SHOW_EMD = True                  # draw the EMD example in panel (a)
EMD = dict(T_dom=100.0, T_RH=10.0)    # EMD example: X would dominate at 100 GeV, decays at T_RH = 10 GeV

FO_KEY = "fo10"                  # freeze-out curve of panel (b): "fo10" (m = 10 GeV) or "fo1" (1 GeV)
SNR_DETECTOR = "LISA"            # x-axis of panel (b): "LISA", "DECIGO" or "ET"
SNR_THRESHOLD = 10.0             # shaded "detected" region of panel (b)

BOUNDS = dict(
    planck=5.6e-6 * 0.30,        # integrated bound from Planck+BAO, Delta N_eff < 0.3 (drawn as a line)
    simons=5.6e-6 * 0.10,        # projected Simons Observatory reach, Delta N_eff ~ 0.1
    lvk=1.27e-9,                 # LVK O4a upper limit on a flat spectrum
    lvk_band=(20.0, 76.6),       # Hz, band of that limit
)
SHOW_DETECTORS = ("SKA", "LISA", "DECIGO", "ET", "CE", "A+ (HL+LIGO-India)")
FREQ = np.logspace(-12, 4, 700)  # frequencies of panel (a) [Hz]


def history(w):
    """the expansion history used for each w in panel (a).
    To draw a generic w(T) instead, return for example
        tc.Background("w_of_T", T_ini=T_INI, npts=NPTS, w_of_T=lambda T: ..., T_norm=T_TR)"""
    return tc.Background("stiff", T_ini=tc.T_ini_stiff(T_TR, T_K, w, T_min=T_INI), npts=NPTS,
                         T_tr=T_TR, T_K=T_K, w=w)


# =================================================================================================
# SECTION 2. Appearance
# =================================================================================================
FIGSIZE = (7.28, 2.5)            # inches (full text width of an A4 proposal page)
FONT = dict(base=6.5, title=7.0, axis=6.7, legend=5.0, label=5.2, small=4.8)
OUTPUT = dict(pdf="figures/fig1.pdf", png="figures/fig1.png", dpi=300)

COLORS = dict(
    ink="#1f2933", muted="#52606d", grid="#9aa5b1",      # text, secondary text, bound shading
    rd="#3b4754",                                        # RD curve and RD reference line
    stiff_cmap="Reds",                                   # colour map for the w curves
    kination="#1F5F8B", emd="#1F5F8B",                   # w = 1 and EMD curves
    detector="#7b8794", aplus="#1f2933",                 # detector sensitivity curves
    freeze_in="#7A1F2B",                                 # maroon
    freeze_out="#D9822B",                                # orange
    detected="#7A1F2B",                                  # shading of the detectable region in (b)
)
STYLE = dict(
    rd=dict(lw=0.9, ls=(0, (3, 1.5))),
    stiff=dict(lw=1.25),
    kination=dict(lw=0.9, ls=(0, (1, 1))),
    emd=dict(lw=1.0, ls=(0, (4, 1.2, 1, 1.2))),
    solid_b=dict(lw=1.3), dashed_b=dict(lw=1.0, ls=(0, (3, 1.6))),
)
LABELS = dict(
    a_x="frequency $f$ [Hz]", a_y=r"$h^2\Omega_{\rm GW}$",
    a_title="(a)  Inflationary GWs through a stiff era",
    b_x=f"SNR [{SNR_DETECTOR}]", b_y=r"$g_{\rm DM}/g_{\rm DM}^{\rm RD}$",
    b_title="(b)  GW signal vs DM coupling Correlation",
)
LEGEND = dict(
    w=lambda w: "$w=%.1f$" % w, kination="$w=1$ (excluded)", rd="RD", emd="EMD",
    r_high=r"$r=%g$" % R_TENSOR, r_low=r"$r=%g$" % (R_LOW or 0),
)
LIMITS = dict(a_x=(1e-12, 1e4), a_y=(1e-19, 1e-4), b_x=(1e-3, 3e5), b_y=(0.85, 30))
# detector names: (where, height h^2 Omega, horizontal factor or vertical factor)
#   "left_arm"  : left of the falling arm at that height     "right_arm": left of the rising arm
#   "below_min" : below the deepest point                    "upper_end": left of the upper end of the curve
DETECTOR_LABELS = {
    "LISA": ("left_arm", 3e-9, 0.85), "ET": ("left_arm", 1e-11, 0.80), "SKA": ("right_arm", 3e-8, 0.80),
    "DECIGO": ("below_min", None, 0.22), "CE": ("below_min", None, 0.22),
    "A+ (HL+LIGO-India)": ("upper_end", None, 0.85),
}
DETECTOR_NAMES = {"A+ (HL+LIGO-India)": "A+"}            # short names printed on the plot
TEXT = dict(                                             # (x, y) positions in data units
    planck=(2e-12, 1.6), simons=(2e-12, 0.32),           # y as a factor of the bound
    lvk=5.5e-6, t_tr=3e-19,
    detected=(14, 26), rd=(1.3e-3, 1.04),
    freeze_in=(9e2, 11.5, 27), freeze_out=(1.5e4, 2.45, 14),   # (x, y, rotation)
)

plt.rcParams.update({
    "font.size": FONT["base"], "axes.linewidth": 0.5, "axes.titlesize": FONT["title"],
    "axes.labelsize": FONT["axis"], "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "mathtext.fontset": "dejavusans", "pdf.fonttype": 42})

# =================================================================================================
# SECTION 3. Panel (a): spectra
# =================================================================================================
bg_rd = tc.Background("std", T_ini=T_INI, npts=NPTS)
spec_rd = gs.omega_inflation(bg_rd, FREQ, r=R_TENSOR)
backgrounds = {w: history(w) for w in W_LIST + ((1.0,) if SHOW_KINATION else ())}
spec_w = {w: gs.omega_inflation(bg, FREQ, r=R_TENSOR) for w, bg in backgrounds.items()}
f_tr = gs.f_of_T(backgrounds[W_LIST[0]], T_TR)           # mode that re-entered at T_tr
if SHOW_EMD:
    spec_emd = gs.omega_inflation(tc.Background("emd", T_ini=T_INI, npts=NPTS, **EMD), FREQ, r=R_TENSOR)
curves = np.load("data/sensitivity_curves.npz")          # made by make_sensitivity_curves.py

# =================================================================================================
# SECTION 4. Panel (b): rate ratios -> coupling ratios
# =================================================================================================
# Freeze-out: annihilation into SM states through a heavier mediator with fixed mass and SM coupling,
#   so <sigma v> ~ g_DM^2.   Freeze-in: decay of the parent into DM, Gamma ~ g_DM^2.
# Hence g_DM / g_DM^RD = (rate / rate_RD)^(1/2) in both cases.
S = json.load(open("data/snr_vs_w.json"))
for key, val in (("T_TR", T_TR), ("T_K", T_K)):
    if not np.isclose(S["params"][key], val):
        warnings.warn(f"{key} differs from the value used in compute_snr_vs_w.py; re-run that script.")
w_scan = np.array(S["w"])
snr = np.array(S[SNR_DETECTOR]) * (R_TENSOR / S["params"]["r"])   # SNR scales linearly with r
g_fo = np.array(S[FO_KEY]) ** 0.5
g_fi = np.array(S["fi"]) ** 0.5

# =================================================================================================
# SECTION 5. Drawing (normally no changes needed below)
# =================================================================================================
C = COLORS
C_W = {w: plt.get_cmap(C["stiff_cmap"])(0.40 + 0.55 * (w - min(W_LIST)) / max(max(W_LIST) - min(W_LIST), 1e-9))
       for w in W_LIST}


def pow10(x, _):
    if x <= 0:
        return ""
    e = int(round(np.log10(x)))
    return "$1$" if e == 0 else ("$10$" if e == 1 else "$10^{%d}$" % e)


def style_log_axes(axis_obj):
    axis_obj.set_xscale("log"); axis_obj.set_yscale("log")
    for axis in (axis_obj.xaxis, axis_obj.yaxis):
        axis.set_major_formatter(FuncFormatter(pow10)); axis.set_minor_formatter(NullFormatter())
    axis_obj.tick_params(which="both", colors=C["muted"], labelcolor=C["ink"], pad=2)
    for sp in axis_obj.spines.values():
        sp.set_color(C["muted"])


def label_detector(axis_obj, key):
    where, y, fac = DETECTOR_LABELS[key]
    fc, oc = curves["pls_f_" + key], curves["pls_o_" + key]
    k = oc.argmin()
    name = DETECTOR_NAMES.get(key, key)
    col = C["aplus"] if key.startswith("A+") else C["muted"]
    kw = dict(color=col, fontsize=FONT["label"])
    if where == "left_arm":
        x = np.interp(np.log10(y), np.log10(oc[:k + 1][::-1]), fc[:k + 1][::-1])
        axis_obj.text(x * fac, y, name, ha="right", va="center", **kw)
    elif where == "right_arm":
        x = np.interp(np.log10(y), np.log10(oc[k:]), fc[k:])
        axis_obj.text(x * fac, y, name, ha="right", va="center", **kw)
    elif where == "below_min":
        axis_obj.text(fc[k], oc[k] * fac, name, ha="center", va="top", **kw)
    elif where == "upper_end":
        axis_obj.text(fc[0] * fac, oc[0], name, ha="right", va="center", **kw)


fig = plt.figure(figsize=FIGSIZE)
gsp = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22, left=0.06, right=0.99, top=0.91, bottom=0.15)
ax, bx = fig.add_subplot(gsp[0]), fig.add_subplot(gsp[1])

# ---- (a) existing bounds: shaded
P, f1, f2 = BOUNDS["planck"], *BOUNDS["lvk_band"]
ax.axhspan(P, 1e0, color=C["grid"], alpha=0.22, lw=0)
ax.axhline(P, color=C["muted"], lw=0.6)
ax.text(TEXT["planck"][0], P * TEXT["planck"][1], r"Planck+BAO $\Delta N_{\rm eff}$ (excluded)",
        fontsize=FONT["legend"], color=C["muted"])
ax.fill_between([f1, f2], BOUNDS["lvk"], 1e0, color=C["grid"], alpha=0.30, lw=0)
ax.plot([f1, f2], [BOUNDS["lvk"]] * 2, color=C["ink"], lw=0.7)
ax.text(np.sqrt(f1 * f2), TEXT["lvk"], "LVK O4a", color=C["ink"], fontsize=FONT["legend"],
        ha="center", va="center", rotation=90)

# ---- (a) projections: lines only
ax.axhline(BOUNDS["simons"], color=C["muted"], lw=0.6, ls=(0, (3, 1.5)))
ax.text(TEXT["simons"][0], BOUNDS["simons"] * TEXT["simons"][1], "Simons Obs. (proj.)",
        fontsize=FONT["legend"], color=C["muted"])
for key in SHOW_DETECTORS:
    if key.startswith("A+"):
        ax.plot(curves["pls_f_" + key], curves["pls_o_" + key], color=C["aplus"], lw=0.7, ls=(0, (2.2, 1.2)))
    else:
        ax.plot(curves["pls_f_" + key], curves["pls_o_" + key], color=C["detector"], lw=0.6)
    if key in DETECTOR_LABELS:
        label_detector(ax, key)

# ---- (a) spectra
ax.plot(FREQ, spec_rd, color=C["rd"], **STYLE["rd"])
handles, names = [], []
for w in W_LIST:
    ax.plot(FREQ, spec_w[w], color=C_W[w], **STYLE["stiff"])
    handles.append(Line2D([], [], color=C_W[w], **STYLE["stiff"])); names.append(LEGEND["w"](w))
if SHOW_KINATION:
    ax.plot(FREQ, spec_w[1.0], color=C["kination"], **STYLE["kination"])
    handles.append(Line2D([], [], color=C["kination"], **STYLE["kination"])); names.append(LEGEND["kination"])
handles.append(Line2D([], [], color=C["rd"], **STYLE["rd"])); names.append(LEGEND["rd"])
if SHOW_EMD:
    ax.plot(FREQ, spec_emd, color=C["emd"], **STYLE["emd"])
    handles.append(Line2D([], [], color=C["emd"], **STYLE["emd"])); names.append(LEGEND["emd"])
ax.axvline(f_tr, color=C["grid"], lw=0.5, ls=":")
ax.text(f_tr * 1.27, TEXT["t_tr"], r"$T_{\rm tr}$", fontsize=FONT["label"], color=C["muted"])
ax.legend(handles, names, loc="upper left", bbox_to_anchor=(0.0, 0.81), fontsize=FONT["legend"],
          frameon=False, handlelength=1.8, labelspacing=0.25, borderpad=0.3)
style_log_axes(ax)
ax.xaxis.set_major_locator(LogLocator(base=10, numticks=9))
ax.set_xlim(*LIMITS["a_x"]); ax.set_ylim(*LIMITS["a_y"])
ax.set_xlabel(LABELS["a_x"], labelpad=1); ax.set_ylabel(LABELS["a_y"], labelpad=1)
ax.set_title(LABELS["a_title"], loc="left", pad=3, color=C["ink"])

# ---- (b) detectable region and RD reference
bx.axvspan(SNR_THRESHOLD, LIMITS["b_x"][1], color=C["detected"], alpha=0.06, lw=0)
bx.axvline(SNR_THRESHOLD, color=C["muted"], lw=0.6, ls=(0, (3, 1.5)))
bx.text(*TEXT["detected"], f"detected by {SNR_DETECTOR}\n(SNR > {SNR_THRESHOLD:g})",
        fontsize=FONT["legend"], color=C["muted"], va="top")
bx.axhline(1, color=C["rd"], lw=0.7, ls=(0, (3, 1.5)))
bx.text(*TEXT["rd"], "RD: standard coupling", fontsize=FONT["legend"], color=C["rd"], va="bottom")

# ---- (b) coupling curves
for g, col in ((g_fi, C["freeze_in"]), (g_fo, C["freeze_out"])):
    if R_LOW:
        bx.plot(snr * R_LOW / R_TENSOR, g, color=col, **STYLE["dashed_b"])
    bx.plot(snr, g, color=col, **STYLE["solid_b"])
for wm in W_LIST:                                      # mark the w values of panel (a)
    j = int(np.argmin(abs(w_scan - wm)))
    for g, col in ((g_fi, C["freeze_in"]), (g_fo, C["freeze_out"])):
        bx.plot(snr[j], g[j], "o", ms=2.6, mfc=C_W[wm], mec=col, mew=0.5, zorder=5)
    bx.text(snr[j] * 1.6, g_fi[j] * 0.93, r"$w$=%.1f" % wm if wm == W_LIST[0] else "%.1f" % wm,
            fontsize=FONT["small"], color=C["muted"], ha="left", va="top")
x, y, rot = TEXT["freeze_in"]
bx.text(x, y, "freeze-in", fontsize=5.4, color=C["freeze_in"], rotation=rot, ha="center")
x, y, rot = TEXT["freeze_out"]
bx.text(x, y, "freeze-out", fontsize=5.4, color=C["freeze_out"], rotation=rot, ha="center")
if R_LOW:
    bx.legend([Line2D([], [], color=C["muted"], **STYLE["solid_b"]), Line2D([], [], color=C["muted"], **STYLE["dashed_b"])],
              [LEGEND["r_high"], LEGEND["r_low"]], loc="upper left", bbox_to_anchor=(0.0, 0.9),
              fontsize=FONT["label"], frameon=True, framealpha=0.85, edgecolor="none",
              handlelength=2.0, labelspacing=0.25, borderpad=0.3)
style_log_axes(bx)
bx.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
bx.set_xlim(*LIMITS["b_x"]); bx.set_ylim(*LIMITS["b_y"])
bx.set_xlabel(LABELS["b_x"], labelpad=1); bx.set_ylabel(LABELS["b_y"], labelpad=1)
bx.set_title(LABELS["b_title"], loc="left", pad=3, color=C["ink"])

# =================================================================================================
# SECTION 6. Save
# =================================================================================================
fig.savefig(OUTPUT["pdf"])
fig.savefig(OUTPUT["png"], dpi=OUTPUT["dpi"])
print(f"saved {OUTPUT['pdf']} and {OUTPUT['png']}")
