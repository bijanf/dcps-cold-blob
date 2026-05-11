"""Single consolidated cross-era figure (3 panels) covering 5 timescales:

Panel (a):  PALMOD-130k Holocene subpolar NA SST anomaly stack (paleo).
Panel (b):  HadISST 1870-2023 subpolar-subtropical NA SST contrast anomaly
            with CMIP6 historical (1850-2014) and CMIP6 ssp585 (2015-2100)
            ensemble overlays.
Panel (c):  Like-for-like Sen-slope rate comparison across five timescales:
            late Holocene PALMOD, CMIP6 historical, HadISST observed,
            CMIP6 ssp245 projection, CMIP6 ssp585 projection. Horizontal
            bar chart on a linear °C/kyr-equivalent axis. CMIP6 bars use
            ensemble median with p16-p84 across-model spread as error.

No text inside panels; numerical values in the caption.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT


CACHE_PALMOD = CACHE_DIR / "palmod"
CACHE_CROSS  = CACHE_DIR / "cross_era"
CACHE_CMIP6  = CACHE_DIR / "cmip6_contrast"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"


def _cmip6_ensemble(json_data: dict, exp: str
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ensemble median + 16th-84th percentile of contrast anomaly
    time series per year for one CMIP6 experiment."""
    entries = list(json_data.get(exp, {}).values())
    if not entries:
        return None, None, None, None
    # Per-model years + contrast_anom_degC
    yrs_all = []
    arrs_all = []
    for e in entries:
        if "years" not in e: continue
        yrs_all.append(np.array(e["years"]))
        arrs_all.append(np.array(e["contrast_anom_degC"]))
    if not yrs_all:
        return None, None, None, None
    # Common year axis as union; align each model.
    yr_min = min(y.min() for y in yrs_all)
    yr_max = max(y.max() for y in yrs_all)
    years = np.arange(yr_min, yr_max + 1)
    grid = np.full((len(arrs_all), len(years)), np.nan)
    for i, (y, a) in enumerate(zip(yrs_all, arrs_all)):
        idx = np.searchsorted(years, y)
        grid[i, idx] = a
    p50 = np.nanmedian(grid, axis=0)
    p16 = np.nanpercentile(grid, 16, axis=0)
    p84 = np.nanpercentile(grid, 84, axis=0)
    return years, p50, p16, p84


def _scenario_rates(json_data: dict, exp: str) -> tuple[float, float, float]:
    """Median, p16, p84 Sen slope (°C/century) across the ensemble."""
    slopes = [r["sen_degC_per_century"]
              for r in json_data.get(exp, {}).values()]
    if not slopes:
        return float("nan"), float("nan"), float("nan")
    return (float(np.percentile(slopes, 50)),
            float(np.percentile(slopes, 16)),
            float(np.percentile(slopes, 84)))


def main():
    # ----- Load cached results --------------------------------------------
    palmod_npz = np.load(CACHE_PALMOD / "holocene_stack.npz", allow_pickle=True)
    palmod_json = json.loads((CACHE_PALMOD / "holocene_stack.json").read_text())
    cross_npz = np.load(CACHE_CROSS / "cross_era.npz", allow_pickle=True)
    cross_json = json.loads((CACHE_CROSS / "cross_era.json").read_text())
    cmip6_json = json.loads((CACHE_CMIP6 / "cmip6_contrast.json").read_text())

    target_kyr = palmod_npz["target_kyr"]
    basin_mean = palmod_npz["basin_mean"]
    matrix = palmod_npz["matrix"]
    late = palmod_json["windows"]["late_holocene_headline"]

    anthro_years = cross_npz["anthro_years"]
    anthro_contrast_anom = cross_npz["anthro_contrast_anom"]
    anthro_sen_per_century = cross_json["anthro"]["sen_degC_per_century"]

    # CMIP6 ensemble overlays for panel (b)
    yrs_hist, hist_p50, hist_p16, hist_p84 = _cmip6_ensemble(cmip6_json, "historical")
    yrs_585, s585_p50, s585_p16, s585_p84 = _cmip6_ensemble(cmip6_json, "ssp585")
    yrs_245, s245_p50, s245_p16, s245_p84 = _cmip6_ensemble(cmip6_json, "ssp245")

    # Per-scenario Sen-slope ensemble stats (°C/century)
    hist_med, hist_lo, hist_hi = _scenario_rates(cmip6_json, "historical")
    s585_med, s585_lo, s585_hi = _scenario_rates(cmip6_json, "ssp585")
    s245_med, s245_lo, s245_hi = _scenario_rates(cmip6_json, "ssp245")

    # Convert all rates to °C / kyr equivalent, signed.
    paleo_sen = late["sen_slope_degC_per_kyr"]  # +0.23 ka (cooling toward present)
    # paleo cooling is "subpolar absolute cools at +0.23 °C/kyr in age units";
    # the equivalent contrast-style cooling rate is -0.23 °C/kyr toward present
    paleo_rate_kyr = -abs(paleo_sen)
    paleo_lo_kyr   = -abs(late["bootstrap_hi_degC_per_kyr"])   # |hi| (less neg)
    paleo_hi_kyr   = -abs(late["bootstrap_lo_degC_per_kyr"])   # |lo| (more neg)
    anthro_rate_kyr = anthro_sen_per_century * 10.0          # /century -> /kyr
    a_lo = cross_json["anthro"]["bootstrap_lo_per_kyr"]
    a_hi = cross_json["anthro"]["bootstrap_hi_per_kyr"]

    hist_med_kyr = hist_med * 10.0; hist_lo_kyr = hist_lo * 10.0; hist_hi_kyr = hist_hi * 10.0
    s245_med_kyr = s245_med * 10.0; s245_lo_kyr = s245_lo * 10.0; s245_hi_kyr = s245_hi * 10.0
    s585_med_kyr = s585_med * 10.0; s585_lo_kyr = s585_lo * 10.0; s585_hi_kyr = s585_hi * 10.0

    # ----- Build figure ---------------------------------------------------
    fig = plt.figure(figsize=(13.5, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.7, 1.0],
        width_ratios=[1.50, 1.05],
    )

    # Panel (a): PALMOD Holocene stack
    ax_a = fig.add_subplot(gs[0, 0])
    for i in range(matrix.shape[0]):
        ax_a.plot(target_kyr, matrix[i, :], color="0.72", lw=0.6, alpha=0.55)
    ax_a.plot(target_kyr, basin_mean, color="C0", lw=2.2)
    m_late = (target_kyr <= 5.0) & np.isfinite(basin_mean)
    if m_late.sum() >= 2:
        coef = np.polyfit(target_kyr[m_late], basin_mean[m_late], 1)
        ax_a.plot(target_kyr[m_late], np.polyval(coef, target_kyr[m_late]),
                  color="C0", lw=1.4, linestyle="--")
    ax_a.axvspan(0.0, 5.0, color="C0", alpha=0.10, zorder=0)
    ax_a.axvspan(5.0, 7.0, color="0.85", alpha=0.65, zorder=0)
    ax_a.axhline(0, color="0.4", lw=0.5)
    ax_a.invert_xaxis()
    ax_a.set_xlabel("age (ka BP)")
    ax_a.set_ylabel("SST anomaly (°C, ref 5--7 ka)")
    ax_a.set_title("(a) PALMOD Holocene subpolar NA stack",
                   loc="left", fontsize=10)
    ax_a.grid(alpha=0.20)

    # Panel (b): HadISST + CMIP6 ensemble overlays
    ax_b = fig.add_subplot(gs[0, 1])
    # CMIP6 historical band + median
    if yrs_hist is not None:
        ax_b.fill_between(yrs_hist, hist_p16, hist_p84,
                           color="0.55", alpha=0.30)
        ax_b.plot(yrs_hist, hist_p50, color="0.30", lw=1.4)
    # CMIP6 ssp585 (future projection)
    if yrs_585 is not None:
        ax_b.fill_between(yrs_585, s585_p16, s585_p84,
                           color="C3", alpha=0.18)
        ax_b.plot(yrs_585, s585_p50, color="C3", lw=1.4, linestyle="-")
    # CMIP6 ssp245 (future projection, moderate)
    if yrs_245 is not None:
        ax_b.plot(yrs_245, s245_p50, color="C1", lw=1.2, linestyle="--")
    # HadISST observed (foreground)
    ax_b.plot(anthro_years, anthro_contrast_anom, color="k", lw=1.8)
    # Linear fit on HadISST
    coef = np.polyfit(anthro_years, anthro_contrast_anom, 1)
    ax_b.plot(anthro_years, np.polyval(coef, anthro_years),
              color="k", lw=1.0, linestyle="--")
    ax_b.axhline(0, color="0.4", lw=0.5)
    ax_b.axvline(2014.5, color="0.4", lw=0.5, linestyle=":")
    ax_b.set_xlabel("year")
    ax_b.set_ylabel("contrast anomaly (°C)")
    ax_b.set_title("(b) HadISST 1870--2023 + CMIP6 historical and ssp245/585",
                   loc="left", fontsize=10)
    ax_b.set_xlim(1865, 2105)
    ax_b.grid(alpha=0.20)

    # Panel (c): Five-bar rate comparison (linear °C/kyr equivalent)
    ax_c = fig.add_subplot(gs[1, :])
    labels = [
        "Late Holocene\nPALMOD (0-5 ka)",
        "CMIP6 historical\n(1870-2014)",
        "Observed HadISST\n(1870-2023)",
        "CMIP6 ssp245\n(2015-2100)",
        "CMIP6 ssp585\n(2015-2100)",
    ]
    rates = [paleo_rate_kyr, hist_med_kyr, anthro_rate_kyr,
              s245_med_kyr, s585_med_kyr]
    lo_errs = [
        max(paleo_rate_kyr - paleo_lo_kyr, 0.001),
        max(hist_med_kyr - hist_lo_kyr, 0.001),
        max(anthro_rate_kyr - a_lo, 0.001),
        max(s245_med_kyr - s245_lo_kyr, 0.001),
        max(s585_med_kyr - s585_lo_kyr, 0.001),
    ]
    hi_errs = [
        max(paleo_hi_kyr - paleo_rate_kyr, 0.001),
        max(hist_hi_kyr - hist_med_kyr, 0.001),
        max(a_hi - anthro_rate_kyr, 0.001),
        max(s245_hi_kyr - s245_med_kyr, 0.001),
        max(s585_hi_kyr - s585_med_kyr, 0.001),
    ]
    colors = ["C0", "0.5", "k", "C1", "C3"]
    y_pos = np.arange(len(labels))[::-1]
    ax_c.barh(y_pos, rates, height=0.55,
              xerr=[lo_errs, hi_errs],
              color=colors, alpha=0.85, edgecolor="k", linewidth=0.6,
              error_kw=dict(ecolor="0.2", capsize=4, lw=1.1))
    ax_c.axvline(0, color="0.3", lw=0.8)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(labels, fontsize=9)
    ax_c.set_xlabel("contrast trend (°C per kyr equivalent; negative = subpolar cools relative to subtropics)")
    ax_c.set_title("(c) Like-for-like trend comparison across five timescales",
                   loc="left", fontsize=10)
    ax_c.grid(axis="x", which="both", alpha=0.25)

    out_fig = MANUSCRIPT_FIGS / "fig_cross_era.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")
    print()
    print("Rate ladder (°C/kyr equivalent):")
    for lab, r, lo, hi in zip(labels, rates, lo_errs, hi_errs):
        print(f"  {lab.replace(chr(10), ' '):42s}  {r:+7.3f}  "
              f"[lo:{r-lo:+7.3f}  hi:{r+hi:+7.3f}]")


if __name__ == "__main__":
    main()
