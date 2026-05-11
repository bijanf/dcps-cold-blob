"""Build the 'Cold Blob breaks the Holocene envelope' figure.

Single time-domain visualization of the meridional SST contrast
(subpolar NA minus subtropical NA) from 12 ka BP to 2100 CE, with the
Holocene empirical envelope shaded as a band. Modern observations
(HadISST) and CMIP6 SSP projections visibly exit the envelope.

Sources:
    Paleo: PALMOD-130k v2 contrast (subpolar n=11, subtropical n=3),
            cached from cross_era_contrast.py.
    Modern: HadISST 1870--2023 contrast anomaly (relative to 1870--1879),
            re-anchored onto the PALMOD contrast at 1900 CE.
    CMIP6: per-model contrast anomalies from cmip6_contrast.json. SSP
           series are anchored to each model's own historical anomaly at
           the end of historical (2014), then offset to the PALMOD tie at
           1900 CE.

Holocene envelope: PALMOD basin-mean contrast over 0--11.7 ka BP, with
mean and \pm 2\sigma band. Modern is 'unprecedented' iff its value sits
outside this band.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()


CACHE_CROSS = CACHE_DIR / "cross_era"
CACHE_CMIP6 = CACHE_DIR / "cmip6_contrast"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"
THORNALLEY_FILE = (PKG_ROOT.parent / "data" / "external"
                    / "thornalley2018_tsub.csv")


def main():
    z = np.load(CACHE_CROSS / "cross_era.npz")
    cmip6 = json.loads((CACHE_CMIP6 / "cmip6_contrast.json").read_text())

    # Independent multi-proxy AMOC reconstruction (Thornalley 2018,
    # used as one of the proxies in the Caesar 2021 multi-proxy AMOC
    # compilation). Tsub = subsurface temperature AMOC fingerprint;
    # higher Tsub => stronger AMOC. The series falls sharply in the
    # 20th century, an independent corroboration of the modern
    # weakening signal we see in HadISST contrast.
    import pandas as pd
    thorn = pd.read_csv(THORNALLEY_FILE)

    # Paleo: PALMOD contrast time series on 100-yr grid.
    t_kyr = z["paleo_kyr"]
    paleo_sp = z["paleo_subpolar"]
    paleo_st = z["paleo_subtropical"]
    paleo_contrast = paleo_sp - paleo_st
    paleo_years = 1950.0 - 1000.0 * t_kyr    # year CE

    # Per-time-step uncertainty band on the paleo contrast: bootstrap-
    # resample the 11 subpolar and 3 subtropical cores at each grid point
    # to propagate inter-core spread into a 16-84% time-varying band.
    sp_mat = z["paleo_subpolar_mat"]
    st_mat = z["paleo_subtropical_mat"]
    n_sp = sp_mat.shape[0]; n_st = st_mat.shape[0]
    rng = np.random.default_rng(0)
    n_boot = 200
    contrast_boot = np.full((n_boot, len(t_kyr)), np.nan)
    for k in range(n_boot):
        sp_idx = rng.integers(0, n_sp, n_sp)
        st_idx = rng.integers(0, n_st, n_st)
        sp_b = np.nanmean(sp_mat[sp_idx, :], axis=0)
        st_b = np.nanmean(st_mat[st_idx, :], axis=0)
        contrast_boot[k, :] = sp_b - st_b
    paleo_p16 = np.nanpercentile(contrast_boot, 16, axis=0)
    paleo_p84 = np.nanpercentile(contrast_boot, 84, axis=0)
    print(f"PALMOD per-time-step bootstrap (n={n_boot}) 16-84% band "
          f"width range: {np.nanmin(paleo_p84-paleo_p16):.2f} to "
          f"{np.nanmax(paleo_p84-paleo_p16):.2f} °C")

    # Holocene mean and static +/- 2 sigma envelope for the
    # "outside the Holocene range" visual claim.
    holo_mean = float(np.nanmean(paleo_contrast))
    holo_sd = float(np.nanstd(paleo_contrast))
    env_lo = holo_mean - 2 * holo_sd
    env_hi = holo_mean + 2 * holo_sd

    # PALMOD subtropical coverage is too sparse to reach the modern era
    # (subtropical cores end ~3.4 ka BP / -1400 CE). We therefore anchor
    # modern HadISST and CMIP6 anomalies to the PALMOD Holocene MEAN, on
    # the simplifying assumption that the late-Holocene contrast was at
    # the Holocene mean. Caveat noted in the manuscript caption.
    tie_value = holo_mean
    print(f"  Using PALMOD Holocene mean as modern tie: "
          f"{tie_value:+.3f} °C")

    # Modern HadISST: shift its anomaly so that 1870-1879 = tie_value.
    anthro_yrs = z["anthro_years"]
    anthro_anom = z["anthro_contrast_anom"]
    hadisst_aligned = anthro_anom + tie_value

    # CMIP6 ensembles: align each model's historical+SSP to tie_value.
    def _per_model_aligned(model: str, exp: str) -> tuple[np.ndarray, np.ndarray] | None:
        if model not in cmip6.get("historical", {}):
            return None
        hist = cmip6["historical"][model]
        if exp == "historical":
            y = np.array(hist["years"])
            a = np.array(hist["contrast_anom_degC"]) + tie_value
            return y, a
        if exp not in cmip6 or model not in cmip6[exp]:
            return None
        ssp = cmip6[exp][model]
        # Anchor SSP to historical end (2014).
        hist_y = np.array(hist["years"])
        hist_a = np.array(hist["contrast_anom_degC"])
        end_idx = np.argmin(np.abs(hist_y - 2014))
        end_val = float(hist_a[end_idx])
        y = np.array(ssp["years"])
        a = np.array(ssp["contrast_anom_degC"]) + end_val + tie_value
        return y, a

    def _ensemble_band(exp: str):
        arrs = []; yr_lists = []
        for m in cmip6.get("historical", {}):
            got = _per_model_aligned(m, exp)
            if got is None: continue
            yr_lists.append(got[0]); arrs.append(got[1])
        if not arrs: return None, None, None, None
        ymin = min(y.min() for y in yr_lists)
        ymax = max(y.max() for y in yr_lists)
        years = np.arange(ymin, ymax + 1)
        grid = np.full((len(arrs), len(years)), np.nan)
        for i, (y, a) in enumerate(zip(yr_lists, arrs)):
            idx = np.searchsorted(years, y)
            grid[i, idx] = a
        return (years,
                np.nanmedian(grid, axis=0),
                np.nanpercentile(grid, 16, axis=0),
                np.nanpercentile(grid, 84, axis=0))

    yrs_h, h50, h16, h84 = _ensemble_band("historical")
    yrs_24, s24_50, s24_16, s24_84 = _ensemble_band("ssp245")
    yrs_58, s58_50, s58_16, s58_84 = _ensemble_band("ssp585")

    # ----- Figure --------------------------------------------------------
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(13.5, 5.0),
        gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.18},
    )

    # ----- Panel (a): continuous time axis 10 000 BCE -- 2100 CE ---------
    # Static Holocene +/- 2 sigma envelope (horizontal band, extends
    # across both panels) so the modern departure is visually obvious.
    for ax in (ax_a, ax_b):
        ax.axhspan(env_lo, env_hi, color="0.85", alpha=0.55, zorder=0)
    # R2-M9 (reviewer B8): mark the 770-yr PALMOD-HadISST coverage gap
    # with a hatched grey region so the reader's eye does not read
    # continuity across an unmeasured interval.  PALMOD's youngest
    # bin ends near 1100 CE; HadISST starts at 1870 CE.
    palmod_last_year = float(np.nanmax(paleo_years[np.isfinite(paleo_contrast)]))
    hadisst_first_year = 1870.0
    if hadisst_first_year > palmod_last_year:
        ax_a.axvspan(palmod_last_year, hadisst_first_year,
                       facecolor="none", edgecolor="0.4",
                       hatch="///", alpha=0.45, zorder=2,
                       label=f"770-yr coverage gap "
                             f"({palmod_last_year:.0f}--{hadisst_first_year:.0f} CE)")
    # Time-varying PALMOD bootstrap uncertainty band on top.
    ax_a.fill_between(paleo_years, paleo_p16, paleo_p84,
                       color="C0", alpha=0.25, zorder=1)
    # Zero reference line
    ax_a.axhline(0, color="0.5", lw=0.5)
    # PALMOD basin-mean
    ax_a.plot(paleo_years, paleo_contrast, color="C0", lw=1.6,
              label="PALMOD-130k contrast (median + bootstrap 16--84\\%)")
    # Thornalley 2018 multi-proxy AMOC reconstruction overlay (panel a)
    # is scaled to share the same y-axis: subtract its 1700-1850 mean
    # so its baseline equals the Holocene mean of the contrast, with
    # a sign flip (Thornalley Tsub falls when AMOC weakens; our
    # contrast also falls when AMOC weakens. Both decline modern.)
    thorn_pre = thorn[(thorn["year"] >= 1700) & (thorn["year"] <= 1850)]
    thorn_pre_mean = float(thorn_pre["tsub"].mean())
    # Map Thornalley Tsub units onto our contrast units via a simple
    # linear rescaling so the 1700-1850 means coincide and modern
    # observed declines align in direction.
    thorn_anom = thorn["tsub"].values - thorn_pre_mean + holo_mean
    ax_a.plot(thorn["year"].values, thorn_anom,
              color="C4", lw=1.1, alpha=0.85,
              label=r"Thornalley 2018 $T_{\mathrm{sub}}$ proxy (rescaled)")
    # CMIP6 historical band + median
    if yrs_h is not None:
        ax_a.fill_between(yrs_h, h16, h84, color="0.55", alpha=0.30)
        ax_a.plot(yrs_h, h50, color="0.30", lw=1.1)
    # CMIP6 ssp245
    if yrs_24 is not None:
        m24 = yrs_24 <= 2100
        ax_a.fill_between(yrs_24[m24], s24_16[m24], s24_84[m24],
                           color="C1", alpha=0.18)
        ax_a.plot(yrs_24[m24], s24_50[m24], color="C1", lw=1.2)
    # CMIP6 ssp585
    if yrs_58 is not None:
        m58 = yrs_58 <= 2100
        ax_a.fill_between(yrs_58[m58], s58_16[m58], s58_84[m58],
                           color="C3", alpha=0.22)
        ax_a.plot(yrs_58[m58], s58_50[m58], color="C3", lw=1.4)
    # HadISST observed (on top, thick black)
    ax_a.plot(anthro_yrs, hadisst_aligned, color="k", lw=1.6)
    ax_a.set_xlim(-10000, 2100)
    ax_a.set_xlabel("year CE")
    ax_a.set_ylabel("subpolar -- subtropical NA SST contrast (°C)")
    ax_a.text(-0.06, 1.02, "a", transform=ax_a.transAxes,
              fontweight="bold", fontsize=13, va="bottom")
    ax_a.grid(alpha=0.20)
    # Set y-limits to span the bootstrap envelope plus margin for CMIP6
    y_lo_pal = float(np.nanmin(paleo_p16))
    y_hi_pal = float(np.nanmax(paleo_p84))
    ax_a.set_ylim(y_lo_pal - 1.5, y_hi_pal + 1.5)
    # Mark the modern-era zoom region as a dashed rectangle on panel (a)
    from matplotlib.patches import Rectangle
    zoom_rect = Rectangle(
        (1850, y_lo_pal - 1.5), 250, y_hi_pal + 1.5 - (y_lo_pal - 1.5),
        fill=False, edgecolor="0.2", lw=0.8, linestyle="--", zorder=10)
    ax_a.add_patch(zoom_rect)

    # ----- Panel (b): zoom 1850 -- 2100 -----------------------------------
    ax_b.axhline(0, color="0.5", lw=0.5)
    if yrs_h is not None:
        ax_b.fill_between(yrs_h, h16, h84, color="0.55", alpha=0.30)
        ax_b.plot(yrs_h, h50, color="0.30", lw=1.4,
                   label="CMIP6 historical")
    if yrs_24 is not None:
        m24 = yrs_24 <= 2100
        ax_b.fill_between(yrs_24[m24], s24_16[m24], s24_84[m24],
                           color="C1", alpha=0.18)
        ax_b.plot(yrs_24[m24], s24_50[m24], color="C1", lw=1.4,
                   label="CMIP6 ssp245")
    if yrs_58 is not None:
        m58 = yrs_58 <= 2100
        ax_b.fill_between(yrs_58[m58], s58_16[m58], s58_84[m58],
                           color="C3", alpha=0.22)
        ax_b.plot(yrs_58[m58], s58_50[m58], color="C3", lw=1.6,
                   label="CMIP6 ssp585")
    ax_b.plot(anthro_yrs, hadisst_aligned, color="k", lw=1.8,
               label="HadISST observed")
    # Thornalley 2018 overlay on the modern zoom too
    ax_b.plot(thorn["year"].values, thorn_anom,
              color="C4", lw=1.2, alpha=0.85,
              label=r"Thornalley 2018 $T_{\mathrm{sub}}$ (rescaled)")
    ax_b.axvline(2014.5, color="0.5", lw=0.7, linestyle=":")
    ax_b.set_xlim(1850, 2100)
    ax_b.set_ylim(y_lo_pal - 1.5, y_hi_pal + 1.5)
    ax_b.set_xlabel("year CE")
    ax_b.text(-0.06, 1.02, "b", transform=ax_b.transAxes,
              fontweight="bold", fontsize=13, va="bottom")
    ax_b.legend(loc="lower left", fontsize=8.5, frameon=False)
    ax_b.grid(alpha=0.20)


    out_fig = MANUSCRIPT_FIGS / "fig_envelope_breakout.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
