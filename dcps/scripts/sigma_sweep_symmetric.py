"""R2-BLOCKER 2 (peer review B1): sigma-sensitivity sweep of the
symmetric-noise unprecedented test.

The Round-2 reviewer correctly notes that the band-matched
sigma=30 yr choice in cold_blob_unprecedented_symmetric.py is a
free parameter introduced after the literal sigma=200 yr test
killed the modern signal. To justify the headline, we sweep sigma
over {10, 20, 30, 50, 100, 200} yr applied symmetrically to both
PALMOD and HadISST, then report |z| at 500/1000/2000-yr windows
for each sigma. The sigma at which |z| crosses 3 marks the
boundary of the unprecedented claim.
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from cross_era_contrast import (
    _hadisst_basin_mean, SUBPOLAR_NA, SUBTROPICAL_NA,
    ANTHRO_START, ANTHRO_END,
)


CACHE_OUT = CACHE_DIR / "cold_blob_unprecedented"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

GRID_KYR = 0.1
SIGMAS_YR = [10.0, 20.0, 30.0, 50.0, 100.0, 200.0]
WINDOW_KYRS = [0.5, 1.0, 2.0]
B = 500


def sen_slope(x, y):
    if y.size < 2:
        return float("nan")
    slopes = []
    for i in range(len(y) - 1):
        slopes.append((y[i + 1:] - y[i]) / (x[i + 1:] - x[i]))
    if not slopes:
        return float("nan")
    return float(np.median(np.concatenate(slopes)))


def sliding_sen_slopes(contrast, kyr_grid, window_kyr, min_pts=3):
    n_per = int(round(window_kyr / GRID_KYR))
    slopes = []
    for i in range(len(contrast) - n_per + 1):
        seg = contrast[i:i + n_per]
        t = kyr_grid[i:i + n_per]
        valid = np.isfinite(seg)
        if valid.sum() < min_pts:
            continue
        slopes.append(sen_slope(t[valid], seg[valid]))
    return np.asarray(slopes)


def _smooth_palmod_grid(contrast_raw, sigma_yr):
    out = contrast_raw.copy()
    finite = np.isfinite(out)
    if not finite.any():
        return out
    filled = np.where(finite, out, np.nanmean(out))
    smoothed = gaussian_filter1d(filled, sigma=sigma_yr / 100.0)
    edge = int(np.ceil(3 * sigma_yr / 100.0))
    smoothed[:edge] = np.nan
    smoothed[-edge:] = np.nan
    smoothed[~finite] = np.nan
    return smoothed


def _modern_smoothed_sen(years, contrast_annual, sigma_yr):
    """Smooth HadISST annual contrast and compute Sen slope on the
    central kernel-supported window."""
    if sigma_yr < 0.5:
        smoothed = contrast_annual.copy()
        ctr = slice(None)
    else:
        smoothed = gaussian_filter1d(contrast_annual, sigma=sigma_yr)
        edge = int(np.ceil(sigma_yr))
        if smoothed.size > 2 * edge + 5:
            ctr = slice(edge, smoothed.size - edge)
        else:
            ctr = slice(None)
    sen_per_yr = sen_slope(years[ctr].astype(float), smoothed[ctr])
    return sen_per_yr * 1000.0  # per_kyr


def main():
    CACHE_OUT.mkdir(parents=True, exist_ok=True)
    cross = np.load(CACHE_DIR / "cross_era" / "cross_era.npz")
    cross_json = json.loads(
        (CACHE_DIR / "cross_era" / "cross_era.json").read_text())
    t_kyr = cross["paleo_kyr"]
    contrast_palmod_raw = (cross["paleo_subpolar"].astype(float)
                            - cross["paleo_subtropical"].astype(float))

    sp_anthro = _hadisst_basin_mean(SUBPOLAR_NA, ANTHRO_START, ANTHRO_END)
    st_anthro = _hadisst_basin_mean(SUBTROPICAL_NA, ANTHRO_START, ANTHRO_END)
    years = sp_anthro["year"].values.astype(float)
    contrast_modern_raw = sp_anthro.values - st_anthro.values
    base_mask = (years >= 1870) & (years <= 1879)
    contrast_modern_annual = (contrast_modern_raw
                                - contrast_modern_raw[base_mask].mean())

    raw_sen_per_century = cross_json["anthro"]["sen_degC_per_century"]
    print("=" * 70)
    print(" R2-BLOCKER 2: sigma-sensitivity sweep of symmetric-noise test")
    print(f" Raw modern Sen slope: {raw_sen_per_century:+.3f} degC/century")
    print("=" * 70)

    by_sigma = {}
    for sigma_yr in SIGMAS_YR:
        modern_sen_kyr = _modern_smoothed_sen(years, contrast_modern_annual,
                                                  sigma_yr)
        modern_sen_century = modern_sen_kyr / 10.0  # 1000/100
        palmod_smoothed = _smooth_palmod_grid(contrast_palmod_raw, sigma_yr)
        print(f"\n sigma = {sigma_yr:>5.0f} yr  modern Sen slope = "
              f"{modern_sen_century:+.3f} degC/century  "
              f"({100*modern_sen_century/raw_sen_per_century:5.1f}% of raw)")

        results_for_sigma = {}
        rng = np.random.default_rng(int(sigma_yr) + 17)
        for w_kyr in WINDOW_KYRS:
            slopes = sliding_sen_slopes(palmod_smoothed, t_kyr, w_kyr)
            if slopes.size < 5:
                continue
            mu = float(np.nanmean(slopes))
            sd = float(np.nanstd(slopes))
            z_point = abs((modern_sen_kyr - mu) / sd) if sd > 0 else float("inf")
            zs = np.empty(B)
            for b in range(B):
                idx = np.sort(rng.integers(0, slopes.size, size=slopes.size))
                mu_b = float(np.nanmean(slopes[idx]))
                sd_b = float(np.nanstd(slopes[idx]))
                zs[b] = abs((modern_sen_kyr - mu_b) / sd_b) if sd_b > 0 else np.nan
            z_med = float(np.nanmedian(zs))
            z_lo = float(np.nanpercentile(zs, 2.5))
            z_hi = float(np.nanpercentile(zs, 97.5))
            results_for_sigma[f"W_{int(w_kyr*1000)}yr"] = dict(
                window_kyr=w_kyr,
                z_median=z_med,
                z_ci95=[z_lo, z_hi],
                modern_sen_per_century=float(modern_sen_century),
            )
            tick = ">3" if z_med > 3.0 else "<=3"
            print(f"   W={w_kyr:.1f} kyr  |z|_med = {z_med:5.2f}  "
                  f"[{z_lo:.2f}, {z_hi:.2f}]  {tick}")
        by_sigma[f"sigma_{int(sigma_yr)}yr"] = dict(
            sigma_yr=sigma_yr,
            modern_sen_per_century=float(modern_sen_century),
            modern_sen_ratio_to_raw=float(modern_sen_century / raw_sen_per_century),
            by_window=results_for_sigma,
        )

    # Identify the sigma at which |z| crosses 3 for each window
    print()
    print(" Crossing-sigma summary (|z|_median = 3):")
    crossings = {}
    for w_kyr in WINDOW_KYRS:
        key_w = f"W_{int(w_kyr*1000)}yr"
        zs = [(s, by_sigma[f"sigma_{int(s)}yr"]["by_window"][key_w]["z_median"])
              for s in SIGMAS_YR
              if key_w in by_sigma[f"sigma_{int(s)}yr"]["by_window"]]
        below = [s for s, z in zs if z <= 3.0]
        above = [s for s, z in zs if z > 3.0]
        sigma_cross = None
        if above and below:
            # Largest sigma still giving |z| > 3
            sigma_cross = max(above) if max(above) < min(below) else None
        crossings[key_w] = dict(
            sigma_values=[s for s, _ in zs],
            z_medians=[z for _, z in zs],
            sigma_max_sustaining_z3=max(above) if above else None,
        )
        max_sigma = max(above) if above else None
        print(f"   {key_w}: |z|>3 sustained up to sigma = "
              f"{max_sigma if max_sigma is not None else 'never'} yr")

    summary = dict(
        sigma_sweep_yr=SIGMAS_YR,
        windows_kyr=WINDOW_KYRS,
        by_sigma=by_sigma,
        crossings=crossings,
        raw_modern_sen_per_century=raw_sen_per_century,
    )
    with open(CACHE_OUT / "sigma_sweep.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {CACHE_OUT / 'sigma_sweep.json'}")

    # Figure: |z| vs sigma per window
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    colors = ["C0", "C1", "C2"]
    for col, w_kyr in zip(colors, WINDOW_KYRS):
        key_w = f"W_{int(w_kyr*1000)}yr"
        xs = []
        ys = []
        los = []
        his = []
        for s in SIGMAS_YR:
            r = by_sigma[f"sigma_{int(s)}yr"]["by_window"].get(key_w)
            if r is None: continue
            xs.append(s)
            ys.append(r["z_median"])
            los.append(r["z_ci95"][0])
            his.append(r["z_ci95"][1])
        xs = np.asarray(xs); ys = np.asarray(ys)
        los = np.asarray(los); his = np.asarray(his)
        ax.plot(xs, ys, "o-", color=col, label=f"{int(w_kyr*1000)}-yr window",
                  linewidth=2)
        ax.fill_between(xs, los, his, color=col, alpha=0.2)
    ax.axhline(3.0, color="black", linestyle="--", linewidth=1.5,
                  label="|z| = 3 threshold")
    ax.set_xscale("log")
    ax.set_xlabel(r"Symmetric smoothing $\sigma$ (yr)")
    ax.set_ylabel(r"$|z|$ median (with 95\% bootstrap CI)")
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(MANUSCRIPT_FIGS / "fig_sigma_sweep.pdf")
    plt.close(fig)
    print(f"Wrote {MANUSCRIPT_FIGS / 'fig_sigma_sweep.pdf'}")


if __name__ == "__main__":
    main()
