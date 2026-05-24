"""BLOCKER 1 (peer review A3, A4): symmetric noise treatment for
PALMOD vs HadISST unprecedented test.

Reviewer concern: the published |z| scores compare a bioturbated,
calibration-smoothed PALMOD Holocene null to a noise-preserved
HadISST modern observation. The high-frequency variance suppressed
in PALMOD is not suppressed in HadISST, mechanically inflating |z|.

Symmetric remedy:
  1. Apply the same sigma=200 yr Gaussian low-pass to BOTH:
     - PALMOD basin-mean contrast on its 100-yr grid (sigma = 2 grid
       steps = 200 yr in physical time).
     - HadISST annual contrast on its 1-yr grid (sigma = 200 samples
       = 200 yr in physical time).
  2. Re-compute the Holocene Sen-slope distribution and the modern
     Sen slope on the matched-smooth records.
  3. Re-tabulate |z| at 500, 1000, 2000-yr windows.
  4. Drop the 5000-yr window (reviewer A4: only ~2-3 effectively
     independent 5000-yr windows in 11.65 kyr).

  Honest expectation: |z| will drop substantially.  At 500-yr the
  median bootstrap |z| (2.94 raw) will likely fall below 3; the
  longer-window claims may survive.  We accept the truth either way.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()


CACHE_OUT = CACHE_DIR / "cold_blob_unprecedented"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

GRID_KYR = 0.1
# Two symmetric smoothing scales reported.  The literal reviewer
# request is sigma=200 yr (matches the forward-proxy bootstrap kernel
# applied to PALMOD).  At a 154-yr HadISST record this kernel extends
# 3*sigma = 600 yr beyond the data and effectively annihilates the
# modern signal.  The band-matched alternative sigma=30 yr suppresses
# the sub-decadal-to-multi-decadal variance that PALMOD bioturbation
# would also suppress while preserving the multi-decadal trend.  Both
# results are reported.
SIGMA_YR_LITERAL = 200.0
SIGMA_YR_BAND_MATCHED = 30.0
WINDOW_KYRS = [0.5, 1.0, 2.0]  # 5000-yr dropped per reviewer A4
B = 1000


def sen_slope(x_index, y):
    if y.size < 2:
        return float("nan")
    slopes = []
    for i in range(len(y) - 1):
        slopes.append((y[i + 1:] - y[i]) / (x_index[i + 1:] - x_index[i]))
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


def main():
    CACHE_OUT.mkdir(parents=True, exist_ok=True)
    cross = np.load(CACHE_DIR / "cross_era" / "cross_era.npz")
    cross_json = json.loads(
        (CACHE_DIR / "cross_era" / "cross_era.json").read_text())

    # PALMOD basin-mean contrast on the 100-yr grid
    t_kyr = cross["paleo_kyr"]
    sp = cross["paleo_subpolar"].astype(float)
    st = cross["paleo_subtropical"].astype(float)
    contrast_palmod_raw = sp - st

    # Helper: smooth PALMOD on its 100-yr grid with given sigma in years.
    def _smooth_palmod(sigma_yr):
        out = contrast_palmod_raw.copy()
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

    # Modern HadISST contrast: load directly from HadISST monthly file
    # and compute annual subpolar-subtropical contrast anomaly.
    import sys
    sys.path.insert(0, str(PKG_ROOT / "scripts"))
    from cross_era_contrast import (
        _hadisst_basin_mean, SUBPOLAR_NA, SUBTROPICAL_NA,
        ANTHRO_START, ANTHRO_END,
    )
    sp_anthro = _hadisst_basin_mean(SUBPOLAR_NA, ANTHRO_START, ANTHRO_END)
    st_anthro = _hadisst_basin_mean(SUBTROPICAL_NA, ANTHRO_START, ANTHRO_END)
    years = sp_anthro["year"].values.astype(float)
    contrast_modern_raw = (sp_anthro.values - st_anthro.values)
    # Anomaly relative to 1870-1879 (matches cross_era_contrast convention)
    base_mask = (years >= 1870) & (years <= 1879)
    contrast_modern_annual = contrast_modern_raw - contrast_modern_raw[base_mask].mean()
    print(f"  HadISST annual contrast: n = {len(years)}  "
          f"first/last anomaly = {contrast_modern_annual[0]:+.2f} / "
          f"{contrast_modern_annual[-1]:+.2f} degC")

    # Helper: smooth HadISST annual contrast and compute Sen slope on
    # the central kernel-supported window.
    def _modern_smoothed_sen(sigma_yr):
        smoothed = gaussian_filter1d(contrast_modern_annual, sigma=sigma_yr)
        edge = int(np.ceil(sigma_yr))
        if smoothed.size > 2 * edge + 5:
            ctr = slice(edge, smoothed.size - edge)
        else:
            ctr = slice(None)
        m_yr = sen_slope(years[ctr].astype(float), smoothed[ctr])
        return m_yr * 1000.0, m_yr * 100.0  # per_kyr, per_century

    raw_modern_sen_per_kyr = cross_json["anthro"]["sen_degC_per_kyr"]
    raw_modern_sen_per_century = cross_json["anthro"]["sen_degC_per_century"]

    sen_literal_kyr, sen_literal_century = _modern_smoothed_sen(SIGMA_YR_LITERAL)
    sen_band_kyr, sen_band_century = _modern_smoothed_sen(SIGMA_YR_BAND_MATCHED)

    print(f"Raw HadISST Sen slope:                  "
          f"{raw_modern_sen_per_century:+.3f} degC/century  "
          f"({raw_modern_sen_per_kyr:+.2f} degC/kyr)")
    print(f"sigma={SIGMA_YR_LITERAL:.0f} yr smoothed (literal, reviewer A3): "
          f"{sen_literal_century:+.3f} degC/century  "
          f"({sen_literal_kyr:+.2f} degC/kyr)")
    print(f"sigma={SIGMA_YR_BAND_MATCHED:.0f} yr smoothed (band-matched):   "
          f"{sen_band_century:+.3f} degC/century  "
          f"({sen_band_kyr:+.2f} degC/kyr)")
    print()

    # Holocene null: sliding Sen slopes on the smoothed PALMOD, for
    # each of the two smoothing scales.
    def _run_symmetric(sigma_yr, modern_sen_per_kyr_smoothed, label):
        print("=" * 70)
        print(f" Symmetric unprecedented test (sigma={sigma_yr:.0f} yr, {label})")
        print("=" * 70)
        contrast_palmod = _smooth_palmod(sigma_yr)
        out = {}
        rng = np.random.default_rng(42)
        for w_kyr in WINDOW_KYRS:
            slopes = sliding_sen_slopes(contrast_palmod, t_kyr, w_kyr)
            if slopes.size < 5:
                continue
            mu = float(np.nanmean(slopes))
            sd = float(np.nanstd(slopes))
            z_point = abs((modern_sen_per_kyr_smoothed - mu) / sd) if sd > 0 else float("inf")
            zs = np.empty(B)
            for b in range(B):
                idx = np.sort(rng.integers(0, slopes.size, size=slopes.size))
                mu_b = float(np.nanmean(slopes[idx]))
                sd_b = float(np.nanstd(slopes[idx]))
                zs[b] = abs((modern_sen_per_kyr_smoothed - mu_b) / sd_b) if sd_b > 0 else np.nan
            z_med = float(np.nanmedian(zs))
            z_lo = float(np.nanpercentile(zs, 2.5))
            z_hi = float(np.nanpercentile(zs, 97.5))
            frac_above = float((zs > 3.0).mean())
            out[f"W_{int(w_kyr*1000)}yr"] = {
                "window_kyr": w_kyr,
                "n_holocene_windows": int(slopes.size),
                "holocene_mean_slope_per_kyr": mu,
                "holocene_std_slope_per_kyr": sd,
                "modern_smoothed_sen_per_kyr": modern_sen_per_kyr_smoothed,
                "z_point": float(z_point),
                "z_median_boot": z_med,
                "z_ci95": [z_lo, z_hi],
                "frac_boot_above_3": frac_above,
            }
            verdict = "above |z|=3" if z_med > 3.0 else "below |z|=3"
            print(f"  W = {w_kyr:.2f} kyr  n={slopes.size:>3}  "
                  f"|z|_med = {z_med:5.2f}  [{z_lo:.2f}, {z_hi:.2f}]  "
                  f"{100*frac_above:5.1f}% > 3   -> {verdict}")
        return out

    literal = _run_symmetric(SIGMA_YR_LITERAL, sen_literal_kyr,
                                "literal A3: kernel longer than HadISST -> null")
    print()
    band = _run_symmetric(SIGMA_YR_BAND_MATCHED, sen_band_kyr,
                              "band-matched: suppresses sub-decadal noise only")

    summary = {
        "method": "symmetric Gaussian smoothing of both PALMOD and "
                  "HadISST at two scales: literal sigma=200 yr "
                  "(reviewer A3) and band-matched sigma=30 yr",
        "raw_modern_sen_per_century": raw_modern_sen_per_century,
        "raw_modern_sen_per_kyr": raw_modern_sen_per_kyr,
        "sigma_literal_yr": SIGMA_YR_LITERAL,
        "sigma_band_matched_yr": SIGMA_YR_BAND_MATCHED,
        "modern_literal_smoothed_sen_per_century": sen_literal_century,
        "modern_band_matched_sen_per_century": sen_band_century,
        "by_window_literal": literal,
        "by_window_band_matched": band,
        "windows_excluded": ["5000 yr (insufficient independent windows; reviewer A4)"],
        "note": "Literal sigma=200 yr extends 3*sigma=600 yr beyond the "
                "154-year HadISST record and annihilates the modern trend; "
                "this is itself a finding (the proxy's effective Nyquist "
                "is too coarse to admit a symmetric test). The "
                "band-matched sigma=30 yr filter suppresses sub-decadal "
                "noise that the proxy network cannot resolve while "
                "preserving the multi-decadal trend that it can.",
    }
    with open(CACHE_OUT / "symmetric_z_scores.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {CACHE_OUT / 'symmetric_z_scores.json'}")

    # Comparison figure: raw vs both symmetric variants at each window
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    windows = WINDOW_KYRS
    x = np.arange(len(windows))
    width = 0.28

    # Raw (from bootstrap_results.json)
    raw_path = CACHE_OUT / "bootstrap_results.json"
    if raw_path.exists():
        raw_data = json.loads(raw_path.read_text())["by_window"]
        raw_med = [raw_data[f"W_{int(w*1000)}yr"]["block_bootstrap"]["z_median"]
                   for w in windows]
    else:
        raw_med = [np.nan] * len(windows)

    lit_med = [literal.get(f"W_{int(w*1000)}yr", {}).get("z_median_boot", np.nan)
               for w in windows]
    bnd_med = [band.get(f"W_{int(w*1000)}yr", {}).get("z_median_boot", np.nan)
               for w in windows]
    bnd_lo = [band.get(f"W_{int(w*1000)}yr", {}).get("z_ci95", [np.nan, np.nan])[0]
              for w in windows]
    bnd_hi = [band.get(f"W_{int(w*1000)}yr", {}).get("z_ci95", [np.nan, np.nan])[1]
              for w in windows]

    ax.bar(x - width, raw_med, width, color="lightgray", edgecolor="black",
              label="published (asymmetric noise)")
    ax.bar(x, lit_med, width, color="crimson", edgecolor="black", alpha=0.7,
              label=r"symmetric $\sigma=200$ yr (literal)")
    ax.bar(x + width, bnd_med, width, color="steelblue", edgecolor="black",
              yerr=[np.asarray(bnd_med) - np.asarray(bnd_lo),
                    np.asarray(bnd_hi) - np.asarray(bnd_med)],
              capsize=4,
              label=r"symmetric $\sigma=30$ yr (band-matched)")
    ax.axhline(3.0, color="black", linestyle="--", linewidth=1.5,
                  label="|z| = 3 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(w*1000)}-yr" for w in windows])
    ax.set_ylabel(r"$|z|$ median (with 95% bootstrap CI)")
    ax.set_xlabel("Sliding-window size")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(MANUSCRIPT_FIGS / "fig_symmetric_z.pdf")
    plt.close(fig)
    print(f"Wrote {MANUSCRIPT_FIGS / 'fig_symmetric_z.pdf'}")


if __name__ == "__main__":
    main()
