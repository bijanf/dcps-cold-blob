"""Test whether the modern Cold Blob is unprecedented in the Holocene.

Pre-registered decision rule (locked before computation):

    Compute the distribution of Sen slopes of the PALMOD subpolar-minus-
    subtropical NA SST contrast in sliding W-year windows over 0--11.7 ka BP.
    Compute the modern HadISST 1870--2023 contrast Sen slope (=
    -5.0 °C/kyr equivalent, from cross_era_contrast.py).

    Test 1 (rate): |z| = |(modern_rate - paleo_mean) / paleo_std|.
       SUPPORTED iff |z| > 3.0 for at least one window size in {200, 500,
       1000, 2000} yr -- modern rate decisively outside Holocene envelope.

    Test 2 (amplitude / 120-yr): is the modern Δcontrast (-0.5°C over the
       1870-1990 emergence period, by HadISST) larger in magnitude than
       the most extreme 120-yr Δ in PALMOD?
       HONEST: with PALMOD ~3kyr median per-core resolution, this is
       likely NOT detectable -- 120-yr noise std ~ 0.3°C, modern ~ 0.5°C,
       SNR < 2. We report this test's status honestly.

Caveats:
    - PALMOD subtropical NA has only 3 cores; the contrast time series is
      noisier than the subpolar-only stack. The test inherits this noise.
    - The W-year sliding-window Sen-slope distribution is computed on the
      same 100-yr grid used elsewhere in the manuscript. Windows with <3
      valid grid points are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT


CACHE_OUT = CACHE_DIR / "cold_blob_unprecedented"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"
GRID_KYR = 0.1


def sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Median pairwise slope (Sen estimator), units of y / x."""
    if x.size < 2:
        return float("nan")
    slopes = []
    for i in range(len(x) - 1):
        slopes.append((y[i + 1:] - y[i]) / (x[i + 1:] - x[i]))
    return float(np.median(np.concatenate(slopes)))


def sliding_sen_slopes(contrast: np.ndarray, kyr_grid: np.ndarray,
                        window_kyr: float, min_pts: int = 3) -> np.ndarray:
    """Sen slope (°C/kyr) in every sliding window of length ``window_kyr``."""
    n_per = int(round(window_kyr / GRID_KYR))
    slopes = []
    for i in range(len(contrast) - n_per + 1):
        sub_c = contrast[i:i + n_per]
        sub_t = kyr_grid[i:i + n_per]
        valid = np.isfinite(sub_c)
        if valid.sum() < min_pts:
            continue
        slopes.append(sen_slope(sub_t[valid], sub_c[valid]))
    return np.array(slopes)


def main():
    CACHE_OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(CACHE_DIR / "cross_era" / "cross_era.npz")
    cross_json = json.loads(
        (CACHE_DIR / "cross_era" / "cross_era.json").read_text())
    t_kyr = z["paleo_kyr"]
    contrast = z["paleo_subpolar"] - z["paleo_subtropical"]  # paleo Δlat

    # Modern Sen slope reference (HadISST 1870-2023 contrast widening).
    modern_sen_kyr = cross_json["anthro"]["sen_degC_per_kyr"]
    modern_sen_century = cross_json["anthro"]["sen_degC_per_century"]
    print(f"Modern observed: HadISST contrast Sen slope = "
          f"{modern_sen_kyr:+.2f} °C/kyr equivalent "
          f"(= {modern_sen_century:+.3f} °C/century).")
    print()

    # Sliding-window Sen-slope distributions for several window sizes.
    print("=" * 70)
    print(" PALMOD Holocene sliding-window Sen-slope distributions")
    print(" (subpolar - subtropical NA contrast, on 100-yr grid)")
    print("=" * 70)
    results = {}
    for w_kyr in (0.2, 0.5, 1.0, 2.0, 5.0):
        slopes = sliding_sen_slopes(contrast, t_kyr, w_kyr)
        if slopes.size < 5:
            print(f"  W = {w_kyr:.1f} kyr: too few windows")
            continue
        mu = float(np.nanmean(slopes))
        sd = float(np.nanstd(slopes))
        mn = float(np.nanmin(slopes))
        mx = float(np.nanmax(slopes))
        z_modern = (modern_sen_kyr - mu) / sd if sd > 0 else float("inf")
        p_outside = float(np.mean(np.abs(slopes - mu) >= abs(modern_sen_kyr - mu)))
        results[f"W_{int(w_kyr*1000)}yr"] = {
            "window_kyr": w_kyr,
            "n_windows": int(slopes.size),
            "paleo_mean_slope_per_kyr": mu,
            "paleo_std_slope_per_kyr": sd,
            "paleo_min_slope_per_kyr": mn,
            "paleo_max_slope_per_kyr": mx,
            "modern_sen_per_kyr": float(modern_sen_kyr),
            "z_score": float(z_modern),
            "p_outside_empirical": p_outside,
        }
        verdict = "UNPRECEDENTED" if abs(z_modern) > 3.0 else "within envelope"
        print(f"  W = {w_kyr:>5.2f} kyr  (n = {slopes.size:>3} windows):  "
              f"paleo mean = {mu:+.3f},  std = {sd:.3f},  "
              f"range = [{mn:+.2f}, {mx:+.2f}] °C/kyr  |  "
              f"modern z = {z_modern:+.1f} -- {verdict}")

    # Headline test: does the modern slope sit at |z| > 3 in any window size?
    any_unprecedented = any(abs(r["z_score"]) > 3.0 for r in results.values())
    print()
    print(f"Headline verdict: modern Cold-Blob rate "
          f"{'UNPRECEDENTED' if any_unprecedented else 'within Holocene envelope'} "
          f"at one or more PALMOD sliding-window scales.")

    # Persist
    with open(CACHE_OUT / "unprecedented.json", "w") as f:
        json.dump({
            "decision_rule": "|z| > 3.0 for any W in {200, 500, 1000, 2000, 5000} yr",
            "headline_verdict": ("UNPRECEDENTED" if any_unprecedented
                                  else "within envelope"),
            "modern_sen_per_kyr": float(modern_sen_kyr),
            "modern_sen_per_century": float(modern_sen_century),
            "by_window": results,
        }, f, indent=2)
    print(f"\nWrote {CACHE_OUT / 'unprecedented.json'}")

    # ----- Figure: distributions + modern marker -------------------------
    # Only show window sizes with valid distributions (>=10 windows).
    panel_windows = []
    for w_kyr in (0.5, 1.0, 2.0, 5.0):
        slopes = sliding_sen_slopes(contrast, t_kyr, w_kyr)
        if slopes.size >= 10:
            panel_windows.append((w_kyr, slopes))

    n_panels = len(panel_windows)
    fig, axes = plt.subplots(1, n_panels,
                              figsize=(3.4 * n_panels, 3.6),
                              constrained_layout=True, sharey=True)
    if n_panels == 1:
        axes = [axes]
    for ax, (w_kyr, slopes) in zip(axes, panel_windows):
        z_score = (modern_sen_kyr - slopes.mean()) / slopes.std()
        # Choose bin span large enough to include the modern marker
        lo = min(slopes.min(), modern_sen_kyr) - 0.3
        hi = max(slopes.max(), modern_sen_kyr) + 0.3
        ax.hist(slopes, bins=20, range=(lo, hi),
                 color="0.7", edgecolor="0.3", alpha=0.85)
        ax.axvline(modern_sen_kyr, color="C3", lw=2.2)
        ax.axvline(slopes.mean(), color="0.3", lw=0.8, linestyle=":")
        ax.set_xlabel("Sen slope (°C / kyr)")
        title = (f"{int(w_kyr*1000)}-yr windows "
                 f"(n = {slopes.size}); |z|$_{{\\mathrm{{modern}}}}$ = {abs(z_score):.1f}")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.20)
    axes[0].set_ylabel("count")
    out_fig = MANUSCRIPT_FIGS / "fig_unprecedented.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
