"""Multi-proxy gap-fill unprecedented test.

Per-proxy |z| > 3 Sen-slope distribution test for each of the nine
paleoclimate reconstructions in the Caesar 2021 multi-proxy AMOC
supplementary compilation. Closes the PALMOD-to-HadISST 770-yr
resolution gap with high-resolution (annual to decadal) records.

Pre-registered decision rule (locked before computation):

  For each proxy:
    1. Compute sliding-window Sen slope distribution at window sizes
       W in {100, 200, 500} yr over the proxy's pre-1850 coverage.
    2. Compute the proxy's modern Sen slope as the linear-trend rate
       over the most recent W years where data is available.
    3. The proxy declares "modern is unprecedented" iff
       |z_modern| = |(modern_slope - paleo_mean_slope) / paleo_std| > 3
       for at least one W in {100, 200, 500} yr.

  Aggregate (multi-proxy) decision rule:
    The Cold Blob is unprecedented at centennial-decadal scales iff
    >= 5 of the 9 proxies independently satisfy |z| > 3.

The 9 proxies have heterogeneous units (degC, micrometers, permil,
percent, standardised anomalies). The test is run in each proxy's
native units, and z-scores are intrinsically unit-free. Direction
of the modern signal is also reported per proxy; we expect the
direction to align with weakening AMOC across all proxies whose
sign-convention is interpreted in the AMOC literature.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()


XLSX_FILE = (Path.home() / "Documents" / "NEW_Theory" / "data"
              / "external" / "caesar2021_multiproxy.xlsx")
OUT_DIR = CACHE_DIR / "multi_proxy"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

# Sheet name -> (display name, expected_direction_for_weakening_AMOC, units, value_col)
# Direction: negative => modern decline indicates weakening AMOC.
# value_col: 1 by default (column B after year).
PROXIES = [
    ("Thornalley et al. (2018), Tsub",      "Thornalley 2018 $T_{sub}$",       "−",  "°C",      1),
    ("Sheerwood et al. (2011)",              "Sherwood 2011 δ$^{15}$N",         "+",  "‰",       1),
    ("Thornalley et al. (2018) s-silt",      "Thornalley 2018 sortable silt",   "−",  "μm",      1),
    ("Spooner et al. (2020)",                "Spooner 2020 $T.$ quinqueloba",   "+",  "%",       1),
    ("Osmann et al. (2019)",                 "Osmann 2019 MAS productivity",    "−",  "std",     1),
    ("Caesar et al. (2018)",                 "Caesar 2018 AMOC index",          "−",  "K",       1),
    ("Cheng et al. (2017)",                  "Cheng 2017 OHC",                  "+",  "ZJ",      1),
    ("Thibedeau et al. (2018)",              "Thibodeau 2018",                  "−",  "‰",       1),
    ("Rahmstorf et al. (2015)",              "Rahmstorf 2015 AMOC index",       "−",  "°C",      1),
]


def _load_proxy(sheet: str, value_col: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Load (year_CE, values) from one sheet, dropping non-numeric rows."""
    wb = openpyxl.load_workbook(XLSX_FILE, data_only=True)
    ws = wb[sheet]
    yrs, vals = [], []
    # Skip up to 6 header rows
    for r in ws.iter_rows(min_row=2, values_only=True):
        if not r or r[0] is None: continue
        try:
            y = float(r[0]); v = float(r[value_col])
        except (TypeError, ValueError):
            continue
        if y < 0: continue
        yrs.append(y); vals.append(v)
    arr = np.array(sorted(zip(yrs, vals), key=lambda p: p[0]))
    return arr[:, 0], arr[:, 1]


def _sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Median pairwise slope (Sen estimator) in y per x."""
    if x.size < 2: return float("nan")
    slopes = []
    for i in range(len(x) - 1):
        slopes.append((y[i + 1:] - y[i]) / (x[i + 1:] - x[i]))
    return float(np.median(np.concatenate(slopes)))


def _sliding_sen_slopes(years: np.ndarray, vals: np.ndarray,
                          window_yr: int, step_yr: int = 25,
                          min_n: int = 5,
                          year_end_cutoff: float | None = None
                          ) -> np.ndarray:
    """Sliding-window Sen slopes over the proxy record.

    If year_end_cutoff is given, restrict the sliding-window scan to
    end at or before that year (e.g., pre-1850 distribution).
    Returns slopes per window position (one slope per step_yr step).
    """
    if year_end_cutoff is not None:
        m = years <= year_end_cutoff
        years = years[m]; vals = vals[m]
    if len(years) < min_n: return np.array([])
    yr_lo = years.min()
    yr_hi = years.max() - window_yr
    starts = np.arange(yr_lo, yr_hi + 1, step_yr)
    out = []
    for s in starts:
        m = (years >= s) & (years <= s + window_yr)
        if m.sum() < min_n: continue
        sl = _sen_slope(years[m], vals[m])
        if np.isfinite(sl): out.append(sl)
    return np.array(out)


def _modern_sen_slope(years: np.ndarray, vals: np.ndarray,
                       window_yr: int, modern_end: float = 2020,
                       min_n: int = 5) -> float:
    """Modern Sen slope over the most recent window_yr years up to
    modern_end."""
    m = (years >= modern_end - window_yr) & (years <= modern_end)
    if m.sum() < min_n: return float("nan")
    return _sen_slope(years[m], vals[m])


def run_one_proxy(sheet: str, display: str, direction: str,
                   units: str, value_col: int = 1,
                   pre_1850_cutoff: float = 1850) -> dict:
    """Full unprecedented test for one proxy."""
    try:
        yrs, vals = _load_proxy(sheet, value_col)
    except Exception as e:
        return {"display": display, "error": str(e)}
    out = {"display": display, "direction": direction, "units": units,
            "n_samples": int(len(yrs)),
            "year_min": float(yrs.min()), "year_max": float(yrs.max()),
            "windows": {}}
    pre_n = int((yrs <= pre_1850_cutoff).sum())
    out["n_pre_1850"] = pre_n
    if pre_n < 10:
        out["verdict_overall"] = "modern-only-no-test"
        return out

    for W in (100, 200, 500):
        # Pre-industrial distribution
        slopes = _sliding_sen_slopes(yrs, vals, W,
                                      year_end_cutoff=pre_1850_cutoff)
        if slopes.size < 5:
            out["windows"][f"W_{W}"] = {"verdict": "insufficient-windows"}
            continue
        modern = _modern_sen_slope(yrs, vals, W)
        if not np.isfinite(modern):
            out["windows"][f"W_{W}"] = {"verdict": "no-modern-window"}
            continue
        mu = float(slopes.mean()); sd = float(slopes.std())
        z = float((modern - mu) / sd) if sd > 0 else float("inf")
        verdict = "UNPRECEDENTED" if abs(z) > 3.0 else "within-envelope"
        # Direction check
        modern_signs_correct = (
            (direction == "−" and modern < 0) or
            (direction == "+" and modern > 0)
        )
        out["windows"][f"W_{W}"] = {
            "n_pre_windows": int(slopes.size),
            "paleo_mean_slope": mu,
            "paleo_std_slope": sd,
            "paleo_min_slope": float(slopes.min()),
            "paleo_max_slope": float(slopes.max()),
            "modern_slope": modern,
            "z_score": z,
            "verdict": verdict,
            "direction_correct": modern_signs_correct,
        }

    # Aggregate over windows: unprecedented at >=1 window?
    z_max = 0.0; any_unprec = False
    for w, info in out["windows"].items():
        if "z_score" in info and abs(info["z_score"]) > abs(z_max):
            z_max = info["z_score"]
        if info.get("verdict") == "UNPRECEDENTED":
            any_unprec = True
    out["verdict_overall"] = "UNPRECEDENTED" if any_unprec else "within-envelope"
    out["max_abs_z"] = float(abs(z_max))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading 9 proxies from {XLSX_FILE.name}")
    results = []
    for sheet, display, direction, units, vcol in PROXIES:
        res = run_one_proxy(sheet, display, direction, units, vcol)
        results.append(res)
        if "verdict_overall" in res:
            print(f"  {display:38} N={res['n_samples']:4} pre1850={res['n_pre_1850']:4}  "
                  f"max|z|={res.get('max_abs_z', 0):5.1f}  {res['verdict_overall']}")
        else:
            print(f"  {display}: {res.get('error', 'failed')}")

    # Aggregate
    testable = [r for r in results if r.get("verdict_overall") not in
                 (None, "modern-only-no-test")]
    unprec = [r for r in testable if r["verdict_overall"] == "UNPRECEDENTED"]
    print()
    print(f"Testable proxies (>=10 pre-1850 samples): {len(testable)} of 9")
    print(f"  Unprecedented (|z|>3 at >=1 window):   {len(unprec)}")
    pass_rule = len(unprec) >= 5
    print(f"  Aggregate verdict (>=5/9 rule): "
          f"{'UNPRECEDENTED' if pass_rule else 'not unprecedented'}")
    print(f"  Aggregate over testable subset alone "
          f"({len(unprec)}/{len(testable)}): "
          f"{'MAJORITY UNPRECEDENTED' if len(unprec) > len(testable)/2 else 'minority'}")

    with open(OUT_DIR / "multi_proxy.json", "w") as f:
        json.dump({
            "n_proxies": len(results),
            "n_testable": len(testable),
            "n_unprecedented": len(unprec),
            "aggregate_verdict": "UNPRECEDENTED" if pass_rule else "not unprecedented",
            "per_proxy": results,
        }, f, indent=2, default=float)
    print(f"\nWrote {OUT_DIR / 'multi_proxy.json'}")

    # ----- Figure: only testable proxies with at least one valid
    # sliding-window slope in a clean grid ----------------------------
    def _has_valid_window(r):
        for W in (100, 200, 500):
            info = r.get("windows", {}).get(f"W_{W}")
            if info and "z_score" in info:
                return True
        return False

    testable_results = [r for r in results
                         if r.get("verdict_overall") not in
                         (None, "modern-only-no-test")
                         and _has_valid_window(r)]
    excluded_names = [
        f"{r['display']} (modern-only)"
        for r in results
        if r.get("verdict_overall") == "modern-only-no-test"
    ] + [
        f"{r['display']} (range too short)"
        for r in results
        if r.get("verdict_overall") not in (None, "modern-only-no-test")
        and not _has_valid_window(r)
    ]

    n = len(testable_results)
    # Layout: enough panels to hold n testable + 1 inset of excluded.
    n_panels = n + 1
    ncol = 2 if n_panels <= 4 else 3
    nrow = (n_panels + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol,
                              figsize=(3.6 * ncol, 3.0 * nrow),
                              constrained_layout=True)
    flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    panel_letters = "abcdefghi"
    for slot, ax in enumerate(flat):
        if slot >= n:
            # First post-data slot: list excluded proxies
            if slot == n and excluded_names:
                ax.axis("off")
                lines = ["Excluded from test"
                          " (insufficient pre-1850 baseline):"]
                for name in excluded_names:
                    lines.append(f"  • {name}")
                ax.text(0.0, 0.92, "\n".join(lines),
                         transform=ax.transAxes,
                         ha="left", va="top", fontsize=8.5)
            else:
                ax.axis("off")
            continue
        letter = panel_letters[slot]
        proxy = testable_results[slot]
        ax.set_title(proxy["display"], fontsize=8.5, loc="center", pad=4)
        ax.text(-0.10, 1.06, letter, transform=ax.transAxes,
                 fontweight="bold", fontsize=10)
        # Show distribution at the largest W with valid data
        best_w = None
        for W in (500, 200, 100):
            info = proxy["windows"].get(f"W_{W}")
            if info and "z_score" in info:
                best_w = (W, info); break
        if best_w is None:
            ax.axis("off")
            continue
        W, info = best_w
        # Recompute slopes for plotting
        sheet_name = [s for s in PROXIES if s[1] == proxy["display"]][0][0]
        yrs, vals = _load_proxy(sheet_name)
        slopes = _sliding_sen_slopes(yrs, vals, W, year_end_cutoff=1850)
        ax.hist(slopes, bins=15, color="0.7", edgecolor="0.3", alpha=0.85)
        ax.axvline(info["modern_slope"], color="C3", lw=2.0)
        ax.axvline(info["paleo_mean_slope"], color="0.3",
                    lw=0.7, linestyle=":")
        ax.set_xlabel(f"slope ({proxy['units']}/yr), W={W} yr", fontsize=7)
        ax.tick_params(labelsize=6)
        z_txt = f"|z|={abs(info['z_score']):.1f}"
        verdict_short = "✓" if info["verdict"] == "UNPRECEDENTED" else "—"
        ax.text(0.96, 0.95, f"{z_txt}\n{verdict_short}",
                 transform=ax.transAxes, ha="right", va="top",
                 fontsize=7,
                 color="C3" if info["verdict"] == "UNPRECEDENTED" else "0.4")

    out_fig = MANUSCRIPT_FIGS / "fig_multi_proxy.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
