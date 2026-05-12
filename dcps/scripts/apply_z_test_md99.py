"""Apply the pre-registered |z|>3 Sen-slope test to the deep core
that the original multi_proxy_unprecedented.py skipped because it
only inspected the *first* core in each dual-core sheet:

  Thibodeau et al. 2018, GRL
    Core MD99-2220 -- Laurentian Slope, ~44 N, 55 W, ~3700 m depth
    Benthic delta-18O on Cibicidoides spp.
    708 -- 1962 CE, n = 179 samples
    n_pre1850 = 156 (passes the >=10 pre-1850 screening rule)
    n_post1850 = 23

The first core in the same sheet (CR02-23) is modern-only and
excluded.  But the second core, hidden after the blank divider
column in the spreadsheet, is a strong deep-NA paleo record with
solid pre-1850 sampling.

Modern interval limitation: the record ends in 1962, so the |z|
test compares the post-1850 portion (1850-1962, n=23) against the
pre-1850 sliding Sen-slope distribution.  This is the most direct
test the data support; we report the limitation honestly.

This script uses the same _sen_slope and _sliding_sen_slopes math
as multi_proxy_unprecedented.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openpyxl


REPO = Path("/home/bijanf/Documents/NEW_Theory")
XLSX = REPO / "data/external/caesar2021_multiproxy.xlsx"
OUT = REPO / "dcps/cache/depth_split/augmented_proxies.json"


def _sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    slopes = []
    for i in range(len(x) - 1):
        slopes.append((y[i + 1:] - y[i]) / (x[i + 1:] - x[i]))
    return float(np.median(np.concatenate(slopes)))


def _sliding_sen_slopes(years, vals, window_yr, step_yr=25, min_n=5,
                          year_end_cutoff=None):
    if year_end_cutoff is not None:
        m = years <= year_end_cutoff
        years = years[m]; vals = vals[m]
    if len(years) < min_n:
        return np.array([])
    yr_lo = years.min(); yr_hi = years.max() - window_yr
    starts = np.arange(yr_lo, yr_hi + 1, step_yr)
    out = []
    for s in starts:
        m = (years >= s) & (years <= s + window_yr)
        if m.sum() < min_n:
            continue
        sl = _sen_slope(years[m], vals[m])
        if np.isfinite(sl):
            out.append(sl)
    return np.array(out)


def _modern_sen_slope(years, vals, window_yr, modern_end, min_n=5):
    m = (years >= modern_end - window_yr) & (years <= modern_end)
    if m.sum() < min_n:
        return float("nan")
    return _sen_slope(years[m], vals[m])


def load_md99_2220():
    """Load (year, d18O) for the second core (MD99-2220) in the
    Thibodeau sheet. The sheet stacks two cores side by side:
    columns A-E for CR02-23, columns G-K for MD99-2220.  So we
    pull (col 7 = year, col 8 = d18O) using 1-indexed openpyxl.
    """
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["Thibedeau et al. (2018)"]
    yrs, vals = [], []
    for r in ws.iter_rows(min_row=2, values_only=True):
        # 0-indexed: r[6] = year.1, r[7] = d18O.1
        if r is None or len(r) < 8:
            continue
        y, v = r[6], r[7]
        try:
            y = float(y); v = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(y) or not np.isfinite(v):
            continue
        if y < 0:
            continue
        yrs.append(y); vals.append(v)
    arr = np.array(sorted(zip(yrs, vals), key=lambda p: p[0]))
    return arr[:, 0], arr[:, 1]


def run_z_test(years, vals, modern_end, label, pre_1850_cutoff=1850):
    out = dict(label=label,
                n_samples=int(len(years)),
                year_min=float(years.min()),
                year_max=float(years.max()),
                modern_end_used=float(modern_end),
                pre_1850_cutoff=pre_1850_cutoff,
                n_pre_1850=int((years <= pre_1850_cutoff).sum()),
                n_post_1850=int((years > pre_1850_cutoff).sum()),
                windows={})
    for W in (100, 200, 500):
        slopes = _sliding_sen_slopes(years, vals, W,
                                       year_end_cutoff=pre_1850_cutoff)
        if slopes.size < 5:
            out["windows"][f"W_{W}"] = {"verdict": "insufficient-windows",
                                          "n_pre_windows": int(slopes.size)}
            continue
        modern = _modern_sen_slope(years, vals, W, modern_end=modern_end)
        if not np.isfinite(modern):
            out["windows"][f"W_{W}"] = {"verdict": "no-modern-window"}
            continue
        mu = float(slopes.mean()); sd = float(slopes.std())
        z = float((modern - mu) / sd) if sd > 0 else float("inf")
        verdict = "UNPRECEDENTED" if abs(z) > 3.0 else "within-envelope"
        out["windows"][f"W_{W}"] = dict(
            n_pre_windows=int(slopes.size),
            paleo_mean_slope=mu, paleo_std_slope=sd,
            paleo_min_slope=float(slopes.min()),
            paleo_max_slope=float(slopes.max()),
            modern_slope=modern, z_score=z, verdict=verdict)
    z_max = 0.0; any_unprec = False
    for info in out["windows"].values():
        if "z_score" in info and abs(info["z_score"]) > abs(z_max):
            z_max = info["z_score"]
        if info.get("verdict") == "UNPRECEDENTED":
            any_unprec = True
    out["max_abs_z"] = float(abs(z_max))
    out["verdict_overall"] = "UNPRECEDENTED" if any_unprec else "within-envelope"
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" |z|>3 Sen-slope test: Thibodeau 2018 MD99-2220")
    print(" Laurentian Slope, ~44N 55W, ~3700 m depth")
    print(" Benthic delta-18O on Cibicidoides spp.")
    print("=" * 70)
    yrs, vals = load_md99_2220()
    print(f"  N samples: {len(yrs)}")
    print(f"  Year span: {yrs.min():.0f} -- {yrs.max():.0f}")
    print(f"  n pre-1850: {(yrs <= 1850).sum()}")
    print(f"  n post-1850: {(yrs > 1850).sum()}")
    # modern_end set to data's own latest year, not 2020
    modern_end = float(yrs.max())
    res = run_z_test(yrs, vals, modern_end=modern_end,
                      label="Thibodeau 2018 MD99-2220 d18O")
    print()
    for w, info in res["windows"].items():
        if "z_score" in info:
            print(f"  W = {w}:  modern slope = {info['modern_slope']:+.5f}"
                  f"   z = {info['z_score']:+.2f}   {info['verdict']}")
        else:
            print(f"  W = {w}:  {info['verdict']}")
    print(f"\n  Overall:  max |z| = {res['max_abs_z']:.2f}"
          f"   verdict = {res['verdict_overall']}")

    # Persist to augmented proxies json
    augmented = {
        "thibodeau_md99_2220": dict(
            source="Thibodeau et al. 2018 GRL",
            core="MD99-2220",
            lat=44.0, lon=-55.0, depth_m=3700,
            proxy_type="benthic delta-18O on Cibicidoides spp.",
            depth_class="deep",
            **res,
        )
    }
    with open(OUT, "w") as f:
        json.dump(augmented, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
