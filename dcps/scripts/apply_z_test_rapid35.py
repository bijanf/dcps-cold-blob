"""Apply the pre-registered |z|>3 Sen-slope test to RAPiD-35-COM
(Moffa-Sanchez et al. 2015, QSR; PANGAEA 899382).

Core RAPiD-35-COM
  Location: 57.504 N, -48.722 W (Eirik Drift, south of Greenland)
  Water depth: 3484 m (DEEP)
  Proxy: sortable silt mean grain size (near-bottom current speed
         proxy for the Denmark Strait Overflow Water pathway)
  Time coverage: ~2000 BCE to 1914 CE  (228 samples)
  n_pre_1850: 223 (passes >=10 screening rule)
  n_post_1850: 5  (constrains the modern Sen-slope window)

The record ends in 1914 CE, so the modern interval cannot reach the
post-1970 AMOC slowdown era.  We apply the test honestly with
modern_end = 1914 CE and report the constraint in the manuscript.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path("/home/bijanf/Documents/NEW_Theory")
TAB = REPO / "data/external/moffa_sanchez/rapid35com.tab"
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


def load_rapid35():
    with open(TAB) as f:
        for i, l in enumerate(f):
            if l.strip().startswith("Depth sed"):
                header = i
                break
    df = pd.read_csv(TAB, sep="\t", skiprows=header)
    df["year_CE"] = 1950.0 - df["Age [ka BP]"] * 1000.0
    df = df.sort_values("year_CE").reset_index(drop=True)
    return df["year_CE"].values, df["SS avg [µm]"].values


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
    print(" |z|>3 Sen-slope test: Moffa-Sanchez 2015 RAPiD-35-COM")
    print(" Eirik Drift, 57.5N 48.7W, 3484 m (DEEP)")
    print(" Sortable silt: bottom-current speed of DSOW")
    print("=" * 70)
    yrs, vals = load_rapid35()
    print(f"  N samples: {len(yrs)}")
    print(f"  Year span: {yrs.min():.0f} -- {yrs.max():.0f}")
    print(f"  n_pre_1850: {(yrs <= 1850).sum()}")
    print(f"  n_post_1850: {(yrs > 1850).sum()}")
    modern_end = float(yrs.max())
    print(f"  Modern interval anchored at {modern_end:.0f} CE "
          f"(record ends pre-RAPID era)")
    res = run_z_test(yrs, vals, modern_end=modern_end,
                      label="Moffa-Sanchez 2015 RAPiD-35-COM sortable silt")
    print()
    for w, info in res["windows"].items():
        if "z_score" in info:
            print(f"  W = {w}:  modern slope = {info['modern_slope']:+.5f}"
                  f"   z = {info['z_score']:+.2f}   {info['verdict']}")
        else:
            print(f"  W = {w}:  {info['verdict']}")
    print(f"\n  Overall:  max |z| = {res['max_abs_z']:.2f}"
          f"   verdict = {res['verdict_overall']}")

    # Merge into augmented_proxies.json
    if OUT.exists():
        augmented = json.loads(OUT.read_text())
    else:
        augmented = {}
    augmented["moffa_sanchez_rapid_35_com"] = dict(
        source="Moffa-Sanchez et al. 2015 QSR; PANGAEA 899382",
        core="RAPiD-35-COM",
        lat=57.504, lon=-48.722, depth_m=3484,
        proxy_type="sortable silt mean grain size (DSOW current speed)",
        depth_class="deep",
        **res,
    )
    with open(OUT, "w") as f:
        json.dump(augmented, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
