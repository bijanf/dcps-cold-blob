"""Detection-and-attribution: how often does internal variability (CMIP6
piControl) produce a 100-year subpolar-minus-subtropical NA SST contrast
trend as negative as the HadISST observed -0.50 deg C/century?

Pre-registered (commit-anchored):
  For each of ~7 CMIP6 models with piControl tos on Pangeo:
    - Compute annual subpolar NA (50-65 N, 50-10 W) and subtropical NA
      (20-35 N, 50-10 W) SST means via cosine-lat area weighting.
    - Take the contrast time series (subpolar - subtropical).
    - Slide a 100-year window over the full piControl record (no overlap
      cap; consecutive windows stepped by 10 years).
    - Compute Sen slope per window in degrees C per century.
  Pool all (model, window) slope values into a single null distribution.

  Pre-registered pass condition (locked):
    fraction(piControl_slopes <= -0.50 degC/century) < 0.01
      => observed rate is rarer than 1% of internal-variability
         100-yr trends; unprecedented claim survives a DnA test.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR
from dcps.nature_style import apply_nature_style
apply_nature_style()


OUT_DIR = CACHE_DIR / "picontrol_dna"

SUBPOLAR_NA  = dict(lon_min=-50, lon_max=-10, lat_min=50, lat_max=65)
SUBTROPICAL  = dict(lon_min=-50, lon_max=-10, lat_min=20, lat_max=35)

OBSERVED_RATE = -0.50  # deg C / century

MODELS = [
    "CanESM5", "CMCC-CM2-SR5", "MIROC6", "MPI-ESM1-2-HR",
    "CESM2", "ACCESS-CM2", "MRI-ESM2-0", "IPSL-CM6A-LR", "NorESM2-MM",
]
WINDOW_YEARS = 100
WINDOW_STEP_YEARS = 10


def _basin_mean_annual(ds, window):
    """Area-weighted basin mean as annual time series."""
    tos = ds["tos"]
    lat_name = next((n for n in ("lat", "latitude") if n in tos.coords), None)
    lon_name = next((n for n in ("lon", "longitude") if n in tos.coords), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"can't find lat/lon coords; available: {list(tos.coords)}")
    lat2 = tos[lat_name]
    lon2 = tos[lon_name]
    lon_180 = ((lon2 + 180) % 360) - 180
    mask = ((lat2 >= window["lat_min"]) & (lat2 <= window["lat_max"])
            & (lon_180 >= window["lon_min"]) & (lon_180 <= window["lon_max"]))
    coslat = np.cos(np.deg2rad(lat2))
    weight = (coslat * mask).where(mask)
    masked = tos.where(mask)
    num = (masked * weight).sum(dim=[d for d in tos.dims if d != "time"], skipna=True)
    den = weight.where(np.isfinite(masked)).sum(
        dim=[d for d in tos.dims if d != "time"], skipna=True)
    ts = num / den
    return ts.groupby("time.year").mean("time")


def _sen_slope_per_century(x):
    """Sen slope on an annual time series; return per-century rate."""
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    x = x[valid]
    n = x.size
    if n < 5:
        return np.nan
    slopes = []
    for i in range(n - 1):
        slopes.append((x[i + 1:] - x[i]) / np.arange(1, n - i, dtype=float))
    sen = float(np.median(np.concatenate(slopes)))
    return sen * 100.0


def sliding_sen_slopes(contrast, window_years=WINDOW_YEARS, step=WINDOW_STEP_YEARS):
    out = []
    n = len(contrast)
    if n < window_years:
        return out
    for start in range(0, n - window_years + 1, step):
        seg = contrast[start:start + window_years]
        out.append(_sen_slope_per_century(seg))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import intake

    print("=" * 70)
    print(" CMIP6 piControl detection-and-attribution against observed rate")
    print(f" Observed HadISST 1870-2014 Sen slope: {OBSERVED_RATE:+.2f} deg C / century")
    print("=" * 70)

    print("\nConnecting to Pangeo CMIP6 catalog (Google Cloud)...")
    cat = intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    print(f"  catalog rows: {len(cat.df)}")

    q = cat.search(
        source_id=MODELS, experiment_id="piControl",
        variable_id="tos", table_id="Omon",
    )
    print(f"  piControl tos rows: {len(q.df)}")

    per_model = {}
    all_slopes = []

    for source_id in MODELS:
        rows = q.df[q.df["source_id"] == source_id]
        if rows.empty:
            print(f"  {source_id}: NO piControl tos on Pangeo")
            continue
        # Prefer 'gn' grid; first member.
        if "gn" in rows["grid_label"].values:
            rows = rows[rows["grid_label"] == "gn"]
        rows = rows.sort_values("member_id").iloc[:1]
        zstore = rows.iloc[0]["zstore"]

        t0 = time.time()
        try:
            ds = xr.open_zarr(zstore, consolidated=True, chunks={})
            sp = _basin_mean_annual(ds, SUBPOLAR_NA).load()
            st = _basin_mean_annual(ds, SUBTROPICAL).load()
            yrs = sp["year"].values
            contrast = (sp - st).values
            slopes = sliding_sen_slopes(contrast)
            per_model[source_id] = {
                "n_years": int(len(yrs)),
                "n_windows": len(slopes),
                "slopes_per_century": [float(s) for s in slopes],
                "year_min": int(yrs.min()) if len(yrs) else None,
                "year_max": int(yrs.max()) if len(yrs) else None,
            }
            all_slopes.extend([s for s in slopes if np.isfinite(s)])
            print(f"  {source_id:<18}  n_years={len(yrs):>4}  "
                  f"n_windows={len(slopes):>3}  "
                  f"slope_p50={np.nanmedian(slopes):+.3f} degC/cent  "
                  f"({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  {source_id}: FAILED -- {type(e).__name__}: {e}")

    all_slopes = np.asarray(all_slopes, dtype=float)
    n_total = all_slopes.size
    if n_total == 0:
        print("\nNO piControl data fetched.  Cannot complete DnA test.")
        return

    n_exceed = int(np.sum(all_slopes <= OBSERVED_RATE))
    frac = n_exceed / n_total
    print()
    print("=" * 70)
    print(" Detection-and-attribution summary")
    print("=" * 70)
    print(f"  Total piControl 100-yr windows pooled: {n_total}")
    print(f"  Observed rate -0.50 degC/century exceeds (more negative than) "
          f"{n_exceed}/{n_total} = {100*frac:.3f}% of internal-variability windows")
    if frac < 0.01:
        verdict = "UNPRECEDENTED (frac < 1%)"
    elif frac < 0.05:
        verdict = "RARE (1% <= frac < 5%)"
    else:
        verdict = "NOT unprecedented (frac >= 5%)"
    print(f"  Pre-registered verdict: {verdict}")

    out = {
        "observed_rate_degC_per_century": OBSERVED_RATE,
        "n_total_windows": int(n_total),
        "n_exceed": n_exceed,
        "fraction_exceed": float(frac),
        "verdict": verdict,
        "per_model": per_model,
        "all_slopes": [float(s) for s in all_slopes],
        "slope_p_001": float(np.percentile(all_slopes, 0.1)),
        "slope_p_01": float(np.percentile(all_slopes, 1)),
        "slope_p_05": float(np.percentile(all_slopes, 5)),
        "slope_p_50": float(np.percentile(all_slopes, 50)),
        "slope_p_95": float(np.percentile(all_slopes, 95)),
    }
    with open(OUT_DIR / "dna_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'dna_results.json'}")

    # Histogram figure
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.hist(all_slopes, bins=40, color="grey", alpha=0.7, edgecolor="black")
    ax.axvline(OBSERVED_RATE, color="red", linewidth=2,
                label=f"HadISST observed ({OBSERVED_RATE:+.2f})")
    ax.set_xlabel("100-yr Sen slope, subpolar$-$subtropical SST contrast (deg C / century)")
    ax.set_ylabel("piControl windows")
    ax.legend(loc="upper left")
    ax.text(0.04, 0.86,
              f"n = {n_total} piControl windows from {len(per_model)} models\n"
              f"Fraction $\\leq$ observed: {100*frac:.3f}%",
              transform=ax.transAxes, fontsize=9,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    out_fig = Path("/home/bijanf/Documents/NEW_Theory/manuscript/figs/fig_picontrol_dna.pdf")
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
