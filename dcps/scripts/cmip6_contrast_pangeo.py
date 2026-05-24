"""Fetch CMIP6 historical + ssp245 + ssp585 SST contrasts via Pangeo
cloud catalog and compute per-model MK + Sen on the meridional contrast
(subpolar NA minus subtropical NA, same windows used elsewhere).

This script reads CMIP6 tos (sea surface temperature) lazily from Google
Cloud Storage via intake-esm + zarr. Only the NA subset is materialised,
so total network use is small.

Pre-registered decision rule (locked):
    For each (model, scenario) tuple, compute the annual basin-mean SST
    in subpolar NA (50-65 N, 50W-10W) and subtropical NA (20-35 N,
    50W-10W), then the contrast (subpolar - subtropical). Run
    Mann-Kendall on each contrast time series over its experiment's
    time window. Report:
        historical: 1870-2014 (to overlap HadISST start year),
        ssp245:    2015-2100,
        ssp585:    2015-2100.
    Per-scenario ensemble statistics: median Sen slope and 16th-84th
    percentile across models.
"""

from __future__ import annotations

import json
import time

import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR


SUBPOLAR_NA  = dict(lon_min=-50, lon_max=-10, lat_min=50, lat_max=65)
SUBTROPICAL  = dict(lon_min=-50, lon_max=-10, lat_min=20, lat_max=35)

OUT_DIR = CACHE_DIR / "cmip6_contrast"

# A focused ensemble: well-known CMIP6 models with all three scenarios on
# Pangeo, modest grid sizes, and good documentation for the AMOC literature.
MODELS = [
    "ACCESS-CM2",
    "CanESM5",
    "CESM2",
    "CMCC-CM2-SR5",
    "CNRM-CM6-1",
    "IPSL-CM6A-LR",
    "MIROC6",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0",
    "NorESM2-MM",
    "UKESM1-0-LL",
]
EXPERIMENTS = ["historical", "ssp245", "ssp585"]

# Time windows for the MK trend test on each experiment.
EXPERIMENT_WINDOWS = {
    "historical": ("1870-01-01", "2014-12-31"),
    "ssp245":     ("2015-01-01", "2100-12-31"),
    "ssp585":     ("2015-01-01", "2100-12-31"),
}


def _basin_mean(ds: xr.Dataset, window: dict) -> xr.DataArray:
    """Cosine-lat area mean of `tos` over a (lat, lon) window. Returns
    annual mean as a 1-D DataArray on year."""
    tos = ds["tos"]
    # CMIP6 longitude can be 0..360 or -180..180. Normalise.
    lon = tos["lon"] if "lon" in tos.coords else tos["longitude"]
    lat = tos["lat"] if "lat" in tos.coords else tos["latitude"]

    # Find the actual lat/lon names.
    lat_name = next((n for n in ("lat", "latitude") if n in tos.coords), None)
    lon_name = next((n for n in ("lon", "longitude") if n in tos.coords), None)
    if lat_name is None or lon_name is None:
        # Some CMIP6 use 2-D nav_lat / nav_lon coords on a curvilinear grid.
        raise ValueError(f"can't find lat/lon coords; available: {list(tos.coords)}")

    # Use full 2-D coordinate arrays in case of curvilinear grid.
    lat2 = tos[lat_name]
    lon2 = tos[lon_name]
    # Map lon to [-180, 180].
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


def _mann_kendall(x: np.ndarray) -> tuple[float, float, float]:
    """Standard MK + Sen on a 1-D array (unit time step)."""
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    x = x[valid]
    n = x.size
    if n < 5:
        return 0.0, 1.0, 0.0
    s = 0
    for i in range(n - 1):
        s += np.sum(np.sign(x[i + 1:] - x[i]))
    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        return 0.0, 1.0, 0.0
    if s > 0: z = (s - 1) / np.sqrt(var_s)
    elif s < 0: z = (s + 1) / np.sqrt(var_s)
    else: z = 0.0
    from scipy.stats import norm
    p = 2.0 * (1 - norm.cdf(abs(z)))
    slopes = []
    for i in range(n - 1):
        slopes.append((x[i + 1:] - x[i]) /
                       np.arange(1, n - i, dtype=float))
    sen = float(np.median(np.concatenate(slopes)))
    return float(z), float(p), sen


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import intake

    print("Connecting to Pangeo CMIP6 catalog (Google Cloud)...")
    cat = intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    print(f"  catalog rows: {len(cat.df)}")

    results = {}
    for exp in EXPERIMENTS:
        print(f"\n==== {exp} ====")
        results[exp] = {}
        # Search query: monthly ocean SST, r1i1p1f1, gn (native) preferred.
        q = cat.search(
            source_id=MODELS, experiment_id=exp,
            variable_id="tos", table_id="Omon",
            member_id="r1i1p1f1",
        )
        # If r1i1p1f1 unavailable for some models (UKESM uses f2), try f2.
        missing = set(MODELS) - set(q.df["source_id"].unique())
        if missing:
            q2 = cat.search(
                source_id=list(missing), experiment_id=exp,
                variable_id="tos", table_id="Omon",
                member_id="r1i1p1f2",
            )
            q = q.search(activity_id=q.df["activity_id"].unique()) if len(q.df) else q
            # Merge by concatenating catalogues' dataframes manually:
            import pandas as pd
            merged_df = pd.concat([q.df, q2.df], ignore_index=True)
            q = type(q)(esmcat=q.esmcat, df=merged_df) if False else q
            # The simpler path: just iterate q.df and q2.df separately.
            dfs = [q.df, q2.df]
        else:
            dfs = [q.df]

        for df in dfs:
            for source_id in df["source_id"].unique():
                if source_id in results[exp]: continue
                t0 = time.time()
                # Pick first row matching this model (some models have
                # multiple grid_label or version variants).
                row_df = df[df["source_id"] == source_id]
                # Prefer 'gn' over 'gr' if available
                if "gn" in row_df["grid_label"].values:
                    row_df = row_df[row_df["grid_label"] == "gn"]
                row = row_df.iloc[0]
                zstore = row["zstore"]
                try:
                    ds = xr.open_zarr(zstore, consolidated=True, chunks={})
                    ts_sp = _basin_mean(ds, SUBPOLAR_NA).load()
                    ts_st = _basin_mean(ds, SUBTROPICAL).load()
                    yrs = ts_sp["year"].values
                    sp_v = ts_sp.values
                    st_v = ts_st.values
                    contrast = sp_v - st_v
                    # Subset to experiment time window.
                    win = EXPERIMENT_WINDOWS[exp]
                    wy0, wy1 = int(win[0][:4]), int(win[1][:4])
                    m = (yrs >= wy0) & (yrs <= wy1)
                    yr_w = yrs[m]
                    c_w = contrast[m]
                    # Anomaly relative to first decade of window.
                    base = c_w[:10].mean() if len(c_w) >= 10 else c_w[0]
                    c_anom = c_w - base
                    z, p, sen = _mann_kendall(c_anom)
                    sen_per_century = sen * 100  # sen per year * 100 = per century
                    sen_per_kyr = sen * 1000
                    results[exp][source_id] = {
                        "MK_z": float(z), "MK_p": float(p),
                        "sen_degC_per_year": float(sen),
                        "sen_degC_per_century": float(sen_per_century),
                        "sen_degC_per_kyr": float(sen_per_kyr),
                        "n_years": int(len(yr_w)),
                        "year_min": int(yr_w.min()),
                        "year_max": int(yr_w.max()),
                        "subpolar_first": float(sp_v[m][0]),
                        "subpolar_last": float(sp_v[m][-1]),
                        "subtropical_first": float(st_v[m][0]),
                        "subtropical_last": float(st_v[m][-1]),
                        # Save per-year contrast anomaly time series
                        # (anomaly relative to first decade of window).
                        "years": [int(y) for y in yr_w],
                        "contrast_anom_degC": [float(v) for v in c_anom],
                    }
                    print(f"  {source_id:<20}  z={z:+6.2f}  p={p:.4f}  "
                          f"Sen={sen_per_century:+.3f} °C/cent  "
                          f"({time.time()-t0:.1f}s)")
                except Exception as e:
                    print(f"  {source_id}: FAILED -- {type(e).__name__}: {e}")

    # ----- Aggregate ensemble statistics ---------------------------------
    print()
    print("=" * 70)
    print(" CMIP6 ensemble Sen slopes per scenario (°C / century)")
    print("=" * 70)
    print(f"{'scenario':<12}  {'n':>3}  {'p16':>7}  {'p50':>7}  {'p84':>7}")
    print("-" * 50)
    for exp in EXPERIMENTS:
        slopes = [r["sen_degC_per_century"]
                  for r in results[exp].values()]
        if not slopes: continue
        p16, p50, p84 = np.percentile(slopes, [16, 50, 84])
        print(f"{exp:<12}  {len(slopes):>3}  {p16:+.3f}  {p50:+.3f}  {p84:+.3f}")

    with open(OUT_DIR / "cmip6_contrast.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'cmip6_contrast.json'}")


if __name__ == "__main__":
    main()
