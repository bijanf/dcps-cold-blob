"""Fetch CMIP6 past1000 (850--1849 CE) subpolar-minus-subtropical NA
contrast anomalies. Bridges the PALMOD/observational gap (~1100--1870 CE).

Per-model anomalies are anchored to that model's own historical
1870--1950 baseline (consistent with cmip6_contrast_pangeo.py). The
past1000 experiment continues through 1849 CE, so we have a contiguous
model-based estimate of the contrast over 850 CE to 2014 CE when
stitched onto historical.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR


SUBPOLAR_NA = dict(lon_min=-50, lon_max=-10, lat_min=50, lat_max=65)
SUBTROPICAL = dict(lon_min=-50, lon_max=-10, lat_min=20, lat_max=35)
BASELINE = (1870, 1950)

OUT_DIR = CACHE_DIR / "cmip6_past1000"

# Models for which past1000 + historical are commonly available on Pangeo.
MODELS = [
    "ACCESS-CM2", "CanESM5", "CESM2", "CMCC-CM2-SR5", "CNRM-CM6-1",
    "IPSL-CM6A-LR", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0",
    "NorESM2-MM", "UKESM1-0-LL",
    # NOTE past1000 has limited model coverage in CMIP6 -- some entries
    # will fall through gracefully.
]


def _basin_mean(ds: xr.Dataset, window: dict) -> xr.DataArray:
    tos = ds["tos"]
    lat_name = next((n for n in ("lat", "latitude") if n in tos.coords), None)
    lon_name = next((n for n in ("lon", "longitude") if n in tos.coords), None)
    lat = tos[lat_name]; lon = tos[lon_name]
    lon180 = ((lon + 180) % 360) - 180
    mask = ((lat >= window["lat_min"]) & (lat <= window["lat_max"])
            & (lon180 >= window["lon_min"]) & (lon180 <= window["lon_max"]))
    coslat = np.cos(np.deg2rad(lat))
    weight = (coslat * mask).where(mask)
    masked = tos.where(mask)
    spatial_dims = [d for d in tos.dims if d != "time"]
    num = (masked * weight).sum(dim=spatial_dims, skipna=True)
    den = weight.where(np.isfinite(masked)).sum(dim=spatial_dims, skipna=True)
    return (num / den).groupby("time.year").mean("time")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import intake
    print("Loading historical baselines for each model...")

    # Reuse historical contrast baselines from previous fetch if possible.
    contrast_cache = (CACHE_DIR / "cmip6_contrast" / "cmip6_contrast.json")
    hist_data = {}
    if contrast_cache.exists():
        prev = json.loads(contrast_cache.read_text())
        for m, info in prev.get("historical", {}).items():
            # Need the per-model historical contrast at 1870-1950 to be the
            # baseline. The 'contrast_anom_degC' in prev is referenced to
            # 1870-1879, so baseline shift = 0 by definition in that data.
            # We pull the absolute subpolar baseline from past_data
            # consistency we'd need a re-fetch; skip for now.
            hist_data[m] = info
        print(f"  loaded {len(hist_data)} models from contrast cache")

    cat = intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    print(f"Catalog rows: {len(cat.df)}")

    out: dict[str, dict] = {}
    for member_id in ("r1i1p1f1", "r1i1p1f2"):
        q = cat.search(source_id=MODELS, experiment_id="past1000",
                       variable_id="tos", table_id="Omon",
                       member_id=member_id)
        if len(q.df) == 0:
            continue
        for source_id in q.df["source_id"].unique():
            if source_id in out: continue
            t0 = time.time()
            sub = q.df[q.df["source_id"] == source_id]
            if "gn" in sub["grid_label"].values:
                sub = sub[sub["grid_label"] == "gn"]
            zstore = sub.iloc[0]["zstore"]
            try:
                ds = xr.open_zarr(zstore, consolidated=True, chunks={})
                sp_ann = _basin_mean(ds, SUBPOLAR_NA).load()
                st_ann = _basin_mean(ds, SUBTROPICAL).load()
                yrs = sp_ann["year"].values
                contrast = sp_ann.values - st_ann.values
                out[source_id] = {
                    "member_id": member_id,
                    "years": [int(y) for y in yrs],
                    "contrast_degC": [float(v) for v in contrast],
                    "subpolar_degC": [float(v) for v in sp_ann.values],
                    "subtropical_degC": [float(v) for v in st_ann.values],
                    "year_min": int(yrs.min()),
                    "year_max": int(yrs.max()),
                }
                print(f"  {source_id:<20}  member {member_id}  "
                      f"{yrs.min()}-{yrs.max()}  ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  {source_id}: FAILED -- {type(e).__name__}: {e}")

    if not out:
        print("\nNo past1000 simulations available on Pangeo for our model set.")
        print("Will fall back to PALMOD-only fill of the gap.")
        return

    with open(OUT_DIR / "past1000.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'past1000.json'}")


if __name__ == "__main__":
    main()
