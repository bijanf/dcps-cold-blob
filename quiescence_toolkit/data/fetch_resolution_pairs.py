"""Fetch CMIP6 resolution-variant pairs for the Quiescence Index
resolution-dependence test (B3 / Prediction P3).

HighResMIP itself is not on Pangeo, but standard-CMIP6 model
families that ship in HR + LR variants are.  Five usable pairs:

  CESM2 (HR, ~1 deg)        vs  CESM2-FV2 (LR, ~2 deg)
  HadGEM3-GC31-MM (HR)      vs  HadGEM3-GC31-LL (LR)
  NorESM2-MM (HR)           vs  NorESM2-LM (LR)
  CNRM-CM6-1-HR (HR)        vs  CNRM-CM6-1 (LR)
  MPI-ESM1-2-HR (HR)        vs  MPI-ESM1-2-LR (LR)

NA basin subset, annual mean, persist as netcdf for the Q-index
script to consume.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import xarray as xr


OUT_DIR = Path("/home/bijanf/Documents/NEW_Theory/dcps/cache/highresmip")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("CESM2", "CESM2-FV2"),
    ("HadGEM3-GC31-MM", "HadGEM3-GC31-LL"),
    ("NorESM2-MM", "NorESM2-LM"),
    ("CNRM-CM6-1-HR", "CNRM-CM6-1"),
    ("MPI-ESM1-2-HR", "MPI-ESM1-2-LR"),
]
NA_LAT = (0, 75)
NA_LON = (-80, 0)
YEARS = (1950, 2014)


def fetch_one(cat, source_id):
    """Try several member_id values to find one that works."""
    for member in ("r1i1p1f1", "r1i1p1f2", "r1i1p1f3"):
        q = cat.search(source_id=source_id, experiment_id="historical",
                          variable_id="tos", table_id="Omon",
                          member_id=member)
        if len(q.df) > 0:
            return q.df.iloc[0]
    return None


def subset_and_persist(row, source_id, group):
    zstore = row["zstore"]
    t0 = time.time()
    try:
        ds = xr.open_zarr(zstore, consolidated=True, chunks={})
        # HadGEM3 uses 360-day calendar where Dec 31 doesn't exist;
        # use Dec 30 as the inclusive slice end to be safe.
        ds = ds.sel(time=slice(f"{YEARS[0]}-01-01",
                                  f"{YEARS[1]}-12-30"))
        # Find lat/lon coords (regular or curvilinear)
        coords = list(ds["tos"].coords)
        if "lat" in coords and "lon" in coords:
            lat_name, lon_name = "lat", "lon"
        elif "latitude" in coords and "longitude" in coords:
            lat_name, lon_name = "latitude", "longitude"
        elif "nav_lat" in coords and "nav_lon" in coords:
            lat_name, lon_name = "nav_lat", "nav_lon"
        else:
            print(f"  {source_id}: cannot find lat/lon (coords={coords})")
            return None
        lon_180 = ((ds[lon_name] + 180) % 360) - 180
        mask = ((ds[lat_name] >= NA_LAT[0])
                & (ds[lat_name] <= NA_LAT[1])
                & (lon_180 >= NA_LON[0])
                & (lon_180 <= NA_LON[1]))
        tos_na = ds["tos"].where(mask)
        tos_annual = tos_na.groupby("time.year").mean("time")
        n_yr = tos_annual.sizes["year"]
        out_path = OUT_DIR / f"{source_id}_{group}_tos_na.nc"
        tos_annual.to_netcdf(out_path)
        return dict(file=str(out_path), n_years=int(n_yr),
                     grid_label=row["grid_label"])
    except Exception as e:
        print(f"  {source_id}: {type(e).__name__}: {e}")
        return None


def main():
    import intake
    cat = intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    print(f"Pangeo catalog: {len(cat.df)} rows")

    inventory = {"high_res": {}, "low_res": {}}
    for hi, lo in PAIRS:
        for source_id, group in [(hi, "high_res"), (lo, "low_res")]:
            cache_file = OUT_DIR / f"{source_id}_{group}_tos_na.nc"
            if cache_file.exists():
                print(f"  {source_id} ({group}): already cached")
                inventory[group][source_id] = dict(
                    file=str(cache_file), n_years="(cached)", grid_label="?")
                continue
            row = fetch_one(cat, source_id)
            if row is None:
                print(f"  {source_id} ({group}): no row found in catalog")
                continue
            print(f"  {source_id} ({group}): fetching ...", end=" ", flush=True)
            t0 = time.time()
            result = subset_and_persist(row, source_id, group)
            if result is not None:
                inventory[group][source_id] = result
                print(f"{result['n_years']} yrs in {time.time()-t0:.0f}s")

    with open(OUT_DIR / "resolution_pairs_inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'resolution_pairs_inventory.json'}")
    print(f"  high_res:  {len(inventory['high_res'])} models")
    print(f"  low_res:   {len(inventory['low_res'])} models")


if __name__ == "__main__":
    main()
