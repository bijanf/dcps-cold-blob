"""Fetch CMIP6 HighResMIP `tos` (sea-surface temperature) via Pangeo
for the resolution-dependence test of the Quiescence Signature.

High-resolution group (nominal <= 0.25 deg ocean):
  - EC-Earth3P-HR
  - HadGEM3-GC31-HM
  - CNRM-CM6-1-HR

Low-resolution group (nominal >= 1 deg ocean):
  - CanESM5
  - CNRM-CM6-1
  - IPSL-CM6A-LR

We pull only the historical 1950-2014 window for each model, basin-
subset to the North Atlantic, and persist to dcps/cache/highresmip/.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import xarray as xr


OUT_DIR = Path("/home/bijanf/Documents/NEW_Theory/dcps/cache/highresmip")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIGH_RES = ["EC-Earth3P-HR", "HadGEM3-GC31-HM", "CNRM-CM6-1-HR"]
LOW_RES = ["CanESM5", "CNRM-CM6-1", "IPSL-CM6A-LR"]
NA_LAT = (0, 75)
NA_LON = (-80, 0)
YEARS = (1950, 2014)


def main():
    import intake
    cat = intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    print(f"Pangeo catalog: {len(cat.df)} rows")

    summary = {"high_res": {}, "low_res": {}}
    for group_name, models in [("high_res", HIGH_RES), ("low_res", LOW_RES)]:
        for source_id in models:
            # HighResMIP uses experiment_id 'hist-1950' for high-res
            # historical; the low-res group is standard 'historical'.
            for exp in ("hist-1950", "historical"):
                q = cat.search(
                    source_id=source_id, experiment_id=exp,
                    variable_id="tos", table_id="Omon",
                    member_id="r1i1p1f1",
                )
                if len(q.df) == 0:
                    q = cat.search(
                        source_id=source_id, experiment_id=exp,
                        variable_id="tos", table_id="Omon",
                        member_id="r1i1p1f2",
                    )
                if len(q.df) == 0:
                    continue
                row = q.df.iloc[0]
                zstore = row["zstore"]
                grid_label = row["grid_label"]
                t0 = time.time()
                try:
                    ds = xr.open_zarr(zstore, consolidated=True, chunks={})
                    # Subset to 1950-2014
                    ds = ds.sel(time=slice(f"{YEARS[0]}-01-01",
                                              f"{YEARS[1]}-12-31"))
                    # Get lat/lon names
                    lat_name = next((n for n in ("lat", "latitude")
                                       if n in ds.coords), None)
                    lon_name = next((n for n in ("lon", "longitude")
                                       if n in ds.coords), None)
                    if lat_name is None or lon_name is None:
                        print(f"  {source_id}/{exp}: can't find lat/lon; skip")
                        continue
                    # NA basin mask: lat in [0, 75], lon in [-80, 0]
                    lon_180 = ((ds[lon_name] + 180) % 360) - 180
                    mask = ((ds[lat_name] >= NA_LAT[0])
                            & (ds[lat_name] <= NA_LAT[1])
                            & (lon_180 >= NA_LON[0])
                            & (lon_180 <= NA_LON[1]))
                    tos_na = ds["tos"].where(mask)
                    # Compute annual mean to keep cache size small
                    tos_annual = tos_na.groupby("time.year").mean("time")
                    n_yr = tos_annual.sizes["year"]
                    print(f"  {source_id}/{exp}/{grid_label}: "
                          f"{n_yr} annual fields in {time.time()-t0:.0f}s")
                    out_path = OUT_DIR / f"{source_id}_{exp}_{grid_label}_tos_na.nc"
                    tos_annual.to_netcdf(out_path)
                    summary[group_name][source_id] = dict(
                        experiment=exp, grid_label=grid_label,
                        n_years=int(n_yr), file=str(out_path),
                    )
                    break   # got one experiment, move on
                except Exception as e:
                    print(f"  {source_id}/{exp}: FAILED -- {type(e).__name__}: {e}")
                    continue
            else:
                print(f"  {source_id}: no hist-1950 or historical found")

    with open(OUT_DIR / "highresmip_inventory.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {OUT_DIR / 'highresmip_inventory.json'}")


if __name__ == "__main__":
    main()
