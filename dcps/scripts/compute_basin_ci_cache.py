"""Per-basin bootstrap CI cache for Plot R3 column 1 (|grad SSH| proxy).

For each basin (atlantic, pacific, southern), reload the ORAS5
SST + SSH pipeline that ``multi_basin_quiescence.py`` runs in-memory,
extract the cell-wise (x = |grad SSH|, y = <r_loc>, lat, lon) arrays,
save them to disk in basin_cellwise_<basin>.nc so subsequent CI
recomputes are fast, and run ``spatial_block_bootstrap`` (block_km=1500,
B=1000) to get a Fisher-z 95 percent CI on rho per basin.

Writes ``dcps/cache/multi_basin/q_ci_bootstrap.json`` with
``{basin: {rho_observed, ci_low, ci_high, n_cells, n_eff}}``.

If ``basin_cellwise_<basin>.nc`` already exists, the ORAS5 pipeline is
skipped for that basin and only the bootstrap runs.  This keeps
re-runs cheap once the first slow pass has populated the cache.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.spatial_stats import spatial_block_bootstrap

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import (
    BASINS,
    load_oras5_basin,
    regrid_basin,
    preprocess_anomaly,
    instantaneous_phase,
    local_r_mean,
)


OUT_DIR = CACHE_DIR / "multi_basin"
BLOCK_KM = 1500.0
B = 1000


def _compute_basin_arrays(basin: str) -> xr.Dataset:
    """Run ORAS5 + Hilbert + r_loc + |grad SSH| for a basin, return as a
    cell-wise Dataset.
    """
    sst, lat2d, rlon2d = load_oras5_basin("sosstsst", basin)
    ssh, _, _ = load_oras5_basin("sossheig", basin)
    sst_rg = regrid_basin(sst, lat2d, rlon2d, basin)
    ssh_rg = regrid_basin(ssh, lat2d, rlon2d, basin)
    sst_anom = preprocess_anomaly(sst_rg)
    phi = instantaneous_phase(sst_anom)
    n_t = phi.sizes["time"]
    phi = phi.isel(time=slice(6, n_t - 6))
    rl_mean = local_r_mean(phi, radius_km=500.0)
    ssh_mean = ssh_rg.mean("time")
    grad_mag = np.sqrt(ssh_mean.differentiate("lat") ** 2
                       + ssh_mean.differentiate("rlon") ** 2)

    lat = rl_mean["lat"].values
    rlon = rl_mean["rlon"].values
    LAT, RLON = np.meshgrid(lat, rlon, indexing="ij")
    lon_offset = BASINS[basin]["lon_offset"]
    actual_lon = ((RLON + lon_offset + 180.0) % 360.0) - 180.0

    return xr.Dataset(
        data_vars=dict(
            r_loc=(("lat", "rlon"), rl_mean.values),
            grad_ssh=(("lat", "rlon"), grad_mag.values),
            actual_lat=(("lat", "rlon"), LAT),
            actual_lon=(("lat", "rlon"), actual_lon),
        ),
        coords=dict(lat=lat, rlon=rlon),
    )


def _bootstrap_basin(basin: str, ds: xr.Dataset, seed: int) -> dict:
    r = ds["r_loc"].values.ravel()
    g = ds["grad_ssh"].values.ravel()
    lat = ds["actual_lat"].values.ravel()
    lon = ds["actual_lon"].values.ravel()
    print(f"  [{basin}] spatial_block_bootstrap: block_km={BLOCK_KM}, B={B}")
    t0 = time.time()
    res = spatial_block_bootstrap(
        r, g, lat, lon, block_km=BLOCK_KM, B=B, seed=seed,
    )
    rho = float(res["rho_cells"])  # cell-wise observed rho
    rho_block = float(res["rho_observed"])
    print(f"    rho_cells={rho:+.3f}  rho_blocks={rho_block:+.3f}  "
          f"CI=({res['ci_low']:+.3f}, {res['ci_high']:+.3f})  "
          f"n_eff={res['n_eff']}  ({time.time() - t0:.0f}s)")
    return dict(
        basin=basin,
        rho_observed=rho,
        rho_block=rho_block,
        ci_low=float(res["ci_low"]),
        ci_high=float(res["ci_high"]),
        n_cells=int(res["n_cells"]),
        n_blocks=int(res["n_blocks"]),
        n_eff=int(res["n_eff"]),
        block_km=BLOCK_KM,
        B=B,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" Plot R3 col 1: per-basin bootstrap CIs on (<r_loc>, |grad SSH|)")
    print("=" * 70)

    results = {}
    for i, basin in enumerate(BASINS):
        cache_nc = OUT_DIR / f"basin_cellwise_{basin}.nc"
        print(f"\n[{basin}] {BASINS[basin]['label']}")
        if cache_nc.exists():
            print(f"  cache hit: {cache_nc}")
            ds = xr.open_dataset(cache_nc)
        else:
            t0 = time.time()
            print(f"  cache miss; running ORAS5 pipeline ...")
            ds = _compute_basin_arrays(basin)
            ds.to_netcdf(cache_nc)
            print(f"  wrote {cache_nc} ({time.time() - t0:.0f}s pipeline)")
        results[basin] = _bootstrap_basin(basin, ds, seed=100 + i)

    out_json = OUT_DIR / "q_ci_bootstrap.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
