"""Sidecar: compute bootstrap CIs for the eddy-resolving (r_loc, EKE)
correlation per basin, reading the cached 2-deg NetCDFs that
``eke_quiescence_eddy_resolving.py`` writes.

Outputs ``dcps/cache/eke_eddy_resolving/q_ci_bootstrap_eddy.json`` with
``{basin: {rho_observed, ci_low, ci_high, n_cells, n_eff}}`` for each
basin whose ``rl_mean_2deg_glorys12<suf>.nc`` and
``eke_2deg_from_native<suf>.nc`` are present.  Basins without the
cached NetCDFs are skipped.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.spatial_stats import spatial_block_bootstrap

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import BASINS  # noqa: E402


EKE_DIR = CACHE_DIR / "eke_eddy_resolving"
BLOCK_KM = 1500.0
B = 1000


def _suf(basin):
    return "" if basin == "atlantic" else f"_{basin}"


def _open_basin(basin):
    rl_path = EKE_DIR / f"rl_mean_2deg_glorys12{_suf(basin)}.nc"
    eke_path = EKE_DIR / f"eke_2deg_from_native{_suf(basin)}.nc"
    if not rl_path.exists() or not eke_path.exists():
        return None
    rl_ds = xr.open_dataset(rl_path)
    rl = rl_ds[list(rl_ds.data_vars)[0]]
    eke = xr.open_dataset(eke_path)["EKE_box_avg_2deg"]
    lat = rl["lat"].values
    rlon = rl["rlon"].values
    LAT, RLON = np.meshgrid(lat, rlon, indexing="ij")
    lon_offset = BASINS[basin]["lon_offset"]
    actual_lon = ((RLON + lon_offset + 180.0) % 360.0) - 180.0
    return rl.values, eke.values, LAT, actual_lon


def main():
    print("=" * 70)
    print(" Plot R3 col 2: eddy-resolving EKE bootstrap CIs per basin")
    print("=" * 70)
    results = {}
    for i, basin in enumerate(BASINS):
        data = _open_basin(basin)
        if data is None:
            print(f"[{basin}] cache not present yet -- skip")
            continue
        r, e, lat2d, lon2d = data
        r_flat = r.ravel(); e_flat = e.ravel()
        lat_flat = lat2d.ravel(); lon_flat = lon2d.ravel()
        # Apply the equatorial geostrophic mask consistently with the
        # rest of the eddy-resolving pipeline.
        eq = np.abs(lat_flat) < 15.0
        r_flat = np.where(eq, np.nan, r_flat)
        e_flat = np.where(eq, np.nan, e_flat)
        print(f"\n[{basin}] cells before mask: {r_flat.size}")
        t0 = time.time()
        res = spatial_block_bootstrap(
            r_flat, e_flat, lat_flat, lon_flat,
            block_km=BLOCK_KM, B=B, seed=200 + i,
        )
        rho = float(res["rho_cells"])
        rho_block = float(res["rho_observed"])
        print(f"  rho_cells={rho:+.3f}  rho_blocks={rho_block:+.3f}  "
              f"CI=({res['ci_low']:+.3f}, {res['ci_high']:+.3f})  "
              f"n_eff={res['n_eff']}  ({time.time() - t0:.0f}s)")
        results[basin] = dict(
            basin=basin,
            rho_observed=rho,
            rho_block=rho_block,
            ci_low=float(res["ci_low"]),
            ci_high=float(res["ci_high"]),
            n_cells=int(res["n_cells"]),
            n_blocks=int(res["n_blocks"]),
            n_eff=int(res["n_eff"]),
            block_km=BLOCK_KM, B=B,
        )

    out_json = EKE_DIR / "q_ci_bootstrap_eddy.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
