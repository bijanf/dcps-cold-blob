"""BLOCKER 2 of the peer review: replace cell-count p-values for the
published Quiescence correlations with spatial-block permutation
p-values.

For each of the three basins (NA, NP, ACC), reconstruct the cell-wise
(<r_loc>_t, |grad SSH|) data and run a 500-km spatial-block
permutation Pearson test.  Report the new p_perm (bounded below by
1/B) and the effective sample size n_eff = number of blocks.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.spatial_stats import spatial_block_permutation, morans_i
from dcps.nature_style import apply_nature_style
apply_nature_style()

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import (
    BASINS,
    load_oras5_basin,
    regrid_basin,
    preprocess_anomaly,
    instantaneous_phase,
    local_r_mean,
)


OUT_DIR = CACHE_DIR / "spatial_perm"
B = 500
BLOCK_KM = 500.0


def run_basin(basin):
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
    # Convert rotated lon to actual lon
    lon_offset = BASINS[basin]["lon_offset"]
    actual_lon = ((RLON + lon_offset + 180.0) % 360.0) - 180.0

    a = rl_mean.values.ravel()
    b = grad_mag.values.ravel()
    lat_flat = LAT.ravel()
    lon_flat = actual_lon.ravel()

    print(f"  {basin}: running spatial-block permutation (B={B}, "
          f"block_km={BLOCK_KM}) ...")
    t0 = time.time()
    result = spatial_block_permutation(a, b, lat_flat, lon_flat,
                                            block_km=BLOCK_KM, B=B, seed=42)
    # Also compute Moran's I on r_loc field for diagnostic
    I_r, E_I, n_eff_I = morans_i(a, lat_flat, lon_flat, k_neighbours=8)
    print(f"    rho = {result['rho_observed']:+.3f}  "
          f"n_cells = {result['n_cells']}  "
          f"n_blocks = {result['n_blocks']}  "
          f"p_perm = {result['p_perm']:.4f}  "
          f"({time.time()-t0:.0f}s)")
    print(f"    Moran I on <r_loc>_t = {I_r:+.3f} (E = {E_I:.3f}, "
          f"n_eff_Moran = {n_eff_I})")
    out = {
        "basin": basin,
        "rho_observed": result["rho_observed"],
        "p_perm": result["p_perm"],
        "n_cells": result["n_cells"],
        "n_blocks_500km": result["n_blocks"],
        "morans_i_r_loc": I_r,
        "morans_i_expected": E_I,
        "n_eff_morans": n_eff_I,
        "B": B,
    }
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" BLOCKER 2: spatial-block permutation for Quiescence Pearson tests")
    print("=" * 70)

    results = {}
    for basin in BASINS:
        try:
            results[basin] = run_basin(basin)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {basin}: FAILED -- {e}")

    print()
    print("=" * 70)
    print(" Summary: spatial-block permutation vs cell-count p-value")
    print("=" * 70)
    print(f"{'basin':<10} {'rho':>7} {'n_cells':>9} {'n_blocks':>10} "
          f"{'p_perm':>10} {'p_published':>14}")
    print("-" * 70)
    # Reference: original cell-count p-values from multi_basin cache
    ref = json.loads(
        (CACHE_DIR / "multi_basin" / "multi_basin_quiescence.json").read_text())
    for b, r in results.items():
        p_pub = ref[b]["p"]
        print(f"{b:<10} {r['rho_observed']:+7.3f} {r['n_cells']:>9} "
              f"{r['n_blocks_500km']:>10} {r['p_perm']:>10.4f} "
              f"{p_pub:>14.2e}")

    with open(OUT_DIR / "spatial_perm.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'spatial_perm.json'}")


if __name__ == "__main__":
    main()
