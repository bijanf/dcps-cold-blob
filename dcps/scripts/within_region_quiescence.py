"""MAJOR 6 of the peer review: within-region Quiescence anti-correlation.

The basin-scale rho(<r_loc>_t, |grad SSH|) may be dominated by the
geographic contrast between two regions (subpolar gyre interior and
Gulf Stream / NAC pathway) rather than by a fine-grained cell-wise
mechanism.  Within-region tests answer: does the anti-correlation
survive when restricted to each region alone?
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.spatial_stats import (
    spatial_block_permutation,
    subpolar_gyre_mask,
    gulf_stream_pathway_mask,
)
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

OUT_DIR = CACHE_DIR / "within_region"
B = 500


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" MAJOR 6: within-region Quiescence anti-correlation (NA only)")
    print("=" * 70)

    basin = "atlantic"
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

    a = rl_mean.values.ravel()
    b = grad_mag.values.ravel()
    lat_flat = LAT.ravel()
    lon_flat = actual_lon.ravel()

    # Whole-basin reference
    full_r = spatial_block_permutation(a, b, lat_flat, lon_flat,
                                            block_km=500, B=B, seed=42)
    print(f"\n  Whole NA basin: rho={full_r['rho_observed']:+.3f}  "
          f"n_cells={full_r['n_cells']}  n_blocks={full_r['n_blocks']}  "
          f"p_perm={full_r['p_perm']:.4f}")

    # Subpolar gyre interior
    sp = subpolar_gyre_mask(lat_flat, lon_flat)
    sp_r = spatial_block_permutation(a[sp], b[sp], lat_flat[sp], lon_flat[sp],
                                          block_km=500, B=B, seed=42)
    print(f"  Subpolar gyre interior (lat>=45, lon in [-50,-10]): "
          f"rho={sp_r['rho_observed']:+.3f}  "
          f"n_cells={sp_r['n_cells']}  n_blocks={sp_r['n_blocks']}  "
          f"p_perm={sp_r['p_perm']:.4f}")

    # Gulf Stream / NAC pathway
    gs = gulf_stream_pathway_mask(lat_flat, lon_flat)
    gs_r = spatial_block_permutation(a[gs], b[gs], lat_flat[gs], lon_flat[gs],
                                          block_km=500, B=B, seed=42)
    print(f"  Gulf Stream + NAC (lat in [30,45], lon in [-75,-20]): "
          f"rho={gs_r['rho_observed']:+.3f}  "
          f"n_cells={gs_r['n_cells']}  n_blocks={gs_r['n_blocks']}  "
          f"p_perm={gs_r['p_perm']:.4f}")

    out = {
        "whole_NA": {k: v for k, v in full_r.items() if k != "rho_null"},
        "subpolar_gyre": {k: v for k, v in sp_r.items() if k != "rho_null"},
        "gulf_stream": {k: v for k, v in gs_r.items() if k != "rho_null"},
    }
    with open(OUT_DIR / "within_region.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'within_region.json'}")


if __name__ == "__main__":
    main()
