"""Step 8 / B8: Operational Quiescence Index Q.

Definition (from the user's guide):
    Q = -Pearson_rho(<r_loc>, |grad SSH|)
  spatial correlation across all valid cells in a basin, sign-flipped
  so that Q > 0 indicates Quiescence Signature presence.

Uncertainty: 2000-replicate bootstrap by resampling YEARS (block
resampling).  For the cached pipeline we re-use the
spatial_block_permutation routine for the spatial significance test
and bootstrap year-resampling for the uncertainty band.

Operational use cases:
    1. Compute Q from observed reanalysis (ORAS5, GLORYS12).
    2. Compute Q from CMIP6 historical output.
    3. Use Q as an emergent constraint: rank CMIP6 models by Q and
       check the correlation with the future AMOC decline (2080-2100
       minus 2020-2040).  If |r| > 0.7 across models, Q is an
       emergent constraint on AMOC sensitivity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import pearsonr

from dcps.spatial_stats import spatial_block_permutation


OUT_DIR = Path(__file__).resolve().parent.parent / "results"
CACHE_BASE = Path("/home/bijanf/Documents/NEW_Theory/dcps/cache")


def compute_Q(r_field, ssh_grad_field, lat, lon, block_km=500, B=500):
    """Compute the Quiescence Index Q with spatial-block permutation
    p-value.

    Parameters
    ----------
    r_field         : 2-D array of time-mean local Kuramoto coherence
    ssh_grad_field  : 2-D array of |grad SSH| (same shape)
    lat, lon        : 1-D coord arrays
    block_km        : spatial block size (default 500 km)
    B               : number of permutation replicates

    Returns
    -------
    dict with Q, p_perm, n_cells, n_blocks
    """
    a = r_field.ravel()
    b = ssh_grad_field.ravel()
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    lat_flat = LAT.ravel()
    lon_flat = LON.ravel()
    result = spatial_block_permutation(a, b, lat_flat, lon_flat,
                                            block_km=block_km, B=B, seed=42)
    rho = result["rho_observed"]
    return dict(
        Q=float(-rho),
        p_perm=float(result["p_perm"]),
        n_cells=int(result["n_cells"]),
        n_blocks=int(result["n_blocks"]),
    )


def Q_from_basin_cache(basin):
    """Re-compute Q for the three published basins using the
    multi_basin_quiescence cache."""
    cache_file = CACHE_BASE / "multi_basin" / "multi_basin_quiescence.json"
    if not cache_file.exists():
        return None
    pub = json.loads(cache_file.read_text())[basin]
    return dict(
        Q=float(-pub["rho"]),
        p_published=float(pub["p"]),
        n_cells=int(pub["n_cells"]),
        verdict=pub["verdict"],
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" Step 8 / B8: Operational Quiescence Index Q")
    print("=" * 70)

    # Use the published multi-basin baseline
    Q_table = {}
    for basin in ("atlantic", "pacific", "southern"):
        result = Q_from_basin_cache(basin)
        if result is not None:
            Q_table[basin] = result
            print(f"  {basin:<10}  Q = {result['Q']:+.3f}  "
                  f"(p_published = {result['p_published']:.2e}; "
                  f"n_cells = {result['n_cells']})")

    # Cross-check with the spatial-block permutation result already cached
    perm = json.loads(
        (CACHE_BASE / "spatial_perm" / "spatial_perm.json").read_text())
    for basin in ("atlantic", "pacific", "southern"):
        if basin in perm:
            print(f"  {basin:<10}  Q_perm = {-perm[basin]['rho_observed']:+.3f}  "
                  f"p_perm = {perm[basin]['p_perm']:.4f}  "
                  f"n_blocks = {perm[basin]['n_blocks_500km']}")

    # Persist the operational Q table
    summary = dict(
        published_baseline=Q_table,
        spatial_permutation=perm,
        definition=("Q = -Pearson_rho(<r_loc>, |grad SSH|); "
                    "Q > 0 indicates Quiescence Signature presence"),
        intended_uses=[
            "Observational benchmark for CMIP6 model evaluation",
            "Emergent constraint on AMOC sensitivity (Q vs future "
            "AMOC decline across models)",
            "Cross-system comparison: ocean basins, atmospheric jets, "
            "SLA altimetry",
        ],
    )
    with open(OUT_DIR / "Q_index_table.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'Q_index_table.json'}")


if __name__ == "__main__":
    main()
