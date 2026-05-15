"""Spatial statistics utilities (BLOCKER 2, MAJOR 6 of the peer review).

Two needs:

1. Spatial-block permutation test for cell-wise Pearson correlations.
   `n = 1105` cells in the NA basin is the grid size, not the effective
   sample size; SST decorrelation length ~500 km gives `n_eff = O(30-50)`.
   The reviewer (A5) wants block-permutation p-values, not cell-count
   ones.

2. Region masks for within-region anti-correlation (MAJOR 6).  The
   basin-scale rho may be dominated by the geographic contrast between
   two regions (subpolar gyre interior vs Gulf Stream / NAC pathway).
"""

from __future__ import annotations

import numpy as np

from .geo import EARTH_R_KM, haversine_km as _haversine_km  # noqa: F401


def make_spatial_blocks(lat, lon, block_km=500.0):
    """Partition a flat (lat, lon) array into geographic blocks.

    Greedy assignment: scan cells in (lat, lon) order; for each cell,
    assign to the first existing block whose centroid is within
    block_km/2 of the cell, or open a new block.

    Returns: integer array `block_id` of same length as lat, lon.
    """
    lat = np.asarray(lat, dtype=float).ravel()
    lon = np.asarray(lon, dtype=float).ravel()
    n = lat.size
    block_id = np.full(n, -1, dtype=int)
    centroids = []
    radius = block_km / 2.0
    for i in range(n):
        if not (np.isfinite(lat[i]) and np.isfinite(lon[i])):
            continue
        if centroids:
            cs = np.asarray(centroids)
            d = _haversine_km(lat[i], lon[i], cs[:, 0], cs[:, 1])
            j = int(np.argmin(d))
            if d[j] <= radius:
                block_id[i] = j
                continue
        block_id[i] = len(centroids)
        centroids.append((lat[i], lon[i]))
    return block_id


def spatial_block_permutation(x, y, lat, lon, block_km=500.0, B=1000, seed=0):
    """Block-permutation Pearson correlation test.

    Strategy: partition cells into geographic blocks of radius block_km/2;
    for each replicate, randomly reassign each block to a destination
    block (shuffle the block-mean of x while keeping y fixed); recompute
    Pearson at cell level by broadcasting the shuffled block-means back
    to cells; the resulting empirical distribution gives a two-sided
    p-value bounded below by 1/B.

    Parameters
    ----------
    x, y     : 1-D arrays of cell-wise values (NaNs allowed; filtered)
    lat, lon : 1-D arrays of cell coordinates (same length as x, y)
    block_km : block diameter in km (default 500, matches SST
               decorrelation length)
    B        : number of permutations
    seed     : RNG seed

    Returns
    -------
    dict with:
      rho_observed : observed Pearson rho on valid cells
      n_cells      : number of cells used
      n_blocks     : number of spatial blocks
      n_eff        : effective sample size ~ number of blocks
      p_perm       : empirical two-sided p-value
      rho_null     : ndarray of B null-rho values
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    lon = np.asarray(lon, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(lat) & np.isfinite(lon)
    x, y = x[valid], y[valid]
    lat, lon = lat[valid], lon[valid]

    if x.size < 10:
        return dict(rho_observed=float("nan"), p_perm=float("nan"),
                    n_cells=int(x.size), n_blocks=0, n_eff=0,
                    rho_null=np.array([]))

    rho_obs = float(np.corrcoef(x, y)[0, 1])
    block_id = make_spatial_blocks(lat, lon, block_km=block_km)
    n_blocks = block_id.max() + 1
    rng = np.random.default_rng(seed)
    rho_null = np.empty(B)
    for b in range(B):
        # Permute block-mean of x; broadcast back to cells.
        # Each block keeps its members, but its x-values get replaced by
        # those of a randomly chosen other block (pseudo-resampling).
        perm = rng.permutation(n_blocks)
        x_shuffled = x.copy()
        for src, dst in enumerate(perm):
            mask = (block_id == src)
            if not mask.any():
                continue
            # Use shuffled-block's mean for the destination cells:
            x_shuffled[mask] = x[block_id == dst].mean()
        rho_null[b] = float(np.corrcoef(x_shuffled, y)[0, 1])
    # Two-sided empirical p-value
    p_perm = float((np.abs(rho_null) >= abs(rho_obs)).mean())
    p_perm = max(p_perm, 1.0 / B)
    return dict(
        rho_observed=rho_obs,
        p_perm=p_perm,
        n_cells=int(x.size),
        n_blocks=int(n_blocks),
        n_eff=int(n_blocks),
        rho_null=rho_null,
    )


# -----------------------------------------------------------------------------
#   Region masks for MAJOR 6 (within-region anti-correlation)
# -----------------------------------------------------------------------------

def subpolar_gyre_mask(lat, lon):
    """Subpolar gyre interior: lat > 45 N AND lon in (-50, -10).
    Excludes the Labrador Sea convection sites by lon > -50.
    """
    lat = np.asarray(lat); lon = np.asarray(lon)
    return (lat >= 45.0) & (lat <= 65.0) & (lon >= -50.0) & (lon <= -10.0)


def gulf_stream_pathway_mask(lat, lon):
    """Gulf Stream + North Atlantic Current pathway: 30-45 N,
    -75 to -20 W.  The main eddy-active corridor."""
    lat = np.asarray(lat); lon = np.asarray(lon)
    return (lat >= 30.0) & (lat <= 45.0) & (lon >= -75.0) & (lon <= -20.0)


# -----------------------------------------------------------------------------
#   Moran's I and effective sample size (additional diagnostic)
# -----------------------------------------------------------------------------

def morans_i(field, lat, lon, k_neighbours=8):
    """Compute global Moran's I of a flat field using k-nearest-neighbour
    spatial weights.  Returns (I, expected_I_under_null, n_eff_estimate).

    The Dale & Fortin (1999) effective-sample-size adjustment based on
    Moran's I is:
        n_eff ~ n * (1 - I) / (1 + I)
    valid for moderate positive autocorrelation.
    """
    field = np.asarray(field, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    lon = np.asarray(lon, dtype=float).ravel()
    valid = np.isfinite(field) & np.isfinite(lat) & np.isfinite(lon)
    z = field[valid]
    z = z - z.mean()
    lat_v = lat[valid]; lon_v = lon[valid]
    n = z.size
    if n < 5:
        return float("nan"), float("nan"), 0

    # k-NN spatial weights
    W = np.zeros((n, n))
    for i in range(n):
        d = _haversine_km(lat_v[i], lon_v[i], lat_v, lon_v)
        d[i] = np.inf
        nn = np.argpartition(d, k_neighbours)[:k_neighbours]
        W[i, nn] = 1.0
    # Row-normalize
    rs = W.sum(axis=1, keepdims=True)
    W = np.where(rs > 0, W / rs, 0.0)
    S0 = W.sum()
    num = (W * np.outer(z, z)).sum()
    den = (z * z).sum()
    I = (n / S0) * (num / den) if den > 0 and S0 > 0 else float("nan")
    E_I = -1.0 / (n - 1)
    n_eff = max(int(n * (1.0 - I) / (1.0 + I)), 1) if np.isfinite(I) and (1 + I) > 0 else n
    return float(I), float(E_I), int(n_eff)


__all__ = [
    "spatial_block_permutation",
    "make_spatial_blocks",
    "subpolar_gyre_mask",
    "gulf_stream_pathway_mask",
    "morans_i",
]
