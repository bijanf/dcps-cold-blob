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
    """Spatial-block permutation Pearson correlation test.

    Tests whether two spatially-autocorrelated cell-wise fields are more
    correlated than expected by chance, accounting for the loss of
    effective sample size caused by spatial autocorrelation.

    Strategy
    --------
    1. Partition cells into geographic blocks of approximate radius
       ``block_km / 2`` (`make_spatial_blocks`).  The block diameter must
       be at least the autocorrelation length of the fields, so that
       distinct blocks can be treated as approximately independent units
       under the null.
    2. Reduce each block to its arithmetic mean of *x* and *y*; this
       gives ``n_blocks`` paired observations whose effective autocorrelation
       is low.
    3. Observed test statistic: Pearson correlation on the block-mean
       pairs.
    4. For each of *B* replicates, permute the block labels of *x*
       (equivalently of *y*) and recompute the Pearson on the permuted
       pairs.  This generates a calibrated null distribution under H0
       of independence.
    5. The two-sided empirical p-value is the fraction of replicates with
       ``|rho_null| >= |rho_blocks|``, bounded below by ``1 / B``.

    The cell-wise Pearson on the full-resolution arrays is also returned
    as ``rho_observed`` for descriptive context, but **inference uses the
    block-mean p-value** because cell-wise rho's significance is
    inflated by spatial autocorrelation.  Effective sample size for the
    test is ``n_blocks``.

    Calibration: under H0 of two independent spatially-autocorrelated
    Gaussian fields the empirical Type-I error rate at alpha = 0.05
    lands in [0.025, 0.085] (binomial 95% CI for n_trials = 100); see
    ``dcps/tests/test_spatial_stats.py::test_h0_type_i_error_calibration``.

    Notes
    -----
    Earlier implementations (commits prior to the 2026-05 NCOMMS revision)
    replaced source-block cells with the *block-mean* of a destination
    block and computed the null Pearson against the *full-resolution* y;
    that mixed scales and biased the null variance.  An interim cell-level
    label-swap implementation failed a calibration check at alpha = 0.05
    because permuted x_shuffled still carried within-block autocorrelation
    that the cell-level rho_obs lacked.  The current implementation runs
    the inference on block means, which is both standard and calibrated.

    Parameters
    ----------
    x, y     : 1-D arrays of cell-wise values (NaNs allowed; filtered).
    lat, lon : 1-D arrays of cell coordinates (same length as x, y).
    block_km : block diameter in km (default 500, matches SST
               decorrelation length).
    B        : number of permutations.
    seed     : RNG seed.

    Returns
    -------
    dict with:
      rho_observed : cell-wise Pearson rho on valid cells (descriptive).
      rho_blocks   : Pearson rho on block-mean pairs (the test statistic).
      n_cells      : number of cells used.
      n_blocks     : number of spatial blocks (effective sample size).
      n_eff        : alias of n_blocks for backward compatibility.
      p_perm       : empirical two-sided p-value on the block-mean null.
      rho_null     : ndarray of B null rho values.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    lon = np.asarray(lon, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(lat) & np.isfinite(lon)
    x, y = x[valid], y[valid]
    lat, lon = lat[valid], lon[valid]

    if x.size < 10:
        return dict(rho_observed=float("nan"), rho_blocks=float("nan"),
                    p_perm=float("nan"),
                    n_cells=int(x.size), n_blocks=0, n_eff=0,
                    rho_null=np.array([]))

    rho_cells = float(np.corrcoef(x, y)[0, 1])
    block_id = make_spatial_blocks(lat, lon, block_km=block_km)
    n_blocks = int(block_id.max() + 1)

    # Block means of x and y.  Empty blocks (no cells assigned) are NaN
    # and excluded from the test.
    xb = np.full(n_blocks, np.nan)
    yb = np.full(n_blocks, np.nan)
    for b in range(n_blocks):
        cells = np.where(block_id == b)[0]
        if cells.size:
            xb[b] = x[cells].mean()
            yb[b] = y[cells].mean()
    finite = np.isfinite(xb) & np.isfinite(yb)
    xb_v = xb[finite]
    yb_v = yb[finite]
    n_eff = int(finite.sum())

    if n_eff < 4:
        # Not enough blocks for a meaningful permutation test.
        return dict(rho_observed=rho_cells, rho_blocks=float("nan"),
                    p_perm=float("nan"),
                    n_cells=int(x.size), n_blocks=n_blocks, n_eff=n_eff,
                    rho_null=np.array([]))

    rho_blocks = float(np.corrcoef(xb_v, yb_v)[0, 1])

    rng = np.random.default_rng(seed)
    rho_null = np.empty(B)
    for b in range(B):
        perm = rng.permutation(n_eff)
        rho_null[b] = float(np.corrcoef(xb_v[perm], yb_v)[0, 1])
    p_perm = float((np.abs(rho_null) >= abs(rho_blocks)).mean())
    p_perm = max(p_perm, 1.0 / B)
    return dict(
        rho_observed=rho_cells,
        rho_blocks=rho_blocks,
        p_perm=p_perm,
        n_cells=int(x.size),
        n_blocks=n_blocks,
        n_eff=n_eff,
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
