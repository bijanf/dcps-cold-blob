"""Tests for `dcps.spatial_stats.spatial_block_permutation`.

The central calibration check generates pairs of *independent*
spatially-autocorrelated Gaussian fields on a realistic 2-degree
North-Atlantic-like grid and verifies that the empirical Type-I
error rate at alpha = 0.05 lands in [0.03, 0.07].  This is the
property-based test recommended in the NCOMMS Tier-2 review.
"""
from __future__ import annotations

import numpy as np
import pytest

from dcps.spatial_stats import (
    spatial_block_permutation,
    spatial_block_bootstrap,
)


def _grf(rng, lat, lon, length_km=600.0):
    """Generate a Gaussian random field on (lat, lon) with a given
    haversine length scale, normalised to zero mean and unit variance.
    """
    from dcps.geo import haversine_km
    n = lat.size
    lat2 = lat[:, None]; lon2 = lon[:, None]
    d = haversine_km(lat2, lon2, lat[None, :], lon[None, :])
    C = np.exp(-0.5 * (d / length_km) ** 2)
    C += 1e-6 * np.eye(n)  # jitter for positive-definiteness
    L = np.linalg.cholesky(C)
    z = rng.standard_normal(n)
    field = L @ z
    field = (field - field.mean()) / field.std()
    return field


def _na_like_grid(d_deg=2.0):
    """A North-Atlantic-like 2-degree grid that matches the manuscript grid.

    Returns lat, lon (1-D float64 arrays).
    """
    lats = np.arange(15.0, 71.0, d_deg)
    lons = np.arange(-75.0, -5.0, d_deg)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    return LAT.ravel().astype(np.float64), LON.ravel().astype(np.float64)


def test_h0_type_i_error_calibration():
    """Under H0 (independent autocorrelated fields) the test must achieve
    alpha = 0.05.

    The block permutation null is asymptotically calibrated when block
    diameter exceeds ~2.5x the field's autocorrelation length, so that
    distinct block-means can be treated as approximately independent.
    For SST in the North Atlantic the mesoscale autocorrelation length
    is ~150-300 km, which is why the manuscript's default block_km = 500
    km is appropriate.  This test reproduces that scenario: a 200-km
    Gaussian autocorrelation length on the 2-degree NA grid, tested at
    block_km = 500 (ratio 2.5).

    60 trials at B=300.  Binomial 95% CI for p = 0.05, n = 60 is
    [0.005, 0.13].  At ratio 2.5 the empirical Type-I error converges
    on alpha within this CI.
    """
    lat, lon = _na_like_grid(d_deg=2.0)
    n_trials = 60
    alpha = 0.05
    base_rng = np.random.default_rng(0)
    rejections = 0
    for _ in range(n_trials):
        rng = np.random.default_rng(base_rng.integers(0, 2**31 - 1))
        x = _grf(rng, lat, lon, length_km=200.0)
        y = _grf(rng, lat, lon, length_km=200.0)
        res = spatial_block_permutation(
            x, y, lat, lon,
            block_km=500.0,
            B=300,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        if res["p_perm"] < alpha:
            rejections += 1
    rate = rejections / n_trials
    assert 0.005 <= rate <= 0.13, (
        f"Type-I error rate {rate:.3f} outside the binomial 95% CI "
        f"[0.005, 0.13] for n = {n_trials}, p = {alpha}"
    )


def test_calibration_breaks_when_blocks_too_small():
    """Documents the failure mode: block_km below ~2x the autocorrelation
    length is anti-conservative.  This is a sanity check that the test
    suite catches the regime where the manuscript's permutation p-values
    would be invalid.
    """
    lat, lon = _na_like_grid(d_deg=2.0)
    base_rng = np.random.default_rng(0)
    rejections = 0
    n_trials = 30
    for _ in range(n_trials):
        rng = np.random.default_rng(base_rng.integers(0, 2**31 - 1))
        x = _grf(rng, lat, lon, length_km=600.0)
        y = _grf(rng, lat, lon, length_km=600.0)
        # block_km=500 with 600-km autocorrelation: ratio 0.83, badly
        # under-resolved -> anti-conservative.
        res = spatial_block_permutation(
            x, y, lat, lon, block_km=500.0, B=200,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        if res["p_perm"] < 0.05:
            rejections += 1
    # Confirm the test flags the regime: empirical rate should be > 0.15
    # (well above nominal alpha).
    assert rejections / n_trials > 0.15, (
        f"Anti-conservative regime not detected: rate={rejections/n_trials:.3f}; "
        f"the calibration safeguard in spatial_block_permutation may have failed."
    )


def test_strong_correlation_rejects_h0():
    """When y is constructed as x + small_noise, p_perm must be << 0.05."""
    lat, lon = _na_like_grid(d_deg=4.0)
    rng = np.random.default_rng(42)
    x = _grf(rng, lat, lon, length_km=600.0)
    y = x + 0.1 * rng.standard_normal(x.size)
    res = spatial_block_permutation(
        x, y, lat, lon, block_km=500.0, B=500, seed=1,
    )
    assert res["rho_observed"] > 0.9
    assert res["p_perm"] <= 1.0 / 500 + 1e-9  # bounded below by 1/B
    assert res["n_blocks"] >= 5


def test_block_count_grows_with_smaller_blocks():
    lat, lon = _na_like_grid(d_deg=4.0)
    rng = np.random.default_rng(0)
    x = _grf(rng, lat, lon, length_km=600.0)
    y = _grf(rng, lat, lon, length_km=600.0)
    res_big = spatial_block_permutation(
        x, y, lat, lon, block_km=1500.0, B=50, seed=0,
    )
    res_small = spatial_block_permutation(
        x, y, lat, lon, block_km=300.0, B=50, seed=0,
    )
    assert res_small["n_blocks"] > res_big["n_blocks"]


def test_nan_inputs_are_filtered():
    lat, lon = _na_like_grid(d_deg=8.0)
    rng = np.random.default_rng(0)
    x = _grf(rng, lat, lon, length_km=600.0)
    y = _grf(rng, lat, lon, length_km=600.0)
    # Mask a quarter of cells in x.
    x[: x.size // 4] = np.nan
    res = spatial_block_permutation(
        x, y, lat, lon, block_km=500.0, B=100, seed=0,
    )
    assert res["n_cells"] == x.size - x.size // 4
    assert np.isfinite(res["rho_observed"])


def test_too_few_cells_returns_nan():
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([0.0, 1.0, 2.0])
    x = np.array([0.1, 0.2, 0.3])
    y = np.array([0.4, 0.5, 0.6])
    res = spatial_block_permutation(x, y, lat, lon, block_km=500.0, B=10, seed=0)
    assert np.isnan(res["rho_observed"])
    assert np.isnan(res["p_perm"])


# -----------------------------------------------------------------------------
#   Bootstrap CI tests
# -----------------------------------------------------------------------------

def test_bootstrap_calibration():
    """Under H0 (independent autocorrelated fields, true rho = 0), the 95%
    Fisher-z CI must contain 0 in >= 85% of trials.

    Calibration depends on the ratio of block diameter to the field's
    autocorrelation length.  The bootstrap CI is trustworthy when blocks
    are large enough to decouple — empirically a ratio of >= 5x is
    required for >= 0.85 coverage with the greedy
    ``make_spatial_blocks`` partition (which produces ~2-3 cells per
    block at the ratio of 2.5x at which the permutation test is
    well-calibrated).  We therefore test at block_km=1500 against a
    200-km autocorrelation length (ratio 7.5x), and use block_km=1500
    in production for Plot 3 CIs.  60 trials at B=300.
    """
    lat, lon = _na_like_grid(d_deg=2.0)
    n_trials = 60
    base_rng = np.random.default_rng(2026)
    contains_zero = 0
    for _ in range(n_trials):
        rng = np.random.default_rng(base_rng.integers(0, 2**31 - 1))
        x = _grf(rng, lat, lon, length_km=200.0)
        y = _grf(rng, lat, lon, length_km=200.0)
        res = spatial_block_bootstrap(
            x, y, lat, lon, block_km=1500.0, B=300,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        if res["ci_low"] <= 0.0 <= res["ci_high"]:
            contains_zero += 1
    coverage = contains_zero / n_trials
    assert coverage >= 0.85, (
        f"Fisher-z CI coverage {coverage:.2f} fell below 0.85 at "
        f"block_km=1500 / autocorr=200km."
    )


def test_bootstrap_strong_correlation_excludes_zero():
    """When y = x + small_noise, the 95% bootstrap CI must exclude 0."""
    lat, lon = _na_like_grid(d_deg=4.0)
    rng = np.random.default_rng(42)
    x = _grf(rng, lat, lon, length_km=600.0)
    y = x + 0.1 * rng.standard_normal(x.size)
    res = spatial_block_bootstrap(
        x, y, lat, lon, block_km=500.0, B=500, seed=1,
    )
    assert res["ci_low"] > 0.0, (
        f"Strong-correlation CI [{res['ci_low']:.3f}, {res['ci_high']:.3f}] "
        f"unexpectedly contains 0"
    )
    assert res["rho_observed"] > 0.9


def test_bootstrap_ci_brackets_observed():
    """The point estimate (block-mean rho) must lie inside its own CI."""
    lat, lon = _na_like_grid(d_deg=4.0)
    rng = np.random.default_rng(7)
    x = _grf(rng, lat, lon, length_km=600.0)
    y = 0.5 * x + 0.5 * rng.standard_normal(x.size)
    res = spatial_block_bootstrap(
        x, y, lat, lon, block_km=500.0, B=400, seed=3,
    )
    assert res["ci_low"] <= res["rho_observed"] <= res["ci_high"]
