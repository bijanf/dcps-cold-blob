"""Unit tests for quiescence_toolkit.src.quiescence_index."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path("/home/bijanf/Documents/NEW_Theory")
sys.path.insert(0, str(REPO / "quiescence_toolkit" / "src"))
sys.path.insert(0, str(REPO / "dcps"))

from quiescence_index import compute_Q, Q_from_basin_cache


def test_compute_Q_returns_expected_keys():
    """compute_Q must return a dict with Q, p_perm, n_cells, n_blocks.

    Synthetic 2-D field on a 10x10 lat-lon grid.
    """
    rng = np.random.default_rng(0)
    lat = np.linspace(10, 60, 10)
    lon = np.linspace(-80, -10, 10)
    r = rng.uniform(0.5, 0.9, (10, 10))
    d = rng.uniform(0.0, 1.0, (10, 10))
    result = compute_Q(r, d, lat, lon, B=20)
    for key in ("Q", "p_perm", "n_cells", "n_blocks"):
        assert key in result, f"missing key {key}"


def test_compute_Q_sign_convention():
    """For a synthetic anti-correlated field, Q should be positive."""
    rng = np.random.default_rng(1)
    lat = np.linspace(10, 60, 10)
    lon = np.linspace(-80, -10, 10)
    d = rng.uniform(0.1, 1.0, (10, 10))
    r = 1.0 - d + rng.normal(0, 0.05, (10, 10))
    result = compute_Q(r, d, lat, lon, B=20)
    assert result["Q"] > 0, f"Q should be > 0 for anti-corr; got {result['Q']}"


def test_Q_from_basin_cache_matches_published():
    """Cached basin Q values must match published within +/- 0.05."""
    published = {"atlantic": 0.32, "pacific": 0.35, "southern": 0.46}
    for basin, q_expected in published.items():
        result = Q_from_basin_cache(basin)
        if result is None:
            pytest.skip(f"{basin} cache not available")
        assert abs(result["Q"] - q_expected) <= 0.05, (
            f"{basin}: Q = {result['Q']:.3f}, expected ~{q_expected:.3f}"
        )
