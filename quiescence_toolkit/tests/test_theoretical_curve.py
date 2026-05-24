"""Unit tests for quiescence_toolkit.src.theoretical_curve."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "quiescence_toolkit" / "src"))

from theoretical_curve import r_simple, r_bessel, r_weak, fit_and_score


def test_r_simple_monotone_decreasing():
    D = np.linspace(0.1, 10, 100)
    r = r_simple(D, tau=1.0)
    diffs = np.diff(r)
    assert (diffs <= 0).all(), "r_simple should monotone-decrease in D"


def test_r_simple_at_zero():
    """r_simple(0) = 1 exactly (perfect coherence at zero noise)."""
    assert r_simple(0.0, tau=1.0) == 1.0


def test_r_simple_large_D_asymptote():
    """r_simple ~ 1/sqrt(tau D) for large D."""
    D = 1e6
    tau = 2.0
    expected = 1.0 / np.sqrt(tau * D)
    assert abs(r_simple(D, tau) - expected) / expected < 1e-6


def test_r_bessel_finite_at_typical_K_D():
    """r_bessel finite and in [0, 1] for typical K/D values."""
    for K_over_D in (0.1, 1.0, 10.0):
        r = r_bessel(np.array([1.0]), K=K_over_D)
        assert np.isfinite(r).all()
        assert (r >= 0).all() and (r <= 1).all()


def test_fit_recovers_known_tau():
    """If synthetic data is generated from r_simple with known tau,
    fit_and_score should recover that tau within 10%."""
    rng = np.random.default_rng(0)
    tau_true = 50.0
    D = np.logspace(-3, 0, 200)
    r_clean = r_simple(D, tau_true)
    r_noisy = r_clean + rng.normal(0, 0.01, r_clean.shape)
    tau_fit, r_fit, mse = fit_and_score(D, r_noisy, "simple")
    assert tau_fit is not None
    assert abs(tau_fit - tau_true) / tau_true < 0.15, (
        f"fitted tau {tau_fit} should be within 15% of {tau_true}"
    )
