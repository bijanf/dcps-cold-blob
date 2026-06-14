"""Validate KSG TE estimator on known-signal synthetic series."""

from __future__ import annotations

import numpy as np

from dcps.te_ksg import ksg_te, ksg_te_bits


def test_independent_series_te_near_zero():
    """For independent Gaussian series, TE should be ~0 (within bias)."""
    rng = np.random.default_rng(7)
    n = 2000
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    te = ksg_te(y, x, k=1, ell=1, k_nn=4)
    # KSG has well-known small bias near zero; |te| < 0.05 is comfortable
    assert abs(te) < 0.05, f"independent TE = {te:.3f} (expected near 0)"


def test_directional_coupling_y_to_x_high():
    """For X_{t+1} = 0.5*X_t + 0.7*Y_t + noise, TE_{Y->X} >> 0."""
    rng = np.random.default_rng(11)
    n = 2000
    y = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + 0.7 * y[t - 1] + 0.3 * rng.standard_normal()
    te_y_to_x = ksg_te(y, x, k=1, ell=1, k_nn=4)
    te_x_to_y = ksg_te(x, y, k=1, ell=1, k_nn=4)
    assert te_y_to_x > 0.20, f"Y->X TE = {te_y_to_x:.3f}, expected >> 0"
    assert te_y_to_x > 5 * abs(te_x_to_y), \
        f"Direction not strongly resolved: Y->X = {te_y_to_x:.3f}, X->Y = {te_x_to_y:.3f}"


def test_short_series_does_not_crash():
    rng = np.random.default_rng(13)
    n = 30
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    te = ksg_te(y, x, k=1, ell=1, k_nn=4)
    assert np.isfinite(te), f"TE on short series returned non-finite: {te}"


if __name__ == "__main__":
    test_independent_series_te_near_zero()
    test_directional_coupling_y_to_x_high()
    test_short_series_does_not_crash()
    print("All KSG TE tests passed.")

    # Diagnostic
    rng = np.random.default_rng(11)
    n = 2000
    y = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + 0.7 * y[t - 1] + 0.3 * rng.standard_normal()
    print(f"  AR coupling Y->X: TE_KSG = {ksg_te(y, x):.4f} nats "
          f"({ksg_te_bits(y, x):.4f} bits)")
    print(f"  AR coupling X->Y: TE_KSG = {ksg_te(x, y):.4f} nats "
          f"(should be near zero)")
