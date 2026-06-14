"""H1 test: Pearson correlation between standardised RAPID AMOC transport
Psi(t) and the basin-scale Kuramoto order parameter R(t).

Pre-registered decision rule (from Methods):
    H1 supported iff rho >= 0.6 with p < 0.01 over the RAPID overlap.
    0.3 <= rho < 0.6: suggestive.
    rho < 0.3: falsified.

Both signals are low-passed at 1 yr to suppress month-to-month observational
noise before correlation, and significance is assessed with a circular block
bootstrap (block length 12 months) following the Methods specification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr


@dataclass
class H1Result:
    rho: float
    p_value: float
    n_months: int
    bootstrap_low: float    # 2.5% quantile
    bootstrap_high: float   # 97.5% quantile
    threshold_strong: float = 0.6
    threshold_suggestive: float = 0.3
    pass_strong: bool = False
    pass_suggestive: bool = False
    falsified: bool = False
    overlap_start: str = ""
    overlap_end: str = ""


def lowpass_1yr(da: xr.DataArray, cutoff_years: float = 1.0, fs_per_year: float = 12.0,
                 order: int = 4) -> xr.DataArray:
    """Forward-backward Butterworth low-pass at the given cutoff (years)."""
    nyq = 0.5 * fs_per_year
    fc = 1.0 / cutoff_years
    b, a = butter(order, fc / nyq, btype="low")
    out = filtfilt(b, a, da.values.astype(np.float64))
    return da.copy(data=out.astype(da.dtype))


def standardise(da: xr.DataArray) -> xr.DataArray:
    return (da - da.mean()) / da.std()


def circular_block_bootstrap_pearson(
    x: np.ndarray,
    y: np.ndarray,
    block_len: int = 12,
    n_resample: int = 10_000,
    seed: int = 0,
) -> np.ndarray:
    """Distribution of Pearson rho under circular block bootstrap.

    The two arrays are treated as a single bivariate stream; we resample blocks
    of length ``block_len`` (with circular wrap) to preserve serial correlation,
    then compute rho per resample.
    """
    n = x.size
    if y.size != n:
        raise ValueError("x, y length mismatch")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    rhos = np.empty(n_resample, dtype=np.float64)
    for k in range(n_resample):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel() % n
        idx = idx[:n]
        rhos[k] = np.corrcoef(x[idx], y[idx])[0, 1]
    return rhos


def test_h1(
    R: xr.DataArray,
    psi: xr.DataArray,
    block_len: int = 12,
    n_resample: int = 10_000,
    seed: int = 0,
) -> H1Result:
    """Run the H1 correlation test over the RAPID-R overlap window.

    Both inputs are low-passed at 1 yr, standardised, then correlated.
    """
    # Align on year-month period (ORAS5 timestamps are mid-month, RAPID start-of-month)
    R_yymm = np.array([str(t)[:7] for t in R["time"].values])
    P_yymm = np.array([str(t)[:7] for t in psi["time"].values])
    common_yymm = np.intersect1d(R_yymm, P_yymm)
    if common_yymm.size < 24:
        raise ValueError(f"Only {common_yymm.size} months of overlap; H1 not testable.")
    R_idx = np.array([i for i, ym in enumerate(R_yymm) if ym in set(common_yymm)])
    P_idx = np.array([i for i, ym in enumerate(P_yymm) if ym in set(common_yymm)])
    R_o = R.isel(time=R_idx).astype(np.float64)
    P_o = psi.isel(time=P_idx).astype(np.float64)

    R_lp = lowpass_1yr(R_o)
    P_lp = lowpass_1yr(P_o)

    R_z = standardise(R_lp).values
    P_z = standardise(P_lp).values

    rho, p = pearsonr(R_z, P_z)
    rhos = circular_block_bootstrap_pearson(R_z, P_z, block_len, n_resample, seed)
    lo, hi = np.quantile(rhos, [0.025, 0.975])

    res = H1Result(
        rho=float(rho),
        p_value=float(p),
        n_months=int(common_yymm.size),
        bootstrap_low=float(lo),
        bootstrap_high=float(hi),
        overlap_start=str(min(common_yymm)),
        overlap_end=str(max(common_yymm)),
    )
    res.pass_strong = (res.rho >= res.threshold_strong) and (res.p_value < 0.01)
    res.pass_suggestive = (res.threshold_suggestive <= res.rho < res.threshold_strong) \
        or (res.rho >= res.threshold_strong and res.p_value >= 0.01)
    res.falsified = res.rho < res.threshold_suggestive
    return res
