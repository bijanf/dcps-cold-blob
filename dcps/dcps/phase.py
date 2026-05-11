"""Hilbert-transform phase extraction per network node.

Given a bandpassed (time, lat, lon) anomaly field, return the analytic-signal
instantaneous amplitude a_i(t) and instantaneous phase phi_i(t), and apply a
6-month edge trim to remove finite-length Hilbert artefacts.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal import hilbert

from .config import EDGE_TRIM_MONTHS


def _hilbert_per_cell(arr: np.ndarray) -> np.ndarray:
    """Apply scipy.signal.hilbert along axis=0 per (lat,lon) cell, NaN-safe.

    Cells whose time series contains any NaN return all-NaN columns; valid cells
    return the analytic signal.
    """
    T = arr.shape[0]
    flat = arr.reshape(T, -1)
    out = np.full(flat.shape, np.nan, dtype=np.complex128)
    valid = np.isfinite(flat).all(axis=0)
    if valid.any():
        out[:, valid] = hilbert(flat[:, valid], axis=0)
    return out.reshape(arr.shape)


def analytic_signal(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return (analytic, amplitude, phase) DataArrays from a bandpassed field.

    Phase is wrapped to (-pi, pi]. Amplitude is non-negative. Land cells stay NaN.
    """
    arr = da.transpose("time", ...).values.astype(np.float64)
    z = _hilbert_per_cell(arr)
    amp = np.abs(z).astype(np.float32)
    phi = np.angle(z).astype(np.float32)

    coords = {k: da.coords[k] for k in da.dims}
    base = {"dims": da.dims, "coords": coords}
    z_da = xr.DataArray(z.astype(np.complex64), name="z", **base, attrs={
        "long_name": "analytic signal",
    })
    a_da = xr.DataArray(amp, name="amp", **base, attrs={
        "long_name": "instantaneous amplitude",
        "units": getattr(da, "units", ""),
    })
    p_da = xr.DataArray(phi, name="phase", **base, attrs={
        "long_name": "instantaneous phase",
        "units": "radians",
        "wrap": "(-pi, pi]",
    })
    return z_da, a_da, p_da


def edge_trim(da: xr.DataArray, n_months: int = EDGE_TRIM_MONTHS) -> xr.DataArray:
    """Drop the first and last ``n_months`` along the time axis."""
    if "time" not in da.dims:
        raise ValueError("Expected a time dim.")
    if 2 * n_months >= da.sizes["time"]:
        raise ValueError(
            f"Trim {n_months} would leave {da.sizes['time'] - 2 * n_months} samples."
        )
    return da.isel(time=slice(n_months, da.sizes["time"] - n_months))
