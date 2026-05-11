"""Climatology removal, linear detrend, and bandpass filtering for the
DCPS phase-network preprocessing pipeline.

All operations are vectorised over the spatial dims and applied independently
per (lat, lon) cell. Cells that are NaN throughout the record stay NaN.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt

from .config import BANDPASS_HI_YEARS, BANDPASS_LO_YEARS, BUTTER_ORDER


def remove_climatology(da: xr.DataArray) -> xr.DataArray:
    """Subtract the monthly-mean climatology (12 monthly means) over the full record."""
    if "time" not in da.dims:
        raise ValueError("Expected a 'time' dim.")
    clim = da.groupby("time.month").mean("time", skipna=True)
    out = da.groupby("time.month") - clim
    out = out.drop_vars("month", errors="ignore")
    out.attrs["preproc_climatology_removed"] = (
        "monthly climatology computed over full input record"
    )
    return out


def detrend_linear(da: xr.DataArray) -> xr.DataArray:
    """Subtract a least-squares linear trend per spatial cell.

    Implementation note: ``xarray.DataArray.polyfit`` returns the polynomial
    coefficients; we re-evaluate to subtract. Cells whose time series is all-NaN
    keep their NaNs.
    """
    if "time" not in da.dims:
        raise ValueError("Expected a 'time' dim.")
    coeffs = da.polyfit(dim="time", deg=1, skipna=True)
    fit = xr.polyval(da["time"], coeffs.polyfit_coefficients)
    out = da - fit
    out.attrs["preproc_detrend"] = "linear least-squares per cell"
    return out


def bandpass_filter(
    da: xr.DataArray,
    lo_years: float = BANDPASS_LO_YEARS,
    hi_years: float = BANDPASS_HI_YEARS,
    order: int = BUTTER_ORDER,
    fs_per_year: float = 12.0,
) -> xr.DataArray:
    """Forward-backward Butterworth bandpass per cell.

    Pass band: periods between ``lo_years`` and ``hi_years``.
    Sampling rate: monthly (fs = 12 / yr by default).
    Cells whose time series contains any NaN are skipped (kept NaN); we expect
    spatial coverage to be uniform after the regrid step, so this should only
    bite on coastal slivers.
    """
    nyq = 0.5 * fs_per_year                          # cycles per year
    f_hi = 1.0 / lo_years                            # high-frequency corner
    f_lo = 1.0 / hi_years                            # low-frequency corner
    if not (0 < f_lo < f_hi < nyq):
        raise ValueError(
            f"Bad bandpass: f_lo={f_lo}, f_hi={f_hi}, nyq={nyq} (cycles/yr)"
        )
    b, a = butter(order, [f_lo / nyq, f_hi / nyq], btype="band")

    arr = da.transpose("time", ...).values            # (T, lat, lon)
    T = arr.shape[0]
    flat = arr.reshape(T, -1)                         # (T, N)
    out = np.full_like(flat, np.nan)
    valid = np.isfinite(flat).all(axis=0)
    if valid.any():
        # filtfilt requires len(x) > padlen; default padlen ~ 3*max(len(a), len(b)).
        # Our records are >>> filter order, so we don't worry about that here.
        out[:, valid] = filtfilt(b, a, flat[:, valid], axis=0).astype(arr.dtype)

    out_arr = out.reshape(arr.shape)
    da_out = xr.DataArray(
        out_arr,
        dims=da.dims,
        coords=da.coords,
        name=(da.name or "") + "_anom",
        attrs={
            **da.attrs,
            "preproc_bandpass": (
                f"4th-order Butterworth filtfilt, periods {lo_years}-{hi_years} yr"
            ),
        },
    )
    return da_out


def preprocess_pipeline(
    da: xr.DataArray,
    lo_years: float = BANDPASS_LO_YEARS,
    hi_years: float = BANDPASS_HI_YEARS,
) -> xr.DataArray:
    """Full preprocessing chain: climatology -> detrend -> bandpass.

    Returns a new DataArray with the same dims/coords; attributes record the
    operations applied.
    """
    step1 = remove_climatology(da)
    step2 = detrend_linear(step1)
    step3 = bandpass_filter(step2, lo_years=lo_years, hi_years=hi_years)
    step3.attrs["preproc_chain"] = "remove_climatology -> detrend_linear -> bandpass_filter"
    return step3
