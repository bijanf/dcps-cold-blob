"""Loaders for ORAS5 monthly reanalysis files and the RAPID 26.5 N record."""

from __future__ import annotations

import glob
import os
import re

import xarray as xr

from .config import (
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    ORAS5_DIR,
    RAPID_FILE,
    TIME_END,
    TIME_START,
)


def _files_in_window(var_token: str, start: str, end: str) -> list[str]:
    """Sorted list of ORAS5 monthly files whose YYYYMM stamp falls in [start, end]."""
    pattern = os.path.join(str(ORAS5_DIR), f"{var_token}_*.nc")
    out = []
    s = start.replace("-", "")[:6]
    e = end.replace("-", "")[:6]
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"_(\d{6})_", os.path.basename(f))
        if m and s <= m.group(1) <= e:
            out.append(f)
    return out


def _na_mask(nav_lat: xr.DataArray, nav_lon: xr.DataArray) -> xr.DataArray:
    """Boolean mask selecting the North Atlantic basin on the 2D ORCA grid."""
    lon = ((nav_lon + 180.0) % 360.0) - 180.0   # wrap to [-180, 180)
    return (
        (nav_lat >= LAT_MIN)
        & (nav_lat <= LAT_MAX)
        & (lon >= LON_MIN)
        & (lon <= LON_MAX)
    )


def load_oras5_var(
    var_token: str,
    start: str = TIME_START,
    end: str = TIME_END,
) -> xr.DataArray:
    """Open all monthly ORAS5 files for one variable in [start, end] as one DataArray.

    Returns a (time, y, x) array on the native ORCA025 tripolar grid, masked to the
    North Atlantic basin. Out-of-basin cells are NaN. The 2D coordinates `nav_lat`
    and `nav_lon` are preserved (taken from the first file; identical across files).

    Implementation note: at >~120 files, ``open_mfdataset`` with default merge
    settings tries to align the 2D nav_lat/nav_lon across files and explodes the
    dask graph to a 4D (time, y, x, time) tensor. We explicitly drop nav coords
    from the per-file datasets, concatenate cleanly, and reattach the nav coords
    once at the end.
    """
    files = _files_in_window(var_token, start, end)
    if not files:
        raise FileNotFoundError(f"No ORAS5 files matched {var_token!r} in {start}..{end}")

    # Read nav coords once from the first file (identical across all files).
    with xr.open_dataset(files[0]) as ds0:
        nav_lat = ds0["nav_lat"].load()
        nav_lon = ds0["nav_lon"].load()

    def _preprocess(ds: xr.Dataset) -> xr.Dataset:
        # Keep only the data variable and time coord; drop nav coords to avoid
        # the multi-file merge blow-up.
        return ds[[var_token]].reset_coords(drop=True)

    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time_counter",
        preprocess=_preprocess,
        parallel=False,
        compat="override",
        coords="minimal",
        data_vars="minimal",
    )

    da = ds[var_token].rename({"time_counter": "time"})
    da = da.assign_coords(nav_lat=nav_lat, nav_lon=nav_lon)

    # Mask to NA basin (uses nav coords; broadcasts cleanly because they're 2D).
    mask = _na_mask(nav_lat, nav_lon)
    da = da.where(mask)
    return da


def load_rapid_amoc(start: str | None = None, end: str | None = None) -> xr.DataArray:
    """Return RAPID-MOCHA 26.5 N AMOC transport (Sv) as a (time,) DataArray.

    Variable extracted: `moc_mar_hc10` (total overturning, monthly mean).
    """
    ds = xr.open_dataset(RAPID_FILE)
    da = ds["moc_mar_hc10"]
    da.attrs.setdefault("units", "Sv")
    da.attrs.setdefault("long_name", "RAPID 26.5N AMOC transport")
    if start is not None or end is not None:
        da = da.sel(time=slice(start, end))
    return da


def list_oras5_coverage(var_token: str) -> dict[str, int]:
    """{year: month_count} for diagnostic / gap-detection use."""
    files = sorted(glob.glob(os.path.join(str(ORAS5_DIR), f"{var_token}_*.nc")))
    out: dict[str, int] = {}
    for f in files:
        m = re.search(r"_(\d{4})\d{2}_", os.path.basename(f))
        if m:
            out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out
