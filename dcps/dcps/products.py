"""Multi-reanalysis loaders for the DCPS pipeline.

Three products are wired up:

    ORAS5     : ECMWF, 1958-2023, ORCA025 tripolar grid, native vars
                sosstsst (SST) + sossheig (SSH).
    GLORYS12  : Copernicus Marine, 1993-2025, regular 1/12 deg lat/lon grid
                (NA-subset already on disk), thetao surface (SST) + zos (SSH).
    ECCO V4r4 : NASA, 1992-2017, regular 0.5 deg lat/lon, THETA surface (SST).
                No SSH variable available in the local cache.

Each loader returns a (time, y, x) DataArray on the *native* grid with 2D
coordinates ``nav_lat``, ``nav_lon`` (broadcast for regular grids) so the
downstream regrid step (``dcps.regrid.regrid_to_2deg``) works unchanged.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import xarray as xr

from .config import (
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    ORAS5_DIR,
    TIME_END,
    TIME_START,
)


ARDP_DATA = Path.home() / "Documents" / "AMOC_renalysis" / "data"
GLORYS12_DIR = ARDP_DATA / "glorys12"
ECCO_DIR = ARDP_DATA / "ecco" / "sal"


# -----------------------------------------------------------------------------
#   Native-to-2D coord helpers
# -----------------------------------------------------------------------------

def _broadcast_2d_coords(lat_1d: np.ndarray, lon_1d: np.ndarray) -> tuple[xr.DataArray, xr.DataArray]:
    """Broadcast 1-D regular lat/lon to 2-D nav_lat/nav_lon DataArrays."""
    LAT, LON = np.meshgrid(lat_1d, lon_1d, indexing="ij")
    return (
        xr.DataArray(LAT.astype(np.float32), dims=("y", "x")),
        xr.DataArray(LON.astype(np.float32), dims=("y", "x")),
    )


def _na_mask_2d(nav_lat: xr.DataArray, nav_lon: xr.DataArray) -> xr.DataArray:
    lon = ((nav_lon + 180.0) % 360.0) - 180.0
    return (
        (nav_lat >= LAT_MIN) & (nav_lat <= LAT_MAX)
        & (lon >= LON_MIN) & (lon <= LON_MAX)
    )


# Basin masks for the multi-basin GLORYS12 EKE pipeline.  Mirrors the
# BASINS registry in ``dcps/scripts/multi_basin_quiescence.py`` but
# expressed as a 2-D mask on (nav_lat, nav_lon) rather than as a
# longitude-rotation rule.
_BASIN_MASK_PARAMS = {
    "atlantic": {"lat": (0.0, 75.0), "lon_offset": -80.0, "lon_extent": 80.0},
    "pacific":  {"lat": (0.0, 60.0), "lon_offset": 130.0, "lon_extent": 110.0},
    "southern": {"lat": (-65.0, -45.0), "lon_offset": 130.0, "lon_extent": 160.0},
}


def _basin_mask_2d(nav_lat: xr.DataArray, nav_lon: xr.DataArray,
                     basin: str) -> xr.DataArray:
    """Boolean mask in (y, x) that selects cells inside ``basin``.

    Uses the same longitude-rotation trick as
    ``multi_basin_quiescence.rotated_lon`` so that basins which cross
    the dateline are contiguous in the rotated coordinate.
    """
    p = _BASIN_MASK_PARAMS[basin]
    lat_min, lat_max = p["lat"]
    rot_lon = (nav_lon - p["lon_offset"]) % 360.0
    return (
        (nav_lat >= lat_min) & (nav_lat <= lat_max)
        & (rot_lon >= 0) & (rot_lon <= p["lon_extent"])
    )


# -----------------------------------------------------------------------------
#   ORAS5  (re-export the existing loader unchanged in spirit)
# -----------------------------------------------------------------------------

def _oras5_files(var_token: str, start: str, end: str) -> list[str]:
    pattern = os.path.join(str(ORAS5_DIR), f"{var_token}_*.nc")
    s = start.replace("-", "")[:6]
    e = end.replace("-", "")[:6]
    out = []
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"_(\d{6})_", os.path.basename(f))
        if m and s <= m.group(1) <= e:
            out.append(f)
    return out


def load_oras5_var(var_token: str, start: str = TIME_START, end: str = TIME_END) -> xr.DataArray:
    files = _oras5_files(var_token, start, end)
    if not files:
        raise FileNotFoundError(f"ORAS5 {var_token!r} {start}..{end}: no files")
    with xr.open_dataset(files[0]) as ds0:
        nav_lat = ds0["nav_lat"].load()
        nav_lon = ds0["nav_lon"].load()

    def _pre(ds):
        return ds[[var_token]].reset_coords(drop=True)

    ds = xr.open_mfdataset(
        files, combine="nested", concat_dim="time_counter",
        preprocess=_pre, parallel=False,
        compat="override", coords="minimal", data_vars="minimal",
    )
    da = ds[var_token].rename({"time_counter": "time"})
    da = da.assign_coords(nav_lat=nav_lat, nav_lon=nav_lon)
    return da.where(_na_mask_2d(nav_lat, nav_lon))


# -----------------------------------------------------------------------------
#   GLORYS12
# -----------------------------------------------------------------------------

GLORYS12_VARS = {"sst": ("thetao", 0), "ssh": ("zos", None)}   # (var, depth_index_or_None)


def load_glorys12_var(
    alias: str, start: str = TIME_START, end: str = TIME_END,
    basin: str = "atlantic",
) -> xr.DataArray:
    """Load GLORYS12 surface SST (thetao depth=0) or SSH (zos), basin-masked.

    The GLORYS12 yearly files store full 3-D thetao; we slice ``depth=0`` for
    SST. ``zos`` is already 2-D.

    ``basin`` selects the geographic mask applied to the returned field.
    Supported values: ``"atlantic"`` (the manuscript NA domain;
    default for back-compat), ``"pacific"``, ``"southern"``.
    """
    if alias not in GLORYS12_VARS:
        raise ValueError(f"GLORYS12 alias {alias!r} not in {list(GLORYS12_VARS)}")
    native, depth_idx = GLORYS12_VARS[alias]

    start_yr = int(start[:4])
    end_yr = int(end[:4])
    files = sorted(glob.glob(str(GLORYS12_DIR / "glorys12_*.nc")))
    files = [f for f in files
             if start_yr <= int(re.search(r"_(\d{4})\.nc$", f).group(1)) <= end_yr]
    if not files:
        raise FileNotFoundError(f"GLORYS12 {alias} {start}..{end}: no files")

    def _pre(ds):
        if depth_idx is None:
            keep = ds[[native]]
        else:
            keep = ds[[native]].isel(depth=depth_idx, drop=True)
        return keep.reset_coords(drop=True)

    ds = xr.open_mfdataset(
        files, combine="nested", concat_dim="time",
        preprocess=_pre, parallel=False,
        compat="override", coords="minimal", data_vars="minimal",
    )
    # Rename to (time, y, x) and attach 2-D nav coords.
    da = ds[native].rename({"latitude": "y", "longitude": "x"})
    da = da.sel(time=slice(start, end))
    with xr.open_dataset(files[0]) as ds0:
        lat_1d = ds0["latitude"].values
        lon_1d = ds0["longitude"].values
    nav_lat, nav_lon = _broadcast_2d_coords(lat_1d, lon_1d)
    da = da.assign_coords(nav_lat=nav_lat, nav_lon=nav_lon)
    if basin == "atlantic":
        mask = _na_mask_2d(nav_lat, nav_lon)
    else:
        mask = _basin_mask_2d(nav_lat, nav_lon, basin)
    return da.where(mask)


# -----------------------------------------------------------------------------
#   ECCO V4r4  (SST only -- no SSH in local cache)
# -----------------------------------------------------------------------------

def _ecco_ts_files(start: str, end: str) -> list[str]:
    """OCEAN_TEMPERATURE_SALINITY monthly files, deduped by year-month."""
    pattern = os.path.join(str(ECCO_DIR), "*", "OCEAN_TEMPERATURE_SALINITY_mon_mean_*.nc")
    s = start.replace("-", "")[:6]
    e = end.replace("-", "")[:6]
    by_yymm: dict[str, str] = {}
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"_(\d{4})-(\d{2})_", os.path.basename(f))
        if m and s <= m.group(1) + m.group(2) <= e:
            ym = m.group(1) + m.group(2)
            by_yymm.setdefault(ym, f)        # keep first occurrence per month
    return [by_yymm[k] for k in sorted(by_yymm)]


def load_ecco_var(alias: str, start: str = TIME_START, end: str = TIME_END) -> xr.DataArray:
    """ECCO V4r4 SST (THETA surface). ``alias`` must be "sst" -- ECCO local
    cache does not include the SSH product."""
    if alias != "sst":
        raise NotImplementedError(f"ECCO local cache lacks {alias}; only 'sst' supported.")
    files = _ecco_ts_files(start, end)
    if not files:
        raise FileNotFoundError(f"ECCO {alias} {start}..{end}: no files")

    def _pre(ds):
        # THETA dim order (time, Z, latitude, longitude); take Z=0 surface
        return ds[["THETA"]].isel(Z=0, drop=True).reset_coords(drop=True)

    ds = xr.open_mfdataset(
        files, combine="nested", concat_dim="time",
        preprocess=_pre, parallel=False,
        compat="override", coords="minimal", data_vars="minimal",
    )
    da = ds["THETA"].rename({"latitude": "y", "longitude": "x"})
    with xr.open_dataset(files[0]) as ds0:
        lat_1d = ds0["latitude"].values
        lon_1d = ds0["longitude"].values
    nav_lat, nav_lon = _broadcast_2d_coords(lat_1d, lon_1d)
    da = da.assign_coords(nav_lat=nav_lat, nav_lon=nav_lon)
    return da.where(_na_mask_2d(nav_lat, nav_lon))


# -----------------------------------------------------------------------------
#   Unified dispatcher
# -----------------------------------------------------------------------------

PRODUCTS = {
    "oras5":   {"loader": load_oras5_var,    "vars": {"sst": "sosstsst", "ssh": "sossheig"}},
    "glorys12":{"loader": load_glorys12_var, "vars": {"sst": "sst", "ssh": "ssh"}},
    "ecco":    {"loader": load_ecco_var,     "vars": {"sst": "sst"}},
}


def load_product_var(
    product: str, alias: str, start: str = TIME_START, end: str = TIME_END,
) -> xr.DataArray:
    if product not in PRODUCTS:
        raise ValueError(f"Unknown product {product!r}; known: {list(PRODUCTS)}")
    p = PRODUCTS[product]
    if alias not in p["vars"]:
        raise ValueError(f"{product} doesn't have {alias!r}; available: {list(p['vars'])}")
    return p["loader"](p["vars"][alias], start=start, end=end)


def product_window(product: str) -> tuple[str, str]:
    """Default analysis window for a product (intersected with config TIME_*)."""
    native_starts = {"oras5": "1958-01-01", "glorys12": "1993-01-01", "ecco": "1992-01-01"}
    native_ends = {"oras5": "2023-12-31", "glorys12": "2024-12-31", "ecco": "2017-12-31"}
    return (max(native_starts[product], TIME_START),
            min(native_ends[product], TIME_END))
