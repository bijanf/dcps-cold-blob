"""Box-mean regrid: ORCA025 tripolar -> regular 2 deg x 2 deg lat/lon.

For our purposes (Kuramoto network on 2-deg cells) we want a simple
area-weighted box average of all native cells whose centre falls inside each
target cell. We use ``scipy.stats.binned_statistic_2d`` for the bin assignment;
weighting is by ORCA cos(lat) (a good approximation of cell area on the
tripolar grid where the full area metric is not stored alongside the surface
fields).
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import binned_statistic_2d

from .config import GRID_DEG, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN


def _target_grid(grid_deg: float = GRID_DEG) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (lat_centres, lon_centres, lat_edges, lon_edges) of the target grid."""
    lat_edges = np.arange(LAT_MIN, LAT_MAX + grid_deg, grid_deg)
    lon_edges = np.arange(LON_MIN, LON_MAX + grid_deg, grid_deg)
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    return lat_c, lon_c, lat_edges, lon_edges


def regrid_to_2deg(
    da: xr.DataArray,
    grid_deg: float = GRID_DEG,
    min_count: int = 4,
) -> xr.DataArray:
    """Box-mean regrid a (time, y, x) ORCA025 DataArray to (time, lat, lon).

    Parameters
    ----------
    da : xarray.DataArray
        Native field with dims (time, y, x) and 2D coords nav_lat, nav_lon.
        NaNs in the source are treated as missing and excluded from each box.
    grid_deg : float
        Target cell size, default 2.
    min_count : int
        Minimum number of valid native points required per (target cell, time)
        before a value is reported. Cells below this count become NaN. Guards
        against thin coastal or land-fragment cells.

    Returns
    -------
    xarray.DataArray
        Shape (time, lat, lon), regular grid, dims and coords as named.
    """
    if "time" not in da.dims:
        raise ValueError("Expected a 'time' dimension on the input DataArray.")

    lat_c, lon_c, lat_edges, lon_edges = _target_grid(grid_deg)
    n_lat = lat_c.size
    n_lon = lon_c.size

    # Flatten the spatial coordinates once. Wrap longitude to [-180, 180).
    nav_lat = da["nav_lat"].values.ravel()
    nav_lon = ((da["nav_lon"].values.ravel() + 180.0) % 360.0) - 180.0

    # Cosine-of-latitude weighting, one number per native cell.
    coslat = np.cos(np.deg2rad(nav_lat))

    # Pre-mask once: only consider native points inside the target grid bbox.
    in_box = (
        (nav_lat >= lat_edges[0])
        & (nav_lat <= lat_edges[-1])
        & (nav_lon >= lon_edges[0])
        & (nav_lon <= lon_edges[-1])
    )

    out = np.full((da.sizes["time"], n_lat, n_lon), np.nan, dtype=np.float32)

    for t_idx in range(da.sizes["time"]):
        flat = da.isel(time=t_idx).values.ravel()
        valid = in_box & np.isfinite(flat)
        if not valid.any():
            continue

        x = nav_lat[valid]
        y = nav_lon[valid]
        v = flat[valid]
        w = coslat[valid]

        # Weighted sum and weight sum per target box; divide for weighted mean.
        wsum, _, _, _ = binned_statistic_2d(
            x, y, w, statistic="sum", bins=[lat_edges, lon_edges]
        )
        wvsum, _, _, _ = binned_statistic_2d(
            x, y, w * v, statistic="sum", bins=[lat_edges, lon_edges]
        )
        cnt, _, _, _ = binned_statistic_2d(
            x, y, np.ones_like(v), statistic="count", bins=[lat_edges, lon_edges]
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(wsum > 0, wvsum / wsum, np.nan)
        mean[cnt < min_count] = np.nan
        out[t_idx, :, :] = mean.astype(np.float32)

    return xr.DataArray(
        out,
        dims=("time", "lat", "lon"),
        coords={
            "time": da["time"].values,
            "lat": lat_c,
            "lon": lon_c,
        },
        name=da.name,
        attrs={
            **(da.attrs or {}),
            "regrid_method": f"area-weighted (cos lat) box mean to {grid_deg} deg",
            "regrid_min_count": int(min_count),
        },
    )
