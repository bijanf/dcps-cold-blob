"""Kuramoto order parameters (global and local) on a 2-D phase field.

Global:
    R(t) = | (1/N) sum_j  exp(i phi_j(t)) |

Local (Chimera detection):
    r_loc(x, t) = | (1/|W|) sum_{j in W(x; ell)} exp(i phi_j(t)) |

where W(x; ell) is the set of network nodes within great-circle distance ``ell``
of position x. We pre-compute the neighbour index list once per grid.

All inputs are NaN-aware: land cells (NaN throughout) are excluded both from the
global sum and from each local window. Cells with too few valid neighbours
become NaN in the local field (controlled by ``min_neighbours``).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


# -----------------------------------------------------------------------------
#   Global R(t)
# -----------------------------------------------------------------------------

def global_R(phase: xr.DataArray) -> xr.DataArray:
    """Basin-wide Kuramoto order parameter R(t) from a (time, lat, lon) phase field.

    Land cells (NaN) are excluded from the sum. Output: 1-D DataArray on time.
    """
    arr = phase.transpose("time", ...).values
    z = np.exp(1j * arr)
    z = np.where(np.isfinite(arr), z, 0.0 + 0.0j)
    n_valid = np.isfinite(arr).sum(axis=(1, 2))                # per timestep
    z_sum = z.sum(axis=(1, 2))
    R = np.abs(z_sum) / np.where(n_valid > 0, n_valid, 1)
    R = np.where(n_valid > 0, R, np.nan).astype(np.float32)
    return xr.DataArray(
        R,
        dims=("time",),
        coords={"time": phase["time"].values},
        name="R",
        attrs={
            "long_name": "global Kuramoto order parameter",
            "units": "dimensionless",
            "n_nodes": int(np.isfinite(arr).any(axis=0).sum()),
            "definition": "R(t) = | (1/N) sum_j exp(i phi_j(t)) |",
        },
    )


def global_R_pooled(phase_a: xr.DataArray, phase_b: xr.DataArray) -> xr.DataArray:
    """R(t) computed over both phase fields treated as 2N independent nodes."""
    if phase_a.sizes != phase_b.sizes:
        raise ValueError("phase_a and phase_b must have identical shapes.")
    a = phase_a.transpose("time", ...).values
    b = phase_b.transpose("time", ...).values
    z = np.where(np.isfinite(a), np.exp(1j * a), 0.0 + 0.0j) + \
        np.where(np.isfinite(b), np.exp(1j * b), 0.0 + 0.0j)
    n_valid = np.isfinite(a).sum(axis=(1, 2)) + np.isfinite(b).sum(axis=(1, 2))
    z_sum = z.sum(axis=(1, 2))
    R = np.abs(z_sum) / np.where(n_valid > 0, n_valid, 1)
    R = np.where(n_valid > 0, R, np.nan).astype(np.float32)
    return xr.DataArray(
        R, dims=("time",), coords={"time": phase_a["time"].values}, name="R_pooled",
        attrs={"definition": "R from pooled SST+SSH phase fields, 2N nodes"},
    )


# -----------------------------------------------------------------------------
#   Great-circle distance and neighbour index lookup
# -----------------------------------------------------------------------------

_EARTH_R_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km."""
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlat = lat2r - lat1r
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_R_KM * np.arcsin(np.sqrt(a))


def neighbour_indices(
    lat: np.ndarray, lon: np.ndarray, radius_km: float
) -> list[np.ndarray]:
    """For each cell on a regular (lat, lon) grid, return a flat-index array of
    its neighbours within ``radius_km`` (inclusive). Length == n_lat*n_lon.
    """
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    flat_lat = LAT.ravel()
    flat_lon = LON.ravel()
    out: list[np.ndarray] = []
    for i in range(flat_lat.size):
        d = _haversine_km(flat_lat[i], flat_lon[i], flat_lat, flat_lon)
        out.append(np.where(d <= radius_km)[0].astype(np.int32))
    return out


# -----------------------------------------------------------------------------
#   Local r_loc(x, t)
# -----------------------------------------------------------------------------

def local_r(
    phase: xr.DataArray,
    radius_km: float = 500.0,
    min_neighbours: int = 4,
) -> xr.DataArray:
    """Spatially local Kuramoto OP on a circular window of given radius.

    Returns a (time, lat, lon) field. Cells with fewer than ``min_neighbours``
    *valid* (non-NaN) neighbours at a given timestep become NaN.
    """
    if "time" not in phase.dims or "lat" not in phase.dims or "lon" not in phase.dims:
        raise ValueError("Expected dims (time, lat, lon).")

    lat = phase["lat"].values
    lon = phase["lon"].values
    nbr = neighbour_indices(lat, lon, radius_km)

    # Pre-compute the *land mask* once: a cell is land if it's NaN throughout time.
    arr = phase.transpose("time", "lat", "lon").values    # (T, ny, nx)
    T, ny, nx = arr.shape
    is_land = ~np.isfinite(arr).any(axis=0)                # (ny, nx) — True where land
    flat_land = is_land.ravel()

    # Convert phases -> complex unit vectors once. Treat NaN as zero contribution.
    valid_mask = np.isfinite(arr)                          # (T, ny, nx)
    z_full = np.where(valid_mask, np.exp(1j * arr), 0.0 + 0.0j)
    z_flat = z_full.reshape(T, -1)                         # (T, n_cells)
    valid_flat = valid_mask.reshape(T, -1)                 # bool

    out = np.full((T, ny * nx), np.nan, dtype=np.float32)
    for cell in range(ny * nx):
        if flat_land[cell]:
            continue
        idx = nbr[cell]
        if idx.size < min_neighbours:
            continue
        # Vectorise over time: sum complex unit vectors and valid count per window.
        z_sum = z_flat[:, idx].sum(axis=1)                 # (T,)
        n_valid = valid_flat[:, idx].sum(axis=1)           # (T,)
        ok = n_valid >= min_neighbours
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(ok, np.abs(z_sum) / np.maximum(n_valid, 1), np.nan)
        out[:, cell] = r.astype(np.float32)

    out_3d = out.reshape(T, ny, nx)
    return xr.DataArray(
        out_3d,
        dims=("time", "lat", "lon"),
        coords={"time": phase["time"].values, "lat": lat, "lon": lon},
        name="r_loc",
        attrs={
            "long_name": "local Kuramoto order parameter",
            "units": "dimensionless",
            "radius_km": float(radius_km),
            "min_neighbours": int(min_neighbours),
        },
    )
