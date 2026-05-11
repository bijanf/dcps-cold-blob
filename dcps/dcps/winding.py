"""Topological-defect diagnostic for a 2-D phase field.

Inspired by Brin & Stuck (2002) Sec. 7.1 (rotation numbers / Poincare
classification) and the standard plaquette technique for detecting phase
dislocations in oscillatory media (e.g. Aranson & Kramer 2002 review).

For each plaquette (i, j) -- the 2 x 2 block of cells (i, j), (i+1, j),
(i+1, j+1), (i, j+1) -- compute the winding number

    W = (1 / 2 pi) sum of wrapped phase differences around the closed loop.

W is exactly an integer in the continuum limit; on a discrete grid it
takes values ~{-1, 0, +1} except where two defects coincide. A non-zero W
indicates that the loop encloses a phase singularity.

The time-mean *defect density* D(x) is the fraction of plaquettes within a
500 km circular window of x that carry |W| >= 0.5, time-averaged. D(x) is
the topological analogue of the (1 - <r_loc>(x)) "incoherence" field, and
predicts a positive correlation with |grad SSH| under the Quiescence
hypothesis (vigorous mean flow seeds defects).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


_EARTH_R_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1r, lat2r = np.deg2rad(lat1), np.deg2rad(lat2)
    dlat = lat2r - lat1r
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_R_KM * np.arcsin(np.sqrt(a))


def _phase_diff_wrapped(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Phase difference b - a wrapped to (-pi, pi] via the complex unit circle."""
    return np.angle(np.exp(1j * (b - a)))


def plaquette_winding(phase: xr.DataArray, lat_dim: str = "lat",
                       lon_dim: str = "rlon") -> xr.DataArray:
    """Winding number on every (i, j) plaquette of a (time, lat, lon) phase field.

    Returns a DataArray with dims (time, ``lat_dim``, ``lon_dim``) and the
    spatial dims shorter by one cell. The plaquette is centred at the
    midpoint of its four corner cells.

    NaNs (land, masked) propagate: any NaN among the four corners gives NaN.
    """
    if "time" not in phase.dims or lat_dim not in phase.dims or lon_dim not in phase.dims:
        raise ValueError(f"Expected dims (time, {lat_dim}, {lon_dim}).")

    p = phase.transpose("time", lat_dim, lon_dim).values     # (T, ny, nx)
    # Counter-clockwise traversal in (lat north, lon east) convention:
    # SW (i, j) -> SE (i, j+1) -> NE (i+1, j+1) -> NW (i+1, j) -> SW.
    a = p[:, :-1, :-1]    # SW
    b = p[:, :-1, 1:]     # SE
    c = p[:, 1:,  1:]     # NE
    d = p[:, 1:,  :-1]    # NW

    valid = (np.isfinite(a) & np.isfinite(b)
             & np.isfinite(c) & np.isfinite(d))
    s = (_phase_diff_wrapped(a, b)
         + _phase_diff_wrapped(b, c)
         + _phase_diff_wrapped(c, d)
         + _phase_diff_wrapped(d, a)) / (2 * np.pi)
    s = np.where(valid, s, np.nan).astype(np.float32)

    lat_centres = 0.5 * (phase[lat_dim].values[:-1] + phase[lat_dim].values[1:])
    lon_centres = 0.5 * (phase[lon_dim].values[:-1] + phase[lon_dim].values[1:])
    return xr.DataArray(
        s, dims=("time", lat_dim, lon_dim),
        coords={"time": phase["time"].values,
                lat_dim: lat_centres, lon_dim: lon_centres},
        name="winding",
        attrs={
            "long_name": "plaquette winding number of the phase field",
            "units": "dimensionless",
            "definition": ("W = (1/2pi) sum of wrapped phase differences "
                           "around a 2x2 plaquette."),
        },
    )


def defect_density_field(winding: xr.DataArray, target_lat: np.ndarray,
                          target_lon: np.ndarray, radius_km: float = 500.0,
                          min_neighbours: int = 4,
                          defect_threshold: float = 0.5,
                          lat_dim: str = "lat",
                          lon_dim: str = "rlon") -> xr.DataArray:
    """Time-mean defect density D(x) = <fraction of plaquettes within ``radius_km``
    of x with |W| >= ``defect_threshold``>_t, evaluated on (target_lat, target_lon).

    The plaquette field has its own (lat, lon) grid (cell midpoints).
    ``target_lat`` and ``target_lon`` are typically the original cell-centre
    grid -- so the output D(x) is on the same grid as <r_loc>(x), which is
    what we correlate against |grad SSH|.
    """
    plaq_lat = winding[lat_dim].values
    plaq_lon = winding[lon_dim].values
    P_LAT, P_LON = np.meshgrid(plaq_lat, plaq_lon, indexing="ij")
    flat_plat = P_LAT.ravel()
    flat_plon = P_LON.ravel()

    arr = winding.transpose("time", lat_dim, lon_dim).values
    T = arr.shape[0]
    flat_w = arr.reshape(T, -1)
    is_defect = (np.abs(flat_w) >= defect_threshold).astype(np.float32)
    is_valid_p = np.isfinite(flat_w).astype(np.float32)

    n_lat, n_lon = target_lat.size, target_lon.size
    out = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    for i, la in enumerate(target_lat):
        for j, lo in enumerate(target_lon):
            d = _haversine_km(la, lo, flat_plat, flat_plon)
            idx = np.where(d <= radius_km)[0]
            if idx.size < min_neighbours:
                continue
            valid_t = is_valid_p[:, idx].sum(axis=1)
            defect_t = is_defect[:, idx].sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                density_t = np.where(valid_t >= min_neighbours,
                                     defect_t / np.maximum(valid_t, 1), np.nan)
            mean_density = np.nanmean(density_t)
            if np.isfinite(mean_density):
                out[i, j] = mean_density

    return xr.DataArray(
        out, dims=(lat_dim, lon_dim),
        coords={lat_dim: target_lat, lon_dim: target_lon},
        name="defect_density",
        attrs={
            "long_name": "time-mean phase-defect density",
            "units": "fraction",
            "radius_km": float(radius_km),
            "defect_threshold": float(defect_threshold),
            "definition": ("D(x) = mean_t [ #{plaquettes in W(x; r) with |W| >= "
                           "thr} / #{valid plaquettes in W(x; r)} ]."),
        },
    )


# ---------------------------------------------------------------------------
#   Self-tests on synthetic phase fields
# ---------------------------------------------------------------------------

def _synthetic_vortex(ny=20, nx=20, charge: int = +1) -> xr.DataArray:
    """A single phase vortex of integer ``charge`` near the grid centre.

    phi(x, y) = charge * atan2(y - y0, x - x0). With even ny/nx the
    singularity falls in the interior of a single plaquette, which must
    therefore carry W = charge; plaquettes far from the singularity have
    W = 0.
    """
    lat = np.linspace(-10, 10, ny)
    lon = np.linspace(-10, 10, nx)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    phi = charge * np.arctan2(LAT, LON)
    arr = phi[None, :, :].astype(np.float32)
    return xr.DataArray(arr, dims=("time", "lat", "rlon"),
                         coords={"time": [0.0], "lat": lat, "rlon": lon})


def _synthetic_smooth(ny=20, nx=20) -> xr.DataArray:
    """A smooth (no-defect) phase field. Every plaquette must have W = 0."""
    lat = np.linspace(-10, 10, ny)
    lon = np.linspace(-10, 10, nx)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    phi = 0.05 * (LAT + LON)
    arr = phi[None, :, :].astype(np.float32)
    return xr.DataArray(arr, dims=("time", "lat", "rlon"),
                         coords={"time": [0.0], "lat": lat, "rlon": lon})


def _self_test():
    """Round-trip checks. Run via:  python -m dcps.winding"""
    phi = _synthetic_vortex(charge=+1)
    W = plaquette_winding(phi)
    w_max = float(np.nanmax(np.abs(W.values)))
    w_corner = float(W.values[0, 0, 0])
    assert 0.9 <= w_max <= 1.1, f"vortex max |W| = {w_max}, expected ~+1"
    assert abs(w_corner) < 0.1, f"vortex far-field W = {w_corner}, expected ~0"
    # And the sign: total winding over the basin (sum of plaquette charges)
    # should equal the topological charge (+1 for charge=+1 input).
    w_sum = float(np.nansum(W.values))
    assert 0.9 <= w_sum <= 1.1, f"total winding sum = {w_sum}, expected +1"

    phi_smooth = _synthetic_smooth()
    W_smooth = plaquette_winding(phi_smooth)
    assert np.nanmax(np.abs(W_smooth.values)) < 0.01, "smooth field has nonzero W"

    print("dcps.winding self-test: vortex max|W| = "
          f"{w_max:.3f}, sum_W = {w_sum:+.3f}, smooth max|W| = "
          f"{float(np.nanmax(np.abs(W_smooth.values))):.3e} -> OK")


if __name__ == "__main__":
    _self_test()
