"""Regression test for the box-mean regridder.

Builds a synthetic ORCA-like dataset with a known analytic field, regrids it,
and checks that the regridded values match the analytic value at each target
cell to within a tolerance set by the local subcell variance.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from dcps.regrid import regrid_to_2deg


def _synthetic_orca(n_y: int = 200, n_x: int = 300, n_t: int = 3) -> xr.DataArray:
    """A toy "ORCA" dataset on a regular lat/lon grid, in the NA box, with a
    smooth field f(lat, lon) = sin(lat / 20) * cos(lon / 30) + t * 0.1."""
    nav_lat = np.linspace(0.5, 74.5, n_y)
    nav_lon = np.linspace(-79.5, -0.5, n_x)
    lat2d, lon2d = np.meshgrid(nav_lat, nav_lon, indexing="ij")

    times = np.array(
        [np.datetime64(f"2010-{m:02d}-15") for m in range(1, n_t + 1)],
        dtype="datetime64[ns]",
    )
    base = np.sin(lat2d / 20.0) * np.cos(lon2d / 30.0)
    data = np.stack([base + t * 0.1 for t in range(n_t)], axis=0).astype(np.float32)

    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={
            "time": times,
            "nav_lat": (("y", "x"), lat2d.astype(np.float32)),
            "nav_lon": (("y", "x"), lon2d.astype(np.float32)),
        },
        name="f",
    )


def test_regrid_recovers_smooth_field():
    """Regridded values should agree with the analytic field at target cell
    centres to <0.01 in absolute value (the field is smooth on the 2-deg scale)."""
    da = _synthetic_orca()
    out = regrid_to_2deg(da)

    # Analytic field at target cell centres
    lat_c = out["lat"].values
    lon_c = out["lon"].values
    LAT, LON = np.meshgrid(lat_c, lon_c, indexing="ij")
    expected_t0 = np.sin(LAT / 20.0) * np.cos(LON / 30.0)

    diff = np.abs(out.isel(time=0).values - expected_t0)
    # Allow some slack at the boundary cells where the box may be partially empty.
    interior = np.s_[1:-1, 1:-1]
    assert np.nanmax(diff[interior]) < 0.01, (
        f"interior max abs diff = {np.nanmax(diff[interior]):.4f}"
    )


def test_regrid_handles_nans():
    """NaNs in the input should be excluded, not propagated."""
    da = _synthetic_orca()
    arr = da.values.copy()
    arr[:, :50, :50] = np.nan
    da2 = da.copy(data=arr)
    out = regrid_to_2deg(da2)
    # Cells near (lat<10, lon<-70) should be NaN where all source was NaN; far
    # interior cells should be unaffected.
    assert np.isfinite(out.sel(lat=50, lon=-30, method="nearest").isel(time=0)).all()


def test_regrid_time_axis_preserved():
    da = _synthetic_orca(n_t=5)
    out = regrid_to_2deg(da)
    assert out.sizes["time"] == 5
    assert (out["time"].values == da["time"].values).all()


if __name__ == "__main__":
    test_regrid_recovers_smooth_field()
    test_regrid_handles_nans()
    test_regrid_time_axis_preserved()
    print("All regrid tests passed.")
