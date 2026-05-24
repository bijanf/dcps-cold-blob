"""Step 4.1 / B4: Atmospheric Quiescence Signature in midlatitude
Z500.

ERA5 monthly Z500 2000-2023, global 60S to 60N, 1-deg grid (already
fetched: data/external/era5_z500/era5_z500_global_midlat_2000_2023.nc).

For each hemisphere midlatitude band (NH 30-60N, SH 60-30S) we:
  1. Compute monthly Z500 anomalies (climatology removal + detrend).
  2. Bandpass to 2-10 yr (matching the SST pipeline; though for
     synoptic Z500 the relevant band is 2-10 days, we don't have
     daily data, so we apply 1-5 yr on monthly data and report
     honestly).
  3. Compute Hilbert phase per cell.
  4. Compute <r_loc>_t on a 1000-km spatial window (matched to
     synoptic Rossby radius at midlatitudes).
  5. Compute |grad Z500| from the time-mean Z500.
  6. Correlate <r_loc>_t with |grad Z500|; report spatial-block
     permutation p.

The prediction (P1 in the predictions document) is rho < 0:
jet stream cores have high |grad Z500| and should have low local
phase coherence.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert as scipy_hilbert

REPO = Path("/home/bijanf/Documents/NEW_Theory")
sys.path.insert(0, str(REPO / "dcps"))
from dcps.spatial_stats import spatial_block_permutation


ERA5_FILE = REPO / "data/external/era5_z500/era5_z500_global_midlat_2000_2023.nc"
OUT_DIR = REPO / "quiescence_toolkit" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bandpass_2d(arr_3d, lo_yr=1.0, hi_yr=5.0, fs_per_yr=12.0, order=4):
    nyq = 0.5 * fs_per_yr
    b, a = butter(order, [1.0 / hi_yr / nyq, 1.0 / lo_yr / nyq], btype="band")
    T, ny, nx = arr_3d.shape
    flat = arr_3d.reshape(T, -1)
    valid = np.isfinite(flat).all(axis=0)
    out = np.full_like(flat, np.nan)
    if valid.any():
        out[:, valid] = filtfilt(b, a, flat[:, valid], axis=0).astype(arr_3d.dtype)
    return out.reshape(T, ny, nx)


def local_r_2d(phase_3d, lat, lon, radius_km=1000.0, min_n=4):
    T, ny, nx = phase_3d.shape
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    flat_lat = LAT.ravel()
    flat_lon = LON.ravel()
    z = np.exp(1j * phase_3d).reshape(T, -1)
    valid_mask = np.isfinite(phase_3d).reshape(T, -1)
    out = np.full((T, ny * nx), np.nan, dtype=np.float32)
    EARTH_R = 6371.0
    for cell in range(ny * nx):
        if not valid_mask[:, cell].any():
            continue
        lat1r = np.deg2rad(flat_lat[cell])
        lat2r = np.deg2rad(flat_lat)
        dlat = lat2r - lat1r
        dlon = np.deg2rad(flat_lon - flat_lon[cell])
        a = (np.sin(dlat / 2) ** 2
             + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2)
        d = 2 * EARTH_R * np.arcsin(np.sqrt(a))
        idx = np.where(d <= radius_km)[0]
        if idx.size < min_n:
            continue
        z_sum = z[:, idx].sum(axis=1)
        n_valid = valid_mask[:, idx].sum(axis=1)
        ok = n_valid >= min_n
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(ok, np.abs(z_sum) / np.maximum(n_valid, 1), np.nan)
        out[:, cell] = r.astype(np.float32)
    return out.reshape(T, ny, nx).mean(axis=0)


def run_hemisphere(z500, lat_range, lon_range, name):
    print(f"\n  {name} hemisphere: lat {lat_range}, lon {lon_range}")
    lat_arr = z500["latitude"].values
    lon_arr = z500["longitude"].values
    lat_mask = (lat_arr >= lat_range[0]) & (lat_arr <= lat_range[1])
    z = z500.isel(latitude=lat_mask).load()
    lat = z["latitude"].values
    lon = z["longitude"].values
    arr = z.values.astype(np.float64)
    print(f"    shape {arr.shape}")

    # Climatology + detrend
    n_t = arr.shape[0]
    months = np.arange(n_t) % 12
    arr_anom = np.zeros_like(arr)
    for m in range(12):
        m_idx = months == m
        arr_anom[m_idx] = arr[m_idx] - arr[m_idx].mean(axis=0, keepdims=True)
    # Linear detrend
    t = np.arange(n_t).astype(float)
    flat = arr_anom.reshape(n_t, -1)
    for j in range(flat.shape[1]):
        if np.isfinite(flat[:, j]).all():
            coef = np.polyfit(t, flat[:, j], 1)
            flat[:, j] -= np.polyval(coef, t)
    arr_anom = flat.reshape(arr.shape)

    # Bandpass 1-5 yr
    t0 = time.time()
    arr_bp = bandpass_2d(arr_anom, lo_yr=1.0, hi_yr=5.0)
    print(f"    bandpass in {time.time()-t0:.0f}s")

    # Hilbert phase
    t0 = time.time()
    flat = arr_bp.reshape(n_t, -1)
    valid = np.isfinite(flat).all(axis=0)
    phi_flat = np.full(flat.shape, np.nan, dtype=np.float32)
    if valid.any():
        zh = scipy_hilbert(flat[:, valid].astype(np.float64), axis=0)
        phi_flat[:, valid] = np.angle(zh).astype(np.float32)
    phi = phi_flat.reshape(arr.shape)
    # Trim edges (6 months)
    phi = phi[6:-6]
    print(f"    Hilbert in {time.time()-t0:.0f}s")

    # <r_loc>_t
    t0 = time.time()
    rl_mean = local_r_2d(phi, lat, lon, radius_km=1000.0)
    print(f"    <r_loc> in {time.time()-t0:.0f}s; "
          f"range [{np.nanmin(rl_mean):.3f}, {np.nanmax(rl_mean):.3f}]")

    # |grad Z500|
    z500_mean = z.mean("valid_time").values
    grad_lat = np.gradient(z500_mean, axis=0)
    grad_lon = np.gradient(z500_mean, axis=1)
    grad_mag = np.sqrt(grad_lat ** 2 + grad_lon ** 2)

    # Spatial block permutation
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    a = rl_mean.ravel()
    b = grad_mag.ravel()
    result = spatial_block_permutation(a, b, LAT.ravel(), LON.ravel(),
                                            block_km=1000, B=200, seed=42)
    print(f"    rho = {result['rho_observed']:+.3f}  "
          f"n_cells = {result['n_cells']}  "
          f"n_blocks = {result['n_blocks']}  "
          f"p_perm = {result['p_perm']:.4f}")
    return dict(
        hemisphere=name,
        rho=float(result["rho_observed"]),
        p_perm=float(result["p_perm"]),
        n_cells=int(result["n_cells"]),
        n_blocks=int(result["n_blocks"]),
        Q=float(-result["rho_observed"]),
    )


def main():
    print("=" * 70)
    print(" B4: Atmospheric Quiescence Signature in midlatitude Z500")
    print("=" * 70)
    z500 = xr.open_dataset(ERA5_FILE)
    # ERA5 names: variable 'z' (geopotential in m^2/s^2), divide by g for height
    var = z500["z"] / 9.80665
    var = var.squeeze(drop=True)
    print(f"  Z500 dims: {dict(var.sizes)}")

    nh = run_hemisphere(var, (30, 60), (-180, 180), "Northern")
    sh = run_hemisphere(var, (-60, -30), (-180, 180), "Southern")

    summary = dict(NH=nh, SH=sh,
                   prediction="rho < 0 (jet cores: high gradient, low coherence)",
                   verdict_NH=("supports prediction" if nh["rho"] < 0
                                else "does not support"),
                   verdict_SH=("supports prediction" if sh["rho"] < 0
                                else "does not support"))
    with open(OUT_DIR / "atmospheric_quiescence.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'atmospheric_quiescence.json'}")
    print(f"\n  NH verdict: {summary['verdict_NH']}  (rho = {nh['rho']:+.3f})")
    print(f"  SH verdict: {summary['verdict_SH']}  (rho = {sh['rho']:+.3f})")


if __name__ == "__main__":
    main()
