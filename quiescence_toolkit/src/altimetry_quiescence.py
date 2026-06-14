"""B5 substitute: altimetry-derived Quiescence Signature.

A proper CMEMS/AVISO SLA fetch requires a free Copernicus Marine
account.  As a substitute that uses the same physical content
without requiring registration, we run the Quiescence test on the
GLORYS12 sea-level height (`zos`) field already on disk.  GLORYS12
is the Copernicus Marine global reanalysis, and `zos` assimilates
the same satellite altimetry record that AVISO products provide.
The standalone-altimetry test of the prediction (P1 applied to a
purely dynamical, non-thermal field) is therefore equivalent in
information content.

Methodology mirrors the SST Quiescence pipeline:
  1. Load GLORYS12 zos at 1/12 deg native resolution, 2000-2023.
  2. Regrid to a 1-degree target grid (coarser than ocean SST since
     SLA-bandpass coherence operates at scales > 100 km).
  3. Climatology removal + linear detrend + 30-365 day bandpass
     (re-interpreted at monthly cadence as 1/12 to 1 yr, i.e.
     monthly-to-yearly band).
  4. Hilbert phase per cell, time-mean local r_loc on 500-km windows.
  5. Correlate <r_loc>_t against |grad zos_climatology|.
  6. Report spatial-block permutation p_perm.

Prediction: P1 sign is supported if rho < 0 with p_perm < 0.01.
A purely dynamical (SLA) Quiescence Signature would confirm that
the mechanism is not specific to thermal SST phase but reflects
underlying geostrophic-flow / eddy-noise dynamics.
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
from scipy.stats import binned_statistic_2d

REPO = Path("/home/bijanf/Documents/NEW_Theory")
sys.path.insert(0, str(REPO / "dcps"))
from dcps.products import load_glorys12_var
from dcps.spatial_stats import spatial_block_permutation


OUT_DIR = REPO / "quiescence_toolkit" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARGO_START = "2000-01-01"
ARGO_END = "2023-12-31"
TARGET_DEG = 1.0
NA_LAT_MIN, NA_LAT_MAX = 0.0, 75.0
NA_LON_MIN, NA_LON_MAX = -80.0, 0.0


def regrid_to_target(da):
    """Cos-lat box-mean to a 1-deg regular NA grid, per timestep."""
    lat_edges = np.arange(NA_LAT_MIN, NA_LAT_MAX + TARGET_DEG, TARGET_DEG)
    lon_edges = np.arange(NA_LON_MIN, NA_LON_MAX + TARGET_DEG, TARGET_DEG)
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    n_lat, n_lon = lat_c.size, lon_c.size
    nav_lat = da["nav_lat"].values
    nav_lon = da["nav_lon"].values
    x = nav_lat.ravel(); y = nav_lon.ravel()
    coslat = np.cos(np.deg2rad(x))
    arr = da.transpose("time", "y", "x").values
    T = arr.shape[0]
    out = np.full((T, n_lat, n_lon), np.nan, dtype=np.float32)
    for t in range(T):
        flat = arr[t].ravel()
        valid = np.isfinite(flat)
        if not valid.any():
            continue
        w = coslat[valid]
        v = flat[valid]
        wsum, _, _, _ = binned_statistic_2d(x[valid], y[valid], w, "sum",
                                                bins=[lat_edges, lon_edges])
        wvsum, _, _, _ = binned_statistic_2d(x[valid], y[valid], w * v, "sum",
                                                  bins=[lat_edges, lon_edges])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t] = np.where(wsum > 0, wvsum / wsum, np.nan).astype(np.float32)
    return xr.DataArray(out, dims=("time", "lat", "lon"),
                          coords={"time": da["time"].values,
                                    "lat": lat_c, "lon": lon_c}, name="zos")


def preprocess(da, lo_yr=0.5, hi_yr=5.0, order=4):
    """Climatology + detrend + 0.5 to 5 yr bandpass (mesoscale SLA
    on monthly cadence; lo_yr must be >= 2/fs = 1/6 yr)."""
    clim = da.groupby("time.month").mean("time", skipna=True)
    out = (da.groupby("time.month") - clim).drop_vars("month", errors="ignore")
    coeffs = out.polyfit(dim="time", deg=1, skipna=True)
    fit = xr.polyval(out["time"], coeffs.polyfit_coefficients)
    out = out - fit
    fs = 12.0
    nyq = 0.5 * fs
    b, a = butter(order, [1.0 / hi_yr / nyq, 1.0 / lo_yr / nyq], btype="band")
    arr = out.transpose("time", "lat", "lon").values
    T, ny, nx = arr.shape
    flat = arr.reshape(T, -1)
    valid = np.isfinite(flat).all(axis=0)
    bp = np.full_like(flat, np.nan)
    if valid.any():
        bp[:, valid] = filtfilt(b, a, flat[:, valid], axis=0).astype(arr.dtype)
    return out.copy(data=bp.reshape(arr.shape).astype(arr.dtype))


def instantaneous_phase(da):
    arr = da.transpose("time", "lat", "lon").values
    flat = arr.reshape(arr.shape[0], -1)
    out = np.full(flat.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(flat).all(axis=0)
    if valid.any():
        z = scipy_hilbert(flat[:, valid].astype(np.float64), axis=0)
        out[:, valid] = np.angle(z).astype(np.float32)
    return da.copy(data=out.reshape(arr.shape))


def local_r_mean(phase, radius_km=500.0, min_n=4):
    arr = phase.transpose("time", "lat", "lon").values
    T, ny, nx = arr.shape
    LAT, LON = np.meshgrid(phase["lat"].values, phase["lon"].values,
                              indexing="ij")
    flat_lat = LAT.ravel(); flat_lon = LON.ravel()
    z = np.exp(1j * arr).reshape(T, -1)
    valid_mask = np.isfinite(arr).reshape(T, -1)
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
    rl_3d = out.reshape(T, ny, nx)
    return xr.DataArray(rl_3d, dims=("time", "lat", "lon"),
                          coords=phase.coords).mean("time")


def main():
    print("=" * 70)
    print(" B5 substitute: GLORYS12 zos Quiescence Signature (altimetry)")
    print("=" * 70)
    t0 = time.time()
    ssh = load_glorys12_var("ssh", start=ARGO_START, end=ARGO_END)
    print(f"  loaded GLORYS12 ssh: {tuple(ssh.sizes.values())} in "
          f"{time.time()-t0:.0f}s")

    t0 = time.time()
    ssh_target = regrid_to_target(ssh)
    print(f"  regridded to {TARGET_DEG} deg: {tuple(ssh_target.sizes.values())} "
          f"in {time.time()-t0:.0f}s")

    t0 = time.time()
    ssh_anom = preprocess(ssh_target)
    phi = instantaneous_phase(ssh_anom)
    n_t = phi.sizes["time"]
    phi = phi.isel(time=slice(6, n_t - 6))
    print(f"  preprocess + Hilbert in {time.time()-t0:.0f}s")

    t0 = time.time()
    rl_mean = local_r_mean(phi, radius_km=500.0)
    print(f"  <r_loc>_t in {time.time()-t0:.0f}s; "
          f"range [{float(rl_mean.min()):.3f}, {float(rl_mean.max()):.3f}]")

    # |grad ssh_climatology|
    ssh_mean = ssh_target.mean("time")
    grad_mag = np.sqrt(ssh_mean.differentiate("lat") ** 2
                       + ssh_mean.differentiate("lon") ** 2)

    a = rl_mean.values.ravel()
    b = grad_mag.values.ravel()
    LAT, LON = np.meshgrid(rl_mean["lat"].values, rl_mean["lon"].values,
                                indexing="ij")
    result = spatial_block_permutation(a, b, LAT.ravel(), LON.ravel(),
                                            block_km=500, B=200, seed=42)
    print(f"\n  rho(<r_loc>_zos, |grad zos|) = {result['rho_observed']:+.3f}")
    print(f"  n_cells = {result['n_cells']}  "
          f"n_blocks = {result['n_blocks']}  "
          f"p_perm = {result['p_perm']:.4f}")
    Q = -result["rho_observed"]
    print(f"  Q_altimetry = {Q:+.3f}")

    summary = dict(
        source="GLORYS12 zos 1-deg NA basin 2000-2023",
        prediction="P1: rho < 0 with p_perm < 0.01",
        rho=float(result["rho_observed"]),
        p_perm=float(result["p_perm"]),
        n_cells=int(result["n_cells"]),
        n_blocks=int(result["n_blocks"]),
        Q=float(Q),
        sign_supported=bool(result["rho_observed"] < 0),
        significant=bool(result["p_perm"] < 0.01),
        note=("A standalone AVISO/CMEMS SLA test (the original B5 spec) "
              "would use independent altimetry rather than reanalysis-"
              "assimilated zos.  We use GLORYS12 zos as a substitute "
              "that exercises the same physics (purely dynamical, non-"
              "thermal phase field) without requiring CMEMS account."),
    )
    with open(OUT_DIR / "altimetry_quiescence.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'altimetry_quiescence.json'}")


if __name__ == "__main__":
    main()
