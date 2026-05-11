"""Same-data, same-resolution version of the WOW figure.

Compute the Quiescence pipeline on GLORYS12 1/2 deg regridded SST and
1/2 deg box-averaged 1/12 deg EKE, so both panels of the figure use
the same data product at the same resolution.

The published 2 deg result (rho = -0.35) remains the headline cell-
wise correlation; this 1/2 deg version exists purely to make the
visual comparison apples-to-apples.

Panels:
  a) 1/12 deg native EKE box-averaged to 1/2 deg (NA basin).
  b) 1/2 deg <r_loc> from GLORYS12 SST.
  c) Cell-wise scatter at 1/2 deg.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert as scipy_hilbert
from scipy.stats import binned_statistic_2d, pearsonr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
from dcps.products import load_glorys12_var
apply_nature_style()


OUT_DIR = CACHE_DIR / "eke_eddy_resolving"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

ARGO_START = "2000-01-01"
ARGO_END = "2023-12-31"
TARGET_DEG = 0.5

# NA basin extent (matches multi_basin_quiescence rotated lon)
NA_LAT_MIN, NA_LAT_MAX = 0.0, 75.0
NA_LON_MIN, NA_LON_MAX = -80.0, 0.0   # actual lon, NOT rotated


def regrid_to_half_deg(da):
    """Cos-lat box-mean GLORYS12 -> 1/2 deg NA grid."""
    lat_edges = np.arange(NA_LAT_MIN, NA_LAT_MAX + TARGET_DEG, TARGET_DEG)
    lon_edges = np.arange(NA_LON_MIN, NA_LON_MAX + TARGET_DEG, TARGET_DEG)
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    n_lat, n_lon = lat_c.size, lon_c.size
    nav_lat = da["nav_lat"].values
    nav_lon = da["nav_lon"].values
    x = nav_lat.ravel()
    y = nav_lon.ravel()
    coslat = np.cos(np.deg2rad(x))
    arr = da.transpose("time", "y", "x").values
    T = arr.shape[0]
    out = np.full((T, n_lat, n_lon), np.nan, dtype=np.float32)
    for t in range(T):
        flat = arr[t].ravel()
        valid = np.isfinite(flat)
        if not valid.any(): continue
        wsum, _, _, _ = binned_statistic_2d(
            x[valid], y[valid], coslat[valid], "sum",
            bins=[lat_edges, lon_edges])
        wvsum, _, _, _ = binned_statistic_2d(
            x[valid], y[valid], (coslat * flat)[valid], "sum",
            bins=[lat_edges, lon_edges])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t] = np.where(wsum > 0, wvsum / wsum, np.nan).astype(np.float32)
        if (t + 1) % 50 == 0:
            print(f"      regrid t={t+1}/{T}")
    return xr.DataArray(
        out, dims=("time", "lat", "lon"),
        coords={"time": da["time"].values, "lat": lat_c, "lon": lon_c},
        name=da.name,
    )


def box_average_to_half_deg(eke_native):
    """Box-average native 1/12 deg EKE map to the 1/2 deg target grid."""
    lat_edges = np.arange(NA_LAT_MIN, NA_LAT_MAX + TARGET_DEG, TARGET_DEG)
    lon_edges = np.arange(NA_LON_MIN, NA_LON_MAX + TARGET_DEG, TARGET_DEG)
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    nav_lat = eke_native["nav_lat"].values
    nav_lon = eke_native["nav_lon"].values
    x = nav_lat.ravel(); y = nav_lon.ravel()
    v = eke_native.values.ravel()
    valid = np.isfinite(v)
    cnt, _, _, _ = binned_statistic_2d(
        x[valid], y[valid], np.ones(valid.sum()), "count",
        bins=[lat_edges, lon_edges])
    vsum, _, _, _ = binned_statistic_2d(
        x[valid], y[valid], v[valid], "sum",
        bins=[lat_edges, lon_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, vsum / cnt, np.nan)
    return xr.DataArray(mean.astype(np.float32),
                          dims=("lat", "lon"),
                          coords={"lat": lat_c, "lon": lon_c})


def preprocess_anomaly(da):
    """Climatology, detrend, 1-10 yr Butterworth bandpass."""
    clim = da.groupby("time.month").mean("time", skipna=True)
    out = (da.groupby("time.month") - clim).drop_vars("month", errors="ignore")
    coeffs = out.polyfit(dim="time", deg=1, skipna=True)
    fit = xr.polyval(out["time"], coeffs.polyfit_coefficients)
    out = out - fit
    nyq = 0.5 * 12.0
    b, a = butter(4, [1.0 / 10.0 / nyq, 1.0 / 1.0 / nyq], btype="band")
    arr = out.transpose("time", "lat", "lon").values
    flat = arr.reshape(arr.shape[0], -1)
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


_EARTH_R_KM = 6371.0


def local_r_vectorized(phase, lat_arr, lon_arr, radius_km=500.0,
                          min_neighbours=4):
    """Time-mean local r at every cell.

    Builds a sparse neighbour-set per cell once (haversine in vectorised
    form) and reuses it across all timesteps.  This is what makes the
    1/2 deg native run tractable.
    """
    arr = phase.transpose("time", "lat", "lon").values
    T, ny, nx = arr.shape
    LAT, LON = np.meshgrid(lat_arr, lon_arr, indexing="ij")
    flat_lat = LAT.ravel(); flat_lon = LON.ravel()
    valid_mask = np.isfinite(arr)
    flat_land = ~valid_mask.any(axis=0).ravel()
    z_full = np.where(np.isfinite(arr), np.exp(1j * arr), 0.0 + 0.0j)
    z_flat = z_full.reshape(T, -1)
    valid_flat = valid_mask.reshape(T, -1)
    out = np.full(flat_lat.size, np.nan, dtype=np.float32)
    n_cells = flat_lat.size
    print(f"      cell loop: n={n_cells} (land excluded: {int(flat_land.sum())})")
    t0 = time.time()
    for cell in range(n_cells):
        if flat_land[cell]:
            continue
        lat1r = np.deg2rad(flat_lat[cell])
        lat2r = np.deg2rad(flat_lat)
        dlat = lat2r - lat1r
        dlon = np.deg2rad(flat_lon - flat_lon[cell])
        a = (np.sin(dlat / 2) ** 2
             + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2)
        d = 2 * _EARTH_R_KM * np.arcsin(np.sqrt(a))
        idx = np.where(d <= radius_km)[0]
        if idx.size < min_neighbours:
            continue
        z_sum = z_flat[:, idx].sum(axis=1)
        n_valid = valid_flat[:, idx].sum(axis=1)
        ok = n_valid >= min_neighbours
        with np.errstate(invalid="ignore", divide="ignore"):
            r_t = np.where(ok, np.abs(z_sum) / np.maximum(n_valid, 1), np.nan)
        out[cell] = float(np.nanmean(r_t))
        if (cell + 1) % 1000 == 0:
            print(f"        cell {cell+1}/{n_cells}  "
                  f"({time.time()-t0:.0f}s elapsed)")
    return xr.DataArray(
        out.reshape(ny, nx), dims=("lat", "lon"),
        coords={"lat": lat_arr, "lon": lon_arr},
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading GLORYS12 SST at native 1/12 deg")
    sst = load_glorys12_var("sst", start=ARGO_START, end=ARGO_END)

    print(f"Regridding SST -> {TARGET_DEG} deg")
    sst_target = regrid_to_half_deg(sst)
    del sst

    print("Preprocess anomaly + Hilbert phase")
    sst_anom = preprocess_anomaly(sst_target)
    phi = instantaneous_phase(sst_anom)
    n_t = phi.sizes["time"]
    phi = phi.isel(time=slice(6, n_t - 6))
    print(f"  phi shape: {tuple(phi.sizes.values())}")

    print(f"Computing <r_loc> at {TARGET_DEG} deg, 500-km window")
    rl_mean = local_r_vectorized(phi, sst_target.lat.values,
                                       sst_target.lon.values,
                                       radius_km=500.0)
    rl_mean.to_netcdf(OUT_DIR / f"rl_mean_{int(1/TARGET_DEG)}thdeg.nc")

    # Box-average native 1/12 deg EKE to 1/2 deg
    print("Loading native EKE and box-averaging to 1/2 deg")
    eke_native = xr.open_dataset(
        OUT_DIR / "eke_native_1_12deg.nc")["EKE_eddy_resolving"]
    eke_target = box_average_to_half_deg(eke_native)
    eke_target.to_netcdf(OUT_DIR / f"eke_{int(1/TARGET_DEG)}thdeg.nc")

    # Cell-wise correlation at 1/2 deg
    a = rl_mean.values.ravel()
    b = eke_target.values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    rho, p = pearsonr(a[m], b[m])
    n = int(m.sum())
    print(f"\nrho(<r_loc>_0.5deg, EKE_0.5deg) = {rho:+.3f}  p = {p:.2e}  n={n}")

    out = {
        "target_deg": TARGET_DEG,
        "rho": float(rho), "p": float(p), "n_cells": n,
        "data_product": "GLORYS12V1",
        "window": f"{ARGO_START} to {ARGO_END}",
    }
    with open(OUT_DIR / "eke_eddy_test_halfdeg.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_DIR / 'eke_eddy_test_halfdeg.json'}")

    # -------- WOW figure at matched resolution ---------------------------
    fig = plt.figure(figsize=(14.5, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[3, 0.05, 3, 0.05, 2])
    ax_eke = fig.add_subplot(gs[0, 0])
    cax_eke = fig.add_subplot(gs[0, 1])
    ax_rl = fig.add_subplot(gs[0, 2])
    cax_rl = fig.add_subplot(gs[0, 3])
    ax_sc = fig.add_subplot(gs[0, 4])

    LAT = eke_target["lat"].values
    LON = eke_target["lon"].values

    # Common valid-data mask: a cell is shown only where BOTH EKE and
    # r_loc have valid values.  This makes the "no-data" areas
    # (continents + equatorial band + masked edges) identical in the
    # two spatial panels.
    eke_v = eke_target.values.copy()
    R = rl_mean.values.copy()
    common_valid = np.isfinite(eke_v) & np.isfinite(R)
    eke_v[~common_valid] = np.nan
    R[~common_valid] = np.nan

    extent = [LON.min() - TARGET_DEG/2, LON.max() + TARGET_DEG/2,
              LAT.min() - TARGET_DEG/2, LAT.max() + TARGET_DEG/2]

    # Panel a: 1/2 deg EKE
    eke_lo = max(np.nanpercentile(eke_v, 5), 1e-4)
    eke_hi = np.nanpercentile(eke_v, 99)
    im_eke = ax_eke.imshow(
        eke_v, extent=extent,
        origin="lower", aspect="auto",
        cmap="inferno", norm=LogNorm(vmin=eke_lo, vmax=eke_hi),
        interpolation="bilinear", rasterized=True,
    )
    fig.colorbar(im_eke, cax=cax_eke,
                    label="Geostrophic EKE (m$^{2}$/s$^{2}$)")
    cb_box_a = Rectangle((-50, 45), width=35, height=15, linewidth=2.0,
                            edgecolor="white", facecolor="none",
                            linestyle="--", zorder=10)
    ax_eke.add_patch(cb_box_a)
    ax_eke.set_xlabel("Longitude")
    ax_eke.set_ylabel("Latitude")
    ax_eke.set_xlim(-80, 0)
    ax_eke.set_ylim(0, 75)
    ax_eke.text(-0.10, 1.02, "a", transform=ax_eke.transAxes,
                  fontweight="bold", fontsize=14)

    # Panel b: 1/2 deg r_loc
    im_rl = ax_rl.imshow(
        R, extent=extent,
        origin="lower", aspect="auto",
        cmap="viridis", interpolation="bilinear", rasterized=True,
        vmin=np.nanpercentile(R, 3), vmax=np.nanpercentile(R, 97),
    )
    fig.colorbar(im_rl, cax=cax_rl,
                    label=r"$\langle r_{\mathrm{loc}}\rangle_t$  (phase coherence)")
    cb_box_b = Rectangle((-50, 45), width=35, height=15, linewidth=2.0,
                            edgecolor="white", facecolor="none",
                            linestyle="--", zorder=10)
    ax_rl.add_patch(cb_box_b)
    ax_rl.set_xlabel("Longitude")
    ax_rl.set_ylabel("Latitude")
    ax_rl.set_xlim(-80, 0)
    ax_rl.set_ylim(0, 75)
    ax_rl.text(-0.10, 1.02, "b", transform=ax_rl.transAxes,
                  fontweight="bold", fontsize=14)

    # Panel c: scatter
    ax_sc.scatter(b[m], a[m], s=8, alpha=0.35, color="C0",
                     edgecolors="none")
    ax_sc.set_xscale("log")
    if m.sum() > 30:
        x_pos = b[m][b[m] > 0]
        y_pos = a[m][b[m] > 0]
        coef = np.polyfit(np.log10(x_pos), y_pos, 1)
        xline = np.logspace(np.log10(x_pos.min()), np.log10(x_pos.max()), 50)
        ax_sc.plot(xline, np.polyval(coef, np.log10(xline)),
                       color="C3", linewidth=2.5)
    ax_sc.set_xlabel(f"EKE 1/12$^{{\\circ}}\\rightarrow${int(1/TARGET_DEG)}"
                        "$^{\\circ}$ (m$^{2}$/s$^{2}$)")
    ax_sc.set_ylabel(r"$\langle r_{\mathrm{loc}}\rangle_t$  ("
                        f"1/{int(1/TARGET_DEG)}$^{{\\circ}}$)")
    ax_sc.text(0.04, 0.96,
                  f"$\\rho = {rho:+.3f}$\n$n = {n}$",
                  transform=ax_sc.transAxes, fontsize=10, va="top",
                  bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="white", edgecolor="0.5"))
    ax_sc.text(-0.18, 1.02, "c", transform=ax_sc.transAxes,
                  fontweight="bold", fontsize=14)
    ax_sc.grid(alpha=0.3)

    out_fig = MANUSCRIPT_FIGS / "fig_wow_eddy_quiescence.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
