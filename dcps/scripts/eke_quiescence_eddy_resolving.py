"""Eddy-resolving NA validation of the EKE-vs-Quiescence mechanism test.

Pre-registered (commit-anchored):
  Compute geostrophic EKE from GLORYS12 sea-surface-height at *native*
  1/12 deg resolution, box-average to the published 2 deg Quiescence
  grid, and correlate cell-wise against the 2 deg time-mean local
  Kuramoto coherence in the North Atlantic basin.

  Pass condition (locked):
    rho(<r_loc>_t, EKE_eddy_resolving) <= -0.30 (at 2 deg, NA)
      => eddy-resolving validation upheld, report the new value.
    -0.30 < rho <= -0.20  => "consistent with mechanism but not
                              significantly improved by resolution".
    rho > -0.20           => null; frame as future work.

  The scientific question is whether the 2 deg EKE baseline
  (rho = -0.20 in the published Quiescence test) systematically
  underestimates mesoscale variance.  The native 1/12 deg EKE captures
  mesoscale variance directly; box-averaging back to 2 deg preserves
  that mesoscale contribution.
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.signal import hilbert as scipy_hilbert
from scipy.stats import binned_statistic_2d, pearsonr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
from dcps.products import load_glorys12_var
apply_nature_style()

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import (
    BASINS,
    basin_target_grid,
    instantaneous_phase,
    local_r_mean,
    preprocess_anomaly,
)


OUT_DIR = CACHE_DIR / "eke_eddy_resolving"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

GRAVITY = 9.81
OMEGA_EARTH = 7.2921e-5
DEG2M = 111e3
ARGO_START = "2000-01-01"
ARGO_END = "2023-12-31"


def coriolis(lat_deg):
    return 2 * OMEGA_EARTH * np.sin(np.deg2rad(lat_deg))


def native_eke(ssh, lat_dim="y", lon_dim="x"):
    """Time-mean surface geostrophic EKE on the native GLORYS12 grid.

    Streaming over timesteps to keep peak memory bounded to a few
    snapshots worth, not the full (time, y, x) array.

    SSH is in metres on a regular 1/12 deg lat-lon grid, with NA
    basin already masked.  Returns a 2-D map on the same grid.
    Equatorial mask |lat| < 15 deg avoids the 1/f singularity.
    """
    nav_lat = ssh["nav_lat"].values
    nav_lon = ssh["nav_lon"].values

    dlat = float(np.abs(np.diff(np.unique(nav_lat))).mean())
    dlon = float(np.abs(np.diff(np.unique(nav_lon))).mean())
    dy_m = dlat * DEG2M
    dx_m_2d = (dlon * DEG2M * np.cos(np.deg2rad(nav_lat))).astype(np.float32)
    f = coriolis(nav_lat).astype(np.float32)
    inv_f = np.where(np.abs(f) > 1e-8, GRAVITY / f, np.nan).astype(np.float32)
    eq_mask = np.abs(nav_lat) < 15.0

    # First pass: time mean of SSH (streaming)
    print("    pass 1: SSH time mean (streaming)...")
    n_t = ssh.sizes["time"]
    ssh_sum = None
    ssh_cnt = None
    for t in range(n_t):
        s = ssh.isel(time=t).values.astype(np.float32)
        if ssh_sum is None:
            ssh_sum = np.zeros_like(s, dtype=np.float64)
            ssh_cnt = np.zeros_like(s, dtype=np.int32)
        valid = np.isfinite(s)
        ssh_sum[valid] += s[valid]
        ssh_cnt[valid] += 1
        if (t + 1) % 50 == 0:
            print(f"      t={t+1}/{n_t}")
    ssh_mean = np.where(ssh_cnt > 0, ssh_sum / np.maximum(ssh_cnt, 1), np.nan).astype(np.float32)

    # Second pass: per-timestep geostrophic anomaly velocities, accumulate u^2 + v^2
    print("    pass 2: geostrophic velocity anomaly variance (streaming)...")
    sum_eke = np.zeros_like(ssh_mean, dtype=np.float64)
    cnt_eke = np.zeros_like(ssh_mean, dtype=np.int32)
    for t in range(n_t):
        s = ssh.isel(time=t).values.astype(np.float32)
        anom = s - ssh_mean
        u = np.full_like(anom, np.nan)
        v = np.full_like(anom, np.nan)
        u[1:-1, :] = -inv_f[1:-1, :] * (anom[2:, :] - anom[:-2, :]) / (2 * dy_m)
        v[:, 1:-1] = inv_f[:, 1:-1] * (anom[:, 2:] - anom[:, :-2]) / (2 * dx_m_2d[:, 1:-1])
        u[eq_mask] = np.nan
        v[eq_mask] = np.nan
        eke_t = 0.5 * (u * u + v * v)
        valid = np.isfinite(eke_t)
        sum_eke[valid] += eke_t[valid]
        cnt_eke[valid] += 1
        if (t + 1) % 50 == 0:
            print(f"      t={t+1}/{n_t}")
    eke_mean = np.where(cnt_eke > 0, sum_eke / np.maximum(cnt_eke, 1), np.nan).astype(np.float32)

    return xr.DataArray(
        eke_mean, dims=(lat_dim, lon_dim),
        coords={
            "nav_lat": (("y", "x"), nav_lat),
            "nav_lon": (("y", "x"), nav_lon),
        },
        name="EKE_eddy_resolving",
    )


def box_average_to_grid(eke_native, nav_lat, nav_lon, basin="atlantic"):
    """Box-average a native-grid EKE field to the 2 deg multi-basin grid."""
    lat_c, rlon_c, lat_edges, rlon_edges = basin_target_grid(basin)
    # Rotate native lon to basin-relative coords
    lon_offset = BASINS[basin]["lon_offset"]
    rot_lon = (nav_lon - lon_offset) % 360.0

    x = nav_lat.ravel()
    y = rot_lon.ravel()
    v = eke_native.values.ravel()
    valid = np.isfinite(v)
    cnt, _, _, _ = binned_statistic_2d(x[valid], y[valid], np.ones(valid.sum()),
                                          "count", bins=[lat_edges, rlon_edges])
    vsum, _, _, _ = binned_statistic_2d(x[valid], y[valid], v[valid],
                                            "sum", bins=[lat_edges, rlon_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, vsum / cnt, np.nan)
    return xr.DataArray(
        mean.astype(np.float32), dims=("lat", "rlon"),
        coords={"lat": lat_c, "rlon": rlon_c},
        name="EKE_box_avg_2deg",
    )


def regrid_glorys_to_2deg(da, basin="atlantic"):
    """Cos-lat box-mean GLORYS12 native grid -> 2 deg basin grid, per time step."""
    lat_c, rlon_c, lat_edges, rlon_edges = basin_target_grid(basin)
    lon_offset = BASINS[basin]["lon_offset"]
    nav_lat = da["nav_lat"].values
    nav_lon = da["nav_lon"].values
    rot_lon = (nav_lon - lon_offset) % 360.0
    x = nav_lat.ravel()
    y = rot_lon.ravel()
    coslat = np.cos(np.deg2rad(x))
    n_lat, n_rlon = lat_c.size, rlon_c.size
    arr = da.transpose("time", "y", "x").values
    T = arr.shape[0]
    out = np.full((T, n_lat, n_rlon), np.nan, dtype=np.float32)
    for t in range(T):
        flat = arr[t].ravel()
        valid = np.isfinite(flat)
        if not valid.any(): continue
        w = coslat[valid]
        v = flat[valid]
        wsum, _, _, _ = binned_statistic_2d(x[valid], y[valid], w, "sum",
                                                bins=[lat_edges, rlon_edges])
        wvsum, _, _, _ = binned_statistic_2d(x[valid], y[valid], w * v, "sum",
                                                  bins=[lat_edges, rlon_edges])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t] = np.where(wsum > 0, wvsum / wsum, np.nan).astype(np.float32)
    return xr.DataArray(
        out, dims=("time", "lat", "rlon"),
        coords={"time": da["time"].values, "lat": lat_c, "rlon": rlon_c},
        name=da.name,
    )


def main():
    import os
    BASIN = os.environ.get("BASIN", "atlantic")
    if BASIN not in BASINS:
        raise SystemExit(f"BASIN env var '{BASIN}' not in BASINS={list(BASINS)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f" GLORYS12 1/12 deg eddy-resolving validation -- basin={BASIN}")
    print("=" * 70)

    t0 = time.time()
    print(f"\nLoading GLORYS12 SSH (zos) at native 1/12 deg, {ARGO_START}..{ARGO_END}")
    ssh = load_glorys12_var("ssh", start=ARGO_START, end=ARGO_END)
    print(f"  SSH shape: {tuple(ssh.sizes.values())} in {time.time()-t0:.1f}s")

    t0 = time.time()
    print(f"\nLoading GLORYS12 SST (thetao surface) at native 1/12 deg")
    sst = load_glorys12_var("sst", start=ARGO_START, end=ARGO_END)
    print(f"  SST shape: {tuple(sst.sizes.values())} in {time.time()-t0:.1f}s")

    # ------ EKE on native 1/12 deg ---------------------------------------
    t0 = time.time()
    print("\nComputing geostrophic EKE on native 1/12 deg...")
    eke_native = native_eke(ssh)
    print(f"  native EKE shape: {tuple(eke_native.sizes.values())} in {time.time()-t0:.1f}s")

    # Save native EKE for figure inset — basin-suffixed to avoid collisions.
    _suf_native = "" if BASIN == "atlantic" else f"_{BASIN}"
    eke_native.to_netcdf(OUT_DIR / f"eke_native_1_12deg{_suf_native}.nc")
    print(f"  wrote {OUT_DIR / f'eke_native_1_12deg{_suf_native}.nc'}")

    # ------ Box-average native EKE to 2 deg basin grid -------------------
    t0 = time.time()
    print("\nBox-averaging native EKE to 2 deg NA basin grid...")
    nav_lat = ssh["nav_lat"].values
    nav_lon = ssh["nav_lon"].values
    eke_2deg_from_native = box_average_to_grid(eke_native, nav_lat, nav_lon, BASIN)
    print(f"  box-averaged shape: {tuple(eke_2deg_from_native.sizes.values())} in {time.time()-t0:.1f}s")

    # ------ Compute 2 deg r_loc using GLORYS12 SST (self-consistent test) -
    t0 = time.time()
    print("\nRegridding GLORYS12 SST to 2 deg...")
    sst_2deg = regrid_glorys_to_2deg(sst, BASIN)
    print(f"  sst_2deg shape: {tuple(sst_2deg.sizes.values())} in {time.time()-t0:.1f}s")

    print("\nQuiescence pipeline on 2 deg GLORYS12 SST...")
    sst_anom = preprocess_anomaly(sst_2deg)
    phi = instantaneous_phase(sst_anom)
    n_t = phi.sizes["time"]
    phi = phi.isel(time=slice(6, n_t - 6))
    print(f"  Hilbert + edge trim: {phi.sizes['time']} months remain")
    rl_mean = local_r_mean(phi, radius_km=500.0)
    print("  r_loc done")

    # ------ Cell-wise correlation: r_loc vs EKE_eddy_resolving -----------
    a = rl_mean.values.ravel()
    b = eke_2deg_from_native.values.ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    rho, p = pearsonr(a[mask], b[mask])
    n = int(mask.sum())
    print(f"\nrho(<r_loc>_2deg, EKE_1/12deg->2deg) = {rho:+.3f}  p = {p:.2e}  n={n}")

    if rho <= -0.30:
        verdict = "UPHELD (rho <= -0.30)"
    elif rho <= -0.20:
        verdict = "consistent (rho in (-0.30, -0.20])"
    else:
        verdict = "null (rho > -0.20)"
    print(f"  Pre-registered verdict: {verdict}")

    # ------ Save summary --------------------------------------------------
    out = {
        "basin": BASIN,
        "resolution_native_deg": 1.0 / 12.0,
        "resolution_eval_deg": 2.0,
        "rho_eke_eddy_resolving": float(rho),
        "p": float(p),
        "n_cells": n,
        "verdict": verdict,
        "comparison_published_2deg_rho": -0.20,
        "improvement": float(rho - (-0.20)),
        "window": f"{ARGO_START} to {ARGO_END}",
        "data_product": "GLORYS12V1",
    }
    # Basin-suffix the outputs when not running atlantic, so multi-basin runs
    # don't clobber each other.
    suf = "" if BASIN == "atlantic" else f"_{BASIN}"
    with open(OUT_DIR / f"eke_eddy_test{suf}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR / f'eke_eddy_test{suf}.json'}")

    # Save the rl_mean and EKE-2deg maps too
    rl_mean.to_netcdf(OUT_DIR / f"rl_mean_2deg_glorys12{suf}.nc")
    eke_2deg_from_native.to_netcdf(OUT_DIR / f"eke_2deg_from_native{suf}.nc")
    # eke_native_1_12deg already written with basin suffix above (line ~218).

    # ------ Figure: native EKE map + 2 deg scatter -----------------------
    fig = plt.figure(figsize=(11, 4.3), constrained_layout=True)
    ax_map = fig.add_subplot(1, 2, 1)
    im = ax_map.pcolormesh(
        eke_native["nav_lon"].values, eke_native["nav_lat"].values,
        eke_native.values, cmap="magma", shading="auto",
        vmax=np.nanpercentile(eke_native.values, 99),
    )
    plt.colorbar(im, ax=ax_map, label="EKE (m$^2$/s$^2$)")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.text(-0.12, 1.03, "a", transform=ax_map.transAxes,
                  fontweight="bold", fontsize=13)

    ax_sc = fig.add_subplot(1, 2, 2)
    ax_sc.scatter(b[mask], a[mask], s=6, alpha=0.35, color="C0")
    ax_sc.set_xlabel("EKE (1/12$^{\\circ}$ box-avg to 2$^{\\circ}$) (m$^2$/s$^2$)")
    ax_sc.set_ylabel(r"$\langle r_{\mathrm{loc}}\rangle_t$ (2$^{\circ}$)")
    ax_sc.text(0.04, 0.94, f"$\\rho = {rho:+.2f}$\n$n = {n}$",
                  transform=ax_sc.transAxes, fontsize=10)
    ax_sc.text(-0.13, 1.03, "b", transform=ax_sc.transAxes,
                  fontweight="bold", fontsize=13)

    out_fig = MANUSCRIPT_FIGS / "fig_eke_eddy_resolving.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")

    return out


if __name__ == "__main__":
    main()
