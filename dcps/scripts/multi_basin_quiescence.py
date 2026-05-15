"""Multi-basin replication of the H1*-c (Quiescence) spatial result.

Tests whether the negative spatial correlation between time-mean local
Kuramoto coherence and the climatological |grad SSH| holds in two basins
beyond the North Atlantic:
    - North Pacific (Kuroshio extension region)
    - Southern Ocean (Pacific sector, ACC confluence band)

Uses ORAS5 only. Pre-registered: a positive replication is rho <= -0.3,
p < 0.01 in each basin.

Implementation: each basin defines a longitude-offset and an extent so that
the rotated basin lon is contiguous in [0, basin_extent_deg], avoiding any
dateline-wrap headaches in the mask and the target regrid.
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import binned_statistic_2d, pearsonr
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert as scipy_hilbert

from dcps.config import CACHE_DIR, ORAS5_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()


ARGO_START = "2000-01-01"
ARGO_END = "2023-12-31"

MULTI_BASIN_DIR = CACHE_DIR / "multi_basin"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

GRID_DEG = 2.0


# Each basin: lat range and a (lon_offset, lon_extent) pair.
#   The "rotated" longitude r_lon = (nav_lon - lon_offset) mod 360 is in
#   [0, lon_extent] over the basin and outside otherwise.
BASINS = {
    "atlantic": {                      # 80W..0 = lon in [-80, 0]
        "lat": (0.0, 75.0), "lon_offset": -80.0, "lon_extent": 80.0,
        "label": "North Atlantic", "color": "C0",
        "lon_display": (-80, 0),
    },
    "pacific": {                       # 130E..120W (wraps dateline)
        "lat": (0.0, 60.0), "lon_offset": 130.0, "lon_extent": 110.0,
        "label": "North Pacific", "color": "C1",
        "lon_display": (130, -120),    # plot wraps
    },
    "southern": {                      # 130E..70W (wraps dateline) zonal -65..-45
        "lat": (-65.0, -45.0), "lon_offset": 130.0, "lon_extent": 160.0,
        "label": "Southern Ocean (Pacific sector ACC)", "color": "C2",
        "lon_display": (130, -70),
    },
}


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


def rotated_lon(nav_lon_array: np.ndarray, lon_offset: float) -> np.ndarray:
    """Rotate longitude so the basin is contiguous in [0, basin_extent]."""
    return (nav_lon_array - lon_offset) % 360.0


def load_oras5_basin(var_token: str, basin: str,
                      start: str = ARGO_START, end: str = ARGO_END) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """Open ORAS5 monthly files masked to ``basin``.

    Returns (da_native, nav_lat_2d, rotated_nav_lon_2d) where the rotated
    longitude is what we use downstream.
    """
    files = _oras5_files(var_token, start, end)
    if not files:
        raise FileNotFoundError(f"ORAS5 {var_token!r} {start}..{end}: no files")

    with xr.open_dataset(files[0]) as ds0:
        nav_lat = ds0["nav_lat"].values.astype(np.float64)
        nav_lon = ds0["nav_lon"].values.astype(np.float64)

    rot_lon = rotated_lon(nav_lon, BASINS[basin]["lon_offset"])
    lat_min, lat_max = BASINS[basin]["lat"]
    extent = BASINS[basin]["lon_extent"]
    mask = (
        (nav_lat >= lat_min) & (nav_lat <= lat_max)
        & (rot_lon >= 0) & (rot_lon <= extent)
    )

    def _pre(ds):
        return ds[[var_token]].reset_coords(drop=True)

    ds = xr.open_mfdataset(
        files, combine="nested", concat_dim="time_counter",
        preprocess=_pre, parallel=False,
        compat="override", coords="minimal", data_vars="minimal",
    )
    da = ds[var_token].rename({"time_counter": "time"})
    # Apply the basin mask in place: set out-of-basin to NaN
    da = da.where(xr.DataArray(mask, dims=("y", "x")))
    return da, nav_lat, rot_lon


def basin_target_grid(basin: str, grid_deg: float = GRID_DEG):
    """Target regular grid in (lat, rotated_lon) coords."""
    lat_min, lat_max = BASINS[basin]["lat"]
    extent = BASINS[basin]["lon_extent"]
    lat_edges = np.arange(lat_min, lat_max + grid_deg, grid_deg)
    rlon_edges = np.arange(0.0, extent + grid_deg, grid_deg)
    return (
        0.5 * (lat_edges[:-1] + lat_edges[1:]),
        0.5 * (rlon_edges[:-1] + rlon_edges[1:]),
        lat_edges, rlon_edges,
    )


def regrid_basin(da: xr.DataArray, nav_lat_2d: np.ndarray, rot_lon_2d: np.ndarray,
                  basin: str, min_count: int = 4) -> xr.DataArray:
    """Cos-lat-weighted box-mean to a regular (lat, r_lon) target grid."""
    lat_c, rlon_c, lat_edges, rlon_edges = basin_target_grid(basin)
    n_lat, n_rlon = lat_c.size, rlon_c.size

    nav_lat = nav_lat_2d.ravel()
    nav_lon = rot_lon_2d.ravel()
    coslat = np.cos(np.deg2rad(nav_lat))
    in_box = (
        (nav_lat >= lat_edges[0]) & (nav_lat <= lat_edges[-1])
        & (nav_lon >= rlon_edges[0]) & (nav_lon <= rlon_edges[-1])
    )
    out = np.full((da.sizes["time"], n_lat, n_rlon), np.nan, dtype=np.float32)
    for t_idx in range(da.sizes["time"]):
        flat = da.isel(time=t_idx).values.ravel()
        valid = in_box & np.isfinite(flat)
        if not valid.any():
            continue
        x = nav_lat[valid]; y = nav_lon[valid]
        v = flat[valid]; w = coslat[valid]
        wsum, _, _, _ = binned_statistic_2d(x, y, w, "sum", bins=[lat_edges, rlon_edges])
        wvsum, _, _, _ = binned_statistic_2d(x, y, w * v, "sum", bins=[lat_edges, rlon_edges])
        cnt, _, _, _ = binned_statistic_2d(x, y, np.ones_like(v), "count",
                                            bins=[lat_edges, rlon_edges])
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(wsum > 0, wvsum / wsum, np.nan)
        mean[cnt < min_count] = np.nan
        out[t_idx, :, :] = mean.astype(np.float32)
    return xr.DataArray(
        out, dims=("time", "lat", "rlon"),
        coords={"time": da["time"].values, "lat": lat_c, "rlon": rlon_c},
        name=da.name,
    )


def preprocess_anomaly(da: xr.DataArray, lo_yr=1.0, hi_yr=10.0, order=4) -> xr.DataArray:
    clim = da.groupby("time.month").mean("time", skipna=True)
    out = (da.groupby("time.month") - clim).drop_vars("month", errors="ignore")
    coeffs = out.polyfit(dim="time", deg=1, skipna=True)
    fit = xr.polyval(out["time"], coeffs.polyfit_coefficients)
    out = out - fit
    nyq = 0.5 * 12.0
    f_hi = 1.0 / lo_yr; f_lo = 1.0 / hi_yr
    b, a = butter(order, [f_lo / nyq, f_hi / nyq], btype="band")
    arr = out.transpose("time", ...).values
    flat = arr.reshape(arr.shape[0], -1)
    valid = np.isfinite(flat).all(axis=0)
    bp = np.full_like(flat, np.nan)
    if valid.any():
        bp[:, valid] = filtfilt(b, a, flat[:, valid], axis=0).astype(arr.dtype)
    return out.copy(data=bp.reshape(arr.shape).astype(arr.dtype))


def instantaneous_phase(da: xr.DataArray) -> xr.DataArray:
    arr = da.transpose("time", "lat", "rlon").values
    flat = arr.reshape(arr.shape[0], -1)
    out = np.full(flat.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(flat).all(axis=0)
    if valid.any():
        z = scipy_hilbert(flat[:, valid].astype(np.float64), axis=0)
        out[:, valid] = np.angle(z).astype(np.float32)
    return da.copy(data=out.reshape(arr.shape))


from dcps.geo import EARTH_R_KM as _EARTH_R_KM, haversine_km as _haversine_km  # noqa: F401


def local_r_mean(phase: xr.DataArray, radius_km: float = 500.0,
                  min_neighbours: int = 4) -> xr.DataArray:
    arr = phase.transpose("time", "lat", "rlon").values
    T, ny, nx = arr.shape
    LAT, LON = np.meshgrid(phase.lat.values, phase.rlon.values, indexing="ij")
    flat_lat = LAT.ravel(); flat_lon = LON.ravel()
    z_full = np.where(np.isfinite(arr), np.exp(1j * arr), 0.0 + 0.0j)
    valid_mask = np.isfinite(arr)
    z_flat = z_full.reshape(T, -1)
    valid_flat = valid_mask.reshape(T, -1)
    is_land = ~valid_mask.any(axis=0)
    flat_land = is_land.ravel()
    out_t = np.full((T, ny * nx), np.nan, dtype=np.float32)
    for cell in range(ny * nx):
        if flat_land[cell]:
            continue
        d = _haversine_km(flat_lat[cell], flat_lon[cell], flat_lat, flat_lon)
        idx = np.where(d <= radius_km)[0]
        if idx.size < min_neighbours:
            continue
        z_sum = z_flat[:, idx].sum(axis=1)
        n_valid = valid_flat[:, idx].sum(axis=1)
        ok = n_valid >= min_neighbours
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(ok, np.abs(z_sum) / np.maximum(n_valid, 1), np.nan)
        out_t[:, cell] = r.astype(np.float32)
    out_3d = out_t.reshape(T, ny, nx)
    return xr.DataArray(
        out_3d, dims=("time", "lat", "rlon"),
        coords={"time": phase.time.values, "lat": phase.lat.values, "rlon": phase.rlon.values},
    ).mean("time")


def run_basin(basin: str) -> dict:
    print(f"\n{'='*60}\n{basin.upper()}: {BASINS[basin]['label']}\n{'='*60}")
    MULTI_BASIN_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sst, lat2d, rlon2d = load_oras5_basin("sosstsst", basin)
    ssh, _, _ = load_oras5_basin("sossheig", basin)
    print(f"  loaded SST shape={tuple(sst.sizes.values())} in {time.time()-t0:.1f}s")

    t0 = time.time()
    sst_rg = regrid_basin(sst, lat2d, rlon2d, basin)
    ssh_rg = regrid_basin(ssh, lat2d, rlon2d, basin)
    print(f"  regrid -> {tuple(sst_rg.sizes.values())} in {time.time()-t0:.1f}s")

    sst_anom = preprocess_anomaly(sst_rg)
    print("  preprocessed (climatology + detrend + bandpass)")

    phi = instantaneous_phase(sst_anom)
    n_t = phi.sizes["time"]
    phi = phi.isel(time=slice(6, n_t - 6))
    print(f"  Hilbert + edge trim, {phi.sizes['time']} months remain")

    rl_mean = local_r_mean(phi, radius_km=500.0)
    ssh_mean = ssh_rg.mean("time")
    grad_mag = np.sqrt(ssh_mean.differentiate("lat") ** 2
                        + ssh_mean.differentiate("rlon") ** 2)

    a = rl_mean.values.ravel()
    b = grad_mag.values.ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a_v, b_v = a[mask], b[mask]
    rho, p = pearsonr(a_v, b_v)
    print(f"  H1*-c: rho(<r_loc>, |grad SSH|) = {rho:+.3f}  p = {p:.2e}  n_cells={a_v.size}")
    verdict = "SUPPORTED" if (rho <= -0.3 and p < 0.01) else "falsified"
    print(f"  -> {verdict}")
    return {"basin": basin, "rho": float(rho), "p": float(p),
            "n_cells": int(a_v.size), "verdict": verdict,
            "rl_mean": rl_mean, "grad_mag": grad_mag}


def main():
    results: dict[str, dict] = {}
    for b in BASINS:
        try:
            results[b] = run_basin(b)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {b}: FAILED -- {e}")

    # ----- summary --------------------------------------------------------
    print()
    print("=" * 70)
    print(" Multi-basin Quiescence summary (ORAS5, Argo era)")
    print("=" * 70)
    print(f"{'basin':<12} {'rho':>7} {'p':>10} {'n_cells':>8}  {'verdict'}")
    print("-" * 70)
    for b, r in results.items():
        print(f"{b:<12} {r['rho']:+.3f} {r['p']:.2e} {r['n_cells']:>8}  {r['verdict']}")

    out_json = MULTI_BASIN_DIR / "multi_basin_quiescence.json"
    with open(out_json, "w") as f:
        json.dump({b: {"rho": r["rho"], "p": r["p"], "n_cells": r["n_cells"],
                       "verdict": r["verdict"]} for b, r in results.items()},
                  f, indent=2)
    print(f"\nWrote {out_json}")

    # ----- figure --------------------------------------------------------
    fig = plt.figure(figsize=(11, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.3])
    ax_scatter = fig.add_subplot(gs[3, :])

    def _native_tick_label(rlon_value: float, lon_offset: float) -> str:
        native = ((rlon_value + lon_offset + 180) % 360) - 180
        if native >= 0:
            return f"{native:.0f}°E"
        return f"{-native:.0f}°W"

    panel_letters = "abcdefg"
    for row, (b, r) in enumerate(results.items()):
        lon_offset = BASINS[b]["lon_offset"]
        rlon = r["rl_mean"].rlon.values
        rl_plot = r["rl_mean"].values
        grad_plot = r["grad_mag"].values

        ax_rl = fig.add_subplot(gs[row, 0])
        im = ax_rl.pcolormesh(rlon, r["rl_mean"].lat, rl_plot,
                               cmap="RdYlBu_r", vmin=0, vmax=1, shading="auto")
        plt.colorbar(im, ax=ax_rl, label=r"$\langle r_{loc}\rangle_t$")
        ax_rl.set_xlabel("longitude")
        ax_rl.set_ylabel("latitude")
        n_ticks = 6
        tick_rlon = np.linspace(rlon.min(), rlon.max(), n_ticks)
        ax_rl.set_xticks(tick_rlon)
        ax_rl.set_xticklabels([_native_tick_label(t, lon_offset) for t in tick_rlon])
        ax_rl.text(-0.18, 1.02, panel_letters[2 * row],
                    transform=ax_rl.transAxes,
                    fontweight="bold", fontsize=11,
                    verticalalignment="bottom")

        ax_grad = fig.add_subplot(gs[row, 1])
        im = ax_grad.pcolormesh(rlon, r["grad_mag"].lat, grad_plot,
                                 cmap="magma", shading="auto")
        plt.colorbar(im, ax=ax_grad, label=r"$|\nabla\overline{SSH}|$")
        ax_grad.set_xlabel("longitude")
        ax_grad.set_ylabel("latitude")
        ax_grad.set_xticks(tick_rlon)
        ax_grad.set_xticklabels([_native_tick_label(t, lon_offset) for t in tick_rlon])
        ax_grad.text(-0.18, 1.02, panel_letters[2 * row + 1],
                      transform=ax_grad.transAxes,
                      fontweight="bold", fontsize=11,
                      verticalalignment="bottom")

        a = r["rl_mean"].values.ravel()
        bb = r["grad_mag"].values.ravel()
        mask = np.isfinite(a) & np.isfinite(bb)
        ax_scatter.scatter(bb[mask], a[mask], s=4, alpha=0.30,
                            color=BASINS[b]["color"], label=BASINS[b]["label"])

    ax_scatter.set_xlabel(r"$|\nabla \overline{SSH}|$")
    ax_scatter.set_ylabel(r"$\langle r_\mathrm{loc}\rangle_t$")
    ax_scatter.legend(loc="lower left", fontsize=9, frameon=False)
    ax_scatter.grid(alpha=0.25)
    ax_scatter.text(-0.08, 1.02, "g",
                     transform=ax_scatter.transAxes,
                     fontweight="bold", fontsize=11,
                     verticalalignment="bottom")

    out_fig = MANUSCRIPT_FIGS / "fig5_multi_basin_quiescence.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
