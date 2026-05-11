"""MAJOR 8 of the peer review: cell-wise atmospheric-forcing regression
with ERA5 surface fluxes.

Reviewer (A9): NAO/AMO are basin-scale indices; 3.8% variance is too
easy to pass.  Proper cell-wise regression against local atmospheric
forcing (turbulent heat flux, wind stress curl, freshwater flux)
would qualify the "dominantly oceanic" claim.

ERA5 monthly NA-basin 2000-2023 already on disk
(data/external/era5/era5_na_monthly_2000_2023.nc).
Variables: sshf (sensible heat flux), slhf (latent heat flux),
ewss (eastward stress), nsss (northward stress), tp
(precipitation), e (evaporation).

We regress local r_loc(t) cell-by-cell against the seven
atmospheric variables (after 1-10 yr bandpass match):
  - turbulent heat flux total (sshf + slhf)
  - wind stress magnitude sqrt(ewss^2 + nsss^2)
  - wind stress curl (computed locally)
  - freshwater flux (tp - e)
Report cell-wise R^2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import (
    BASINS, load_oras5_basin, regrid_basin,
    preprocess_anomaly, instantaneous_phase,
)
# Use the time-resolved r_loc from nao_phase_regression
sys.path.insert(0, str(PKG_ROOT / "scripts"))
from nao_phase_regression import _local_r_with_time


OUT_DIR = CACHE_DIR / "era5_regression"
ERA5_FILE = (PKG_ROOT.parent / "data" / "external" / "era5"
             / "era5_na_monthly_2000_2023.nc")
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"


def bandpass_2d(arr_3d, lo_yr=1.0, hi_yr=10.0, fs_per_yr=12.0, order=4):
    """Apply 1-10 yr Butterworth bandpass to (T, ny, nx) array along
    the time axis.  NaN-safe: cells with insufficient valid points
    return NaN."""
    nyq = 0.5 * fs_per_yr
    b, a = butter(order, [1.0 / hi_yr / nyq, 1.0 / lo_yr / nyq], btype="band")
    T, ny, nx = arr_3d.shape
    flat = arr_3d.reshape(T, -1)
    valid = np.isfinite(flat).all(axis=0)
    out = np.full_like(flat, np.nan)
    if valid.any():
        out[:, valid] = filtfilt(b, a, flat[:, valid], axis=0).astype(arr_3d.dtype)
    return out.reshape(T, ny, nx)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" MAJOR 8: ERA5 cell-wise atmospheric-forcing regression")
    print("=" * 70)

    # 1. Load r_loc(t) for the NA basin on the published 2 deg grid.
    print("\nStep 1: load and recompute r_loc(t) on 2 deg NA grid...")
    sst, lat2d, rlon2d = load_oras5_basin("sosstsst", "atlantic")
    sst_rg = regrid_basin(sst, lat2d, rlon2d, "atlantic")
    sst_anom = preprocess_anomaly(sst_rg)
    phi = instantaneous_phase(sst_anom)
    n_t = phi.sizes["time"]
    phi = phi.isel(time=slice(6, n_t - 6))
    rloc_t = _local_r_with_time(phi, radius_km=500.0)
    print(f"  r_loc(t) shape: {tuple(rloc_t.sizes.values())}")
    # Convert to actual lon
    lon_offset = BASINS["atlantic"]["lon_offset"]
    lat_rg = rloc_t["lat"].values
    rlon_rg = rloc_t["rlon"].values
    actual_lon = ((rlon_rg + lon_offset + 180.0) % 360.0) - 180.0
    # The r_loc grid is (T, lat, rlon).  Reorder lon ascending.
    sort_idx = np.argsort(actual_lon)
    rloc_v = rloc_t.values[:, :, sort_idx]
    lon_sorted = actual_lon[sort_idx]
    lat_sorted = lat_rg

    # 2. Load ERA5; regrid to 2 deg by bilinear interpolation.
    print(f"\nStep 2: load ERA5 from {ERA5_FILE}")
    era = xr.open_dataset(ERA5_FILE)
    # ERA5 latitude is 75 -> 0 (descending); reorder to ascending.
    era = era.reindex(latitude=sorted(era.latitude.values))
    era = era.rename({"valid_time": "time"})
    # Interp to 2 deg NA grid
    era_2deg = era.interp(latitude=lat_sorted, longitude=lon_sorted)
    print(f"  ERA5 regridded shape: {dict(era_2deg.sizes)}")

    # Take only the time range covered by r_loc (it's edge-trimmed by 6)
    rloc_time = rloc_t["time"].values
    era_2deg = era_2deg.reindex(time=rloc_time, method="nearest", tolerance=np.timedelta64(45, "D"))

    # 3. Composite predictors
    print("\nStep 3: assemble atmospheric predictors")
    sshf = era_2deg["sshf"].values.astype(np.float32)
    slhf = era_2deg["slhf"].values.astype(np.float32)
    ewss = era_2deg["ewss"].values.astype(np.float32)
    nsss = era_2deg["nsss"].values.astype(np.float32)
    tp = era_2deg["tp"].values.astype(np.float32)
    evap = era_2deg["e"].values.astype(np.float32)
    qturb = sshf + slhf
    wsmag = np.sqrt(ewss ** 2 + nsss ** 2)
    # Curl ~ dnsss/dx - dewss/dy approximated by finite differences
    dy = np.gradient(nsss, axis=1)
    dx = np.gradient(ewss, axis=2)
    curl = dy - dx
    fw = tp - evap

    # 4. Bandpass everything to 1-10 yr.
    print("Step 4: 1-10 yr Butterworth bandpass on r_loc and atmospheric...")
    rloc_bp = bandpass_2d(rloc_v)
    qturb_bp = bandpass_2d(qturb)
    wsmag_bp = bandpass_2d(wsmag)
    curl_bp = bandpass_2d(curl)
    fw_bp = bandpass_2d(fw)

    # 5. Cell-wise R^2 from OLS regression of r_loc on the four predictors.
    print("\nStep 5: cell-wise OLS regression...")
    T, ny, nx = rloc_bp.shape
    r2_map = np.full((ny, nx), np.nan, dtype=np.float32)
    n_used = 0
    for i in range(ny):
        for j in range(nx):
            y = rloc_bp[:, i, j]
            x1 = qturb_bp[:, i, j]
            x2 = wsmag_bp[:, i, j]
            x3 = curl_bp[:, i, j]
            x4 = fw_bp[:, i, j]
            m = (np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
                 & np.isfinite(x3) & np.isfinite(x4))
            if m.sum() < 30:
                continue
            X = np.column_stack([x1[m], x2[m], x3[m], x4[m]])
            X_c = X - X.mean(axis=0)
            y_c = y[m] - y[m].mean()
            var_y = float(np.sum(y_c ** 2))
            if var_y <= 0:
                continue
            try:
                beta, *_ = np.linalg.lstsq(X_c, y_c, rcond=None)
                resid = y_c - X_c @ beta
                r2 = float(1.0 - np.sum(resid ** 2) / var_y)
                r2_map[i, j] = r2
                n_used += 1
            except np.linalg.LinAlgError:
                continue
    print(f"  cells with valid R^2: {n_used} / {ny*nx}")

    mean_r2 = float(np.nanmean(r2_map))
    p10, p50, p90 = np.nanpercentile(r2_map, [10, 50, 90])
    print(f"  cell-wise R^2: mean = {100*mean_r2:.2f}%   "
          f"p10/50/90 = {100*p10:.1f}/{100*p50:.1f}/{100*p90:.1f}%")

    summary = dict(
        n_cells_valid=int(n_used),
        mean_r2=mean_r2,
        r2_p10=float(p10), r2_p50=float(p50), r2_p90=float(p90),
        predictors=["turbulent heat flux (sshf+slhf)",
                    "wind stress magnitude",
                    "wind stress curl",
                    "freshwater flux (tp - e)"],
        note=("Mean cell-wise R^2 is the percentage of bandpass-1-10-yr "
              "<r_loc>(t) variance jointly explained by four ERA5 surface "
              "atmospheric forcing predictors over the 2000-2023 NA basin."),
    )
    with open(OUT_DIR / "era5_regression.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'era5_regression.json'}")

    # Save the R^2 map
    xr.DataArray(r2_map, dims=("lat", "lon"),
                  coords={"lat": lat_sorted, "lon": lon_sorted},
                  name="R2_atmospheric").to_netcdf(OUT_DIR / "r2_map.nc")

    # Quick figure
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    im = ax.pcolormesh(lon_sorted, lat_sorted, np.clip(r2_map, 0, 0.5),
                          cmap="magma", shading="auto")
    plt.colorbar(im, ax=ax, label=r"$R^{2}$ (capped at 0.5)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.text(0.04, 0.96,
              f"mean $R^2$ = {100*mean_r2:.1f}%\np50 = {100*p50:.1f}%",
              transform=ax.transAxes, fontsize=10, va="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    fig.savefig(MANUSCRIPT_FIGS / "fig_era5_regression.pdf")
    plt.close(fig)
    print(f"Wrote {MANUSCRIPT_FIGS / 'fig_era5_regression.pdf'}")


if __name__ == "__main__":
    main()
