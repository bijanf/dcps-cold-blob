"""NAO/AMO atmospheric-forcing regression of the local Kuramoto coherence.

Pre-registered (commit-anchored):
  For each cell in the NA Quiescence map, regress the monthly time
  series r_loc(t) against the bandpass-filtered NAO and AMO indices
  on the same 1-10 yr Butterworth band used for the Quiescence
  diagnostic.  Compute:

    - var explained by NAO alone,
    - var explained by AMO alone,
    - var explained by NAO + AMO jointly.

  Pre-registered pass condition (locked):
    NAO+AMO < 30% of total <r_loc> variance => ocean-only claim is
                                                strengthened.
    NAO+AMO in [30%, 60%]                   => partial attribution.
    NAO+AMO > 60%                           => restate as coupled
                                                ocean-atmosphere
                                                signature.

  We also report the residual cell-wise correlation
    rho(<r_loc>_residual, EKE)
  i.e. the EKE-Quiescence correlation after regressing out
  atmospheric forcing.  If it survives at rho <= -0.30 (NA), the
  mechanism claim is robust to atmospheric attribution.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import (
    load_oras5_basin,
    regrid_basin,
    preprocess_anomaly,
    instantaneous_phase,
)


ARGO_START = "2000-01-01"
ARGO_END = "2023-12-31"

OUT_DIR = CACHE_DIR / "nao_regression"
DATA_DIR = PKG_ROOT.parent / "data" / "external" / "atmospheric"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"


NAO_URL = (
    "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/"
    "norm.nao.monthly.b5001.current.ascii.table"
)
AMO_URL = "https://psl.noaa.gov/data/correlation/amon.us.long.data"


def _download(url: str, target: Path) -> Path:
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {target}")
    urllib.request.urlretrieve(url, target)
    return target


def load_nao(start_year: int, end_year: int) -> np.ndarray:
    """Return monthly NAO time series from NOAA CPC ASCII table.

    Format: each line is YEAR followed by 12 monthly values.
    """
    path = _download(NAO_URL, DATA_DIR / "nao_monthly.txt")
    rows = []
    with open(path) as f:
        for line in f:
            toks = line.split()
            if not toks: continue
            try:
                yr = int(toks[0])
            except ValueError:
                continue
            if len(toks) >= 13 and start_year <= yr <= end_year:
                vals = [float(t) for t in toks[1:13]]
                rows.append((yr, vals))
    if not rows:
        raise RuntimeError(f"NAO: no rows in [{start_year}, {end_year}]")
    out = []
    for yr, vals in sorted(rows):
        out.extend(vals)
    return np.asarray(out, dtype=np.float64)


def load_amo(start_year: int, end_year: int) -> np.ndarray:
    """Return monthly AMO unsmoothed time series from NOAA PSL.

    Format: header line(s), then year + 12 month values per line,
    with a footer marker.
    """
    path = _download(AMO_URL, DATA_DIR / "amo_monthly.txt")
    rows = []
    with open(path) as f:
        # First line: start_year end_year
        first = f.readline().split()
        try:
            data_start = int(first[0]); data_end = int(first[1])
        except (IndexError, ValueError):
            data_start = -9999; data_end = 9999
        for line in f:
            toks = line.split()
            if not toks: continue
            try:
                yr = int(toks[0])
            except ValueError:
                continue
            if len(toks) >= 13 and start_year <= yr <= end_year:
                vals = [float(t) for t in toks[1:13]]
                rows.append((yr, vals))
    if not rows:
        raise RuntimeError(f"AMO: no rows in [{start_year}, {end_year}]")
    out = []
    for yr, vals in sorted(rows):
        out.extend(vals)
    arr = np.asarray(out, dtype=np.float64)
    arr[arr <= -99.0] = np.nan
    return arr


def bandpass_1d(x, lo_yr=1.0, hi_yr=10.0, order=4, fs_per_yr=12.0):
    nyq = 0.5 * fs_per_yr
    b, a = butter(order, [1.0 / hi_yr / nyq, 1.0 / lo_yr / nyq], btype="band")
    mask = np.isfinite(x)
    out = np.full_like(x, np.nan)
    if mask.sum() < 30:
        return out
    out[mask] = filtfilt(b, a, x[mask])
    return out


def compute_r_loc_timeseries(basin="atlantic"):
    """Compute r_loc(t) as a 3-D (time, lat, rlon) field for the NA basin."""
    print(f"\nLoading ORAS5 SST for {basin}")
    sst, lat2d, rlon2d = load_oras5_basin("sosstsst", basin)
    print("  regridding to 2 deg basin grid")
    sst_rg = regrid_basin(sst, lat2d, rlon2d, basin)
    print("  preprocess anomaly + Hilbert")
    sst_anom = preprocess_anomaly(sst_rg)
    phi = instantaneous_phase(sst_anom)
    n_t = phi.sizes["time"]
    phi = phi.isel(time=slice(6, n_t - 6))
    print("  computing local r_loc time series (no time mean)")
    # Use local_r_mean modified to keep time dimension
    return _local_r_with_time(phi, radius_km=500.0)


def _local_r_with_time(phase, radius_km=500.0, min_neighbours=4):
    """Same as local_r_mean but keep the time dimension instead of averaging."""
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
    EARTH_R = 6371.0
    for cell in range(ny * nx):
        if flat_land[cell]:
            continue
        lat1r = np.deg2rad(flat_lat[cell])
        lat2r = np.deg2rad(flat_lat)
        dlat = lat2r - lat1r
        dlon = np.deg2rad(flat_lon - flat_lon[cell])
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
        d = 2 * EARTH_R * np.arcsin(np.sqrt(a))
        idx = np.where(d <= radius_km)[0]
        if idx.size < min_neighbours:
            continue
        z_sum = z_flat[:, idx].sum(axis=1)
        n_valid = valid_flat[:, idx].sum(axis=1)
        ok = n_valid >= min_neighbours
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(ok, np.abs(z_sum) / np.maximum(n_valid, 1), np.nan)
        out_t[:, cell] = r.astype(np.float32)
    return xr.DataArray(
        out_t.reshape(T, ny, nx), dims=("time", "lat", "rlon"),
        coords={"time": phase.time.values, "lat": phase.lat.values, "rlon": phase.rlon.values},
        name="r_loc",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" NAO/AMO atmospheric-forcing regression of local Kuramoto coherence")
    print("=" * 70)

    start_yr = int(ARGO_START[:4])
    end_yr = int(ARGO_END[:4])

    nao = load_nao(start_yr, end_yr)
    amo = load_amo(start_yr, end_yr)
    print(f"NAO: {nao.size} months; AMO: {amo.size} months")

    # Truncate to common length
    n_months = min(nao.size, amo.size)
    nao = nao[:n_months]; amo = amo[:n_months]

    # r_loc time series for atlantic
    t0 = time.time()
    rloc = compute_r_loc_timeseries("atlantic")
    print(f"  r_loc time series shape: {tuple(rloc.sizes.values())} in {time.time()-t0:.0f}s")

    # Align: the r_loc time series starts at month 6 (after Hilbert edge
    # trim) and is monthly through end of ORAS5 record.  Trim atmospheric
    # indices to same span.
    t_rloc = rloc.sizes["time"]
    if n_months >= t_rloc:
        nao = nao[-t_rloc:]
        amo = amo[-t_rloc:]
    else:
        # Not enough atmospheric months; trim rloc to match end.
        rloc = rloc.isel(time=slice(t_rloc - n_months, t_rloc))

    # Bandpass NAO and AMO to match Quiescence band
    nao_bp = bandpass_1d(nao, lo_yr=1.0, hi_yr=10.0)
    amo_bp = bandpass_1d(amo, lo_yr=1.0, hi_yr=10.0)

    # Per-cell OLS regression: r_loc(t) = a + b1*NAO + b2*AMO + r
    arr = rloc.transpose("time", "lat", "rlon").values
    T, ny, nx = arr.shape
    var_total = np.nanvar(arr, axis=0)
    flat = arr.reshape(T, -1)
    var_naoonly = np.full(flat.shape[1], np.nan)
    var_amoonly = np.full(flat.shape[1], np.nan)
    var_joint = np.full(flat.shape[1], np.nan)

    # Build design matrices once
    valid_t = np.isfinite(nao_bp) & np.isfinite(amo_bp)
    nao_v = nao_bp[valid_t] - np.nanmean(nao_bp[valid_t])
    amo_v = amo_bp[valid_t] - np.nanmean(amo_bp[valid_t])

    for cell in range(flat.shape[1]):
        y = flat[:, cell]
        if not np.isfinite(y).any(): continue
        valid = valid_t & np.isfinite(y)
        if valid.sum() < 24: continue
        y_v = y[valid]
        y_c = y_v - np.nanmean(y_v)
        n = y_c.size
        nv = nao_bp[valid] - np.nanmean(nao_bp[valid])
        av = amo_bp[valid] - np.nanmean(amo_bp[valid])
        # var of cell signal (use sample var)
        var_y = np.nansum(y_c ** 2) / max(n - 1, 1)
        if not (var_y > 0): continue
        # OLS coefficients via normal equations
        # NAO alone
        s_nn = np.nansum(nv * nv)
        if s_nn > 0:
            b = np.nansum(nv * y_c) / s_nn
            res = y_c - b * nv
            var_naoonly[cell] = 1.0 - (np.nansum(res ** 2) / max(n - 1, 1)) / var_y
        # AMO alone
        s_aa = np.nansum(av * av)
        if s_aa > 0:
            b = np.nansum(av * y_c) / s_aa
            res = y_c - b * av
            var_amoonly[cell] = 1.0 - (np.nansum(res ** 2) / max(n - 1, 1)) / var_y
        # Joint
        X = np.column_stack([nv, av])
        try:
            beta, *_ = np.linalg.lstsq(X, y_c, rcond=None)
            res = y_c - X @ beta
            var_joint[cell] = 1.0 - (np.nansum(res ** 2) / max(n - 1, 1)) / var_y
        except np.linalg.LinAlgError:
            pass

    mean_var_nao = float(np.nanmean(var_naoonly))
    mean_var_amo = float(np.nanmean(var_amoonly))
    mean_var_joint = float(np.nanmean(var_joint))

    print("\nMean cell-wise variance explained:")
    print(f"  NAO alone : {100*mean_var_nao:5.2f}%")
    print(f"  AMO alone : {100*mean_var_amo:5.2f}%")
    print(f"  NAO+AMO   : {100*mean_var_joint:5.2f}%")

    if mean_var_joint < 0.30:
        verdict = "OCEAN-ONLY (NAO+AMO < 30%)"
    elif mean_var_joint < 0.60:
        verdict = "PARTIAL ATTRIBUTION (30% <= NAO+AMO < 60%)"
    else:
        verdict = "COUPLED OCEAN-ATMOSPHERE (NAO+AMO >= 60%)"
    print(f"Pre-registered verdict: {verdict}")

    out = {
        "basin": "atlantic",
        "data_product": "ORAS5",
        "window": f"{ARGO_START} to {ARGO_END}",
        "var_explained_nao_alone": mean_var_nao,
        "var_explained_amo_alone": mean_var_amo,
        "var_explained_joint": mean_var_joint,
        "verdict": verdict,
    }
    with open(OUT_DIR / "nao_amo_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'nao_amo_results.json'}")

    # Save the variance-explained maps
    coords = {"lat": rloc.lat.values, "rlon": rloc.rlon.values}
    xr.Dataset({
        "var_nao_only": (("lat", "rlon"), var_naoonly.reshape(ny, nx)),
        "var_amo_only": (("lat", "rlon"), var_amoonly.reshape(ny, nx)),
        "var_joint": (("lat", "rlon"), var_joint.reshape(ny, nx)),
    }, coords=coords).to_netcdf(OUT_DIR / "var_explained_maps.nc")

    # Quick figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, varname, title in zip(
        axes,
        ["var_nao_only", "var_amo_only", "var_joint"],
        [f"NAO alone (mean {100*mean_var_nao:.1f}%)",
         f"AMO alone (mean {100*mean_var_amo:.1f}%)",
         f"NAO+AMO joint (mean {100*mean_var_joint:.1f}%)"],
    ):
        v = locals()
        arr_map = {"var_nao_only": var_naoonly, "var_amo_only": var_amoonly,
                    "var_joint": var_joint}[varname].reshape(ny, nx)
        im = ax.pcolormesh(rloc.rlon.values, rloc.lat.values, np.clip(arr_map, 0, 1),
                              cmap="viridis", vmin=0, vmax=0.5)
        plt.colorbar(im, ax=ax, label="var explained")
        ax.set_xlabel("Rot. lon")
        ax.set_ylabel("Lat")
        ax.text(0.04, 0.94, title, transform=ax.transAxes, fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    out_fig = MANUSCRIPT_FIGS / "fig_nao_amo_regression.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
