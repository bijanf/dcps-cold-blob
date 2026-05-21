"""Phase A pilot: Quiescence-corridor detection on ONE CMIP6 model.

For MPI-ESM1-2-LR (default; configurable via --model):

  1. Load piControl tos + zos at Omon resolution via Pangeo intake-esm.
  2. Regrid to the 2-deg NA basin grid (the same grid used in
     multi_basin_quiescence.py).
  3. Split piControl into non-overlapping 30-yr segments; for each
     segment, compute Q = -rho(<r_loc>, |grad SSH|).  Pool to get the
     piControl null distribution of Q.
  4. Load historical + ssp585; on the concatenated record, compute Q
     on sliding 30-yr windows (stride 5 yr).
  5. Detect first window whose Q exceeds the piControl 95-th percentile
     -- the "Holocene exit year" (window midpoint).
  6. Mann-Kendall trend test on the piControl Q sequence (stationarity
     gate; manuscript pre-registration requires p > 0.05).
  7. Output: Q-trajectory plot + JSON summary.

Outputs:
  dcps/cache/holocene_exit/pilot_<model>_<basin>.json
  manuscript/figs/fig_holocene_pilot_<model>_<basin>.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert as scipy_hilbert
from scipy.stats import binned_statistic_2d, pearsonr, kendalltau

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style, SINGLE_COL_IN, DOUBLE_COL_IN
apply_nature_style()

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import (
    BASINS, basin_target_grid,
    instantaneous_phase, local_r_mean,
)


OUT_DIR = CACHE_DIR / "holocene_exit"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

WINDOW_YEARS = 30
PI_SEGMENT_STRIDE_YEARS = 30      # non-overlapping piControl segments
HIST_WINDOW_STRIDE_YEARS = 5      # sliding 30-yr windows in historical+ssp
PI_SPINUP_YEARS = 50              # exclude first 50 yr of piControl
BANDPASS_LO_YR, BANDPASS_HI_YR = 1.0, 10.0
RLOC_RADIUS_KM = 500.0
EXIT_PCTILE = 95.0


def _open_pangeo(model: str, experiment: str, variable: str) -> xr.DataArray:
    """Stream a CMIP6 variable from Pangeo zarr.  Returns a DataArray with
    a 'time' dim and 2-D (lat, lon) coords (CMIP6 ocean grids are often
    curvilinear)."""
    import intake
    cat = intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    rows = cat.df[
        (cat.df.source_id == model)
        & (cat.df.experiment_id == experiment)
        & (cat.df.variable_id == variable)
        & (cat.df.table_id == "Omon")
    ]
    if rows.empty:
        raise FileNotFoundError(
            f"{model} {experiment} {variable} not in Pangeo CMIP6 catalog")
    if "gn" in rows["grid_label"].values:
        rows = rows[rows["grid_label"] == "gn"]
    row = rows.sort_values("member_id").iloc[0]
    zstore = row["zstore"]
    print(f"  open {model} {experiment} {variable}: {zstore[-80:]}")
    ds = xr.open_zarr(zstore, consolidated=True, chunks={})
    return ds[variable]


def _basin_subset_2deg(da: xr.DataArray, basin: str) -> xr.DataArray:
    """Box-average a CMIP6 native (time, y, x) or (time, lat, lon) field
    to the canonical 2-deg basin grid."""
    lat_c, rlon_c, lat_edges, rlon_edges = basin_target_grid(basin)
    lon_offset = BASINS[basin]["lon_offset"]
    lat_name = next((n for n in ("lat", "latitude") if n in da.coords), None)
    lon_name = next((n for n in ("lon", "longitude") if n in da.coords), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"can't find lat/lon in {list(da.coords)}")
    lat2 = da[lat_name].values
    lon2 = da[lon_name].values
    if lat2.ndim == 1:
        LAT2, LON2 = np.meshgrid(lat2, lon2, indexing="ij")
    else:
        LAT2, LON2 = lat2, lon2
    rot_lon = (LON2 - lon_offset) % 360.0
    x = LAT2.ravel(); y = rot_lon.ravel()
    in_box = ((x >= lat_edges[0]) & (x <= lat_edges[-1])
               & (y >= rlon_edges[0]) & (y <= rlon_edges[-1]))
    coslat = np.cos(np.deg2rad(x))

    arr = da.transpose("time", ..., missing_dims="ignore").values
    if arr.ndim == 3:
        T = arr.shape[0]
    else:
        T = 1
        arr = arr[None, ...]
    nL, nR = lat_c.size, rlon_c.size
    out = np.full((T, nL, nR), np.nan, dtype=np.float32)
    for t in range(T):
        flat = arr[t].ravel()
        valid = in_box & np.isfinite(flat)
        if not valid.any():
            continue
        w = coslat[valid]
        v = flat[valid]
        wsum, _, _, _ = binned_statistic_2d(x[valid], y[valid], w, "sum",
                                              bins=[lat_edges, rlon_edges])
        wvsum, _, _, _ = binned_statistic_2d(x[valid], y[valid], w * v, "sum",
                                                bins=[lat_edges, rlon_edges])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t] = np.where(wsum > 0, wvsum / wsum,
                                np.nan).astype(np.float32)
    return xr.DataArray(
        out, dims=("time", "lat", "rlon"),
        coords={"time": da["time"].values[:T], "lat": lat_c, "rlon": rlon_c},
        name=da.name,
    )


def _months_from_time(times) -> np.ndarray:
    """Extract integer month-of-year from a time coord regardless of
    whether it's numpy.datetime64, pandas Timestamp, or cftime."""
    out = np.empty(len(times), dtype=np.int32)
    for i, t in enumerate(times):
        try:
            out[i] = pd.Timestamp(t).month
        except Exception:
            out[i] = int(t.month)
    return out


def _bandpass_anomaly(da: xr.DataArray, lo_yr=BANDPASS_LO_YR,
                       hi_yr=BANDPASS_HI_YR, order=4) -> xr.DataArray:
    """Same recipe as multi_basin_quiescence.preprocess_anomaly, adapted
    to a generic monthly CMIP6 calendar (cftime-safe).  Removes
    climatology, linearly detrends, bandpasses 1-10 yr Butterworth
    forward-backward."""
    months = _months_from_time(da["time"].values)
    da = da.assign_coords(month=("time", months))
    clim = da.groupby("month").mean("time", skipna=True)
    out = (da.groupby("month") - clim).drop_vars("month", errors="ignore")
    # Linear detrend over integer time index (cftime-safe).
    n_t = out.sizes["time"]
    t_idx = np.arange(n_t, dtype=float)
    arr = out.transpose("time", ...).values
    flat = arr.reshape(arr.shape[0], -1)
    valid = np.isfinite(flat).all(axis=0)
    if valid.any():
        # vectorised linear fit per valid column
        A = np.column_stack([t_idx, np.ones_like(t_idx)])
        coeffs, *_ = np.linalg.lstsq(A, flat[:, valid], rcond=None)
        flat[:, valid] -= A @ coeffs
    nyq = 0.5 * 12.0
    f_hi = 1.0 / lo_yr; f_lo = 1.0 / hi_yr
    b, a = butter(order, [f_lo / nyq, f_hi / nyq], btype="band")
    bp = np.full_like(flat, np.nan)
    if valid.any():
        bp[:, valid] = filtfilt(b, a, flat[:, valid],
                                  axis=0).astype(arr.dtype)
    return out.copy(data=bp.reshape(arr.shape).astype(arr.dtype))


def _Q_for_window(tos_win: xr.DataArray, zos_win: xr.DataArray,
                   basin: str) -> tuple[float, int]:
    """Compute Q on a single time window already restricted to that
    window's time range.  Returns (Q, n_cells_used).
    Pipeline: regrid -> tos anomaly+bandpass+Hilbert -> r_loc; zos time
    mean -> |grad SSH|; Pearson on cells; Q = -rho."""
    tos_2d = _basin_subset_2deg(tos_win, basin)
    zos_2d = _basin_subset_2deg(zos_win, basin)
    sst_anom = _bandpass_anomaly(tos_2d)
    phi = instantaneous_phase(sst_anom)
    n_t = phi.sizes["time"]
    if n_t < 12 * WINDOW_YEARS * 0.8:
        return float("nan"), 0
    phi = phi.isel(time=slice(6, n_t - 6))
    rl_mean = local_r_mean(phi, radius_km=RLOC_RADIUS_KM)
    ssh_mean = zos_2d.mean("time")
    grad = np.sqrt(ssh_mean.differentiate("lat") ** 2
                   + ssh_mean.differentiate("rlon") ** 2)
    a = rl_mean.values.ravel()
    b = grad.values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float("nan"), int(m.sum())
    rho, _ = pearsonr(a[m], b[m])
    return float(-rho), int(m.sum())


def _year_of(t) -> int:
    try:
        return int(pd.Timestamp(t).year)
    except Exception:
        try: return int(t.year)
        except Exception: return int(str(t)[:4])


def _slice_by_year(da: xr.DataArray, year_start: int,
                    year_end: int) -> xr.DataArray:
    """Inclusive slice by calendar year (cftime + numpy datetime safe).

    We extract integer years from each time index and select boolean,
    rather than relying on .sel(time=slice(...)) which has corner cases
    in cftime calendars.
    """
    years = np.fromiter(
        (_year_of(t) for t in da["time"].values),
        count=da["time"].size, dtype=np.int32,
    )
    keep = (years >= year_start) & (years <= year_end)
    return da.isel(time=np.where(keep)[0])


def _mann_kendall(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float("nan"), float("nan")
    tau, p = kendalltau(np.arange(x.size), x)
    return float(tau), float(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="MPI-ESM1-2-LR")
    ap.add_argument("--basin", default="atlantic",
                     choices=list(BASINS.keys()))
    ap.add_argument("--n-pi-segments", type=int, default=10,
                     help="how many non-overlapping piControl segments")
    ap.add_argument("--years-hist-start", type=int, default=1850)
    ap.add_argument("--years-hist-end", type=int, default=2014)
    ap.add_argument("--years-ssp-end", type=int, default=2100)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f" Holocene-Q pilot: model={args.model}  basin={args.basin}")
    print("=" * 70)

    # ------------------ piControl null ----------------------------
    t0 = time.time()
    print("\n[piControl] loading tos and zos ...")
    pi_tos = _open_pangeo(args.model, "piControl", "tos")
    pi_zos = _open_pangeo(args.model, "piControl", "zos")
    yr0 = _year_of(pi_tos["time"].values[0])
    yr1 = _year_of(pi_tos["time"].values[-1])
    print(f"  piControl span: {yr0}--{yr1} ({yr1-yr0+1} yr)")

    pi_Q = []
    pi_starts = []
    seg_start = yr0 + PI_SPINUP_YEARS
    while (seg_start + WINDOW_YEARS - 1) <= yr1 and len(pi_Q) < args.n_pi_segments:
        seg_end = seg_start + WINDOW_YEARS - 1
        t1 = time.time()
        tos_seg = _slice_by_year(pi_tos, seg_start, seg_end)
        zos_seg = _slice_by_year(pi_zos, seg_start, seg_end)
        Q, n = _Q_for_window(tos_seg, zos_seg, args.basin)
        pi_Q.append(Q); pi_starts.append(seg_start)
        print(f"  pi segment {seg_start}-{seg_end}: Q={Q:+.3f} n={n} ({time.time()-t1:.0f}s)")
        seg_start += PI_SEGMENT_STRIDE_YEARS

    pi_Q_arr = np.asarray(pi_Q, dtype=float)
    threshold = float(np.nanpercentile(pi_Q_arr, EXIT_PCTILE))
    tau, p_mk = _mann_kendall(pi_Q_arr)
    print(f"  piControl Q mean={np.nanmean(pi_Q_arr):+.3f} sd={np.nanstd(pi_Q_arr):.3f}  "
          f"95th-pctile={threshold:+.3f}  MK tau={tau:+.3f} p={p_mk:.3f}")

    # ------------------ historical + ssp585 ------------------------
    print("\n[historical+ssp585] loading tos and zos ...")
    hi_tos = _open_pangeo(args.model, "historical", "tos")
    hi_zos = _open_pangeo(args.model, "historical", "zos")
    try:
        ssp_tos = _open_pangeo(args.model, "ssp585", "tos")
        ssp_zos = _open_pangeo(args.model, "ssp585", "zos")
        tos_all = xr.concat([hi_tos, ssp_tos], dim="time")
        zos_all = xr.concat([hi_zos, ssp_zos], dim="time")
        end_year = args.years_ssp_end
    except Exception as e:
        print(f"  ssp585 not loaded ({type(e).__name__}); using historical only")
        tos_all = hi_tos; zos_all = hi_zos
        end_year = args.years_hist_end

    hi_Q = []
    hi_centres = []
    win_start = args.years_hist_start
    while win_start + WINDOW_YEARS - 1 <= end_year:
        win_end = win_start + WINDOW_YEARS - 1
        t1 = time.time()
        tos_seg = _slice_by_year(tos_all, win_start, win_end)
        zos_seg = _slice_by_year(zos_all, win_start, win_end)
        Q, n = _Q_for_window(tos_seg, zos_seg, args.basin)
        hi_Q.append(Q); hi_centres.append(win_start + WINDOW_YEARS // 2)
        print(f"  hist window {win_start}-{win_end}: Q={Q:+.3f} n={n} ({time.time()-t1:.0f}s)")
        win_start += HIST_WINDOW_STRIDE_YEARS

    hi_Q_arr = np.asarray(hi_Q, dtype=float)
    centres = np.asarray(hi_centres)

    # First exit: first centre where Q exceeds the 95th-pctile of pi.
    exit_idx = np.where(hi_Q_arr > threshold)[0]
    first_exit_year = int(centres[exit_idx[0]]) if exit_idx.size else None

    summary = dict(
        model=args.model, basin=args.basin,
        window_years=WINDOW_YEARS,
        n_pi_segments=int(len(pi_Q)),
        pi_Q=pi_Q, pi_starts=pi_starts,
        pi_mean=float(np.nanmean(pi_Q_arr)),
        pi_sd=float(np.nanstd(pi_Q_arr)),
        pi_p95_threshold=threshold,
        pi_mk_tau=tau, pi_mk_p=p_mk,
        stationarity_gate_passed=bool(p_mk > 0.05),
        hist_Q=hi_Q, hist_centres=[int(c) for c in centres],
        first_exit_year=first_exit_year,
        exit_pctile=EXIT_PCTILE,
        years_hist_start=args.years_hist_start,
        years_hist_end=args.years_hist_end,
        years_ssp_end=args.years_ssp_end,
    )
    out_json = OUT_DIR / f"pilot_{args.model}_{args.basin}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_json}")
    print(f"  first-exit year for {args.basin} in {args.model}: "
          f"{first_exit_year if first_exit_year is not None else 'never'}")

    # ------------------ figure ------------------------------------
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.4))
    # piControl band as a horizontal envelope across the historical span
    ax.axhspan(np.nanmin(pi_Q_arr), np.nanmax(pi_Q_arr), color="0.85",
                zorder=0, label="piControl Q range")
    ax.axhline(threshold, color="C3", lw=0.8, ls="--",
                label=f"piControl {EXIT_PCTILE:.0f}-th pctile")
    ax.plot(centres, hi_Q_arr, "o-", color="C0", ms=3, lw=1.0,
             label="historical+ssp585 Q")
    if first_exit_year is not None:
        ax.axvline(first_exit_year, color="C3", lw=0.6, ls=":",
                    label=f"first exit {first_exit_year}")
    ax.set_xlabel("Window centre year")
    ax.set_ylabel(r"Quiescence Index  $Q = -\rho$")
    ax.set_xlim(args.years_hist_start, end_year)
    ax.legend(loc="best", frameon=False, fontsize=7)

    out_pdf = (MANUSCRIPT_FIGS
                / f"fig_holocene_pilot_{args.model}_{args.basin}.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
