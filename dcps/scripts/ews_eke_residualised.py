"""EKE-residualised critical-slowing-down indicators on the Caesar AMOC SST
fingerprint.

The published surface-fingerprint AMOC early-warning literature
(Caesar et al. 2018, 2021; Boers 2021; Ditlevsen & Ditlevsen 2023)
reads the subpolar North Atlantic SST anomaly as a thermohaline
weakening signal and reports rising lag-1 autocorrelation and
rising variance as critical-slowing-down (CSD) precursors.

This study identifies a stationary spatial scaling whereby a
non-trivial part of the subpolar SST anomaly is the low-EKE half
of a basin-scale phase-coherence contrast.  If that stationary
mesoscale contribution is folded into the fingerprint, the
slopes of the CSD indicators are partly diagnosing the
EKE-suppression pattern, not pure AMOC thermohaline weakening.

This script delivers the operational correction:

  1. Build the Caesar-style fingerprint F(t) from HadISST monthly
     SST: annual-mean subpolar SST anomaly (45-60 N, 50-15 W)
     minus NH-mean SST anomaly (0-60 N).
  2. Build an EKE-weighted basin temperature index W(t):
     EKE-weighted spatial mean of the SST anomaly field on the
     2-deg Atlantic basin grid; large EKE cells (Gulf Stream / NAC
     corridor) dominate W(t).
  3. Residualise: F_res(t) = F(t) - beta * W(t) where beta is the
     linear-regression coefficient of F on W (least squares).
  4. Compute lag-1 autocorrelation and variance of F and F_res in
     50-yr sliding windows.
  5. Fit the linear trend in each CSD indicator over 1900--2020.
  6. Report (a) the slope of each indicator and (b) the change in
     slope after residualisation; output JSON + figure following
     the project's figure-quality standard.

Inputs and outputs are CLI-flagged so the script runs portably.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "dcps" / "scripts"))

# --- domain definitions --------------------------------------------------
SUBPOLAR_BOX = dict(lat=(45.0, 60.0), lon=(-50.0, -15.0))
NH_BOX = dict(lat=(0.0, 60.0), lon=(-180.0, 180.0))
WINDOW_YEARS = 50
CSD_FIT_START = 1900
CSD_FIT_END = 2020


@dataclass
class CSDResult:
    n_years: int
    n_windows: int
    lag1_slope_raw: float
    lag1_slope_residualised: float
    var_slope_raw: float
    var_slope_residualised: float
    lag1_slope_change_frac: float
    var_slope_change_frac: float
    beta_eke_weight: float
    rho_F_W: float

    def to_json(self) -> dict:
        return asdict(self)


def annual_mean(da: xr.DataArray) -> xr.DataArray:
    return da.groupby("time.year").mean("time")


def box_mean(da: xr.DataArray, lat_range, lon_range,
             lat_name="latitude", lon_name="longitude") -> xr.DataArray:
    lat_lo, lat_hi = sorted(lat_range)
    lon_lo, lon_hi = sorted(lon_range)
    sub = da.sel({lat_name: slice(lat_hi, lat_lo)}) if da[lat_name][0] > da[lat_name][-1] \
        else da.sel({lat_name: slice(lat_lo, lat_hi)})
    sub = sub.sel({lon_name: slice(lon_lo, lon_hi)})
    weights = np.cos(np.deg2rad(sub[lat_name]))
    weights.name = "w"
    return sub.weighted(weights).mean((lat_name, lon_name))


def caesar_fingerprint(sst: xr.DataArray) -> xr.DataArray:
    """Annual-mean subpolar-minus-NH-mean SST anomaly (Caesar-style index).

    SST climatology removed using the full record.
    """
    sst_annual = annual_mean(sst)
    clim = sst_annual.mean("year")
    anom = sst_annual - clim
    subpolar = box_mean(anom, SUBPOLAR_BOX["lat"], SUBPOLAR_BOX["lon"])
    nh = box_mean(anom, NH_BOX["lat"], NH_BOX["lon"])
    fingerprint = (subpolar - nh).rename("fingerprint")
    fingerprint = fingerprint.assign_attrs(
        long_name="Caesar-style subpolar minus NH-mean SST anomaly",
        units="K",
    )
    return fingerprint


def eke_weighted_index(sst: xr.DataArray, eke: xr.DataArray,
                      eke_lat="lat", eke_lon="rlon", lon_offset=-80.0) -> xr.DataArray:
    """Annual-mean spatial average of SST anomaly weighted by EKE on
    the 2-degree Atlantic basin grid.

    The EKE climatology lives on a rotated longitude (rlon) coordinate
    with offset -80 deg.  We flatten the basin grid to a list of cells,
    nearest-neighbour-pull HadISST onto each cell, then take the
    EKE-weighted spatial mean per year.
    """
    sst_annual = annual_mean(sst)
    clim = sst_annual.mean("year")
    anom = sst_annual - clim

    lat_e = np.asarray(eke[eke_lat].values, dtype=np.float64)
    lon_e = (np.asarray(eke[eke_lon].values, dtype=np.float64)
             + lon_offset + 180.0) % 360.0 - 180.0
    lat2, lon2 = np.meshgrid(lat_e, lon_e, indexing="ij")
    lat_flat = lat2.ravel()
    lon_flat = lon2.ravel()
    eke_flat = eke.values.ravel()

    mask = np.isfinite(eke_flat) & (eke_flat > 0)
    if not mask.any():
        raise ValueError("EKE weights sum to zero; check eke grid alignment.")
    lat_flat = lat_flat[mask]
    lon_flat = lon_flat[mask]
    w = eke_flat[mask]
    w_norm = w / w.sum()

    lats_h = anom["latitude"].values.astype(np.float64)
    lons_h = anom["longitude"].values.astype(np.float64)
    ilat = np.argmin(np.abs(lats_h[:, None] - lat_flat[None, :]), axis=0)
    ilon = np.argmin(np.abs(lons_h[:, None] - lon_flat[None, :]), axis=0)

    # Pull (year, cell) array via direct numpy indexing.
    anom_vals = anom.values  # (year, lat, lon)
    pulled = anom_vals[:, ilat, ilon]  # (year, cell)
    finite_cell = np.isfinite(pulled).all(axis=0)
    if not finite_cell.any():
        raise ValueError("All sampled cells are NaN; check fingerprint years.")
    pulled = pulled[:, finite_cell]
    w_norm = w_norm[finite_cell]
    w_norm = w_norm / w_norm.sum()

    W = (pulled * w_norm[None, :]).sum(axis=1)
    out = xr.DataArray(W, dims=("year",),
                       coords={"year": anom["year"].values},
                       name="W_eke_weighted")
    out = out.assign_attrs(
        long_name="EKE-weighted spatial mean of SST anomaly on Atlantic basin",
        units="K",
    )
    return out


def residualise(F: xr.DataArray, W: xr.DataArray) -> tuple[xr.DataArray, float, float]:
    """Linear-regression residualisation: F_res = F - beta * W.

    Returns the residualised fingerprint, the regression coefficient,
    and the raw Pearson correlation rho(F, W).
    """
    F = F.dropna("year")
    W = W.dropna("year")
    common = np.intersect1d(F["year"].values, W["year"].values)
    F = F.sel(year=common)
    W = W.sel(year=common)
    Fv = F.values - F.values.mean()
    Wv = W.values - W.values.mean()
    cov = float((Fv * Wv).mean())
    varW = float((Wv * Wv).mean())
    beta = cov / varW if varW > 0 else 0.0
    rho = float(np.corrcoef(F.values, W.values)[0, 1])
    F_res = F - beta * W
    F_res = F_res.rename("fingerprint_residualised").assign_attrs(
        long_name="Caesar fingerprint with EKE-weighted projection removed",
        units="K",
        beta_eke_weight=beta,
        rho_F_W=rho,
    )
    return F_res, beta, rho


def rolling_window_csd(series: xr.DataArray, window_years: int = WINDOW_YEARS):
    """Lag-1 autocorrelation and variance in a sliding window.

    Returns two xarray DataArrays, both indexed by the right edge of the
    window in years.  Series must be detrended for CSD; we apply a
    Gaussian high-pass with width = window_years / 4.
    """
    from scipy.ndimage import gaussian_filter1d

    y = series.dropna("year").astype(np.float64)
    yv = y.values
    yr = y["year"].values
    smoothed = gaussian_filter1d(yv, sigma=window_years / 4.0, mode="nearest")
    fluct = yv - smoothed

    n = yv.size
    lag1 = np.full(n, np.nan)
    var = np.full(n, np.nan)
    for i in range(window_years - 1, n):
        win = fluct[i - window_years + 1: i + 1]
        win = win - win.mean()
        if win.size < 4 or win.std() <= 0:
            continue
        # Pearson at lag 1
        a = win[:-1]; b = win[1:]
        denom = (a.std() * b.std())
        lag1[i] = float(((a - a.mean()) * (b - b.mean())).mean() / denom) \
            if denom > 0 else np.nan
        var[i] = float(win.var(ddof=1))

    lag1_da = xr.DataArray(lag1, dims=("year",), coords={"year": yr})
    var_da = xr.DataArray(var, dims=("year",), coords={"year": yr})
    return lag1_da, var_da


def trend_slope(series: xr.DataArray, year_start=CSD_FIT_START, year_end=CSD_FIT_END):
    """Least-squares slope of `series` over (year_start, year_end)."""
    finite = np.isfinite(series.values)
    sub = series.where(finite & (series["year"] >= year_start)
                       & (series["year"] <= year_end), drop=True)
    if sub.size < 5:
        return float("nan")
    x = sub["year"].values.astype(np.float64)
    y = sub.values.astype(np.float64)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def make_figure(F, F_res, lag1_raw, lag1_res, var_raw, var_res,
                slopes: CSDResult, out_path: Path) -> None:
    """Four-panel figure following the project figure-quality standard:
    panel letters as bold corner text; no titles inside axes;
    legends placed so they cannot overlap data.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })

    fig = plt.figure(figsize=(7.09, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    def _panel(ax, label, dx=-0.10):
        # Panel letter OUTSIDE the axes (top-left), no bbox.
        ax.text(dx, 1.02, label, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="left")

    # (a) raw vs residualised fingerprint
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(F["year"], F.values, color="C3", lw=1.0, label="raw")
    ax_a.plot(F_res["year"], F_res.values, color="C0", lw=1.0,
              label="EKE-residualised")
    ax_a.axhline(0, color="0.7", lw=0.5)
    ax_a.set_xlabel("year")
    ax_a.set_ylabel("fingerprint anomaly [K]")
    ax_a.legend(loc="lower left", frameon=False)
    ax_a.tick_params(direction="in", length=2.5)
    _panel(ax_a, "a")

    # (b) lag-1 AC
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(lag1_raw["year"], lag1_raw.values, color="C3", lw=1.0)
    ax_b.plot(lag1_res["year"], lag1_res.values, color="C0", lw=1.0)
    ax_b.set_xlabel("year (right edge of 50-yr window)")
    ax_b.set_ylabel("lag-1 autocorrelation")
    ax_b.tick_params(direction="in", length=2.5)
    _panel(ax_b, "b")

    # (c) variance
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.plot(var_raw["year"], var_raw.values, color="C3", lw=1.0)
    ax_c.plot(var_res["year"], var_res.values, color="C0", lw=1.0)
    ax_c.set_xlabel("year (right edge of 50-yr window)")
    ax_c.set_ylabel(r"variance [K$^{2}$]")
    ax_c.tick_params(direction="in", length=2.5)
    _panel(ax_c, "c")

    # (d) slope comparison
    ax_d = fig.add_subplot(gs[1, 1])
    labels = ["lag-1 AC slope", "variance slope"]
    vals_raw = [slopes.lag1_slope_raw, slopes.var_slope_raw]
    vals_res = [slopes.lag1_slope_residualised, slopes.var_slope_residualised]
    xpos = np.arange(len(labels))
    width = 0.35
    ax_d.bar(xpos - width / 2, vals_raw, width, color="C3", label="raw")
    ax_d.bar(xpos + width / 2, vals_res, width, color="C0",
             label="residualised")
    ax_d.axhline(0, color="0.4", lw=0.5)
    ax_d.set_xticks(xpos)
    ax_d.set_xticklabels(labels)
    ax_d.set_ylabel("trend over 1900-2020 [yr$^{-1}$]")
    ax_d.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                ncol=2, frameon=False)
    ax_d.tick_params(direction="in", length=2.5)
    _panel(ax_d, "d")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hadisst", type=Path, required=True,
                   help="HadISST monthly SST NetCDF (1870-present, 1 deg).")
    p.add_argument("--eke", type=Path, required=True,
                   help="GLORYS12 EKE climatology on 2-deg basin grid.")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-fig",  type=Path, required=True)
    p.add_argument("--year-start", type=int, default=CSD_FIT_START)
    p.add_argument("--year-end",   type=int, default=CSD_FIT_END)
    p.add_argument("--window-years", type=int, default=WINDOW_YEARS)
    args = p.parse_args(argv)

    print(f"loading HadISST {args.hadisst}")
    ds = xr.open_dataset(args.hadisst)
    sst = ds["sst"].where(ds["sst"] > -10)  # mask sea-ice fill values
    # Trim to the post-1900 record for the fingerprint
    sst = sst.sel(time=slice(f"{args.year_start - 5}-01-01",
                             f"{args.year_end + 1}-12-31"))

    print(f"loading EKE climatology {args.eke}")
    eke = xr.open_dataarray(args.eke)

    print("building Caesar fingerprint ...")
    F = caesar_fingerprint(sst)
    print(f"  fingerprint: {F.sizes['year']} years, "
          f"std = {float(F.std()):.3f} K")

    print("building EKE-weighted index ...")
    W = eke_weighted_index(sst, eke)
    print(f"  W: {W.sizes['year']} years, std = {float(W.std()):.3f} K")

    F_res, beta, rho = residualise(F, W)
    print(f"  beta = {beta:+.3f}, rho(F, W) = {rho:+.3f}")

    print(f"CSD on {args.window_years}-yr sliding windows ...")
    lag1_raw, var_raw = rolling_window_csd(F, args.window_years)
    lag1_res, var_res = rolling_window_csd(F_res, args.window_years)

    slope_lag1_raw = trend_slope(lag1_raw, args.year_start, args.year_end)
    slope_lag1_res = trend_slope(lag1_res, args.year_start, args.year_end)
    slope_var_raw = trend_slope(var_raw, args.year_start, args.year_end)
    slope_var_res = trend_slope(var_res, args.year_start, args.year_end)

    def _frac(a, b):
        return float((b - a) / a) if a not in (0.0, float("nan")) and np.isfinite(a) else float("nan")

    res = CSDResult(
        n_years=int(F.sizes["year"]),
        n_windows=int(np.isfinite(lag1_raw.values).sum()),
        lag1_slope_raw=slope_lag1_raw,
        lag1_slope_residualised=slope_lag1_res,
        var_slope_raw=slope_var_raw,
        var_slope_residualised=slope_var_res,
        lag1_slope_change_frac=_frac(slope_lag1_raw, slope_lag1_res),
        var_slope_change_frac=_frac(slope_var_raw, slope_var_res),
        beta_eke_weight=beta,
        rho_F_W=rho,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(res.to_json(), indent=2))
    print(f"wrote {args.out_json}")
    print(json.dumps(res.to_json(), indent=2))

    make_figure(F, F_res, lag1_raw, lag1_res, var_raw, var_res, res,
                args.out_fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
