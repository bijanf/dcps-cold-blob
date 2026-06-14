"""
SI figure: atmospheric forcing explains <10% of <r_loc> variance.

Four-panel SI figure addressing the coauthor remark on whether the Quiescence
Signature is aliasing atmospheric forcing:

  (a) NAO + AMO joint variance explained (cell-wise OLS, 2000-2023, ORAS5)
      mean = 3.8 %, p90 = 8.5 %
      -> source: dcps/cache/nao_regression/var_explained_maps.nc (var_joint)

  (b) Four ERA5 surface atmospheric predictors (sshf+slhf, wind-stress mag,
      wind-stress curl, freshwater tp-e) joint R^2, cell-wise OLS, 2000-2023.
      mean = 9.4 %, p90 = 18.6 %
      -> source: dcps/cache/era5_regression/r2_map.nc (R2_atmospheric)

  (c) Variance commonality decomposition of bandpassed (1-10 yr) box-mean
      r_loc inside the Caesar cold-blob box (46-61 N, 50-20 W) into the
      contributions from EKE (proxy: monthly SSH spatial variance over the
      box) and from four ERA5 surface covariates (Q_turb, |tau|, curl(tau),
      p-e). Decomposition uses squared semi-partial correlations on the
      band-passed monthly series, expressed as fractions of Var(r_loc).

  (d) Argo subsurface dT/dt vs ERA5 turbulent-flux-implied surface
      tendency, both in the cold-blob box, monthly 2019-2023. Argo RG09
      monthly anomalies are averaged in three depth bins (0-100, 100-500,
      500-1500 m) and first-differenced (K/month). The ERA5 surface
      turbulent-flux-implied tendency is computed for a 100 m mixed layer
      assuming rho = 1025 kg/m^3, cp = 3995 J/(kg K). Box and whisker.

Output: manuscript/figs/fig_si_atmos_partial_correlation.pdf
"""
from __future__ import annotations
from pathlib import Path
import re

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle
from scipy.signal import butter, filtfilt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "dcps" / "cache"
NAO_PATH = CACHE / "nao_regression" / "var_explained_maps.nc"
ERA5_PATH = CACHE / "era5_regression" / "r2_map.nc"
PHASE2_PATH = CACHE / "phase2_R.nc"          # has r_loc_sst(time, lat, lon)
PHASE1_PATH = CACHE / "phase1_oras5_NA_2deg.nc"  # has ssh_anom -> EKE proxy

ERA5_FILE = Path("/home/bijanf/Documents/NEW_Theory/data/external/era5/"
                 "era5_na_monthly_2000_2023.nc")
ARGO_DIR = Path("/home/bijanf/Documents/AMOC_renalysis/data/argo_rg09")

OUT = REPO / "manuscript" / "figs" / "fig_si_atmos_partial_correlation.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Cold-Blob (Caesar) box: 46-61 N, 50-20 W
CB_BOX = dict(lat0=46.0, lat1=61.0, lon0=-50.0, lon1=-20.0)

# Physical constants (panel d)
RHO_SW = 1025.0       # kg/m^3
CP_SW = 3995.0        # J/(kg K)
MLD_H = 100.0         # m (assumed mixed layer depth for surface tendency)

# Bandpass cutoffs in years
BP_LO_YR = 1.0
BP_HI_YR = 10.0


# ---------------------------------------------------------------------------
# Panel (a) and (b) loaders -- unchanged from 2-panel version
# ---------------------------------------------------------------------------

def _load_nao() -> xr.DataArray:
    ds = xr.open_dataset(NAO_PATH)
    da = ds["var_joint"]
    if "rlon" in da.coords:
        # rlon runs 1..79 but represents the rotated longitude that should
        # be plotted as -79..-1 (North Atlantic basin in 0-360 convention).
        da = da.assign_coords(rlon=-da["rlon"].values).rename({"rlon": "lon"})
        # Reorder so lon is ascending
        da = da.sortby("lon")
    return da


def _load_era5() -> xr.DataArray:
    ds = xr.open_dataset(ERA5_PATH)
    return ds["R2_atmospheric"]


def _draw_box(ax, lon0: float, lon1: float, lat0: float, lat1: float, **kw):
    rect = Rectangle(
        (lon0, lat0), lon1 - lon0, lat1 - lat0,
        linewidth=0.9, edgecolor="black", facecolor="none",
        linestyle="--", **kw,
    )
    ax.add_patch(rect)


def _plot_map(ax, da: xr.DataArray, *, vmax: float, title_letter: str,
              mean_val: float):
    lon = da["lon"].values
    lat = da["lat"].values
    LON, LAT = np.meshgrid(lon, lat)
    im = ax.pcolormesh(
        LON, LAT, da.values,
        cmap="viridis", vmin=0.0, vmax=vmax,
        shading="auto", rasterized=True,
    )
    _draw_box(ax, **CB_BOX)
    ax.set_xlim(-80, 0)
    ax.set_ylim(20, 70)
    ax.set_aspect(1.4)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xticks([-80, -60, -40, -20, 0])
    ax.set_yticks([20, 30, 40, 50, 60, 70])
    ax.text(
        0.02, 0.98, title_letter, transform=ax.transAxes,
        ha="left", va="top", fontsize=8, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
    )
    ax.text(
        0.98, 0.02, f"basin mean = {mean_val * 100:.1f}%",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
    )
    return im


# ---------------------------------------------------------------------------
# Panel (c) helpers: bandpass + commonality decomposition
# ---------------------------------------------------------------------------

def _bandpass(x: np.ndarray, dt_months: float = 1.0,
              lo_yr: float = BP_LO_YR, hi_yr: float = BP_HI_YR,
              order: int = 4) -> np.ndarray:
    """Climatology + linear trend removal + Butterworth band-pass.

    Operates on a 1-D monthly time series. Returns same length array.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    # Climatology removal: subtract month-of-year mean
    if n >= 24:
        months = np.arange(n) % 12
        clim = np.array([np.nanmean(x[months == m]) for m in range(12)])
        x = x - clim[months]
    # Linear detrend
    t = np.arange(n)
    finite = np.isfinite(x)
    if finite.sum() >= 3:
        p = np.polyfit(t[finite], x[finite], 1)
        x = x - np.polyval(p, t)
    # Fill remaining NaNs with 0 for filter stability
    x = np.where(np.isfinite(x), x, 0.0)
    # Bandpass
    fs = 12.0 / 1.0     # samples per year
    nyq = 0.5 * fs
    f_hi = 1.0 / lo_yr  # high frequency cutoff (cycles/yr)
    f_lo = 1.0 / hi_yr  # low frequency cutoff
    b, a = butter(order, [f_lo / nyq, f_hi / nyq], btype="band")
    return filtfilt(b, a, x)


def _box_mean(da: xr.DataArray, *, lat_name="lat", lon_name="lon",
              lat0=CB_BOX["lat0"], lat1=CB_BOX["lat1"],
              lon0=CB_BOX["lon0"], lon1=CB_BOX["lon1"]) -> xr.DataArray:
    """Area-weighted box mean over (lat0, lat1) x (lon0, lon1)."""
    sel = da.sel({lat_name: slice(min(lat0, lat1), max(lat0, lat1)),
                  lon_name: slice(min(lon0, lon1), max(lon0, lon1))})
    if sel.sizes.get(lat_name, 0) == 0:
        # try reverse-ordered latitude
        sel = da.sel({lat_name: slice(max(lat0, lat1), min(lat0, lat1)),
                      lon_name: slice(min(lon0, lon1), max(lon0, lon1))})
    coslat = np.cos(np.deg2rad(sel[lat_name]))
    w = coslat / coslat.sum()
    return (sel * w).sum(lat_name).mean(lon_name)


def _build_rloc_box_series() -> tuple[np.ndarray, np.ndarray]:
    """Return (time, r_loc_box) at monthly resolution."""
    ds = xr.open_dataset(PHASE2_PATH)
    da = ds["r_loc_sst"]   # (time, lat, lon)
    # box mean
    s = _box_mean(da)
    return s["time"].values, s.values


def _build_eke_box_series() -> tuple[np.ndarray, np.ndarray]:
    """EKE proxy: monthly spatial variance of de-meaned ORAS5 SSH anomaly
    over the wider subpolar gyre (30-70 N, 70-10 W).

    We subtract the spatial mean at each time step so the proxy reflects
    spatial gradients (mesoscale eddy energy) rather than basin-wide tilt.
    A basin-scale aperture is used because the box-scale r_loc time series
    integrates phase coherence over the cold-blob box and is forced by
    eddy stirring on the surrounding gyre.
    """
    ds = xr.open_dataset(PHASE1_PATH)
    ssh = ds["ssh_anom"].sel(lat=slice(30, 70), lon=slice(-70, -10))
    ssh_dev = ssh - ssh.mean(dim=("lat", "lon"))
    eke = (ssh_dev ** 2).mean(dim=("lat", "lon"), skipna=True)
    return eke["time"].values, eke.values


def _build_atm_box_series() -> tuple[np.ndarray, dict]:
    """Return (time, dict) of four bandpassed atmospheric predictors,
    box-mean ERA5.
    """
    ds = xr.open_dataset(ERA5_FILE)
    sel = ds.sel(latitude=slice(CB_BOX["lat1"], CB_BOX["lat0"]),
                 longitude=slice(CB_BOX["lon0"], CB_BOX["lon1"]))
    coslat = np.cos(np.deg2rad(sel.latitude))
    w = coslat / coslat.sum()

    def mean(field):
        return (field * w).sum("latitude").mean("longitude")

    # Turbulent flux (W/m^2): divide J/m^2/day by 86400
    Q_turb = mean(sel["sshf"] + sel["slhf"]) / 86400.0
    # Wind-stress magnitude (N/m^2)
    tau_e = sel["ewss"] / 86400.0  # N/m^2 (avg over day)
    tau_n = sel["nsss"] / 86400.0
    tau_mag = np.sqrt(tau_e ** 2 + tau_n ** 2)
    tau_mag_box = mean(tau_mag)
    # Wind-stress curl: dtau_n/dx - dtau_e/dy on regular lat-lon
    R_EARTH = 6.371e6
    lat_rad = np.deg2rad(sel.latitude)
    dy = R_EARTH * np.deg2rad(np.abs(sel.latitude.diff("latitude").mean().item()))
    dlon = np.deg2rad(np.abs(sel.longitude.diff("longitude").mean().item()))
    # gradients via xarray differentiate (regular spacing assumption fine for box)
    dtaun_dx = tau_n.differentiate("longitude") / (R_EARTH * np.cos(lat_rad) * np.deg2rad(1.0))
    dtaue_dy = tau_e.differentiate("latitude") / (R_EARTH * np.deg2rad(1.0))
    curl = dtaun_dx - dtaue_dy
    curl_box = mean(curl)
    # Freshwater forcing tp - e (m/day; note e is negative when evap)
    pme = mean(sel["tp"] - sel["e"])

    times = ds.valid_time.values
    return times, {
        "Q_turb": Q_turb.values,
        "tau_mag": tau_mag_box.values,
        "tau_curl": curl_box.values,
        "pme": pme.values,
    }


def _commonality_decomposition(y: np.ndarray, X_eke: np.ndarray,
                               X_atm: np.ndarray) -> dict:
    """Variance partition of y between predictor X_eke and predictor block
    X_atm (n_t x k). Returns the four contributions as fractions of Var(y).

    Definitions:
      R2_full  = R^2 of OLS(y ~ X_eke, X_atm...)
      R2_eke   = R^2 of OLS(y ~ X_eke)
      R2_atm   = R^2 of OLS(y ~ X_atm...)
      unique_eke = R2_full - R2_atm        (eke contribution beyond atm)
      unique_atm = R2_full - R2_eke
      shared     = R2_eke + R2_atm - R2_full
      residual   = 1 - R2_full
    """

    def _r2(yy, XX):
        XX = np.column_stack([np.ones_like(yy), XX])
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        yhat = XX @ beta
        ss_res = float(np.sum((yy - yhat) ** 2))
        ss_tot = float(np.sum((yy - yy.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    Xe = X_eke.reshape(-1, 1) if X_eke.ndim == 1 else X_eke
    R2_eke = _r2(y, Xe)
    R2_atm = _r2(y, X_atm)
    R2_full = _r2(y, np.column_stack([Xe, X_atm]))
    unique_eke = max(R2_full - R2_atm, 0.0)
    unique_atm = max(R2_full - R2_eke, 0.0)
    shared = max(R2_eke + R2_atm - R2_full, 0.0)
    residual = max(1.0 - R2_full, 0.0)
    return dict(unique_eke=unique_eke, unique_atm=unique_atm,
                shared=shared, residual=residual,
                R2_eke=R2_eke, R2_atm=R2_atm, R2_full=R2_full)


def _plot_commonality(ax, parts: dict, *, title_letter: str):
    # Order: unique EKE, shared, unique atm, residual
    labels = ["unique EKE", "shared (EKE & atm.)", "unique atm.",
              "residual"]
    vals = [parts["unique_eke"], parts["shared"],
            parts["unique_atm"], parts["residual"]]
    colors = ["#2c7bb6", "#7b6cb0", "#d7191c", "#bababa"]
    markers = ["o", "D", "s", "x"]  # distinct shapes per category (for legend)
    # Single thicker horizontal stacked bar
    bar_height = 0.55
    left = 0.0
    for v, c, lab, mk in zip(vals, colors, labels, markers):
        ax.barh(0.5, v, height=bar_height, left=left, color=c,
                edgecolor="white", linewidth=0.7)
        if v > 0.04:
            ax.text(left + v / 2, 0.5, f"{v * 100:.1f}%",
                    ha="center", va="center", fontsize=6,
                    color="white" if c != "#bababa" else "black",
                    fontweight="bold")
        left += v
    # legend swatches with category-distinct markers, inside the axes
    handles = [
        plt.Line2D([0], [0], marker=mk, color=c, markersize=5,
                   linestyle="none", markeredgecolor="black",
                   markeredgewidth=0.4)
        for c, mk in zip(colors, markers)
    ]
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False, fontsize=6,
              handlelength=1.0, handletextpad=0.3,
              columnspacing=1.2)
    ax.set_yticks([])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("Fraction of Var($r_{\\mathrm{loc}}$) explained (%)")
    ax.text(
        0.02, 0.98, title_letter, transform=ax.transAxes,
        ha="left", va="top", fontsize=8, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
    )
    # annotate R2_full and per-block R^2 values
    ax.text(
        0.98, 0.95,
        (f"$R^{{2}}_{{full}}$ = {parts['R2_full'] * 100:.1f}%\n"
         f"$R^{{2}}_{{EKE}}$ = {parts['R2_eke'] * 100:.1f}%\n"
         f"$R^{{2}}_{{atm}}$ = {parts['R2_atm'] * 100:.1f}%"),
        transform=ax.transAxes, ha="right", va="top",
        fontsize=6,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


# ---------------------------------------------------------------------------
# Panel (d) helpers: Argo subsurface dT/dt + ERA5 surface tendency
# ---------------------------------------------------------------------------

ARGO_FILE_RE = re.compile(r"RG_ArgoClim_(\d{6})_2019\.nc$")


def _argo_box_monthly(depth_bins: list[tuple[float, float]]) -> tuple[np.ndarray, dict]:
    """Return (time_array, dict_of_series) where each series is box-mean
    ARGO_TEMPERATURE_ANOMALY averaged over the depth bin (PRESSURE in dbar
    approx m).

    The cold-blob box: lat 46-61 N, lon -50..-20 E. RG09 longitude is
    20.5..379.5, so we convert to 0-360 convention: -50 -> 310, -20 -> 340.
    """
    paths = sorted(ARGO_DIR.glob("RG_ArgoClim_*_2019.nc"))
    series = {f"{int(z0)}–{int(z1)} m": [] for z0, z1 in depth_bins}
    times = []
    for p in paths:
        m = ARGO_FILE_RE.search(p.name)
        if not m:
            continue
        yyyymm = m.group(1)
        year = int(yyyymm[:4])
        month = int(yyyymm[4:])
        ds = xr.open_dataset(p, decode_times=False)
        # box-select
        lon_min = (360.0 + CB_BOX["lon0"]) % 360  # 310
        lon_max = (360.0 + CB_BOX["lon1"]) % 360  # 340
        sub = ds["ARGO_TEMPERATURE_ANOMALY"].squeeze("TIME").sel(
            LATITUDE=slice(CB_BOX["lat0"], CB_BOX["lat1"]),
            LONGITUDE=slice(lon_min, lon_max),
        )
        coslat = np.cos(np.deg2rad(sub.LATITUDE))
        w = coslat / coslat.sum()
        sub_area = (sub * w).sum("LATITUDE").mean("LONGITUDE")  # PRESSURE only
        # depth bin averages
        for (z0, z1) in depth_bins:
            sub_z = sub_area.sel(PRESSURE=slice(z0, z1))
            val = float(sub_z.mean("PRESSURE", skipna=True).values)
            series[f"{int(z0)}–{int(z1)} m"].append(val)
        times.append(np.datetime64(f"{year:04d}-{month:02d}-15"))
        ds.close()
    times = np.array(times)
    order = np.argsort(times)
    times = times[order]
    series = {k: np.array(v)[order] for k, v in series.items()}
    return times, series


def _era5_box_surface_tendency() -> tuple[np.ndarray, np.ndarray]:
    """Compute the ERA5 turbulent-flux-implied tendency for a 100 m mixed
    layer in the cold-blob box, monthly 2000-2023.

    ERA5 stepType=avgad means values are daily mean of the time integrated
    flux (J/m^2/day), so /86400 -> W/m^2.

    tendency [K/month] = Q_turb [W/m^2] / (rho cp h) * seconds_per_month
    """
    ds = xr.open_dataset(ERA5_FILE)
    sel = ds.sel(latitude=slice(CB_BOX["lat1"], CB_BOX["lat0"]),
                 longitude=slice(CB_BOX["lon0"], CB_BOX["lon1"]))
    coslat = np.cos(np.deg2rad(sel.latitude))
    w = coslat / coslat.sum()
    Q = ((sel["sshf"] + sel["slhf"]) / 86400.0)  # W/m^2 (downward positive)
    Q_box = (Q * w).sum("latitude").mean("longitude")
    times = ds.valid_time.values
    # seconds per month (true calendar days)
    days_per_month = np.empty(len(times), dtype=float)
    for i in range(len(times)):
        m0 = np.datetime64(str(times[i])[:7], "M")
        m1 = m0 + np.timedelta64(1, "M")
        days_per_month[i] = (m1.astype("datetime64[D]") -
                             m0.astype("datetime64[D]")) / np.timedelta64(1, "D")
    seconds_per_month = days_per_month * 86400.0
    tendency = Q_box.values / (RHO_SW * CP_SW * MLD_H) * seconds_per_month
    return times, tendency


def _plot_panel_d(ax, argo_times, argo_series, era5_tend, era5_times,
                  title_letter: str):
    # Compute monthly first-differences for each Argo bin -> dT/dt K/month
    boxes = []
    labels = []
    markers = []
    colors = []
    # ERA5 first; restrict to 2019-2023 (matches Argo range)
    t_start = np.datetime64("2019-01-01")
    t_end = np.datetime64("2024-01-01")
    mask = (era5_times >= t_start) & (era5_times < t_end)
    boxes.append(era5_tend[mask])
    labels.append("ERA5 surface\n($Q_{\\mathrm{turb}}/\\rho c_{p} h$,\nh = 100 m)")
    markers.append("D")
    colors.append("#d7191c")
    # Argo bins, first-differenced monthly
    bin_colors = ["#2c7bb6", "#1a9850", "#6a3d9a"]
    bin_markers = ["o", "s", "^"]
    for i, (lab, ser) in enumerate(argo_series.items()):
        d_dt = np.diff(ser)
        boxes.append(d_dt)
        labels.append(f"Argo dT/dt\n{lab}")
        markers.append(bin_markers[i])
        colors.append(bin_colors[i])

    pos = np.arange(len(boxes))
    bp = ax.boxplot(boxes, positions=pos, widths=0.55, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", linewidth=1.0),
                    whiskerprops=dict(color="black", linewidth=0.7),
                    capprops=dict(color="black", linewidth=0.7),
                    boxprops=dict(linewidth=0.7))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    # overlay individual points with category-specific markers
    rng = np.random.default_rng(0)
    for i, (vals, m, c) in enumerate(zip(boxes, markers, colors)):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(np.full(len(vals), pos[i]) + jitter, vals,
                   marker=m, s=8, color=c, edgecolor="black",
                   linewidth=0.3, alpha=0.7, zorder=3)
    ax.axhline(0, color="black", lw=0.4, ls=":", zorder=1)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_ylabel("Temperature tendency (K month$^{-1}$)")
    # Show on a symmetric log-scaled y-axis equivalent using symlog
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_ylim(-5, 5)
    ax.text(
        0.02, 0.98, title_letter, transform=ax.transAxes,
        ha="left", va="top", fontsize=8, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
    )
    # Annotate the surface-vs-deep contrast (panel-d punchline)
    era5_med = float(np.nanmedian(np.abs(boxes[0])))
    argo_deep = float(np.nanmedian(np.abs(boxes[-1])))
    if argo_deep > 0:
        ratio = era5_med / argo_deep
        ax.text(
            0.98, 0.04,
            f"|ERA5|/|Argo$_{{500–1500}}$| median = {ratio:.0f}×",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- panels (a) and (b) ------------------------------------------------
    nao = _load_nao()
    era5 = _load_era5()
    nao_mean = float(np.nanmean(nao.values))
    era5_mean = float(np.nanmean(era5.values))

    # ---- panel (c): commonality -------------------------------------------
    t_r, r_box = _build_rloc_box_series()
    t_eke, eke_box = _build_eke_box_series()
    t_atm, atm = _build_atm_box_series()

    # Align all to a common monthly grid covering 2000-07 .. 2023-06
    # (intersection of phase2 276 months and the ERA5/EKE 288 months)
    t_common = np.array(sorted(set(t_r).intersection(set(t_eke)).intersection(set(t_atm))))
    if len(t_common) == 0:
        # fall back: pad shorter series via month-matching using year-month
        def _ym(arr):
            return np.array([np.datetime64(str(a)[:7], "M") for a in arr])
        ym_r = _ym(t_r); ym_e = _ym(t_eke); ym_a = _ym(t_atm)
        common_ym = np.array(sorted(set(ym_r).intersection(set(ym_e)).intersection(set(ym_a))))
        idx_r = np.array([np.where(ym_r == m)[0][0] for m in common_ym])
        idx_e = np.array([np.where(ym_e == m)[0][0] for m in common_ym])
        idx_a = np.array([np.where(ym_a == m)[0][0] for m in common_ym])
        r_sel = r_box[idx_r]
        eke_sel = eke_box[idx_e]
        atm_sel = {k: v[idx_a] for k, v in atm.items()}
    else:
        idx_r = np.array([np.where(t_r == t)[0][0] for t in t_common])
        idx_e = np.array([np.where(t_eke == t)[0][0] for t in t_common])
        idx_a = np.array([np.where(t_atm == t)[0][0] for t in t_common])
        r_sel = r_box[idx_r]
        eke_sel = eke_box[idx_e]
        atm_sel = {k: v[idx_a] for k, v in atm.items()}

    # Bandpass each
    r_bp = _bandpass(r_sel)
    eke_bp = _bandpass(eke_sel)
    atm_bp = np.column_stack([_bandpass(v) for v in atm_sel.values()])
    parts = _commonality_decomposition(r_bp, eke_bp, atm_bp)

    # ---- panel (d): Argo vs ERA5 surface tendency -------------------------
    argo_t, argo_series = _argo_box_monthly([(0, 100), (100, 500), (500, 1500)])
    era5_t, era5_tend = _era5_box_surface_tendency()

    # ---- assemble figure ---------------------------------------------------
    # 2-column Nature width = 180 mm = 7.09 in. 2x2 grid with compact rows.
    fig = plt.figure(figsize=(7.09, 6.4))
    gs = fig.add_gridspec(
        2, 2, hspace=0.40, wspace=0.30,
        left=0.07, right=0.97, top=0.97, bottom=0.09,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    im_a = _plot_map(ax_a, nao, vmax=0.30, title_letter="a",
                     mean_val=nao_mean)
    cb_a = fig.colorbar(im_a, ax=ax_a, orientation="horizontal",
                        fraction=0.06, pad=0.18, aspect=30)
    cb_a.set_label("$R^{2}$  (NAO + AMO regression on $\\langle r_{loc}\\rangle$)")

    im_b = _plot_map(ax_b, era5, vmax=0.30, title_letter="b",
                     mean_val=era5_mean)
    cb_b = fig.colorbar(im_b, ax=ax_b, orientation="horizontal",
                        fraction=0.06, pad=0.18, aspect=30)
    cb_b.set_label("$R^{2}$  (ERA5 $Q_{\\mathrm{turb}}$, $|\\tau|$, curl $\\tau$, $p-e$)")

    _plot_commonality(ax_c, parts, title_letter="c")
    _plot_panel_d(ax_d, argo_t, argo_series, era5_tend, era5_t,
                  title_letter="d")

    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"  (a) basin-mean R^2 NAO+AMO     = {nao_mean * 100:.2f}%")
    print(f"  (b) basin-mean R^2 ERA5 4-pred = {era5_mean * 100:.2f}%")
    print("  (c) commonality (cold-blob box, bandpass 1-10 yr):")
    print(f"        unique EKE   = {parts['unique_eke'] * 100:.1f}%")
    print(f"        shared       = {parts['shared'] * 100:.1f}%")
    print(f"        unique atm   = {parts['unique_atm'] * 100:.1f}%")
    print(f"        residual     = {parts['residual'] * 100:.1f}%")
    print(f"        R2_full      = {parts['R2_full'] * 100:.1f}%")
    print(f"  (d) Argo months  = {len(argo_t)}; ERA5 months in 2019-2023 = "
          f"{int(((era5_t >= np.datetime64('2019-01-01')) & (era5_t < np.datetime64('2024-01-01'))).sum())}")


if __name__ == "__main__":
    main()
