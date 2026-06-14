"""Fig. 7 panel (c) -- JET pathway.

European DJF jet-latitude regression on the bandpass-filtered cold-blob
index. For each CMIP6 historical model the cold-blob index is built from
the model's own tas inside the 50-56 N, 45-25 W box (same recipe as the
Sahel script), so the regression measures the model-internal jet
response to its own North Atlantic cooling rather than borrowing the
observed (HadISST) cold-blob trajectory. The observational anchor still
uses HadISST cold-blob index versus NCEP R1 uwnd.

For each CMIP6 historical model with monthly ua available:

  1. Compute DJF zonal-mean U_850 over the Atlantic sector (60 W - 0 E),
     yielding U(lat) per DJF season.
  2. Define jet latitude as argmax over 30-65 N, with parabolic
     refinement using the maximum and its two neighbours for sub-grid
     precision.
  3. Compute the model's annual cold-blob index = - <tas>_(50-56 N,
     45-25 W).  Sign flipped so that positive = stronger cold blob.
  4. Compute the model's annual global-mean tas (for partialling out
     global warming).
  5. Bandpass cold-blob index in the 10-30 yr band; regress DJF jet
     latitude on bandpassed CB index partialling out global-mean tas
     (HAC Newey-West, 4-yr lag).

Observational anchor: NCEP/NCAR Reanalysis 1 monthly uwnd at 850 hPa,
HadISST cold-blob index; same bandpass and partial-regression recipe.

Outputs
-------
  dcps/cache/fig7_panel_c/jet.json          -- slopes, medians, CI
  dcps/cache/fig7_panel_c/jet_series.npz    -- per-model bandpassed series
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from scipy import signal, stats

# -----------------------------------------------------------------------------
# paths
# -----------------------------------------------------------------------------
REPO = Path("/home/bijanf/Documents/NEW_Theory")
COLD_BLOB_NC = Path(
    "/home/bijanf/Documents/AMOC_renalysis/data/results/cold_blob_timeseries_hadisst.nc"
)
NCEP_UA_NC = Path(
    "/home/bijanf/Documents/arctic-regime-diagram/data/reanalysis/ncep_r1_uwnd.mon.mean.nc"
)
CMIP6_ATMOS = REPO / "data" / "external" / "cmip6_atmos"
CACHE_DIR = REPO / "dcps" / "cache" / "fig7_panel_c"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = CACHE_DIR / "jet.json"
OUT_NPZ = CACHE_DIR / "jet_series.npz"

# -----------------------------------------------------------------------------
# analysis parameters
# -----------------------------------------------------------------------------
ATL_LON_W = -60.0  # western edge of Atlantic sector
ATL_LON_E = 0.0    # eastern edge (Greenwich)
JET_LAT_MIN = 30.0
JET_LAT_MAX = 65.0
PLEV_TARGET = 85000.0  # 850 hPa in Pa
BAND_LOW_YR = 10.0   # bandpass low cut (long-period)
BAND_HIGH_YR = 30.0  # bandpass high cut (long-period)
EPOCH_START = 1900
EPOCH_END = 2014

# Cold-blob tas box: same as the Sahel script (50-56 N, 45-25 W)
CB_LAT = (50.0, 56.0)
CB_LON_WEST = -45.0
CB_LON_EAST = -25.0

# Target historical models for the regression
TARGET_HISTORICAL = [
    "CNRM-CM6-1",
    "CanESM5",
    "GISS-E2-1-G",
    "MIROC6",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "UKESM1-0-LL",
]
# Target ssp245 models. Note: these enter only if local tas is available.
TARGET_SSP245 = [
    "BCC-CSM2-MR",
    "CESM2-WACCM",
    "CanESM5",
    "MIROC6",
    "MPI-ESM1-2-LR",
]


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _select_atlantic_sector(da: xr.DataArray, lon_name: str) -> xr.DataArray:
    """Select 60W-0E from a DataArray; tolerates 0-360 or -180-180 lon."""
    lon = da[lon_name].values
    if np.nanmax(lon) > 180.0:
        # 0-360 convention; Atlantic sector is 300-360
        mask = (lon >= 300.0) | (lon <= 0.0)
        return da.isel({lon_name: np.where(mask)[0]})
    else:
        return da.sel({lon_name: slice(ATL_LON_W, ATL_LON_E)})


def _coerce_lat_lon_plev(ds: xr.Dataset) -> tuple[str, str, str, str]:
    """Return canonical (lat, lon, plev, u) names for either CMIP6 or NCEP."""
    lat = "lat" if "lat" in ds.dims else ("latitude" if "latitude" in ds.dims else None)
    lon = "lon" if "lon" in ds.dims else ("longitude" if "longitude" in ds.dims else None)
    if "plev" in ds.dims:
        plev = "plev"
    elif "level" in ds.dims:
        plev = "level"
    elif "pressure_level" in ds.dims:
        plev = "pressure_level"
    else:
        plev = None
    if "ua" in ds.data_vars:
        u = "ua"
    elif "uwnd" in ds.data_vars:
        u = "uwnd"
    elif "u" in ds.data_vars:
        u = "u"
    else:
        raise KeyError(f"no u wind variable found in {list(ds.data_vars)}")
    return lat, lon, plev, u


def _select_850hpa(da: xr.DataArray, plev_name: str) -> xr.DataArray:
    """Select 850 hPa whether the plev coordinate is in Pa or hPa."""
    plev = da[plev_name].values
    # CMIP6: Pa (85000); NCEP: hPa (850)
    if np.nanmax(plev) > 5000.0:
        target = PLEV_TARGET
    else:
        target = 850.0
    # Require exact (or near-exact) match: within 5% of target
    idx = int(np.argmin(np.abs(plev - target)))
    if abs(plev[idx] - target) > 0.05 * target:
        raise ValueError(f"plev level closest to {target}: {plev[idx]} > 5% off; refusing")
    return da.isel({plev_name: idx})


def _djf_seasonal_mean(da: xr.DataArray, time_name: str = "time") -> xr.DataArray:
    """Group by DJF season; label by the year of January."""
    times = da[time_name]
    months = times.dt.month
    years = times.dt.year.values
    labels = np.where(months.values == 12, years + 1,
                      np.where(months.values <= 2, years, -9999))
    da = da.assign_coords({"djfyear": (time_name, labels)})
    da = da.where(da["djfyear"] != -9999, drop=True)
    counts = da.groupby("djfyear").count(dim=time_name)
    means = da.groupby("djfyear").mean(dim=time_name)
    if "djfyear" in counts.dims:
        valid_years = counts["djfyear"].values[(counts.max(
            dim=[d for d in counts.dims if d != "djfyear"]
        ).values >= 3)]
        means = means.sel(djfyear=valid_years)
    return means.rename({"djfyear": "year"})


def _parabolic_refine(values: np.ndarray, lats: np.ndarray) -> float:
    """Quadratic refinement of argmax to sub-grid precision."""
    j = int(np.nanargmax(values))
    if j <= 0 or j >= len(values) - 1:
        return float(lats[j])
    y0, y1, y2 = values[j - 1], values[j], values[j + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if not np.isfinite(denom) or denom == 0:
        return float(lats[j])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(lats[j] + delta * (lats[j] - lats[j - 1]))


def _jet_latitude_series(u_djf: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    """Given DJF U(year, lat, lon), return jet lat per year (degrees)."""
    u_sector = _select_atlantic_sector(u_djf, lon_name)
    u_zm = u_sector.mean(dim=lon_name, skipna=True)
    u_band = u_zm.sortby(lat_name).sel({lat_name: slice(JET_LAT_MIN, JET_LAT_MAX)})
    lats = u_band[lat_name].values
    out = np.full(u_band["year"].size, np.nan)
    arr = u_band.values
    for i in range(arr.shape[0]):
        v = arr[i]
        if np.all(np.isnan(v)):
            continue
        out[i] = _parabolic_refine(v, lats)
    return xr.DataArray(out, coords={"year": u_band["year"].values}, dims=["year"])


def _bandpass(x: np.ndarray, low_yr: float, high_yr: float) -> np.ndarray:
    """Bandpass filter annual series; returns NaN where input is NaN."""
    mask = np.isfinite(x)
    if mask.sum() < 30:
        return np.full_like(x, np.nan, dtype=float)
    xi = x.copy().astype(float)
    xi[~mask] = np.nanmean(xi[mask])
    fs = 1.0
    low = 1.0 / high_yr
    high = 1.0 / low_yr
    nyq = 0.5 * fs
    sos = signal.butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    y = signal.sosfiltfilt(sos, xi)
    y[~mask] = np.nan
    return y


def _partial_regression_hac(y: np.ndarray, x: np.ndarray, z: np.ndarray,
                            hac_lag: int = 4) -> dict:
    """Regress y = a + b*x + c*z + e; report slope b with Newey-West HAC SE."""
    m = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
    if m.sum() < 15:
        return {"slope": float("nan"), "se": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p": float("nan"), "n": int(m.sum()), "r2": float("nan")}
    yi, xi, zi = y[m], x[m], z[m]
    n = yi.size
    X = np.column_stack([np.ones(n), xi, zi])
    beta, *_ = np.linalg.lstsq(X, yi, rcond=None)
    yhat = X @ beta
    resid = yi - yhat
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((yi - yi.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    XtX_inv = np.linalg.inv(X.T @ X)
    L = hac_lag
    S = (resid[:, None] * X).T @ (resid[:, None] * X)
    for ll in range(1, L + 1):
        w = 1.0 - ll / (L + 1)
        u = (resid[:-ll, None] * X[:-ll]).T @ (resid[ll:, None] * X[ll:])
        S = S + w * (u + u.T)
    cov = XtX_inv @ S @ XtX_inv
    se_b = float(np.sqrt(max(cov[1, 1], 0.0)))
    slope = float(beta[1])
    dof = n - 3
    tcrit = stats.t.ppf(0.975, dof)
    ci_lo = slope - tcrit * se_b
    ci_hi = slope + tcrit * se_b
    tval = slope / se_b if se_b > 0 else 0.0
    pval = 2.0 * (1.0 - stats.t.cdf(abs(tval), dof))
    return {"slope": slope, "se": se_b, "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi), "p": float(pval), "n": int(n),
            "r2": float(r2)}


def _sel_box(da: xr.DataArray, lat_range: tuple[float, float],
             lon_west: float, lon_east: float) -> xr.DataArray:
    """Area-mean over a lat/lon box, handling 0-360 / -180-180 conventions."""
    lat_name = "lat" if "lat" in da.dims else ("latitude" if "latitude" in da.dims else None)
    lon_name = "lon" if "lon" in da.dims else ("longitude" if "longitude" in da.dims else None)
    if lat_name is None or lon_name is None:
        raise RuntimeError(f"no lat/lon dim: {list(da.dims)}")
    lats = da[lat_name].values
    lons = da[lon_name].values

    lo_lat, hi_lat = lat_range
    if lats[0] > lats[-1]:
        da = da.sel({lat_name: slice(hi_lat, lo_lat)})
    else:
        da = da.sel({lat_name: slice(lo_lat, hi_lat)})

    if lons.min() >= 0:
        lw = lon_west % 360
        le = lon_east % 360
        if lw <= le:
            da = da.sel({lon_name: slice(lw, le)})
        else:
            da = xr.concat([
                da.sel({lon_name: slice(lw, 360)}),
                da.sel({lon_name: slice(0, le)}),
            ], dim=lon_name)
    else:
        da = da.sel({lon_name: slice(lon_west, lon_east)})

    weights = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(weights).mean(dim=(lat_name, lon_name))


# -----------------------------------------------------------------------------
# data loading
# -----------------------------------------------------------------------------
def load_cold_blob_index() -> xr.DataArray:
    ds = xr.open_dataset(COLD_BLOB_NC)
    a = ds["caesar_box_anomaly"].astype(float)
    return a


def load_global_anom() -> xr.DataArray:
    ds = xr.open_dataset(COLD_BLOB_NC)
    return ds["global_anomaly"].astype(float)


def load_obs_jet_lat() -> Optional[xr.DataArray]:
    if not NCEP_UA_NC.exists():
        return None
    ds = xr.open_dataset(NCEP_UA_NC)
    lat, lon, plev, u = _coerce_lat_lon_plev(ds)
    u850 = _select_850hpa(ds[u], plev)
    u_djf = _djf_seasonal_mean(u850)
    jet = _jet_latitude_series(u_djf, lat, lon)
    return jet


def _list_ua_files(exp: str, model: str) -> list[Path]:
    d = CMIP6_ATMOS / exp / model / "Amon" / "ua"
    if not d.is_dir():
        return []
    ncs = sorted(d.glob("ua_Amon_*.nc"))
    # filter tiny/corrupt files (<1 MB)
    ncs = [p for p in ncs if p.stat().st_size > 1 * 1024 * 1024]
    return ncs


def _list_tas_files(exp: str, model: str) -> list[Path]:
    d = CMIP6_ATMOS / exp / model / "Amon" / "tas"
    if not d.is_dir():
        return []
    ncs = sorted(d.glob("tas_Amon_*.nc"))
    ncs = [p for p in ncs if p.stat().st_size > 100 * 1024]
    return ncs


def _open_concat(files: list[Path]) -> Optional[xr.Dataset]:
    try:
        if len(files) == 1:
            return xr.open_dataset(files[0], use_cftime=True)
        return xr.open_mfdataset([str(p) for p in files], combine="by_coords",
                                 use_cftime=True, parallel=False)
    except Exception as exc:
        print(f"  open failed: {exc}")
        return None


def model_jet_lat(files: list[Path]) -> Optional[xr.DataArray]:
    ds = _open_concat(files)
    if ds is None:
        return None
    try:
        lat, lon, plev, u = _coerce_lat_lon_plev(ds)
        u850 = _select_850hpa(ds[u], plev)
        months = u850["time"].dt.month
        u850 = u850.where((months == 12) | (months <= 2), drop=True)
        u_djf = _djf_seasonal_mean(u850.load())
        jet = _jet_latitude_series(u_djf, lat, lon)
        return jet
    except Exception as exc:
        print(f"  jet extraction failed: {exc}")
        return None
    finally:
        ds.close()


def model_cb_and_global(tas_files: list[Path]) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (years, cb_index, global_mean) annual arrays from model tas.

    cb_index is sign-flipped (-<tas>_box) so that positive = stronger cold blob.
    """
    ds = _open_concat(tas_files)
    if ds is None:
        return None
    try:
        ta = ds["tas"]
        # Cold-blob box annual mean
        cb_da = _sel_box(ta, CB_LAT, CB_LON_WEST, CB_LON_EAST)
        cb_vals = cb_da.values
        years_t = np.array([t.year for t in cb_da["time"].values])
        uniq_t = np.unique(years_t)
        cb_ann = np.array([cb_vals[years_t == y].mean() for y in uniq_t])
        # Global-mean tas annual
        lat_name = "lat" if "lat" in ta.dims else "latitude"
        ta_arr = ta.values
        lat_vals = ta[lat_name].values
        w = np.cos(np.deg2rad(lat_vals))
        w = w / w.sum()
        zon = np.nanmean(ta_arr, axis=2)
        glob_mon = (zon * w[None, :]).sum(axis=1)
        years_g = np.array([t.year for t in ta["time"].values])
        uniq_g = np.unique(years_g)
        glob_ann = np.array([glob_mon[years_g == y].mean() for y in uniq_g])
        all_years = np.intersect1d(uniq_t, uniq_g).astype(int)
        cb_aligned = np.array([cb_ann[uniq_t == y][0] for y in all_years])
        gl_aligned = np.array([glob_ann[uniq_g == y][0] for y in all_years])
        cb_anom = cb_aligned - np.nanmean(cb_aligned)
        gl_anom = gl_aligned - np.nanmean(gl_aligned)
        # sign-flip so + = cold blob
        return all_years, -cb_anom, gl_anom
    except Exception as exc:
        print(f"  tas-box extraction failed: {exc}")
        import traceback; traceback.print_exc()
        return None
    finally:
        ds.close()


# -----------------------------------------------------------------------------
# per-model driver
# -----------------------------------------------------------------------------
def process_model(exp: str, name: str) -> Optional[dict]:
    ua_files = _list_ua_files(exp, name)
    tas_files = _list_tas_files(exp, name)
    if not ua_files:
        print(f"  [{exp}/{name}] no ua files; skipping")
        return None
    if not tas_files:
        print(f"  [{exp}/{name}] no tas files; skipping")
        return None
    print(f"  [{exp}/{name}] ua={len(ua_files)} tas={len(tas_files)} files")
    jet = model_jet_lat(ua_files)
    if jet is None:
        return None
    jet_years = jet["year"].values.astype(int)
    jet_vals = jet.values
    cbres = model_cb_and_global(tas_files)
    if cbres is None:
        return None
    tas_years, cb_anom, gl_anom = cbres
    # Restrict to EPOCH and to common years between jet and tas
    common = np.intersect1d(jet_years, tas_years)
    common = common[(common >= EPOCH_START) & (common <= EPOCH_END)]
    if len(common) < 30:
        print(f"  [{exp}/{name}] insufficient common years ({len(common)}); skipping")
        return None
    j_map = {int(y): i for i, y in enumerate(jet_years)}
    t_map = {int(y): i for i, y in enumerate(tas_years)}
    j_aligned = np.array([jet_vals[j_map[int(y)]] for y in common])
    cb_aligned = np.array([cb_anom[t_map[int(y)]] for y in common])
    gl_aligned = np.array([gl_anom[t_map[int(y)]] for y in common])
    # Bandpass CB in 10-30 yr; jet stays raw; partial out raw global tas
    cb_bp = _bandpass(cb_aligned, BAND_LOW_YR, BAND_HIGH_YR)
    jet_bp = _bandpass(j_aligned, BAND_LOW_YR, BAND_HIGH_YR)
    reg = _partial_regression_hac(jet_bp, cb_bp, gl_aligned)
    # Linear trend on raw jet (deg / decade)
    valid = np.isfinite(j_aligned)
    if valid.sum() >= 10:
        tr = stats.linregress(common[valid].astype(float), j_aligned[valid]).slope * 10.0
    else:
        tr = float("nan")
    return {
        "experiment": exp,
        "slope_deg_per_K_CB": reg["slope"],
        "se": reg["se"],
        "ci_lo": reg["ci_lo"],
        "ci_hi": reg["ci_hi"],
        "p_value": reg["p"],
        "n_years": reg["n"],
        "r2": reg["r2"],
        "trend_deg_per_decade": tr,
        "mean_jet_lat_N": float(np.nanmean(j_aligned)),
        "cb_amp_K": float(np.nanstd(cb_bp)),
        "_series_years": common.astype(int),
        "_series_jet": j_aligned,
        "_series_jet_bp": jet_bp,
        "_series_cb_bp": cb_bp,
        "_series_gl_anom": gl_aligned,
    }


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    print("[jet] loading obs cold-blob index & global anomaly (HadISST)")
    cb_idx = load_cold_blob_index()
    g_idx = load_global_anom()
    cb_full = cb_idx.values.astype(float)
    cb_years = cb_idx["year"].values
    print(f"[jet] cold-blob years {cb_years[0]}..{cb_years[-1]}, n={len(cb_years)}")

    # Observational anchor (NCEP R1)
    print("[jet] computing NCEP R1 DJF jet latitude")
    obs_jet = load_obs_jet_lat()
    if obs_jet is None:
        raise RuntimeError("NCEP R1 ua file not found")
    obs_years = obs_jet["year"].values.astype(int)
    obs_vals = obs_jet.values
    print(f"  NCEP jet years {obs_years[0]}..{obs_years[-1]}, mean={np.nanmean(obs_vals):.2f} N")

    # Build HadISST-driven obs regression (sign-flipped CB)
    cb_align_full = np.full_like(obs_vals, np.nan, dtype=float)
    g_align = np.full_like(obs_vals, np.nan, dtype=float)
    cb_year_to_idx = {int(y): i for i, y in enumerate(cb_years)}
    for j, y in enumerate(obs_years):
        if int(y) in cb_year_to_idx:
            i = cb_year_to_idx[int(y)]
            cb_align_full[j] = -cb_full[i]  # K, + = cold blob
            g_align[j] = g_idx.values[i]
    cb_bp_obs = _bandpass(cb_align_full, BAND_LOW_YR, BAND_HIGH_YR)
    obs_bp = _bandpass(obs_vals, BAND_LOW_YR, BAND_HIGH_YR)
    obs_reg = _partial_regression_hac(obs_bp, cb_bp_obs, g_align)
    print(f"  NCEP: slope = {obs_reg['slope']:.3f} deg/K  (p={obs_reg['p']:.3g}, "
          f"n={obs_reg['n']}, SE={obs_reg['se']:.3f})")

    valid = np.isfinite(obs_vals)
    if valid.sum() >= 10:
        fit = stats.linregress(obs_years[valid].astype(float), obs_vals[valid])
        obs_trend = fit.slope * 10.0
    else:
        obs_trend = float("nan")

    # CMIP6 historical models
    print("[jet] === processing CMIP6 historical models ===")
    historical_results = {}
    series_payload = {}
    for name in TARGET_HISTORICAL:
        print(f"[jet] {name}")
        r = process_model("historical", name)
        if r is None:
            continue
        sy = r.pop("_series_years")
        sj = r.pop("_series_jet")
        sjbp = r.pop("_series_jet_bp")
        scbp = r.pop("_series_cb_bp")
        sgl = r.pop("_series_gl_anom")
        historical_results[name] = r
        series_payload[f"hist__{name}__years"] = sy
        series_payload[f"hist__{name}__jet"] = sj
        series_payload[f"hist__{name}__jet_bp"] = sjbp
        series_payload[f"hist__{name}__cb_bp"] = scbp
        series_payload[f"hist__{name}__gl"] = sgl
        sig = "*" if (r["p_value"] is not None and r["p_value"] < 0.05) else " "
        print(f"  {name}: slope={r['slope_deg_per_K_CB']:+.3f} deg/K  "
              f"SE={r['se']:.3f}  p={r['p_value']:.3g}{sig}  "
              f"trend={r['trend_deg_per_decade']:+.3f} deg/dec  "
              f"mean={r['mean_jet_lat_N']:.2f}N  n={r['n_years']}")

    # CMIP6 SSP245 (only if tas is available locally)
    print("[jet] === processing CMIP6 ssp245 models ===")
    ssp245_results = {}
    for name in TARGET_SSP245:
        print(f"[jet] {name}")
        r = process_model("ssp245", name)
        if r is None:
            continue
        sy = r.pop("_series_years")
        sj = r.pop("_series_jet")
        sjbp = r.pop("_series_jet_bp")
        scbp = r.pop("_series_cb_bp")
        sgl = r.pop("_series_gl_anom")
        ssp245_results[name] = r
        series_payload[f"ssp245__{name}__years"] = sy
        series_payload[f"ssp245__{name}__jet"] = sj
        series_payload[f"ssp245__{name}__jet_bp"] = sjbp
        series_payload[f"ssp245__{name}__cb_bp"] = scbp
        series_payload[f"ssp245__{name}__gl"] = sgl

    # ---- summary statistics on historical ensemble ----
    slopes_hist = np.array([v["slope_deg_per_K_CB"] for v in historical_results.values()])
    slopes_hist = slopes_hist[np.isfinite(slopes_hist)]
    n_hist = len(slopes_hist)
    if n_hist >= 4:
        med = float(np.nanmedian(slopes_hist))
        q25 = float(np.nanpercentile(slopes_hist, 25))
        q75 = float(np.nanpercentile(slopes_hist, 75))
        rng = np.random.default_rng(42)
        boots = np.array([np.nanmedian(rng.choice(slopes_hist, size=n_hist, replace=True))
                          for _ in range(5000)])
        ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        fallback = None
        ci_str = (f"95% CI of multi-model median = [{ci_lo:+.2f}, {ci_hi:+.2f}] "
                  f"deg/K (n={n_hist})")
    else:
        med = q25 = q75 = ci_lo = ci_hi = float("nan")
        fallback = "obs_only"
        ci_str = (f"obs-only fallback: NCEP slope = {obs_reg['slope']:+.2f} deg/K "
                  f"(p={obs_reg['p']:.2g}, n={obs_reg['n']})")

    n_sig = int(np.sum([(v["p_value"] is not None and v["p_value"] < 0.05)
                        for v in historical_results.values()]))

    # Combined ensemble (historical + ssp245) as a secondary diagnostic
    all_slopes = np.array([v["slope_deg_per_K_CB"] for v in historical_results.values()] +
                          [v["slope_deg_per_K_CB"] for v in ssp245_results.values()])
    all_slopes = all_slopes[np.isfinite(all_slopes)]
    if len(all_slopes) >= 4:
        med_all = float(np.nanmedian(all_slopes))
        q25_all = float(np.nanpercentile(all_slopes, 25))
        q75_all = float(np.nanpercentile(all_slopes, 75))
    else:
        med_all = q25_all = q75_all = float("nan")

    # Effect strength wording
    if not np.isnan(med):
        if abs(med) >= 0.7:
            strength_med = "strong"
        elif abs(med) >= 0.3:
            strength_med = "moderate"
        elif abs(med) >= 0.1:
            strength_med = "weak"
        else:
            strength_med = "near zero"
    else:
        strength_med = "undetermined"

    if obs_reg["p"] is not None and obs_reg["p"] < 0.05 and abs(obs_reg["slope"]) >= 0.5:
        strength_obs = "strong (consistent with Gervais 2019 / Hu-Fedorov 2020)"
    elif obs_reg["slope"] is not None and abs(obs_reg["slope"]) >= 0.3:
        strength_obs = "moderate"
    else:
        strength_obs = "weak"

    strength = (f"obs (NCEP, HadISST-driven): {strength_obs}; "
                f"CMIP6 historical median ({n_hist} models): {strength_med} "
                f"({med:+.2f} deg/K) with {n_sig} model(s) p<0.05")

    summary = {
        "pathway": "jet",
        "epoch": (f"{EPOCH_START}-{EPOCH_END} (CMIP6); "
                  f"{int(obs_years.min())}-{int(obs_years.max())} (NCEP R1)"),
        "bandpass_yr": [BAND_LOW_YR, BAND_HIGH_YR],
        "atlantic_sector_lon": [ATL_LON_W, ATL_LON_E],
        "jet_search_lat": [JET_LAT_MIN, JET_LAT_MAX],
        "cold_blob_box": {"lat": list(CB_LAT),
                          "lon_west": CB_LON_WEST,
                          "lon_east": CB_LON_EAST,
                          "sign": "positive = stronger cold blob (-<tas>_box)"},
        "method": ("Per-model: CB index from model's own tas inside 50-56 N, "
                   "45-25 W, sign-flipped. DJF jet latitude from model ua850 "
                   "over 60W-0E by parabolic refinement of argmax in 30-65 N. "
                   "CB bandpassed 10-30 yr; jet bandpassed 10-30 yr; partial "
                   "regression of bandpassed jet on bandpassed CB partialling "
                   "out raw global-mean tas; Newey-West HAC SE with 4-yr lag."),
        "obs": {
            "source": "NCEP/NCAR Reanalysis 1 monthly uwnd; HadISST cold-blob index",
            "slope_deg_per_K_CB": obs_reg["slope"],
            "se": obs_reg["se"],
            "ci95": [obs_reg["ci_lo"], obs_reg["ci_hi"]],
            "p_value": obs_reg["p"],
            "n_years": obs_reg["n"],
            "trend_deg_per_decade": obs_trend,
            "mean_jet_lat_N": float(np.nanmean(obs_vals)),
            "years": [int(obs_years.min()), int(obs_years.max())],
        },
        "cmip6_models": historical_results,
        "cmip6_models_ssp245": ssp245_results,
        "multi_model": {
            "median_slope_deg_per_K_CB": med,
            "q25": q25,
            "q75": q75,
            "ci95_median": [ci_lo, ci_hi],
            "n_models": n_hist,
            "n_significant_p05": n_sig,
            "fallback": fallback,
        },
        "multi_model_combined_hist_ssp245": {
            "median_slope_deg_per_K_CB": med_all,
            "q25": q25_all,
            "q75": q75_all,
            "n_models": int(len(all_slopes)),
        },
        "effect_strength": strength,
        "ci_string": ci_str,
        "interpretation": (
            "Positive slope = poleward DJF jet shift per K of cold-blob "
            "amplification, consistent with Gervais et al. 2019 and "
            "Hu & Fedorov 2020."
        ),
    }

    def _json_clean(obj):
        if isinstance(obj, dict):
            return {k: _json_clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_clean(v) for v in obj]
        if isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return v if np.isfinite(v) else None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    with OUT_JSON.open("w") as f:
        json.dump(_json_clean(summary), f, indent=2)
    print(f"[jet] wrote {OUT_JSON}")

    # NPZ for plotting
    npz_payload = {
        "obs_years": obs_years.astype(int),
        "obs_jet": obs_vals.astype(float),
        "obs_jet_bp": obs_bp,
        "cb_bp_obs_window": cb_bp_obs,
        "obs_slope": np.array([obs_reg["slope"], obs_reg["p"]]),
    }
    for k, v in series_payload.items():
        npz_payload[k] = v
    np.savez_compressed(OUT_NPZ, **npz_payload)
    print(f"[jet] wrote {OUT_NPZ}")

    print("\n=== SUMMARY ===")
    print(f"  historical models in regression: {n_hist}")
    print(f"  median slope (historical): {med:+.3f} deg/K")
    print(f"  IQR: [{q25:+.3f}, {q75:+.3f}]")
    print(f"  per-model p<0.05: {n_sig} of {n_hist}")
    print(f"  ssp245 models also processed: {len(ssp245_results)}")
    print(f"  effect: {strength}")
    return summary


# -----------------------------------------------------------------------------
# subpanel drawing for composing Fig. 7c
# -----------------------------------------------------------------------------
def draw_subpanel(ax):
    """Draw the JET subpanel onto an existing matplotlib Axes."""
    if not OUT_JSON.exists() or not OUT_NPZ.exists():
        ax.text(0.5, 0.5, "jet panel not computed", transform=ax.transAxes,
                ha="center", va="center", fontsize=6)
        ax.set_axis_off()
        return
    with OUT_JSON.open() as f:
        summ = json.load(f)
    npz = np.load(OUT_NPZ, allow_pickle=True)

    cb_x = npz["cb_bp_obs_window"]
    jet_y = npz["obs_jet_bp"]
    mask = np.isfinite(cb_x) & np.isfinite(jet_y)
    cb_x = cb_x[mask]
    jet_y = jet_y[mask]

    ax.scatter(cb_x, jet_y, s=10, marker="o",
               facecolor="none", edgecolor="black", linewidth=0.6,
               alpha=0.95, zorder=3)

    obs_slope_raw = summ["obs"]["slope_deg_per_K_CB"]
    obs_slope = float(obs_slope_raw) if obs_slope_raw is not None else float("nan")
    if np.isfinite(obs_slope) and cb_x.size > 1:
        xx = np.array([np.nanmin(cb_x), np.nanmax(cb_x)])
        ax.plot(xx, obs_slope * xx, color="black", linewidth=1.0, zorder=2)

    mm = summ.get("multi_model", {})

    def _f(x):
        return float(x) if x is not None else float("nan")

    med = _f(mm.get("median_slope_deg_per_K_CB"))
    q25 = _f(mm.get("q25"))
    q75 = _f(mm.get("q75"))
    if np.isfinite(med) and cb_x.size > 1:
        xx = np.array([np.nanmin(cb_x), np.nanmax(cb_x)])
        ax.plot(xx, med * xx, color="#D17A22", linewidth=1.2, linestyle="-",
                zorder=2)
        if np.isfinite(q25) and np.isfinite(q75):
            ax.fill_between(xx, q25 * xx, q75 * xx, color="#D17A22",
                            alpha=0.22, linewidth=0, zorder=1)

    ax.axhline(0, color="0.55", linewidth=0.4, linestyle=":", alpha=0.9)
    ax.axvline(0, color="0.55", linewidth=0.4, linestyle=":", alpha=0.9)
    ax.set_xlabel(r"Cold-Blob index, 10-30 yr band (K)")
    ax.set_ylabel(r"DJF jet lat. ($^\circ$N, 10-30 yr)", labelpad=2)

    # Stats (NCEP slope, HAC p-value, CMIP6 median + IQR, 'sign not
    # constrained') live in the LaTeX caption -- ZERO in-axes annotation.
    # Build proxy artists so the figure-foot legend can identify the two
    # series even though the scatter has no native label.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="black", marker="o", linestyle="none",
               markerfacecolor="none", markeredgewidth=0.6,
               markersize=3.5, label="NCEP/NCAR R1 DJF jet ($n=77$ yr)"),
        Line2D([], [], color="#D17A22", linestyle="-", linewidth=1.2,
               label=f"CMIP6 historical jet median ($n={mm.get('n_models', 0)}$)"),
    ]
    labels = [h.get_label() for h in handles]
    return handles, labels


if __name__ == "__main__":
    main()
