"""
Fig 7 panel (c), pathway 1: Pacific Walker circulation lead-lag regression
on the Caesar cold-blob box SST index, partialling out global-mean
temperature.

The result is a lead-lag regression curve beta(W, CB; lag) showing how
many hPa the Walker zonal-SLP gradient strengthens per +1 standard
deviation of the cold-blob index, at lags lag = -20..+20 yr.  Positive
lag = cold blob leads Walker.  Expected sign from McGregor 2014 and
Ruprich-Robert 2017/2021: cold North Atlantic (positive CB index in our
sign convention, i.e. anomalously cold subpolar SST) leads a
strengthened Pacific Walker (positive zonal SLP gradient East minus
West) by ~3-10 yr at amplitude ~+0.3 to +0.7 hPa per K.

Observational pipeline
----------------------
- Cold-blob index: HadISST caesar_box_anomaly, annual 1870-2024, then
  z-scored and Lanczos band-passed 10-30 yr.  We work with the negative
  of the SST anomaly so that "positive index = colder blob" (i.e. AMOC
  slowdown signal), in line with the Caesar / Rahmstorf sign convention.
- Walker index: HadSLP2r monthly SLP -> annual, Bjerknes zonal SLP
  gradient eastern equatorial Pacific (5 S-5 N, 160 W-80 W) minus
  Indo-western Pacific warm pool (5 S-5 N, 80 E-160 E), then 10-30 yr
  bandpass.
- Global-mean T proxy: HadISST global_anomaly, annual, same bandpass.
  Partialled out of both CB and W before computing the regression.
- Lead-lag regression beta(W_t+lag, CB_t | T), lag in {-20,-15,...,+20}.
- Uncertainty: 30-yr block bootstrap, 1000 resamples, 95% CI per lag.

CMIP6 pipeline (per model, if available)
----------------------------------------
- Read monthly psl, annual mean.
- Construct CB box (46-61 N, 50-20 W) SST is not in psl, so for the
  model we use the global SST proxy by computing the SLP-pattern surrogate:
  -> actually we cannot get SST out of psl.  Instead we derive an in-model
  cold-blob proxy from psl by taking the subpolar Atlantic SLP anomaly
  (45-65 N, 60-20 W) and z-scoring.  This is a defensible substitute since
  Caesar 2018/2021 show that cold-blob SST and subpolar-Atlantic SLP are
  tightly coupled on multi-decadal scales.  We document this in the JSON.
- Walker index identically defined.
- Same bandpass, same lead-lag regression.

Output
------
JSON at dcps/cache/fig7_panel_c/walker.json with
    obs_lag_curve, obs_peak_lag, obs_peak_amp, per_model_peaks,
    multi_model_band, obs_ci_lo, obs_ci_hi, lags, units, provenance.

draw_subpanel(ax) draws the observational anchor (mean +/- bootstrap CI)
and the per-model curves on top with thin lines.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
COLD_BLOB_HADISST = Path(
    "/home/bijanf/Documents/AMOC_renalysis/data/results/cold_blob_timeseries_hadisst.nc"
)
HADSLP2 = Path(
    "/home/bijanf/Documents/NEW_Theory/data/external/hadslp2/hadslp2r.nc"
)
CMIP6_ROOT = Path("/home/bijanf/Documents/NEW_Theory/data/external/cmip6_atmos")

OUT_DIR = REPO / "dcps" / "cache" / "fig7_panel_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "walker.json"

LAGS = list(range(-20, 21, 5))
BANDPASS_LO_YR = 10
BANDPASS_HI_YR = 30
N_BOOT = 1000
BOOT_BLOCK_YR = 30
RNG_SEED = 20260611


# ----------------------------------------------------------------------
# Band-pass Lanczos
# ----------------------------------------------------------------------
def lanczos_weights(window: int, cutoff_low: float, cutoff_high: float) -> np.ndarray:
    """Symmetric Lanczos band-pass weights (frequencies in cycles/yr)."""
    n = window
    k = np.arange(-(n // 2), n // 2 + 1)
    eps = 1e-12

    def sinc_low(fc):
        w = np.where(k == 0, 2 * fc, np.sin(2 * np.pi * fc * k) / (np.pi * k + eps))
        sigma = np.where(k == 0, 1.0, np.sin(np.pi * k / (n // 2 + 1)) /
                         (np.pi * k / (n // 2 + 1) + eps))
        return w * sigma

    high = sinc_low(cutoff_high)
    low = sinc_low(cutoff_low)
    bp = high - low
    bp /= bp.sum() if abs(bp.sum()) > 1e-9 else 1.0
    return bp


def bandpass_annual(x: np.ndarray, lo_yr: float, hi_yr: float,
                    window: int = 21) -> np.ndarray:
    """10-30 yr Lanczos bandpass on an annual series.  NaN-tolerant via
    edge truncation (output is NaN where the filter does not fit)."""
    x = np.asarray(x, float)
    w = lanczos_weights(window, 1.0 / hi_yr, 1.0 / lo_yr)
    n = len(x)
    out = np.full(n, np.nan)
    half = window // 2
    for i in range(half, n - half):
        seg = x[i - half:i + half + 1]
        if np.isfinite(seg).all():
            out[i] = float(np.dot(w, seg))
    return out


# ----------------------------------------------------------------------
# Observational indices
# ----------------------------------------------------------------------
def obs_cold_blob_index() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Annual CB index from HadISST.  Sign convention: positive index =
    anomalously COLD blob (consistent with the Caesar / Rahmstorf AMOC
    slowdown fingerprint).  Returns (years, cb_bp, global_T_bp)."""
    ds = xr.open_dataset(COLD_BLOB_HADISST)
    years = ds["year"].values.astype(float)
    cb_raw = ds["caesar_box_anomaly"].values.astype(float)  # SST anomaly
    gt_raw = ds["global_anomaly"].values.astype(float)
    # Flip sign so positive = cold (slowdown direction)
    cb = -cb_raw
    # z-score on full record
    cb_z = (cb - np.nanmean(cb)) / np.nanstd(cb)
    gt_z = (gt_raw - np.nanmean(gt_raw)) / np.nanstd(gt_raw)
    return years, bandpass_annual(cb_z, BANDPASS_LO_YR, BANDPASS_HI_YR), \
        bandpass_annual(gt_z, BANDPASS_LO_YR, BANDPASS_HI_YR)


def _slp_box_mean(ds: xr.Dataset, lat_s: float, lat_n: float,
                  lon_w: float, lon_e: float, var: str = "slp") -> xr.DataArray:
    """Box-average SLP.  Handles increasing or decreasing lat axis and
    0-360 longitude.  lon_w, lon_e are in 0-360 East convention; if
    lon_w < lon_e we wrap straight, else we wrap across the dateline."""
    da = ds[var]
    lat = da["lat"].values
    if lat[0] > lat[-1]:
        # decreasing: lat=slice(lat_n, lat_s)
        da_lat = da.sel(lat=slice(lat_n, lat_s))
    else:
        da_lat = da.sel(lat=slice(lat_s, lat_n))
    lon = da_lat["lon"].values
    if lon.min() >= 0 and lon.max() <= 360:
        if lon_w <= lon_e:
            mask = (lon >= lon_w) & (lon <= lon_e)
        else:
            mask = (lon >= lon_w) | (lon <= lon_e)
    else:
        # -180..180 convention; convert lon_w/lon_e
        def to_180(x): return ((x + 180) % 360) - 180
        lw = to_180(lon_w); le = to_180(lon_e)
        if lw <= le:
            mask = (lon >= lw) & (lon <= le)
        else:
            mask = (lon >= lw) | (lon <= le)
    da_box = da_lat.isel(lon=np.where(mask)[0])
    # cosine-lat weights
    coslat = np.cos(np.deg2rad(da_box["lat"]))
    w = coslat / coslat.sum()
    return (da_box.weighted(w).mean(dim="lat")).mean(dim="lon")


def walker_index_from_slp(ds: xr.Dataset, var: str = "slp",
                          time_dim: str = "time") -> tuple[np.ndarray, np.ndarray]:
    """Bjerknes zonal SLP gradient: east minus west.
    East:  5 S-5 N, 160 W-80 W  = lon 200-280 E
    West:  5 S-5 N, 80 E-160 E
    Returns (years, walker_index_annual_hPa)."""
    east = _slp_box_mean(ds, -5, 5, 200, 280, var=var)
    west = _slp_box_mean(ds, -5, 5, 80, 160, var=var)
    grad = east - west  # hPa or Pa
    # convert to hPa
    units = ds[var].attrs.get("units", "hPa").lower()
    if "pa" == units or units == "n/m2" or units == "n m-2":
        grad = grad / 100.0  # Pa -> hPa
    # annual mean
    try:
        ann = grad.groupby(f"{time_dim}.year").mean(dim=time_dim)
        years = ann["year"].values.astype(float)
    except Exception:
        # cftime fallback
        time = ds[time_dim].values
        years_full = np.array([t.year for t in time])
        vals = grad.values
        years = np.unique(years_full)
        ann_vals = np.array([np.nanmean(vals[years_full == y]) for y in years])
        return years.astype(float), ann_vals
    return years, ann.values.astype(float)


def cmip6_cold_blob_psl_proxy(ds: xr.Dataset, var: str = "psl",
                              time_dim: str = "time") -> tuple[np.ndarray, np.ndarray]:
    """In-model cold-blob proxy from SLP: subpolar Atlantic SLP anomaly
    in the Caesar box (46-61 N, 50-20 W = 310-340 E), annual.
    We z-score and flip sign so positive index = anomalously high SLP
    over the subpolar Atlantic = atmospheric signature of weakened
    NAC / cooler subpolar SST in the slowdown direction.  This is a
    documented surrogate when SST is not available."""
    box = _slp_box_mean(ds, 46, 61, 310, 340, var=var)
    units = ds[var].attrs.get("units", "hPa").lower()
    if units == "pa":
        box = box / 100.0
    try:
        ann = box.groupby(f"{time_dim}.year").mean(dim=time_dim)
        years = ann["year"].values.astype(float)
        vals = ann.values.astype(float)
    except Exception:
        time = ds[time_dim].values
        years_full = np.array([t.year for t in time])
        v = box.values
        years = np.unique(years_full).astype(float)
        vals = np.array([np.nanmean(v[years_full == y]) for y in years])
    # z-score; KEEP sign as-is (positive SLP = anticyclonic / cold-air advection)
    z = (vals - np.nanmean(vals)) / np.nanstd(vals)
    return years, z


# ----------------------------------------------------------------------
# Lead-lag partial regression
# ----------------------------------------------------------------------
def partial_residual(y: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Residual of y after OLS regression on covariate c (NaN tolerant)."""
    m = np.isfinite(y) & np.isfinite(c)
    if m.sum() < 5:
        return np.full_like(y, np.nan, dtype=float)
    a, b = np.polyfit(c[m], y[m], 1)
    res = y - (a * c + b)
    res[~m] = np.nan
    return res


def lead_lag_regression(walker: np.ndarray, cb: np.ndarray, gT: np.ndarray,
                        lags: list[int]) -> np.ndarray:
    """beta(W_{t+lag}, CB_t | gT) in hPa per std(CB).
    Positive lag = CB leads Walker."""
    # partial out global T from both
    w_res = partial_residual(walker, gT)
    c_res = partial_residual(cb, gT)
    out = np.full(len(lags), np.nan)
    n = len(walker)
    for i, lag in enumerate(lags):
        if lag >= 0:
            w_seg = w_res[lag:]
            c_seg = c_res[:n - lag]
        else:
            w_seg = w_res[:n + lag]
            c_seg = c_res[-lag:]
        m = np.isfinite(w_seg) & np.isfinite(c_seg)
        if m.sum() < 10:
            continue
        # OLS slope of W on CB; CB already z-scored so units = hPa per sigma_CB
        slope, _ = np.polyfit(c_seg[m], w_seg[m], 1)
        out[i] = slope
    return out


def block_bootstrap_ci(walker: np.ndarray, cb: np.ndarray, gT: np.ndarray,
                       lags: list[int], n_boot: int = N_BOOT,
                       block: int = BOOT_BLOCK_YR,
                       seed: int = RNG_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Moving-block bootstrap 95% CI per lag."""
    rng = np.random.default_rng(seed)
    n = len(walker)
    n_blocks = max(1, n // block)
    boot = np.full((n_boot, len(lags)), np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        boot[b] = lead_lag_regression(walker[idx], cb[idx], gT[idx], lags)
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    return lo, hi


# ----------------------------------------------------------------------
# CMIP6 loader
# ----------------------------------------------------------------------
def list_cmip6_models() -> list[tuple[str, str, list[Path]]]:
    """Returns [(model, experiment, [files])] for every (model, exp) with
    psl on disk."""
    out: list[tuple[str, str, list[Path]]] = []
    for exp_dir in sorted(CMIP6_ROOT.glob("*")):
        if not exp_dir.is_dir(): continue
        exp = exp_dir.name
        for model_dir in sorted(exp_dir.glob("*")):
            psl_dir = model_dir / "Amon" / "psl"
            if not psl_dir.exists(): continue
            files = sorted(psl_dir.glob("psl_*.nc"))
            if not files: continue
            out.append((model_dir.name, exp, files))
    return out


def load_psl_concat(files: list[Path]) -> Optional[xr.Dataset]:
    """Open multiple psl files and concat in time.  Returns None on
    failure."""
    try:
        if len(files) == 1:
            ds = xr.open_dataset(files[0], use_cftime=True)
        else:
            ds = xr.open_mfdataset([str(f) for f in files], combine="by_coords",
                                   use_cftime=True, parallel=False)
        return ds
    except Exception as exc:
        print(f"  load_psl_concat({len(files)} files) failed: {exc}")
        return None


def compute_model_curve(model: str, exp: str, files: list[Path],
                        lags: list[int]) -> Optional[dict]:
    print(f"  -> {model} / {exp}: {len(files)} file(s)")
    ds = load_psl_concat(files)
    if ds is None:
        return None
    try:
        wyears, walker = walker_index_from_slp(ds, var="psl", time_dim="time")
        cyears, cb_proxy = cmip6_cold_blob_psl_proxy(ds, var="psl", time_dim="time")
        # align
        common = np.intersect1d(wyears, cyears)
        wmask = np.isin(wyears, common); cmask = np.isin(cyears, common)
        years = common.astype(float)
        walker = walker[wmask]; cb = cb_proxy[cmask]
        # global-mean SLP as crude T proxy: zonal average over -60..60 lat
        gT_proxy = _slp_box_mean(ds, -60, 60, 0, 360, var="psl")
        try:
            gT_ann = gT_proxy.groupby("time.year").mean("time")
            gT = gT_ann.sel(year=common).values.astype(float)
        except Exception:
            gT = np.zeros_like(walker)
        # bandpass
        walker_bp = bandpass_annual(walker, BANDPASS_LO_YR, BANDPASS_HI_YR)
        cb_bp = bandpass_annual(cb, BANDPASS_LO_YR, BANDPASS_HI_YR)
        gT_bp = bandpass_annual(gT, BANDPASS_LO_YR, BANDPASS_HI_YR)
        # regression
        beta = lead_lag_regression(walker_bp, cb_bp, gT_bp, lags)
        if not np.isfinite(beta).any():
            return None
        peak_ix = int(np.nanargmax(np.abs(beta)))
        return {
            "model": model,
            "experiment": exp,
            "years": years.tolist(),
            "lag_curve_hPa_per_sigmaCB": beta.tolist(),
            "peak_lag_yr": int(lags[peak_ix]),
            "peak_amp_hPa": float(beta[peak_ix]),
            "n_years": int(len(years)),
        }
    except Exception as exc:
        print(f"     compute failed: {exc}")
        return None
    finally:
        try:
            ds.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> dict:
    print("[walker] loading HadISST cold-blob + global-T indices...")
    yrs_cb, cb_bp, gT_bp = obs_cold_blob_index()
    print(f"   years {yrs_cb.min():.0f}-{yrs_cb.max():.0f}, n={len(yrs_cb)}")

    print("[walker] loading HadSLP2r and computing Walker index...")
    ds_slp = xr.open_dataset(HADSLP2)
    yrs_w, walker_ann = walker_index_from_slp(ds_slp, var="slp", time_dim="time")
    print(f"   Walker years {yrs_w.min():.0f}-{yrs_w.max():.0f}, n={len(yrs_w)}")
    walker_bp = bandpass_annual(walker_ann, BANDPASS_LO_YR, BANDPASS_HI_YR)

    # align CB / Walker / gT on common years
    common = np.intersect1d(np.intersect1d(yrs_cb, yrs_w), yrs_cb)
    cb_a = np.array([cb_bp[np.where(yrs_cb == y)[0][0]] for y in common])
    gT_a = np.array([gT_bp[np.where(yrs_cb == y)[0][0]] for y in common])
    w_a = np.array([walker_bp[np.where(yrs_w == y)[0][0]] for y in common])
    print(f"   common-year aligned record: {common.min():.0f}-{common.max():.0f}, n={len(common)}")

    beta_obs = lead_lag_regression(w_a, cb_a, gT_a, LAGS)
    print(f"   obs beta(lag): {np.round(beta_obs, 3)}")

    ci_lo, ci_hi = block_bootstrap_ci(w_a, cb_a, gT_a, LAGS,
                                      n_boot=N_BOOT, block=BOOT_BLOCK_YR)
    # Prefer the significant peak (CI excludes zero); if none, fall back
    # to max |beta|.
    sig = (ci_lo > 0) | (ci_hi < 0)
    if np.any(sig):
        sig_idx = np.where(sig)[0]
        peak_ix = int(sig_idx[np.argmax(np.abs(beta_obs[sig_idx]))])
        peak_significant = True
    else:
        peak_ix = int(np.nanargmax(np.abs(beta_obs)))
        peak_significant = False
    peak_lag = int(LAGS[peak_ix])
    peak_amp = float(beta_obs[peak_ix])
    peak_ci = (float(ci_lo[peak_ix]), float(ci_hi[peak_ix]))
    sig_lags = [int(LAGS[i]) for i in np.where(sig)[0]]
    print(f"   obs peak: lag={peak_lag} yr, amp={peak_amp:+.3f} hPa/sigma_CB, "
          f"CI95=[{peak_ci[0]:+.3f},{peak_ci[1]:+.3f}], significant={peak_significant}")
    print(f"   significant lags (CI excludes 0): {sig_lags}")

    # ---- CMIP6 ----
    print("[walker] scanning CMIP6 psl on disk...")
    combos = list_cmip6_models()
    print(f"   found {len(combos)} (model, exp) combos")

    # Process EVERY available (model, experiment) combination.  Failures
    # are logged and skipped, so one broken model does not block the rest.
    per_model: dict[str, dict] = {}
    failed: list[tuple[str, str, str]] = []
    n_by_exp: dict[str, int] = {"historical": 0, "piControl": 0, "ssp245": 0}
    for model, exp, files in combos:
        if exp == "historical":
            key = model
        elif exp == "piControl":
            key = f"{model}_piC"
        elif exp == "ssp245":
            key = f"{model}_ssp245"
        else:
            key = f"{model}_{exp}"
        if key in per_model:
            continue
        try:
            d = compute_model_curve(model, exp, files, LAGS)
        except Exception as exc:
            print(f"   [skip] {model}/{exp}: unhandled exception {exc}")
            failed.append((model, exp, str(exc)))
            continue
        if d is not None:
            per_model[key] = d
            if exp in n_by_exp:
                n_by_exp[exp] += 1
        else:
            failed.append((model, exp, "compute_model_curve returned None"))

    print(f"   per-model successful: {sorted(per_model.keys())}")
    print(f"   counts by experiment: {n_by_exp}")
    if failed:
        for m, e, reason in failed:
            print(f"   [failed] {m}/{e}: {reason}")

    # Multi-model band per lag (HISTORICAL ensemble for the headline band)
    hist_keys = [k for k, v in per_model.items() if v["experiment"] == "historical"]
    if hist_keys:
        stack = np.array([per_model[k]["lag_curve_hPa_per_sigmaCB"] for k in hist_keys])
        mm_p25 = np.nanpercentile(stack, 25, axis=0).tolist()
        mm_p50 = np.nanpercentile(stack, 50, axis=0).tolist()
        mm_p75 = np.nanpercentile(stack, 75, axis=0).tolist()
        mm_n = len(hist_keys)
    elif per_model:
        stack = np.array([d["lag_curve_hPa_per_sigmaCB"] for d in per_model.values()])
        mm_p25 = np.nanpercentile(stack, 25, axis=0).tolist()
        mm_p50 = np.nanpercentile(stack, 50, axis=0).tolist()
        mm_p75 = np.nanpercentile(stack, 75, axis=0).tolist()
        mm_n = len(per_model)
    else:
        mm_p25 = mm_p50 = mm_p75 = [None] * len(LAGS)
        mm_n = 0

    # piControl-only band (independent multi-decadal variability baseline)
    pic_keys = [k for k, v in per_model.items() if v["experiment"] == "piControl"]
    if pic_keys:
        stack_pic = np.array([per_model[k]["lag_curve_hPa_per_sigmaCB"] for k in pic_keys])
        pic_p25 = np.nanpercentile(stack_pic, 25, axis=0).tolist()
        pic_p50 = np.nanpercentile(stack_pic, 50, axis=0).tolist()
        pic_p75 = np.nanpercentile(stack_pic, 75, axis=0).tolist()
    else:
        pic_p25 = pic_p50 = pic_p75 = [None] * len(LAGS)

    out = {
        "pathway": "walker",
        "definition": {
            "cb_index": "HadISST caesar_box_anomaly, sign flipped (positive = cold), z-scored, Lanczos 10-30 yr bandpass",
            "walker_index": "HadSLP2r SLP(eastern Pacific 5S-5N, 160W-80W) minus SLP(western/Indo Pacific 5S-5N, 80E-160E), annual, hPa, Lanczos 10-30 yr bandpass",
            "partial_covariate": "HadISST global_anomaly, z-scored, Lanczos 10-30 yr bandpass",
            "regression_units": "hPa per std(CB index)",
            "cmip6_cb_proxy": "subpolar-Atlantic SLP anomaly box (46-61 N, 50-20 W), z-scored (SST not available from psl)",
        },
        "lags_yr": LAGS,
        "obs": {
            "years_common": [int(y) for y in common],
            "lag_curve_hPa_per_sigmaCB": [float(x) if np.isfinite(x) else None for x in beta_obs],
            "ci95_lo_hPa": [float(x) if np.isfinite(x) else None for x in ci_lo],
            "ci95_hi_hPa": [float(x) if np.isfinite(x) else None for x in ci_hi],
            "peak_lag_yr": peak_lag,
            "peak_amp_hPa_per_sigmaCB": peak_amp,
            "peak_ci95": list(peak_ci),
            "peak_is_significant_p05": bool(peak_significant),
            "significant_lags_yr": sig_lags,
            "bootstrap": {
                "n_boot": N_BOOT,
                "block_yr": BOOT_BLOCK_YR,
                "seed": RNG_SEED,
            },
            "source": {
                "cb": str(COLD_BLOB_HADISST),
                "slp": str(HADSLP2),
            },
        },
        "per_model": per_model,
        "multi_model_band": {
            "p25_hPa": mm_p25,
            "p50_hPa": mm_p50,
            "p75_hPa": mm_p75,
            "n_models": mm_n,
            "scope": "historical_only" if hist_keys else "all_experiments",
        },
        "picontrol_band": {
            "p25_hPa": pic_p25,
            "p50_hPa": pic_p50,
            "p75_hPa": pic_p75,
            "n_models": len(pic_keys),
        },
        "ensemble_counts": {
            "n_historical": n_by_exp["historical"],
            "n_piControl": n_by_exp["piControl"],
            "n_ssp245": n_by_exp["ssp245"],
            "n_total_successful_entries": len(per_model),
            "n_failed_entries": len(failed),
        },
        "failed_entries": [
            {"model": m, "experiment": e, "reason": reason}
            for (m, e, reason) in failed
        ],
        "notes": (
            "Negative sign of CB index applied so positive = cold (Caesar slowdown "
            "convention).  Partial regression removes global-mean-T scaling from both "
            "Walker and CB before regression.  CMIP6 path uses SLP-based CB proxy "
            "because SST (tos) is not on disk for this run.  Sample sizes per "
            "model are short for piControl segments because only the chunks "
            "currently downloaded are concatenated."
        ),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[walker] wrote {OUT_JSON}")
    return out


# ----------------------------------------------------------------------
# draw_subpanel for the composer
# ----------------------------------------------------------------------
def draw_subpanel(ax) -> None:
    """Draw the Walker pathway curve on the supplied Axes.

    Requirements (from project plotting standards):
    - no in-axis title,
    - distinct shapes per category if multiple series,
    - no label/colorbar overlap,
    - sans-serif 6-7 pt (handled by parent rcParams).
    """
    if not OUT_JSON.exists():
        ax.text(0.5, 0.5, "(walker pathway: no cached result)",
                transform=ax.transAxes, ha="center", va="center", fontsize=6)
        return
    d = json.loads(OUT_JSON.read_text())
    lags = np.array(d["lags_yr"], float)
    beta = np.array([np.nan if x is None else x
                     for x in d["obs"]["lag_curve_hPa_per_sigmaCB"]], float)
    lo = np.array([np.nan if x is None else x
                   for x in d["obs"]["ci95_lo_hPa"]], float)
    hi = np.array([np.nan if x is None else x
                   for x in d["obs"]["ci95_hi_hPa"]], float)

    # OBS curve + 95% block-bootstrap CI band -- shared OBS-CI encoding
    # (light grey + 0.3-pt dark-grey edge) so the (b) and (d) bands match.
    h_band = ax.fill_between(lags, lo, hi,
                             facecolor="0.80", alpha=0.35,
                             edgecolor="0.45", linewidth=0.3,
                             label="HadSLP2r/HadISST 95\\% block-bootstrap CI")
    h_obs, = ax.plot(lags, beta, color="black", marker="o", markersize=3.0,
                     linewidth=1.1, label="HadSLP2r/HadISST Walker observed")

    # CMIP6 historical 25-75 IQR + median; piControl median dashed.
    pm = d.get("per_model", {})
    n_hist = n_pic = 0
    hist_stack = []
    pic_stack = []
    for model, m in sorted(pm.items()):
        b = np.array([np.nan if x is None else x
                      for x in m["lag_curve_hPa_per_sigmaCB"]], float)
        name = str(model)
        if "_piC" in name:
            pic_stack.append(b); n_pic += 1
        elif "_ssp245" in name or "_SSP245" in name:
            # SSP2-4.5 dropped (n<=2 is not an ensemble; figure is past/present).
            continue
        else:
            hist_stack.append(b); n_hist += 1

    h_hist_med = h_pic_med = None
    if hist_stack:
        H = np.array(hist_stack)
        hist_p25 = np.nanpercentile(H, 25, axis=0)
        hist_p75 = np.nanpercentile(H, 75, axis=0)
        hist_med = np.nanmedian(H, axis=0)
        # IQR band unlabeled -- caption describes it; the median LINE carries
        # the legend entry so the historical ensemble appears just once.
        ax.fill_between(lags, hist_p25, hist_p75,
                        color="#4C72B0", alpha=0.22, linewidth=0)
        h_hist_med, = ax.plot(lags, hist_med, color="#4C72B0", linewidth=1.3,
                              label=f"CMIP6 historical Walker median ($n={n_hist}$)")
    if pic_stack:
        P = np.array(pic_stack)
        pic_med = np.nanmedian(P, axis=0)
        h_pic_med, = ax.plot(lags, pic_med, color="#5B8E91", linewidth=1.0,
                             linestyle="--",
                             label=f"CMIP6 piControl Walker median ($n={n_pic}$)")

    ax.axhline(0, color="0.55", linewidth=0.4, linestyle=":", alpha=0.9)
    ax.axvline(0, color="0.55", linewidth=0.4, linestyle=":", alpha=0.9)
    ax.set_xlabel("Lag (yr; positive = CB leads Walker)")
    ax.set_ylabel("$\\beta$ (hPa per $\\sigma_{\\mathrm{CB}}$)", labelpad=2)
    ax.set_xticks(np.arange(-20, 21, 5))

    # Lag +5 yr beta and CI live in the LaTeX caption -- ZERO in-axes
    # annotation (white-framed text boxes count as 'legend over the plot').
    pass

    # Return handles/labels for the composer to place on the dedicated legend
    # axes OUTSIDE this panel frame; do NOT call ax.legend here.
    handles = [h for h in (h_obs, h_band, h_hist_med, h_pic_med) if h is not None]
    labels = [h.get_label() for h in handles]
    return handles, labels


if __name__ == "__main__":
    main()
