"""
SI figure: Atlantic-to-Pacific Walker teleconnection -- deep dive (4 panels).

(a) Observed lead-lag regression curve beta(Walker, CB; lag) at lags
    -20..+20 yr from HadSLP2r / HadISST 1870-2019, with 95% block
    bootstrap CI (30-yr blocks).  Reads the cached curve in
    dcps/cache/fig7_panel_c/walker.json which was computed by
    compute_fig7_walker.py.

(b) Regression map of DJF SLP from HadSLP2r at each grid point onto the
    bandpassed standardised cold-blob index at lag +5 yr (1900-2019);
    units hPa per sigma_CB, with stippling at p<0.05 (block-bootstrap on
    autocorrelated series, Neff via Bretherton 1999).  Focus on the
    tropical Pacific (120E-80W, 30S-30N).  Expected pattern: positive
    SLP anomaly over eastern Pacific, negative over Maritime Continent
    -- the canonical La-Nina-like Walker strengthening.

(c) Sliding 30-yr-window CB-to-Walker slope from 1870 to 2019 (obs)
    plus the CMIP6 historical+ssp245 multi-model band for the same
    diagnostic, taken over whatever historical / ssp245 psl files are
    currently on disk.

(d) CMIP6-model scatter of per-model CB->Walker regression slope at
    lag +5 yr (y) versus the model's Fig.5 Q-effect underestimation
    factor (x) = Q_obs / Q_model_modern.  Distinct marker shape per
    institute family.

Tolerates missing CMIP6 models: if fewer than two are available, panel
(c) and (d) become "to be expanded as additional models become
available" placeholders, but panels (a) and (b) always render.

Output: manuscript/figs/fig_si_atl_pac_walker.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Springer Nature plotting standards
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "manuscript" / "figs" / "fig_si_atl_pac_walker.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

WALKER_CACHE = REPO / "dcps" / "cache" / "fig7_panel_c" / "walker.json"
HOL_CACHE = REPO / "dcps" / "cache" / "holocene_exit" / "bulk"
SI_CACHE = REPO / "dcps" / "cache" / "walker_si"
SI_CACHE.mkdir(parents=True, exist_ok=True)

HADSLP2 = Path("/home/bijanf/Documents/NEW_Theory/data/external/hadslp2/hadslp2r.nc")
COLD_BLOB_HADISST = Path(
    "/home/bijanf/Documents/AMOC_renalysis/data/results/cold_blob_timeseries_hadisst.nc"
)
CMIP6_ROOT = Path("/home/bijanf/Documents/NEW_Theory/data/external/cmip6_atmos")

# Lanczos band-pass like the headline diagnostic
BANDPASS_LO_YR = 10
BANDPASS_HI_YR = 30
LAG_FOR_MAP = 5
LAG_FOR_SCATTER = 5
WIN_YR = 30
N_BOOT = 1000
BOOT_BLOCK_YR = 30
RNG_SEED = 20260611

Q_OBS = 0.32  # observed North Atlantic Quiescence Index (main text)


# ---------------------------------------------------------------------------
# Filtering utilities (kept self-contained from compute_fig7_walker.py)
# ---------------------------------------------------------------------------
def lanczos_weights(window: int, cutoff_low: float, cutoff_high: float) -> np.ndarray:
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


def _slp_box_mean(ds: xr.Dataset, lat_s: float, lat_n: float,
                  lon_w: float, lon_e: float, var: str = "slp") -> xr.DataArray:
    da = ds[var]
    lat = da["lat"].values
    if lat[0] > lat[-1]:
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
        def to_180(x): return ((x + 180) % 360) - 180
        lw = to_180(lon_w); le = to_180(lon_e)
        if lw <= le:
            mask = (lon >= lw) & (lon <= le)
        else:
            mask = (lon >= lw) | (lon <= le)
    da_box = da_lat.isel(lon=np.where(mask)[0])
    coslat = np.cos(np.deg2rad(da_box["lat"]))
    w = coslat / coslat.sum()
    return (da_box.weighted(w).mean(dim="lat")).mean(dim="lon")


def walker_index_from_slp(ds: xr.Dataset, var: str = "slp",
                          time_dim: str = "time") -> tuple[np.ndarray, np.ndarray]:
    east = _slp_box_mean(ds, -5, 5, 200, 280, var=var)
    west = _slp_box_mean(ds, -5, 5, 80, 160, var=var)
    grad = east - west
    units = ds[var].attrs.get("units", "hPa").lower()
    if "pa" == units or units == "n/m2" or units == "n m-2":
        grad = grad / 100.0
    try:
        ann = grad.groupby(f"{time_dim}.year").mean(dim=time_dim)
        years = ann["year"].values.astype(float)
        vals = ann.values.astype(float)
    except Exception:
        t = ds[time_dim].values
        years_full = np.array([tt.year for tt in t])
        v = grad.values
        years = np.unique(years_full).astype(float)
        vals = np.array([np.nanmean(v[years_full == y]) for y in years])
    return years, vals


def cmip6_cold_blob_psl_proxy(ds: xr.Dataset, var: str = "psl",
                              time_dim: str = "time") -> tuple[np.ndarray, np.ndarray]:
    box = _slp_box_mean(ds, 46, 61, 310, 340, var=var)
    units = ds[var].attrs.get("units", "hPa").lower()
    if units == "pa":
        box = box / 100.0
    try:
        ann = box.groupby(f"{time_dim}.year").mean(dim=time_dim)
        years = ann["year"].values.astype(float)
        vals = ann.values.astype(float)
    except Exception:
        t = ds[time_dim].values
        years_full = np.array([tt.year for tt in t])
        v = box.values
        years = np.unique(years_full).astype(float)
        vals = np.array([np.nanmean(v[years_full == y]) for y in years])
    z = (vals - np.nanmean(vals)) / np.nanstd(vals)
    return years, z


# ---------------------------------------------------------------------------
# Observational CB / Walker / globalT (panel a uses cache; panel b/c rebuild)
# ---------------------------------------------------------------------------
def obs_cold_blob_index() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(COLD_BLOB_HADISST)
    years = ds["year"].values.astype(float)
    cb_raw = ds["caesar_box_anomaly"].values.astype(float)
    gt_raw = ds["global_anomaly"].values.astype(float)
    cb = -cb_raw  # positive = cold
    cb_z = (cb - np.nanmean(cb)) / np.nanstd(cb)
    gt_z = (gt_raw - np.nanmean(gt_raw)) / np.nanstd(gt_raw)
    return years, bandpass_annual(cb_z, BANDPASS_LO_YR, BANDPASS_HI_YR), \
        bandpass_annual(gt_z, BANDPASS_LO_YR, BANDPASS_HI_YR)


def partial_residual(y: np.ndarray, c: np.ndarray) -> np.ndarray:
    m = np.isfinite(y) & np.isfinite(c)
    if m.sum() < 5:
        return np.full_like(y, np.nan, dtype=float)
    a, b = np.polyfit(c[m], y[m], 1)
    res = y - (a * c + b)
    res[~m] = np.nan
    return res


def lead_lag_slope(walker: np.ndarray, cb: np.ndarray, gT: np.ndarray,
                   lag: int) -> float:
    """Return single beta at one specific lag (positive lag = CB leads W)."""
    w_res = partial_residual(walker, gT)
    c_res = partial_residual(cb, gT)
    n = len(walker)
    if lag >= 0:
        w_seg = w_res[lag:]; c_seg = c_res[:n - lag]
    else:
        w_seg = w_res[:n + lag]; c_seg = c_res[-lag:]
    m = np.isfinite(w_seg) & np.isfinite(c_seg)
    if m.sum() < 10:
        return np.nan
    slope, _ = np.polyfit(c_seg[m], w_seg[m], 1)
    return float(slope)


# ---------------------------------------------------------------------------
# Panel (b): per-grid regression map with Neff-aware significance
# ---------------------------------------------------------------------------
def lag1_autocorr(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return 0.0
    xm = x - x.mean()
    num = np.sum(xm[:-1] * xm[1:])
    den = np.sum(xm * xm)
    if den <= 0:
        return 0.0
    return float(num / den)


def neff_bretherton(x: np.ndarray, y: np.ndarray) -> int:
    """Bretherton 1999 effective sample size for two AR(1) series."""
    n = len(x)
    rx = lag1_autocorr(x); ry = lag1_autocorr(y)
    rx = max(-0.95, min(0.95, rx))
    ry = max(-0.95, min(0.95, ry))
    neff = n * (1 - rx * ry) / (1 + rx * ry)
    return max(5, int(neff))


def regression_pvalue_neff(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS slope of y on x and a two-sided p-value using a t-test with
    Bretherton effective sample size (treats x, y as AR(1) autocorrelated)."""
    from math import erf, sqrt
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return np.nan, np.nan
    xv = x[m]; yv = y[m]
    xm = xv - xv.mean(); ym = yv - yv.mean()
    sxx = np.sum(xm * xm)
    if sxx <= 0:
        return np.nan, np.nan
    slope = float(np.sum(xm * ym) / sxx)
    yhat = slope * xm
    resid = ym - yhat
    neff = neff_bretherton(xv, yv)
    dof = max(2, neff - 2)
    sigma2 = np.sum(resid ** 2) / dof
    se = float(np.sqrt(sigma2 / sxx))
    if se <= 0:
        return slope, np.nan
    t = slope / se
    # 2-sided normal approximation (good for neff>=20)
    p = 2 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))
    return slope, float(p)


def block_bootstrap_pvalue(x: np.ndarray, y: np.ndarray, slope_obs: float,
                           block_yr: int = 30, n_boot: int = 500,
                           rng: np.random.Generator | None = None
                           ) -> float:
    """Two-sided block-bootstrap p-value for the slope of y on x.  We
    resample y in 30-yr blocks (preserving autocorrelation) while keeping
    x fixed; the bootstrap distribution is the slope under the null of
    arbitrary autocorrelated noise.  p = fraction of |null slope| >=
    |observed slope|."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return np.nan
    xv = x[m]; yv = y[m]
    n = len(xv)
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    # how many blocks needed to fill n with margin
    n_blocks = max(1, int(np.ceil(n / block_yr)) + 1)
    xm = xv - xv.mean()
    sxx = np.sum(xm * xm)
    if sxx <= 0:
        return np.nan
    null_slopes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block_yr + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_yr) for s in starts])[:n]
        yb = yv[idx]
        ybm = yb - yb.mean()
        null_slopes[b] = float(np.sum(xm * ybm) / sxx)
    return float((np.abs(null_slopes) >= abs(slope_obs)).mean())


def compute_djf_regression_map(ds_slp: xr.Dataset, cb_years: np.ndarray,
                               cb_bp_z: np.ndarray,
                               years_lo: int = 1900,
                               years_hi: int = 2019,
                               lag: int = LAG_FOR_MAP) -> xr.Dataset:
    """
    Annual DJF mean SLP -> regress at each grid cell onto bandpassed
    standardised CB at the same point in time but with CB shifted by lag.
    Positive lag = CB leads SLP.  Output beta in hPa per sigma_CB,
    and p-value map.
    """
    cache = SI_CACHE / "djf_regression_map_lag5.nc"
    if cache.exists():
        try:
            return xr.open_dataset(cache)
        except Exception:
            pass

    slp = ds_slp["slp"]
    # DJF mean indexed by Dec-year; e.g. 1900 winter -> Dec 1900 + Jan/Feb 1901
    months = slp["time"].dt.month
    djf_mask = months.isin([12, 1, 2])
    slp_djf = slp.where(djf_mask)
    # Group: take Dec-Jan-Feb mean labelled by the Dec year
    year_of = xr.where(months == 12, slp["time"].dt.year, slp["time"].dt.year - 1)
    slp_djf = slp_djf.assign_coords(year=year_of)
    djf_ann = slp_djf.groupby("year").mean(dim="time")
    yrs_slp = djf_ann["year"].values.astype(float)

    # bandpass each grid cell
    djf_vals = djf_ann.values  # (year, lat, lon)
    nlat, nlon = djf_vals.shape[1], djf_vals.shape[2]
    djf_bp = np.full_like(djf_vals, np.nan, dtype=float)
    for ilat in range(nlat):
        for ilon in range(nlon):
            ts = djf_vals[:, ilat, ilon].astype(float)
            if np.isfinite(ts).sum() < 30:
                continue
            djf_bp[:, ilat, ilon] = bandpass_annual(ts, BANDPASS_LO_YR,
                                                   BANDPASS_HI_YR)

    # align CB and SLP on common years within [years_lo, years_hi]
    common = np.intersect1d(yrs_slp, cb_years)
    common = common[(common >= years_lo) & (common <= years_hi - lag)]
    # SLP at time t (target), CB at time t-lag (leads by `lag`)
    slp_idx = np.array([np.where(yrs_slp == y)[0][0] for y in common])
    cb_idx = np.array([np.where(cb_years == (y - lag))[0][0] for y in common
                       if (y - lag) in cb_years])
    common = common[: len(cb_idx)]
    slp_idx = slp_idx[: len(cb_idx)]

    cb_seg = cb_bp_z[cb_idx]
    # cb_seg is already z-scored bandpassed, but limit to its in-band finite range
    finite_cb = np.isfinite(cb_seg)

    rng = np.random.default_rng(RNG_SEED)
    beta_map = np.full((nlat, nlon), np.nan)
    p_map = np.full((nlat, nlon), np.nan)
    for ilat in range(nlat):
        for ilon in range(nlon):
            slp_seg = djf_bp[slp_idx, ilat, ilon]
            m = finite_cb & np.isfinite(slp_seg)
            if m.sum() < 30:
                continue
            beta, _ = regression_pvalue_neff(cb_seg[m], slp_seg[m])
            beta_map[ilat, ilon] = beta
            # Block-bootstrap p (preserves autocorrelation in y; x kept
            # fixed); 500 resamples per cell.
            p_map[ilat, ilon] = block_bootstrap_pvalue(
                cb_seg[m], slp_seg[m], beta,
                block_yr=BOOT_BLOCK_YR, n_boot=500, rng=rng,
            )

    out = xr.Dataset(
        {
            "beta_hPa_per_sigmaCB": (("lat", "lon"), beta_map),
            "p_value": (("lat", "lon"), p_map),
        },
        coords={"lat": djf_ann["lat"].values,
                "lon": djf_ann["lon"].values},
        attrs={"lag_yr": lag, "years_used": f"{int(common.min())}-{int(common.max())}"},
    )
    try:
        out.to_netcdf(cache)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Panel (c): sliding-window CB->Walker slope at lag +5
# ---------------------------------------------------------------------------
def sliding_window_slope(cb_bp: np.ndarray, walker_bp: np.ndarray,
                         gT_bp: np.ndarray, years: np.ndarray,
                         win: int = WIN_YR, lag: int = LAG_FOR_SCATTER,
                         min_finite: int = 18,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window slope of Walker on CB (with gT partialled out), at
    a fixed lag.  Window labels are CENTRE years."""
    n = len(years)
    centres, slopes = [], []
    half = win // 2
    for i in range(half, n - half):
        ws = slice(i - half, i + half + 1)
        cb_seg = cb_bp[ws]; w_seg = walker_bp[ws]; gT_seg = gT_bp[ws]
        # apply lag inside the window
        if lag >= 0:
            w_l = w_seg[lag:]; c_l = cb_seg[:len(cb_seg) - lag]; g_l = gT_seg[:len(gT_seg) - lag]
        else:
            w_l = w_seg[:len(w_seg) + lag]; c_l = cb_seg[-lag:]; g_l = gT_seg[-lag:]
        c_res = partial_residual(c_l, g_l)
        w_res = partial_residual(w_l, g_l)
        m = np.isfinite(c_res) & np.isfinite(w_res)
        if m.sum() < min_finite:
            continue
        try:
            slope, _ = np.polyfit(c_res[m], w_res[m], 1)
            centres.append(years[i])
            slopes.append(float(slope))
        except Exception:
            continue
    return np.array(centres, float), np.array(slopes, float)


def load_cmip6_model_psl(model: str, exp: str) -> xr.Dataset | None:
    psl_dir = CMIP6_ROOT / exp / model / "Amon" / "psl"
    if not psl_dir.exists():
        return None
    files = sorted(psl_dir.glob("psl_*.nc"))
    if not files:
        return None
    try:
        if len(files) == 1:
            return xr.open_dataset(files[0], use_cftime=True)
        return xr.open_mfdataset([str(f) for f in files],
                                 combine="by_coords", use_cftime=True,
                                 parallel=False)
    except Exception as exc:
        print(f"   load failed {model} {exp}: {exc}")
        return None


def model_sliding_curve(model: str) -> dict | None:
    """Build sliding-window CB->Walker slope curve from concatenated
    historical+ssp245 psl for one CMIP6 model.  Returns dict with
    centres, slopes, plus the scalar slope at lag +5 over the 1900-2019
    matching window (for panel d)."""
    ds_h = load_cmip6_model_psl(model, "historical")
    if ds_h is None:
        return None
    ds_s = load_cmip6_model_psl(model, "ssp245")

    def annual_walker_cb(ds):
        wy, w = walker_index_from_slp(ds, var="psl", time_dim="time")
        cy, c = cmip6_cold_blob_psl_proxy(ds, var="psl", time_dim="time")
        gT_box = _slp_box_mean(ds, -60, 60, 0, 360, var="psl")
        try:
            gT_ann = gT_box.groupby("time.year").mean("time")
            gy = gT_ann["year"].values.astype(float)
            gT = gT_ann.values.astype(float)
        except Exception:
            t = ds["time"].values
            years_full = np.array([tt.year for tt in t])
            v = gT_box.values
            gy = np.unique(years_full).astype(float)
            gT = np.array([np.nanmean(v[years_full == y]) for y in gy])
        common = np.intersect1d(np.intersect1d(wy, cy), gy)
        wmask = np.isin(wy, common); cmask = np.isin(cy, common); gmask = np.isin(gy, common)
        return common, w[wmask], c[cmask], gT[gmask]

    yh, wh, ch, gh = annual_walker_cb(ds_h)
    try:
        ds_h.close()
    except Exception:
        pass
    if ds_s is not None:
        ys, ws, cs_, gs = annual_walker_cb(ds_s)
        try:
            ds_s.close()
        except Exception:
            pass
        # ssp245 starts in 2015; historical typically ends 2014
        years = np.concatenate([yh, ys[ys > yh.max()]])
        walker = np.concatenate([wh, ws[ys > yh.max()]])
        cb = np.concatenate([ch, cs_[ys > yh.max()]])
        gT = np.concatenate([gh, gs[ys > yh.max()]])
    else:
        years, walker, cb, gT = yh, wh, ch, gh

    walker_bp = bandpass_annual(walker, BANDPASS_LO_YR, BANDPASS_HI_YR)
    cb_bp = bandpass_annual(cb, BANDPASS_LO_YR, BANDPASS_HI_YR)
    gT_bp = bandpass_annual(gT, BANDPASS_LO_YR, BANDPASS_HI_YR)

    centres, slopes = sliding_window_slope(cb_bp, walker_bp, gT_bp, years,
                                           win=WIN_YR, lag=LAG_FOR_SCATTER)
    # Long-record scalar slope at lag +5 over 1900-2019 (or full overlap if shorter)
    m_full = (years >= 1900) & (years <= 2019)
    if m_full.sum() >= 30:
        s_lag5 = lead_lag_slope(walker_bp[m_full], cb_bp[m_full], gT_bp[m_full],
                                LAG_FOR_SCATTER)
    else:
        s_lag5 = lead_lag_slope(walker_bp, cb_bp, gT_bp, LAG_FOR_SCATTER)

    return dict(model=model, centres=centres.tolist(),
                slopes=slopes.tolist(),
                slope_lag5=s_lag5, n_years=int(len(years)),
                year_range=(int(years.min()), int(years.max())))


def collect_cmip6_models(force: bool = False) -> dict:
    """Discover and compute sliding-window curves for every model with
    psl historical (and ssp245 if present).  When the on-disk inventory
    grows, the previous cache becomes stale; we therefore invalidate the
    cache automatically if the set of historical models has changed."""
    cache = SI_CACHE / "cmip6_walker_sliding.json"
    hist_models = sorted(
        [d.name for d in (CMIP6_ROOT / "historical").glob("*")
         if (d / "Amon" / "psl").exists()
         and any((d / "Amon" / "psl").glob("psl_*.nc"))]
    )
    if cache.exists() and not force:
        try:
            cached = json.loads(cache.read_text())
            if set(cached.keys()) >= set(hist_models):
                return cached
            print(f"   cache miss: on-disk historical models {hist_models} "
                  f"not all present in cached set {sorted(cached.keys())}; "
                  f"rebuilding.")
        except Exception:
            pass
    out: dict[str, dict] = {}
    for model in hist_models:
        print(f"  panel (c)/(d) sliding curve: {model}")
        try:
            rec = model_sliding_curve(model)
        except Exception as exc:
            print(f"    [skip] {model}: {exc}")
            continue
        if rec is not None and len(rec["centres"]) > 0:
            out[model] = rec
    cache.write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# Panel (d): Q-effect underestimation factor per model
# ---------------------------------------------------------------------------
def q_underestimation_factor(model: str) -> float | None:
    f = HOL_CACHE / f"{model}_atlantic.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    hist_q = d.get("hist_Q", [])
    if not hist_q:
        return None
    # Use the median of the last 3 windows (i.e. the modern epoch) as
    # the model's representative Q.  Same convention as the main-text
    # multi-basin Q reporting (Fig 5).
    modern = [v for v in hist_q[-3:] if v is not None and np.isfinite(v)]
    if not modern:
        return None
    q_model = float(np.median(modern))
    if q_model <= 0:
        # The underestimation factor is only physically meaningful for a
        # positive Q.  Models with Q <= 0 are off-scale; we cap them at
        # a generous upper bound so they still appear on the scatter but
        # at the rightmost margin.
        return 10.0
    return Q_OBS / q_model


# ---------------------------------------------------------------------------
# Institute family marker mapping (distinct shapes per family)
# ---------------------------------------------------------------------------
INSTITUTE_FAMILY = {
    "CESM2": "NCAR", "CESM2-WACCM": "NCAR",
    "CanESM5": "CCCma", "CanESM5-CanOE": "CCCma",
    "UKESM1-0-LL": "MOHC", "HadGEM3-GC31-LL": "MOHC", "HadGEM3-GC31-MM": "MOHC",
    "CNRM-CM6-1": "CNRM", "CNRM-CM6-1-HR": "CNRM", "CNRM-ESM2-1": "CNRM",
    "MPI-ESM1-2-LR": "MPI-M", "MPI-ESM1-2-HR": "MPI-M",
    "IPSL-CM6A-LR": "IPSL",
    "ACCESS-CM2": "CSIRO",
    "GFDL-ESM4": "NOAA-GFDL",
    "MIROC6": "MIROC", "MIROC-ES2L": "MIROC",
    "GISS-E2-1-G": "NASA-GISS",
    "BCC-CSM2-MR": "BCC", "CAMS-CSM1-0": "CAMS",
    "MRI-ESM2-0": "MRI",
    "NorESM2-MM": "NorESMHub",
    "INM-CM4-8": "INM", "INM-CM5-0": "INM",
    "EC-Earth3": "EC-Earth-Cons", "EC-Earth3-Veg-LR": "EC-Earth-Cons",
    "FGOALS-g3": "CAS",
    "FIO-ESM-2-0": "FIO",
    "NESM3": "NUIST",
    "E3SM-1-1": "E3SM-Project",
    "CMCC-CM2-SR5": "CMCC",
    "CIESM": "THU",
}
FAMILY_MARKERS = [
    "o", "s", "^", "D", "v", "P", "X", "*",
    "<", ">", "h", "p", "H", "8", "d",
]


def family_marker(model: str, family_order: list[str]) -> str:
    fam = INSTITUTE_FAMILY.get(model, model)
    if fam not in family_order:
        family_order.append(fam)
    i = family_order.index(fam)
    return FAMILY_MARKERS[i % len(FAMILY_MARKERS)]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def draw_panel_a(ax) -> None:
    """Lead-lag curve from the cached HadSLP2r/HadISST result."""
    if not WALKER_CACHE.exists():
        ax.text(0.5, 0.5, "(no cached HadSLP2r walker result)",
                transform=ax.transAxes, ha="center", va="center")
        return
    d = json.loads(WALKER_CACHE.read_text())
    lags = np.array(d["lags_yr"], float)
    beta = np.array(d["obs"]["lag_curve_hPa_per_sigmaCB"], float)
    lo = np.array(d["obs"]["ci95_lo_hPa"], float)
    hi = np.array(d["obs"]["ci95_hi_hPa"], float)
    ax.fill_between(lags, lo, hi, color="0.78", alpha=0.6,
                    linewidth=0, label="95\\% block-bootstrap CI")
    ax.plot(lags, beta, color="black", marker="o", markersize=3.6,
            linewidth=1.1, label="HadSLP2r / HadISST 1870--2019")
    sig_lags = d["obs"].get("significant_lags_yr", [])
    if sig_lags:
        for sl in sig_lags:
            i = int(np.where(lags == sl)[0][0])
            ax.plot([sl], [beta[i]], marker="o", markersize=6,
                    markerfacecolor="none", markeredgecolor="#C44E52",
                    markeredgewidth=1.1, zorder=4)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.6)
    ax.axvline(0, color="black", linewidth=0.4, linestyle=":", alpha=0.6)
    ax.axvline(LAG_FOR_MAP, color="#4C72B0", linewidth=0.6, linestyle="--",
               alpha=0.7, label=f"lag = +{LAG_FOR_MAP} yr (panel b)")
    ax.set_xticks(lags)
    ax.set_xlabel("Lag (yr) -- positive = cold blob leads Walker")
    ax.set_ylabel(r"$\beta$  (hPa per $\sigma_{\mathrm{CB}}$)")
    ax.legend(loc="lower center", frameon=False, fontsize=5.2,
              handletextpad=0.5, borderpad=0.2)


def draw_panel_b(ax, reg_ds: xr.Dataset) -> None:
    """Tropical-Pacific DJF SLP regression map at lag +5 yr."""
    beta = reg_ds["beta_hPa_per_sigmaCB"].values
    p = reg_ds["p_value"].values
    lat = reg_ds["lat"].values
    lon = reg_ds["lon"].values

    # focus 30S-30N, 120E-280E (=80W)
    lat_mask = (lat >= -30) & (lat <= 30)
    lon_mask = (lon >= 120) & (lon <= 280)
    lat_sel = lat[lat_mask]
    lon_sel = lon[lon_mask]
    beta_sel = beta[np.ix_(lat_mask, lon_mask)]
    p_sel = p[np.ix_(lat_mask, lon_mask)]

    vmax = float(np.nanpercentile(np.abs(beta_sel), 95))
    vmax = max(vmax, 0.05)
    levels = np.linspace(-vmax, vmax, 13)
    cf = ax.contourf(lon_sel, lat_sel, beta_sel, levels=levels,
                     cmap="RdBu_r", extend="both")
    # stippling at p<0.05 (block-bootstrap)
    sig = p_sel < 0.05
    if sig.any():
        LON, LAT = np.meshgrid(lon_sel, lat_sel)
        ax.scatter(LON[sig], LAT[sig], s=4.0, c="black",
                   marker=".", alpha=0.9, linewidths=0, zorder=4)
    # Walker-index east and west boxes for context
    for (lw, le) in [(80, 160), (200, 280)]:
        ax.plot([lw, le, le, lw, lw], [-5, -5, 5, 5, -5],
                color="black", linewidth=0.6, linestyle="--", alpha=0.85)
    ax.set_xlabel("Longitude ($^\\circ$E)")
    ax.set_ylabel("Latitude ($^\\circ$)")
    ax.set_xticks([120, 160, 200, 240, 280])
    ax.set_xticklabels(["120$^\\circ$E", "160$^\\circ$E", "160$^\\circ$W",
                        "120$^\\circ$W", "80$^\\circ$W"])
    ax.set_yticks([-30, -15, 0, 15, 30])
    ax.set_ylim(-30, 30)
    ax.set_xlim(120, 280)
    cb = plt.colorbar(cf, ax=ax, orientation="vertical",
                      fraction=0.04, pad=0.02)
    cb.set_label(r"$\beta_{\mathrm{SLP}}$  (hPa per $\sigma_{\mathrm{CB}}$)")
    cb.ax.tick_params(labelsize=5)


def draw_panel_c(ax, obs_centres, obs_slopes, models: dict) -> None:
    """Sliding-window slope, obs + CMIP6 multi-model band."""
    # Obs
    ax.plot(obs_centres, obs_slopes, color="black", linewidth=1.2,
            marker="o", markersize=2.5, markevery=5,
            label=f"HadSLP2r/HadISST (30-yr window, lag +{LAG_FOR_SCATTER} yr)")
    # CMIP6 band -- align onto a common centre grid
    if models:
        all_c = np.unique(np.concatenate(
            [np.array(m["centres"]) for m in models.values()
             if len(m["centres"]) > 0]
        ))
        stack = []
        for m in models.values():
            cs = np.array(m["centres"]); ss = np.array(m["slopes"])
            interp = np.full(len(all_c), np.nan)
            if len(cs) >= 2:
                interp = np.interp(all_c, cs, ss, left=np.nan, right=np.nan)
            else:
                # single-point fallback
                idx = np.searchsorted(all_c, cs)
                idx = np.clip(idx, 0, len(all_c) - 1)
                interp[idx] = ss
            stack.append(interp)
        stack = np.vstack(stack)
        p25 = np.nanpercentile(stack, 25, axis=0)
        p50 = np.nanpercentile(stack, 50, axis=0)
        p75 = np.nanpercentile(stack, 75, axis=0)
        ax.fill_between(all_c, p25, p75, color="#4C72B0",
                        alpha=0.22, linewidth=0,
                        label=f"CMIP6 25--75\\% (n={len(models)})")
        ax.plot(all_c, p50, color="#4C72B0", linewidth=1.0,
                label="CMIP6 median")
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.6)
    ax.set_xlabel("Window centre year")
    ax.set_ylabel(r"$\beta_{\mathrm{Walker} \leftarrow \mathrm{CB}}$  "
                  r"(hPa per $\sigma_{\mathrm{CB}}$, lag +5 yr)")
    ax.legend(loc="upper left", frameon=False, fontsize=5.5)


def _spearman_rho(xs: np.ndarray, ys: np.ndarray) -> tuple[float, int]:
    """Pearson correlation on ranks = Spearman rho.  Returns (rho, n_used)."""
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 3:
        return float("nan"), int(m.sum())
    xr = np.argsort(np.argsort(xs[m])).astype(float)
    yr = np.argsort(np.argsort(ys[m])).astype(float)
    xr -= xr.mean(); yr -= yr.mean()
    denom = np.sqrt((xr * xr).sum() * (yr * yr).sum())
    if denom <= 0:
        return float("nan"), int(m.sum())
    return float((xr * yr).sum() / denom), int(m.sum())


def draw_panel_d(ax, models: dict) -> tuple[list[str], dict[str, str]]:
    """Per-model scatter: Q-effect underestimation factor (x) vs
    sliding-window slope at lag +5 yr (y)."""
    if not models:
        ax.text(0.5, 0.5,
                "to be expanded as additional models\nbecome available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=6.5, color="0.4")
        return [], {}
    family_order: list[str] = []
    family_for_model: dict[str, str] = {}
    xs, ys, names = [], [], []
    for model, rec in sorted(models.items()):
        x = q_underestimation_factor(model)
        y = rec.get("slope_lag5", np.nan)
        if x is None or not np.isfinite(y):
            continue
        xs.append(x); ys.append(y); names.append(model)

    # Always persist whatever points we managed to build, even if too few
    # to plot, so downstream tools can introspect the run state.
    rho, n_used = _spearman_rho(np.array(xs, float), np.array(ys, float)) \
        if xs else (float("nan"), 0)
    (SI_CACHE / "walker_emergent_scatter.json").write_text(json.dumps({
        "x_label": "Q_obs / Q_model_modern (Fig.5 underestimation factor)",
        "y_label": "beta_Walker<-CB (hPa per sigma_CB, lag +5 yr)",
        "lag_yr": LAG_FOR_SCATTER,
        "models": names,
        "x": [float(x) for x in xs],
        "y": [float(y) for y in ys],
        "spearman_rho": rho if np.isfinite(rho) else None,
        "n_used": n_used,
        "n_total_models_available": len(models),
    }, indent=2))

    if not xs:
        ax.text(0.5, 0.5,
                "no CMIP6 model with both Q cache\nand Walker slope on disk",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=6.5, color="0.4")
        return [], {}

    # Plot
    plotted_fams: set[str] = set()
    for x, y, m in zip(xs, ys, names):
        mk = family_marker(m, family_order)
        fam = INSTITUTE_FAMILY.get(m, m)
        family_for_model[m] = fam
        ax.scatter([x], [y], s=55, marker=mk, facecolors="#DD8452",
                   edgecolors="black", linewidth=0.6, zorder=3,
                   label=fam if fam not in plotted_fams else None)
        plotted_fams.add(fam)
    # Observational anchor: x=1 means Q_model == Q_obs == 0.32; the obs
    # slope at lag +5 (cached in walker.json)
    if WALKER_CACHE.exists():
        d = json.loads(WALKER_CACHE.read_text())
        lags = np.array(d["lags_yr"], float)
        beta = np.array(d["obs"]["lag_curve_hPa_per_sigmaCB"], float)
        i5 = int(np.where(lags == LAG_FOR_SCATTER)[0][0])
        ax.axhline(beta[i5], color="black", linestyle="--", linewidth=0.6,
                   alpha=0.7,
                   label=f"Obs slope ({beta[i5]:+.3f}) at lag +{LAG_FOR_SCATTER} yr")
    ax.axvline(1.0, color="grey", linestyle=":", linewidth=0.6,
               label="$Q_{\\mathrm{model}}=Q_{\\mathrm{obs}}$")
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.6)

    ax.set_xlabel(r"Q-effect underestimation factor  "
                  r"$Q_{\mathrm{obs}}/Q_{\mathrm{model}}$ "
                  "(Fig.\\,5)")
    ax.set_ylabel(r"$\beta_{\mathrm{Walker} \leftarrow \mathrm{CB}}$  "
                  r"(hPa per $\sigma_{\mathrm{CB}}$, lag +5 yr)")
    # widen the x-axis a touch so the rightmost point isn't cropped and
    # the legend has room
    xs_arr = np.array(xs); ys_arr = np.array(ys)
    x_lo, x_hi = float(xs_arr.min()), float(xs_arr.max())
    pad_x = max(0.4, 0.15 * (x_hi - x_lo))
    ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
    # dedupe legend
    h, lab = ax.get_legend_handles_labels()
    seen = set(); h_u, l_u = [], []
    for hh, ll in zip(h, lab):
        if ll not in seen:
            seen.add(ll); h_u.append(hh); l_u.append(ll)
    ax.legend(h_u, l_u, loc="upper right", frameon=False, fontsize=5,
              handletextpad=0.4, borderpad=0.3)
    # Spearman rho annotation in lower-left so it does not overlap legend
    if np.isfinite(rho) and len(xs) >= 3:
        ax.text(0.03, 0.04,
                rf"Spearman $\rho={rho:+.2f}$  (n={n_used})",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=5.5, color="0.15")
    return names, family_for_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[walker SI] obs CB index...")
    yrs_cb, cb_bp, gT_bp = obs_cold_blob_index()
    print(f"   CB years {int(yrs_cb.min())}-{int(yrs_cb.max())}, n={len(yrs_cb)}")

    print("[walker SI] HadSLP2r DJF regression map at lag +5...")
    # Need ds_slp open for both panel (b) and panel (c) observational sliding
    ds_slp = xr.open_dataset(HADSLP2)
    reg_ds = compute_djf_regression_map(ds_slp, yrs_cb, cb_bp,
                                        years_lo=1900, years_hi=2019,
                                        lag=LAG_FOR_MAP)

    # Panel (c) observational sliding-window slope, full HadSLP2r record
    print("[walker SI] obs sliding-window slope...")
    yrs_w, walker_ann = walker_index_from_slp(ds_slp, var="slp", time_dim="time")
    walker_bp = bandpass_annual(walker_ann, BANDPASS_LO_YR, BANDPASS_HI_YR)
    common = np.intersect1d(yrs_cb, yrs_w)
    cb_a = np.array([cb_bp[np.where(yrs_cb == y)[0][0]] for y in common])
    gT_a = np.array([gT_bp[np.where(yrs_cb == y)[0][0]] for y in common])
    w_a = np.array([walker_bp[np.where(yrs_w == y)[0][0]] for y in common])
    obs_centres, obs_slopes = sliding_window_slope(cb_a, w_a, gT_a, common,
                                                   win=WIN_YR,
                                                   lag=LAG_FOR_SCATTER)
    try:
        ds_slp.close()
    except Exception:
        pass

    # CMIP6 model curves
    print("[walker SI] CMIP6 historical+ssp245 sliding curves...")
    models = collect_cmip6_models()
    print(f"   CMIP6 sliding-window curves built for: {sorted(models.keys())}")

    # Cache obs sliding curve
    (SI_CACHE / "obs_sliding_lag5.json").write_text(json.dumps(dict(
        centres=obs_centres.tolist(), slopes=obs_slopes.tolist(),
        window_yr=WIN_YR, lag_yr=LAG_FOR_SCATTER,
    ), indent=2))

    # ------------------------------------------------------------------
    # Figure layout: 2-column = 180 mm = 7.09 in
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(7.09, 6.0), constrained_layout=False)
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.075, right=0.985,
                           bottom=0.085, top=0.97,
                           wspace=0.32, hspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b, reg_ds)
    draw_panel_c(ax_c, obs_centres, obs_slopes, models)
    draw_panel_d(ax_d, models)

    # Panel letters as corner text only (NO in-axis titles); panel-(b)
    # is placed outside the data area so it does not overlap the
    # blue tropical-Pacific cells or the y-axis label.
    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        if letter == "b":
            ax.text(-0.13, 1.04, letter, transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=9, fontweight="bold")
        else:
            ax.text(0.02, 0.98, letter, transform=ax.transAxes,
                    ha="left", va="top", fontsize=9, fontweight="bold")

    fig.savefig(OUT, bbox_inches="tight")
    print(f"[walker SI] wrote {OUT}")


if __name__ == "__main__":
    main()
