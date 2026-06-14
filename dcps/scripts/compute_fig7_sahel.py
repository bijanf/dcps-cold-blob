"""Fig. 7 panel (c): SAHEL pathway.

Compute the Sahel JJAS precipitation regression onto the bandpass-filtered
HadISST cold-blob box index (Caesar 46-61 N, 50-20 W -> here 50-56 N,
45-25 W as stored in the HadISST file), after removing the global-mean
SST anomaly (i.e. partialling out global warming).

For CMIP6 we use historical pr from priority models stored at
/home/bijanf/Documents/NEW_Theory/data/external/cmip6_atmos/historical/
<model>/Amon/pr/*.nc with a model-internal cold-blob SST index built from
tas as a proxy (i.e. <tas> over the same 50-56 N, 45-25 W box on the model
grid). The model index is bandpass-filtered identically to obs.

Sahel region: 10-20 N, 20 W-10 E, Jun-Sep monthly precipitation averaged
to JJAS annual mean.

Outputs:
  /home/bijanf/Documents/NEW_Theory/dcps/cache/fig7_panel_c/sahel.json
  /home/bijanf/Documents/NEW_Theory/dcps/cache/fig7_panel_c/sahel.npz

A draw_subpanel(ax) is also exported for the composing agent.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from scipy import signal

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO = Path("/home/bijanf/Documents/NEW_Theory")
HADISST = Path("/home/bijanf/Documents/AMOC_renalysis/data/results/cold_blob_timeseries_hadisst.nc")
CMIP6_ROOT = REPO / "data" / "external" / "cmip6_atmos" / "historical"
OUT_DIR = REPO / "dcps" / "cache" / "fig7_panel_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "sahel.json"
OUT_NPZ = OUT_DIR / "sahel.npz"

# Cold-blob box: use the same box stored in the HadISST file (50-56 N, 45-25 W)
# for the model-internal index, to keep the same "what" being regressed.
CB_LAT = (50.0, 56.0)
CB_LON_WEST = -45.0  # -180..180
CB_LON_EAST = -25.0

# Sahel JJAS
SAHEL_LAT = (10.0, 20.0)
SAHEL_LON_WEST = -20.0
SAHEL_LON_EAST = 10.0

# Bandpass: AMOC/cold-blob multi-decadal teleconnection: keep 10-80 yr
BANDPASS_LOW_YR = 10.0
BANDPASS_HIGH_YR = 80.0

# Regression epoch (annual)
EPOCH = (1900, 2014)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _bandpass_yearly(x: np.ndarray, low_yr: float, high_yr: float) -> np.ndarray:
    """Butterworth 4-pole bandpass; periods in years; annual sampling."""
    nyq = 0.5  # cycles per year
    low = 1.0 / high_yr / nyq  # high period -> low cutoff freq
    high = 1.0 / low_yr / nyq  # low period -> high cutoff freq
    b, a = signal.butter(4, [low, high], btype="band")
    valid = np.isfinite(x)
    if valid.sum() < 30:
        return np.full_like(x, np.nan, dtype=float)
    xf = np.array(x, float)
    # fill gaps by linear interp (rare for annual SST anomaly)
    if not valid.all():
        idx = np.arange(x.size)
        xf[~valid] = np.interp(idx[~valid], idx[valid], xf[valid])
    y = signal.filtfilt(b, a, xf, padlen=min(50, len(xf) // 3))
    y[~valid] = np.nan
    return y


def _normalise_lon(lon: np.ndarray) -> tuple[np.ndarray, str]:
    """Return (-180, 180]-style lon and the convention name of the input."""
    lon = np.asarray(lon, float)
    if (lon.min() >= 0) and (lon.max() > 180):
        return ((lon + 180) % 360) - 180, "0_360"
    return lon, "neg180_180"


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
    if lats[0] > lats[-1]:  # decreasing latitude
        da = da.sel({lat_name: slice(hi_lat, lo_lat)})
    else:
        da = da.sel({lat_name: slice(lo_lat, hi_lat)})

    # longitude: convert request to whatever convention the data uses
    if lons.min() >= 0:  # data in 0-360
        lw = lon_west % 360
        le = lon_east % 360
        if lw <= le:
            da = da.sel({lon_name: slice(lw, le)})
        else:
            # wraparound
            da = xr.concat([
                da.sel({lon_name: slice(lw, 360)}),
                da.sel({lon_name: slice(0, le)}),
            ], dim=lon_name)
    else:  # data in -180..180
        da = da.sel({lon_name: slice(lon_west, lon_east)})

    weights = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(weights).mean(dim=(lat_name, lon_name))


def _open_var(model: str, var: str) -> Optional[xr.Dataset]:
    """Open all NetCDF files for (model, var) under CMIP6_ROOT."""
    d = CMIP6_ROOT / model / "Amon" / var
    files = sorted(d.glob(f"{var}_Amon_*.nc"))
    if not files:
        return None
    try:
        if len(files) == 1:
            ds = xr.open_dataset(files[0], use_cftime=True)
        else:
            ds = xr.open_mfdataset([str(f) for f in files], combine="by_coords",
                                   use_cftime=True, parallel=False)
        return ds
    except Exception as e:
        print(f"  [{model}] open {var} failed: {e}")
        return None


def _load_obs_cb() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (years, cb_anomaly degC, global_anomaly degC)."""
    ds = xr.open_dataset(HADISST)
    years = ds["year"].values.astype(int)
    cb = ds["caesar_box_anomaly"].values.astype(float)
    gl = ds["global_anomaly"].values.astype(float)
    ds.close()
    return years, cb, gl


def _partial_regression(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> dict:
    """OLS: y = a + b*x + c*z + e.  Returns slope on x with HAC-ish CI.

    Uses standard OLS with Newey-West (lag=4) HAC standard error on b.
    """
    m = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
    if m.sum() < 15:
        return {"slope": np.nan, "se": np.nan, "n": int(m.sum()),
                "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "r2": np.nan}
    y, x, z = y[m], x[m], z[m]
    n = y.size
    X = np.column_stack([np.ones(n), x, z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    # Newey-West HAC SE on beta
    XtX_inv = np.linalg.inv(X.T @ X)
    L = 4  # 4 years lag (multi-decadal residuals shouldn't be very autocorrelated after bandpass)
    S = (resid[:, None] * X).T @ (resid[:, None] * X)
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1)
        u = (resid[:-l, None] * X[:-l]).T @ (resid[l:, None] * X[l:])
        S = S + w * (u + u.T)
    cov = XtX_inv @ S @ XtX_inv
    se_b = float(np.sqrt(max(cov[1, 1], 0.0)))
    slope = float(beta[1])
    # t-based 95% CI / p
    from scipy.stats import t as tdist
    dof = n - 3
    tcrit = tdist.ppf(0.975, dof)
    ci_lo = slope - tcrit * se_b
    ci_hi = slope + tcrit * se_b
    tval = slope / se_b if se_b > 0 else 0.0
    pval = 2.0 * (1.0 - tdist.cdf(abs(tval), dof))
    return {"slope": slope, "se": se_b, "n": int(n),
            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "p": float(pval), "r2": float(r2)}


# -----------------------------------------------------------------------------
# Per-model analysis
# -----------------------------------------------------------------------------
def analyze_model(model: str) -> Optional[dict]:
    """Returns regression dict or None if data missing."""
    ds_pr = _open_var(model, "pr")
    ds_ts = _open_var(model, "tas")
    if ds_pr is None or ds_ts is None:
        print(f"  [{model}] missing pr or tas; skipping")
        return None
    try:
        pr = ds_pr["pr"]
        ta = ds_ts["tas"]

        # Pre-filter to EPOCH range FIRST (this drops 50 years of data we don't need)
        # then load into memory once. Both pr and tas have monthly time.
        years_pr = np.array([t.year for t in pr["time"].values])
        keep_pr = (years_pr >= EPOCH[0]) & (years_pr <= EPOCH[1])
        pr = pr.isel(time=keep_pr)
        years_ta = np.array([t.year for t in ta["time"].values])
        keep_ta = (years_ta >= EPOCH[0]) & (years_ta <= EPOCH[1])
        ta = ta.isel(time=keep_ta)

        # build Sahel JJAS pr (mm/day): pr is kg m-2 s-1; multiply by 86400
        sahel_da = _sel_box(pr, SAHEL_LAT, SAHEL_LON_WEST, SAHEL_LON_EAST) * 86400.0
        sahel_vals = sahel_da.values  # (time,)
        months_pr = np.array([t.month for t in sahel_da["time"].values])
        years_m = np.array([t.year for t in sahel_da["time"].values])
        jjas_mask = (months_pr >= 6) & (months_pr <= 9)
        sahel_jjas = sahel_vals[jjas_mask]
        years_jjas = years_m[jjas_mask]
        uniq = np.unique(years_jjas)
        sahel_ann = np.array([sahel_jjas[years_jjas == y].mean() for y in uniq])

        # Cold-blob box tas annual mean
        cb_da = _sel_box(ta, CB_LAT, CB_LON_WEST, CB_LON_EAST)
        cb_vals = cb_da.values
        years_t = np.array([t.year for t in cb_da["time"].values])
        uniq_t = np.unique(years_t)
        cb_ann = np.array([cb_vals[years_t == y].mean() for y in uniq_t])

        # Global-mean tas annual (for partialling-out global warming)
        # Load tas eagerly as a numpy array first, then compute area-weighted
        # mean via direct numpy (xarray's .weighted is slow on small grids).
        lat_name = "lat" if "lat" in ta.dims else "latitude"
        lon_name = "lon" if "lon" in ta.dims else "longitude"
        ta_arr = ta.values  # (time, lat, lon)
        lat_vals = ta[lat_name].values
        w = np.cos(np.deg2rad(lat_vals))
        w = w / w.sum()
        # mean over lon, then weighted mean over lat
        zon_mean = np.nanmean(ta_arr, axis=2)  # (time, lat)
        glob_monthly = (zon_mean * w[None, :]).sum(axis=1)  # (time,)
        years_g = np.array([t.year for t in ta["time"].values])
        uniq_g = np.unique(years_g)
        glob_ann = np.array([glob_monthly[years_g == y].mean() for y in uniq_g])

        # align all three onto the common year axis
        all_years = np.intersect1d(np.intersect1d(uniq, uniq_t), uniq_g).astype(int)
        s = np.array([sahel_ann[uniq == y][0] for y in all_years])
        c = np.array([cb_ann[uniq_t == y][0] for y in all_years])
        g = np.array([glob_ann[uniq_g == y][0] for y in all_years])

        # Anomalies (subtract epoch mean)
        s_anom = s - np.nanmean(s)
        c_anom = c - np.nanmean(c)
        g_anom = g - np.nanmean(g)

        # Bandpass filter cold-blob (focus on multi-decadal CB variability).
        # Sign convention: "cold-blob INDEX" = -(SST anomaly), so positive index
        # means stronger cold-blob cooling (Caesar's slowdown convention).
        c_bp = _bandpass_yearly(-c_anom, BANDPASS_LOW_YR, BANDPASS_HIGH_YR)
        # Don't bandpass Sahel — let the regression pick out the band response.
        # Partial out global-mean tas anomaly (NOT bandpassed; we strip the
        # warming trend regardless of timescale).
        reg = _partial_regression(s_anom, c_bp, g_anom)
        reg["model"] = model
        reg["n_years"] = int(len(all_years))
        reg["epoch"] = list(EPOCH)
        reg["cb_amp"] = float(np.nanstd(c_bp))
        reg["sahel_amp"] = float(np.nanstd(s_anom))
        return reg
    except Exception as e:
        print(f"  [{model}] analysis failed: {e}")
        import traceback; traceback.print_exc()
        return None
    finally:
        ds_pr.close(); ds_ts.close()


# -----------------------------------------------------------------------------
# Observational anchor (HadISST CB driving CRU/GPCC ?? — we have no CMAP/CRU
# pulled, so the observational anchor is *only* the SST-driving series. We
# report the obs cold-blob index used as the driver, and we *cannot* compute
# an obs slope without an obs precip product. We anchor by reporting the
# obs cold-blob amplitude and bandpass spectral structure used as the driver
# for the model-internal regression, and by noting the published expected
# slope range (-0.1 .. -0.4 mm/day per K CB) from Zhang & Delworth 2006.)
# -----------------------------------------------------------------------------
def obs_anchor() -> dict:
    years, cb, gl = _load_obs_cb()
    # Cold-blob INDEX = -(SST anomaly) so positive index = stronger cooling.
    cb_index = -cb
    cb_bp = _bandpass_yearly(cb_index, BANDPASS_LOW_YR, BANDPASS_HIGH_YR)
    return {
        "years": years.tolist(),
        "cb_index_negsst": cb_index.tolist(),
        "global_anomaly": gl.tolist(),
        "cb_bandpass": cb_bp.tolist(),
        "amp_bandpass_K": float(np.nanstd(cb_bp)),
        "expected_sahel_slope_mm_day_per_K_CB": [-0.4, -0.1],
        "sign_convention": "positive CB index = stronger cold-blob cooling (Caesar/AMOC-slowdown convention)",
        "source": "HadISST cold_blob_timeseries_hadisst.nc",
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    priority = [
        "UKESM1-0-LL", "CESM2", "CanESM5", "GFDL-ESM4",
        "MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "MIROC6", "BCC-CSM2-MR",
        "CNRM-CM6-1", "GISS-E2-1-G", "IPSL-CM6A-LR", "ACCESS-CM2",
    ]
    per_model = {}
    for m in priority:
        print(f"\n=== {m} ===")
        r = analyze_model(m)
        if r is not None:
            per_model[m] = r
            print(f"  slope = {r['slope']:+.3f} mm/d per K CB  "
                  f"(SE={r['se']:.3f}, p={r['p']:.3f}, n={r['n']})")

    obs = obs_anchor()

    # ensemble stats
    slopes = np.array([v["slope"] for v in per_model.values()])
    n_models = len(slopes)
    if n_models > 0:
        med = float(np.nanmedian(slopes))
        p25 = float(np.nanpercentile(slopes, 25))
        p75 = float(np.nanpercentile(slopes, 75))
    else:
        med = p25 = p75 = float("nan")

    # effect strength
    if n_models >= 4:
        # robust if median is clearly negative AND p25..p75 doesn't cross zero
        if med < 0 and p75 < 0:
            strength = "moderate-strong: ensemble median negative, IQR excludes zero"
        elif med < 0:
            strength = "weak-moderate: median negative but IQR straddles zero"
        elif med > 0:
            strength = "unexpected positive median"
        else:
            strength = "null"
    elif n_models > 0:
        strength = f"too few models ({n_models}) for ensemble verdict; obs-anchor only"
    else:
        strength = "no model data available; obs-anchor only"

    out = {
        "pathway": "sahel",
        "description": "Sahel JJAS pr regressed on bandpassed cold-blob SST index, partialling out global-mean tas.",
        "epoch": list(EPOCH),
        "bandpass_period_yr": [BANDPASS_LOW_YR, BANDPASS_HIGH_YR],
        "cold_blob_box": {"lat": list(CB_LAT), "lon_west": CB_LON_WEST, "lon_east": CB_LON_EAST},
        "sahel_box": {"lat": list(SAHEL_LAT), "lon_west": SAHEL_LON_WEST, "lon_east": SAHEL_LON_EAST},
        "per_model": per_model,
        "ensemble": {
            "n_models": n_models,
            "median_slope_mm_day_per_K": med,
            "p25_slope": p25,
            "p75_slope": p75,
        },
        "effect_strength": strength,
        "obs_anchor": {k: v for k, v in obs.items() if k != "years" and k != "cb_anomaly" and k != "global_anomaly" and k != "cb_bandpass"},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x)))

    # also save full arrays as npz for plotting
    np.savez_compressed(
        OUT_NPZ,
        obs_years=np.array(obs["years"]),
        obs_cb_index=np.array(obs["cb_index_negsst"]),
        obs_cb_bp=np.array(obs["cb_bandpass"]),
        obs_global_anom=np.array(obs["global_anomaly"]),
        model_names=np.array(list(per_model.keys())),
        model_slopes=slopes if n_models else np.array([]),
        model_se=np.array([v["se"] for v in per_model.values()]),
        model_p=np.array([v["p"] for v in per_model.values()]),
    )

    print("\n=== SUMMARY ===")
    print(f"  models analysed: {n_models}")
    print(f"  median slope: {med:+.3f} mm/day per K CB")
    print(f"  IQR: [{p25:+.3f}, {p75:+.3f}]")
    print(f"  effect strength: {strength}")
    print(f"  wrote {OUT_JSON}")
    print(f"  wrote {OUT_NPZ}")
    return out


# -----------------------------------------------------------------------------
# Subpanel for the composite figure
# -----------------------------------------------------------------------------
def draw_subpanel(ax):
    """Draw the SAHEL pathway subpanel onto an existing matplotlib Axes.

    Layout:
      - Horizontal box+strip plot of per-model regression slopes (mm/day per K CB),
        with each model shown as a distinct shape; vertical line at zero;
        shaded band for the published expected range (-0.4 .. -0.1).
      - Title-style anchor text on the right with median +/- IQR.
    """
    import matplotlib.pyplot as plt  # noqa: F401
    if not OUT_JSON.exists():
        ax.text(0.5, 0.5, "sahel.json missing — run compute_fig7_sahel.py",
                transform=ax.transAxes, ha="center", va="center", fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
        return

    d = json.loads(OUT_JSON.read_text())
    per = d["per_model"]
    ens = d["ensemble"]
    obs = d["obs_anchor"]

    # Uniform marker per model -- per-family encoding cannot be supported at n=12
    # and the corresponding author cannot decode 12 different shapes at 5.5 pt.
    SHORT_NAME = {
        "UKESM1-0-LL": "UKESM1-LL",
        "CESM2": "CESM2",
        "CanESM5": "CanESM5",
        "GFDL-ESM4": "GFDL",
        "MPI-ESM1-2-HR": "MPI-HR",
        "MPI-ESM1-2-LR": "MPI-LR",
        "MIROC6": "MIROC",
        "BCC-CSM2-MR": "BCC",
        "CNRM-CM6-1": "CNRM",
        "GISS-E2-1-G": "GISS",
        "IPSL-CM6A-LR": "IPSL",
        "ACCESS-CM2": "ACCESS",
    }

    # Expected range band (Zhang & Delworth 2006; Folland 1986).
    exp_lo, exp_hi = obs.get("expected_sahel_slope_mm_day_per_K_CB", [-0.4, -0.1])
    ax.axvspan(exp_lo, exp_hi, facecolor="#F0D8C8", alpha=0.55,
               edgecolor="#C9A37A", linewidth=0.3, zorder=0,
               label="Zhang \\& Delworth 2006 expected Sahel range")

    # Zero line -- match figure-wide reference-line convention.
    ax.axvline(0, color="0.55", linewidth=0.4, linestyle=":", alpha=0.9)

    # Per-model points with error bars
    models = list(per.keys())
    n = len(models)
    if n == 0:
        ax.text(0.5, 0.5, "no CMIP6 models available",
                transform=ax.transAxes, ha="center", va="center", fontsize=6)
    else:
        # Reserve y=-1.0 for the ensemble median bar
        ENS_ROW = -1.0
        ys = np.arange(n)
        for i, m in enumerate(models):
            r = per[m]
            ax.errorbar(r["slope"], i, xerr=1.96 * r["se"],
                        marker="o", markersize=4.5,
                        markerfacecolor="#4C72B0", markeredgecolor="black",
                        ecolor="0.4", elinewidth=0.6, capsize=2,
                        linestyle="none", zorder=3)
        ytlabels = [SHORT_NAME.get(m, m) for m in models] + ["ENS"]
        ax.set_yticks(list(ys) + [ENS_ROW])
        ax.set_yticklabels(ytlabels, fontsize=6)
        ax.set_ylim(ENS_ROW - 0.6, n - 0.2)
        ax.invert_yaxis()

        # Ensemble median + IQR on the dedicated ENS row.
        med = ens["median_slope_mm_day_per_K"]
        p25 = ens["p25_slope"]; p75 = ens["p75_slope"]
        ax.hlines(ENS_ROW, p25, p75, colors="#2F3E46", linewidth=2.8,
                  alpha=0.85, zorder=2,
                  label=(f"CMIP6 ensemble median $\\pm$ IQR (Sahel, "
                         f"$n={ens['n_models']}$)"))
        ax.plot(med, ENS_ROW, marker="|", color="#2F3E46", markersize=12,
                markeredgewidth=1.8, zorder=4)
        # Highlight ENS row with a light separator between ENS and first model
        ax.axhline((ys[0] + ENS_ROW) / 2, color="0.7", linewidth=0.4, linestyle=":")

    ax.set_xlabel(r"Sahel JJAS pr regression on cold-blob index (mm day$^{-1}$ per K)",
                  labelpad=1)
    ax.tick_params(axis="x", pad=1, length=2)
    ax.tick_params(axis="y", pad=1, length=0, labelsize=5)
    # Return handles for the figure-foot legend; NO ax.legend here.
    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


if __name__ == "__main__":
    main()
