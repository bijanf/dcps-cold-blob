"""H3 test: transfer-entropy collapse vs critical-slowing-down rise.

We build two regional time series (subtropical, subpolar) from the bandpassed
SST anomaly and compute on a sliding 10-year window:
  * transfer entropy T_{ST -> SP}(t)  -- binary-discretised, history k=l=1
  * subpolar variance sigma^2(t)
  * subpolar lag-1 autocorrelation alpha_1(t)

Threshold-crossing times (from Methods):
  * t_TE: first year TE drops below baseline_mean - 2 sigma and stays
    there for 5 yr.
  * t_var, t_alpha: first year each rises above baseline_mean + 2 sigma.

Pre-registered decision rule:
  H3 supported iff t_TE < min(t_var, t_alpha) - 3 yr.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


# ---- Binary-discretised transfer entropy ----------------------------------

def _binary_te(x: np.ndarray, y: np.ndarray) -> float:
    """T_{Y -> X} with binary symbols and history length 1.

    Discretises each series at its window median, then estimates joint
    probabilities by counting. Returns transfer entropy in bits; 0 if
    degenerate (all-same-symbol windows).
    """
    if x.size != y.size or x.size < 4:
        return float("nan")
    xb = (x > np.median(x)).astype(np.int8)
    yb = (y > np.median(y)).astype(np.int8)
    # Triples (x_{t+1}, x_t, y_t)
    x_next = xb[1:]
    x_prev = xb[:-1]
    y_prev = yb[:-1]

    # Joint and marginal counts -> probabilities
    n = x_next.size
    p_xyy = np.zeros((2, 2, 2), dtype=np.float64)
    for a, b, c in zip(x_next, x_prev, y_prev):
        p_xyy[a, b, c] += 1
    p_xyy /= n
    p_xy = p_xyy.sum(axis=2)              # p(x_next, x_prev)
    p_xy_y = p_xyy.sum(axis=0)            # p(x_prev, y_prev)
    p_x = p_xy.sum(axis=0)                # p(x_prev) (broadcast index)

    te = 0.0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                if p_xyy[a, b, c] <= 0:
                    continue
                num = p_xyy[a, b, c] / p_xy_y[b, c] if p_xy_y[b, c] > 0 else 0.0
                den = p_xy[a, b] / p_x[b] if p_x[b] > 0 else 0.0
                if num > 0 and den > 0:
                    te += p_xyy[a, b, c] * np.log2(num / den)
    return float(te)


def _lag1_autocorr(x: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    var = (x * x).mean()
    if var <= 0:
        return float("nan")
    return float((x[:-1] * x[1:]).mean() / var)


# ---- Regional aggregator --------------------------------------------------

SUBTROPICAL_BOX = dict(lat=slice(20, 40), lon=slice(-60, -10))
SUBPOLAR_BOX = dict(lat=slice(45, 60), lon=slice(-40, -15))


def regional_mean_series(da: xr.DataArray, lat_range: slice, lon_range: slice) -> xr.DataArray:
    """Area-weighted (cos lat) mean over a box -> 1-D time series."""
    box = da.sel(lat=lat_range, lon=lon_range)
    w = np.cos(np.deg2rad(box.lat)).broadcast_like(box)
    w = w.where(box.notnull())
    out = (box * w).sum(["lat", "lon"], skipna=True) / w.sum(["lat", "lon"], skipna=True)
    return out


# ---- Sliding-window EWS ---------------------------------------------------

def sliding_window_ews(
    sst_anom: xr.DataArray,
    window_years: int = 10,
    step_months: int = 1,
) -> xr.Dataset:
    """Compute T_{ST->SP}(t), sigma_SP^2(t), alpha1_SP(t) on sliding windows.

    Returns one value per window centre, indexed by the centre time.
    """
    st = regional_mean_series(sst_anom, SUBTROPICAL_BOX["lat"], SUBTROPICAL_BOX["lon"])
    sp = regional_mean_series(sst_anom, SUBPOLAR_BOX["lat"], SUBPOLAR_BOX["lon"])

    n = st.size
    win = window_years * 12
    if n < win + 1:
        raise ValueError(f"Series of {n} months too short for {win}-month window.")

    centres = []
    te_vals, var_vals, ac_vals = [], [], []
    for start in range(0, n - win + 1, step_months):
        end = start + win
        x_st = st.values[start:end]
        x_sp = sp.values[start:end]
        if not (np.isfinite(x_st).all() and np.isfinite(x_sp).all()):
            te_vals.append(np.nan); var_vals.append(np.nan); ac_vals.append(np.nan)
        else:
            te_vals.append(_binary_te(x_sp, x_st))   # (X=SP target, Y=ST source)
            var_vals.append(float(np.var(x_sp)))
            ac_vals.append(_lag1_autocorr(x_sp))
        centres.append(st.time.values[start + win // 2])

    coords = {"time": np.array(centres, dtype="datetime64[ns]")}
    return xr.Dataset(
        {
            "TE_st_to_sp": ("time", np.array(te_vals, dtype=np.float32)),
            "var_sp": ("time", np.array(var_vals, dtype=np.float32)),
            "ac1_sp": ("time", np.array(ac_vals, dtype=np.float32)),
        },
        coords=coords,
        attrs={
            "window_years": int(window_years),
            "step_months": int(step_months),
            "te_method": "binary-discretised, history k=l=1, source=ST, target=SP",
        },
    )


# ---- Threshold crossings + H3 decision ------------------------------------

@dataclass
class H3Result:
    t_te_cross: str | None
    t_var_cross: str | None
    t_alpha_cross: str | None
    lead_years_te_vs_var: float
    lead_years_te_vs_alpha: float
    pass_h3: bool
    note: str = ""


def _baseline_stats(series: xr.DataArray, baseline_start: str, baseline_end: str) -> tuple[float, float]:
    s = series.sel(time=slice(baseline_start, baseline_end)).values
    s = s[np.isfinite(s)]
    if s.size < 12:
        return (float("nan"), float("nan"))
    return float(s.mean()), float(s.std())


def _first_persistent_crossing(
    series: xr.DataArray,
    threshold: float,
    direction: str,
    persist_years: float = 5.0,
    fs_per_year: float = 12.0,
) -> str | None:
    """First time the series crosses the threshold and stays there for
    ``persist_years``. ``direction`` is 'below' or 'above'."""
    vals = series.values
    times = series.time.values
    n_persist = int(round(persist_years * fs_per_year))
    if direction == "below":
        cross = vals < threshold
    elif direction == "above":
        cross = vals > threshold
    else:
        raise ValueError(direction)
    for i in range(vals.size - n_persist):
        if cross[i] and cross[i:i + n_persist].all():
            return str(times[i])[:10]
    return None


def test_h3(
    ews: xr.Dataset,
    baseline: tuple[str, str] = ("1980-01-01", "2010-12-31"),
    persist_years: float = 5.0,
    minimum_lead_years: float = 3.0,
) -> H3Result:
    te = ews.TE_st_to_sp
    var = ews.var_sp
    ac = ews.ac1_sp

    te_mu, te_sd = _baseline_stats(te, *baseline)
    var_mu, var_sd = _baseline_stats(var, *baseline)
    ac_mu, ac_sd = _baseline_stats(ac, *baseline)

    t_te = _first_persistent_crossing(te, te_mu - 2 * te_sd, "below", persist_years)
    t_var = _first_persistent_crossing(var, var_mu + 2 * var_sd, "above", persist_years)
    t_alpha = _first_persistent_crossing(ac, ac_mu + 2 * ac_sd, "above", persist_years)

    def _yrdiff(a: str | None, b: str | None) -> float:
        if a is None or b is None:
            return float("nan")
        ay = float(a[:4]) + (float(a[5:7]) - 1) / 12
        by = float(b[:4]) + (float(b[5:7]) - 1) / 12
        return ay - by

    lead_te_var = _yrdiff(t_var, t_te)
    lead_te_alpha = _yrdiff(t_alpha, t_te)
    lead = min(lead_te_var, lead_te_alpha) if not (np.isnan(lead_te_var) and np.isnan(lead_te_alpha)) else float("nan")

    pass_h3 = (
        t_te is not None
        and (t_var is not None or t_alpha is not None)
        and not np.isnan(lead)
        and lead >= minimum_lead_years
    )

    note = ""
    if t_te is None:
        note = "TE did not cross its baseline threshold persistently."
    elif t_var is None and t_alpha is None:
        note = "Neither variance nor AC1 crossed their thresholds."

    return H3Result(
        t_te_cross=t_te,
        t_var_cross=t_var,
        t_alpha_cross=t_alpha,
        lead_years_te_vs_var=lead_te_var,
        lead_years_te_vs_alpha=lead_te_alpha,
        pass_h3=pass_h3,
        note=note,
    )
