"""H3 test on CMIP6 piControl long records.

Pre-registered in chat transcript:
  For each model, on a 30-yr sliding window stepped by 1 yr:
    TE(subtropical-AMOC -> subpolar-AMOC)  binary-discretised, k=l=1
    var(subpolar-AMOC), AC1(subpolar-AMOC)
  Then lagged Pearson rho(TE(t), var(t+tau)) for tau in {-10,-5,0,+5,+10,+15} yr.
  Per-model SUPPORTED iff argmin_tau rho is at tau > 0 with rho < -0.2.
  Aggregate SUPPORTED iff >=60% of models pass.

AMOC at latitude L is defined as max over depth of the streamfunction
Psi(t, z, L) = integral_0^z v_zonal(z', L, t) dz'.

Subtropical: 26.5N (RAPID latitude). Subpolar: 50N (subpolar gyre).
"""

from __future__ import annotations

import glob
import json
import os
import time as _time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.h3 import _binary_te, _lag1_autocorr


CMIP6_DIR = Path("/home/bijanf/Documents/AMOC_renalysis/data/cmip6_fullfield")
H3_PIC_DIR = CACHE_DIR / "h3_picontrol"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

# Pre-registered parameters
SUBTROPICAL_LAT = 26.5      # RAPID latitude
SUBPOLAR_LAT = 50.0         # subpolar gyre
WINDOW_YEARS = 30
LAGS_YEARS = [-10, -5, 0, 5, 10, 15]
PER_MODEL_RHO_BAR = -0.2
AGGREGATE_FRACTION = 0.60


def _coder():
    return xr.coders.CFDatetimeCoder(use_cftime=True)


def amoc_at_latitude(vo_zonal_file: str, target_lat: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (years, AMOC time series in Sv) at the latitude closest to target.

    AMOC = max over depth of cumulative-trapezoid integral of v_zonal over depth.
    We aggregate to annual means before returning.
    """
    ds = xr.open_dataset(vo_zonal_file, decode_times=_coder())
    depth = ds.depth.values
    lat = ds.lat.values
    v = ds.v_zonal.values            # (time, depth, lat) m^2/s zonally integrated

    # cumulative trapezoid integration along depth axis
    psi = cumulative_trapezoid(v, depth, axis=1, initial=0)   # (time, depth, lat)
    # AMOC strength: max over depth, taking absolute value
    psi_max = np.nanmax(psi, axis=1)                            # (time, lat)

    j = int(np.argmin(np.abs(lat - target_lat)))
    psi_lat = psi_max[:, j]                                     # monthly

    # Convert to Sv (1 Sv = 1e6 m^3/s)
    psi_lat_Sv = psi_lat / 1.0e6

    # Annual means
    times = ds.time.values
    years = np.array([t.year for t in times])
    uy = np.unique(years)
    annual = np.array([psi_lat_Sv[years == y].mean() for y in uy])
    ds.close()
    return uy.astype(float), annual


def sliding_diagnostics(amoc_st: np.ndarray, amoc_sp: np.ndarray,
                         years: np.ndarray, window_years: int = WINDOW_YEARS):
    """Compute sliding-window TE, var, AC1.  Inputs are annual time series."""
    n = len(years)
    if n < window_years + 5:
        return None
    centres, te_vals, var_vals, ac_vals = [], [], [], []
    for start in range(0, n - window_years + 1):
        end = start + window_years
        x_st = amoc_st[start:end]
        x_sp = amoc_sp[start:end]
        if not (np.isfinite(x_st).all() and np.isfinite(x_sp).all()):
            te_vals.append(np.nan); var_vals.append(np.nan); ac_vals.append(np.nan)
        else:
            te_vals.append(_binary_te(x_sp, x_st))    # X=target=subpolar, Y=source=subtropical
            var_vals.append(float(np.var(x_sp)))
            ac_vals.append(_lag1_autocorr(x_sp))
        centres.append(years[start + window_years // 2])
    return (np.array(centres),
            np.array(te_vals), np.array(var_vals), np.array(ac_vals))


def lag_correlation(te: np.ndarray, var: np.ndarray, lag_years: int) -> float:
    """Pearson rho(te(t), var(t+lag_years)).  lag>0 means TE leads var."""
    finite = np.isfinite(te) & np.isfinite(var)
    if finite.sum() < 30:
        return float("nan")
    if lag_years == 0:
        a, b = te[finite], var[finite]
    elif lag_years > 0:
        a = te[:-lag_years]
        b = var[lag_years:]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
    else:
        L = -lag_years
        a = te[L:]
        b = var[:-L]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
    if a.size < 30 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_model(vo_zonal_file: str) -> dict:
    name = os.path.basename(vo_zonal_file).split("_piControl")[0]
    print(f"  {name}", end=" ", flush=True)
    t0 = _time.time()
    yr1, st = amoc_at_latitude(vo_zonal_file, SUBTROPICAL_LAT)
    yr2, sp = amoc_at_latitude(vo_zonal_file, SUBPOLAR_LAT)
    n_yr = min(yr1.size, yr2.size)
    if n_yr < WINDOW_YEARS + 5:
        print(f"  too short ({n_yr} yr) - skip")
        return {"model": name, "n_years": int(n_yr), "skipped": "too_short"}
    yr = yr1[:n_yr]; st = st[:n_yr]; sp = sp[:n_yr]
    diag = sliding_diagnostics(st, sp, yr)
    if diag is None:
        return {"model": name, "n_years": int(n_yr), "skipped": "diag_failed"}
    centres, te, var, ac = diag

    # Lagged Pearson rho(TE, var) at each tau
    rhos = {tau: lag_correlation(te, var, tau) for tau in LAGS_YEARS}
    finite_rhos = {k: v for k, v in rhos.items() if np.isfinite(v)}
    if not finite_rhos:
        return {"model": name, "n_years": int(n_yr), "rhos": rhos,
                "argmin_tau": None, "passed": False}
    argmin_tau = min(finite_rhos, key=lambda k: finite_rhos[k])
    rho_at_argmin = finite_rhos[argmin_tau]
    passed = (argmin_tau > 0) and (rho_at_argmin <= PER_MODEL_RHO_BAR)
    print(f"  {n_yr} yr  argmin_tau = {argmin_tau:+d}, rho = {rho_at_argmin:+.2f}  "
          f"{'pass' if passed else 'fail'}  ({_time.time()-t0:.1f}s)")
    return {
        "model": name, "n_years": int(n_yr), "rhos": rhos,
        "argmin_tau": argmin_tau, "rho_at_argmin": rho_at_argmin,
        "passed": passed,
        "amoc_st_mean": float(np.mean(st)),
        "amoc_sp_mean": float(np.mean(sp)),
    }


def main():
    H3_PIC_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(CMIP6_DIR / "*_piControl_vo_zonal.nc")))
    print(f"Found {len(files)} CMIP6 piControl vo_zonal files\n")
    print("H3-piControl test (pre-registered):")
    print(f"  subtropical lat: {SUBTROPICAL_LAT}")
    print(f"  subpolar lat:    {SUBPOLAR_LAT}")
    print(f"  window:          {WINDOW_YEARS} yr")
    print(f"  lag set:         {LAGS_YEARS} yr")
    print(f"  per-model bar:   argmin_tau > 0 AND rho <= {PER_MODEL_RHO_BAR}")
    print(f"  aggregate bar:   >= {AGGREGATE_FRACTION:.0%} of models pass")
    print()

    results: list[dict] = []
    for f in files:
        try:
            r = run_model(f)
            results.append(r)
        except Exception as e:
            print(f"  {os.path.basename(f).split('_piControl')[0]}: ERROR {e}")
            results.append({"model": os.path.basename(f).split("_piControl")[0],
                            "error": str(e)})

    valid = [r for r in results if "passed" in r]
    n_pass = sum(1 for r in valid if r["passed"])
    pass_frac = n_pass / max(1, len(valid))
    print("\n=== Aggregate H3-piControl ===")
    print(f"  Models tested: {len(valid)} of {len(results)}")
    print(f"  Per-model passes: {n_pass}/{len(valid)}  ({pass_frac:.0%})")
    aggregate_supported = pass_frac >= AGGREGATE_FRACTION
    print(f"  AGGREGATE: {'SUPPORTED' if aggregate_supported else 'falsified'}")

    # Median lagged rho across models
    print("\nMedian rho(TE, var) across models, by lag:")
    for tau in LAGS_YEARS:
        rhos = [r["rhos"][tau] for r in valid if "rhos" in r and np.isfinite(r["rhos"].get(tau, np.nan))]
        if rhos:
            print(f"  tau = {tau:+3d} yr: median rho = {np.median(rhos):+.3f}  "
                  f"(n={len(rhos)} models)")

    out_json = H3_PIC_DIR / "h3_picontrol_results.json"
    with open(out_json, "w") as f:
        json.dump({"results": results,
                   "aggregate": {
                       "n_tested": len(valid), "n_pass": n_pass,
                       "pass_frac": float(pass_frac),
                       "supported": bool(aggregate_supported),
                   }}, f, indent=2, default=str)
    print(f"\nWrote {out_json}")

    # ----- figure --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    cmap = plt.get_cmap("tab20")
    for i, r in enumerate(valid):
        rhos = [r["rhos"].get(tau, np.nan) for tau in LAGS_YEARS]
        ax.plot(LAGS_YEARS, rhos, "-", color=cmap(i % 20), alpha=0.5, lw=0.8,
                 label=r["model"])
    median_rho = [np.median([r["rhos"][tau] for r in valid
                              if "rhos" in r and np.isfinite(r["rhos"].get(tau, np.nan))])
                  for tau in LAGS_YEARS]
    ax.plot(LAGS_YEARS, median_rho, "k-", lw=2.5, label="median", marker="s")
    ax.axhline(0, color="grey", lw=0.4)
    ax.axvline(0, color="grey", lw=0.4)
    ax.set_xlabel(r"lag $\tau$  (yr; $\tau>0$: TE leads variance)")
    ax.set_ylabel(r"$\rho(T_{ST\to SP}(t),\ \sigma^2_{SP}(t+\tau))$")
    ax.legend(loc="best", fontsize=7, ncol=2, frameon=False)
    out_fig = MANUSCRIPT_FIGS / "fig6_h3_picontrol.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
