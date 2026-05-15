"""Flexible-form refit of the <r_loc>-EKE relationship for the primary 30-yr SST.

Four parametric families fit cell-wise on the Atlantic 2-deg grid:
  1. registered: r = (1 + tau*EKE)^-0.5         (1 free: tau)
  2. power:      r = (alpha + tau*EKE)^-beta    (3 free: alpha, tau, beta)
  3. exp:        r = r0 * exp(-tau*EKE)         (2 free: r0, tau)
  4. saturating: r = r_inf + (r0-r_inf)*exp(-tau*EKE)  (3 free: r_inf, r0, tau)

For each: chi^2_nu, AIC, BIC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

sys.path.insert(0, "/home/fallah/NEW_Theory/dcps/scripts")
from multi_basin_quiescence import (  # noqa: E402
    BASINS,
    instantaneous_phase,
    local_r_mean,
    preprocess_anomaly,
)

SST_PATH = Path("/p/projects/poem/fallah/dlesym_p6_out/p6_30yr_sst.nc")
EKE_PATH = Path("/p/projects/poem/fallah/cache/glorys12_eke_clim_atlantic_2deg.nc")
OUT_JSON = Path("/p/projects/poem/fallah/p6_bundle/data/flex_fit.json")
OUT_NPZ = Path("/p/projects/poem/fallah/p6_bundle/data/flex_fit_cells.npz")


def _registered(eke, tau):
    return (1.0 + tau * eke) ** -0.5


def _power(eke, alpha, tau, beta):
    return (alpha + tau * eke) ** -beta


def _exp(eke, r0, tau):
    return r0 * np.exp(-tau * eke)


def _sat(eke, r_inf, r0, tau):
    return r_inf + (r0 - r_inf) * np.exp(-tau * eke)


FAMILIES = [
    ("registered", _registered, [144.0]),
    ("power",      _power,      [1.0, 144.0, 0.5]),
    ("exp",        _exp,        [1.0, 1.0]),
    ("saturating", _sat,        [0.0, 1.0, 1.0]),
]


def fit_all(x_rloc, y_eke):
    fits = []
    for name, fn, p0 in FAMILIES:
        try:
            popt, _ = curve_fit(fn, y_eke, x_rloc, p0=p0, maxfev=20000)
            pred = fn(y_eke, *popt)
            resid = x_rloc - pred
            k = len(popt)
            n = x_rloc.size
            ss_res = float(np.sum(resid ** 2))
            chi2_nu = ss_res / (max(n - k, 1) * float(np.var(x_rloc, ddof=1)))
            sigma2 = ss_res / n
            ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)
            aic = 2.0 * k - 2.0 * ll
            bic = float(k * np.log(n) - 2.0 * ll)
            fits.append({
                "family": name, "k": k, "n": n,
                "params": [float(v) for v in popt],
                "chi2_nu": float(chi2_nu),
                "AIC": float(aic),
                "BIC": bic,
                "ss_resid": ss_res,
            })
        except Exception as e:
            fits.append({"family": name, "error": str(e)})
    return fits


def main() -> int:
    print(f"loading {SST_PATH}")
    sst_ds = xr.open_dataset(SST_PATH)
    sst = sst_ds.get("sst", next(iter(sst_ds.data_vars.values())))

    # Already lat/lon. Build the 2-deg basin <r_loc> the same way the script does.
    # Reuse healpix_to_basin_grid path through the lat/lon branch.
    sys.path.insert(0, "/home/fallah/NEW_Theory/dcps/scripts")
    from quiescence_dlesym import healpix_to_basin_grid
    sst_basin = healpix_to_basin_grid(SST_PATH, basin="atlantic")
    bp = preprocess_anomaly(sst_basin)
    phase = instantaneous_phase(bp)
    rloc = local_r_mean(phase, radius_km=500.0)

    eke = xr.open_dataarray(EKE_PATH)
    rv = rloc.values
    ev = eke.values
    if rv.shape != ev.shape:
        raise SystemExit(f"shape mismatch: rloc {rv.shape} vs eke {ev.shape}")
    mask = np.isfinite(rv) & np.isfinite(ev)
    x = rv[mask].astype(np.float64)
    y = ev[mask].astype(np.float64)
    print(f"n_cells={x.size}")

    fits = fit_all(x, y)
    # Rank by AIC, lowest first
    fits_valid = [f for f in fits if "AIC" in f]
    fits_valid.sort(key=lambda f: f["AIC"])
    delta_aic = fits_valid[0]["AIC"] if fits_valid else 0.0
    for f in fits_valid:
        f["dAIC"] = f["AIC"] - delta_aic

    out = {
        "n_cells": int(x.size),
        "fits": fits,
        "best_family": fits_valid[0]["family"] if fits_valid else None,
        "rloc_stats": {
            "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()),
        },
        "eke_stats": {
            "mean": float(y.mean()), "min": float(y.min()), "max": float(y.max()),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON}")
    print("\nfits ranked by AIC:")
    for f in fits_valid:
        print(f"  {f['family']:>11s} k={f['k']} chi2_nu={f['chi2_nu']:.3f} "
              f"AIC={f['AIC']:.1f} dAIC={f['dAIC']:.1f} BIC={f['BIC']:.1f}")

    # Save the (x, y) data for downstream story-figure scatter
    np.savez(OUT_NPZ, rloc=x, eke=y)
    print(f"wrote {OUT_NPZ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
