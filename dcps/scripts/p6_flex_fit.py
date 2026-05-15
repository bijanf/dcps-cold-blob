"""Flexible-form refit of the <r_loc>-EKE relationship for the primary 30-yr SST.

Four parametric families fit cell-wise on the Atlantic 2-deg grid:
  1. registered: r = (1 + tau*EKE)^-0.5         (1 free: tau)
  2. power:      r = (alpha + tau*EKE)^-beta    (3 free: alpha, tau, beta)
  3. exp:        r = r0 * exp(-tau*EKE)         (2 free: r0, tau)
  4. saturating: r = r_inf + (r0-r_inf)*exp(-tau*EKE)  (3 free: r_inf, r0, tau)

For each: chi^2_nu, AIC, BIC.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

# Sibling-script imports: add the directory this file lives in to sys.path so
# `from multi_basin_quiescence import ...` resolves regardless of where the
# script is run from. This avoids hard-coded absolute paths.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_basin_quiescence import (  # noqa: E402
    BASINS,
    instantaneous_phase,
    local_r_mean,
    preprocess_anomaly,
)


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


def main(argv=None) -> int:
    p_arg = argparse.ArgumentParser(description=__doc__)
    p_arg.add_argument("--sst", type=Path, required=True,
                       help="Path to monthly-mean SST NetCDF (lat/lon).")
    p_arg.add_argument("--eke", type=Path, required=True,
                       help="Path to EKE climatology on the basin grid.")
    p_arg.add_argument("--out-json", type=Path, required=True)
    p_arg.add_argument("--out-npz",  type=Path, required=True)
    p_arg.add_argument("--basin", default="atlantic")
    args = p_arg.parse_args(argv)

    print(f"loading {args.sst}")
    sst_ds = xr.open_dataset(args.sst)
    sst = sst_ds.get("sst", next(iter(sst_ds.data_vars.values())))

    # Already lat/lon.  Use the same healpix-or-latlon regridder as the P6
    # pipeline so the cell set is identical to the primary run.
    from quiescence_dlesym import healpix_to_basin_grid  # noqa: E402
    sst_basin = healpix_to_basin_grid(args.sst, basin=args.basin)
    bp = preprocess_anomaly(sst_basin)
    phase = instantaneous_phase(bp)
    rloc = local_r_mean(phase, radius_km=500.0)

    eke = xr.open_dataarray(args.eke)
    rv = rloc.values
    ev = eke.values
    if rv.shape != ev.shape:
        raise SystemExit(f"shape mismatch: rloc {rv.shape} vs eke {ev.shape}")
    mask = np.isfinite(rv) & np.isfinite(ev)
    x = rv[mask].astype(np.float64)
    y = ev[mask].astype(np.float64)
    print(f"n_cells={x.size}")

    fits = fit_all(x, y)
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
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out_json}")
    print("\nfits ranked by AIC:")
    for f in fits_valid:
        print(f"  {f['family']:>11s} k={f['k']} chi2_nu={f['chi2_nu']:.3f} "
              f"AIC={f['AIC']:.1f} dAIC={f['dAIC']:.1f} BIC={f['BIC']:.1f}")

    np.savez(args.out_npz, rloc=x, eke=y)
    print(f"wrote {args.out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
