"""Compute (Q, basin-mean EKE) for the ORAS5 reanalysis on the
Atlantic basin, using the same window/regrid pipeline as the CMIP6
holocene_q_pilot computations.

ORAS5 covers 2000-2023 (24 yr, monthly).  This is shorter than the
30-yr window standard but close enough to anchor the CMIP6 joint
attractor analysis to one observation-constrained product.

For comparability with the joint (rel-EKE, Q) figure (which plots
each model's EKE normalised by its own piControl mean), we report
two normalisations:
  - absolute basin-mean |grad SSH|^2
  - rel-EKE = absolute / mean(pi_eke_mean) across all admitted CMIP6
    models.  This anchors ORAS5 to the cross-CMIP6 mean pre-industrial
    EKE level.

Output: dcps/cache/eke_timeseries/ORAS5_atlantic_obs.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR, PKG_ROOT
sys.path.insert(0, str(PKG_ROOT / "scripts"))
from holocene_q_pilot import (  # noqa: E402
    _basin_subset_2deg, _Q_for_window,
)

EKE_TS_DIR = CACHE_DIR / "eke_timeseries"

# Each entry: (source_label, path_to_phase1_nc).  All files share the
# same 2-deg North Atlantic schema (lat=38, lon=40, sst_raw, ssh_raw).
SOURCES = {
    "ORAS5":    CACHE_DIR / "phase1_oras5_NA_2deg.nc",
    "GLORYS12": CACHE_DIR / "multi" / "phase1_glorys12.nc",
}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="ORAS5",
                     choices=list(SOURCES.keys()),
                     help="reanalysis source label")
    args = ap.parse_args()
    src = args.source
    path = SOURCES[src]
    out_path = EKE_TS_DIR / f"{src}_atlantic_obs.json"
    try:
        # _basin_mean_eke_window may not be importable; reimplement inline
        from holocene_q_pilot import _basin_subset_2deg as _bs
    except ImportError:
        pass

    ds = xr.open_dataset(path)
    print(f"{src}: vars={list(ds.data_vars)}  time {ds['time'].values[0]} .. "
          f"{ds['time'].values[-1]}  N={ds.sizes['time']}")

    tos = ds["sst_raw"]
    zos = ds["ssh_raw"]

    # Q on the full available window
    Q, n_cells = _Q_for_window(tos, zos, "atlantic")
    print(f"  Q = {Q:+.4f}  ({n_cells} cells)")

    # basin-mean EKE = mean |grad SSH_mean|^2 on the 2-deg basin grid
    zos_2d = _basin_subset_2deg(zos, "atlantic")
    ssh_mean = zos_2d.mean("time")
    grad2 = (ssh_mean.differentiate("lat") ** 2
              + ssh_mean.differentiate("rlon") ** 2)
    eke_abs = float(grad2.mean(skipna=True).values)
    print(f"  basin-mean EKE (absolute) = {eke_abs:.6f}")

    # Cross-CMIP6 mean of pi_eke_mean (as rel-EKE denominator)
    bulk_pi_means = []
    for p in sorted((CACHE_DIR / "eke_timeseries").glob("*_atlantic_eke_ts.json")):
        # skip scenario-suffixed caches like *_atlantic_ssp245_eke_ts.json
        # and *_atlantic_1pctCO2_eke_ts.json
        stem = p.stem
        if any(tag in stem for tag in ("ssp245", "ssp370", "ssp126",
                                          "ssp119", "1pctCO2",
                                          "abrupt-4xCO2", "obs")):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        mu = d.get("pi_eke_mean")
        if mu is not None and np.isfinite(mu) and mu > 0:
            bulk_pi_means.append(float(mu))
    pi_mean_ensemble = float(np.mean(bulk_pi_means))
    rel_eke = eke_abs / pi_mean_ensemble
    print(f"  cross-CMIP6 pi_eke_mean = {pi_mean_ensemble:.6f} "
          f"(N={len(bulk_pi_means)} models)")
    print(f"  rel-EKE (ORAS5 / cross-CMIP6 pi mean) = {rel_eke:.4f}")

    EKE_TS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(
        source=src,
        basin="atlantic",
        time_range_start=str(ds["time"].values[0]),
        time_range_end=str(ds["time"].values[-1]),
        n_months=int(ds.sizes["time"]),
        Q=float(Q),
        n_cells_used=int(n_cells),
        eke_abs=float(eke_abs),
        rel_eke=float(rel_eke),
        pi_eke_mean_cross_cmip6=pi_mean_ensemble,
        n_cmip6_models_used=len(bulk_pi_means),
    ), indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
