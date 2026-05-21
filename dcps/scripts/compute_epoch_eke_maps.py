"""Per-epoch climatological EKE maps for each CMIP6 model on Pangeo.

For each model with bulk Q results already cached, compute the
time-mean |grad SSH|^2 over THREE epochs on the 2-deg basin grid:

  - piControl  : ~500 yr post-spinup            (Holocene-equivalent proxy)
  - historical : 1850-2014                       (industrial)
  - ssp585     : 2015-2099                       (high-emissions projection)

For each (model, basin), save 3 maps as NetCDF.  These feed into a
later multi-panel figure showing how the EKE field itself shifts
between epochs and how Q corridor exit follows.

Resumable per-(model, basin) cache.  Idempotent re-runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR, PKG_ROOT
sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import BASINS, basin_target_grid  # noqa: E402
from holocene_q_pilot import _basin_subset_2deg, _year_of, _slice_by_year  # noqa: E402


OUT_DIR = CACHE_DIR / "eke_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _open_pangeo_member(model, experiment, variable):
    import intake
    cat = intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    sub = cat.df[
        (cat.df.source_id == model)
        & (cat.df.experiment_id == experiment)
        & (cat.df.variable_id == variable)
        & (cat.df.table_id == "Omon")
    ]
    if sub.empty:
        raise FileNotFoundError(f"{model} {experiment} {variable}")
    if "gn" in sub["grid_label"].values:
        sub = sub[sub["grid_label"] == "gn"]
    sub = sub.sort_values("member_id").iloc[:1]
    zstore = sub.iloc[0]["zstore"]
    ds = xr.open_zarr(zstore, consolidated=True, chunks={})
    return ds[variable]


def _compute_eke_map(zos, basin, year_start=None, year_end=None):
    """Return the 2-deg basin-grid time-mean |grad SSH|^2 field as
    DataArray (lat, rlon)."""
    if year_start is not None:
        zos = _slice_by_year(zos, year_start, year_end)
    zos_2d = _basin_subset_2deg(zos, basin)
    ssh_mean = zos_2d.mean("time")
    grad2 = (ssh_mean.differentiate("lat") ** 2
              + ssh_mean.differentiate("rlon") ** 2)
    return grad2.rename(f"eke_{basin}")


def _process_one(model: str, basin: str) -> bool:
    out_path = OUT_DIR / f"{model}_{basin}_epoch_eke.nc"
    if out_path.exists():
        print(f"  [{model} {basin}] cached, skip")
        return True
    print(f"\n[{model} {basin}] starting ...")
    t0 = time.time()
    maps = {}
    # piControl: take a 165-yr slice after 50-yr spinup
    try:
        zos_pi = _open_pangeo_member(model, "piControl", "zos")
        yr0_pi = _year_of(zos_pi["time"].values[0])
        maps["pi"] = _compute_eke_map(zos_pi, basin,
                                          yr0_pi + 50, yr0_pi + 50 + 164)
        print(f"  piControl ({yr0_pi+50}-{yr0_pi+214}) done ({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"  piControl FAILED: {type(e).__name__}: {e}")
        return False
    t1 = time.time()
    try:
        zos_hi = _open_pangeo_member(model, "historical", "zos")
        maps["historical"] = _compute_eke_map(zos_hi, basin, 1850, 2014)
        print(f"  historical (1850-2014) done ({time.time()-t1:.0f}s)")
    except Exception as e:
        print(f"  historical FAILED: {type(e).__name__}: {e}")
        return False
    t2 = time.time()
    try:
        zos_sp = _open_pangeo_member(model, "ssp585", "zos")
        maps["ssp585"] = _compute_eke_map(zos_sp, basin, 2015, 2099)
        print(f"  ssp585 (2015-2099) done ({time.time()-t2:.0f}s)")
    except Exception as e:
        print(f"  ssp585 FAILED ({type(e).__name__}); piControl+historical only saved")

    ds = xr.Dataset({k: v for k, v in maps.items()})
    ds.attrs["model"] = model
    ds.attrs["basin"] = basin
    ds.to_netcdf(out_path)
    print(f"  wrote {out_path}  (total {time.time()-t0:.0f}s)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic", choices=list(BASINS.keys()))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--max-models", type=int, default=None)
    args = ap.parse_args()

    # Default model list: every model with a bulk JSON
    bulk_dir = CACHE_DIR / "holocene_exit" / "bulk"
    if args.models is None:
        models = sorted([
            p.stem.replace("_atlantic", "") for p in bulk_dir.glob("*.json")
        ])
    else:
        models = args.models
    if args.max_models: models = models[:args.max_models]

    print("=" * 70)
    print(f" Epoch EKE maps  basin={args.basin}  N_models={len(models)}")
    print("=" * 70)

    for i, m in enumerate(models):
        try:
            _process_one(m, args.basin)
        except Exception as e:
            print(f"  [{m}] hard failure: {type(e).__name__}: {e}")

    print(f"\ncached EKE maps at {OUT_DIR}/")


if __name__ == "__main__":
    main()
