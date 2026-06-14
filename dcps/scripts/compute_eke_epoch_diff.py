"""Per-model EKE maps for narrow comparison epochs.

Companion to compute_epoch_eke_maps.py (which uses wide epochs:
piControl, full-historical, full-ssp585).  This script computes
time-mean |grad SSH|^2 on the 2-deg basin grid for three NARROW
epochs that match the way IPCC AR6 reports climate-change maps:

  baseline  : 1850-1900   (early historical)
  mid       : 2030-2060   (mid-future, historical+ssp585 splice if needed)
  far       : 2070-2099   (far-future, ssp585)

Per-model output:
  dcps/cache/eke_epoch_diff/<model>_<basin>_epoch_diff.nc
  data vars: baseline, mid, far  (each on lat, rlon)

Resumable: skips models already on disk.
"""
from __future__ import annotations

import argparse
import sys
import time

import xarray as xr

from dcps.config import CACHE_DIR, PKG_ROOT
sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import BASINS  # noqa: E402
from holocene_q_pilot import _basin_subset_2deg, _slice_by_year  # noqa: E402


OUT_DIR = CACHE_DIR / "eke_epoch_diff"
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
    return xr.open_zarr(zstore, consolidated=True, chunks={})[variable]


def _epoch_mean_eke(zos, basin, year_start, year_end):
    zos_w = _slice_by_year(zos, year_start, year_end)
    zos_2d = _basin_subset_2deg(zos_w, basin)
    ssh_mean = zos_2d.mean("time")
    grad2 = (ssh_mean.differentiate("lat") ** 2
              + ssh_mean.differentiate("rlon") ** 2)
    return grad2


def _process_one(model: str, basin: str, experiment: str = "ssp585") -> bool:
    """Per-model 30-year-mean |grad SSH|^2 maps for one scenario.

    For ssp585: writes baseline (1850-1900), mid (2030-2060), far (2070-2099).
    For other ssp* scenarios: writes mid + far only (baseline is shared
    across scenarios and lives in the canonical ssp585 cache file).
    """
    suffix = "" if experiment == "ssp585" else f"_{experiment}"
    out_path = OUT_DIR / f"{model}_{basin}_epoch_diff{suffix}.nc"
    if out_path.exists():
        print(f"  [{model}/{experiment}] cached, skip"); return True
    print(f"\n[{model} {basin} {experiment}] starting ...")
    t0 = time.time()
    maps = {}
    if experiment == "ssp585":
        try:
            zos_hi = _open_pangeo_member(model, "historical", "zos")
        except Exception as e:
            print(f"  historical FAILED: {e}"); return False
    try:
        zos_sp = _open_pangeo_member(model, experiment, "zos")
    except Exception as e:
        print(f"  {experiment} FAILED ({e}); skipping this model")
        return False
    try:
        if experiment == "ssp585":
            maps["baseline"] = _epoch_mean_eke(zos_hi, basin, 1850, 1900)
            print(f"  baseline (1850-1900) done ({time.time()-t0:.0f}s)")
        t1 = time.time()
        maps["mid"]      = _epoch_mean_eke(zos_sp, basin, 2030, 2060)
        print(f"  mid (2030-2060) done ({time.time()-t1:.0f}s)")
        t2 = time.time()
        maps["far"]      = _epoch_mean_eke(zos_sp, basin, 2070, 2099)
        print(f"  far (2070-2099) done ({time.time()-t2:.0f}s)")
    except Exception as e:
        print(f"  epoch mean FAILED: {type(e).__name__}: {e}")
        return False
    ds = xr.Dataset({k: v for k, v in maps.items()})
    ds.attrs["model"] = model
    ds.attrs["basin"] = basin
    ds.attrs["experiment"] = experiment
    tmp = out_path.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp)
    tmp.rename(out_path)
    print(f"  wrote {out_path}  (total {time.time()-t0:.0f}s)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic",
                     choices=list(BASINS.keys()))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--experiment", default="ssp585",
                     help="scenario to compute (e.g. ssp245, ssp585)")
    args = ap.parse_args()

    bulk_dir = CACHE_DIR / "holocene_exit" / "bulk"
    if args.models is None:
        models = sorted([p.stem.replace(f"_{args.basin}", "")
                          for p in bulk_dir.glob(f"*_{args.basin}.json")])
    else:
        models = args.models
    print("=" * 70)
    print(f" Narrow-epoch EKE maps  basin={args.basin}  experiment={args.experiment}  "
          f"N_models={len(models)}")
    print("=" * 70)
    for m in models:
        try: _process_one(m, args.basin, experiment=args.experiment)
        except Exception as e:
            print(f"  [{m}] hard fail: {type(e).__name__}: {e}")
    print(f"\ndone -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
