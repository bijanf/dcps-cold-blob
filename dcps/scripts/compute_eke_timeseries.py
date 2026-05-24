"""Basin-mean EKE time series per CMIP6 model.

Companion to compute_epoch_eke_maps.py.  Where that script gives 3
spatial snapshots per epoch, this one gives a continuous time series:
30-yr rolling basin-mean |grad SSH|^2 from piControl through
historical+ssp585.

Saves per-(model, basin) JSON with the rolling time series, matching
the schema of the Q JSONs so the same plotter can overlay them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR, PKG_ROOT
sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import BASINS  # noqa: E402
from holocene_q_pilot import (  # noqa: E402
    _basin_subset_2deg, _year_of, _slice_by_year, WINDOW_YEARS,
)


OUT_DIR = CACHE_DIR / "eke_timeseries"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRIDE_YR = 10
PI_SPINUP_YEARS = 50


def _open_pangeo_member(model, experiment):
    import intake
    cat = intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    sub = cat.df[
        (cat.df.source_id == model)
        & (cat.df.experiment_id == experiment)
        & (cat.df.variable_id == "zos")
        & (cat.df.table_id == "Omon")
    ]
    if sub.empty:
        raise FileNotFoundError(f"{model} {experiment} zos")
    if "gn" in sub["grid_label"].values:
        sub = sub[sub["grid_label"] == "gn"]
    sub = sub.sort_values("member_id").iloc[:1]
    zstore = sub.iloc[0]["zstore"]
    return xr.open_zarr(zstore, consolidated=True, chunks={})["zos"]


def _basin_mean_eke_window(zos_window, basin):
    zos_2d = _basin_subset_2deg(zos_window, basin)
    ssh_mean = zos_2d.mean("time")
    grad2 = (ssh_mean.differentiate("lat") ** 2
              + ssh_mean.differentiate("rlon") ** 2)
    return float(grad2.mean(skipna=True).values)


def _process_one(model: str, basin: str,
                   experiment: str = "ssp585") -> bool:
    # ssp585 keeps the legacy no-suffix filename
    suffix = "" if experiment == "ssp585" else f"_{experiment}"
    out_path = OUT_DIR / f"{model}_{basin}{suffix}_eke_ts.json"
    if out_path.exists():
        print(f"  [{model} {basin}] cached, skip"); return True
    print(f"\n[{model} {basin}] starting ...")
    t0 = time.time()

    # piControl segments (sparser stride to keep it fast)
    try:
        zos_pi = _open_pangeo_member(model, "piControl")
        yr0 = _year_of(zos_pi["time"].values[0])
        yr1 = _year_of(zos_pi["time"].values[-1])
        pi_starts = list(range(yr0 + PI_SPINUP_YEARS,
                                  yr1 - WINDOW_YEARS + 1,
                                  3 * STRIDE_YR))  # coarse stride for pi
        pi_ts = []
        for s in pi_starts:
            try:
                v = _basin_mean_eke_window(
                    _slice_by_year(zos_pi, s, s + WINDOW_YEARS - 1), basin)
                pi_ts.append(v)
            except Exception:
                pi_ts.append(float("nan"))
        print(f"  piControl: {len(pi_ts)} windows ({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"  piControl FAILED: {type(e).__name__}: {e}")
        return False

    # historical + <experiment> concatenated
    try:
        zos_hi = _open_pangeo_member(model, "historical")
        try:
            zos_sp = _open_pangeo_member(model, experiment)
            zos_all = xr.concat([zos_hi, zos_sp], dim="time")
            end_year = 2099
        except Exception as ex_sp:
            print(f"  {experiment} open FAILED: {type(ex_sp).__name__}: {ex_sp}")
            zos_all = zos_hi
            end_year = 2014
        hi_ts = []; hi_centres = []
        for s in range(1850, end_year - WINDOW_YEARS + 2, STRIDE_YR):
            try:
                v = _basin_mean_eke_window(
                    _slice_by_year(zos_all, s, s + WINDOW_YEARS - 1), basin)
                hi_ts.append(v)
            except Exception:
                hi_ts.append(float("nan"))
            hi_centres.append(s + WINDOW_YEARS // 2)
        print(f"  historical+ssp: {len(hi_ts)} windows  (total {time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"  historical/ssp FAILED: {type(e).__name__}: {e}")
        return False

    summary = dict(
        model=model, basin=basin,
        window_years=WINDOW_YEARS,
        pi_eke=pi_ts,
        pi_starts=pi_starts,
        pi_eke_mean=float(np.nanmean(pi_ts)) if pi_ts else None,
        pi_eke_sd=float(np.nanstd(pi_ts)) if pi_ts else None,
        hist_eke=hi_ts,
        hist_centres=hi_centres,
        timestamp=int(time.time()),
    )
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic", choices=list(BASINS.keys()))
    ap.add_argument("--experiment", default="ssp585",
                     help="scenario after historical (e.g. ssp585, ssp245, ssp370)")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--max-models", type=int, default=None)
    args = ap.parse_args()

    bulk_dir = CACHE_DIR / "holocene_exit" / "bulk"
    if args.models is None:
        # use the existing ssp585 model list as the universe; we re-use
        # the same admitted ensemble for fair cross-scenario comparison
        models = sorted([p.stem.replace(f"_{args.basin}", "")
                          for p in bulk_dir.glob(f"*_{args.basin}.json")
                          if not any(tag in p.stem for tag in
                                       ("ssp245", "ssp370", "ssp126",
                                        "ssp119", "1pctCO2", "abrupt-4xCO2"))])
    else:
        models = args.models
    if args.max_models: models = models[:args.max_models]

    print("=" * 70)
    print(f" Basin-mean EKE time series  basin={args.basin}  "
          f"N_models={len(models)}")
    print("=" * 70)
    for m in models:
        try:
            _process_one(m, args.basin, experiment=args.experiment)
        except Exception as e:
            print(f"  [{m}] hard fail: {type(e).__name__}: {e}")
    print(f"\ndone -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
