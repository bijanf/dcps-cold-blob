"""Basin-mean EKE time series for the 1pctCO2 and abrupt-4xCO2
idealised CO2-only experiments.

For each (model, experiment) listed in the existing Q caches at
dcps/cache/holocene_exit/auto/pangeo_*_<experiment>_<basin>.json,
opens zos from Pangeo and computes basin-mean |grad <SSH>|^2 on
30-yr rolling windows (stride 10 yr).

Reuses the same _basin_mean_eke_window definition as
compute_eke_timeseries.py so EKE numbers are directly comparable
across experiments.

Output: dcps/cache/eke_timeseries/<model>_<basin>_<experiment>_eke_ts.json
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
from multi_basin_quiescence import BASINS  # noqa: E402
from holocene_q_pilot import (  # noqa: E402
    _basin_subset_2deg, _slice_by_year, _year_of, WINDOW_YEARS,
)

OUT_DIR = CACHE_DIR / "eke_timeseries"
OUT_DIR.mkdir(parents=True, exist_ok=True)
Q_CACHE_DIR = CACHE_DIR / "holocene_exit" / "auto"

STRIDE_YR = 10
EXPERIMENTS = ("1pctCO2", "abrupt-4xCO2")


def _open_pangeo_member(model, experiment, variable="zos"):
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


def _basin_mean_eke_window(zos_window, basin):
    zos_2d = _basin_subset_2deg(zos_window, basin)
    ssh_mean = zos_2d.mean("time")
    grad2 = (ssh_mean.differentiate("lat") ** 2
              + ssh_mean.differentiate("rlon") ** 2)
    return float(grad2.mean(skipna=True).values)


def _process_one(model: str, experiment: str, basin: str) -> bool:
    out_path = OUT_DIR / f"{model}_{basin}_{experiment}_eke_ts.json"
    if out_path.exists():
        print(f"  [{model} {experiment} {basin}] cached, skip")
        return True
    print(f"\n[{model} {experiment} {basin}] starting ...")
    t0 = time.time()
    try:
        zos = _open_pangeo_member(model, experiment, "zos")
    except Exception as e:
        print(f"  open zos FAILED: {type(e).__name__}: {e}")
        return False
    yr0 = _year_of(zos["time"].values[0])
    yr1 = _year_of(zos["time"].values[-1])
    print(f"  zos {yr0}..{yr1}  ({yr1-yr0+1} yr)")

    ts = []
    centres = []
    for s in range(yr0, yr1 - WINDOW_YEARS + 2, STRIDE_YR):
        e = s + WINDOW_YEARS - 1
        try:
            v = _basin_mean_eke_window(_slice_by_year(zos, s, e), basin)
            ts.append(v)
        except Exception as ex:
            print(f"  {s}-{e} FAILED ({type(ex).__name__})")
            ts.append(float("nan"))
        centres.append(s + WINDOW_YEARS // 2)

    arr_f = np.asarray([x for x in ts if np.isfinite(x)], dtype=float)
    summary = dict(
        model=model, basin=basin, experiment=experiment,
        window_years=WINDOW_YEARS, stride_yr=STRIDE_YR,
        year_range=[int(yr0), int(yr1)],
        eke=ts, centres=centres,
        eke_mean=float(arr_f.mean()) if arr_f.size else None,
        eke_sd=float(arr_f.std()) if arr_f.size else None,
        n_windows=len(ts),
        elapsed_s=round(time.time() - t0, 1),
        timestamp=int(time.time()),
    )
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.rename(out_path)
    print(f"  wrote {out_path}  ({time.time()-t0:.0f}s)")
    return True


def _models_with_q_cache(experiment, basin):
    models = set()
    for p in sorted(Q_CACHE_DIR.glob(
            f"pangeo_*_{experiment}_{basin}.json")):
        try:
            d = json.loads(p.read_text())
            m = d.get("target", {}).get("model")
            if m:
                models.add(m)
        except Exception:
            continue
    return sorted(models)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    ap.add_argument("--experiment", choices=list(EXPERIMENTS) + ["both"],
                     default="both")
    args = ap.parse_args()

    exps = EXPERIMENTS if args.experiment == "both" else (args.experiment,)
    for exp in exps:
        models = _models_with_q_cache(exp, args.basin)
        print(f"\n=== {exp}  basin={args.basin}  models={len(models)} ===")
        ok = 0
        for m in models:
            if _process_one(m, exp, args.basin):
                ok += 1
        print(f"\n{exp}: {ok}/{len(models)} OK")


if __name__ == "__main__":
    main()
