"""Phase B / D bulk runner: loop Q-corridor detection over all CMIP6
models with full chain (piControl + historical + ssp585, both tos+zos).

For each model, takes a SINGLE member (the first r-index that has
both tos and zos), runs:
  - piControl: 8 non-overlapping 30-yr segments post-spinup -> Q null
  - historical+ssp585: rolling 30-yr windows 1850-2099, stride 10 yr
  - first-exit year: first window centre where Q > piControl 95-th pctile
  - Mann-Kendall on the piControl Q segments for stationarity gate

Writes one JSON per model under
``dcps/cache/holocene_exit/bulk/<model>_<basin>.json`` so progress is
incremental and recoverable.  Models that already have a JSON on disk
are skipped (resume-after-crash behaviour).

After all models complete, writes
``dcps/cache/holocene_exit/bulk_summary_<basin>.json`` with the
per-model first-exit-year + the cross-model distribution.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT
sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import BASINS  # noqa: E402
from holocene_q_pilot import (  # noqa: E402
    _open_pangeo, _Q_for_window, _slice_by_year, _year_of, _mann_kendall,
    WINDOW_YEARS, PI_SEGMENT_STRIDE_YEARS, PI_SPINUP_YEARS, EXIT_PCTILE,
)

OUT_DIR = CACHE_DIR / "holocene_exit"
BULK_DIR = OUT_DIR / "bulk"


def _process_one_model(model: str, basin: str, n_pi: int,
                        year_hist_start: int, year_ssp_end: int,
                        hist_stride: int, experiment: str = "ssp585"
                        ) -> dict | None:
    """Returns the per-model summary or None on hard failure."""
    # Backwards compatibility: ssp585 keeps the no-suffix filename so
    # all earlier code that reads <model>_<basin>.json still works.
    suffix = "" if experiment == "ssp585" else f"_{experiment}"
    out_path = BULK_DIR / f"{model}_{basin}{suffix}.json"
    if out_path.exists():
        try:
            cached = json.loads(out_path.read_text())
            print(f"  [{model}] cache hit -> first_exit_year={cached.get('first_exit_year')}")
            return cached
        except Exception:
            pass

    t0 = time.time()
    try:
        pi_tos = _open_pangeo(model, "piControl",  "tos")
        pi_zos = _open_pangeo(model, "piControl",  "zos")
    except Exception as e:
        print(f"  [{model}] piControl open FAILED -- {type(e).__name__}: {e}")
        return None

    yr0 = _year_of(pi_tos["time"].values[0])
    yr1 = _year_of(pi_tos["time"].values[-1])

    # piControl Q segments ---------------------------------------------
    pi_Q = []; pi_starts = []
    seg = yr0 + PI_SPINUP_YEARS
    while (seg + WINDOW_YEARS - 1) <= yr1 and len(pi_Q) < n_pi:
        s, e = seg, seg + WINDOW_YEARS - 1
        try:
            Q, n = _Q_for_window(_slice_by_year(pi_tos, s, e),
                                   _slice_by_year(pi_zos, s, e), basin)
            pi_Q.append(Q); pi_starts.append(int(s))
            print(f"  [{model}] pi {s}-{e}: Q={Q:+.3f} n={n}")
        except Exception as ex:
            print(f"  [{model}] pi {s}-{e}: FAILED ({type(ex).__name__})")
        seg += PI_SEGMENT_STRIDE_YEARS

    if len(pi_Q) < 4:
        print(f"  [{model}] too few piControl segments ({len(pi_Q)}); skip")
        return None

    pi_arr = np.asarray(pi_Q, dtype=float)
    threshold = float(np.nanpercentile(pi_arr, EXIT_PCTILE))
    tau, p_mk = _mann_kendall(pi_arr)
    print(f"  [{model}] pi mean={np.nanmean(pi_arr):+.3f} sd={np.nanstd(pi_arr):.3f}  "
          f"thr={threshold:+.3f}  MK p={p_mk:.3f}")

    # historical + <experiment> ----------------------------------------
    try:
        hi_tos = _open_pangeo(model, "historical", "tos")
        hi_zos = _open_pangeo(model, "historical", "zos")
    except Exception as ex:
        print(f"  [{model}] historical open FAILED: {ex}")
        return None
    try:
        sp_tos = _open_pangeo(model, experiment, "tos")
        sp_zos = _open_pangeo(model, experiment, "zos")
        import xarray as xr
        tos_all = xr.concat([hi_tos, sp_tos], dim="time")
        zos_all = xr.concat([hi_zos, sp_zos], dim="time")
        end_year = year_ssp_end
    except Exception as ex:
        print(f"  [{model}] {experiment} open FAILED: {ex}")
        tos_all, zos_all = hi_tos, hi_zos
        end_year = 2014

    hi_Q = []; centres = []
    win = year_hist_start
    while win + WINDOW_YEARS - 1 <= end_year:
        try:
            Q, n = _Q_for_window(_slice_by_year(tos_all, win, win + WINDOW_YEARS - 1),
                                   _slice_by_year(zos_all, win, win + WINDOW_YEARS - 1),
                                   basin)
            hi_Q.append(Q); centres.append(win + WINDOW_YEARS // 2)
        except Exception:
            hi_Q.append(float("nan")); centres.append(win + WINDOW_YEARS // 2)
        win += hist_stride

    hi_arr = np.asarray(hi_Q, dtype=float)
    centres_arr = np.asarray(centres)
    exit_idx = np.where(hi_arr > threshold)[0]
    first_exit = int(centres_arr[exit_idx[0]]) if exit_idx.size else None
    print(f"  [{model}] first_exit_year={first_exit}  ({time.time()-t0:.0f}s)")

    summary = dict(
        model=model, basin=basin,
        n_pi_segments=int(len(pi_Q)),
        pi_Q=pi_Q, pi_starts=pi_starts,
        pi_mean=float(np.nanmean(pi_arr)),
        pi_sd=float(np.nanstd(pi_arr)),
        pi_p95_threshold=threshold,
        pi_mk_tau=tau, pi_mk_p=p_mk,
        stationarity_gate_passed=bool(p_mk > 0.05),
        hist_Q=hi_Q, hist_centres=[int(c) for c in centres_arr],
        first_exit_year=first_exit,
        exit_pctile=EXIT_PCTILE,
        elapsed_s=round(time.time() - t0, 1),
    )
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic",
                     choices=list(BASINS.keys()))
    ap.add_argument("--n-pi-segments", type=int, default=8)
    ap.add_argument("--hist-stride", type=int, default=10)
    ap.add_argument("--year-hist-start", type=int, default=1850)
    ap.add_argument("--year-ssp-end",   type=int, default=2099)
    ap.add_argument("--experiment", default="ssp585",
                     help="scenario after historical (e.g. ssp585, ssp245, ssp370)")
    ap.add_argument("--models", nargs="*", default=None,
                     help="explicit model list; default = full-chain inventory")
    ap.add_argument("--max-models", type=int, default=None,
                     help="cap the number of models (for testing)")
    args = ap.parse_args()

    BULK_DIR.mkdir(parents=True, exist_ok=True)

    if args.models:
        models = args.models
    else:
        inv = json.loads((OUT_DIR / "models_inventory.json").read_text())
        models = inv["models_full_chain"]
    if args.max_models:
        models = models[:args.max_models]

    print("=" * 70)
    print(f" Bulk Q corridor detection  basin={args.basin}  N_models={len(models)}")
    print("=" * 70)

    per_model = {}
    for i, m in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] {m}")
        res = _process_one_model(
            m, args.basin, args.n_pi_segments,
            args.year_hist_start, args.year_ssp_end, args.hist_stride,
            experiment=args.experiment,
        )
        if res is not None:
            per_model[m] = res

    # Aggregate summary -------------------------------------------------
    exits = [r["first_exit_year"] for r in per_model.values()
             if r["first_exit_year"] is not None]
    summary = dict(
        basin=args.basin,
        n_models_attempted=len(models),
        n_models_ok=len(per_model),
        n_models_exit_detected=len(exits),
        first_exit_years=exits,
        first_exit_year_p10=float(np.percentile(exits, 10)) if exits else None,
        first_exit_year_p50=float(np.percentile(exits, 50)) if exits else None,
        first_exit_year_p90=float(np.percentile(exits, 90)) if exits else None,
        per_model_keys=list(per_model.keys()),
    )
    summary_path = OUT_DIR / f"bulk_summary_{args.basin}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {summary_path}")
    print(f"  models OK: {len(per_model)}/{len(models)}")
    if exits:
        print(f"  first-exit year distribution: "
              f"p10={summary['first_exit_year_p10']:.0f}  "
              f"p50={summary['first_exit_year_p50']:.0f}  "
              f"p90={summary['first_exit_year_p90']:.0f}")


if __name__ == "__main__":
    main()
