"""Dense piControl Q time series for the Holocene-equivalent forcing.

For each long-piControl CMIP6 model on Pangeo, compute Q on rolling
30-yr windows across the FULL piControl record (typically 500-1200 yr
per model).  Pool across models to build a Holocene-equivalent Q
distribution that's dense in time -- typically 100-200 Q points per
model -- and write a compact summary JSON.

This is the no-download alternative to CESM-LME ingest: we use
piControl as the Holocene-like null and treat the model-year axis as
nominal (any 30-yr piControl window is equally a Holocene-equivalent
estimate of Q).
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
    _open_pangeo, _Q_for_window, _slice_by_year, _year_of,
    WINDOW_YEARS,
)


OUT_DIR = CACHE_DIR / "holocene_exit" / "dense_picontrol"
STRIDE_YR = 10                          # rolling window stride


# Long-piControl models with both tos+zos on Pangeo, prioritised by
# piControl length and trustworthiness.  These three together span
# ~2700 model years of Holocene-equivalent forcing.
LONG_PI_MODELS = [
    "CESM2",                            # ~1200 yr
    "HadGEM3-GC31-LL",                  # ~500 yr
    "IPSL-CM6A-LR",                     # ~1000 yr
    "CanESM5",                          # ~1000 yr  (already used in pilot)
    "MPI-ESM1-2-LR",                    # ~1000 yr
    "MIROC6",                           # ~800 yr
]


def _process_model(model: str, basin: str) -> dict | None:
    out_path = OUT_DIR / f"{model}_{basin}_dense_pi.json"
    if out_path.exists():
        print(f"[{model}] cache hit -> skip")
        return json.loads(out_path.read_text())
    print(f"\n[{model}] streaming piControl ...")
    try:
        pi_tos = _open_pangeo(model, "piControl", "tos")
        pi_zos = _open_pangeo(model, "piControl", "zos")
    except Exception as e:
        print(f"  open FAILED: {type(e).__name__}: {e}")
        return None
    yr0 = _year_of(pi_tos["time"].values[0])
    yr1 = _year_of(pi_tos["time"].values[-1])
    span_yr = yr1 - yr0 + 1
    print(f"  span: {yr0}-{yr1} ({span_yr} yr)")

    Q_list = []; year_centres = []
    seg = yr0 + 50                       # 50-yr spin-up exclusion
    while (seg + WINDOW_YEARS - 1) <= yr1:
        s, e = seg, seg + WINDOW_YEARS - 1
        t0 = time.time()
        try:
            Q, n = _Q_for_window(
                _slice_by_year(pi_tos, s, e),
                _slice_by_year(pi_zos, s, e), basin,
            )
            Q_list.append(Q); year_centres.append(s + WINDOW_YEARS // 2)
            print(f"  {s}-{e}: Q={Q:+.3f} n={n} ({time.time()-t0:.0f}s)")
        except Exception as ex:
            print(f"  {s}-{e}: FAILED ({type(ex).__name__})")
            Q_list.append(float("nan")); year_centres.append(s + WINDOW_YEARS // 2)
        seg += STRIDE_YR

    summary = dict(
        model=model, basin=basin,
        window_years=WINDOW_YEARS, stride_yr=STRIDE_YR,
        n_windows=len(Q_list),
        Q=Q_list,
        year_centres=year_centres,
        Q_mean=float(np.nanmean(Q_list)) if Q_list else None,
        Q_sd=float(np.nanstd(Q_list)) if Q_list else None,
        Q_p95=float(np.nanpercentile(Q_list, 95)) if Q_list else None,
        elapsed_sec=time.time() - t0,
    )
    # Atomic write
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.rename(out_path)
    print(f"  wrote {out_path}  (n_windows={len(Q_list)})")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic",
                     choices=list(BASINS.keys()))
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = args.models or LONG_PI_MODELS
    print("=" * 70)
    print(f" Dense piControl Q timeline  basin={args.basin}  "
          f"N_models={len(models)}")
    print("=" * 70)
    per_model = {}
    for m in models:
        r = _process_model(m, args.basin)
        if r is not None:
            per_model[m] = r
    # Aggregate
    all_Q = []
    for r in per_model.values():
        all_Q.extend(r["Q"])
    all_Q = np.asarray(all_Q, dtype=float)
    all_Q = all_Q[np.isfinite(all_Q)]
    print()
    print(f"=== aggregate piControl Q ({len(per_model)} models) ===")
    print(f"  total 30-yr windows: {all_Q.size}")
    print(f"  Q mean: {all_Q.mean():+.3f}")
    print(f"  Q sd:   {all_Q.std():.3f}")
    print(f"  Q 95th pctile: {np.percentile(all_Q, 95):+.3f}")
    print(f"  Q 99th pctile: {np.percentile(all_Q, 99):+.3f}")
    out_summary = OUT_DIR / f"summary_{args.basin}.json"
    out_summary.write_text(json.dumps({
        "basin": args.basin,
        "n_models": len(per_model),
        "n_windows_total": int(all_Q.size),
        "Q_mean": float(all_Q.mean()),
        "Q_sd": float(all_Q.std()),
        "Q_p95": float(np.percentile(all_Q, 95)),
        "Q_p99": float(np.percentile(all_Q, 99)),
        "models": list(per_model.keys()),
    }, indent=2))
    print(f"wrote {out_summary}")


if __name__ == "__main__":
    main()
