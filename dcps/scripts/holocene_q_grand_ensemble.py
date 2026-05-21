"""Grand-ensemble Q corridor: loop over members of a single model.

For a chosen model and experiment, iterate over the available
``member_id`` values on Pangeo and compute Q on rolling 30-yr windows
per member.  Outputs the ensemble-mean Q(t), per-member spread, and
the piControl null distribution pooled across all piControl members.

Designed to be re-used for any (model, experiment) combo that has
multiple members.  Default is CanESM5 historical (65 members on
Pangeo); CanESM5 1pctCO2 (6 members) and ssp585 (50 members) work
the same way.

Outputs:
  dcps/cache/holocene_exit/grand_<model>_<exp>_<basin>.json
  manuscript/figs/fig_grand_<model>_<exp>_<basin>.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style, DOUBLE_COL_IN
apply_nature_style()

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import BASINS  # noqa: E402
from holocene_q_pilot import (  # noqa: E402
    _Q_for_window, _slice_by_year, _year_of,
    WINDOW_YEARS, EXIT_PCTILE, PI_SPINUP_YEARS,
)


OUT_DIR = CACHE_DIR / "holocene_exit"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"


def _open_pangeo_member(model: str, experiment: str, variable: str,
                         member: str) -> xr.DataArray:
    """Open a SPECIFIC member of a CMIP6 zarr store."""
    import intake
    cat = intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    rows = cat.df[
        (cat.df.source_id == model)
        & (cat.df.experiment_id == experiment)
        & (cat.df.variable_id == variable)
        & (cat.df.table_id == "Omon")
        & (cat.df.member_id == member)
    ]
    if rows.empty:
        raise FileNotFoundError(
            f"{model} {experiment} {variable} member={member} not found")
    if "gn" in rows["grid_label"].values:
        rows = rows[rows["grid_label"] == "gn"]
    zstore = rows.sort_values("version").iloc[-1]["zstore"]
    ds = xr.open_zarr(zstore, consolidated=True, chunks={})
    return ds[variable]


def _list_members(model: str, experiment: str) -> list[str]:
    """List member_ids that have BOTH tos and zos on Pangeo.

    If ``experiment`` contains '+', return the INTERSECTION of members
    that have tos+zos in EVERY part (e.g. 'historical+ssp585' returns
    only members where the same r-index has both historical and ssp585
    with tos and zos).
    """
    import intake
    cat = intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    parts = experiment.split("+") if "+" in experiment else [experiment]
    sets = []
    for exp in parts:
        base = cat.df[
            (cat.df.source_id == model)
            & (cat.df.experiment_id == exp)
            & (cat.df.table_id == "Omon")
        ]
        tos_m = set(base[base.variable_id == "tos"]["member_id"])
        zos_m = set(base[base.variable_id == "zos"]["member_id"])
        sets.append(tos_m & zos_m)
    common = sets[0]
    for s in sets[1:]:
        common = common & s
    return sorted(common,
                   key=lambda m: (int(m.split("i")[0][1:]) if m.startswith("r")
                                  else 999))


def _run_member(model, experiment, member, basin, year_starts):
    """Compute Q on each 30-yr window for ONE member.

    If experiment includes a '+' (e.g. 'historical+ssp585'), open BOTH
    runs for the same member and concatenate along time so the
    rolling window extends past 2014 into the SSP era.
    """
    print(f"  open {model} {experiment} member={member}")
    if "+" in experiment:
        parts = experiment.split("+")
        tos_list = []; zos_list = []
        for exp in parts:
            try:
                t = _open_pangeo_member(model, exp, "tos", member)
                z = _open_pangeo_member(model, exp, "zos", member)
                tos_list.append(t); zos_list.append(z)
            except Exception as e:
                print(f"    skipping {exp} for {member}: {type(e).__name__}")
        if not tos_list:
            raise FileNotFoundError(f"no parts loaded for {experiment}")
        import xarray as xr
        tos = xr.concat(tos_list, dim="time")
        zos = xr.concat(zos_list, dim="time")
    else:
        tos = _open_pangeo_member(model, experiment, "tos", member)
        zos = _open_pangeo_member(model, experiment, "zos", member)
    Qs = []
    for win_start in year_starts:
        win_end = win_start + WINDOW_YEARS - 1
        t0 = time.time()
        tos_win = _slice_by_year(tos, win_start, win_end)
        zos_win = _slice_by_year(zos, win_start, win_end)
        if tos_win.sizes.get("time", 0) < 12 * WINDOW_YEARS * 0.8:
            Qs.append(float("nan")); continue
        Q, n = _Q_for_window(tos_win, zos_win, basin)
        Qs.append(Q)
        print(f"    {experiment[:4]} {win_start}-{win_end}  Q={Q:+.3f}  "
              f"n={n}  ({time.time() - t0:.0f}s)")
    return Qs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="CanESM5")
    ap.add_argument("--experiment", default="historical")
    ap.add_argument("--basin", default="atlantic",
                     choices=list(BASINS.keys()))
    ap.add_argument("--n-members", type=int, default=10,
                     help="how many members to use (first N by r-index)")
    ap.add_argument("--year-start", type=int, default=1850)
    ap.add_argument("--year-end",   type=int, default=2014)
    ap.add_argument("--stride", type=int, default=10,
                     help="window-start stride in years")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f" Grand ensemble Q: model={args.model}  exp={args.experiment}  "
          f"basin={args.basin}")
    print("=" * 70)

    members = _list_members(args.model, args.experiment)
    print(f"\n  available members (tos+zos): {len(members)} -> first "
          f"{args.n_members} used:")
    members = members[:args.n_members]
    for m in members: print(f"    {m}")

    year_starts = list(range(args.year_start,
                                args.year_end - WINDOW_YEARS + 2,
                                args.stride))
    centres = [ys + WINDOW_YEARS // 2 for ys in year_starts]
    print(f"\n  {len(year_starts)} windows per member "
          f"({args.year_start}-{args.year_end}, stride={args.stride}yr)")

    # Per-member checkpoint cache lets the run resume after kill/crash.
    member_cache_dir = (OUT_DIR / "members" /
                         f"{args.model}_{args.experiment}_{args.basin}")
    member_cache_dir.mkdir(parents=True, exist_ok=True)

    all_Q = np.full((len(members), len(year_starts)), np.nan)
    for i, m in enumerate(members):
        cache_path = member_cache_dir / f"{m}.json"
        if cache_path.exists():
            try:
                row = json.loads(cache_path.read_text())
                if len(row.get("Q", [])) == len(year_starts):
                    all_Q[i, :] = row["Q"]
                    print(f"  member {m}: cached, skipping")
                    continue
            except Exception:
                pass
        try:
            row_Q = _run_member(args.model, args.experiment, m,
                                  args.basin, year_starts)
            all_Q[i, :] = row_Q
            # Atomic checkpoint write
            tmp = cache_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"member": m, "Q": list(row_Q)},
                                         indent=2))
            tmp.rename(cache_path)
        except Exception as e:
            print(f"  member {m}: FAILED -- {type(e).__name__}: {e}")

    mean_Q = np.nanmean(all_Q, axis=0)
    median_Q = np.nanmedian(all_Q, axis=0)
    p10 = np.nanpercentile(all_Q, 10, axis=0)
    p90 = np.nanpercentile(all_Q, 90, axis=0)

    summary = dict(
        model=args.model, experiment=args.experiment, basin=args.basin,
        n_members_used=int(len(members)), members=list(members),
        window_years=WINDOW_YEARS, stride=args.stride,
        year_centres=[int(c) for c in centres],
        mean_Q=[float(x) for x in mean_Q],
        median_Q=[float(x) for x in median_Q],
        p10_Q=[float(x) for x in p10],
        p90_Q=[float(x) for x in p90],
        per_member_Q=[[float(x) for x in row] for row in all_Q],
    )
    out_json = (OUT_DIR
                 / f"grand_{args.model}_{args.experiment}_{args.basin}.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_json}")

    # Compare against pilot's piControl null if available.
    pi_path = OUT_DIR / f"pilot_{args.model}_{args.basin}.json"
    pi_threshold = None
    pi_min = pi_max = None
    if pi_path.exists():
        pi_json = json.loads(pi_path.read_text())
        pi_threshold = pi_json["pi_p95_threshold"]
        pi_min = min(pi_json["pi_Q"])
        pi_max = max(pi_json["pi_Q"])
        print(f"  reading piControl null from {pi_path}: "
              f"threshold={pi_threshold:+.3f}")
    else:
        print(f"  no piControl pilot at {pi_path}; "
              f"using model-internal range for envelope")

    # ---- figure ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.4))
    centres_arr = np.asarray(centres)
    if pi_min is not None:
        ax.axhspan(pi_min, pi_max, color="0.85", zorder=0,
                    label="piControl Q range")
    if pi_threshold is not None:
        ax.axhline(pi_threshold, color="C3", lw=0.8, ls="--",
                    label=f"piControl {EXIT_PCTILE:.0f}-th pctile")
    # per-member faint
    for row in all_Q:
        ax.plot(centres_arr, row, color="C0", alpha=0.18, lw=0.5)
    # spread band
    finite = np.isfinite(p10) & np.isfinite(p90)
    ax.fill_between(centres_arr[finite], p10[finite], p90[finite],
                     color="C0", alpha=0.25, label="member 10-90 pctile")
    # ensemble median line
    ax.plot(centres_arr, median_Q, "o-", color="C0", ms=3, lw=1.2,
             label=f"ensemble median (N={len(members)})")
    # exit year if any
    if pi_threshold is not None:
        exit_idx = np.where(median_Q > pi_threshold)[0]
        if exit_idx.size:
            yr = int(centres_arr[exit_idx[0]])
            ax.axvline(yr, color="C3", lw=0.6, ls=":",
                        label=f"first exit (median) {yr}")
            print(f"  first-exit year (ensemble median): {yr}")
    ax.set_xlabel("Window centre year")
    ax.set_ylabel(r"Quiescence Index  $Q = -\rho$")
    ax.set_xlim(args.year_start, args.year_end)
    ax.legend(loc="best", frameon=False, fontsize=7)

    out_pdf = (MANUSCRIPT_FIGS
                / f"fig_grand_{args.model}_{args.experiment}_{args.basin}.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
