"""Multi-model first-exit-year summary on the Atlantic Q corridor.

Reads every gate-passing bulk JSON in
``dcps/cache/holocene_exit/bulk/`` and produces:

  - dcps/cache/holocene_exit/audit/first_exit_<basin>.json
  - manuscript/figs/fig_first_exit_multimodel_<basin>.pdf

Two-panel figure:
  (a) per-model first-exit year as a horizontal scatter sorted by year,
      shape encodes whether the exit lies in the early-industrial
      cluster, the aerosol-cooling gap, or the post-1990 cluster.
  (b) histogram + Gaussian KDE of first-exit years, dashed vertical
      lines at the pre-registered cluster boundaries 1940 and 1990.

Pre-registered cluster split (sec. 7, PREREGISTRATION_ATLANTIC.md):
  early-industrial  (.., 1940]
  aerosol-gap       (1940, 1990)
  post-1990         [1990, ..)
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style, DOUBLE_COL_IN
apply_nature_style()

BULK_DIR = CACHE_DIR / "holocene_exit" / "bulk"
AUDIT_DIR = CACHE_DIR / "holocene_exit" / "audit"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

EARLY_END = 1940
LATE_START = 1990
ALPHA = 0.05


def _cluster(year: int) -> str:
    if year <= EARLY_END: return "early"
    if year >= LATE_START: return "late"
    return "gap"


def _load_rows(basin: str) -> list[dict]:
    rows = []
    for p in sorted(BULK_DIR.glob(f"*_{basin}.json")):
        d = json.loads(p.read_text())
        if "pi_mk_p" not in d: continue
        gate = bool(d.get("stationarity_gate_passed",
                           d["pi_mk_p"] > ALPHA))
        if not gate: continue
        rows.append(dict(
            model=d["model"], basin=d["basin"],
            first_exit_year=d.get("first_exit_year"),
            pi_p95=float(d.get("pi_p95_threshold", float("nan"))),
            pi_mean=float(d.get("pi_mean", float("nan"))),
            pi_sd=float(d.get("pi_sd", float("nan"))),
            mk_p=float(d["pi_mk_p"]),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    args = ap.parse_args()

    rows = _load_rows(args.basin)
    if not rows:
        print(f"no gate-passing bulk JSONs for basin={args.basin}")
        return
    with_exit = [r for r in rows if r["first_exit_year"] is not None]
    no_exit = [r for r in rows if r["first_exit_year"] is None]

    years = np.asarray([r["first_exit_year"] for r in with_exit],
                        dtype=float)
    for r in with_exit:
        r["cluster"] = _cluster(int(r["first_exit_year"]))

    n_early = sum(r["cluster"] == "early" for r in with_exit)
    n_gap = sum(r["cluster"] == "gap" for r in with_exit)
    n_late = sum(r["cluster"] == "late" for r in with_exit)

    print(f"basin={args.basin}  admitted={len(rows)}  "
          f"with-exit={len(with_exit)}  no-exit={len(no_exit)}")
    print(f"  early(<= {EARLY_END}): {n_early}")
    print(f"  gap   ({EARLY_END}-{LATE_START}): {n_gap}")
    print(f"  late  (>= {LATE_START}): {n_late}")
    if no_exit:
        print(f"  no-exit models: {', '.join(r['model'] for r in no_exit)}")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = AUDIT_DIR / f"first_exit_{args.basin}.json"
    summary = dict(
        basin=args.basin, n_admitted=len(rows),
        n_with_exit=len(with_exit), n_no_exit=len(no_exit),
        cluster_thresholds=dict(early_end=EARLY_END,
                                  late_start=LATE_START),
        cluster_counts=dict(early=n_early, gap=n_gap, late=n_late),
        median_exit_year=float(np.median(years)) if years.size else None,
        min_exit_year=int(years.min()) if years.size else None,
        max_exit_year=int(years.max()) if years.size else None,
        no_exit_models=[r["model"] for r in no_exit],
        per_model=with_exit + no_exit,
    )
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")

    # ---------- figure ----------
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.4),
        gridspec_kw=dict(width_ratios=[1.0, 1.1], wspace=0.32),
    )

    sorted_rows = sorted(with_exit, key=lambda r: r["first_exit_year"])
    colour = dict(early="C0", gap="C7", late="C3")
    marker = dict(early="o", gap="s", late="^")
    y_pos = np.arange(len(sorted_rows))
    for i, r in enumerate(sorted_rows):
        c = r["cluster"]
        axL.scatter(r["first_exit_year"], i, s=14,
                      c=colour[c], marker=marker[c],
                      edgecolor="0.2", linewidth=0.4, zorder=3)
    axL.axvline(EARLY_END, color="0.5", lw=0.5, ls="--")
    axL.axvline(LATE_START, color="0.5", lw=0.5, ls="--")
    axL.set_yticks(y_pos)
    axL.set_yticklabels([r["model"] for r in sorted_rows], fontsize=5)
    axL.set_xlabel("First-exit year (window centre)")
    axL.set_xlim(1850, 2100)
    axL.set_ylim(-0.5, len(sorted_rows) - 0.5)
    axL.text(0.02, 1.04, "(a)", transform=axL.transAxes,
              ha="left", va="bottom", fontsize=8, fontweight="bold")

    bins = np.arange(1850, 2105, 10)
    axR.hist(years, bins=bins, color="0.7", edgecolor="white",
              linewidth=0.5, alpha=0.9)
    if years.size >= 3:
        kde = gaussian_kde(years, bw_method=0.25)
        xx = np.linspace(1850, 2100, 400)
        kde_curve = kde(xx) * len(years) * 10
        axR.plot(xx, kde_curve, color="C3", lw=1.0,
                  label="KDE (bw=0.25)")
    axR.axvline(EARLY_END, color="0.5", lw=0.5, ls="--")
    axR.axvline(LATE_START, color="0.5", lw=0.5, ls="--")
    axR.text(EARLY_END - 5, axR.get_ylim()[1] * 0.95, "1940",
              ha="right", va="top", fontsize=5, color="0.4")
    axR.text(LATE_START + 5, axR.get_ylim()[1] * 0.95, "1990",
              ha="left", va="top", fontsize=5, color="0.4")
    axR.set_xlim(1850, 2100)
    axR.set_xlabel("First-exit year")
    axR.set_ylabel("Models per 10-yr bin")
    axR.legend(loc="upper right", frameon=False, fontsize=6)
    axR.text(0.02, 1.04, "(b)", transform=axR.transAxes,
              ha="left", va="bottom", fontsize=8, fontweight="bold")
    axR.text(0.50, -0.32,
              f"early {n_early}  |  gap {n_gap}  |  late {n_late}  |  "
              f"no-exit {len(no_exit)}",
              transform=axR.transAxes,
              ha="center", va="top", fontsize=6, color="0.3")

    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    out_pdf = (MANUSCRIPT_FIGS
                / f"fig_first_exit_multimodel_{args.basin}.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
