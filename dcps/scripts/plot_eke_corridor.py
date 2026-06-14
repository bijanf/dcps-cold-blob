"""EKE-corridor analogue of plot_picontrol_corridor.py.

Pools basin-mean EKE across the same gate-passed CMIP6 admission list
used for the Q corridor.  Because absolute EKE units differ across
models, each model's basin-mean EKE is normalised by its own
piControl-mean EKE before pooling.  The resulting quantity is
dimensionless ("relative EKE", 1.0 = each model's piControl mean).

Two panels:
  (a) per-model piControl relative-EKE segments (one column per model;
      grey 10-90 percentile envelope of pooled piControl rel-EKE).
  (b) pooled piControl relative-EKE histogram + KDE, with the IQR
      of (i) historical 1850-1900, (ii) mid-future 2030-2060, and
      (iii) far-future 2070-2099 relative-EKE overlaid as strips.

Output: manuscript/figs/fig_eke_corridor_<basin>.pdf
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
EKE_TS_DIR = CACHE_DIR / "eke_timeseries"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"
ALPHA = 0.05


def _admitted_models(basin: str) -> set[str]:
    out = set()
    for p in sorted(BULK_DIR.glob(f"*_{basin}.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "pi_mk_p" not in d:
            continue
        if not d.get("stationarity_gate_passed",
                      d["pi_mk_p"] > ALPHA):
            continue
        out.add(d["model"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    args = ap.parse_args()

    admit = _admitted_models(args.basin)

    rows = []
    pooled_pi = []
    pooled_hist = []
    pooled_mid = []
    pooled_far = []
    for p in sorted(EKE_TS_DIR.glob(f"*_{args.basin}_eke_ts.json")):
        d = json.loads(p.read_text())
        if d.get("model") not in admit:
            continue
        pi = [x for x in (d.get("pi_eke") or [])
              if x is not None and np.isfinite(x)]
        if not pi:
            continue
        pi_mean = float(np.mean(pi))
        if not np.isfinite(pi_mean) or pi_mean <= 0:
            continue
        pi_rel = [v / pi_mean for v in pi]
        centres = d.get("hist_centres") or []
        hist_eke = d.get("hist_eke") or []
        hist_rel = {c: (q / pi_mean) for c, q in zip(centres, hist_eke)
                    if q is not None and np.isfinite(q)}
        rows.append(dict(model=d["model"], pi_rel=pi_rel,
                          pi_mean=pi_mean,
                          hist_rel=hist_rel))
        pooled_pi.extend(pi_rel)
        for c, q in hist_rel.items():
            if 1850 <= c <= 1900: pooled_hist.append(q)
            if 2030 <= c <= 2060: pooled_mid.append(q)
            if 2070 <= c <= 2099: pooled_far.append(q)

    if not rows:
        print(f"no admitted models with EKE timeseries for basin={args.basin}")
        return

    pooled_pi = np.asarray(pooled_pi)
    pooled_hist = np.asarray(pooled_hist)
    pooled_mid = np.asarray(pooled_mid)
    pooled_far = np.asarray(pooled_far)
    N = len(rows)
    p975 = float(np.percentile(pooled_pi, 97.5))
    print(f"basin={args.basin}  N_models={N}  "
          f"n_pi_segments_total={pooled_pi.size}")
    print(f"  piControl rel-EKE corridor (pooled):  "
          f"p2.5={np.percentile(pooled_pi,2.5):.3f}  "
          f"median={np.median(pooled_pi):.3f}  "
          f"p97.5={p975:.3f}  max={pooled_pi.max():.3f}")
    for name, arr in [("hist", pooled_hist), ("mid", pooled_mid),
                       ("far", pooled_far)]:
        if arr.size:
            pct = (arr > p975).mean() * 100
            print(f"  {name}: n={arr.size}  median={np.median(arr):.3f}  "
                  f"max={arr.max():.3f}  "
                  f"pct_above_corridor_p97.5={pct:.1f}%")

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.42),
        gridspec_kw=dict(width_ratios=[1.2, 1.0], wspace=0.28),
    )

    rows_sorted = sorted(rows, key=lambda r: float(np.mean(r["pi_rel"])))
    rng = np.random.default_rng(0)
    for i, r in enumerate(rows_sorted):
        x_jit = i + 0.18 * rng.standard_normal(len(r["pi_rel"]))
        axL.scatter(x_jit, r["pi_rel"], s=8, c="0.35",
                      marker="o", alpha=0.7, linewidths=0, zorder=3)
    p10 = float(np.percentile(pooled_pi, 10))
    p90 = float(np.percentile(pooled_pi, 90))
    median = float(np.median(pooled_pi))
    axL.axhspan(p10, p90, color="C0", alpha=0.15, zorder=1,
                  label="piControl 10-90 %")
    axL.axhline(median, color="C0", lw=0.8, zorder=2,
                  label=f"median = {median:.3f}")
    axL.axhline(p975, color="C3", lw=0.6, ls="--", zorder=2,
                  label=f"corridor p97.5 = {p975:.3f}")
    axL.set_xticks(np.arange(N))
    axL.set_xticklabels([r["model"] for r in rows_sorted],
                          rotation=90, fontsize=4)
    axL.set_xlim(-0.5, N - 0.5)
    axL.set_ylabel(r"rel-EKE $= \langle \mathrm{EKE} \rangle / \langle \mathrm{EKE} \rangle_{\mathrm{piC}}$")
    y_top = max(1.6, pooled_far.max() if pooled_far.size else 1.5)
    axL.set_ylim(0.7, y_top)
    axL.legend(loc="upper left", fontsize=5, frameon=False)
    axL.text(0.02, 1.02, "(a)", transform=axL.transAxes,
              ha="left", va="bottom", fontsize=8, fontweight="bold")

    bins = np.linspace(0.7, y_top, 60)
    axR.hist(pooled_pi, bins=bins, color="C0", alpha=0.35,
              edgecolor="white", linewidth=0.4, density=True,
              label=f"piControl  N={pooled_pi.size}")
    xx = np.linspace(0.7, y_top, 400)
    if pooled_pi.size >= 3:
        kde = gaussian_kde(pooled_pi, bw_method=0.25)
        axR.plot(xx, kde(xx), color="C0", lw=1.0)
    axR.axvline(p975, color="C3", lw=0.6, ls="--", zorder=2,
                  label=f"corridor p97.5 = {p975:.3f}")
    if pooled_hist.size:
        med_e = float(np.median(pooled_hist))
        p25, p75 = np.percentile(pooled_hist, [25, 75])
        axR.axvspan(p25, p75, color="0.55", alpha=0.25, zorder=1,
                      label=f"historical 1850-1900  IQR  (med={med_e:.2f})")
        axR.axvline(med_e, color="0.3", lw=0.8, ls="-")
    if pooled_mid.size:
        med_m = float(np.median(pooled_mid))
        p25_m, p75_m = np.percentile(pooled_mid, [25, 75])
        axR.axvspan(p25_m, p75_m, color="C1", alpha=0.20, zorder=1,
                      label=f"ssp585 2030-2060  IQR  (med={med_m:.2f})")
        axR.axvline(med_m, color="C1", lw=0.8, ls="-")
    if pooled_far.size:
        med_f = float(np.median(pooled_far))
        p25_f, p75_f = np.percentile(pooled_far, [25, 75])
        axR.axvspan(p25_f, p75_f, color="C3", alpha=0.20, zorder=1,
                      label=f"ssp585 2070-2099  IQR  (med={med_f:.2f})")
        axR.axvline(med_f, color="C3", lw=0.8, ls="-")
    axR.set_xlabel("rel-EKE")
    axR.set_ylabel("density")
    axR.set_xlim(0.7, y_top)
    axR.legend(loc="upper right", fontsize=5, frameon=False)
    axR.text(0.02, 1.02, "(b)", transform=axR.transAxes,
              ha="left", va="bottom", fontsize=8, fontweight="bold")

    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    out_pdf = MANUSCRIPT_FIGS / f"fig_eke_corridor_{args.basin}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
