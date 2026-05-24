"""Multi-model piControl 'Holocene-equivalent' Q corridor.

For every gate-passing bulk JSON in dcps/cache/holocene_exit/bulk/,
collect the 8 piControl Q segments and aggregate them into a
multi-model corridor: the union of internal-variability Q under the
Holocene-equivalent CMIP6 piControl forcing.

Two panels:
  (a) per-model piControl Q segments scattered across a notional
      Holocene-equivalent time axis (one column per model, dots = the
      8 segments); ensemble 10-90 % envelope as a grey band; per-model
      95-th-percentile thresholds as red ticks on a right margin.
  (b) pooled histogram + KDE of every piControl Q value across all
      admitted models, with the spread of (i) historical 1850-1900 Q
      and (ii) far-future 2070-2099 Q overlaid as vertical strips.
      This is the formal Holocene-equivalent corridor against which
      the exit claim is tested.

Output: manuscript/figs/fig_picontrol_corridor_<basin>.pdf
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
PAST1000_DIR = CACHE_DIR / "cmip5_past1000"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"
ALPHA = 0.05


def _early_far_Q(d, early=(1850, 1900), far=(2070, 2099)):
    """Pick hist_Q values whose window centres fall in the given
    epoch.  Returns (early_list, far_list)."""
    centres = d.get("hist_centres") or []
    Qs = d.get("hist_Q") or []
    e_list, f_list = [], []
    for c, q in zip(centres, Qs):
        if early[0] <= c <= early[1]: e_list.append(q)
        if far[0]   <= c <= far[1]:   f_list.append(q)
    return e_list, f_list


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    args = ap.parse_args()

    rows = []
    pooled_pi = []
    pooled_early = []
    pooled_far = []
    for p in sorted(BULK_DIR.glob(f"*_{args.basin}.json")):
        d = json.loads(p.read_text())
        if "pi_mk_p" not in d: continue
        if not d.get("stationarity_gate_passed",
                       d["pi_mk_p"] > ALPHA): continue
        pi = d.get("pi_Q") or []
        rows.append(dict(model=d["model"], pi_Q=pi,
                          pi_p95=float(d["pi_p95_threshold"]),
                          pi_mean=float(d.get("pi_mean", float("nan"))),
                          pi_sd=float(d.get("pi_sd", float("nan")))))
        pooled_pi.extend([q for q in pi if q is not None
                           and np.isfinite(q)])
        e, f = _early_far_Q(d)
        pooled_early.extend([q for q in e if np.isfinite(q)])
        pooled_far.extend([q for q in f if np.isfinite(q)])
    if not rows:
        print(f"no admitted models for basin={args.basin}"); return

    # Holocene-equivalent past1000 forced trajectories (CMIP5/PMIP3).
    # Restrict to windows centred at or before 1849 CE, the official
    # end of the past1000 experiment.  Some institutes ship past1000
    # files merged with historical (e.g. bcc-csm1-1 covers 850-2005);
    # those post-1850 windows belong to the historical strip, not the
    # Holocene-equivalent reference.
    pooled_p1k = []
    p1k_models = []
    P1K_END_YEAR = 1849
    for p in sorted(PAST1000_DIR.glob(f"*_{args.basin}.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        qs = d.get("Q") or []
        cs = d.get("year_centres") or []
        kept = []
        for q, c in zip(qs, cs):
            if q is None or not np.isfinite(q):
                continue
            if c is None or c > P1K_END_YEAR:
                continue
            kept.append(q)
        if not kept:
            continue
        pooled_p1k.extend(kept)
        p1k_models.append(d.get("model", p.stem))

    pooled_pi = np.asarray(pooled_pi)
    pooled_early = np.asarray(pooled_early)
    pooled_far = np.asarray(pooled_far)
    pooled_p1k = np.asarray(pooled_p1k)
    N = len(rows)
    Np1k = len(p1k_models)
    print(f"basin={args.basin}  N_models={N}  "
          f"n_pi_segments_total={pooled_pi.size}")
    if Np1k:
        print(f"  past1000 forced trajectories: N_models={Np1k}  "
              f"n_windows={pooled_p1k.size}  "
              f"median={np.median(pooled_p1k):+.3f}  "
              f"models=[{', '.join(p1k_models)}]")
    print(f"  piControl Q corridor (pooled):  "
          f"p2.5={np.percentile(pooled_pi,2.5):+.3f}  "
          f"median={np.median(pooled_pi):+.3f}  "
          f"p97.5={np.percentile(pooled_pi,97.5):+.3f}")
    print(f"  early Q (1850-1900):  n={pooled_early.size}  "
          f"median={np.median(pooled_early):+.3f}")
    print(f"  far   Q (2070-2099):  n={pooled_far.size}  "
          f"median={np.median(pooled_far):+.3f}"
          if pooled_far.size else "  far Q: not yet available")

    # ---- figure ----
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.42),
        gridspec_kw=dict(width_ratios=[1.2, 1.0], wspace=0.28),
    )

    # (a) per-model piControl segments scattered across model columns
    rows_sorted = sorted(rows, key=lambda r: r["pi_mean"])
    rng = np.random.default_rng(0)
    for i, r in enumerate(rows_sorted):
        x_jit = i + 0.18 * rng.standard_normal(len(r["pi_Q"]))
        axL.scatter(x_jit, r["pi_Q"], s=8, c="0.35",
                      marker="o", alpha=0.7, linewidths=0, zorder=3)
        # threshold tick
        axL.scatter([i + 0.35], [r["pi_p95"]], s=10, c="C3",
                      marker="_", linewidth=1.2, zorder=4)
    # 10-90 pctile envelope of piControl segments across models
    p10 = np.percentile(pooled_pi, 10)
    p90 = np.percentile(pooled_pi, 90)
    median = np.median(pooled_pi)
    axL.axhspan(p10, p90, color="C0", alpha=0.15, zorder=1,
                  label="piControl 10-90 %")
    axL.axhline(median, color="C0", lw=0.8, zorder=2,
                  label=f"median = {median:+.3f}")
    axL.set_xticks(np.arange(N))
    axL.set_xticklabels([r["model"] for r in rows_sorted],
                          rotation=90, fontsize=4)
    axL.set_xlim(-0.5, N - 0.5)
    axL.set_ylabel(r"$Q = -\rho$  on 30-yr piControl segments")
    axL.set_ylim(-0.5, 0.65)
    axL.legend(loc="upper left", fontsize=5, frameon=False)
    axL.text(0.02, 1.02, "(a)", transform=axL.transAxes,
              ha="left", va="bottom", fontsize=8, fontweight="bold")

    # (b) pooled distributions
    bins = np.linspace(-0.5, 0.65, 60)
    axR.hist(pooled_pi, bins=bins, color="C0", alpha=0.35,
              edgecolor="white", linewidth=0.4, density=True,
              label=f"piControl  N={pooled_pi.size}")
    xx = np.linspace(-0.5, 0.65, 400)
    if pooled_pi.size >= 3:
        kde = gaussian_kde(pooled_pi, bw_method=0.25)
        axR.plot(xx, kde(xx), color="C0", lw=1.0)
    if pooled_p1k.size:
        med_p = np.median(pooled_p1k)
        p25_p, p75_p = np.percentile(pooled_p1k, [25, 75])
        axR.axvspan(p25_p, p75_p, color="C2", alpha=0.20, zorder=1,
                      label=f"past1000 850-1849  IQR  "
                            f"(med={med_p:+.2f}, N={Np1k})")
        axR.axvline(med_p, color="C2", lw=0.8, ls="-")
        if pooled_p1k.size >= 3:
            kde_p = gaussian_kde(pooled_p1k, bw_method=0.25)
            axR.plot(xx, kde_p(xx), color="C2", lw=1.0)
    if pooled_early.size:
        med_e = np.median(pooled_early)
        p25_e, p75_e = np.percentile(pooled_early, [25, 75])
        axR.axvspan(p25_e, p75_e, color="0.55", alpha=0.25, zorder=1,
                      label=f"historical 1850-1900  IQR  (med={med_e:+.2f})")
        axR.axvline(med_e, color="0.3", lw=0.8, ls="-")
    if pooled_far.size:
        med_f = np.median(pooled_far)
        p25_f, p75_f = np.percentile(pooled_far, [25, 75])
        axR.axvspan(p25_f, p75_f, color="C3", alpha=0.2, zorder=1,
                      label=f"ssp585 2070-2099  IQR  (med={med_f:+.2f})")
        axR.axvline(med_f, color="C3", lw=0.8, ls="-")
    axR.set_xlabel(r"$Q$")
    axR.set_ylabel("density")
    axR.set_xlim(-0.5, 0.65)
    axR.legend(loc="upper left", fontsize=5, frameon=False)
    axR.text(0.02, 1.02, "(b)", transform=axR.transAxes,
              ha="left", va="bottom", fontsize=8, fontweight="bold")

    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    out_pdf = MANUSCRIPT_FIGS / f"fig_picontrol_corridor_{args.basin}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
