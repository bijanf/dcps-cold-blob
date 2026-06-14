"""Joint (rel-EKE, Q) corridor of the Holocene-equivalent piControl
state space, with epoch overlays.

Replaces the two parallel univariate corridor figures (Q corridor and
EKE corridor) by a single corner-plot: a central joint scatter with
piControl bivariate-density contours, plus matching marginals on top
and on the right.  Each model-window contributes one paired point in
(rel-EKE, Q); pre-industrial piControl segments define the corridor,
historical and SSP5-8.5 epochs are overlaid in colour, and the multi-
model-mean trajectory through phase space is drawn as a sequence of
arrows.

This makes two scientific statements that the parallel layout obscured:
  - Q and rel-EKE are not independent indicators — they evolve along a
    well-defined direction in phase space whose shape is set by the
    Quiescence-Signature mechanism.
  - A joint p95 density-contour exit can be deeper than either marginal
    exit, because future points may sit inside both marginal corridors
    while landing in low-density joint regions.

Output: manuscript/figs/fig_q_eke_joint_corridor_<basin>.pdf
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import gaussian_kde

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style, DOUBLE_COL_IN
apply_nature_style()

BULK_DIR = CACHE_DIR / "holocene_exit" / "bulk"
EKE_TS_DIR = CACHE_DIR / "eke_timeseries"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

ALPHA = 0.05
EPOCHS = {
    "historical": (1850, 1900),
    "mid":        (2030, 2060),
    "late":       (2070, 2099),
}
COLOURS = {
    "pi":         "#1f77b4",
    "historical": "0.45",
    "mid":        "#ff7f0e",
    "late":       "#d62728",
}


def _admitted_models(basin):
    out = {}
    for p in sorted(BULK_DIR.glob(f"*_{basin}.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "pi_mk_p" not in d:
            continue
        if not d.get("stationarity_gate_passed", d["pi_mk_p"] > ALPHA):
            continue
        out[d["model"]] = d
    return out


def _eke_caches(basin):
    out = {}
    for p in sorted(EKE_TS_DIR.glob(f"*_{basin}_eke_ts.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        out[d["model"]] = d
    return out


def _epoch_pairs(bulk, eke, lo, hi):
    bc = bulk.get("hist_centres") or []
    bq = bulk.get("hist_Q") or []
    ec = eke.get("hist_centres") or []
    ev = eke.get("hist_eke") or []
    mu = eke.get("pi_eke_mean") or float("nan")
    if not (np.isfinite(mu) and mu > 0):
        return []
    eke_by_c = {c: v for c, v in zip(ec, ev)}
    pairs = []
    for c, q in zip(bc, bq):
        if c < lo or c > hi:
            continue
        if q is None or not np.isfinite(q):
            continue
        v = eke_by_c.get(c)
        if v is None or not np.isfinite(v):
            continue
        pairs.append((v / mu, q))
    return pairs


def _pi_pairs(bulk, eke):
    bs = bulk.get("pi_starts") or []
    bq = bulk.get("pi_Q") or []
    es = eke.get("pi_starts") or []
    ev = eke.get("pi_eke") or []
    mu = eke.get("pi_eke_mean") or float("nan")
    if not (np.isfinite(mu) and mu > 0):
        return []
    eke_by_s = {s: v for s, v in zip(es, ev)}
    pairs = []
    for s, q in zip(bs, bq):
        if q is None or not np.isfinite(q):
            continue
        v = eke_by_s.get(s)
        if v is None or not np.isfinite(v):
            continue
        pairs.append((v / mu, q))
    return pairs


def _kde_contour_levels(kde, points, quantiles=(0.50, 0.95)):
    """Return density levels enclosing the requested quantiles of
    POINTS (count-based, not density-weighted).  A level L for
    quantile q satisfies: fraction of input points with
    kde(p) >= L is approximately q."""
    dens = kde(points.T)
    dens_asc = np.sort(dens)
    n = len(dens_asc)
    levels = []
    for q in quantiles:
        idx = int(np.floor((1.0 - q) * n))
        idx = max(0, min(idx, n - 1))
        levels.append(dens_asc[idx])
    return sorted(levels)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    args = ap.parse_args()

    admit = _admitted_models(args.basin)
    eke = _eke_caches(args.basin)
    common = sorted(set(admit) & set(eke))
    print(f"basin={args.basin}  models with paired Q+EKE caches = {len(common)}")

    pi_pts, hist_pts, mid_pts, late_pts = [], [], [], []
    for m in common:
        b = admit[m]
        e = eke[m]
        pi_pts.extend(_pi_pairs(b, e))
        hist_pts.extend(_epoch_pairs(b, e, *EPOCHS["historical"]))
        mid_pts.extend(_epoch_pairs(b, e, *EPOCHS["mid"]))
        late_pts.extend(_epoch_pairs(b, e, *EPOCHS["late"]))
    pi_pts   = np.asarray(pi_pts)
    hist_pts = np.asarray(hist_pts)
    mid_pts  = np.asarray(mid_pts)
    late_pts = np.asarray(late_pts)
    print(f"  piControl joint pairs:  N={len(pi_pts)}")
    print(f"  historical pairs:       N={len(hist_pts)}")
    print(f"  mid-future pairs:       N={len(mid_pts)}")
    print(f"  late-future pairs:      N={len(late_pts)}")

    # ---- joint piControl density + 95% contour exit fraction ------
    kde = gaussian_kde(pi_pts.T, bw_method=0.30)
    levels = _kde_contour_levels(kde, pi_pts, quantiles=(0.50, 0.95))
    lvl95 = levels[0]  # lowest level = the 95-th-percentile contour
    def _exit_fraction(pts):
        if len(pts) == 0:
            return 0.0
        return float((kde(pts.T) < lvl95).mean())
    f_pi   = _exit_fraction(pi_pts)
    f_hist = _exit_fraction(hist_pts)
    f_mid  = _exit_fraction(mid_pts)
    f_late = _exit_fraction(late_pts)
    print("  joint p95-contour exit fraction:")
    print(f"    piControl   : {100*f_pi  :5.1f}%  (target ~5%)")
    print(f"    historical  : {100*f_hist:5.1f}%")
    print(f"    mid-future  : {100*f_mid :5.1f}%")
    print(f"    late-future : {100*f_late:5.1f}%")

    # ---- figure layout ------------------------------------------------
    fig = plt.figure(figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.7),
                       constrained_layout=False)
    gs = GridSpec(
        2, 2,
        width_ratios=[4.0, 1.4],
        height_ratios=[1.2, 4.0],
        wspace=0.06, hspace=0.06,
        left=0.10, right=0.985, bottom=0.11, top=0.97,
        figure=fig,
    )
    ax_joint = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_joint)
    ax_legend = fig.add_subplot(gs[0, 1]); ax_legend.axis("off")

    # axis ranges - full data extent + 10 % pad so KDE tails are fully visible
    all_x = np.concatenate([pi_pts[:,0], hist_pts[:,0] if len(hist_pts) else [],
                              mid_pts[:,0] if len(mid_pts) else [],
                              late_pts[:,0] if len(late_pts) else []])
    all_y = np.concatenate([pi_pts[:,1], hist_pts[:,1] if len(hist_pts) else [],
                              mid_pts[:,1] if len(mid_pts) else [],
                              late_pts[:,1] if len(late_pts) else []])
    xlo, xhi = all_x.min(), all_x.max()
    ylo, yhi = all_y.min(), all_y.max()
    xpad = (xhi - xlo) * 0.10
    ypad = (yhi - ylo) * 0.10
    xlo -= xpad; xhi += xpad; ylo -= ypad; yhi += ypad

    # joint contours
    gx, gy = np.linspace(xlo, xhi, 200), np.linspace(ylo, yhi, 200)
    GX, GY = np.meshgrid(gx, gy)
    Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    ax_joint.contourf(GX, GY, Z, levels=[lvl95, levels[1], Z.max()],
                       colors=["#9ecae1", "#3182bd"], alpha=0.45, zorder=1)
    ax_joint.contour(GX, GY, Z, levels=[lvl95], colors=["#08519c"],
                       linewidths=0.7, zorder=2)

    # uniform marker size across all four epochs - shape + colour
    # alone distinguish them
    S = 12
    ax_joint.scatter(pi_pts[:,0], pi_pts[:,1], s=S, c=COLOURS["pi"],
                       marker="o", alpha=0.55, linewidths=0, zorder=3,
                       label=f"piControl  ({len(pi_pts)})")
    if len(hist_pts):
        ax_joint.scatter(hist_pts[:,0], hist_pts[:,1], s=S,
                           c=COLOURS["historical"], marker="D",
                           edgecolor="white", linewidth=0.25,
                           alpha=0.8, zorder=4,
                           label=f"hist 1850-1900  ({len(hist_pts)})")
    if len(mid_pts):
        ax_joint.scatter(mid_pts[:,0], mid_pts[:,1], s=S,
                           c=COLOURS["mid"], marker="s",
                           edgecolor="white", linewidth=0.25,
                           alpha=0.85, zorder=5,
                           label=f"SSP585 2030-2060  ({len(mid_pts)})")
    if len(late_pts):
        ax_joint.scatter(late_pts[:,0], late_pts[:,1], s=S,
                           c=COLOURS["late"], marker="^",
                           edgecolor="white", linewidth=0.25,
                           alpha=0.9, zorder=6,
                           label=f"SSP585 2070-2099  ({len(late_pts)})")

    # (epoch-median markers removed - the data clouds and the
    # piControl p95 contour already convey the per-epoch position.)

    ax_joint.axhline(0, color="0.7", lw=0.3, zorder=0)
    ax_joint.axvline(1, color="0.7", lw=0.3, zorder=0)
    ax_joint.set_xlim(xlo, xhi); ax_joint.set_ylim(ylo, yhi)
    ax_joint.set_xlabel(r"rel EKE  $=\langle|\nabla SSH|^2\rangle / \mu_{\mathrm{piC}}$",
                          labelpad=2)
    ax_joint.set_ylabel(r"Quiescence Index $Q$", labelpad=2)

    # ---- top marginal: rel-EKE ----------------------------------------
    xx = np.linspace(xlo, xhi, 400)
    for key, pts in [("pi", pi_pts), ("historical", hist_pts),
                      ("mid", mid_pts), ("late", late_pts)]:
        if len(pts) < 3: continue
        k = gaussian_kde(pts[:,0], bw_method=0.30)
        ax_top.plot(xx, k(xx), color=COLOURS[key], lw=1.0)
        ax_top.fill_between(xx, 0, k(xx), color=COLOURS[key], alpha=0.18)
    ax_top.axvline(1, color="0.7", lw=0.3)
    ax_top.set_ylabel("density", fontsize=6)
    ax_top.tick_params(axis="x", labelbottom=False)

    # ---- right marginal: Q -------------------------------------------
    yy = np.linspace(ylo, yhi, 400)
    for key, pts in [("pi", pi_pts), ("historical", hist_pts),
                      ("mid", mid_pts), ("late", late_pts)]:
        if len(pts) < 3: continue
        k = gaussian_kde(pts[:,1], bw_method=0.30)
        ax_right.plot(k(yy), yy, color=COLOURS[key], lw=1.0)
        ax_right.fill_betweenx(yy, 0, k(yy), color=COLOURS[key], alpha=0.18)
    ax_right.axhline(0, color="0.7", lw=0.3)
    ax_right.set_xlabel("density", fontsize=6)
    ax_right.tick_params(axis="y", labelleft=False)

    # legend in corner panel - bigger, distinct markers
    handles = [
        plt.Line2D([0],[0], marker=".", color="none",
                    markerfacecolor=COLOURS["pi"],
                    markeredgecolor=COLOURS["pi"], markersize=7,
                    label=f"piControl  ({len(pi_pts)})"),
        plt.Line2D([0],[0], marker="D", color="none",
                    markerfacecolor=COLOURS["historical"],
                    markeredgecolor="0.2", markeredgewidth=0.3,
                    markersize=5,
                    label=f"hist 1850-1900  ({len(hist_pts)})"),
        plt.Line2D([0],[0], marker="s", color="none",
                    markerfacecolor=COLOURS["mid"],
                    markeredgecolor="0.2", markeredgewidth=0.3,
                    markersize=5,
                    label=f"SSP585 2030-2060  ({len(mid_pts)})"),
        plt.Line2D([0],[0], marker="^", color="none",
                    markerfacecolor=COLOURS["late"],
                    markeredgecolor="0.2", markeredgewidth=0.3,
                    markersize=6,
                    label=f"SSP585 2070-2099  ({len(late_pts)})"),
        plt.Line2D([0],[0], color="#08519c", lw=0.8,
                    label="piControl $p_{95}$ contour"),
    ]
    ax_legend.legend(handles=handles, loc="center", fontsize=5,
                       frameon=False, handlelength=1.2,
                       borderpad=0.2, labelspacing=0.55)

    # (joint exit-fraction inset removed per user request - numbers
    # are reported in the manuscript prose, not in the figure.)

    # panel letters
    ax_top.text(-0.02, 1.05, "(a)", transform=ax_top.transAxes,
                  ha="right", va="bottom", fontsize=8, fontweight="bold")
    ax_right.text(1.02, 1.05, "(c)", transform=ax_right.transAxes,
                    ha="left", va="bottom", fontsize=8, fontweight="bold")
    ax_joint.text(-0.10, 1.00, "(b)", transform=ax_joint.transAxes,
                    ha="right", va="bottom", fontsize=8, fontweight="bold")

    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    out_pdf = MANUSCRIPT_FIGS / f"fig_q_eke_joint_corridor_{args.basin}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
