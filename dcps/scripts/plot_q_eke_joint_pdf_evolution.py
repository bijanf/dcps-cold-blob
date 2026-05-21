"""Joint $(rel-EKE, Q)$ PDF evolution across epochs.

A five-panel figure that visualises the deformation of the bivariate
$(EKE, Q)$ attractor under anthropogenic forcing:

  (a) piControl joint PDF      (the unforced attractor)
  (b) historical 1850-1900     (early-anthropogenic state)
  (c) SSP5-8.5 2030-2060       (mid-future)
  (d) SSP5-8.5 2070-2099       (far-future)
  (e) log_2( P_late / P_piC )  (displacement: where the attractor has
                                  gained / lost density in phase space)

Each panel is a 2D KDE on a fixed grid with a shared color scale.
The piControl $p_{50}$ and $p_{95}$ density contours are overlaid
on every panel, so the eye can track how each future cloud sits
relative to the piControl attractor.  The last panel is the punchline:
red = future density excess (new climate states), blue = piControl
states that the future ensemble has abandoned.

Input: same paired (Q, rel-EKE) data structure as
plot_q_eke_joint_corridor.py.

Output: manuscript/figs/fig_q_eke_joint_pdf_evolution_<basin>.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
        if c < lo or c > hi or q is None or not np.isfinite(q):
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


def _level_for_quantile(kde, points, q):
    dens = kde(points.T)
    dens_asc = np.sort(dens)
    n = len(dens_asc)
    idx = int(np.floor((1.0 - q) * n))
    idx = max(0, min(idx, n - 1))
    return dens_asc[idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    args = ap.parse_args()

    admit = _admitted_models(args.basin)
    eke = _eke_caches(args.basin)
    common = sorted(set(admit) & set(eke))

    pi_pts, hist_pts, mid_pts, late_pts = [], [], [], []
    for m in common:
        b = admit[m]; e = eke[m]
        pi_pts.extend(_pi_pairs(b, e))
        hist_pts.extend(_epoch_pairs(b, e, *EPOCHS["historical"]))
        mid_pts.extend(_epoch_pairs(b, e, *EPOCHS["mid"]))
        late_pts.extend(_epoch_pairs(b, e, *EPOCHS["late"]))
    pi_pts   = np.asarray(pi_pts)
    hist_pts = np.asarray(hist_pts)
    mid_pts  = np.asarray(mid_pts)
    late_pts = np.asarray(late_pts)
    print(f"basin={args.basin}  pi={len(pi_pts)}  hist={len(hist_pts)}  "
          f"mid={len(mid_pts)}  late={len(late_pts)}")

    all_x = np.concatenate([pi_pts[:,0], hist_pts[:,0],
                              mid_pts[:,0], late_pts[:,0]])
    all_y = np.concatenate([pi_pts[:,1], hist_pts[:,1],
                              mid_pts[:,1], late_pts[:,1]])
    # full data extent + 20 % pad so KDE tails are fully visible
    xlo, xhi = all_x.min(), all_x.max()
    ylo, yhi = all_y.min(), all_y.max()
    xpad = (xhi - xlo) * 0.20; ypad = (yhi - ylo) * 0.20
    xlo -= xpad; xhi += xpad; ylo -= ypad; yhi += ypad

    gx = np.linspace(xlo, xhi, 180)
    gy = np.linspace(ylo, yhi, 180)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.vstack([GX.ravel(), GY.ravel()])

    def _Z(pts, bw=0.30):
        if len(pts) < 5:
            return np.zeros_like(GX), None
        k = gaussian_kde(pts.T, bw_method=bw)
        return k(grid).reshape(GX.shape), k

    Z_pi,  kde_pi  = _Z(pi_pts)
    Z_hi,  _       = _Z(hist_pts)
    Z_mid, _       = _Z(mid_pts)
    Z_la,  _       = _Z(late_pts)

    # Pinned vmax = 30 - aggressive cap that saturates the piControl
    # ridge well below its true peak (~45) so the future-epoch
    # lower-amplitude features get nearly all the grey-scale dynamic
    # range.  Round ticks every 10.
    vmax = 10.0
    levels_lin = np.linspace(0, vmax, 21)
    cb_ticks = [0, 2, 4, 6, 8, 10]

    # piControl 50% and 95% contour levels (for overlay on every panel)
    lvl_p50 = _level_for_quantile(kde_pi, pi_pts, 0.50)
    lvl_p95 = _level_for_quantile(kde_pi, pi_pts, 0.95)
    # historical 95% contour level - second reference state, gives the
    # reader an "early-anthropogenic" envelope to compare future epochs
    # against in addition to piControl.
    kde_hi_obj = gaussian_kde(hist_pts.T, bw_method=0.30) \
        if len(hist_pts) >= 5 else None
    lvl_hi_p95 = (_level_for_quantile(kde_hi_obj, hist_pts, 0.95)
                    if kde_hi_obj is not None else None)

    # log2 ratio of (late+eps) / (piC+eps), masked where piControl is
    # near-zero (no reference state to compare against)
    eps = 1e-3 * vmax
    R = np.log2((Z_la + eps) / (Z_pi + eps))
    R_mask = np.where((Z_la < 1e-3 * vmax) & (Z_pi < 1e-3 * vmax),
                       np.nan, R)
    # Symmetric +/- 10 log2 range (per user request) - shows the
    # extreme tails of the future-vs-piControl density ratio.
    rmax = 10.0

    # ---- figure: row 1 holds (a-d) + shared viridis cbar; row 2 holds
    #              (e) in a SINGLE column (same width as one row-1 panel)
    fig = plt.figure(figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.62),
                       constrained_layout=False)
    gs = GridSpec(2, 5,
                    width_ratios=[1, 1, 1, 1, 0.07],
                    height_ratios=[1.0, 1.05],
                    wspace=0.18, hspace=0.34,
                    left=0.07, right=0.93, bottom=0.10, top=0.95,
                    figure=fig)
    ax_pi  = fig.add_subplot(gs[0, 0])
    ax_hi  = fig.add_subplot(gs[0, 1], sharey=ax_pi, sharex=ax_pi)
    ax_mid = fig.add_subplot(gs[0, 2], sharey=ax_pi, sharex=ax_pi)
    ax_la  = fig.add_subplot(gs[0, 3], sharey=ax_pi, sharex=ax_pi)
    ax_cb  = fig.add_subplot(gs[0, 4])
    # row 2: log-ratio panel = ONE column only (same width as (a)).
    # Intentionally NOT sharex with ax_pi - we want to zoom only (a-d)
    # and keep panel (e) at its full rel-EKE extent.
    ax_rat = fig.add_subplot(gs[1, 0], sharey=ax_pi)

    panels = [
        (ax_pi,  Z_pi,  pi_pts,   "(a) piControl"),
        (ax_hi,  Z_hi,  hist_pts, "(b) historical 1850-1900"),
        (ax_mid, Z_mid, mid_pts,  "(c) SSP5-8.5 2030-2060"),
        (ax_la,  Z_la,  late_pts, "(d) SSP5-8.5 2070-2099"),
    ]

    # Reanalysis anchors: each is one observation-constrained
    # (rel-EKE, Q) coordinate to compare with the CMIP6 clouds.
    # Different marker + colour per source so the reader can tell them
    # apart at a glance.
    # ORAS5 only: its native 0.25 deg resolution is comparable to the
    # CMIP6 HighResMIP subset (0.25-0.5 deg), giving a fair
    # apples-to-apples observational anchor against the CMIP6
    # piControl corridor.  GLORYS12 (1/12 deg eddy-resolving) is
    # discussed in the prose for the resolution-coherence pathway
    # but not overlaid here.
    OBS_SOURCES = [
        ("ORAS5", "*", "#ffd400", 50),   # yellow star
    ]
    obs_points = []
    for src, marker, color, size in OBS_SOURCES:
        p = CACHE_DIR / "eke_timeseries" / f"{src}_atlantic_obs.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
            xy = (float(d["rel_eke"]), float(d["Q"]))
            obs_points.append((src, xy, marker, color, size))
            print(f"  {src} anchor: rel-EKE={xy[0]:.3f}, Q={xy[1]:+.3f}")
        except Exception as ex:
            print(f"  failed to load {src} anchor: {ex}")
    last_im = None
    for ax, Z, pts, title in panels:
        im = ax.contourf(GX, GY, Z, levels=levels_lin, cmap="Greys",
                          extend="max")
        last_im = im
        # piControl reference contours (black on the Greys map)
        ax.contour(GX, GY, Z_pi, levels=[lvl_p95], colors="k",
                    linewidths=0.7, linestyles="--", alpha=0.9)
        ax.contour(GX, GY, Z_pi, levels=[lvl_p50], colors="k",
                    linewidths=0.6, linestyles=":", alpha=0.75)
        # historical reference contour: a saturated orange that
        # contrasts with both white and dark grey
        if lvl_hi_p95 is not None:
            ax.contour(GX, GY, Z_hi, levels=[lvl_hi_p95],
                        colors="#e25822", linewidths=0.8,
                        linestyles="-", alpha=0.95)
        # Reanalysis anchors (same markers on every PDF panel)
        for src, xy, marker, color, size in obs_points:
            ax.scatter(*xy, marker=marker, s=size,
                        facecolor=color, edgecolor="k",
                        linewidth=0.7, zorder=10)
        # zoom rel-EKE axis on the four PDF panels so the data-rich
        # region fills the panel; panel (e) keeps the full extent.
        ax.set_xlim(0.02, 1.5); ax.set_ylim(ylo, yhi)
        ax.set_title(title, fontsize=6, pad=2)
        ax.axhline(0, color="0.85", lw=0.25, zorder=0)
        ax.axvline(1, color="0.85", lw=0.25, zorder=0)
        ax.tick_params(axis="both", labelsize=5)
        # only the bottom row of row-1 panels shows the x label;
        # since row 1 is one row, we still need the x label here
        ax.set_xlabel(r"rel EKE", fontsize=6, labelpad=2)
    ax_pi.set_ylabel(r"Quiescence Index $Q$", fontsize=6, labelpad=2)
    for ax in (ax_hi, ax_mid, ax_la):
        ax.tick_params(axis="y", labelleft=False)
        ax.set_ylabel("")

    cb = fig.colorbar(last_im, cax=ax_cb, ticks=cb_ticks)
    cb.set_label("joint density", fontsize=5, labelpad=2)
    cb.ax.tick_params(labelsize=5)

    # (e) log-ratio panel
    norm = TwoSlopeNorm(vmin=-rmax, vcenter=0.0, vmax=rmax)
    imR = ax_rat.imshow(R_mask, origin="lower", aspect="auto",
                          extent=(xlo, xhi, ylo, yhi),
                          cmap="RdBu_r", norm=norm, interpolation="bilinear")
    ax_rat.contour(GX, GY, Z_pi, levels=[lvl_p95], colors="k",
                    linewidths=0.6, linestyles="--", alpha=0.9)
    ax_rat.contour(GX, GY, Z_pi, levels=[lvl_p50], colors="k",
                    linewidths=0.5, linestyles=":", alpha=0.6)
    if lvl_hi_p95 is not None:
        ax_rat.contour(GX, GY, Z_hi, levels=[lvl_hi_p95],
                        colors="#e25822", linewidths=0.8,
                        linestyles="-", alpha=0.95)
    for src, xy, marker, color, size in obs_points:
        ax_rat.scatter(*xy, marker=marker, s=size + 10,
                        facecolor=color, edgecolor="k",
                        linewidth=0.6, zorder=10)
    # contour-style key placed outside the (e) panel - figure-level
    # legend in the unused empty area below row 1 cols 1-3
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0],[0], color="k",       lw=0.8, ls="--",
               label=r"piControl $p_{95}$"),
        Line2D([0],[0], color="k",       lw=0.7, ls=":",
               label=r"piControl $p_{50}$"),
        Line2D([0],[0], color="#e25822", lw=0.8, ls="-",
               label=r"historical $p_{95}$"),
        Line2D([0],[0], marker="*", color="none",
               markerfacecolor="#ffd400", markeredgecolor="k",
               markeredgewidth=0.6, markersize=7,
               label="ORAS5 reanalysis 1995-2024"),
    ]
    fig.legend(handles=legend_handles, loc="center",
                  bbox_to_anchor=(0.65, 0.27), fontsize=6,
                  frameon=False, handlelength=2.0,
                  labelspacing=0.5, title="reference contours",
                  title_fontsize=6)
    ax_rat.axhline(0, color="0.4", lw=0.25, zorder=0)
    ax_rat.axvline(1, color="0.4", lw=0.25, zorder=0)
    ax_rat.set_xlim(xlo, xhi); ax_rat.set_ylim(ylo, yhi)
    ax_rat.set_title(r"(e) $\log_2(P_{\mathrm{late}}/P_{\mathrm{piC}})$",
                       fontsize=6, pad=2)
    ax_rat.set_xlabel(r"rel EKE", fontsize=6, labelpad=2)
    ax_rat.set_ylabel(r"Quiescence Index $Q$", fontsize=6, labelpad=2)
    ax_rat.tick_params(axis="both", labelsize=5)
    # colorbar for (e) attached to its right edge via axes-divider
    div = make_axes_locatable(ax_rat)
    ax_cbR = div.append_axes("right", size="6%", pad=0.05)
    cbR = fig.colorbar(imR, cax=ax_cbR,
                          ticks=[-10, -5, 0, 5, 10])
    cbR.set_label(r"$\log_2$ ratio", fontsize=5, labelpad=2)
    cbR.ax.tick_params(labelsize=5)

    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    out_pdf = MANUSCRIPT_FIGS / f"fig_q_eke_joint_pdf_evolution_{args.basin}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
