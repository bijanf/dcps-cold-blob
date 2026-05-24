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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
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


def _admitted_models_245(basin):
    """SSP2-4.5 Q bulk caches; piControl gate inherited from SSP585 cache."""
    out = {}
    for p in sorted(BULK_DIR.glob(f"*_{basin}_ssp245.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
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
        # restrict to default ssp585 caches: drop the variants
        name = p.stem
        if any(t in name for t in ("ssp245", "ssp370", "ssp126",
                                    "1pctCO2", "abrupt-4xCO2")):
            continue
        out[d["model"]] = d
    return out


def _eke_caches_245(basin):
    out = {}
    for p in sorted(EKE_TS_DIR.glob(f"*_{basin}_ssp245_eke_ts.json")):
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

    # SSP2-4.5 sibling caches (use the same stationarity-passed subset)
    admit245 = _admitted_models_245(args.basin)
    eke245 = _eke_caches_245(args.basin)
    common245 = sorted(set(admit) & set(admit245) & set(eke245))

    pi_pts, hist_pts, mid_pts, late_pts = [], [], [], []
    for m in common:
        b = admit[m]; e = eke[m]
        pi_pts.extend(_pi_pairs(b, e))
        hist_pts.extend(_epoch_pairs(b, e, *EPOCHS["historical"]))
        mid_pts.extend(_epoch_pairs(b, e, *EPOCHS["mid"]))
        late_pts.extend(_epoch_pairs(b, e, *EPOCHS["late"]))
    mid_pts_245, late_pts_245 = [], []
    for m in common245:
        mid_pts_245.extend(_epoch_pairs(admit245[m], eke245[m],
                                          *EPOCHS["mid"]))
        late_pts_245.extend(_epoch_pairs(admit245[m], eke245[m],
                                           *EPOCHS["late"]))
    pi_pts   = np.asarray(pi_pts)
    hist_pts = np.asarray(hist_pts)
    mid_pts  = np.asarray(mid_pts)
    late_pts = np.asarray(late_pts)
    mid_pts_245 = np.asarray(mid_pts_245)
    late_pts_245 = np.asarray(late_pts_245)
    print(f"basin={args.basin}  pi={len(pi_pts)}  hist={len(hist_pts)}  "
          f"mid585={len(mid_pts)}  late585={len(late_pts)}  "
          f"mid245={len(mid_pts_245)}  late245={len(late_pts_245)}  "
          f"(n_models 585={len(common)}, 245={len(common245)})")

    all_x = np.concatenate([pi_pts[:,0], hist_pts[:,0],
                              mid_pts[:,0], late_pts[:,0],
                              mid_pts_245[:,0], late_pts_245[:,0]])
    all_y = np.concatenate([pi_pts[:,1], hist_pts[:,1],
                              mid_pts[:,1], late_pts[:,1],
                              mid_pts_245[:,1], late_pts_245[:,1]])
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

    Z_pi,    kde_pi = _Z(pi_pts)
    Z_hi,    _      = _Z(hist_pts)
    Z_mid,   _      = _Z(mid_pts)
    Z_la,    _      = _Z(late_pts)
    Z_mid245, _     = _Z(mid_pts_245)
    Z_la245,  _     = _Z(late_pts_245)

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
    R585 = np.log2((Z_la + eps) / (Z_pi + eps))
    R585_mask = np.where((Z_la < 1e-3 * vmax) & (Z_pi < 1e-3 * vmax),
                          np.nan, R585)
    R245 = np.log2((Z_la245 + eps) / (Z_pi + eps))
    R245_mask = np.where((Z_la245 < 1e-3 * vmax) & (Z_pi < 1e-3 * vmax),
                          np.nan, R245)
    # Symmetric +/- 10 log2 range (per user request) - shows the
    # extreme tails of the future-vs-piControl density ratio.
    rmax = 10.0

    # ---- figure: 2 rows x 4 cols of panels + colorbar column
    # Row 0: SSP5-8.5 chain  (a) piC | (b) hist | (c) 585 mid | (d) 585 late | density cbar
    # Row 1: SSP2-4.5 chain + log-ratios
    #        (e) 245 mid | (f) 245 late | (g) log2 585 | (h) log2 245 | ratio cbar
    fig = plt.figure(figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.60),
                       constrained_layout=False)
    gs = GridSpec(2, 5,
                    width_ratios=[1, 1, 1, 1, 0.07],
                    height_ratios=[1.0, 1.0],
                    wspace=0.22, hspace=0.34,
                    left=0.06, right=0.95, bottom=0.14, top=0.95,
                    figure=fig)
    ax_pi    = fig.add_subplot(gs[0, 0])
    ax_hi    = fig.add_subplot(gs[0, 1], sharey=ax_pi, sharex=ax_pi)
    ax_mid   = fig.add_subplot(gs[0, 2], sharey=ax_pi, sharex=ax_pi)
    ax_la    = fig.add_subplot(gs[0, 3], sharey=ax_pi, sharex=ax_pi)
    ax_cb    = fig.add_subplot(gs[0, 4])
    ax_mid245 = fig.add_subplot(gs[1, 0], sharey=ax_pi, sharex=ax_pi)
    ax_la245  = fig.add_subplot(gs[1, 1], sharey=ax_pi, sharex=ax_pi)
    ax_rat585 = fig.add_subplot(gs[1, 2], sharey=ax_pi)
    ax_rat245 = fig.add_subplot(gs[1, 3], sharey=ax_pi, sharex=ax_rat585)
    ax_cbR    = fig.add_subplot(gs[1, 4])

    panels = [
        (ax_pi,     Z_pi,      pi_pts,        "(a) piControl"),
        (ax_hi,     Z_hi,      hist_pts,      "(b) historical 1850-1900"),
        (ax_mid,    Z_mid,     mid_pts,       "(c) SSP5-8.5 2030-2060"),
        (ax_la,     Z_la,      late_pts,      "(d) SSP5-8.5 2070-2099"),
        (ax_mid245, Z_mid245,  mid_pts_245,   "(e) SSP2-4.5 2030-2060"),
        (ax_la245,  Z_la245,   late_pts_245,  "(f) SSP2-4.5 2070-2099"),
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
    ax_mid245.set_ylabel(r"Quiescence Index $Q$", fontsize=6, labelpad=2)
    for ax in (ax_hi, ax_mid, ax_la, ax_la245, ax_rat585, ax_rat245):
        ax.tick_params(axis="y", labelleft=False)
        ax.set_ylabel("")

    cb = fig.colorbar(last_im, cax=ax_cb, ticks=cb_ticks)
    cb.set_label("joint density", fontsize=5, labelpad=2)
    cb.ax.tick_params(labelsize=5)

    # ---- log-ratio panels (g, h) -------------------------------------
    norm = TwoSlopeNorm(vmin=-rmax, vcenter=0.0, vmax=rmax)
    ratio_panels = [
        (ax_rat585, R585_mask,
         r"(g) $\log_2(P_{\mathrm{late\,SSP5\text{-}8.5}}/P_{\mathrm{piC}})$"),
        (ax_rat245, R245_mask,
         r"(h) $\log_2(P_{\mathrm{late\,SSP2\text{-}4.5}}/P_{\mathrm{piC}})$"),
    ]
    last_imR = None
    for ax, R_mask, title in ratio_panels:
        imR = ax.imshow(R_mask, origin="lower", aspect="auto",
                          extent=(xlo, xhi, ylo, yhi),
                          cmap="RdBu_r", norm=norm, interpolation="bilinear")
        last_imR = imR
        ax.contour(GX, GY, Z_pi, levels=[lvl_p95], colors="k",
                    linewidths=0.6, linestyles="--", alpha=0.9)
        ax.contour(GX, GY, Z_pi, levels=[lvl_p50], colors="k",
                    linewidths=0.5, linestyles=":", alpha=0.6)
        if lvl_hi_p95 is not None:
            ax.contour(GX, GY, Z_hi, levels=[lvl_hi_p95],
                        colors="#e25822", linewidths=0.8,
                        linestyles="-", alpha=0.95)
        for src, xy, marker, color, size in obs_points:
            ax.scatter(*xy, marker=marker, s=size + 10,
                        facecolor=color, edgecolor="k",
                        linewidth=0.6, zorder=10)
        ax.axhline(0, color="0.4", lw=0.25, zorder=0)
        ax.axvline(1, color="0.4", lw=0.25, zorder=0)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_title(title, fontsize=6, pad=2)
        ax.set_xlabel(r"rel EKE", fontsize=6, labelpad=2)
        ax.tick_params(axis="both", labelsize=5)

    cbR = fig.colorbar(last_imR, cax=ax_cbR, ticks=[-10, -5, 0, 5, 10])
    cbR.set_label(r"$\log_2$ ratio", fontsize=5, labelpad=2)
    cbR.ax.tick_params(labelsize=5)

    # Contour-style key: single figure-level legend just below the
    # density colorbar.
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
    fig.legend(handles=legend_handles, loc="upper center",
                  bbox_to_anchor=(0.5, 0.04), fontsize=6, ncol=4,
                  frameon=False, handlelength=2.0,
                  labelspacing=0.5, columnspacing=1.2)

    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    out_pdf = MANUSCRIPT_FIGS / f"fig_q_eke_joint_pdf_evolution_{args.basin}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
