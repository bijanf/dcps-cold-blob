"""Plot R3 (reviewer-driven): Quiescence Index Q across basins,
with both EKE definitions side-by-side and 95% CI error bars.

Three basin groups (North Atlantic, North Pacific, Southern Ocean
[Pacific-sector ACC]), each with two bars:
  - |grad SSH| proxy           (multi-basin pipeline; ORAS5 SSH)
  - geostrophic-velocity EKE   (eddy-resolving; GLORYS12 zos)

Vertical 95% CI error bars from spatial_block_bootstrap (block_km=1500,
B=1000).  Horizontal dashed line at Q = 0.2 (reviewer's "robust
signature" threshold).  Distinct colour + hatch per EKE definition so
the figure reads in greyscale (project rule feedback_plot_shapes.md).
No value labels above the bars -- Q numbers go in the caption.

If a basin's eddy-resolving column-2 cache is not yet on disk, that
bar is rendered as an outlined empty bar with an "pending" note in the
caption (the eddy-resolving pipeline for that basin had not completed
when this figure was made).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style, SINGLE_COL_IN
apply_nature_style()


MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"
PROXY_JSON = CACHE_DIR / "multi_basin" / "q_ci_bootstrap.json"
EDDY_JSON = CACHE_DIR / "eke_eddy_resolving" / "q_ci_bootstrap_eddy.json"

BASINS_ORDER = (
    ("atlantic", "N. Atlantic"),
    ("pacific",  "N. Pacific"),
    ("southern", "S. Ocean (ACC)"),
)
BASIN_THRESHOLD = 0.2

COL_PROXY = "#1f77b4"  # blue
COL_EDDY  = "#ff7f0e"  # orange


def _read(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _Q_and_ci(entry):
    """Convert {rho_observed, ci_low, ci_high} to (Q, lo, hi) where Q=-rho."""
    if not entry:
        return None
    rho = entry.get("rho_observed")
    lo = entry.get("ci_low"); hi = entry.get("ci_high")
    if rho is None or lo is None or hi is None:
        return None
    Q = -float(rho)
    # CI in rho is (lo, hi) where lo <= hi; convert to Q-CI:
    return Q, -float(hi), -float(lo)   # flip and negate


def main():
    proxy = _read(PROXY_JSON)
    eddy = _read(EDDY_JSON)
    print(f"proxy CIs: {list(proxy.keys())}")
    print(f"eddy CIs:  {list(eddy.keys())}")

    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, SINGLE_COL_IN * 0.85))

    n_groups = len(BASINS_ORDER)
    x_centres = np.arange(n_groups, dtype=float)
    bar_w = 0.36
    x_proxy = x_centres - bar_w / 2
    x_eddy  = x_centres + bar_w / 2

    proxy_Q = []; proxy_lo = []; proxy_hi = []
    eddy_Q  = []; eddy_lo  = []; eddy_hi  = []
    eddy_present = []
    for (basin, _) in BASINS_ORDER:
        p = _Q_and_ci(proxy.get(basin))
        if p is None:
            proxy_Q.append(np.nan); proxy_lo.append(0); proxy_hi.append(0)
        else:
            Q, lo, hi = p
            proxy_Q.append(Q); proxy_lo.append(Q - lo); proxy_hi.append(hi - Q)
        e = _Q_and_ci(eddy.get(basin))
        if e is None:
            eddy_Q.append(np.nan); eddy_lo.append(0); eddy_hi.append(0)
            eddy_present.append(False)
        else:
            Q, lo, hi = e
            eddy_Q.append(Q); eddy_lo.append(Q - lo); eddy_hi.append(hi - Q)
            eddy_present.append(True)

    # |grad SSH| proxy bars (always present from cached multi_basin run)
    ax.bar(x_proxy, proxy_Q, width=bar_w, color=COL_PROXY,
            edgecolor="black", linewidth=0.6,
            label=r"$|\nabla \mathrm{SSH}|$ proxy")
    ax.errorbar(x_proxy, proxy_Q,
                 yerr=[proxy_lo, proxy_hi], fmt="none",
                 ecolor="black", capsize=2.5, lw=0.8)

    # Geostrophic-velocity EKE bars (NA always; NP/ACC if cached)
    for i, present in enumerate(eddy_present):
        if present:
            ax.bar(x_eddy[i], eddy_Q[i], width=bar_w, color=COL_EDDY,
                    edgecolor="black", linewidth=0.6, hatch="///",
                    label=(r"geostrophic-velocity EKE"
                           if i == 0 else None))
            ax.errorbar(x_eddy[i], eddy_Q[i],
                         yerr=[[eddy_lo[i]], [eddy_hi[i]]], fmt="none",
                         ecolor="black", capsize=2.5, lw=0.8)
        else:
            # Pending: outlined empty bar at the proxy height as placeholder
            ax.bar(x_eddy[i], 0, width=bar_w, facecolor="none",
                    edgecolor="0.55", hatch="///", linewidth=0.6,
                    linestyle="--",
                    label=(r"geostrophic-velocity EKE (pending)"
                           if not any(eddy_present[:i]) and i == 0 else None))

    # Threshold line (value annotated in the caption, not on the plot).
    ax.axhline(BASIN_THRESHOLD, color="0.3", linestyle="--", linewidth=0.6,
                zorder=0)

    ax.set_xticks(x_centres)
    ax.set_xticklabels([lab for (_, lab) in BASINS_ORDER], fontsize=7.5)
    ax.margins(x=0.04)
    ax.set_ylabel(r"Quiescence Index  $Q = -\rho$")
    ax.set_ylim(0, max(0.8,
                         np.nanmax(proxy_Q) + max(proxy_hi),
                         np.nanmax(eddy_Q + [0]) + max(eddy_hi + [0])) + 0.05)
    ax.legend(loc="upper left", frameon=False, fontsize=6.5,
               handlelength=1.4)

    out = MANUSCRIPT_FIGS / "fig_R3_basin_bar_chart.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")

    # Print a per-basin summary table for the caption author.
    print("\n  basin              proxy Q  proxy CI         eddy Q   eddy CI")
    for (basin, lab) in BASINS_ORDER:
        p = _Q_and_ci(proxy.get(basin))
        e = _Q_and_ci(eddy.get(basin))
        ps = "n/a" if p is None else f"{p[0]:+.3f}  [{p[1]:+.3f},{p[2]:+.3f}]"
        es = "n/a" if e is None else f"{e[0]:+.3f}  [{e[1]:+.3f},{e[2]:+.3f}]"
        print(f"  {lab:<18}  {ps:<27}  {es}")


if __name__ == "__main__":
    main()
