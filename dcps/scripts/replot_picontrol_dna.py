"""Replot the 2-panel piControl detection-and-attribution figure from
the cached per-model slope distributions.

Follows the project figure-quality standard:
  - no in-plot titles
  - bold corner panel letters as the only in-axis text identifier
  - legend placed to avoid overlapping data
  - no in-plot statistics text box (statistics live in the caption)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _panel(ax, label, dx=-0.10):
    # Panel letter outside the axes (top-left), no bbox.
    ax.text(dx, 1.02, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom", ha="left")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, required=True,
                   help="dna_results.json (output of cmip6_picontrol_dna.py).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output PDF path.")
    args = p.parse_args(argv)

    d = json.loads(args.json.read_text())
    observed = float(d["observed_rate_degC_per_century"])
    per_model = d["per_model"]
    # Order models by exceedance fraction descending (CMCC first as outlier).
    model_order = sorted(
        per_model.keys(),
        key=lambda m: -sum(1 for s in per_model[m]["slopes_per_century"]
                            if s <= observed) / max(len(per_model[m]["slopes_per_century"]), 1),
    )

    # All-models pooled and CMCC-excluded pooled slopes.
    all_slopes = []
    no_cmcc_slopes = []
    for m, md in per_model.items():
        slopes = md["slopes_per_century"]
        all_slopes.extend(slopes)
        if m != "CMCC-CM2-SR5":
            no_cmcc_slopes.extend(slopes)
    all_slopes = np.asarray(all_slopes)
    no_cmcc_slopes = np.asarray(no_cmcc_slopes)
    n_all = all_slopes.size
    n_nocmcc = no_cmcc_slopes.size

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.5, 4.6),
                                    constrained_layout=True,
                                    gridspec_kw={"width_ratios": [1.2, 1.0]})

    # ---- Panel (a): per-model violin ------------------------------------
    positions = np.arange(len(model_order))
    data_for_violin = [per_model[m]["slopes_per_century"] for m in model_order]
    colors = ["crimson" if m == "CMCC-CM2-SR5" else "0.65" for m in model_order]
    parts = axA.violinplot(data_for_violin, positions=positions, widths=0.85,
                            showmeans=False, showmedians=True, showextrema=True)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor("0.25")
        body.set_alpha(0.75)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("steelblue")
            parts[key].set_linewidth(1.0)

    axA.axhline(observed, color="red", lw=1.5,
                 label=f"HadISST observed ({observed:+.2f})")
    axA.set_xticks(positions)
    axA.set_xticklabels(model_order, rotation=30, ha="right")
    axA.set_ylabel(r"100-yr Sen slope (deg C / century)")

    # Y-axis upper limit pushed up so the per-violin percentage labels
    # never get clipped at the top of the panel.
    ymin = min(min(per_model[m]["slopes_per_century"]) for m in model_order) - 0.3
    ymax = max(max(per_model[m]["slopes_per_century"]) for m in model_order) + 0.55
    axA.set_ylim(ymin, ymax)
    for i, m in enumerate(model_order):
        slopes = per_model[m]["slopes_per_century"]
        n_exc = sum(1 for s in slopes if s <= observed)
        pct = 100.0 * n_exc / max(len(slopes), 1)
        col = "crimson" if m == "CMCC-CM2-SR5" else "0.30"
        axA.text(i, ymax - 0.15, f"{pct:.0f}%",
                  ha="center", va="top", color=col,
                  fontsize=9, fontweight="bold")
    axA.legend(loc="lower right", frameon=False)
    axA.tick_params(direction="in", length=2.5)
    axA.grid(alpha=0.25, axis="y")
    _panel(axA, "a", dx=-0.10)

    # ---- Panel (b): pooled histogram, with and without CMCC -------------
    bins = np.linspace(-3.0, 1.5, 41)
    axB.hist(all_slopes, bins=bins, color="0.7", alpha=0.7,
              edgecolor="0.4", label=f"all 8 models (n={n_all})")
    axB.hist(no_cmcc_slopes, bins=bins, color="steelblue", alpha=0.7,
              edgecolor="0.25", label=f"excluding CMCC (n={n_nocmcc})")
    axB.axvline(observed, color="red", lw=1.5,
                 label=f"observed {observed:+.2f}")
    axB.set_xlabel(r"100-yr Sen slope, subpolar$-$subtropical SST contrast"
                     r" (deg C / century)")
    axB.set_ylabel("piControl windows")
    # Legend on the LEFT side (per author request).  Anchored just
    # below the top so it does not collide with the panel letter
    # placed above the axes at (dx, 1.02).
    axB.legend(loc="upper left", bbox_to_anchor=(0.02, 0.97),
                frameon=False)
    axB.tick_params(direction="in", length=2.5)
    axB.grid(alpha=0.25, axis="y")
    _panel(axB, "b", dx=-0.10)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
