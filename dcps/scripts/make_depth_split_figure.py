"""Dedicated figure for the depth-dependent unprecedented split.

No inset text annotations or descriptive labels on the plot.  All
descriptive content (Fisher's-exact p-value, methodological
caveats, thermocline/abyssal reference depths) lives in the figure
caption.  Plot communicates visually only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dcps.nature_style import apply_nature_style
apply_nature_style()

OUT = (Path(__file__).resolve().parents[2]
       / "manuscript" / "figs" / "fig_depth_split.pdf")


def main():
    # Seven testable proxies: 5 from Caesar 2021 + 2 augmented this
    # revision (Thibodeau MD99-2220 deep, Moffa-Sanchez RAPiD-35-COM
    # deep).  (name, sampling depth in m, |z|_max, group).
    # Augmented proxies use a different marker EDGE color so the
    # reader can distinguish original from added without text on plot.
    proxies = [
        ("Thornalley 2018 $T_{\\mathrm{sub}}$",          400.0,  19.7, "surface", False),
        ("Spooner 2020 $T.$ quinqueloba",                100.0,   4.8, "surface", False),
        ("Rahmstorf 2015 multi-proxy AMOC",               50.0,  22.1, "surface", False),
        ("Thornalley 2018 sortable silt",               2000.0,   0.8, "deep",    False),
        ("Osmann 2019 MAS productivity",                3500.0,   0.0, "deep",    False),
        ("Thibodeau 2018 MD99-2220 $\\delta^{18}$O",    3700.0,   4.65,"deep",    True),
        ("Moffa-Sanchez 2015 RAPiD-35-COM",             3484.0,   1.15,"deep",    True),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)

    for name, depth, z, group, augmented in proxies:
        color = "crimson" if group == "surface" else "steelblue"
        marker = "o" if group == "surface" else "s"
        edge = "black"
        face = color
        if augmented:
            face = "lightsteelblue"
        ax.scatter(depth, z, s=240, c=face, marker=marker,
                      edgecolors=edge, linewidths=1.8, zorder=10,
                      label=None)

    # |z|=3 threshold dashed line
    ax.axhline(3.0, color="black", linestyle="--", linewidth=1.5,
                  zorder=2)
    # Shaded "within envelope" zone below |z|=3
    ax.axhspan(-2.0, 3.0, color="0.92", zorder=1)

    # Reference vertical lines at the thermocline (~400 m) and abyssal
    # (~2000 m) depth horizons.  The lines communicate location; the
    # caption tells the reader what they mean.
    ax.axvline(400, color="0.6", linestyle=":", linewidth=1.0, zorder=2)
    ax.axvline(2000, color="0.6", linestyle=":", linewidth=1.0, zorder=2)

    # Legend: only the two groups, placed where it doesn't overlap data.
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
                  markeredgecolor="black", markersize=12,
                  label="Surface / thermocline (Caesar 2021)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="steelblue",
                  markeredgecolor="black", markersize=12,
                  label="Deep / abyssal (Caesar 2021)"),
        Line2D([0], [0], marker="s", color="w",
                  markerfacecolor="lightsteelblue",
                  markeredgecolor="black", markersize=12,
                  label="Deep / abyssal (augmented this revision)"),
    ]
    ax.legend(handles=legend_elements, loc="center right", fontsize=22,
                  framealpha=0.95)

    ax.set_xscale("log")
    ax.set_xlim(20, 9000)
    ax.set_ylim(-2.0, 28)   # extend lower bound so markers aren't clipped
    ax.set_xlabel("Sampling depth (m, log scale)")
    ax.set_ylabel(r"$|z|$-score: modern Sen slope vs pre-1850 envelope")
    ax.grid(True, alpha=0.3, which="both")

    fig.savefig(OUT)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
