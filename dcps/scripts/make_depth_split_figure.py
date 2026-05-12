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
import numpy as np

from dcps.nature_style import apply_nature_style
apply_nature_style()

OUT = (Path(__file__).resolve().parents[2]
       / "manuscript" / "figs" / "fig_depth_split.pdf")


def main():
    # Five testable Caesar-2021 proxies:
    # (name, sampling depth in m, |z|_max, group)
    proxies = [
        ("Thornalley 2018 $T_{\\mathrm{sub}}$",          400.0,  19.7, "surface"),
        ("Spooner 2020 $T.$ quinqueloba",                100.0,   4.8, "surface"),
        ("Rahmstorf 2015 multi-proxy AMOC",               50.0,  22.1, "surface"),
        ("Thornalley 2018 sortable silt",               2000.0,   0.8, "deep"),
        ("Osmann 2019 MAS productivity",                3500.0,   0.0, "deep"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)

    for name, depth, z, group in proxies:
        color = "crimson" if group == "surface" else "steelblue"
        marker = "o" if group == "surface" else "s"
        ax.scatter(depth, z, s=220, c=color, marker=marker,
                      edgecolors="black", linewidths=1.5, zorder=10,
                      label=None)
        # Proxy name as a marker label next to the data point.
        # Position is chosen so the text never overlaps another
        # marker or the legend.
        text_x_mult = {
            "Thornalley 2018 $T_{\\mathrm{sub}}$":         0.55,
            "Spooner 2020 $T.$ quinqueloba":               1.20,
            "Rahmstorf 2015 multi-proxy AMOC":             1.20,
            "Thornalley 2018 sortable silt":               0.55,
            "Osmann 2019 MAS productivity":                0.55,
        }[name]
        text_y_off = {
            "Thornalley 2018 $T_{\\mathrm{sub}}$":          0.0,
            "Spooner 2020 $T.$ quinqueloba":                0.0,
            "Rahmstorf 2015 multi-proxy AMOC":              0.0,
            "Thornalley 2018 sortable silt":               +0.8,
            "Osmann 2019 MAS productivity":                +0.8,
        }[name]
        ha = "left" if text_x_mult > 1.0 else "right"
        ax.annotate(name, xy=(depth, z),
                        xytext=(depth * text_x_mult, z + text_y_off),
                        fontsize=10, ha=ha, va="center")

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
                  label="Surface / thermocline"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="steelblue",
                  markeredgecolor="black", markersize=12,
                  label="Deep / abyssal"),
    ]
    ax.legend(handles=legend_elements, loc="center right", fontsize=10,
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
