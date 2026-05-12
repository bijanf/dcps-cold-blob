"""Dedicated figure for the depth-dependent unprecedented split.

Reviewer flagged that the new depth-split section was re-using
fig_wow.pdf (the cross-scale ladder), which is also the lead figure
of the unprecedented-test section.  This script produces a distinct
figure: depth on x-axis vs |z|-score on y-axis for the five testable
Caesar-2021 proxies, colour-coded by surface/deep group.
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
    # Five testable Caesar-2021 proxies: (name, sampling depth in m,
    # |z|_max, group)
    proxies = [
        ("Thornalley 2018\n$T_{\\mathrm{sub}}$",       400.0,  19.7, "surface"),
        ("Spooner 2020\n$T.$ quinqueloba",             100.0,   4.8, "surface"),
        ("Rahmstorf 2015\nmulti-proxy AMOC",            50.0,  22.1, "surface"),
        ("Thornalley 2018\nsortable silt",            2000.0,   0.8, "deep"),
        ("Osmann 2019\nMAS productivity",             3500.0,   0.0, "deep"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)

    # Per-proxy marker, colour, and label offset for the five
    # testable proxies.  Two distinct shapes for the two groups so
    # that the figure remains readable in greyscale.
    label_offsets = {
        "Thornalley 2018\n$T_{\\mathrm{sub}}$":          (1.15, -3.0,  "left"),
        "Spooner 2020\n$T.$ quinqueloba":                (1.15, +1.5,  "left"),
        "Rahmstorf 2015\nmulti-proxy AMOC":              (1.15, -3.0,  "left"),
        "Thornalley 2018\nsortable silt":                (1.10, +5.5,  "left"),
        "Osmann 2019\nMAS productivity":                 (0.55, +7.5,  "right"),
    }
    for name, depth, z, group in proxies:
        color = "crimson" if group == "surface" else "steelblue"
        marker = "o" if group == "surface" else "s"
        ax.scatter(depth, z, s=200, c=color, marker=marker,
                      edgecolors="black", linewidths=1.5, zorder=10)
        x_mult, y_off, ha = label_offsets[name]
        ax.annotate(name, xy=(depth, z),
                        xytext=(depth * x_mult, z + y_off),
                        fontsize=9, ha=ha)

    # |z|=3 threshold
    ax.axhline(3.0, color="black", linestyle="--", linewidth=1.5,
                  zorder=2, label="|z| = 3 unprecedented threshold")

    # Shade the "not unprecedented" region below |z|=3
    ax.axhspan(-0.5, 3.0, color="0.92", zorder=1)

    # Mark thermocline (~400 m) and abyssal (~2000 m) reference lines
    ax.axvline(400, color="0.6", linestyle=":", linewidth=1.0, zorder=2)
    ax.axvline(2000, color="0.6", linestyle=":", linewidth=1.0, zorder=2)
    ax.text(400, 25, "thermocline\n(~400 m)", fontsize=8, ha="center",
              color="0.4")
    ax.text(2000, 25, "abyssal\n(~2000 m)", fontsize=8, ha="center",
              color="0.4")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
                  markeredgecolor="black", markersize=12,
                  label="Surface / thermocline (3/3 unprec)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="steelblue",
                  markeredgecolor="black", markersize=12,
                  label="Deep / abyssal (0/2 unprec)"),
        Line2D([0], [0], color="black", linestyle="--",
                  label="|z| = 3 threshold"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.set_xscale("log")
    ax.set_xlim(20, 9000)
    ax.set_ylim(-0.5, 28)
    ax.set_xlabel("Sampling depth (m, log scale)")
    ax.set_ylabel(r"$|z|$-score: modern Sen slope vs pre-1850 envelope")
    ax.grid(True, alpha=0.3, which="both")

    # Inset: Fisher's-exact p-value
    ax.text(0.04, 0.06,
              "Fisher's-exact (3/3 vs 0/2 split):\n"
              "$p = 0.10$ (10,000-perm M.C.: 0.097)\n"
              "Underpowered with only 5 proxies",
              transform=ax.transAxes, fontsize=9, va="bottom",
              bbox=dict(boxstyle="round,pad=0.5",
                          facecolor="white", edgecolor="0.5"))

    fig.savefig(OUT)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
