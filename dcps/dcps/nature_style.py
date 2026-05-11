"""Matplotlib styling for Nature-journal-compatible figures.

Call ``apply_nature_style()`` at import time in any figure-generation
script. Sets sans-serif fonts (Helvetica with fallback to Arial then
DejaVu Sans), removes background grids, applies inward tick marks,
sets sensible default font sizes, and switches to a publication-grade
DPI.

Column widths follow Nature Portfolio convention:
    single column: 89 mm (~3.5 in)
    double column: 183 mm (~7.2 in)

Use ``add_panel_label(ax, "a")`` to add lowercase bold panel labels in
the upper-left corner of a subplot in the Nature style.
"""

from __future__ import annotations

import matplotlib as mpl


def apply_nature_style() -> None:
    """Apply Nature-compatible matplotlib defaults globally."""
    rc = mpl.rcParams
    # Fonts
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    rc["font.size"] = 11
    rc["axes.labelsize"] = 12
    rc["axes.titlesize"] = 12
    rc["xtick.labelsize"] = 10
    rc["ytick.labelsize"] = 10
    rc["legend.fontsize"] = 10
    rc["figure.titlesize"] = 13
    # Tick marks point inward, on both axes
    rc["xtick.direction"] = "in"
    rc["ytick.direction"] = "in"
    rc["xtick.top"] = True
    rc["ytick.right"] = True
    rc["xtick.major.size"] = 3.5
    rc["ytick.major.size"] = 3.5
    rc["xtick.major.width"] = 0.6
    rc["ytick.major.width"] = 0.6
    rc["xtick.minor.size"] = 2.0
    rc["ytick.minor.size"] = 2.0
    # Axes line width
    rc["axes.linewidth"] = 0.6
    rc["axes.grid"] = False
    # Saving
    rc["savefig.dpi"] = 300
    rc["savefig.bbox"] = "tight"
    rc["savefig.pad_inches"] = 0.02
    # Line widths
    rc["lines.linewidth"] = 1.2


def add_panel_label(ax, label: str, x: float = -0.02, y: float = 1.04,
                     fontsize: int = 10) -> None:
    """Add a lowercase bold Nature-style panel label (a, b, c, ...)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontweight="bold", fontsize=fontsize,
            verticalalignment="bottom", horizontalalignment="left")


# Column widths in inches for matplotlib figsize
SINGLE_COL_IN = 3.5    # 89 mm
DOUBLE_COL_IN = 7.2    # 183 mm
