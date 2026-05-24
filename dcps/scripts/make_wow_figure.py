"""WOW figure: cross-era unprecedentedness ladder.

A single-panel summary of the paper's quantitative novelty: the modern
Cold-Blob signal exceeds the |z| > 3 unprecedented threshold at every
timescale from a century to five millennia, in every paleo dataset
that can probe that timescale. The |z|-score itself grows
monotonically with window length, because longer windows have tighter
Holocene Sen-slope distributions.

This is what neither Caesar 2021 (single-scale, last 1000 yr) nor
Thornalley 2018 (single-scale, last 150 yr) showed: a multi-scale
ladder of unprecedentedness, anchored both in the newly-published
PALMOD-130k multi-millennial synthesis and in the high-resolution
multi-proxy compilation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()


MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"


def load_palmod_z_scores() -> dict[int, float]:
    """PALMOD-130k subpolar-minus-subtropical contrast unprecedented
    |z| at sliding-window sizes from cold_blob_unprecedented.json."""
    path = CACHE_DIR / "cold_blob_unprecedented" / "unprecedented.json"
    data = json.loads(path.read_text())
    out = {}
    for k, info in data["by_window"].items():
        w_yr = int(info["window_kyr"] * 1000)
        out[w_yr] = abs(info["z_score"])
    return out


def load_proxy_z_scores() -> dict[str, dict[int, float]]:
    """Per-proxy |z| at each window size, from multi_proxy.json."""
    path = CACHE_DIR / "multi_proxy" / "multi_proxy.json"
    data = json.loads(path.read_text())
    out = {}
    for proxy in data["per_proxy"]:
        if proxy.get("verdict_overall") in (None, "modern-only-no-test"):
            continue
        if "windows" not in proxy: continue
        z_by_w = {}
        for k, info in proxy["windows"].items():
            if "z_score" not in info: continue
            w_yr = int(k.split("_")[1])
            z_by_w[w_yr] = abs(info["z_score"])
        if z_by_w:
            out[proxy["display"]] = z_by_w
    return out


def main():
    palmod = load_palmod_z_scores()
    proxies = load_proxy_z_scores()

    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)

    # Reference bands: |z| < 2 noise floor, |z| = 3 unprecedented threshold
    ax.axhspan(0, 2, color="0.88", alpha=0.6, zorder=0)
    ax.axhline(3, color="0.4", lw=0.9, linestyle="--", zorder=1)

    # PALMOD ladder: large filled red squares
    p_w = sorted(palmod.keys())
    p_z = [palmod[w] for w in p_w]
    ax.plot(p_w, p_z, color="C3", lw=2.0, zorder=4)
    ax.plot(p_w, p_z, "s", markersize=13, color="C3",
             markeredgecolor="k", markeredgewidth=0.8, zorder=5,
             label="PALMOD-130k (this work)")

    # Caesar 2021 multi-proxy: distinct color AND distinct marker
    # per proxy so the four series are distinguishable without
    # color alone.
    proxy_style = {
        "Thornalley 2018 $T_{sub}$":          dict(color="C0", marker="o"),
        "Rahmstorf 2015 AMOC index":          dict(color="C2", marker="^"),
        "Spooner 2020 $T.$ quinqueloba":      dict(color="C1", marker="D"),
        "Thornalley 2018 sortable silt":      dict(color="0.5", marker="v"),
    }
    for name, z_by_w in proxies.items():
        ws = sorted(z_by_w.keys())
        zs = [z_by_w[w] for w in ws]
        sty = proxy_style.get(name, dict(color="0.4", marker="P"))
        ax.plot(ws, zs, sty["marker"], markersize=10,
                 color=sty["color"],
                 markeredgecolor="k", markeredgewidth=0.5,
                 zorder=6, label=name)

    ax.set_xscale("log")
    ax.set_xticks([100, 200, 500, 1000, 2000, 5000])
    ax.set_xticklabels(["100", "200", "500", "1000", "2000", "5000"])
    ax.set_xlim(60, 7000)
    y_top = max(max(p_z),
                 max(max(z.values()) for z in proxies.values())) * 1.18
    ax.set_ylim(0, y_top)
    ax.set_xlabel("Sliding-window timescale (years)")
    ax.set_ylabel(r"$|z|$ vs Holocene/pre-1850 distribution ($\sigma$)")
    ax.legend(loc="upper left", frameon=False, fontsize=22)

    out_fig = MANUSCRIPT_FIGS / "fig_wow.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")
    print("\nLadder summary:")
    for w in sorted(p_w):
        print(f"  PALMOD {w:>5} yr window:  |z| = {palmod[w]:5.1f}")
    for name, z_by_w in proxies.items():
        for w, z in sorted(z_by_w.items()):
            print(f"  {name:<40} {w:>5} yr:  |z| = {z:5.1f}")


if __name__ == "__main__":
    main()
