"""fig_q_cmip6.pdf -- the Quiescence Index per CMIP6 model, plus
observations.  Two-panel:
  (a) Per-model Q with HR (circles) vs LR (squares), one column
      per family, plus observed bands.
  (b) Box-and-strip plot of HR vs LR groups.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/bijanf/Documents/NEW_Theory")
JSON = REPO / "quiescence_toolkit/results/cmip6_quiescence.json"
OUT = REPO / "manuscript/figs/fig_q_cmip6.pdf"

PAIRS = [
    ("CESM2",           "CESM2-FV2",       "CESM2"),
    ("HadGEM3-GC31-MM", "HadGEM3-GC31-LL", "HadGEM3"),
    ("NorESM2-MM",      "NorESM2-LM",      "NorESM2"),
    ("CNRM-CM6-1-HR",   "CNRM-CM6-1",      "CNRM"),
    ("MPI-ESM1-2-HR",   "MPI-ESM1-2-LR",   "MPI-ESM"),
]
Q_OBS_NA_SST = 0.32
Q_OBS_NA_ZOS = 0.28


def main():
    data = json.loads(JSON.read_text())
    Q = data["Q_per_model"]
    hi = []; lo = []; labels = []
    for hi_id, lo_id, fam in PAIRS:
        hi.append(Q["high_res"].get(hi_id, {}).get("Q", np.nan))
        lo.append(Q["low_res"].get(lo_id, {}).get("Q", np.nan))
        labels.append(fam)
    hi = np.array(hi); lo = np.array(lo)
    print("hi:", hi)
    print("lo:", lo)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                    constrained_layout=True,
                                    gridspec_kw={"width_ratios": [3, 1.2]})

    # === Panel (a): per family ===
    x = np.arange(len(PAIRS))
    # Observed bands first (behind)
    ax0.axhspan(Q_OBS_NA_ZOS - 0.02, Q_OBS_NA_SST + 0.03,
                 color="0.85", alpha=0.6, zorder=0)
    ax0.axhline(Q_OBS_NA_SST, color="#2ca02c", linestyle="-",
                 linewidth=1.5, zorder=1)
    ax0.axhline(Q_OBS_NA_ZOS, color="#2ca02c", linestyle="--",
                 linewidth=1.5, zorder=1)
    ax0.axhline(0, color="0.4", linewidth=0.7, linestyle=":")

    ax0.scatter(x - 0.12, hi, marker="o", s=140, c="#d62728",
                 edgecolors="black", linewidths=1.2, zorder=4,
                 label="high resolution")
    ax0.scatter(x + 0.12, lo, marker="s", s=140, c="#1f77b4",
                 edgecolors="black", linewidths=1.2, zorder=4,
                 label="low resolution")
    for i, (h, l) in enumerate(zip(hi, lo)):
        if np.isfinite(h) and np.isfinite(l):
            ax0.plot([x[i] - 0.12, x[i] + 0.12], [h, l],
                       color="0.6", linewidth=0.8, zorder=2)

    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=20, ha="right")
    ax0.set_ylabel(r"Quiescence Index  $Q = -\rho$")
    ax0.set_title("(a)  CMIP6 resolution-variant pairs",
                   fontsize=11, loc="left")
    ax0.set_ylim(-0.10, 0.45)
    ax0.legend(loc="lower right", fontsize=9, framealpha=0.92)
    ax0.grid(True, linewidth=0.3, alpha=0.4)

    # === Panel (b): grouped strip + median bars ===
    hi_v = hi[np.isfinite(hi)]
    lo_v = lo[np.isfinite(lo)]
    # Boxes
    bp = ax1.boxplot([hi_v, lo_v], positions=[0, 1], widths=0.45,
                       patch_artist=True, showmeans=False, zorder=2,
                       medianprops=dict(color="black", linewidth=1.4))
    bp["boxes"][0].set_facecolor("#fbb4ae")
    bp["boxes"][1].set_facecolor("#b3cde3")
    # Strip
    rng = np.random.default_rng(0)
    for j, (vals, mk, col) in enumerate(
        [(hi_v, "o", "#d62728"), (lo_v, "s", "#1f77b4")]
    ):
        jitter = rng.uniform(-0.12, 0.12, size=vals.size)
        ax1.scatter(np.full_like(vals, j) + jitter, vals,
                     marker=mk, s=80, c=col, edgecolors="black",
                     linewidths=1.0, zorder=4)
    ax1.axhspan(Q_OBS_NA_ZOS - 0.02, Q_OBS_NA_SST + 0.03,
                 color="0.85", alpha=0.6, zorder=0)
    ax1.axhline(Q_OBS_NA_SST, color="#2ca02c", linestyle="-",
                 linewidth=1.5, zorder=1)
    ax1.axhline(Q_OBS_NA_ZOS, color="#2ca02c", linestyle="--",
                 linewidth=1.5, zorder=1)
    ax1.axhline(0, color="0.4", linewidth=0.7, linestyle=":")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["HR", "LR"])
    ax1.set_ylim(-0.10, 0.45)
    ax1.set_title("(b)  Grouped", fontsize=11, loc="left")
    ax1.grid(True, linewidth=0.3, alpha=0.4)

    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
