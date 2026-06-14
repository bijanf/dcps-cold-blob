"""
SI figure: decomposition of the 40-yr EKE-leads-Q population-level lag.

5 panels:
  (a) Bootstrap distribution of the median EKE-Q gap (10000 resamples)
      with point estimate, 68% and 95% CI marked.
  (b) Cumulative-distribution function of strict-exit years for Q vs EKE,
      showing that the entire EKE distribution sits left of Q -- i.e. the
      gap is a property of the full distribution, not just the median.
  (c) Per-model paired strict-exit-year scatter for the 5 Atlantic models
      that exit strictly on BOTH Q and EKE; 1:1 line and median paired lag.
  (d) Jackknife sensitivity: leave-one-model-out gap recomputation,
      sorted, with the +40 yr point estimate shown.
  (e) Multi-model median Welch PSD (with 25-75% inter-model band) of
      piControl basin-mean EKE and Q on the cached 30-yr-window grid
      (Nyquist period 60 yr).  The 13-28 yr North Atlantic decadal band
      (Arthun 2021) and the ~24 yr SPG eigenmode (Sevellec & Fedorov 2013)
      are annotated as the physical anchor of the 40-yr gap; these
      periods sit below the resolved frequency range of these 30-yr-window
      series.

Inputs:
  dcps/cache/lag_40yr/window_sensitivity.json (built by compute_lag_40yr_bootstrap.py)
  dcps/cache/lag_40yr/pi_psd_atlantic.json     (built by compute_lag_40yr_pi_psd.py)

Output:
  manuscript/figs/fig_si_lag_40yr_decomposition.pdf
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "dcps" / "cache" / "lag_40yr" / "window_sensitivity.json"
PSD_CACHE = REPO / "dcps" / "cache" / "lag_40yr" / "pi_psd_atlantic.json"
OUT = REPO / "manuscript" / "figs" / "fig_si_lag_40yr_decomposition.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)


def _bootstrap_distribution(q_years, e_years, n_boot=10000, seed=42):
    """Recompute bootstrap distribution of the median gap for plotting."""
    rng = np.random.default_rng(seed)
    q = np.array(q_years); e = np.array(e_years)
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        qs = rng.choice(q, size=len(q), replace=True)
        es = rng.choice(e, size=len(e), replace=True)
        gaps[i] = np.median(qs) - np.median(es)
    return gaps


def main():
    d = json.loads(CACHE.read_text())
    q_years = d["q_side"]["exit_years"]
    e_years = d["eke_side"]["exit_years"]
    gap_point = d["ensemble_gap"]["point_estimate_yr"]
    gap_95 = (d["ensemble_gap"]["ci95_low"], d["ensemble_gap"]["ci95_high"])
    gap_68 = (d["ensemble_gap"]["ci68_low"], d["ensemble_gap"]["ci68_high"])
    paired = d["within_model_paired"]["per_model"]
    paired_median = d["within_model_paired"]["median_lag"]
    jack = d["jackknife"]

    gaps = _bootstrap_distribution(q_years, e_years, n_boot=10000)

    # 2-column Nature width = 180 mm = 7.09 in
    # 3 rows x 2 cols; bottom row (panel e) spans the full width.
    fig = plt.figure(figsize=(7.09, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    # --- (a) bootstrap distribution ---
    ax_a.hist(gaps, bins=40, color="#4C72B0", edgecolor="white",
              linewidth=0.3, alpha=0.85)
    ax_a.axvline(gap_point, color="black", linestyle="-", linewidth=1.0)
    ax_a.axvspan(gap_68[0], gap_68[1], alpha=0.15, color="black")
    ax_a.axvspan(gap_95[0], gap_95[1], alpha=0.08, color="black")
    ax_a.axvline(0, color="grey", linestyle=":", linewidth=0.6)
    ax_a.set_xlabel("median($Q_{\\mathrm{exit}}$) $-$ median($\\mathrm{EKE}_{\\mathrm{exit}}$)  (yr)")
    ax_a.set_ylabel("Bootstrap count (out of 10 000)")
    ax_a.text(0.02, 0.98, "a", transform=ax_a.transAxes,
              ha="left", va="top", fontsize=9, fontweight="bold")
    ax_a.text(0.98, 0.98,
              f"point = $+{gap_point:.0f}$\n68% CI [{gap_68[0]:+.0f}, {gap_68[1]:+.0f}]\n"
              f"95% CI [{gap_95[0]:+.0f}, {gap_95[1]:+.0f}]",
              transform=ax_a.transAxes, ha="right", va="top",
              fontsize=6,
              bbox=dict(facecolor="white", edgecolor="0.7", linewidth=0.4, pad=2))

    # --- (b) CDF of strict-exit years ---
    q_sorted = np.sort(q_years)
    e_sorted = np.sort(e_years)
    q_cdf = np.arange(1, len(q_sorted) + 1) / len(q_sorted)
    e_cdf = np.arange(1, len(e_sorted) + 1) / len(e_sorted)
    ax_b.step(e_sorted, e_cdf, where="post", color="#C44E52",
              linewidth=1.2, label=f"EKE strict-exit ($n={len(e_sorted)}$)",
              marker="s", markersize=3.5)
    ax_b.step(q_sorted, q_cdf, where="post", color="#4C72B0",
              linewidth=1.2, label=f"$Q$ strict-exit ($n={len(q_sorted)}$)",
              marker="o", markersize=3.5)
    ax_b.axvline(np.median(e_sorted), color="#C44E52", linestyle=":", linewidth=0.7)
    ax_b.axvline(np.median(q_sorted), color="#4C72B0", linestyle=":", linewidth=0.7)
    ax_b.set_xlabel("First-exit year")
    ax_b.set_ylabel("Cumulative fraction of strictly-exiting models")
    ax_b.set_xlim(1850, 2060)
    ax_b.legend(loc="lower right", frameon=False)
    ax_b.text(0.02, 0.98, "b", transform=ax_b.transAxes,
              ha="left", va="top", fontsize=9, fontweight="bold")
    ax_b.text(0.40, 0.45,
              f"median:\nEKE = {int(np.median(e_sorted))}\n$Q$    = {int(np.median(q_sorted))}",
              transform=ax_b.transAxes, ha="left", va="top",
              fontsize=6,
              bbox=dict(facecolor="white", edgecolor="0.7", linewidth=0.4, pad=2))

    # --- (c) paired strict-exit scatter for 5 dual-exit models ---
    if paired:
        e_p = np.array([row["eke_exit"] for row in paired])
        q_p = np.array([row["q_exit"] for row in paired])
        names = [row["model"] for row in paired]
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        for i, (em, qm, nm) in enumerate(zip(e_p, q_p, names)):
            mk = markers[i % len(markers)]
            ax_c.scatter(em, qm, s=60, marker=mk, facecolors="#4C72B0",
                         edgecolors="black", linewidth=0.5,
                         label=nm, zorder=3)
        lo, hi = 1850, 2070
        ax_c.plot([lo, hi], [lo, hi], color="grey", linestyle="--", linewidth=0.7,
                  label="1:1")
        ax_c.set_xlim(lo, hi); ax_c.set_ylim(lo, hi)
        ax_c.set_xlabel("Strict-exit year, EKE")
        ax_c.set_ylabel("Strict-exit year, $Q$")
        ax_c.set_aspect("equal")
        ax_c.legend(loc="upper left", frameon=False, fontsize=5.5,
                    ncol=1, handletextpad=0.4)
        ax_c.text(0.98, 0.02,
                  f"paired median ($Q-\\mathrm{{EKE}}$) = ${paired_median:+.0f}$ yr",
                  transform=ax_c.transAxes, ha="right", va="bottom",
                  fontsize=6,
                  bbox=dict(facecolor="white", edgecolor="0.7", linewidth=0.4, pad=2))
    ax_c.text(0.02, 0.98, "c", transform=ax_c.transAxes,
              ha="left", va="top", fontsize=9, fontweight="bold")

    # --- (d) jackknife sensitivity ---
    # Recompute jackknife arrays for display (cheaper than reading them)
    q = np.array(q_years); e = np.array(e_years)
    jack_q = [np.median(np.delete(q, i)) - np.median(e) for i in range(len(q))]
    jack_e = [np.median(q) - np.median(np.delete(e, i)) for i in range(len(e))]
    cats = (["drop Q"] * len(jack_q)) + (["drop EKE"] * len(jack_e))
    vals = jack_q + jack_e
    xs = np.arange(len(vals))
    colors = ["#4C72B0"] * len(jack_q) + ["#C44E52"] * len(jack_e)
    sorted_idx = np.argsort(vals)
    vals_s = [vals[i] for i in sorted_idx]
    colors_s = [colors[i] for i in sorted_idx]
    cats_s = [cats[i] for i in sorted_idx]
    bar = ax_d.bar(xs, vals_s, color=colors_s, edgecolor="black", linewidth=0.3)
    ax_d.axhline(gap_point, color="black", linestyle="-", linewidth=0.8,
                 label=f"point = ${gap_point:+.0f}$ yr")
    ax_d.axhline(0, color="grey", linestyle=":", linewidth=0.6)
    ax_d.axhspan(jack["q_drop_one_min"], jack["q_drop_one_max"],
                 alpha=0.10, color="#4C72B0", label="drop-Q range")
    ax_d.axhspan(jack["e_drop_one_min"], jack["e_drop_one_max"],
                 alpha=0.10, color="#C44E52", label="drop-EKE range")
    ax_d.set_xticks([])
    ax_d.set_xlabel("Leave-one-out replicate (sorted)")
    ax_d.set_ylabel("Recomputed gap  (yr)")
    ax_d.legend(loc="lower right", frameon=False, fontsize=6)
    ax_d.text(0.02, 0.98, "d", transform=ax_d.transAxes,
              ha="left", va="top", fontsize=9, fontweight="bold")

    # --- (e) piControl Welch PSD: EKE vs Q ---
    psd = json.loads(PSD_CACHE.read_text())
    periods = np.array(psd["period_grid_yr"])
    eke_med = np.array(psd["eke"]["median"])
    eke_q25 = np.array(psd["eke"]["q25"])
    eke_q75 = np.array(psd["eke"]["q75"])
    q_med = np.array(psd["q"]["median"])
    q_q25 = np.array(psd["q"]["q25"])
    q_q75 = np.array(psd["q"]["q75"])
    n_eke = psd["eke"]["n_models"]
    n_q = psd["q"]["n_models"]

    # Reference band: 13-28 yr Arthun (2021) and 24 yr Sevellec-Fedorov (2013)
    # Plotted as background shading; lies below the resolved 60-yr Nyquist.
    period_axis_lo = 13.0
    period_axis_hi = 600.0
    ax_e.axvspan(13.0, 28.0, color="#999999", alpha=0.18,
                 label="13--28 yr decadal band (Arthun 2021)", zorder=0)
    ax_e.axvline(24.0, color="#444444", linestyle="--", linewidth=0.7,
                 zorder=1, label=r"$\sim$24 yr SPG eigenmode (Sevellec--Fedorov 2013)")
    # Vertical line marking the spectral floor (Nyquist period) of these series.
    ax_e.axvline(60.0, color="black", linestyle=":", linewidth=0.7, zorder=1)
    ax_e.text(60.0, 0.92, "  spectral floor\n  (60 yr Nyquist)",
              transform=ax_e.get_xaxis_transform(),
              ha="left", va="top", fontsize=5.5, color="black")
    # Hatch the unresolved (sub-60-yr) region of the panel.
    ax_e.axvspan(period_axis_lo, 60.0, facecolor="none",
                 edgecolor="#bbbbbb", hatch="////", linewidth=0.0,
                 alpha=0.6, zorder=0)

    # Multi-model bands: EKE solid + circles, Q dotted + squares.
    ax_e.fill_between(periods, eke_q25, eke_q75,
                       color="#C44E52", alpha=0.18, linewidth=0,
                       label=f"EKE 25--75% inter-model band ($n={n_eke}$)")
    ax_e.fill_between(periods, q_q25, q_q75,
                       color="#4C72B0", alpha=0.18, linewidth=0,
                       label=f"$Q$ 25--75% inter-model band ($n={n_q}$)")
    ax_e.plot(periods, eke_med, color="#C44E52", linestyle="-",
              linewidth=1.4, marker="s", markersize=3.0,
              markevery=4, markerfacecolor="white",
              markeredgewidth=0.7,
              label="EKE multi-model median PSD")
    ax_e.plot(periods, q_med, color="#4C72B0", linestyle=":",
              linewidth=1.4, marker="o", markersize=3.0,
              markevery=4, markerfacecolor="white",
              markeredgewidth=0.7,
              label="$Q$ multi-model median PSD")

    # 40-yr gap marker (lives inside the unresolved region but is a useful
    # visual anchor connecting panels a-d with panel e).
    ax_e.axvline(40.0, color="#7B2D26", linestyle="-", linewidth=0.7,
                 alpha=0.55, zorder=1)
    ax_e.text(40.0, 0.05, "  +40 yr gap",
              transform=ax_e.get_xaxis_transform(),
              ha="left", va="bottom", fontsize=5.5, color="#7B2D26")

    ax_e.set_xscale("log")
    ax_e.set_yscale("log")
    ax_e.set_xlim(period_axis_lo, period_axis_hi)
    ax_e.set_xlabel("Period  (yr)")
    ax_e.set_ylabel("Normalised power spectral density  (yr)")
    # Short period left -> long period right (decadal band on the left,
    # multi-centennial slow tail on the right).
    # Tick locations at canonical periods
    ticks = [13, 24, 28, 60, 100, 200, 400, 600]
    ax_e.set_xticks(ticks)
    ax_e.set_xticklabels([str(t) for t in ticks])
    ax_e.minorticks_off()
    ax_e.legend(loc="lower right", frameon=False, fontsize=5.5,
                ncol=1, handletextpad=0.5)
    ax_e.text(0.01, 0.98, "e", transform=ax_e.transAxes,
              ha="left", va="top", fontsize=9, fontweight="bold")

    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
