"""Snapshot figure for the Holocene-Q-corridor pipeline.

Aggregates all completed cache files in ``dcps/cache/holocene_exit/``
into a single multi-panel figure showing per-model piControl
corridors and historical+ssp585 trajectories.  Re-runnable; it picks
up whatever is currently on disk.

Outputs:
  manuscript/figs/fig_holocene_status.pdf      (multi-model grid)
  manuscript/figs/fig_holocene_canesm5_ge.pdf  (CanESM5 grand-ensemble)
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style, DOUBLE_COL_IN
apply_nature_style()


HOL_DIR = CACHE_DIR / "holocene_exit"
BULK_DIR = HOL_DIR / "bulk"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"
PALMOD_NPZ = CACHE_DIR / "palmod" / "holocene_stack.npz"


def _load_palmod_paleo():
    """Return (year_CE, basin_mean_SST_anom_degC) for the NA Holocene
    sediment-core stack.  Year CE = 1950 - kyr_BP * 1000.  Returns None
    if the stack isn't on disk."""
    if not PALMOD_NPZ.exists():
        return None
    d = np.load(PALMOD_NPZ, allow_pickle=True)
    kyr_bp = d["target_kyr"]
    basin = d["basin_mean"]
    year_CE = 1950 - kyr_bp * 1000.0
    # Sort ascending in time
    order = np.argsort(year_CE)
    return year_CE[order], basin[order]


def _collect_per_model():
    """Returns a dict: model -> per-model summary dict.  Includes
    bulk results AND pilot results (the pilot JSONs have the same
    schema).
    """
    per = {}
    # Bulk
    for p in sorted(BULK_DIR.glob("*_atlantic.json")):
        try:
            d = json.loads(p.read_text())
            per[d["model"]] = d
        except Exception:
            pass
    # Pilots (override if both exist; pilots have more hist windows)
    for p in sorted(HOL_DIR.glob("pilot_*_atlantic.json")):
        try:
            d = json.loads(p.read_text())
            model = d["model"]
            # Don't clobber bulk; pilots are extra info
            if model not in per:
                per[model] = d
        except Exception:
            pass
    return per


def _multi_model_figure(per_model):
    """Each panel shows the Holocene-like corridor (piControl Q range)
    as a green band occupying the pre-1850 epoch, then the
    historical+ssp585 Q trajectory after 1850.  The corridor's vertical
    extent is the piControl Q range; the threshold (red dashed) is the
    95-th percentile.  Conceptual time axis: piControl model years are
    placed in the pre-1850 region to visually mark the "Holocene-like"
    epoch -- the actual model-year integer values aren't real calendar
    years, but the band stands in for the Holocene baseline.
    """
    n = len(per_model)
    if n == 0:
        print("no models with results yet"); return
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(DOUBLE_COL_IN, max(2.2, 2.0 * rows)),
        squeeze=False, constrained_layout=True,
    )
    items = sorted(per_model.items(),
                    key=lambda kv: kv[1].get("first_exit_year") or 9999)

    HOLO_START = 1000  # left edge of the Holocene-like region on the axis
    HOLO_END = 1850    # right edge / industrial-era boundary
    AXIS_END = 2100

    for idx, (model, d) in enumerate(items):
        ax = axes.flat[idx]
        pi_min = min(d["pi_Q"]) if d.get("pi_Q") else 0
        pi_max = max(d["pi_Q"]) if d.get("pi_Q") else 0
        thr = d.get("pi_p95_threshold")

        # Holocene-like corridor: green rectangle in the pre-1850 region,
        # height = piControl Q range.
        ax.add_patch(plt.Rectangle(
            (HOLO_START, pi_min), HOLO_END - HOLO_START, pi_max - pi_min,
            facecolor="#a8d5ba", edgecolor="0.5", linewidth=0.4,
            zorder=0, alpha=0.7,
        ))
        ax.text((HOLO_START + HOLO_END) / 2, pi_min + (pi_max - pi_min) / 2,
                 "piControl =\nHolocene-like\ncorridor",
                 ha="center", va="center", fontsize=5.5, color="0.25")

        # post-1850 grey corridor band as before (for continuity)
        ax.axhspan(pi_min, pi_max, xmin=(HOLO_END - HOLO_START) / (AXIS_END - HOLO_START),
                    color="0.88", zorder=0)
        if thr is not None:
            ax.axhline(thr, color="C3", lw=0.6, ls="--")

        # Industrial era boundary
        ax.axvline(HOLO_END, color="0.4", lw=0.5, ls=":")

        hist_centres = d.get("hist_centres") or []
        hist_Q = d.get("hist_Q") or []
        if hist_centres:
            ax.plot(hist_centres, hist_Q, "o-", color="C0", ms=2.3, lw=0.8)
        exit_yr = d.get("first_exit_year")
        if exit_yr is not None:
            ax.axvline(exit_yr, color="C3", lw=0.5, ls=":")
        gate = d.get("stationarity_gate_passed", False)
        gate_str = "MK pass" if gate else "MK fail"
        ax.text(0.02, 0.98,
                  f"{model}\nexit {exit_yr}  {gate_str}",
                  transform=ax.transAxes, ha="left", va="top",
                  fontsize=5, color="0.2")
        ax.set_xlim(HOLO_START, AXIS_END)
        ax.set_ylim(-0.05, 0.55)
        ax.tick_params(labelsize=6)
        if idx % cols == 0:
            ax.set_ylabel(r"$Q$", fontsize=7)
        if idx // cols == rows - 1:
            ax.set_xlabel("year (Holocene-like ← | → industrial+future)",
                          fontsize=6)

    for k in range(len(items), rows * cols):
        axes.flat[k].axis("off")

    out = MANUSCRIPT_FIGS / "fig_holocene_status.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  ({n} models)")
    return out


def _canesm5_grand_ensemble_figure():
    """CanESM5: piControl band + per-member historical lines + per-member
    1pctCO2 lines, with a separate CO2-ppm secondary axis for 1pctCO2."""
    pi_path = HOL_DIR / "pilot_CanESM5_atlantic.json"
    if not pi_path.exists():
        print("no CanESM5 pilot piControl; skipping grand-ensemble figure")
        return None
    pi = json.loads(pi_path.read_text())
    pi_min = min(pi["pi_Q"]); pi_max = max(pi["pi_Q"])
    thr = pi.get("pi_p95_threshold")

    # Prefer the concatenated historical+ssp585 JSON if it exists (covers
    # 1850-2099); fall back to historical-only (covers 1850-2014).
    hist_path = HOL_DIR / "grand_CanESM5_historical+ssp585_atlantic.json"
    if not hist_path.exists():
        hist_path = HOL_DIR / "grand_CanESM5_historical_atlantic.json"
    onepct_path = HOL_DIR / "grand_CanESM5_1pctCO2_atlantic.json"

    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_IN, 2.6), constrained_layout=True,
        sharey=True,
    )
    # ---- historical+ssp585 panel ----
    ax = axes[0]
    HOLO_START, HOLO_END, AXIS_END = 0, 1850, 2100
    # Faint band for the eye
    ax.add_patch(plt.Rectangle(
        (HOLO_START, pi_min), HOLO_END - HOLO_START, pi_max - pi_min,
        facecolor="#a8d5ba", edgecolor="none",
        zorder=0, alpha=0.35,
    ))
    # ---- Plot the actual piControl Q segments as scatter ----
    # The piControl model-time has no real calendar mapping, so we
    # spread the segments evenly across the Holocene-equivalent period.
    pi_Q_list = pi["pi_Q"]
    n_pi = len(pi_Q_list)
    x_pi = np.linspace(HOLO_START + 100, HOLO_END - 100, n_pi)
    ax.scatter(x_pi, pi_Q_list, s=22, color="#2d8b3e",
                edgecolor="white", linewidth=0.6, zorder=4,
                label=f"piControl Q segments (n={n_pi}; "
                      f"internal-variability bound only, "
                      f"no transient Holocene forcing)")
    if thr is not None:
        ax.axhline(thr, color="C3", lw=0.6, ls="--",
                    label="piControl 95-th pctile")
    # ---- PalMod NA Holocene sediment-core SST anomaly (secondary axis) ----
    paleo = _load_palmod_paleo()
    if paleo is not None:
        year_CE, sst_anom = paleo
        keep = (year_CE >= HOLO_START) & (year_CE <= HOLO_END) & np.isfinite(sst_anom)
        if keep.sum() > 1:
            ax_pal = ax.twinx()
            ax_pal.plot(year_CE[keep], sst_anom[keep],
                          "-", color="#b35900", lw=1.0, marker="o", ms=2.4,
                          label="PalMod NA SST anom. (11 cores)")
            ax_pal.set_ylabel(r"SST anomaly (K)", color="#b35900",
                               fontsize=7)
            ax_pal.tick_params(axis="y", colors="#b35900", labelsize=6)
            ax_pal.spines["right"].set_color("#b35900")
            # Match the visible y-extent of the corridor so the paleo
            # series shares the same vertical band.
            ax_pal.set_ylim(-2.5, 1.5)
            ax_pal.legend(loc="lower left", fontsize=6, frameon=False)
    ax.axhspan(pi_min, pi_max,
                xmin=(HOLO_END - HOLO_START) / (AXIS_END - HOLO_START),
                color="0.88", zorder=0)
    ax.axvline(HOLO_END, color="0.4", lw=0.5, ls=":")
    if hist_path.exists():
        h = json.loads(hist_path.read_text())
        centres = h["year_centres"]
        for row in h["per_member_Q"]:
            ax.plot(centres, row, color="C0", alpha=0.25, lw=0.5)
        median = h["median_Q"]
        ax.plot(centres, median, "o-", color="C0", ms=2.8, lw=1.3,
                 label=f"historical+ssp585 ensemble median (N={h['n_members_used']})")
    ax.text(0.99, 0.02, "CanESM5  historical+ssp585",
              transform=ax.transAxes, ha="right", va="bottom",
              fontsize=6, color="0.3")
    ax.set_xlabel("year (Holocene-like ← | → industrial+future)", fontsize=7)
    ax.set_ylabel(r"$Q = -\rho$")
    ax.set_xlim(HOLO_START, AXIS_END)
    ax.set_ylim(-0.05, 0.55)
    ax.legend(loc="upper left", fontsize=6, frameon=False)

    # ---- 1pctCO2 panel ----
    ax = axes[1]
    ax.axhspan(pi_min, pi_max, color="0.85", zorder=0)
    if thr is not None:
        ax.axhline(thr, color="C3", lw=0.6, ls="--")
    if onepct_path.exists():
        o = json.loads(onepct_path.read_text())
        centres = o["year_centres"]
        # Map model-year to CO2 ppm.  1pctCO2 starts at ~285 ppm at the
        # first window's first year and multiplies by 1.01 per year.
        # Time origin: take the first window-start as year 0.
        year_starts = [c - 15 for c in centres]
        y0 = year_starts[0]
        co2 = [285.0 * (1.01 ** (ys - y0)) for ys in year_starts]
        for row in o["per_member_Q"]:
            ax.plot(centres, row, color="C2", alpha=0.25, lw=0.5)
        median = o["median_Q"]
        ax.plot(centres, median, "s-", color="C2", ms=2.8, lw=1.3,
                 label=f"ensemble median (N={o['n_members_used']})")
        # secondary axis (CO2 ppm) on top
        ax2 = ax.twiny()
        ax2.set_xlim(co2[0], co2[-1])
        ax2.set_xscale("linear")
        # Tick at 285, 380, 570 (2x), 855 (3x), 1140 (4x)
        ticks_ppm = [285, 380, 570, 855, 1140]
        ax2.set_xticks(ticks_ppm)
        ax2.set_xticklabels([f"{t}\nppm" for t in ticks_ppm], fontsize=6)
        ax2.tick_params(labelsize=6)
        # match the data axis range
        ax.set_xlim(centres[0], centres[-1])
    ax.text(0.99, 0.02, "CanESM5  1pctCO2 ramp",
              transform=ax.transAxes, ha="right", va="bottom",
              fontsize=6, color="0.3")
    ax.set_xlabel("model year")
    ax.legend(loc="best", fontsize=6, frameon=False)

    out = MANUSCRIPT_FIGS / "fig_holocene_canesm5_ge.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main():
    per_model = _collect_per_model()
    print(f"models with results on disk: {len(per_model)}")
    for k in sorted(per_model):
        d = per_model[k]
        exit_yr = d.get("first_exit_year")
        print(f"  {k}: exit_year={exit_yr}  "
              f"n_pi={len(d.get('pi_Q', []))}  "
              f"n_hist={len(d.get('hist_Q', []))}")

    _multi_model_figure(per_model)
    _canesm5_grand_ensemble_figure()


if __name__ == "__main__":
    main()
