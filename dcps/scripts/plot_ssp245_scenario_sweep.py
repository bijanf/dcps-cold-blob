"""SSP2-4.5 vs SSP5-8.5 scenario-robustness figure for Q-corridor exit.

Two panels:
  (a) Cumulative distribution of first-exit year for the 29 CMIP6 models
      with the full piControl + historical + scenario chain, computed
      independently under SSP5-8.5 (high forcing) and SSP2-4.5 (moderate
      forcing). Both curves share the same piControl threshold per model,
      so any divergence reflects scenario-forcing strength alone.

  (b) Per-model paired scatter: exit-year under SSP5-8.5 (x) versus
      exit-year under SSP2-4.5 (y). The 1:1 line marks scenario-identical
      timing. Models that fail to exit by 2099 under SSP2-4.5 are placed
      on the upper "no-exit" rail at y=2105; models that fail under SSP5-8.5
      are placed at x=2105.

Output: manuscript/figs/fig_ssp245_scenario_sweep.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style, DOUBLE_COL_IN, add_panel_label
apply_nature_style()

BULK = CACHE_DIR / "holocene_exit" / "bulk"
FIG_OUT = PKG_ROOT.parent / "manuscript" / "figs" / "fig_ssp245_scenario_sweep.pdf"

NO_EXIT_RAIL = 2105
EXIT_RANGE = (1850, 2100)


def _load_exits():
    """Returns {model: (exit_585, exit_245)} for models present in both."""
    out = {}
    for f in sorted(BULK.glob("*_atlantic.json")):
        name = f.stem
        if any(t in name for t in ("ssp245", "ssp370", "ssp126",
                                    "1pctCO2", "abrupt-4xCO2", "historical-")):
            continue
        model = name.replace("_atlantic", "")
        d585 = json.loads(f.read_text())
        f245 = BULK / f"{model}_atlantic_ssp245.json"
        if not f245.exists():
            continue
        d245 = json.loads(f245.read_text())
        out[model] = (d585.get("first_exit_year"),
                       d245.get("first_exit_year"))
    return out


def _cdf(years):
    arr = np.asarray([y for y in years if y is not None], dtype=float)
    arr = np.sort(arr)
    yvals = np.arange(1, len(arr) + 1) / len(years)
    return arr, yvals


def main():
    data = _load_exits()
    n = len(data)
    print(f"Paired models (SSP585 ∩ SSP245): {n}")

    exits_585 = [v[0] for v in data.values()]
    exits_245 = [v[1] for v in data.values()]
    n_exit_585 = sum(1 for x in exits_585 if x is not None)
    n_exit_245 = sum(1 for x in exits_245 if x is not None)
    print(f"  SSP5-8.5 exits: {n_exit_585}/{n}")
    print(f"  SSP2-4.5 exits: {n_exit_245}/{n}")
    print(f"  SSP2-4.5 no-exit models: "
          f"{[m for m, v in data.items() if v[1] is None]}")

    fig = plt.figure(figsize=(DOUBLE_COL_IN, 3.2), constrained_layout=True)
    axA, axB = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.05, 1.0]})

    col585 = "#cc3322"
    col245 = "#225eaa"

    # ----- Panel (a): CDF of first-exit year ----------------------------
    x585, y585 = _cdf(exits_585)
    x245, y245 = _cdf(exits_245)
    n_total_585 = len(exits_585)
    n_total_245 = len(exits_245)
    axA.step(x585, y585, where="post", color=col585, lw=1.6,
              label=f"SSP5-8.5  (n={n_exit_585}/{n_total_585})")
    axA.step(x245, y245, where="post", color=col245, lw=1.6,
              label=f"SSP2-4.5  (n={n_exit_245}/{n_total_245})")
    # median markers
    med_585 = np.median([y for y in exits_585 if y is not None])
    med_245 = np.median([y for y in exits_245 if y is not None])
    axA.axvline(med_585, color=col585, ls=":", lw=0.7, alpha=0.6)
    axA.axvline(med_245, color=col245, ls=":", lw=0.7, alpha=0.6)
    axA.set_xlabel("first Q-exit year")
    axA.set_ylabel("cumulative fraction of models")
    axA.set_xlim(EXIT_RANGE)
    axA.set_ylim(0, 1.02)
    axA.legend(frameon=False, loc="lower right")
    add_panel_label(axA, "a")

    # ----- Panel (b): paired exit-year scatter --------------------------
    xs, ys = [], []
    for m, (a, b) in data.items():
        xs.append(NO_EXIT_RAIL if a is None else a)
        ys.append(NO_EXIT_RAIL if b is None else b)
    xs = np.asarray(xs); ys = np.asarray(ys)

    # 1:1 line
    axB.plot([EXIT_RANGE[0], EXIT_RANGE[1]],
              [EXIT_RANGE[0], EXIT_RANGE[1]],
              color="0.6", lw=0.7, ls="--", zorder=1)

    # shaded no-exit rails
    axB.axhspan(NO_EXIT_RAIL - 5, NO_EXIT_RAIL + 5,
                 color="0.92", zorder=0)
    axB.axvspan(NO_EXIT_RAIL - 5, NO_EXIT_RAIL + 5,
                 color="0.92", zorder=0)
    axB.text(EXIT_RANGE[0] + 5, NO_EXIT_RAIL,
              "no exit by 2099 (SSP2-4.5)", fontsize=7, color="0.4",
              va="center", ha="left")

    # color-coded by whether both, one, or neither exited
    n_both = n_585_only = n_245_only = n_neither = 0
    for m, (a, b) in data.items():
        x = NO_EXIT_RAIL if a is None else a
        y = NO_EXIT_RAIL if b is None else b
        if a is not None and b is not None:
            c, mk = "k", "o"; n_both += 1
        elif a is not None and b is None:
            c, mk = col585, "^"; n_585_only += 1
        elif a is None and b is not None:
            c, mk = col245, "v"; n_245_only += 1
        else:
            c, mk = "0.5", "x"; n_neither += 1
        axB.scatter(x, y, s=22, marker=mk, color=c, alpha=0.9,
                     linewidth=0.8, zorder=3)

    axB.set_xlim(EXIT_RANGE[0], NO_EXIT_RAIL + 5)
    axB.set_ylim(EXIT_RANGE[0], NO_EXIT_RAIL + 5)
    axB.set_xlabel("first Q-exit year under SSP5-8.5")
    axB.set_ylabel("first Q-exit year under SSP2-4.5")

    # custom legend — list every glyph that appears in panel (b)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="k",
                markersize=5, label=f"exit under both scenarios (n={n_both})"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=col585,
                markersize=5, label=f"exit SSP5-8.5 only (n={n_585_only})"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=col245,
                markersize=5, label=f"exit SSP2-4.5 only (n={n_245_only})"),
        Line2D([0], [0], marker="x", color="0.5", linestyle="",
                markersize=5, label=f"no exit either scenario (n={n_neither})"),
    ]
    axB.legend(handles=handles, frameon=False, loc="upper left",
                fontsize=7, handletextpad=0.4)
    add_panel_label(axB, "b")

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT)
    print(f"wrote {FIG_OUT}")

    # Also dump the underlying summary numbers
    summary = {
        "n_paired_models": n,
        "n_exit_ssp585": n_exit_585,
        "n_exit_ssp245": n_exit_245,
        "median_exit_ssp585": float(med_585),
        "median_exit_ssp245": float(med_245),
        "p10_p90_ssp585": [
            float(np.percentile([y for y in exits_585 if y is not None], 10)),
            float(np.percentile([y for y in exits_585 if y is not None], 90)),
        ],
        "p10_p90_ssp245": [
            float(np.percentile([y for y in exits_245 if y is not None], 10)),
            float(np.percentile([y for y in exits_245 if y is not None], 90)),
        ],
        "ssp245_no_exit_models": [m for m, v in data.items() if v[1] is None],
        "per_model": {m: {"ssp585": a, "ssp245": b} for m, (a, b) in data.items()},
    }
    sum_path = CACHE_DIR / "holocene_exit" / "ssp245_scenario_sweep_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {sum_path}")


if __name__ == "__main__":
    main()
