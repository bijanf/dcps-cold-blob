"""Autonomous figure refresher.

Re-runs plot_holocene_status.py every N minutes so the status figure
reflects whatever results have landed since the last refresh.  Safe
to run alongside autonomous_holocene_hunt.py: simply reads cached
JSONs, never touches the hunter's work.

Also builds an aggregated "everywhere we looked" figure showing the
auto/<src>_<model>_<exp>.json results from the hunter, organised by
experiment.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style, DOUBLE_COL_IN
apply_nature_style()


AUTO_DIR = CACHE_DIR / "holocene_exit" / "auto"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"


def _collect_auto():
    """Group cached auto results by experiment for plotting."""
    grouped: dict[str, list[dict]] = {}
    if not AUTO_DIR.exists():
        return grouped
    for p in sorted(AUTO_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            exp = d["target"]["experiment"]
            grouped.setdefault(exp, []).append(d)
        except Exception:
            continue
    return grouped


def _aggregated_figure():
    grouped = _collect_auto()
    if not grouped:
        return None
    # Lay out: one panel per experiment, sorted by Holocene priority
    prio = ["past1000", "midHolocene", "lgm", "lig127k",
            "1pctCO2", "abrupt-4xCO2", "piControl",
            "historical", "ssp585", "ssp245", "ssp126", "ssp370"]
    exps = [e for e in prio if e in grouped] + \
           sorted(set(grouped) - set(prio))
    n = len(exps)
    if n == 0: return None
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(DOUBLE_COL_IN, max(2.2, 2.2 * rows)),
        squeeze=False, constrained_layout=True,
    )
    for i, exp in enumerate(exps):
        ax = axes.flat[i]
        for d in grouped[exp]:
            yrs = d["year_centres"]; Qs = d["Q"]
            model = d["target"]["model"]
            strat = d.get("strategy", "?")
            col = "C0" if strat == "Q-EKE" else "C2"
            ax.plot(yrs, Qs, "-", color=col, alpha=0.5, lw=0.6,
                    label=f"{model} ({strat})")
        ax.set_title(f"{exp}  (n={len(grouped[exp])} runs)",
                     fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_ylabel("Q", fontsize=7)
        ax.axhline(0, color="0.7", lw=0.4)
    for k in range(n, rows * cols):
        axes.flat[k].axis("off")
    out = MANUSCRIPT_FIGS / "fig_holocene_autohunt.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def _refresh_main_status():
    p = Path(__file__).resolve().parent / "plot_holocene_status.py"
    if not p.exists(): return
    try:
        subprocess.run([sys.executable, str(p)], check=False,
                        timeout=120, capture_output=True)
    except Exception:
        pass


def _refresh_aux():
    """Re-render auxiliary figures whose underlying caches grow as the
    background jobs run.  Each call is best-effort; missing inputs are
    handled gracefully by the individual scripts."""
    here = Path(__file__).resolve().parent
    aux = [
        ("audit_stationarity.py",            ["--basin", "atlantic"]),
        ("multi_model_first_exit_summary.py", ["--basin", "atlantic"]),
        ("plot_epoch_eke_composite.py",      ["--basin", "atlantic"]),
        ("compare_q_exit_vs_secondary.py",   ["--basin", "atlantic"]),
        ("plot_eke_epoch_diff.py",           ["--basin", "atlantic"]),
        ("recompute_significant_exit.py",    ["--basin", "atlantic"]),
        ("plot_picontrol_corridor.py",       ["--basin", "atlantic"]),
    ]
    for name, extra in aux:
        script = here / name
        if not script.exists(): continue
        try:
            subprocess.run([sys.executable, str(script), *extra],
                            check=False, timeout=180,
                            capture_output=True)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval-s", type=int, default=600)
    ap.add_argument("--max-iter", type=int, default=10_000)
    args = ap.parse_args()
    for k in range(args.max_iter):
        t0 = time.time()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}] refresh iteration {k + 1}")
        _refresh_main_status()
        _refresh_aux()
        out = _aggregated_figure()
        if out: print(f"  wrote {out}")
        grouped = _collect_auto()
        n_models = sum(len(v) for v in grouped.values())
        print(f"  auto cache: {n_models} runs across "
              f"{len(grouped)} experiments  ({time.time()-t0:.1f}s)")
        elapsed = time.time() - t0
        time.sleep(max(1, args.interval_s - elapsed))


if __name__ == "__main__":
    main()
