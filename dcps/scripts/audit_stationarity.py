"""Per-basin, per-model stationarity audit for the Q corridor.

Reads every cached bulk JSON in ``dcps/cache/holocene_exit/bulk/`` and
builds an audit table summarising the Mann-Kendall trend test on each
model's piControl Q segments.  The gate is documented as part of the
pre-registration: a model's piControl null distribution is only
admissible to the multi-model ensemble if Kendall tau over its 30-yr
piControl Q segments has p > 0.05 (no detectable monotonic trend),
i.e. it is statistically indistinguishable from a stationary record.

Outputs:
  dcps/cache/holocene_exit/audit/stationarity_<basin>.json
  manuscript/figs/fig_stationarity_audit_<basin>.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style, SINGLE_COL_IN
apply_nature_style()

BULK_DIR = CACHE_DIR / "holocene_exit" / "bulk"
AUDIT_DIR = CACHE_DIR / "holocene_exit" / "audit"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

ALPHA = 0.05


def _load_audit(basin: str) -> list[dict]:
    rows = []
    for p in sorted(BULK_DIR.glob(f"*_{basin}.json")):
        d = json.loads(p.read_text())
        if "pi_mk_tau" not in d or d.get("pi_mk_tau") is None:
            continue
        rows.append(dict(
            model=d["model"],
            basin=d["basin"],
            n_segments=int(d.get("n_pi_segments", 0)),
            pi_mean=float(d.get("pi_mean", float("nan"))),
            pi_sd=float(d.get("pi_sd", float("nan"))),
            pi_p95=float(d.get("pi_p95_threshold", float("nan"))),
            mk_tau=float(d["pi_mk_tau"]),
            mk_p=float(d["pi_mk_p"]),
            gate_passed=bool(d.get("stationarity_gate_passed",
                                     d["pi_mk_p"] > ALPHA)),
            first_exit_year=d.get("first_exit_year"),
        ))
    return rows


def _write_audit_json(rows: list[dict], basin: str) -> Path:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIT_DIR / f"stationarity_{basin}.json"
    n_pass = sum(r["gate_passed"] for r in rows)
    summary = dict(
        basin=basin, alpha=ALPHA, n_models=len(rows),
        n_pass=n_pass, n_fail=len(rows) - n_pass,
        pass_rate=(n_pass / len(rows)) if rows else 0.0,
        models_failing_gate=[r["model"] for r in rows
                              if not r["gate_passed"]],
        per_model=rows,
    )
    out.write_text(json.dumps(summary, indent=2))
    return out


def _plot_audit(rows: list[dict], basin: str) -> Path:
    rows_sorted = sorted(rows, key=lambda r: r["mk_tau"])
    models = [r["model"] for r in rows_sorted]
    taus = np.asarray([r["mk_tau"] for r in rows_sorted])
    ps = np.asarray([r["mk_p"] for r in rows_sorted])
    passed = np.asarray([r["gate_passed"] for r in rows_sorted])

    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 0.18 * len(models) + 0.6))
    y = np.arange(len(models))
    ax.axvline(0.0, color="0.4", lw=0.5)
    ax.scatter(taus[passed], y[passed], s=14, c="C0",
                marker="o", label="gate pass", zorder=3)
    ax.scatter(taus[~passed], y[~passed], s=18, c="C3",
                marker="s", label="gate fail", zorder=3)
    for yi, p in zip(y, ps):
        if not np.isfinite(p): continue
        ax.text(0.95, yi, f"p={p:.2f}",
                  transform=ax.get_yaxis_transform(),
                  ha="right", va="center", fontsize=5, color="0.4")
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=5)
    ax.set_xlabel(r"Mann-Kendall $\tau$ on piControl Q segments")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.5, len(models) - 0.5)
    ax.legend(loc="lower left", frameon=False, fontsize=6)
    ax.text(0.02, 0.98, f"({basin})", transform=ax.transAxes,
              ha="left", va="top", fontsize=6, color="0.3")

    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    out = MANUSCRIPT_FIGS / f"fig_stationarity_audit_{basin}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    args = ap.parse_args()

    rows = _load_audit(args.basin)
    if not rows:
        print(f"no bulk JSONs found for basin={args.basin}")
        return
    j = _write_audit_json(rows, args.basin)
    pdf = _plot_audit(rows, args.basin)

    n_pass = sum(r["gate_passed"] for r in rows)
    print(f"basin={args.basin}  models={len(rows)}  "
          f"pass={n_pass}  fail={len(rows) - n_pass}")
    fail = [r["model"] for r in rows if not r["gate_passed"]]
    if fail:
        print(f"  failing-gate: {', '.join(fail)}")
    print(f"wrote {j}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
