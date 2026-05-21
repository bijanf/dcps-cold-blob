"""Cross-validate the Q-exit year against two independent diagnostics.

For each CMIP6 model with a cached bulk Q result, also compute:

  1. Mean-EKE-departure year.  First window-centre where the basin-
     mean |grad SSH|^2 (from compute_eke_timeseries.py) exceeds the
     piControl mean + 2 sigma.

  2. EWS-onset year (lag-1 autocorrelation of the Q series).  Compute
     rolling lag-1 autocorrelation on the historical+ssp585 Q series
     using a 7-window slider; report the first centre year where the
     rolling AC1 crosses 0.5 (a conventional CSD onset proxy).

Outputs:
  dcps/cache/holocene_exit/audit/secondary_<basin>.json
  manuscript/figs/fig_q_exit_vs_secondary_<basin>.pdf

The figure is a small scatter / forest plot comparing the three exit-
year diagnostics per model, with the multi-model median highlighted.
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
from dcps.nature_style import apply_nature_style, DOUBLE_COL_IN
apply_nature_style()

BULK_DIR = CACHE_DIR / "holocene_exit" / "bulk"
EKE_TS_DIR = CACHE_DIR / "eke_timeseries"
AUDIT_DIR = CACHE_DIR / "holocene_exit" / "audit"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

EWS_WINDOW = 7
EWS_THRESHOLD = 0.5
ALPHA = 0.05


def _eke_departure(model: str, basin: str) -> int | None:
    p = EKE_TS_DIR / f"{model}_{basin}_eke_ts.json"
    if not p.exists(): return None
    d = json.loads(p.read_text())
    if not d.get("hist_eke") or d.get("pi_eke_mean") is None:
        return None
    mu = float(d["pi_eke_mean"]); sd = float(d.get("pi_eke_sd") or 0.0)
    if sd <= 0: return None
    threshold = mu + 2.0 * sd
    centres = d["hist_centres"]; hist = d["hist_eke"]
    for c, v in zip(centres, hist):
        if v is not None and np.isfinite(v) and v > threshold:
            return int(c)
    return None


def _ews_onset(hist_Q: list, hist_centres: list) -> int | None:
    arr = np.asarray(hist_Q, dtype=float)
    if arr.size < EWS_WINDOW + 2: return None
    ac1 = np.full(arr.size, np.nan)
    for i in range(EWS_WINDOW, arr.size):
        seg = arr[i - EWS_WINDOW: i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size < 3: continue
        s0 = seg[:-1]; s1 = seg[1:]
        if s0.std(ddof=1) == 0 or s1.std(ddof=1) == 0: continue
        ac1[i] = float(np.corrcoef(s0, s1)[0, 1])
    cross = np.where(np.isfinite(ac1) & (ac1 > EWS_THRESHOLD))[0]
    if cross.size == 0: return None
    return int(hist_centres[cross[0]])


def _load_rows(basin: str) -> list[dict]:
    rows = []
    for p in sorted(BULK_DIR.glob(f"*_{basin}.json")):
        d = json.loads(p.read_text())
        if "pi_mk_p" not in d: continue
        if not d.get("stationarity_gate_passed",
                       d["pi_mk_p"] > ALPHA):
            continue
        rows.append(dict(
            model=d["model"],
            q_exit=d.get("first_exit_year"),
            eke_dep=_eke_departure(d["model"], basin),
            ews_onset=_ews_onset(d.get("hist_Q") or [],
                                    d.get("hist_centres") or []),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    args = ap.parse_args()
    rows = _load_rows(args.basin)
    if not rows:
        print(f"no admitted models for basin={args.basin}")
        return

    qs = np.array([r["q_exit"]    or np.nan for r in rows], dtype=float)
    es = np.array([r["eke_dep"]   or np.nan for r in rows], dtype=float)
    ws = np.array([r["ews_onset"] or np.nan for r in rows], dtype=float)

    print(f"basin={args.basin}  N={len(rows)}")
    print(f"  Q-exit       median: {np.nanmedian(qs):.0f}  "
          f"n_valid={int(np.isfinite(qs).sum())}")
    print(f"  EKE-departure median: {np.nanmedian(es):.0f}  "
          f"n_valid={int(np.isfinite(es).sum())}")
    print(f"  EWS-onset    median: {np.nanmedian(ws):.0f}  "
          f"n_valid={int(np.isfinite(ws).sum())}")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = AUDIT_DIR / f"secondary_{args.basin}.json"
    summary = dict(
        basin=args.basin, n_models=len(rows),
        median_q_exit=float(np.nanmedian(qs)) if np.isfinite(qs).any() else None,
        median_eke_departure=float(np.nanmedian(es)) if np.isfinite(es).any() else None,
        median_ews_onset=float(np.nanmedian(ws)) if np.isfinite(ws).any() else None,
        per_model=rows,
    )
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")

    # ---- figure ----
    sorted_rows = sorted(rows,
                          key=lambda r: r["q_exit"] if r["q_exit"]
                          else 9999)
    y = np.arange(len(sorted_rows))
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN,
                                       0.16 * len(sorted_rows) + 1.0))
    for i, r in enumerate(sorted_rows):
        if r["q_exit"]:
            ax.scatter(r["q_exit"], i, marker="o", s=18,
                          color="C0", edgecolor="0.2", linewidth=0.4,
                          zorder=4,
                          label="Q-exit" if i == 0 else None)
        if r["eke_dep"]:
            ax.scatter(r["eke_dep"], i, marker="^", s=18,
                          color="C2", edgecolor="0.2", linewidth=0.4,
                          zorder=4,
                          label="EKE-departure (mean+2sigma)"
                          if i == 0 else None)
        if r["ews_onset"]:
            ax.scatter(r["ews_onset"], i, marker="s", s=18,
                          color="C3", edgecolor="0.2", linewidth=0.4,
                          zorder=4,
                          label=f"EWS-onset (AC1>{EWS_THRESHOLD})"
                          if i == 0 else None)
    ax.set_yticks(y)
    ax.set_yticklabels([r["model"] for r in sorted_rows], fontsize=5)
    ax.set_xlabel("year (window centre)")
    ax.set_xlim(1850, 2100)
    ax.set_ylim(-0.5, len(sorted_rows) - 0.5)
    ax.axvline(1940, color="0.5", lw=0.4, ls="--")
    ax.axvline(1990, color="0.5", lw=0.4, ls="--")
    ax.legend(loc="lower right", fontsize=6, frameon=False)
    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    out_pdf = (MANUSCRIPT_FIGS
                / f"fig_q_exit_vs_secondary_{args.basin}.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
