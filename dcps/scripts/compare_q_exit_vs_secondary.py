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

    # ---- figure (3-lane horizontal strip; one lane per quantity) ----
    # Native DOUBLE-column width matches Figure 5 panels a-c so all four
    # subfigures scale uniformly when LaTeX includes them at 0.48\linewidth;
    # aspect ~1.55 -> height = DOUBLE_COL_IN / 1.55.
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN / 1.55))
    # Uniform circles -- shape carries no extra information; lane name and
    # colour discriminate the three quantities.
    LANES = [
        ("$Q$-exit",                              qs, "o", "C0"),
        (r"EWS-onset (AC1$>0.5$)",                ws, "o", "C3"),
        (r"EKE-departure ($\mu+2\sigma$)",        es, "o", "C2"),
    ]
    rng = np.random.default_rng(2026)
    handles = []
    for j, (lbl, arr, mk, col) in enumerate(LANES):
        v = arr[np.isfinite(arr)]
        # jittered 1-D scatter inside the lane (jitter ~ 0.18 of lane height)
        yj = j + (rng.uniform(-1, 1, size=v.size) * 0.18)
        h = ax.scatter(v, yj, marker=mk, s=22, color=col,
                       edgecolor="0.2", linewidth=0.35, zorder=3, label=lbl)
        handles.append(h)
        if v.size:
            med = float(np.nanmedian(v))
            p25 = float(np.nanpercentile(v, 25))
            p75 = float(np.nanpercentile(v, 75))
            # IQR whisker (thin) and ensemble-median tick (thick)
            ax.hlines(j, p25, p75, colors="0.35", linewidth=1.2,
                      alpha=0.85, zorder=2)
            ax.vlines(med, j - 0.32, j + 0.32, colors="0.15",
                      linewidth=1.8, zorder=4)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["$Q$-exit", "EWS-onset", "EKE-departure"], fontsize=6)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel("year (window centre)")
    ax.set_xlim(1850, 2100)
    ax.set_ylim(-0.55, 2.55)
    ax.invert_yaxis()  # Q-exit on top -> EKE-departure on bottom (reading order)
    ax.grid(axis="x", color="0.85", linewidth=0.3, linestyle=":")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # Panel letter via set_title (sits above the axes, not inside data).
    ax.set_title("d", loc="left", fontweight="bold", fontsize=10, pad=2)
    # Legend OUTSIDE the data axes -- single foot strip, no frame.
    fig.legend(handles, [h.get_label() for h in handles],
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, frameon=False, fontsize=6,
               handletextpad=0.4, columnspacing=1.4)
    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    out_pdf = (MANUSCRIPT_FIGS
                / f"fig_q_exit_vs_secondary_{args.basin}.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
