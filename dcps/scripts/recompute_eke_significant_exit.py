"""EKE analogue of recompute_significant_exit.py.

For every model with an EKE-timeseries cache and that passes the Q-side
stationarity gate, apply the same pre-registered stricter exit rule on
basin-mean EKE:

    Exit at year y iff Q(y) > pi_p95_threshold + eps * pi_sd
    AND Q > pi_p95_threshold persists for >= persist consecutive windows.

Outputs:
  dcps/cache/holocene_exit/audit/significant_eke_exit_<basin>.json
  manuscript/figs/fig_significant_eke_exit_<basin>.pdf
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

ALPHA = 0.05
DEFAULT_EPS = 1.0
DEFAULT_PERSIST = 3
EARLY_END = 1940
LATE_START = 1990


def _admitted_models(basin: str) -> set[str]:
    out = set()
    for p in sorted(BULK_DIR.glob(f"*_{basin}.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "pi_mk_p" not in d:
            continue
        if not d.get("stationarity_gate_passed",
                      d["pi_mk_p"] > ALPHA):
            continue
        out.add(d["model"])
    return out


def _significant_exit(hist_q, centres, threshold, sd,
                       epsilon_sigma, min_persist):
    arr = np.asarray(hist_q, dtype=float)
    cs = np.asarray(centres, dtype=int)
    above_strict = arr > (threshold + epsilon_sigma * sd)
    above_thresh = arr > threshold
    for i in range(arr.size - min_persist + 1):
        if not above_strict[i]:
            continue
        if np.all(above_thresh[i: i + min_persist]):
            return int(cs[i])
    return None


def _permissive_exit(hist_q, centres, threshold):
    arr = np.asarray(hist_q, dtype=float)
    cs = np.asarray(centres, dtype=int)
    mask = arr > threshold
    if mask.any():
        return int(cs[np.where(mask)[0][0]])
    return None


def _cluster(year):
    if year is None: return "no-exit"
    if year <= EARLY_END: return "early"
    if year >= LATE_START: return "late"
    return "gap"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    ap.add_argument("--epsilon-sigma", type=float, default=DEFAULT_EPS)
    ap.add_argument("--min-persist", type=int, default=DEFAULT_PERSIST)
    args = ap.parse_args()

    eps = args.epsilon_sigma
    persist = args.min_persist
    admit = _admitted_models(args.basin)

    rows = []
    for p in sorted(EKE_TS_DIR.glob(f"*_{args.basin}_eke_ts.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        model = d.get("model")
        if model not in admit:
            continue
        pi = [x for x in (d.get("pi_eke") or [])
              if x is not None and np.isfinite(x)]
        if len(pi) < 3:
            continue
        pi_arr = np.asarray(pi, dtype=float)
        p95 = float(np.percentile(pi_arr, 95))
        sd = float(d.get("pi_eke_sd") or pi_arr.std(ddof=0))
        cs = d.get("hist_centres") or []
        qs = d.get("hist_eke") or []
        pairs = [(c, q) for c, q in zip(cs, qs)
                 if q is not None and np.isfinite(q)]
        if not pairs:
            continue
        cs_a = [c for c, _ in pairs]
        qs_a = [q for _, q in pairs]
        sig = _significant_exit(qs_a, cs_a, p95, sd, eps, persist)
        perm = _permissive_exit(qs_a, cs_a, p95)
        rows.append(dict(
            model=model, permissive_exit=perm, significant_exit=sig,
            pi_p95=p95, pi_sd=sd, margin=eps * sd,
            cluster=_cluster(sig),
        ))
    if not rows:
        print(f"no admitted models with EKE timeseries for basin={args.basin}")
        return

    with_sig = [r for r in rows if r["significant_exit"] is not None]
    no_sig = [r for r in rows if r["significant_exit"] is None]
    years = np.array([r["significant_exit"] for r in with_sig], dtype=float)
    nperm = sum(r["permissive_exit"] is not None for r in rows)

    print(f"basin={args.basin}  with-EKE-timeseries={len(rows)}  "
          f"(Q-gate admitted set has {len(admit)})")
    print(f"  permissive EKE exit count:        {nperm}/{len(rows)}")
    print(f"  significant EKE exit count:       {len(with_sig)}/{len(rows)}  "
          f"(margin >= {eps} sigma, persist >= {persist} windows)")
    print(f"  no-significant-exit:              {len(no_sig)}")
    if with_sig:
        print(f"  median significant exit:          {np.median(years):.0f}")
        print(f"  range:                            {int(years.min())} ... {int(years.max())}")
        n_early = sum(r["cluster"] == "early" for r in with_sig)
        n_gap   = sum(r["cluster"] == "gap" for r in with_sig)
        n_late  = sum(r["cluster"] == "late" for r in with_sig)
        print(f"    early (<={EARLY_END}): {n_early}")
        print(f"    gap   ({EARLY_END}-{LATE_START}): {n_gap}")
        print(f"    late  (>={LATE_START}): {n_late}")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = AUDIT_DIR / f"significant_eke_exit_{args.basin}.json"
    out_json.write_text(json.dumps(dict(
        basin=args.basin, epsilon_sigma=eps, min_persist=persist,
        early_end=EARLY_END, late_start=LATE_START,
        n_admitted=len(rows), n_significant=len(with_sig),
        n_no_sig=len(no_sig),
        per_model=rows,
    ), indent=2))
    print(f"wrote {out_json}")

    # single panel; bar plot removed (redundant with year axis)
    fig, axL = plt.subplots(
        1, 1, figsize=(DOUBLE_COL_IN * 0.65, DOUBLE_COL_IN * 0.42),
        constrained_layout=True,
    )
    sorted_rows = sorted(rows,
                          key=lambda r: r["significant_exit"]
                          if r["significant_exit"] else 9999)
    colour = dict(early="C0", gap="C7", late="C3", **{"no-exit": "0.7"})
    marker = dict(early="o", gap="s", late="^", **{"no-exit": "x"})
    n_no  = sum(1 for r in rows if r["significant_exit"] is None)
    n_e   = sum(1 for r in rows if r["cluster"] == "early")
    n_g   = sum(1 for r in rows if r["cluster"] == "gap")
    n_l   = sum(1 for r in rows if r["cluster"] == "late")
    n_per = sum(1 for r in rows if r["permissive_exit"] is not None)
    y = np.arange(len(sorted_rows))
    for i, r in enumerate(sorted_rows):
        if r["permissive_exit"]:
            axL.scatter(r["permissive_exit"], i, s=10, marker="o",
                          facecolor="none", edgecolor="0.5",
                          linewidth=0.5, zorder=2)
        if r["significant_exit"]:
            c = r["cluster"]
            axL.scatter(r["significant_exit"], i, s=18,
                          marker=marker[c], color=colour[c],
                          edgecolor="0.2", linewidth=0.4, zorder=4)
        else:
            axL.scatter(2100, i, s=14, marker="x",
                          color="0.5", zorder=3)
    axL.axvline(EARLY_END, color="0.5", lw=0.4, ls="--")
    axL.axvline(LATE_START, color="0.5", lw=0.4, ls="--")
    axL.set_yticks(y)
    axL.set_yticklabels([r["model"] for r in sorted_rows], fontsize=5)
    axL.set_xlim(1850, 2110)
    axL.set_ylim(-0.5, len(sorted_rows) - 0.5)
    axL.set_xlabel("first EKE-exit year (window centre)", fontsize=6,
                    labelpad=2)

    handles = [
        plt.Line2D([0],[0], marker="o", color="none",
                    markerfacecolor="none", markeredgecolor="0.5",
                    markeredgewidth=0.5, markersize=4,
                    label=f"permissive crossing (N={n_per})"),
        plt.Line2D([0],[0], marker="o", color="none",
                    markerfacecolor=colour["early"],
                    markeredgecolor="0.2", markeredgewidth=0.4,
                    markersize=5,
                    label=f"strict, early (<={EARLY_END}; N={n_e})"),
        plt.Line2D([0],[0], marker="s", color="none",
                    markerfacecolor=colour["gap"],
                    markeredgecolor="0.2", markeredgewidth=0.4,
                    markersize=5,
                    label=f"strict, gap ({EARLY_END}-{LATE_START}; N={n_g})"),
        plt.Line2D([0],[0], marker="^", color="none",
                    markerfacecolor=colour["late"],
                    markeredgecolor="0.2", markeredgewidth=0.4,
                    markersize=5,
                    label=f"strict, late (>={LATE_START}; N={n_l})"),
        plt.Line2D([0],[0], marker="x", color="0.5",
                    markersize=4, linestyle="none",
                    label=f"no strict exit (N={n_no}; plotted at 2100)"),
    ]
    axL.legend(handles=handles, loc="upper left", fontsize=5,
                  bbox_to_anchor=(1.01, 1.0), frameon=False,
                  borderpad=0.2, labelspacing=0.5,
                  handlelength=1.2)

    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    out_pdf = MANUSCRIPT_FIGS / f"fig_significant_eke_exit_{args.basin}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
