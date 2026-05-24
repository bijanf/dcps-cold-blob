"""past1000 forcing attribution: a millennium of single-forcing Q.

Two-panel SI figure showing that no past1000 forcing combination
(natural-only, solar-only, volcanic-only, anthropogenic-only,
GHG-only, land-use-only, orbital-only, all-forcings) pulls the
Atlantic Quiescence Index out of the CMIP6 piControl corridor.
Only the industrial-era CO2 trajectory does, supporting the claim
that the corridor exit is anthropogenic-CO2-specific rather than
sensitivity to forcing per se.

  (a) Per-model Q time-series 850-1849 CE, one coloured line per
      structurally distinct PMIP3 past1000 model (MPI-ESM-P,
      bcc-csm1-1, CCSM4, MIROC-ESM, GISS-E2-R p121 all-forcings),
      with the modern CMIP6 piControl 10-90 % envelope as a grey
      band for reference.

  (b) GISS-E2-R single-forcing decomposition (Schmidt 2014 GMD):
      one violin per p-number Q distribution.  The piControl 10-90 %
      band and the SSP5-8.5 2070-2099 IQR strip are overlaid so the
      reader can see at a glance which forcings keep Q inside the
      corridor and which (only industrial-era CO2 in CMIP6) do not.

Inputs:
  dcps/cache/cmip5_past1000/*.json   (PMIP3 forced past1000)
  dcps/cache/holocene_exit/bulk/*.json (piControl + far-future)

Output: manuscript/figs/fig_past1000_forcing_attribution_<basin>.pdf
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

PAST1000_DIR = CACHE_DIR / "cmip5_past1000"
BULK_DIR = CACHE_DIR / "holocene_exit" / "bulk"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"
P1K_END = 1849
ALPHA = 0.05

GISS_FORCINGS = {
    "r1i1p121":  "all forcings",
    "r1i1p122":  "natural only",
    "r1i1p1221": "natural only*",
    "r1i1p123":  "anthropogenic only",
    "r1i1p124":  "solar only",
    "r1i1p125":  "volcanic only",
    "r1i1p126":  "orbital only",
    "r1i1p127":  "GHG only",
    "r1i1p128":  "land-use only",
}
GISS_ORDER = [
    "r1i1p121", "r1i1p122", "r1i1p1221", "r1i1p123",
    "r1i1p124", "r1i1p125", "r1i1p126", "r1i1p127", "r1i1p128",
]

MODEL_COLOURS = {
    "MPI-ESM-P":   "#4C72B0",
    "bcc-csm1-1":  "#DD8452",
    "CCSM4":       "#55A868",
    "MIROC-ESM":   "#C44E52",
    "GISS-E2-R":   "#8172B2",
}


def _truncate_to_p1k(qs, cs):
    out = []
    for q, c in zip(qs, cs):
        if q is None or not np.isfinite(q): continue
        if c is None or c > P1K_END:        continue
        out.append((c, q))
    out.sort()
    return out


def _load_past1000(basin):
    """Returns dict by (model, member) -> list of (centre, Q)."""
    out = {}
    for p in sorted(PAST1000_DIR.glob(f"*_{basin}.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        model = d.get("model")
        member = d.get("member")
        if not model or not member: continue
        out[(model, member)] = _truncate_to_p1k(
            d.get("Q") or [], d.get("year_centres") or [])
    return out


def _pooled_picontrol(basin):
    pooled = []
    for p in sorted(BULK_DIR.glob(f"*_{basin}.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "pi_mk_p" not in d: continue
        if not d.get("stationarity_gate_passed",
                       d["pi_mk_p"] > ALPHA): continue
        pi = d.get("pi_Q") or []
        pooled.extend([q for q in pi if q is not None
                         and np.isfinite(q)])
    return np.asarray(pooled, dtype=float)


def _pooled_window(basin, lo, hi):
    pooled = []
    for p in sorted(BULK_DIR.glob(f"*_{basin}.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        centres = d.get("hist_centres") or []
        Qs      = d.get("hist_Q") or []
        for c, q in zip(centres, Qs):
            if q is None or not np.isfinite(q): continue
            if lo <= c <= hi:
                pooled.append(q)
    return np.asarray(pooled, dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    args = ap.parse_args()

    by_member = _load_past1000(args.basin)
    if not by_member:
        print(f"no past1000 caches for {args.basin}"); return

    pooled_pi = _pooled_picontrol(args.basin)
    pooled_hist1850 = _pooled_window(args.basin, 1850, 1900)
    pi_p10, pi_med, pi_p90 = (np.percentile(pooled_pi, q)
                                for q in (10, 50, 90))
    print(f"basin={args.basin}  N_past1000_members={len(by_member)}  "
          f"piControl N={pooled_pi.size}  "
          f"piControl 10-90 % = ({pi_p10:+.3f}, {pi_p90:+.3f})  "
          f"median = {pi_med:+.3f}")
    if pooled_hist1850.size:
        h_med = np.median(pooled_hist1850)
        print(f"  historical 1850-1900  N={pooled_hist1850.size}  "
              f"median={h_med:+.3f}")

    fig = plt.figure(figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.40))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0],
                            wspace=0.28, left=0.07, right=0.985,
                            bottom=0.18, top=0.92)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ---------- panel (a) ----------
    # 5 structurally distinct PMIP3 past1000 models.  GISS shown as
    # p121 (all forcings) so it's directly comparable to MPI/bcc/CCSM4
    # MIROC which are likewise all-forcings past1000.
    primary_targets = [
        ("MPI-ESM-P",  "r1i1p1"),
        ("bcc-csm1-1", "r1i1p1"),
        ("CCSM4",      "r1i1p1"),
        ("MIROC-ESM",  "r1i1p1"),
        ("GISS-E2-R",  "r1i1p121"),
    ]
    axA.axhspan(pi_p10, pi_p90, color="0.6", alpha=0.18, zorder=1,
                  label=f"piControl 10-90 % (N={pooled_pi.size})")
    axA.axhline(pi_med, color="0.4", lw=0.7, zorder=2)
    have_any = False
    for (model, member) in primary_targets:
        ts = by_member.get((model, member))
        if not ts:
            print(f"  panel-a skip {model}/{member}: no data")
            continue
        cs, qs = zip(*ts)
        c_arr = np.asarray(cs); q_arr = np.asarray(qs)
        col = MODEL_COLOURS.get(model, "0.2")
        label = f"{model} ({member}, all forcings)" if model == "GISS-E2-R" \
                else f"{model}"
        axA.plot(c_arr, q_arr, color=col, lw=0.7, alpha=0.85,
                  zorder=3, label=label)
        axA.scatter(c_arr, q_arr, s=4, color=col, alpha=0.7,
                     linewidths=0, zorder=4)
        have_any = True
    if not have_any:
        print("  panel-a has no primary models, aborting"); return
    axA.set_xlabel("window centre (CE)")
    axA.set_ylabel(r"$Q$ (Atlantic, 30-yr window)")
    axA.set_xlim(840, 1860)
    axA.set_ylim(-0.10, 0.60)
    axA.legend(loc="lower right", fontsize=4.5, frameon=False,
                ncol=1, handlelength=1.3)
    axA.text(0.02, 0.98, "(a)", transform=axA.transAxes,
              ha="left", va="top", fontsize=8, fontweight="bold")

    # ---------- panel (b) ----------
    # GISS-E2-R forcing attribution -- one violin per p-number.
    data, labels = [], []
    for member in GISS_ORDER:
        ts = by_member.get(("GISS-E2-R", member))
        if not ts:
            continue
        qs = [q for (_, q) in ts]
        if len(qs) < 5:
            continue
        data.append(np.asarray(qs))
        labels.append(GISS_FORCINGS[member])
    xpos = np.arange(1, len(data) + 1)
    parts = axB.violinplot(data, positions=xpos, widths=0.78,
                             showmeans=False, showmedians=True,
                             showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(MODEL_COLOURS["GISS-E2-R"])
        body.set_edgecolor("0.2")
        body.set_alpha(0.55)
        body.set_linewidth(0.5)
    if "cmedians" in parts:
        parts["cmedians"].set_color("0.15")
        parts["cmedians"].set_linewidth(0.8)
    # piControl band + median line
    axB.axhspan(pi_p10, pi_p90, color="0.6", alpha=0.18, zorder=1,
                  label="piControl 10-90 %")
    axB.axhline(pi_med, color="0.4", lw=0.7, zorder=2)
    # Historical 1850-1900 anchor as a dashed reference line: the
    # industrial-era start of the corridor, not a post-exit state.
    if pooled_hist1850.size:
        h_med = np.median(pooled_hist1850)
        axB.axhline(h_med, color="C3", lw=0.7, ls="--", zorder=2,
                      label=(f"historical 1850-1900 median "
                             f"({h_med:+.2f})"))
    axB.set_xticks(xpos)
    axB.set_xticklabels(labels, rotation=35, ha="right",
                          fontsize=5)
    axB.set_xlim(0.4, len(data) + 0.6)
    axB.set_ylim(-0.10, 0.60)
    axB.set_ylabel(r"$Q$ (Atlantic, 30-yr window)")
    axB.legend(loc="lower right", fontsize=4.5, frameon=False)
    axB.text(0.02, 0.98, "(b)", transform=axB.transAxes,
              ha="left", va="top", fontsize=8, fontweight="bold")
    axB.set_title("GISS-E2-R past1000 forcing decomposition",
                   fontsize=6, pad=2)

    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    out_pdf = MANUSCRIPT_FIGS / \
        f"fig_past1000_forcing_attribution_{args.basin}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
