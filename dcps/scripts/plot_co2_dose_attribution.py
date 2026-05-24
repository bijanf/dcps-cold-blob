"""CO2-dose attribution of Q and EKE corridor exits.

For the two idealised CO2-only experiments (1pctCO2 and abrupt-4xCO2)
this script extracts each model's first strict corridor exit in Q and
in basin-mean EKE, then maps the 1pctCO2 exit year to the prevailing
CO2 concentration assuming the experiment design

    CO2(y) = 284.32 ppm x 1.01 ** y

(pre-industrial concentration ramped by 1 % per year).

For abrupt-4xCO2 the CO2 forcing is constant at 4x PI from year 0,
so we report time-to-exit rather than a CO2 dose.

Exit rule (matches recompute_significant_exit.py): a window centred at
year y counts as a strict exit if the indicator exceeds the model's
own piControl p95 by at least one piControl standard deviation AND
remains above the p95 threshold for at least three consecutive windows.

Output: manuscript/figs/fig_co2_dose_attribution_<basin>.pdf
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style, DOUBLE_COL_IN
apply_nature_style()

BULK_DIR    = CACHE_DIR / "holocene_exit" / "bulk"
EKE_TS_DIR  = CACHE_DIR / "eke_timeseries"
Q_CO2_DIR   = CACHE_DIR / "holocene_exit" / "auto"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

PI_CO2_PPM = 284.32
EPS_SIGMA  = 1.0
MIN_PERSIST = 3
ALPHA = 0.05


def _admitted_models(basin):
    out = {}
    for p in sorted(BULK_DIR.glob(f"*_{basin}.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "pi_mk_p" not in d: continue
        if not d.get("stationarity_gate_passed",
                       d["pi_mk_p"] > ALPHA): continue
        out[d["model"]] = d
    return out


def _strict_exit_index(values, threshold, sd):
    """Return the index of the first window satisfying the strict
    rule (>= threshold + eps*sd AND persist for min_persist windows
    above threshold)."""
    arr = np.asarray(values, dtype=float)
    above_strict = arr > (threshold + EPS_SIGMA * sd)
    above_thresh = arr > threshold
    for i in range(arr.size - MIN_PERSIST + 1):
        if not above_strict[i]: continue
        if np.all(above_thresh[i: i + MIN_PERSIST]):
            return i
    return None


def _q_co2_records(experiment, basin, admitted):
    """For each admitted model with a Q cache for the given CO2
    experiment, compute the strict-exit experiment year."""
    rec = []
    for p in sorted(Q_CO2_DIR.glob(
            f"pangeo_*_{experiment}_{basin}.json")):
        d = json.loads(p.read_text())
        m = d.get("target", {}).get("model")
        if m not in admitted:
            continue
        b = admitted[m]
        try:
            p95 = float(b["pi_p95_threshold"])
            sd = float(b.get("pi_sd") or 0.0)
        except Exception:
            continue
        qs = d.get("Q") or []
        cs = d.get("year_centres") or []
        yr_start = (d.get("year_range") or [None])[0]
        if not qs or not cs or yr_start is None:
            continue
        idx = _strict_exit_index(qs, p95, sd)
        rec.append(dict(
            model=m,
            window_centre=int(cs[idx]) if idx is not None else None,
            year_since_branch=(int(cs[idx]) - int(yr_start))
                                  if idx is not None else None,
            n_windows=len(qs),
        ))
    return rec


def _eke_co2_records(experiment, basin, admitted):
    """For each admitted model with an EKE-CO2 cache, compute strict-
    exit experiment year using historical pi_eke_mean as normaliser
    and a per-model piControl p95 on rel-EKE."""
    rec = []
    for p in sorted(EKE_TS_DIR.glob(
            f"*_{basin}_{experiment}_eke_ts.json")):
        d = json.loads(p.read_text())
        m = d.get("model")
        if m not in admitted:
            continue
        # locate the historical EKE cache to fetch pi-EKE reference
        hist_p = EKE_TS_DIR / f"{m}_{basin}_eke_ts.json"
        if not hist_p.exists(): continue
        h = json.loads(hist_p.read_text())
        pi_eke = h.get("pi_eke") or []
        pi_arr = np.asarray([x for x in pi_eke
                                if x is not None and np.isfinite(x)],
                               dtype=float)
        if pi_arr.size < 4: continue
        mu = h.get("pi_eke_mean") or pi_arr.mean()
        if not (mu and np.isfinite(mu) and mu > 0): continue
        rel_pi = pi_arr / mu
        thr = float(np.percentile(rel_pi, 95))
        sd  = float(rel_pi.std(ddof=0))
        ekes = d.get("eke") or []
        cs   = d.get("centres") or []
        yr_start = (d.get("year_range") or [None])[0]
        if not ekes or not cs or yr_start is None:
            continue
        rel = np.asarray([(e / mu) if (e is not None
                                          and np.isfinite(e))
                              else np.nan for e in ekes], dtype=float)
        # treat NaNs as not above thresh; the rule will skip them
        idx = _strict_exit_index(np.nan_to_num(rel,
                                                  nan=-np.inf), thr, sd)
        rec.append(dict(
            model=m,
            window_centre=int(cs[idx]) if idx is not None else None,
            year_since_branch=(int(cs[idx]) - int(yr_start))
                                  if idx is not None else None,
            n_windows=len(ekes),
        ))
    return rec


def _ppm(y):
    """1pctCO2 CO2 at experiment year y."""
    if y is None: return None
    return PI_CO2_PPM * (1.01 ** y)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    args = ap.parse_args()

    admit = _admitted_models(args.basin)
    q1   = _q_co2_records("1pctCO2",      args.basin, admit)
    q4   = _q_co2_records("abrupt-4xCO2", args.basin, admit)
    e1   = _eke_co2_records("1pctCO2",      args.basin, admit)
    e4   = _eke_co2_records("abrupt-4xCO2", args.basin, admit)
    print(f"basin={args.basin}")
    print(f"  1pctCO2:       Q {len(q1)} models, "
          f"EKE {len(e1)} models")
    print(f"  abrupt-4xCO2:  Q {len(q4)} models, "
          f"EKE {len(e4)} models")

    q1_by = {r["model"]: r for r in q1}
    q4_by = {r["model"]: r for r in q4}
    e1_by = {r["model"]: r for r in e1}
    e4_by = {r["model"]: r for r in e4}

    paired_1pct = []
    for m in sorted(set(q1_by) & set(e1_by)):
        rq = q1_by[m]; re = e1_by[m]
        if (rq["year_since_branch"] is not None
            and re["year_since_branch"] is not None):
            paired_1pct.append((m,
                                  _ppm(rq["year_since_branch"]),
                                  _ppm(re["year_since_branch"])))
    print(f"  paired 1pctCO2 (both Q and EKE strict exit): "
          f"N={len(paired_1pct)}")

    q1_ppm_all = [_ppm(r["year_since_branch"])
                   for r in q1 if r["year_since_branch"] is not None]
    e1_ppm_all = [_ppm(r["year_since_branch"])
                   for r in e1 if r["year_since_branch"] is not None]
    q4_yr_all  = [r["year_since_branch"]
                   for r in q4 if r["year_since_branch"] is not None]
    e4_yr_all  = [r["year_since_branch"]
                   for r in e4 if r["year_since_branch"] is not None]
    print(f"  1pctCO2  Q exits:   N={len(q1_ppm_all)} "
          f"median {np.median(q1_ppm_all):.0f} ppm" if q1_ppm_all else "")
    print(f"  1pctCO2  EKE exits: N={len(e1_ppm_all)} "
          f"median {np.median(e1_ppm_all):.0f} ppm" if e1_ppm_all else "")
    print(f"  abrupt-4xCO2  Q exits:   N={len(q4_yr_all)} "
          f"median {np.median(q4_yr_all):.0f} yr" if q4_yr_all else "")
    print(f"  abrupt-4xCO2  EKE exits: N={len(e4_yr_all)} "
          f"median {np.median(e4_yr_all):.0f} yr" if e4_yr_all else "")

    # ---- figure ------------------------------------------------------
    fig = plt.figure(figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.36),
                       constrained_layout=True)
    gs = GridSpec(1, 3, width_ratios=[1.05, 1.0, 1.0], figure=fig)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # (a) per-model CO2-at-Q-exit vs CO2-at-EKE-exit, 1pctCO2
    if paired_1pct:
        xs = np.array([p[2] for p in paired_1pct])  # EKE-exit ppm
        ys = np.array([p[1] for p in paired_1pct])  # Q-exit ppm
        ax_a.scatter(xs, ys, s=14, c="C0", marker="o",
                       edgecolor="0.2", linewidth=0.4, zorder=3)
        lo = min(min(xs), min(ys)) * 0.9
        hi = max(max(xs), max(ys)) * 1.05
        ax_a.plot([lo, hi], [lo, hi], color="0.5", lw=0.5, ls="--",
                    label="1:1")
        ax_a.set_xlim(lo, hi); ax_a.set_ylim(lo, hi)
    ax_a.axvline(PI_CO2_PPM, color="0.6", lw=0.3, ls=":")
    ax_a.axhline(PI_CO2_PPM, color="0.6", lw=0.3, ls=":")
    ax_a.set_xlabel("CO$_2$ at EKE strict exit  (ppm)",
                       fontsize=6, labelpad=2)
    ax_a.set_ylabel("CO$_2$ at Q strict exit  (ppm)",
                       fontsize=6, labelpad=2)
    ax_a.set_title(f"(a) 1pctCO$_2$  (N={len(paired_1pct)} paired)",
                       fontsize=6, pad=2)
    ax_a.legend(loc="upper left", fontsize=5, frameon=False)

    # (b) histogram of CO2-at-exit, Q vs EKE
    bins = np.linspace(280, 1200, 24)
    if q1_ppm_all:
        ax_b.hist(q1_ppm_all, bins=bins, color="C3", alpha=0.55,
                    edgecolor="white", linewidth=0.4,
                    label=f"Q exits (N={len(q1_ppm_all)})")
    if e1_ppm_all:
        ax_b.hist(e1_ppm_all, bins=bins, color="C0", alpha=0.55,
                    edgecolor="white", linewidth=0.4,
                    label=f"EKE exits (N={len(e1_ppm_all)})")
    ax_b.axvline(PI_CO2_PPM, color="0.4", lw=0.4, ls=":")
    ax_b.axvline(2 * PI_CO2_PPM, color="0.4", lw=0.4, ls=":")
    ax_b.axvline(4 * PI_CO2_PPM, color="0.4", lw=0.4, ls=":")
    ax_b.text(2 * PI_CO2_PPM, ax_b.get_ylim()[1] * 0.9, "2x",
                fontsize=5, ha="center", color="0.4")
    ax_b.text(4 * PI_CO2_PPM, ax_b.get_ylim()[1] * 0.9, "4x",
                fontsize=5, ha="center", color="0.4")
    ax_b.set_xlabel("CO$_2$ at strict exit  (ppm)",
                       fontsize=6, labelpad=2)
    ax_b.set_ylabel("models per 40-ppm bin", fontsize=6, labelpad=2)
    ax_b.set_title("(b) 1pctCO$_2$  CO$_2$-dose distribution",
                       fontsize=6, pad=2)
    ax_b.legend(loc="upper right", fontsize=5, frameon=False)

    # (c) abrupt-4xCO2 time-to-exit
    box_data = []
    box_labels = []
    if q4_yr_all:
        box_data.append(q4_yr_all)
        box_labels.append(f"Q  (N={len(q4_yr_all)})")
    if e4_yr_all:
        box_data.append(e4_yr_all)
        box_labels.append(f"EKE  (N={len(e4_yr_all)})")
    if box_data:
        bp = ax_c.boxplot(box_data, labels=box_labels, widths=0.55,
                            patch_artist=True, showfliers=True)
        for patch, c in zip(bp["boxes"], ["C3", "C0"]):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        for med in bp["medians"]:
            med.set_color("k"); med.set_linewidth(0.8)
    ax_c.set_ylabel("years since 4x CO$_2$ jump",
                       fontsize=6, labelpad=2)
    ax_c.set_title("(c) abrupt-4xCO$_2$  time-to-exit",
                       fontsize=6, pad=2)
    ax_c.tick_params(axis="x", labelsize=5)

    for ax in (ax_a, ax_b, ax_c):
        ax.tick_params(axis="both", labelsize=5)

    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    out_pdf = MANUSCRIPT_FIGS / f"fig_co2_dose_attribution_{args.basin}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
