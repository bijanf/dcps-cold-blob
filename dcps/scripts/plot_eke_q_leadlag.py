"""Lead-lag analysis of EKE and Q in the joint phase plane.

Every previous joint figure paired (EKE(t), Q(t)) at the same window
centre.  The Quiescence-Signature law predicts EKE forces, Q responds
- a contemporaneous pairing is therefore the wrong dynamical object.

Three panels:
  (a) Per-model cross-correlation rho(EKE(t), Q(t+lag)) on the
      hist+future timeseries (1850-2099) for lags in {-30, ..., +60}
      yr.  Each model line in light grey, ensemble median in solid,
      ensemble 25-75 % band shaded.  A peak at positive lag means
      "today's EKE predicts future Q" - EKE leads.

  (b) Joint p95-density-contour exit fraction for the SSP5-8.5
      2070-2099 epoch as a function of the pairing lag Delta.  At
      each Delta we pair (rel-EKE(t-Delta), Q(t)) and rebuild the
      joint corridor from the equivalently-lagged piControl pairs.
      The Delta that maximises late-future exit is the data-driven
      EKE-Q response time.

  (c) Paired strict-exit-year scatter for the small subset of models
      (typically 5) that satisfy the stricter-than-conventional exit
      rule on BOTH indicators.  1:1 line and OLS fit shown.  Sparse,
      so reported as a complementary check rather than a primary
      lag estimator.

Inputs: same paired (Q, EKE) cache structure used by
plot_q_eke_joint_corridor.py + the strict-exit audit JSONs.

Output: manuscript/figs/fig_eke_q_leadlag_<basin>.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import gaussian_kde, pearsonr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style, DOUBLE_COL_IN
apply_nature_style()

BULK_DIR = CACHE_DIR / "holocene_exit" / "bulk"
EKE_TS_DIR = CACHE_DIR / "eke_timeseries"
AUDIT_DIR = CACHE_DIR / "holocene_exit" / "audit"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

ALPHA = 0.05
STRIDE_YR = 10
LATE = (2070, 2099)


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


def _eke_caches(basin):
    return {json.loads(p.read_text())["model"]: json.loads(p.read_text())
             for p in sorted(EKE_TS_DIR.glob(f"*_{basin}_eke_ts.json"))}


def _paired_series(bulk, eke):
    """Return (centres, rel_eke, Q) aligned by window centre over
    the hist+future timeseries."""
    bc = bulk.get("hist_centres") or []
    bq = bulk.get("hist_Q") or []
    ec = eke.get("hist_centres") or []
    ev = eke.get("hist_eke") or []
    mu = eke.get("pi_eke_mean") or float("nan")
    if not (np.isfinite(mu) and mu > 0):
        return np.array([]), np.array([]), np.array([])
    eke_by_c = {c: v for c, v in zip(ec, ev)}
    pairs = []
    for c, q in zip(bc, bq):
        v = eke_by_c.get(c)
        if (v is None or q is None
            or not np.isfinite(v) or not np.isfinite(q)):
            continue
        pairs.append((c, v / mu, q))
    pairs.sort()
    cs = np.asarray([p[0] for p in pairs], dtype=int)
    es = np.asarray([p[1] for p in pairs], dtype=float)
    qs = np.asarray([p[2] for p in pairs], dtype=float)
    return cs, es, qs


def _pi_paired(bulk, eke):
    bs = bulk.get("pi_starts") or []
    bq = bulk.get("pi_Q") or []
    es_ = eke.get("pi_starts") or []
    ev = eke.get("pi_eke") or []
    mu = eke.get("pi_eke_mean") or float("nan")
    if not (np.isfinite(mu) and mu > 0):
        return np.array([]), np.array([])
    eke_by_s = {s: v for s, v in zip(es_, ev)}
    out_e, out_q = [], []
    for s, q in zip(bs, bq):
        v = eke_by_s.get(s)
        if (v is None or q is None
            or not np.isfinite(v) or not np.isfinite(q)):
            continue
        out_e.append(v / mu); out_q.append(q)
    return np.asarray(out_e), np.asarray(out_q)


def _lagged_xcorr(eke, q, lags):
    """ rho( EKE(t), Q(t+lag) ).  lag in stride units (integer)."""
    n = len(eke)
    out = []
    for L in lags:
        if L >= 0:
            x = eke[:n - L] if L > 0 else eke
            y = q[L:] if L > 0 else q
        else:
            x = eke[-L:]
            y = q[:n + L]
        if len(x) < 3:
            out.append(np.nan); continue
        try:
            r, _ = pearsonr(x, y)
        except Exception:
            r = np.nan
        out.append(r)
    return np.asarray(out)


def _level_for_quantile(kde, points, q):
    dens = kde(points.T)
    ds = np.sort(dens)
    n = len(ds)
    idx = int(np.floor((1.0 - q) * n))
    return ds[max(0, min(idx, n - 1))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    args = ap.parse_args()

    admit = _admitted_models(args.basin)
    eke   = _eke_caches(args.basin)
    common = sorted(set(admit) & set(eke))

    # ---- panel (a) per-model cross-correlation ----------------------
    lags_yr = np.arange(-30, 61, 10)
    lags_idx = lags_yr // STRIDE_YR
    model_xcorr = {}
    for m in common:
        c, e, q = _paired_series(admit[m], eke[m])
        if len(c) < 6:
            continue
        # detrend each series (linear) so we measure coherence rather
        # than shared monotonic forcing
        def _detr(x):
            t = np.arange(len(x))
            p = np.polyfit(t, x, 1)
            return x - np.polyval(p, t)
        model_xcorr[m] = _lagged_xcorr(_detr(e), _detr(q), lags_idx)
    if not model_xcorr:
        print("no model has a usable hist+future paired series"); return
    M = np.vstack(list(model_xcorr.values()))
    print(f"cross-correlation: N models = {len(model_xcorr)}, "
          f"lags = {lags_yr.tolist()} yr")
    median_xc = np.nanmedian(M, axis=0)
    p25 = np.nanpercentile(M, 25, axis=0)
    p75 = np.nanpercentile(M, 75, axis=0)
    peak_lag = int(lags_yr[np.nanargmax(median_xc)])
    print(f"  ensemble-median peak at lag = {peak_lag} yr "
          f"(rho_peak = {np.nanmax(median_xc):+.2f})")

    # ---- panel (b) lagged joint p95 late-future exit fraction -------
    sweep_yr = lags_yr.copy()
    sweep_idx = sweep_yr // STRIDE_YR
    exit_frac = []
    for L in sweep_idx:
        pi_E, pi_Q = [], []
        late_E, late_Q = [], []
        for m in common:
            c, e, q = _paired_series(admit[m], eke[m])
            if len(c) < 3:
                continue
            # build lagged pairs over the full hist+future series
            if L >= 0:
                xs = e[:len(c) - L] if L > 0 else e
                ys = q[L:] if L > 0 else q
                cs2 = c[L:] if L > 0 else c   # year associated with Q
            else:
                xs = e[-L:]
                ys = q[:len(c) + L]
                cs2 = c[:len(c) + L]
            for x, y, cc in zip(xs, ys, cs2):
                if LATE[0] <= cc <= LATE[1]:
                    late_E.append(x); late_Q.append(y)
            # piControl lagged pairs (already aligned at lag 0; lag
            # affects only the hist+future axis, since piControl is
            # treated as a stationary cloud)
            pe, pq = _pi_paired(admit[m], eke[m])
            pi_E.extend(pe); pi_Q.extend(pq)
        if len(late_E) < 5 or len(pi_E) < 10:
            exit_frac.append(np.nan); continue
        pi = np.column_stack([pi_E, pi_Q])
        late = np.column_stack([late_E, late_Q])
        try:
            k = gaussian_kde(pi.T, bw_method=0.30)
            lvl = _level_for_quantile(k, pi, 0.95)
            frac = float((k(late.T) < lvl).mean())
        except Exception:
            frac = np.nan
        exit_frac.append(frac)
    exit_frac = np.asarray(exit_frac)
    sweep_peak_lag = int(sweep_yr[np.nanargmax(exit_frac)])
    print(f"  joint late-future exit fraction sweep:")
    for L, f in zip(sweep_yr, exit_frac):
        print(f"    lag={L:+3d} yr  exit={100*f:5.1f}%")
    print(f"  peak exit fraction at lag = {sweep_peak_lag} yr "
          f"(frac = {100*np.nanmax(exit_frac):.1f}%)")

    # ---- panel (c) paired strict-exit-year scatter ------------------
    a = json.loads((AUDIT_DIR / f"significant_exit_{args.basin}.json").read_text())
    b = json.loads((AUDIT_DIR / f"significant_eke_exit_{args.basin}.json").read_text())
    qx = {r["model"]: r["significant_exit"] for r in a["per_model"]
           if r["significant_exit"]}
    ex = {r["model"]: r["significant_exit"] for r in b["per_model"]
           if r["significant_exit"]}
    pairs = sorted((m, qx[m], ex[m]) for m in set(qx) & set(ex))
    paired_lags = [q - e for _, q, e in pairs]
    print(f"  paired strict-exit models: N={len(pairs)}, "
          f"median lag (Q - EKE) = "
          f"{int(np.median(paired_lags)) if paired_lags else 'n/a'} yr")

    # ---- figure -----------------------------------------------------
    fig = plt.figure(figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.36))
    gs = GridSpec(1, 3, width_ratios=[1.1, 1.0, 1.0], wspace=0.55,
                   left=0.06, right=0.985, top=0.92, bottom=0.22,
                   figure=fig)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # (a)
    for v in model_xcorr.values():
        ax_a.plot(lags_yr, v, color="0.7", lw=0.4, alpha=0.6)
    ax_a.fill_between(lags_yr, p25, p75, color="C0", alpha=0.20,
                       label="ensemble 25-75 %")
    ax_a.plot(lags_yr, median_xc, color="C0", lw=1.2,
                label="ensemble median")
    ax_a.axvline(peak_lag, color="C3", lw=0.6, ls="--",
                  label=f"peak @ {peak_lag} yr")
    ax_a.axhline(0, color="0.6", lw=0.3)
    ax_a.axvline(0, color="0.6", lw=0.3)
    ax_a.set_xlabel("lag $\\Delta$ (yr)   $\\rho(\\mathrm{EKE}(t),\\,Q(t+\\Delta))$",
                      fontsize=11)
    ax_a.set_ylabel("Pearson correlation", fontsize=11)
    ax_a.set_xlim(lags_yr.min(), lags_yr.max())
    ax_a.set_ylim(-1, 1)
    ax_a.legend(loc="lower left", fontsize=5, frameon=False)
    ax_a.text(-0.18, 1.02, "(a)", transform=ax_a.transAxes,
                ha="left", va="bottom", fontsize=8, fontweight="bold")

    # (b)
    ax_b.plot(sweep_yr, 100 * exit_frac, "o-", color="C3", lw=0.9,
                ms=3, label="2070-2099 joint exit")
    ax_b.axvline(sweep_peak_lag, color="C3", lw=0.6, ls="--",
                  label=f"peak @ {sweep_peak_lag} yr")
    ax_b.axhline(5, color="0.6", lw=0.3, ls=":")
    ax_b.set_xlabel("EKE-Q pairing lag $\\Delta$ (yr)", fontsize=11)
    ax_b.set_ylabel("joint $p_{95}$ exit fraction (%)", fontsize=11)
    ax_b.set_xlim(sweep_yr.min(), sweep_yr.max())
    ax_b.set_ylim(0, 100)
    ax_b.legend(loc="lower right", fontsize=5, frameon=False)
    ax_b.text(-0.18, 1.02, "(b)", transform=ax_b.transAxes,
                ha="left", va="bottom", fontsize=8, fontweight="bold")

    # (c)
    if pairs:
        markers = ["o", "s", "^", "D", "v", "<", ">", "P", "X", "*"]
        for i, (m, q, e) in enumerate(pairs):
            ax_c.scatter(e, q, s=22, c="C0", edgecolor="0.2",
                          linewidth=0.4, zorder=3,
                          marker=markers[i % len(markers)],
                          label=m)
        lo = min(min([e for _, _, e in pairs]),
                  min([q for _, q, _ in pairs])) - 25
        hi = max(max([e for _, _, e in pairs]),
                  max([q for _, q, _ in pairs])) + 25
        ax_c.plot([lo, hi], [lo, hi], color="0.5", lw=0.5, ls="--",
                    label="1:1")
        med_lag = int(np.median(paired_lags))
        ax_c.plot([lo, hi], [lo + med_lag, hi + med_lag],
                    color="C3", lw=0.7,
                    label=f"median lag = {med_lag:+d} yr")
        ax_c.set_xlim(lo, hi); ax_c.set_ylim(lo, hi)
    ax_c.set_xlabel("EKE strict-exit year", fontsize=11)
    ax_c.set_ylabel("Q strict-exit year", fontsize=11)
    ax_c.legend(loc="lower right", fontsize=4, frameon=False,
                  handlelength=0.8, borderpad=0.2, labelspacing=0.25)
    ax_c.text(-0.18, 1.02, "(c)", transform=ax_c.transAxes,
                ha="left", va="bottom", fontsize=8, fontweight="bold")

    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    out_pdf = MANUSCRIPT_FIGS / f"fig_eke_q_leadlag_{args.basin}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
