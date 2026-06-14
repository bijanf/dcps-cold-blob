"""H3 re-run using the continuous-state KSG transfer-entropy estimator.

This is the methodological follow-up the manuscript recommends in §4.6:
replace binary-discretised TE with the Kraskov-Stoegbauer-Grassberger
estimator (Frenzel-Pompe 2007 conditional MI), and ask whether the
TE-leads-CSD R-tipping precursor signal that was absent under binary
TE re-emerges under continuous-state TE.

Re-runs the same pre-registered protocols on the same data:
  - 13 CMIP6 piControl runs (lagged-correlation rho(TE, var))
  - 16 CMIP6 historical+ssp585 forced trajectories (threshold crossings)
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time as _time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.h3 import _lag1_autocorr
from dcps.te_ksg import ksg_te_bits

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from h3_picontrol import amoc_at_latitude  # noqa: E402
from h3_forced import (
    BASELINE_PERIOD,
    LEAD_YEARS_BAR,
    PERSIST_YEARS,
    SUBPOLAR_LAT,
    SUBTROPICAL_LAT,
    TIPPING_END,
    WEAKEN_FRAC,
    WINDOW_YEARS,
    AGGREGATE_FRAC,
    baseline_stats,
    concat_amoc_trajectory,
    first_persistent_crossing,
)

CMIP6_DIR = Path("/home/bijanf/Documents/AMOC_renalysis/data/cmip6_fullfield")
H3_KSG_DIR = CACHE_DIR / "h3_ksg"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

LAGS_YEARS = [-10, -5, 0, 5, 10, 15]
PER_MODEL_RHO_BAR = -0.2


def sliding_diag_ksg(amoc_st: np.ndarray, amoc_sp: np.ndarray,
                       years: np.ndarray, window: int = WINDOW_YEARS):
    n = len(years)
    centres, te_v, var_v, ac_v = [], [], [], []
    for s in range(0, n - window + 1):
        e = s + window
        x_st = amoc_st[s:e]; x_sp = amoc_sp[s:e]
        if not (np.isfinite(x_st).all() and np.isfinite(x_sp).all()):
            te_v.append(np.nan); var_v.append(np.nan); ac_v.append(np.nan)
        else:
            te_v.append(ksg_te_bits(x_st, x_sp, k=1, ell=1, k_nn=4))
            var_v.append(float(np.var(x_sp)))
            ac_v.append(_lag1_autocorr(x_sp))
        centres.append(years[s + window // 2])
    return (np.array(centres), np.array(te_v), np.array(var_v), np.array(ac_v))


def lag_correlation(te: np.ndarray, var: np.ndarray, lag_years: int) -> float:
    finite = np.isfinite(te) & np.isfinite(var)
    if finite.sum() < 30:
        return float("nan")
    if lag_years == 0:
        a, b = te[finite], var[finite]
    elif lag_years > 0:
        a, b = te[:-lag_years], var[lag_years:]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
    else:
        L = -lag_years
        a, b = te[L:], var[:-L]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
    if a.size < 30 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---- piControl test (KSG) ------------------------------------------------

def picontrol_ksg() -> list[dict]:
    files = sorted(glob.glob(str(CMIP6_DIR / "*_piControl_vo_zonal.nc")))
    print(f"\n=== KSG H3 on piControl ({len(files)} models) ===")
    rows = []
    for f in files:
        name = os.path.basename(f).split("_piControl")[0]
        t0 = _time.time()
        yr1, st = amoc_at_latitude(f, SUBTROPICAL_LAT)
        yr2, sp = amoc_at_latitude(f, SUBPOLAR_LAT)
        n = min(yr1.size, yr2.size)
        if n < WINDOW_YEARS + 5:
            print(f"  {name}: too short -- skip"); continue
        yr = yr1[:n]; st = st[:n]; sp = sp[:n]
        centres, te, var, ac = sliding_diag_ksg(st, sp, yr)
        rhos = {tau: lag_correlation(te, var, tau) for tau in LAGS_YEARS}
        finite = {k: v for k, v in rhos.items() if np.isfinite(v)}
        if not finite:
            rows.append({"model": name, "n_years": int(n), "rhos": rhos,
                         "argmin_tau": None, "passed": False})
            continue
        argmin = min(finite, key=lambda k: finite[k])
        rho_at = finite[argmin]
        passed = (argmin > 0) and (rho_at <= PER_MODEL_RHO_BAR)
        print(f"  {name:<22}: {n} yr, argmin_tau={argmin:+d}, rho={rho_at:+.2f} "
              f"{'pass' if passed else 'fail'}  ({_time.time()-t0:.1f}s)")
        rows.append({"model": name, "n_years": int(n), "rhos": rhos,
                     "argmin_tau": argmin, "rho_at_argmin": rho_at,
                     "passed": passed})
    return rows


# ---- Forced ssp585 test (KSG) -------------------------------------------

def forced_ksg(scenario: str = "ssp585") -> list[dict]:
    models = sorted({os.path.basename(f).split("_historical")[0]
                     for f in glob.glob(str(CMIP6_DIR / "*_historical_vo_zonal.nc"))})
    print(f"\n=== KSG H3 on forced ({scenario}, {len(models)} models) ===")
    rows = []
    for m in models:
        traj = concat_amoc_trajectory(m, scenario)
        if traj is None:
            print(f"  {m}: missing files"); continue
        years, st, sp = traj
        if years.size < 200:
            continue
        t0 = _time.time()
        base_st_mu, _ = baseline_stats(st, years)
        end_mask = (years >= TIPPING_END[0]) & (years <= TIPPING_END[1])
        end_mean = float(st[end_mask][np.isfinite(st[end_mask])].mean())
        weaken = (base_st_mu - end_mean) / base_st_mu
        tipping = bool(weaken >= WEAKEN_FRAC)

        # NB: sliding_diag_ksg signature is (amoc_st, amoc_sp, years).
        centres, te, var, ac = sliding_diag_ksg(st, sp, years)
        bmask = (centres >= BASELINE_PERIOD[0]) & (centres <= BASELINE_PERIOD[1])
        te_mu = float(np.nanmean(te[bmask])); te_sig = float(np.nanstd(te[bmask]))
        var_mu = float(np.nanmean(var[bmask])); var_sig = float(np.nanstd(var[bmask]))
        ac_mu = float(np.nanmean(ac[bmask])); ac_sig = float(np.nanstd(ac[bmask]))

        t_TE = first_persistent_crossing(centres, te, te_mu - 2 * te_sig,
                                         "below", PERSIST_YEARS)
        t_var = first_persistent_crossing(centres, var, var_mu + 2 * var_sig,
                                          "above", PERSIST_YEARS)
        t_alpha = first_persistent_crossing(centres, ac, ac_mu + 2 * ac_sig,
                                            "above", PERSIST_YEARS)

        passed = None
        lead = None
        if tipping:
            if t_TE is None or (t_var is None and t_alpha is None):
                passed = False
            else:
                csd_times = [t for t in (t_var, t_alpha) if t is not None]
                t_csd = min(csd_times)
                lead = float(t_csd - t_TE)
                passed = bool(lead >= LEAD_YEARS_BAR)
        ver = "PASS" if passed else ("fail" if passed is False else "n/a")
        print(f"  {m:<22}: weaken={weaken*100:+5.1f}%  "
              f"{'TIP' if tipping else 'stable':<6}  "
              f"t_TE={t_TE} t_var={t_var}  {ver}  ({_time.time()-t0:.1f}s)")
        rows.append({"model": m, "tipping_prone": tipping,
                     "weaken_frac": float(weaken),
                     "t_TE": t_TE, "t_var": t_var, "t_alpha": t_alpha,
                     "lead_yr": lead, "passed": passed,
                     "centres": centres.tolist(), "TE": te.tolist(),
                     "var": var.tolist(), "ac1": ac.tolist()})
    return rows


# ---- Driver --------------------------------------------------------------

def main():
    H3_KSG_DIR.mkdir(parents=True, exist_ok=True)

    pic = picontrol_ksg()
    valid_pic = [r for r in pic if "passed" in r]
    n_pass_pic = sum(1 for r in valid_pic if r["passed"])
    pic_supported = (len(valid_pic) > 0
                      and n_pass_pic / len(valid_pic) >= AGGREGATE_FRAC)
    print(f"\npiControl KSG: {n_pass_pic}/{len(valid_pic)} pass "
          f"({n_pass_pic / max(1, len(valid_pic)):.0%}).  "
          f"{'SUPPORTED' if pic_supported else 'FALSIFIED'} at 60% bar.")

    fwd = forced_ksg("ssp585")
    fwd_tipping = [r for r in fwd if r.get("tipping_prone")]
    fwd_pass = [r for r in fwd_tipping if r.get("passed") is True]
    fwd_supported = (len(fwd_tipping) > 0
                      and len(fwd_pass) / len(fwd_tipping) >= AGGREGATE_FRAC)
    print(f"\nForced ssp585 KSG: {len(fwd_pass)}/{len(fwd_tipping)} pass "
          f"({len(fwd_pass) / max(1, len(fwd_tipping)):.0%}).  "
          f"{'SUPPORTED' if fwd_supported else 'FALSIFIED'} at 60% bar.")

    summary = {
        "estimator": "KSG (Frenzel-Pompe), k_NN=4, history k=l=1, units bits",
        "piControl": {
            "n_models": len(valid_pic),
            "n_passed": n_pass_pic,
            "pass_frac": (n_pass_pic / len(valid_pic)) if valid_pic else 0.0,
            "supported": bool(pic_supported),
            "per_model": [{k: v for k, v in r.items() if k != "rhos"}
                          for r in valid_pic],
        },
        "forced_ssp585": {
            "n_tipping_prone": len(fwd_tipping),
            "n_passed": len(fwd_pass),
            "pass_frac": (len(fwd_pass) / len(fwd_tipping))
                          if fwd_tipping else 0.0,
            "supported": bool(fwd_supported),
            "per_model": [{k: v for k, v in r.items()
                            if k not in ("centres", "TE", "var", "ac1")}
                          for r in fwd],
        },
    }
    out_json = H3_KSG_DIR / "h3_ksg_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {out_json}")

    # ----- Combined Figure 6 (KSG, replaces binary fig6 + fig7) -----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax_pic, ax_fwd = axes

    # Panel A: piControl lagged correlation
    cmap_a = plt.get_cmap("tab20")
    for i, r in enumerate(valid_pic):
        rhos = [r["rhos"].get(tau, np.nan) for tau in LAGS_YEARS]
        ax_pic.plot(LAGS_YEARS, rhos, "-", color=cmap_a(i % 20),
                    alpha=0.5, lw=0.7, label=r["model"])
    median_pic = [np.nanmedian([r["rhos"].get(tau, np.nan)
                                  for r in valid_pic if "rhos" in r])
                  for tau in LAGS_YEARS]
    ax_pic.plot(LAGS_YEARS, median_pic, "k-", lw=2.5, label="median",
                marker="s")
    ax_pic.axhline(0, color="grey", lw=0.4)
    ax_pic.axvline(0, color="grey", lw=0.4)
    ax_pic.set_xlabel(r"lag $\tau$ (yr)")
    ax_pic.set_ylabel(r"$\rho(T_{ST\to SP}^\mathrm{KSG}(t),\ \sigma^2_{SP}(t+\tau))$")
    ax_pic.legend(loc="best", fontsize=6, ncol=2, frameon=False)

    # Panel B: forced ssp585 standardised TE-KSG trajectories
    cmap_b = plt.get_cmap("tab20")
    for i, r in enumerate([r for r in fwd if r.get("tipping_prone")]):
        centres = np.asarray(r.get("centres", []))
        te = np.asarray(r.get("TE", []))
        if centres.size == 0:
            continue
        bmask = (centres >= BASELINE_PERIOD[0]) & (centres <= BASELINE_PERIOD[1])
        te_z = (te - np.nanmean(te[bmask])) / max(np.nanstd(te[bmask]), 1e-9)
        ax_fwd.plot(centres, te_z, color=cmap_b(i % 20), lw=0.7,
                    alpha=0.7, label=r["model"])
    ax_fwd.axhline(0, color="grey", lw=0.4)
    ax_fwd.axhline(-2, color="grey", lw=0.4, ls=":")
    ax_fwd.axhline(+2, color="grey", lw=0.4, ls=":")
    ax_fwd.set_xlabel("year")
    ax_fwd.set_ylabel(r"standardised $T_{ST\to SP}^\mathrm{KSG}$")
    ax_fwd.legend(loc="upper left", fontsize=6, ncol=4, frameon=False)

    out_fig = MANUSCRIPT_FIGS / "fig8_h3_ksg.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
