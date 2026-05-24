"""H3 test on CMIP6 forced trajectories (historical + ssp585 / ssp245).

Pre-registered (see plan file):
  Per model with paired historical + scenario vo_zonal:
    1. Build 251-yr (251 monthly: 1850-2100) AMOC trajectory at 26.5N and 50N.
    2. Tipping-prone: AMOC@26.5N drops >=25% (1850-1950 baseline -> 2080-2100).
    3. On 30-yr sliding annual window: TE_{ST->SP}, var(SP), AC1(SP).
    4. Baseline (1850-1950) mean+sigma of each EWS metric.
    5. t_TE: first window where TE < base_mu - 2*sigma persistent 5 yr.
       t_var: first window where var > base_mu + 2*sigma persistent 5 yr.
       t_alpha: same for AC1.
    6. Per-model SUPPORTED iff t_TE < min(t_var, t_alpha) - 3 yr.
    7. Aggregate SUPPORTED iff >= 60% of tipping-prone models pass.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time as _time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.h3 import _binary_te, _lag1_autocorr

# Reuse the validated helpers from the piControl driver
import sys
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from h3_picontrol import amoc_at_latitude  # noqa: E402

CMIP6_DIR = Path("/home/bijanf/Documents/AMOC_renalysis/data/cmip6_fullfield")
H3_FORCED_DIR = CACHE_DIR / "h3_forced"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

# Pre-registered parameters
SUBTROPICAL_LAT = 26.5
SUBPOLAR_LAT = 50.0
WINDOW_YEARS = 30
BASELINE_PERIOD = (1850, 1950)
TIPPING_END = (2080, 2100)
WEAKEN_FRAC = 0.25
PERSIST_YEARS = 5
LEAD_YEARS_BAR = 3
AGGREGATE_FRAC = 0.60


def concat_amoc_trajectory(model: str, scenario: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Concatenate historical + scenario AMOC at (26.5N, 50N).
    Returns (years, amoc_st, amoc_sp) annual arrays or None if missing."""
    hist_f = CMIP6_DIR / f"{model}_historical_vo_zonal.nc"
    scen_f = CMIP6_DIR / f"{model}_{scenario}_vo_zonal.nc"
    if not (hist_f.exists() and scen_f.exists()):
        return None
    yr_h, st_h = amoc_at_latitude(str(hist_f), SUBTROPICAL_LAT)
    yr_h2, sp_h = amoc_at_latitude(str(hist_f), SUBPOLAR_LAT)
    yr_s, st_s = amoc_at_latitude(str(scen_f), SUBTROPICAL_LAT)
    yr_s2, sp_s = amoc_at_latitude(str(scen_f), SUBPOLAR_LAT)
    yr = np.concatenate([yr_h, yr_s])
    st = np.concatenate([st_h, st_s])
    sp = np.concatenate([sp_h, sp_s])
    # Drop duplicate years if any (some models repeat 2014-2015 boundary)
    _, idx = np.unique(yr, return_index=True)
    return yr[idx], st[idx], sp[idx]


def baseline_stats(values: np.ndarray, years: np.ndarray, period=BASELINE_PERIOD):
    mask = (years >= period[0]) & (years <= period[1])
    v = values[mask]
    v = v[np.isfinite(v)]
    if v.size < 30:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std())


def first_persistent_crossing(centres, values, threshold, direction, persist_yr):
    """First centre-year of a window whose value crosses threshold in direction
    and stays there for >= persist_yr."""
    arr = np.asarray(values)
    yrs = np.asarray(centres)
    if direction == "below":
        mask = arr < threshold
    elif direction == "above":
        mask = arr > threshold
    else:
        raise ValueError(direction)
    n = arr.size
    n_persist = int(round(persist_yr))
    for i in range(n - n_persist):
        if mask[i] and mask[i:i + n_persist].all():
            return float(yrs[i])
    return None


def sliding_window_ews(years: np.ndarray, st: np.ndarray, sp: np.ndarray,
                        window: int = WINDOW_YEARS):
    n = len(years)
    centres, te_v, var_v, ac_v = [], [], [], []
    for s in range(0, n - window + 1):
        e = s + window
        x_st = st[s:e]; x_sp = sp[s:e]
        if not (np.isfinite(x_st).all() and np.isfinite(x_sp).all()):
            te_v.append(np.nan); var_v.append(np.nan); ac_v.append(np.nan)
        else:
            te_v.append(_binary_te(x_sp, x_st))
            var_v.append(float(np.var(x_sp)))
            ac_v.append(_lag1_autocorr(x_sp))
        centres.append(years[s + window // 2])
    return (np.array(centres),
            np.array(te_v), np.array(var_v), np.array(ac_v))


def run_model_scenario(model: str, scenario: str) -> dict:
    out: dict = {"model": model, "scenario": scenario}
    traj = concat_amoc_trajectory(model, scenario)
    if traj is None:
        out["skipped"] = "missing files"; return out
    years, st, sp = traj
    if years.size < 200:
        out["skipped"] = f"short trajectory ({years.size} yr)"; return out

    base_st_mu, base_st_sig = baseline_stats(st, years)
    out["amoc_baseline_mean_Sv"] = base_st_mu
    out["amoc_baseline_std_Sv"] = base_st_sig

    end_mask = (years >= TIPPING_END[0]) & (years <= TIPPING_END[1])
    end_amoc = st[end_mask]
    end_amoc = end_amoc[np.isfinite(end_amoc)]
    if end_amoc.size == 0 or not np.isfinite(base_st_mu):
        out["skipped"] = "no end-period data"; return out
    end_mean = float(end_amoc.mean())
    weaken_frac = (base_st_mu - end_mean) / base_st_mu
    out["end_period_mean_Sv"] = end_mean
    out["weaken_frac"] = float(weaken_frac)
    out["tipping_prone"] = bool(weaken_frac >= WEAKEN_FRAC)

    # Sliding window EWS over the entire trajectory
    centres, te, var, ac = sliding_window_ews(years, st, sp)
    out["centres"] = centres.tolist()
    out["TE"] = te.tolist()
    out["var"] = var.tolist()
    out["ac1"] = ac.tolist()

    # Baseline EWS stats from windows whose centre falls in baseline period
    bmask = (centres >= BASELINE_PERIOD[0]) & (centres <= BASELINE_PERIOD[1])
    te_mu = float(np.nanmean(te[bmask])) if bmask.any() else float("nan")
    te_sig = float(np.nanstd(te[bmask])) if bmask.any() else float("nan")
    var_mu = float(np.nanmean(var[bmask])) if bmask.any() else float("nan")
    var_sig = float(np.nanstd(var[bmask])) if bmask.any() else float("nan")
    ac_mu = float(np.nanmean(ac[bmask])) if bmask.any() else float("nan")
    ac_sig = float(np.nanstd(ac[bmask])) if bmask.any() else float("nan")

    t_TE = first_persistent_crossing(centres, te, te_mu - 2 * te_sig,
                                     "below", PERSIST_YEARS)
    t_var = first_persistent_crossing(centres, var, var_mu + 2 * var_sig,
                                       "above", PERSIST_YEARS)
    t_alpha = first_persistent_crossing(centres, ac, ac_mu + 2 * ac_sig,
                                         "above", PERSIST_YEARS)
    out["t_TE"] = t_TE
    out["t_var"] = t_var
    out["t_alpha"] = t_alpha

    if out["tipping_prone"]:
        if t_TE is None or (t_var is None and t_alpha is None):
            out["passed"] = False
            out["reason_no_pass"] = "TE or both CSD never crossed"
        else:
            csd_times = [t for t in (t_var, t_alpha) if t is not None]
            t_csd = min(csd_times)
            out["lead_yr"] = float(t_csd - t_TE)
            out["passed"] = bool(t_csd - t_TE >= LEAD_YEARS_BAR)
    else:
        out["passed"] = None      # not eligible
    return out


def aggregate(results: list[dict]) -> dict:
    tipping = [r for r in results if r.get("tipping_prone")]
    passed = [r for r in tipping if r.get("passed") is True]
    return {
        "n_total": len(results),
        "n_with_data": sum(1 for r in results if "skipped" not in r),
        "n_tipping_prone": len(tipping),
        "n_supported": len(passed),
        "pass_frac": (len(passed) / len(tipping)) if tipping else 0.0,
        "supported": (len(tipping) > 0
                       and (len(passed) / len(tipping)) >= AGGREGATE_FRAC),
    }


def list_models() -> list[str]:
    files = glob.glob(str(CMIP6_DIR / "*_historical_vo_zonal.nc"))
    return sorted({os.path.basename(f).split("_historical")[0] for f in files})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="ssp585", choices=["ssp585", "ssp245"])
    args = p.parse_args()
    H3_FORCED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== H3-forced test, scenario = {args.scenario} ===")
    print(f"  Trajectory: 1850-2100 (historical + {args.scenario})")
    print(f"  Tipping-prone: AMOC@26.5N drops >= {WEAKEN_FRAC:.0%} from"
          f" {BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]} baseline by"
          f" {TIPPING_END[0]}-{TIPPING_END[1]} mean")
    print(f"  Per-model: t_TE < min(t_var, t_alpha) - {LEAD_YEARS_BAR} yr")
    print(f"  Aggregate bar: {AGGREGATE_FRAC:.0%} of tipping-prone models\n")

    models = list_models()
    print(f"Models: {len(models)}\n")

    results = []
    for m in models:
        t0 = _time.time()
        r = run_model_scenario(m, args.scenario)
        if "skipped" in r:
            print(f"  {m:<20}: SKIP -- {r['skipped']}")
        else:
            tag = "TIPPING" if r["tipping_prone"] else "stable"
            ver = ("PASS" if r.get("passed") is True else
                   "fail" if r.get("passed") is False else "n/a")
            wf = r["weaken_frac"]
            base = r["amoc_baseline_mean_Sv"]
            print(f"  {m:<20}: base={base:5.1f} Sv  weaken={wf*100:+5.1f}%  "
                  f"{tag:<7}  t_TE={r.get('t_TE')}  t_var={r.get('t_var')}  "
                  f"verdict={ver}  ({_time.time()-t0:.1f}s)")
        results.append(r)

    agg = aggregate(results)
    print(f"\n=== Aggregate H3-forced ({args.scenario}) ===")
    print(f"  Models with data:    {agg['n_with_data']}/{agg['n_total']}")
    print(f"  Tipping-prone:       {agg['n_tipping_prone']}")
    print(f"  Per-model passed:    {agg['n_supported']}/"
          f"{agg['n_tipping_prone']}  ({agg['pass_frac']*100:.0f}%)")
    print(f"  AGGREGATE: {'SUPPORTED' if agg['supported'] else 'FALSIFIED'}"
          f" (bar = {AGGREGATE_FRAC*100:.0f}%)")

    out_path = H3_FORCED_DIR / f"h3_forced_{args.scenario}.json"
    # Strip out heavy time-series arrays before saving the small summary JSON
    summary = {
        "scenario": args.scenario,
        "aggregate": agg,
        "per_model": [{k: v for k, v in r.items()
                       if k not in ("centres", "TE", "var", "ac1")}
                       for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    # Also save raw arrays for plotting
    np.savez_compressed(
        H3_FORCED_DIR / f"h3_forced_{args.scenario}_arrays.npz",
        **{f"{r['model']}__{k}": np.array(r.get(k, []))
           for r in results if "skipped" not in r
           for k in ("centres", "TE", "var", "ac1")},
    )

    # ----- Figure 7 -----
    tipping_results = [r for r in results if r.get("tipping_prone")]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True,
                              constrained_layout=True)
    ax_amoc, ax_te = axes

    cmap = plt.get_cmap("tab20")
    for i, r in enumerate(tipping_results):
        c = cmap(i % 20)
        centres = np.array(r.get("centres", []))
        te = np.array(r.get("TE", []))
        var = np.array(r.get("var", []))
        if centres.size == 0:
            continue
        bmask = (centres >= BASELINE_PERIOD[0]) & (centres <= BASELINE_PERIOD[1])
        te_z = (te - np.nanmean(te[bmask])) / max(np.nanstd(te), 1e-9)
        var_z = (var - np.nanmean(var[bmask])) / max(np.nanstd(var), 1e-9)
        ax_te.plot(centres, te_z, color=c, lw=0.7, alpha=0.7, label=r["model"])
        ax_amoc.plot(centres, var_z, color=c, lw=0.7, alpha=0.7)

    for ax, ylab in [(ax_amoc, r"standardised $\sigma^2_{SP}$"),
                      (ax_te,   r"standardised $T_{ST\to SP}$")]:
        ax.axhline(0, color="grey", lw=0.4)
        ax.axhline(-2, color="grey", lw=0.4, ls=":")
        ax.axhline(+2, color="grey", lw=0.4, ls=":")
        ax.set_ylabel(ylab)
    ax_te.set_xlabel("year")
    ax_amoc.legend(fontsize=6, ncol=4, loc="upper left", frameon=False)

    out_fig = MANUSCRIPT_FIGS / f"fig7_h3_forced_{args.scenario}.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
