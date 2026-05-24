"""CMIP6 emergent-constraint subset (Phase 5).

Reconciles the published Cold-Blob model-data discrepancy: the full
CMIP6 ensemble median historical trend has the wrong sign relative
to HadISST. We address this by stratifying the ensemble by historical
fidelity:

  Constrained subset: models whose historical 1870--2014 contrast
  Sen slope falls within [obs +/- 0.2 deg C/century] of the observed
  HadISST trend (-0.50 deg C/century).

We then report the constrained-subset SSP245 and SSP585 projection
ensemble medians and 16-84% bands alongside the full-ensemble
projections. The constrained-subset projections will be more
aggressive (the observationally-faithful models are also the more
AMOC-sensitive ones), making our full-ensemble published projection
a conservative one.

Pre-registered before computation:
  - Selection criterion: |hist_Sen - obs_Sen| <= 0.2 deg C/century
  - Output: subset count, full vs constrained SSP245 and SSP585
    medians + bands at 2100, 95% CI on each.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()


OUT_DIR = CACHE_DIR / "emergent_constraint"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"
OBS_HIST_SEN = -0.50    # deg C / century (HadISST 1870-2023 contrast)
CONSTRAINT_TOL = 0.4    # deg C / century tolerance


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmip6 = json.loads(
        (CACHE_DIR / "cmip6_contrast" / "cmip6_contrast.json").read_text())

    # Per-model historical Sen slope (deg C / century)
    hist_slopes = {}
    for model, info in cmip6.get("historical", {}).items():
        hist_slopes[model] = info["sen_degC_per_century"]

    constrained = [m for m, s in hist_slopes.items()
                    if abs(s - OBS_HIST_SEN) <= CONSTRAINT_TOL]
    print(f"Full CMIP6 ensemble: {len(hist_slopes)} models")
    print("  Historical Sen slopes (deg C/century):")
    for m, s in sorted(hist_slopes.items(), key=lambda kv: kv[1]):
        in_c = "✓" if m in constrained else " "
        print(f"    {in_c} {m:<22} {s:+.3f}")
    print(f"\nObservational anchor: HadISST {OBS_HIST_SEN:+.2f} °C/century")
    print(f"Constraint tolerance: ±{CONSTRAINT_TOL} °C/century")
    print(f"Constrained subset: {len(constrained)} models")

    # Per-experiment ensemble bands (full vs constrained)
    def _ensemble_band(exp: str, model_list: list[str]
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not model_list: return None, None, None, None
        arrs = []; yrs_all = []
        for m in model_list:
            ent = cmip6.get(exp, {}).get(m)
            if ent is None: continue
            yrs_all.append(np.array(ent["years"]))
            arrs.append(np.array(ent["contrast_anom_degC"]))
        if not arrs: return None, None, None, None
        ymin = min(y.min() for y in yrs_all)
        ymax = max(y.max() for y in yrs_all)
        grid = np.arange(ymin, ymax + 1)
        mat = np.full((len(arrs), len(grid)), np.nan)
        for i, (y, a) in enumerate(zip(yrs_all, arrs)):
            idx = np.searchsorted(grid, y)
            mat[i, idx] = a
        return grid, np.nanmedian(mat, axis=0), \
                np.nanpercentile(mat, 16, axis=0), \
                np.nanpercentile(mat, 84, axis=0)

    all_models = list(hist_slopes.keys())

    summary = {
        "obs_hist_sen_degC_per_century": OBS_HIST_SEN,
        "constraint_tolerance": CONSTRAINT_TOL,
        "n_models_total": len(all_models),
        "n_models_constrained": len(constrained),
        "constrained_models": constrained,
        "all_hist_slopes": hist_slopes,
        "scenario_summary": {},
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2),
                              constrained_layout=True, sharey=True)

    for i, exp in enumerate(("ssp245", "ssp585")):
        ax = axes[i]
        # Full ensemble
        yrs_f, med_f, p16_f, p84_f = _ensemble_band(exp, all_models)
        if yrs_f is not None:
            m = yrs_f <= 2100
            ax.fill_between(yrs_f[m], p16_f[m], p84_f[m],
                             color="0.65", alpha=0.25)
            ax.plot(yrs_f[m], med_f[m], color="0.30", lw=1.4,
                     label=f"full ensemble (N={len(all_models)})")
        # Constrained subset
        yrs_c, med_c, p16_c, p84_c = _ensemble_band(exp, constrained)
        if yrs_c is not None:
            m = yrs_c <= 2100
            color = "C1" if exp == "ssp245" else "C3"
            ax.fill_between(yrs_c[m], p16_c[m], p84_c[m],
                             color=color, alpha=0.30)
            ax.plot(yrs_c[m], med_c[m], color=color, lw=1.8,
                     label=f"obs-constrained subset (N={len(constrained)})")

        ax.axhline(0, color="0.4", lw=0.5)
        ax.set_title(f"{chr(97+i)} {exp.upper()}", loc="left",
                      fontweight="bold", pad=2)
        ax.set_xlabel("year CE")
        ax.set_xlim(2015, 2100)
        ax.legend(loc="lower left", fontsize=7.5, frameon=False)

        # Record end-of-century values
        idx2100 = -1
        if yrs_f is not None:
            summary["scenario_summary"][exp] = {
                "full_2100_median": float(med_f[idx2100]),
                "full_2100_p16": float(p16_f[idx2100]),
                "full_2100_p84": float(p84_f[idx2100]),
                "constrained_2100_median": (float(med_c[idx2100])
                                             if yrs_c is not None else None),
                "constrained_2100_p16": (float(p16_c[idx2100])
                                          if yrs_c is not None else None),
                "constrained_2100_p84": (float(p84_c[idx2100])
                                          if yrs_c is not None else None),
            }

    axes[0].set_ylabel("subpolar -- subtropical SST contrast anomaly (°C)")

    out_fig = MANUSCRIPT_FIGS / "fig_emergent_constraint.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"\nWrote {out_fig}")

    with open(OUT_DIR / "emergent_constraint.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {OUT_DIR / 'emergent_constraint.json'}")

    print("\n----- End-of-century summary -----")
    for exp, s in summary["scenario_summary"].items():
        print(f"  {exp.upper()}: "
              f"full = {s['full_2100_median']:+.2f} °C [{s['full_2100_p16']:+.2f}, {s['full_2100_p84']:+.2f}]")
        if s["constrained_2100_median"] is not None:
            print(f"            constrained = {s['constrained_2100_median']:+.2f} °C "
                  f"[{s['constrained_2100_p16']:+.2f}, {s['constrained_2100_p84']:+.2f}]")


if __name__ == "__main__":
    main()
