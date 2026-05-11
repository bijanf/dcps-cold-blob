"""BLOCKER 3 of the peer review: pre-registered mechanical exclusion
rule for the piControl detection-and-attribution test.

Pre-registered now (NOT at v1.0-pre-registration; added in revision):
  Any model whose 100-yr piControl subpolar-subtropical SST contrast
  Sen-slope VARIANCE exceeds 2 x the multi-model median is excluded
  from the null distribution as an overdispersed outlier.

The original strict-form test (1% threshold) is reported as failed
at 11.4% exceedance without softening.  The mechanical-exclusion
test is reported alongside as a follow-on robustness analysis with
a clearly documented post-hoc exclusion rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcps.config import CACHE_DIR
from dcps.nature_style import apply_nature_style
apply_nature_style()


IN = CACHE_DIR / "picontrol_dna" / "dna_results.json"
OUT_DIR = CACHE_DIR / "picontrol_dna"
OBSERVED = -0.50  # degC / century


def main():
    d = json.loads(IN.read_text())
    per_model = d["per_model"]
    obs = d.get("observed_rate_degC_per_century", OBSERVED)

    # Per-model variance of 100-yr Sen slopes
    per_model_var = {}
    for m, v in per_model.items():
        slopes = np.asarray(v["slopes_per_century"])
        per_model_var[m] = float(np.var(slopes, ddof=1))

    var_arr = np.asarray(list(per_model_var.values()))
    median_var = float(np.median(var_arr))
    threshold = 2.0 * median_var
    excluded = [m for m, v in per_model_var.items() if v > threshold]
    kept = [m for m in per_model if m not in excluded]

    print("=" * 70)
    print(" BLOCKER 3: mechanical exclusion rule for piControl DnA")
    print("=" * 70)
    print(f" Observed rate: {obs:+.2f} degC/century")
    print(f" Multi-model median per-model variance: {median_var:.3f}")
    print(f" Exclusion threshold (2 x median): {threshold:.3f}")
    print()
    print(f" Per-model variance (sorted desc):")
    for m in sorted(per_model_var, key=per_model_var.get, reverse=True):
        flag = "EXCLUDED" if m in excluded else "kept"
        print(f"   {m:<18} var = {per_model_var[m]:6.3f}  {flag}")
    print()
    print(f" Excluded models: {excluded}")
    print()

    # Re-tabulate exceedance
    def _frac(models):
        slopes_pool = []
        for m in models:
            slopes_pool.extend(per_model[m]["slopes_per_century"])
        slopes_pool = np.asarray(slopes_pool, dtype=float)
        n_total = slopes_pool.size
        n_exceed = int(np.sum(slopes_pool <= obs))
        return n_total, n_exceed, n_exceed / max(n_total, 1)

    strict_total, strict_exc, strict_frac = _frac(list(per_model.keys()))
    mech_total, mech_exc, mech_frac = _frac(kept)

    print(" Strict pre-registered test (no exclusion):")
    print(f"   exceedance: {strict_exc}/{strict_total} = {100*strict_frac:.2f}%")
    print(f"   threshold:  1.00%  -- FAILED (well above)")
    print()
    print(f" Mechanical exclusion (variance > 2x multi-model median):")
    print(f"   models kept: {len(kept)} ({', '.join(kept)})")
    print(f"   exceedance: {mech_exc}/{mech_total} = {100*mech_frac:.2f}%")
    print(f"   threshold:  1.00%  -- "
          f"{'PASSED' if mech_frac < 0.01 else 'FAILED'}")

    summary = {
        "observed_rate_degC_per_century": obs,
        "strict_pre_registered": {
            "n_total": strict_total,
            "n_exceed": strict_exc,
            "fraction_exceed": strict_frac,
            "threshold": 0.01,
            "verdict": "FAILED (well above 1%)",
        },
        "mechanical_exclusion": {
            "rule": "exclude models with per-model 100-yr Sen-slope "
                    "variance > 2 x multi-model median",
            "median_variance": median_var,
            "threshold_variance": threshold,
            "excluded": excluded,
            "kept": kept,
            "n_total": mech_total,
            "n_exceed": mech_exc,
            "fraction_exceed": mech_frac,
            "threshold": 0.01,
            "verdict": ("PASSED (below 1%)" if mech_frac < 0.01
                        else f"FAILED (above 1% at {100*mech_frac:.2f}%)"),
            "pre_registration_note": (
                "This exclusion rule was added at the revision stage in "
                "response to peer review BLOCKER 3.  It is NOT anchored "
                "to v1.0-pre-registration."),
        },
        "per_model_variance": per_model_var,
    }
    with open(OUT_DIR / "mechanical_exclusion.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'mechanical_exclusion.json'}")


if __name__ == "__main__":
    main()
