"""BLOCKER 4 of the peer review: emergent constraint with and without
CMCC-CM2-SR5.

The published emergent-constraint subset (CanESM5, CMCC-CM2-SR5,
MIROC6) selects models whose historical Sen slope matches HadISST
(within +/- 0.4 degC/century).  CMCC also turned out to have
overdispersed piControl variance (BLOCKER 3).  The reviewer (A2, A11)
asks: if CMCC's HadISST match is achieved through inflated internal
noise rather than realistic forced response, the emergent constraint
should be re-tabulated without CMCC.
"""
from __future__ import annotations

import json

import numpy as np

from dcps.config import CACHE_DIR


CONTRAST = CACHE_DIR / "cmip6_contrast" / "cmip6_contrast.json"
OUT = CACHE_DIR / "emergent_constraint" / "no_cmcc.json"
OBS_RATE = -0.50  # degC / century
MATCH_TOL = 0.4   # degC / century


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(CONTRAST.read_text())

    # Models matching HadISST historical within MATCH_TOL
    hist = d["historical"]
    constrained_full = {m: r["sen_degC_per_century"] for m, r in hist.items()
                          if abs(r["sen_degC_per_century"] - OBS_RATE) <= MATCH_TOL}

    print("=" * 70)
    print(" BLOCKER 4: emergent-constraint subset with/without CMCC")
    print("=" * 70)
    print(f"  Matching criterion: |historical Sen - {OBS_RATE:+.2f}| <= "
          f"{MATCH_TOL:.2f} degC/century")
    print()
    print("  Constrained subset (full):")
    for m, s in sorted(constrained_full.items()):
        print(f"    {m:<18} historical Sen = {s:+.3f} degC/century")
    print()

    constrained_no_cmcc = {m: s for m, s in constrained_full.items()
                              if "CMCC" not in m}
    print("  Constrained subset (no CMCC):")
    for m, s in sorted(constrained_no_cmcc.items()):
        print(f"    {m:<18} historical Sen = {s:+.3f} degC/century")
    print()

    # SSP585 + SSP245 projections at 2100, both subsets
    def _ssp_summary(models, scenario):
        slopes = []
        endpoints = []
        for m in models:
            r = d.get(scenario, {}).get(m)
            if r is None:
                continue
            slopes.append(r["sen_degC_per_century"])
            # End-of-century value: extrapolate Sen slope over the
            # scenario window (2015-2100 = 85 yr).
            n_yr = r.get("n_years", 86)
            endpoints.append(r["sen_degC_per_year"] * n_yr)
        if not slopes:
            return None
        slopes = np.asarray(slopes)
        endpoints = np.asarray(endpoints)
        return dict(
            n=len(slopes),
            slope_p16=float(np.percentile(slopes, 16)),
            slope_p50=float(np.percentile(slopes, 50)),
            slope_p84=float(np.percentile(slopes, 84)),
            endpoint_p16=float(np.percentile(endpoints, 16)),
            endpoint_p50=float(np.percentile(endpoints, 50)),
            endpoint_p84=float(np.percentile(endpoints, 84)),
        )

    print("  SSP245 / SSP585 projections:")
    out = {"observed_rate": OBS_RATE, "match_tol": MATCH_TOL}
    for scenario in ("ssp245", "ssp585"):
        full = _ssp_summary(list(constrained_full.keys()), scenario)
        no_cmcc = _ssp_summary(list(constrained_no_cmcc.keys()), scenario)
        out[scenario] = dict(full=full, no_cmcc=no_cmcc)
        print(f"    {scenario}:")
        if full:
            print(f"      full constrained (n={full['n']}): "
                  f"endpoint p16/50/84 = "
                  f"{full['endpoint_p16']:+.2f} / "
                  f"{full['endpoint_p50']:+.2f} / "
                  f"{full['endpoint_p84']:+.2f} degC at 2100")
        if no_cmcc:
            print(f"      no-CMCC (n={no_cmcc['n']}):           "
                  f"endpoint p16/50/84 = "
                  f"{no_cmcc['endpoint_p16']:+.2f} / "
                  f"{no_cmcc['endpoint_p50']:+.2f} / "
                  f"{no_cmcc['endpoint_p84']:+.2f} degC at 2100")

    out["constrained_full_models"] = list(constrained_full.keys())
    out["constrained_no_cmcc_models"] = list(constrained_no_cmcc.keys())
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
