"""
Bootstrap CI for the 40-yr EKE-leads-Q median gap, and paired analysis of
the 5 dual-exit Atlantic models.

This addresses one piece of the manuscript's '40-year' claim by attaching
an explicit non-parametric bootstrap 95% confidence interval, and by
splitting the 40-yr ensemble-median gap into two components Xavier asked
about: model-selection (different model subsets exit on Q vs EKE) and the
within-model paired lag.

Inputs (all already on disk; no external fetch needed):
  dcps/cache/holocene_exit/audit/significant_exit_atlantic.json     (Q side)
  dcps/cache/holocene_exit/audit/significant_eke_exit_atlantic.json (EKE side)

Outputs:
  dcps/cache/lag_40yr/window_sensitivity.json   (median + bootstrap CI, paired stats)
  dcps/cache/lag_40yr/per_model_exits.csv       (per-model exit-year table)
"""
from __future__ import annotations
import json
import csv
from pathlib import Path
from statistics import median
import random

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "dcps" / "cache" / "holocene_exit" / "audit"
OUT_DIR = REPO / "dcps" / "cache" / "lag_40yr"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_exit_years(path: Path) -> dict[str, int | None]:
    d = json.loads(path.read_text())
    return {row["model"]: row.get("significant_exit") for row in d["per_model"]}


def bootstrap_median(values: list[int], n_boot: int = 10000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Return (median, p2.5, p97.5) via percentile bootstrap."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    medians = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(median(sample))
    medians.sort()
    return (
        median(values),
        medians[int(0.025 * n_boot)],
        medians[int(0.975 * n_boot)],
    )


def bootstrap_gap(q_years: list[int], eke_years: list[int],
                  n_boot: int = 10000, seed: int = 1) -> dict:
    """Bootstrap CI on (median Q exit) - (median EKE exit)."""
    rng = random.Random(seed)
    nq, ne = len(q_years), len(eke_years)
    gaps = []
    for _ in range(n_boot):
        qs = [q_years[rng.randrange(nq)] for _ in range(nq)]
        es = [eke_years[rng.randrange(ne)] for _ in range(ne)]
        gaps.append(median(qs) - median(es))
    gaps.sort()
    point = median(q_years) - median(eke_years)
    return dict(
        point_estimate_yr=point,
        ci95_low=gaps[int(0.025 * n_boot)],
        ci95_high=gaps[int(0.975 * n_boot)],
        ci68_low=gaps[int(0.16 * n_boot)],
        ci68_high=gaps[int(0.84 * n_boot)],
        n_boot=n_boot,
    )


def main():
    q_map = load_exit_years(AUDIT / "significant_exit_atlantic.json")
    e_map = load_exit_years(AUDIT / "significant_eke_exit_atlantic.json")

    q_strict = sorted([y for y in q_map.values() if y is not None])
    e_strict = sorted([y for y in e_map.values() if y is not None])

    print(f"Q strict-exit models: n={len(q_strict)}, median={median(q_strict)} (range {min(q_strict)}-{max(q_strict)})")
    print(f"EKE strict-exit models: n={len(e_strict)}, median={median(e_strict)} (range {min(e_strict)}-{max(e_strict)})")

    q_stats = bootstrap_median(q_strict)
    e_stats = bootstrap_median(e_strict)
    gap_stats = bootstrap_gap(q_strict, e_strict)

    print(f"Q median exit:   {q_stats[0]} (95% CI {q_stats[1]}-{q_stats[2]})")
    print(f"EKE median exit: {e_stats[0]} (95% CI {e_stats[1]}-{e_stats[2]})")
    print(f"Gap = Q_med - EKE_med = {gap_stats['point_estimate_yr']:+.1f} yr "
          f"(95% CI {gap_stats['ci95_low']:+.0f} to {gap_stats['ci95_high']:+.0f}; "
          f"68% CI {gap_stats['ci68_low']:+.0f} to {gap_stats['ci68_high']:+.0f})")

    # Paired analysis: models that exit strictly on BOTH
    paired_models = sorted(set(q_map) & set(e_map))
    paired = []
    for m in paired_models:
        q = q_map.get(m)
        e = e_map.get(m)
        if q is None or e is None:
            continue
        paired.append((m, e, q, q - e))
    print(f"\nDual-exit models (strict on BOTH Q and EKE): n={len(paired)}")
    paired_lags = [d for (_, _, _, d) in paired]
    if paired_lags:
        med = median(paired_lags)
        print(f"  within-model paired lag (Q - EKE), median: {med:+.0f} yr")
        print(f"  range: {min(paired_lags):+d} to {max(paired_lags):+d} yr")
    for m, e, q, d in paired:
        print(f"    {m:25s} EKE={e}  Q={q}  Q-EKE={d:+d} yr")

    # Sensitivity to subsampling: drop each model in turn, recompute gap
    print("\n=== Jackknife sensitivity (leave-one-model-out) ===")
    jack_gaps_q = []
    for i in range(len(q_strict)):
        sub = q_strict[:i] + q_strict[i + 1:]
        jack_gaps_q.append(median(sub) - median(e_strict))
    jack_gaps_e = []
    for i in range(len(e_strict)):
        sub = e_strict[:i] + e_strict[i + 1:]
        jack_gaps_e.append(median(q_strict) - median(sub))
    print(f"  Drop-one-Q-model gap range:   {min(jack_gaps_q):+.0f} to {max(jack_gaps_q):+.0f} yr")
    print(f"  Drop-one-EKE-model gap range: {min(jack_gaps_e):+.0f} to {max(jack_gaps_e):+.0f} yr")

    out = dict(
        provenance="Computed from cached strict-exit JSONs; "
                   "no new compute beyond bootstrap on per-model exit years.",
        q_side=dict(
            n_strict_exit=len(q_strict),
            median=q_stats[0],
            ci95_low=q_stats[1], ci95_high=q_stats[2],
            range_min=min(q_strict), range_max=max(q_strict),
            exit_years=q_strict,
        ),
        eke_side=dict(
            n_strict_exit=len(e_strict),
            median=e_stats[0],
            ci95_low=e_stats[1], ci95_high=e_stats[2],
            range_min=min(e_strict), range_max=max(e_strict),
            exit_years=e_strict,
        ),
        ensemble_gap=gap_stats,
        within_model_paired=dict(
            n_dual_exit=len(paired),
            median_lag=median(paired_lags) if paired_lags else None,
            range_min=min(paired_lags) if paired_lags else None,
            range_max=max(paired_lags) if paired_lags else None,
            per_model=[
                dict(model=m, eke_exit=e, q_exit=q, q_minus_eke=d)
                for (m, e, q, d) in paired
            ],
        ),
        jackknife=dict(
            q_drop_one_min=min(jack_gaps_q),
            q_drop_one_max=max(jack_gaps_q),
            e_drop_one_min=min(jack_gaps_e),
            e_drop_one_max=max(jack_gaps_e),
        ),
    )
    (OUT_DIR / "window_sensitivity.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_DIR / 'window_sensitivity.json'}")

    # Per-model CSV
    csv_path = OUT_DIR / "per_model_exits.csv"
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["model", "Q_strict_exit", "EKE_strict_exit", "Q_minus_EKE_yr"])
        for m in sorted(set(q_map) | set(e_map)):
            q = q_map.get(m); e = e_map.get(m)
            gap = (q - e) if (q is not None and e is not None) else ""
            w.writerow([m, q if q is not None else "", e if e is not None else "", gap])
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
