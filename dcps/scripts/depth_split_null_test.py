"""R3-1: Null-model permutation test for depth-dependent unprecedented split.

Of the five testable Caesar-2021 proxies in the multi-proxy gap-closure:
  - 3 surface / thermocline records (Thornalley Tsub at ~400 m,
    Spooner T. quinqueloba planktic, Rahmstorf 2015 multi-proxy AMOC
    index): ALL unprecedented at |z| > 3.
  - 2 deep records (Thornalley sortable silt at ~2000 m, Osmann 2019
    MAS productivity at the seafloor): BOTH within Holocene envelope.

This is a 3/3 vs 0/2 split by sampling depth.  We test whether this
split is significantly different from random assignment.

Permutation test:
  - 10,000 random reassignments of the 5 proxies to 3 surface and
    2 deep labels.
  - For each, count: how many "surface" proxies are unprecedented?
    How many "deep" proxies are unprecedented?
  - Fraction of permutations matching or exceeding the observed
    split is the empirical p-value.

The observed split contains 3 unprecedented proxies (Thornalley
Tsub, Spooner, Rahmstorf) and 2 not-unprecedented (Thornalley silt,
Osmann).  Under random labeling, we ask: what is the probability of
all 3 unprecedented proxies being assigned to the "surface" label?
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np

from dcps.config import CACHE_DIR


OUT_DIR = CACHE_DIR / "depth_split"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Original five testable Caesar-2021 proxies + two augmented deep
    # records that passed the same screening rule (>=10 pre-1850
    # samples, post-1850 overlap, decadal-or-better resolution).
    proxies = [
        # --- original 5 ---
        ("Thornalley 2018 Tsub", "surface", True),    # ~400 m thermocline
        ("Spooner 2020 T. quinqueloba", "surface", True),  # planktic surface
        ("Rahmstorf 2015 multi-proxy AMOC", "surface", True),  # surface-derived
        ("Thornalley 2018 sortable silt", "deep", False),  # ~2000 m DWBC
        ("Osmann 2019 MAS productivity", "deep", False),  # seafloor
        # --- augmented (this revision; result-naive at the time the
        #     screening rule was re-applied) ---
        ("Thibodeau 2018 MD99-2220 d18O", "deep", True),
        # ~3700 m Laurentian Slope; |z|_max = 4.65 (1850-1962 modern)
        ("Moffa-Sanchez 2015 RAPiD-35-COM", "deep", False),
        # 3484 m Eirik Drift, DSOW sortable silt; |z|_max = 1.15
        # (1850-1914 modern, pre-RAPID era)
    ]
    n_total = len(proxies)
    n_surface = sum(1 for _, d, _ in proxies if d == "surface")
    n_deep = sum(1 for _, d, _ in proxies if d == "deep")
    n_unprec = sum(1 for _, _, u in proxies if u)

    # Observed: 3 of 3 surface unprec, 0 of 2 deep unprec
    obs_surface_unprec = sum(1 for _, d, u in proxies if d == "surface" and u)
    obs_deep_unprec = sum(1 for _, d, u in proxies if d == "deep" and u)

    print("=" * 70)
    print(" R3-1 (augmented): depth-split null-permutation test")
    print("=" * 70)
    print(f"  {n_total} testable proxies "
          f"(5 Caesar-2021 + 2 augmented from this revision)")
    print(f"  Surface/thermocline: {n_surface}")
    print(f"  Deep: {n_deep}")
    print(f"  Unprecedented (|z| > 3): {n_unprec}")
    print(f"  Observed split: {obs_surface_unprec}/{n_surface} surface "
          f"unprec, {obs_deep_unprec}/{n_deep} deep unprec")
    print()

    # Exhaustive enumeration: C(5,3) = 10 ways to choose 3 of 5 proxies
    # to be the "unprecedented" ones; under random labeling, count how
    # many of these have all 3 unprecedented falling in the surface group.
    proxy_idx = list(range(n_total))
    surface_idx = {i for i, (_, d, _) in enumerate(proxies) if d == "surface"}
    deep_idx = {i for i, (_, d, _) in enumerate(proxies) if d == "deep"}
    obs_unprec_idx = {i for i, (_, _, u) in enumerate(proxies) if u}

    # Exhaustive: choose any 3 of the 5 proxies to be "unprecedented"
    extreme_count = 0
    total = 0
    splits = []
    for combo in combinations(proxy_idx, n_unprec):
        total += 1
        combo_set = set(combo)
        n_surf_in = len(combo_set & surface_idx)
        n_deep_in = len(combo_set & deep_idx)
        splits.append((n_surf_in, n_deep_in))
        # "Extreme" = all unprec in surface (n_surf_in = 3)
        if n_surf_in >= obs_surface_unprec and n_deep_in <= obs_deep_unprec:
            extreme_count += 1

    p_value = extreme_count / total
    print(f"  Exhaustive enumeration: {total} possible label assignments")
    print(f"  Permutations matching observed split or more extreme: "
          f"{extreme_count}")
    print(f"  Empirical p = {p_value:.4f}  (= {extreme_count}/{total})")
    print()

    # Larger Monte Carlo (10,000 reps) for cross-check
    rng = np.random.default_rng(42)
    B = 10000
    extreme_mc = 0
    for b in range(B):
        labels = rng.permutation(n_total)
        # First 3 are "unprecedented" under permutation
        unprec_under_perm = set(labels[:n_unprec].tolist())
        n_surf = len(unprec_under_perm & surface_idx)
        n_deep = len(unprec_under_perm & deep_idx)
        if n_surf >= obs_surface_unprec and n_deep <= obs_deep_unprec:
            extreme_mc += 1
    p_mc = extreme_mc / B
    print(f"  Monte Carlo cross-check: {B} permutations, p = {p_mc:.4f}")
    print()

    out = dict(
        proxies=[dict(name=n, depth=d, unprecedented=u)
                 for n, d, u in proxies],
        n_total=n_total, n_surface=n_surface, n_deep=n_deep,
        n_unprecedented=n_unprec,
        observed_split=dict(surface=obs_surface_unprec,
                              deep=obs_deep_unprec),
        p_exhaustive=p_value,
        p_monte_carlo=p_mc,
        verdict=("non-trivial depth split (p < 0.10)"
                 if p_value < 0.10 else "depth split could arise by chance"),
    )
    with open(OUT_DIR / "depth_split.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_DIR / 'depth_split.json'}")
    print()
    if p_value < 0.10:
        print(f"  Depth-split is significant at p = {p_value:.2f} (one-tailed,")
        print("  Fisher's exact extension): the observed 3/3 vs 0/2 pattern")
        print("  is unlikely under random assignment of depth labels.")


if __name__ == "__main__":
    main()
