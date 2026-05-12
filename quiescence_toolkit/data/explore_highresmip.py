"""Explore what HighResMIP tos data is actually on Pangeo."""
from __future__ import annotations

import intake


def main():
    cat = intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    print(f"Pangeo catalog: {len(cat.df)} rows")

    # First, what HighResMIP activity rows are there?
    hr = cat.search(activity_id="HighResMIP", variable_id="tos")
    print(f"\nHighResMIP tos rows: {len(hr.df)}")
    if len(hr.df) > 0:
        print(f"  source_id values: {sorted(hr.df['source_id'].unique())}")
        print(f"  experiment_id values: {sorted(hr.df['experiment_id'].unique())}")
        print(f"  table_id values: {sorted(hr.df['table_id'].unique())}")
        print(f"  member_id (first 5): {sorted(hr.df['member_id'].unique())[:5]}")

    # CMIP6 resolution-variant pairs (same model family, two resolutions)
    pairs = [
        # (high-res, low-res, family)
        ("CESM2", "CESM2-FV2", "CESM2"),
        ("AWI-CM-1-1-MR", "AWI-CM-1-1-LR", "AWI-CM"),
        ("HadGEM3-GC31-MM", "HadGEM3-GC31-LL", "HadGEM3"),
        ("NorESM2-MM", "NorESM2-LM", "NorESM2"),
        ("CNRM-CM6-1-HR", "CNRM-CM6-1", "CNRM-CM6"),
        ("MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "MPI-ESM"),
    ]
    for hi, lo, family in pairs:
        h = cat.search(source_id=hi, variable_id="tos",
                          experiment_id="historical", table_id="Omon")
        l = cat.search(source_id=lo, variable_id="tos",
                          experiment_id="historical", table_id="Omon")
        print(f"\n{family} pair: {hi} ({len(h.df)} rows)  vs  "
              f"{lo} ({len(l.df)} rows)")
        if len(h.df) > 0:
            print(f"  HR members: {sorted(h.df['member_id'].unique())[:3]}")
        if len(l.df) > 0:
            print(f"  LR members: {sorted(l.df['member_id'].unique())[:3]}")


if __name__ == "__main__":
    main()
