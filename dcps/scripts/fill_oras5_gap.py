"""Fill the 2015-2021 ORAS5 SST+SSH gap in the local ARDP cache.

Pulls ONLY sea_surface_temperature and sea_surface_height (the two variables
the DCPS Phase 1 pipeline consumes) for 2015-2021 from the Copernicus CDS,
landing extracted monthly files alongside the existing oras5 cache.

Skips years that already have all 12 months for both variables on disk.

Usage:
    python scripts/fill_oras5_gap.py
    python scripts/fill_oras5_gap.py --workers 4
    python scripts/fill_oras5_gap.py --years 2015,2016,2017
"""

from __future__ import annotations

import argparse
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi

ORAS5_DIR = Path.home() / "Documents" / "AMOC_renalysis" / "data" / "oras5"
GAP_YEARS = list(range(2015, 2022))   # 2015-2021 inclusive
TARGET_VARS = ["sea_surface_temperature", "sea_surface_height"]
NATIVE_TOKENS = {"sea_surface_temperature": "sosstsst", "sea_surface_height": "sossheig"}


def months_present(token: str, year: int) -> set[int]:
    """Set of month integers present on disk for a given variable+year."""
    pat = re.compile(rf"^{token}_.*_{year}(\d{{2}})_.*\.nc$")
    out = set()
    for f in os.listdir(ORAS5_DIR):
        m = pat.match(f)
        if m:
            out.add(int(m.group(1)))
    return out


def year_complete(year: int) -> bool:
    """True if both target variables have all 12 months for this year."""
    return all(
        months_present(NATIVE_TOKENS[v], year) == set(range(1, 13))
        for v in TARGET_VARS
    )


def download_year(year: int, retries: int = 3) -> str:
    """Pull one year of SST+SSH (single zip), extract to oras5/, retry on failure."""
    if year_complete(year):
        return f"{year}: already complete (skipped)"

    product = "consolidated" if year <= 2014 else "operational"
    out_zip = ORAS5_DIR / f"oras5_dcps_gap_{year}.zip"

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            client = cdsapi.Client(quiet=True)
            client.retrieve(
                "reanalysis-oras5",
                {
                    "product_type": [product],
                    "vertical_resolution": "single_level",
                    "variable": TARGET_VARS,
                    "year": [str(year)],
                    "month": [f"{m:02d}" for m in range(1, 13)],
                },
                str(out_zip),
            )
            if out_zip.exists() and zipfile.is_zipfile(out_zip):
                with zipfile.ZipFile(out_zip, "r") as zf:
                    zf.extractall(ORAS5_DIR)
                out_zip.unlink()
                return f"{year}: extracted ({sum(1 for _ in os.listdir(ORAS5_DIR) if str(year) in _)} files now present)"
            elif out_zip.exists():
                # Single .nc returned, not zipped — keep as is
                return f"{year}: saved as raw .nc ({out_zip.stat().st_size/1e6:.0f} MB)"
            else:
                return f"{year}: no output produced"
        except Exception as e:
            last_err = e
            out_zip.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(20 * attempt)
            else:
                return f"{year}: FAILED after {retries} attempts -- {e}"
    return f"{year}: FAILED -- {last_err}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=3,
                   help="Parallel CDS requests (default 3; CDS allows up to ~10)")
    p.add_argument("--years", type=str, default=None,
                   help="Comma-separated years to fill (default 2015-2021)")
    args = p.parse_args()

    years = [int(y) for y in args.years.split(",")] if args.years else GAP_YEARS
    print(f"ORAS5 gap-fill: {years} into {ORAS5_DIR}")
    print(f"Variables: {TARGET_VARS}")
    print(f"Workers: {args.workers}")
    print()

    # Pre-flight: how many to actually fetch?
    needed = [y for y in years if not year_complete(y)]
    if not needed:
        print("All requested years already complete. Nothing to do.")
        return
    print(f"To fetch: {needed}")
    print(f"Already complete (skip): {sorted(set(years) - set(needed))}\n")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_year, y): y for y in needed}
        for done, fut in enumerate(as_completed(futures), 1):
            print(f"  [{done}/{len(needed)}] {fut.result()}")

    print(f"\nElapsed {time.time() - t0:.0f}s")
    # Final inventory
    print("\nPost-fill coverage:")
    for v in TARGET_VARS:
        token = NATIVE_TOKENS[v]
        for y in years:
            present = len(months_present(token, y))
            flag = "" if present == 12 else f"  <- {12-present} months missing"
            print(f"  {token} {y}: {present}/12{flag}")


if __name__ == "__main__":
    main()
