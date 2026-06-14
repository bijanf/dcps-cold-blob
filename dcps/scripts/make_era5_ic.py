"""Build a local ERA5 IC NetCDF for DLESyMLatLon rollouts.

Compute nodes at PIK have no outbound internet, so we pre-fetch the IC subset
on the login node via earth2studio's ARCO (Google Cloud) source and save a
self-contained NetCDF that the rollout pipeline can read via DataArrayFile.

The 11 raw lat/lon variables + 9 history lead-times match
``DLESyMLatLon.input_coords()`` exactly.

Usage:
    python make_era5_ic.py --date 2010-01-01
    python make_era5_ic.py --date 1990-01-01 --out /custom/path.nc
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

DEFAULT_OUT_DIR = Path("/p/projects/poem/fallah")

# Exactly the 11 raw variables DLESyMLatLon.input_coords() returns
# (z500, z1000, t2m, tcwv, t850, z250, sst, u10m, v10m, z300, z700).
RAW_VARS = ("z500", "z1000", "t2m", "tcwv", "t850", "z250",
            "sst", "u10m", "v10m", "z300", "z700")

# History lead-times the model consumes (-48 .. 0 hours, 6 h step).
LEAD_HOURS = (-48, -42, -36, -30, -24, -18, -12, -6, 0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", required=True,
                   help="IC date as YYYY-MM-DD (UTC midnight assumed)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"Output NetCDF path; default {DEFAULT_OUT_DIR}/era5_ic_<DATE>.nc")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output file")
    args = p.parse_args()

    ic_dt = datetime.fromisoformat(args.date)
    out = args.out or DEFAULT_OUT_DIR / f"era5_ic_{ic_dt.date().isoformat()}.nc"

    if out.exists() and not args.force:
        print(f"[make_era5_ic] {out} already exists ({out.stat().st_size:,} B); "
              f"use --force to overwrite", flush=True)
        return 0

    from earth2studio.data import ARCO, datasource_to_file

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[make_era5_ic] fetching {len(RAW_VARS)} vars × "
          f"{len(LEAD_HOURS)} leads at {ic_dt} -> {out}", flush=True)

    arco = ARCO(cache=True, verbose=False)
    lead = np.array([np.timedelta64(h, "h") for h in LEAD_HOURS])
    datasource_to_file(
        str(out), arco,
        time=[ic_dt], variable=list(RAW_VARS),
        lead_time=lead, backend="netcdf",
    )

    size_gb = out.stat().st_size / (1024 ** 3)
    print(f"[make_era5_ic] wrote {out} ({size_gb:.2f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
