"""Fetch ERA5 monthly Z500 (500-hPa geopotential height) for NH and
SH midlatitudes, 2000-2023.  Step 4.1 of the Quiescence Signature
theory paper: atmospheric extension test.
"""
from __future__ import annotations

from pathlib import Path
import cdsapi


OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "external" / "era5_z500"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "era5_z500_global_midlat_2000_2023.nc"


def main():
    if OUT.exists() and OUT.stat().st_size > 1e6:
        print(f"already on disk: {OUT} ({OUT.stat().st_size/1e6:.0f} MB)")
        return
    c = cdsapi.Client()
    # Northern hemisphere 60N-30N AND southern hemisphere 30S-60S.
    # We fetch one global slab and subset later.
    years = [str(y) for y in range(2000, 2024)]
    months = [f"{m:02d}" for m in range(1, 13)]
    print(f"Fetching ERA5 monthly Z500 global midlat 2000-2023 to {OUT}")
    c.retrieve(
        "reanalysis-era5-pressure-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": "geopotential",
            "pressure_level": "500",
            "year": years,
            "month": months,
            "time": "00:00",
            "format": "netcdf",
            "area": [60, -180, -60, 180],   # global -60 to 60 lat
            "grid": [1.0, 1.0],
        },
        str(OUT),
    )
    print(f"Wrote {OUT} ({OUT.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
