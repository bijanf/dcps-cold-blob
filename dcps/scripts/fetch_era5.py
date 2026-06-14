"""Fetch ERA5 monthly NA-basin surface forcing 2000-2023 for MAJOR 8.

Requested variables (all monthly means on regular 0.25 deg grid):
  - sshf : surface sensible heat flux (W m^-2)
  - slhf : surface latent heat flux (W m^-2)
  - ewss : eastward turbulent surface stress (N m^-2)
  - nsss : northward turbulent surface stress (N m^-2)
  - tp   : total precipitation (m / month)
  - e    : evaporation (m / month)

CDS API needs ~/.cdsapirc (user has it; verified).  Output:
  data/external/era5/era5_na_monthly_2000_2023.nc  (~2-5 GB)
"""
from __future__ import annotations

from pathlib import Path

import cdsapi

DATA_DIR = Path.home() / "Documents/NEW_Theory/data/external/era5"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT = DATA_DIR / "era5_na_monthly_2000_2023.nc"


def main():
    if OUT.exists() and OUT.stat().st_size > 1e6:
        print(f"ERA5 already on disk: {OUT} ({OUT.stat().st_size/1e6:.0f} MB)")
        return
    c = cdsapi.Client()
    # North Atlantic basin extent matching the Quiescence analysis:
    # lat 0..75 N, lon 80W..0E.
    area = [75, -80, 0, 0]   # [north, west, south, east]
    years = [str(y) for y in range(2000, 2024)]
    months = [f"{m:02d}" for m in range(1, 13)]

    print(f"Fetching ERA5 monthly NA-basin 2000-2023 to {OUT}")
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": [
                "surface_sensible_heat_flux",
                "surface_latent_heat_flux",
                "eastward_turbulent_surface_stress",
                "northward_turbulent_surface_stress",
                "total_precipitation",
                "evaporation",
            ],
            "year": years,
            "month": months,
            "time": "00:00",
            "format": "netcdf",
            "area": area,
            "grid": [0.5, 0.5],   # match the 1/2-deg Quiescence target
        },
        str(OUT),
    )
    print(f"Wrote {OUT} ({OUT.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
