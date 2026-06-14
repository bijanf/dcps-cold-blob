"""End-to-end Phase 1 pipeline: ORAS5 SST + SSH -> 2 deg NA cache.

Usage:
    python scripts/run_phase1.py [--smoke]

--smoke runs on a 24-month subset and writes a separate cache file so the full
pipeline isn't disturbed.

Output: cache/phase1_oras5_NA_2deg.nc with variables sst_anom and ssh_anom on a
(time, lat, lon) regular 2 deg grid, plus the regridded raw fields (sst_raw,
ssh_raw) for reference and provenance metadata for the gap in 2015-2021.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import xarray as xr

from dcps import io, regrid, anomaly
from dcps.config import (
    CACHE_DIR,
    GRID_DEG,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    ORAS5_VARS,
    PHASE1_OUTPUT,
    TIME_END,
    TIME_START,
)


def _process_one(var_token: str, alias: str, t0: str, t1: str) -> tuple[xr.DataArray, xr.DataArray]:
    print(f"  [{alias}] open ORAS5 {var_token} {t0}..{t1}")
    tic = time.time()
    raw = io.load_oras5_var(var_token, start=t0, end=t1)
    print(f"        loaded shape={tuple(raw.sizes.values())} in {time.time() - tic:.1f}s")

    print(f"  [{alias}] regrid to {GRID_DEG} deg")
    tic = time.time()
    raw_regrid = regrid.regrid_to_2deg(raw, grid_deg=GRID_DEG).rename(f"{alias}_raw")
    print(f"        regridded shape={tuple(raw_regrid.sizes.values())} in {time.time() - tic:.1f}s")

    print(f"  [{alias}] preprocess: climatology -> detrend -> bandpass")
    tic = time.time()
    anom = anomaly.preprocess_pipeline(raw_regrid).rename(f"{alias}_anom")
    print(f"        preprocessed in {time.time() - tic:.1f}s")
    return raw_regrid, anom


def main(smoke: bool) -> None:
    if smoke:
        # 10 yr window: short enough to be quick, long enough that the 1-10 yr
        # bandpass has enough samples (filtfilt padlen ~ 27).
        t0, t1 = "2005-01-01", "2014-12-31"
        out_path = CACHE_DIR / "phase1_smoke.nc"
    else:
        t0, t1 = TIME_START, TIME_END
        out_path = PHASE1_OUTPUT

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Phase 1 driver: {t0}..{t1} -> {out_path}")
    print(f"  domain {LAT_MIN}-{LAT_MAX} N, {LON_MIN}-{LON_MAX} E, grid {GRID_DEG} deg")

    sst_raw, sst_anom = _process_one(ORAS5_VARS["sst"], "sst", t0, t1)
    ssh_raw, ssh_anom = _process_one(ORAS5_VARS["ssh"], "ssh", t0, t1)

    out = xr.Dataset(
        {
            "sst_raw": sst_raw,
            "ssh_raw": ssh_raw,
            "sst_anom": sst_anom,
            "ssh_anom": ssh_anom,
        }
    )
    out.attrs["title"] = "DCPS Phase 1: ORAS5 North Atlantic, regridded + bandpass anomalies"
    out.attrs["source"] = "ORAS5 reanalysis (ECMWF), local ARDP cache"
    out.attrs["domain"] = (
        f"North Atlantic basin: {LAT_MIN}-{LAT_MAX} N, {LON_MIN}-{LON_MAX} E"
    )
    out.attrs["grid_deg"] = GRID_DEG
    out.attrs["analysis_window"] = f"{t0} to {t1}"
    out.attrs["pipeline"] = (
        "load -> NA mask -> 2 deg box-mean (cos lat) -> climatology removal -> "
        "linear detrend -> 4th-order Butterworth bandpass 1-10 yr (filtfilt)"
    )
    out.attrs["data_gap_note"] = (
        "ORAS5 SST/SSH cache has a 2015-2021 gap (84 months) at the local ARDP "
        "snapshot used here. Phase 1 uses the contiguous 1958-2014 segment to "
        "avoid corrupting bandpass/Hilbert across the gap. Other ORAS5 variables "
        "(salinity, velocity) are complete; only the surface 2D fields are gappy."
    )

    print(f"Writing {out_path} ({sum(v.nbytes for v in out.data_vars.values()) / 1e6:.1f} MB raw)")
    enc = {v: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
           for v in out.data_vars}
    out.to_netcdf(out_path, encoding=enc)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB on disk).")
    print("\nVariables:")
    for name, da in out.data_vars.items():
        print(f"  {name}: shape={tuple(da.sizes.values())}, dtype={da.dtype}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="2-year subset for sanity")
    args = p.parse_args()
    main(args.smoke)
