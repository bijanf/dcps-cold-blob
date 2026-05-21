"""Compute (Q, basin-mean EKE) for reanalysis sources using the full
locally available native period (not the 2000-2023 2-deg cache).

For GLORYS12 we use the per-year native 1/12 deg files in
/home/bijanf/Documents/AMOC_renalysis/data/glorys12/ (1993-2025).  We
take the most recent 30 contiguous years that fit the canonical
WINDOW_YEARS=30 standard, i.e. 1995-2024.

For ORAS5 we use the existing 2-deg basin cache (2000-2023, 24 yr).
Native ORAS5 is not on disk locally; longer span would need download.

Output: dcps/cache/eke_timeseries/<source>_atlantic_obs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR, PKG_ROOT
sys.path.insert(0, str(PKG_ROOT / "scripts"))
from holocene_q_pilot import (  # noqa: E402
    _basin_subset_2deg, _Q_for_window,
)

GLORYS_NATIVE_DIR = Path("/home/bijanf/Documents/AMOC_renalysis/data/glorys12")
ORAS5_NATIVE_DIR = Path("/home/bijanf/Documents/AMOC_renalysis/data/oras5")
ORAS5_2DEG = CACHE_DIR / "phase1_oras5_NA_2deg.nc"
EKE_TS_DIR = CACHE_DIR / "eke_timeseries"


def _process_oras5_native(year_start: int, year_end: int):
    """Stitch ORAS5 native per-month files for sosstsst and sossheig
    over [year_start, year_end].  Returns (sst, ssh) DataArrays with
    (time, lat, lon) — actually ORAS5 native uses curvilinear (y, x)
    with 2-D nav_lat/nav_lon coords, which _basin_subset_2deg handles
    via its 2-D path."""
    sst_paths, ssh_paths = [], []
    for y in range(year_start, year_end + 1):
        for m in range(1, 13):
            tag = f"{y}{m:02d}"
            # files may be either CONS_ or OPER_ variant; glob both
            ss_t = sorted(ORAS5_NATIVE_DIR.glob(
                f"sosstsst_control_monthly_highres_2D_{tag}_*.nc"))
            ss_h = sorted(ORAS5_NATIVE_DIR.glob(
                f"sossheig_control_monthly_highres_2D_{tag}_*.nc"))
            if ss_t and ss_h:
                sst_paths.append(ss_t[0])
                ssh_paths.append(ss_h[0])
    print(f"  found {len(sst_paths)} months of SST and "
          f"{len(ssh_paths)} of SSH")
    if not sst_paths:
        raise RuntimeError("no ORAS5 native files matched")

    sst = xr.open_mfdataset(sst_paths, combine="nested",
                               concat_dim="time_counter",
                               parallel=False, decode_cf=True,
                               data_vars=["sosstsst"])["sosstsst"]
    ssh = xr.open_mfdataset(ssh_paths, combine="nested",
                               concat_dim="time_counter",
                               parallel=False, decode_cf=True,
                               data_vars=["sossheig"])["sossheig"]
    sst = sst.rename({"time_counter": "time"})
    ssh = ssh.rename({"time_counter": "time"})
    # rename nav_lat / nav_lon to lat / lon so _basin_subset_2deg
    # picks them up (it looks for "lat"/"latitude" coord names)
    sst = sst.rename({"nav_lat": "lat", "nav_lon": "lon"})
    ssh = ssh.rename({"nav_lat": "lat", "nav_lon": "lon"})
    # Subset to a generous North Atlantic bounding-box on y/x to keep
    # memory bounded.  The 2-D lat/lon are inside the basin filter
    # downstream so we can be generous here.
    # The ORAS5 tripolar grid maps the North Atlantic basin to
    # y ~ [500, 960], x ~ [820, 1160] (verified empirically by
    # checking where (1<lat<76 AND -80<lon<0) lands).  Generous
    # margins to ensure all NA cells are kept.
    sst = sst.isel(y=slice(500, 970), x=slice(820, 1160)).load()
    ssh = ssh.isel(y=slice(500, 970), x=slice(820, 1160)).load()
    return sst, ssh


def _process_glorys12_native(year_start: int, year_end: int):
    """Concatenate yearly GLORYS12 native files, take surface
    thetao + zos, return as DataArrays on (time, lat, lon)."""
    sst_list, ssh_list = [], []
    for y in range(year_start, year_end + 1):
        p = GLORYS_NATIVE_DIR / f"glorys12_{y}.nc"
        if not p.exists():
            print(f"  [{y}] file missing, skipping")
            continue
        try:
            ds = xr.open_dataset(p)
        except Exception as e:
            print(f"  [{y}] open FAILED: {e}")
            continue
        sst = (ds["thetao"]
                 .isel(depth=0, drop=True)
                 .rename({"latitude": "lat", "longitude": "lon"}))
        ssh = ds["zos"].rename({"latitude": "lat", "longitude": "lon"})
        # subset to North Atlantic box
        sst = sst.sel(lat=slice(0, 76), lon=slice(-80, 0))
        ssh = ssh.sel(lat=slice(0, 76), lon=slice(-80, 0))
        sst_list.append(sst.load())
        ssh_list.append(ssh.load())
        print(f"  [{y}] loaded sst {sst.shape} ssh {ssh.shape}")
    if not sst_list:
        raise RuntimeError("no GLORYS12 native files loaded")
    sst_full = xr.concat(sst_list, dim="time")
    ssh_full = xr.concat(ssh_list, dim="time")
    return sst_full, ssh_full


def _process_one(source: str):
    if source == "GLORYS12":
        # full native record we have: 1993-2025 -> 30-yr window 1995-2024
        year_start, year_end = 1995, 2024
        print(f"GLORYS12 native, window {year_start}-{year_end}")
        sst, ssh = _process_glorys12_native(year_start, year_end)
        t0 = str(sst["time"].values[0])[:10]
        t1 = str(sst["time"].values[-1])[:10]
    elif source == "ORAS5":
        # Native ORAS5 from AMOC_renalysis/data/oras5 (1958-) — same
        # 30-yr window as GLORYS12 (1995-2024) for direct comparability.
        year_start, year_end = 1995, 2024
        print(f"ORAS5 native, window {year_start}-{year_end}")
        sst, ssh = _process_oras5_native(year_start, year_end)
        t0 = str(sst["time"].values[0])[:10]
        t1 = str(sst["time"].values[-1])[:10]
    else:
        raise ValueError(f"unknown source: {source}")

    print(f"  computing Q on full window ...")
    Q, n_cells = _Q_for_window(sst, ssh, "atlantic")
    print(f"  Q = {Q:+.4f}  (n_cells = {n_cells})")

    zos_2d = _basin_subset_2deg(ssh, "atlantic")
    ssh_mean = zos_2d.mean("time")
    grad2 = (ssh_mean.differentiate("lat") ** 2
              + ssh_mean.differentiate("rlon") ** 2)
    eke_abs = float(grad2.mean(skipna=True).values)
    print(f"  basin-mean EKE (absolute) = {eke_abs:.6f}")

    pi_means = []
    for p in sorted(EKE_TS_DIR.glob("*_atlantic_eke_ts.json")):
        if any(tag in p.stem for tag in ("ssp245","ssp370","ssp126",
                                            "ssp119","1pctCO2",
                                            "abrupt-4xCO2","obs")):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        mu = d.get("pi_eke_mean")
        if mu and np.isfinite(mu) and mu > 0:
            pi_means.append(float(mu))
    pi_mean_ensemble = float(np.mean(pi_means))
    rel_eke = eke_abs / pi_mean_ensemble
    print(f"  rel-EKE = {rel_eke:.4f}  (cross-CMIP6 piControl mean "
          f"{pi_mean_ensemble:.6f}, N={len(pi_means)})")

    out = EKE_TS_DIR / f"{source}_atlantic_obs.json"
    out.write_text(json.dumps(dict(
        source=source,
        basin="atlantic",
        time_range_start=t0,
        time_range_end=t1,
        year_window=[year_start, year_end],
        n_years=year_end - year_start + 1,
        Q=float(Q),
        n_cells_used=int(n_cells),
        eke_abs=float(eke_abs),
        rel_eke=float(rel_eke),
        pi_eke_mean_cross_cmip6=pi_mean_ensemble,
        n_cmip6_models_used=len(pi_means),
    ), indent=2))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="GLORYS12",
                     choices=["GLORYS12", "ORAS5"])
    args = ap.parse_args()
    _process_one(args.source)


if __name__ == "__main__":
    main()
