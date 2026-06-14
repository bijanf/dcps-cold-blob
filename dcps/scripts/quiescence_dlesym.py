"""Cross-prediction test of the Quiescence law on NVIDIA DLESyM v1 ERA5.

Registered as prediction P6 in ``PRE_REGISTRATION_P6.md`` at commit
``7960f9f83bd332177f24b7dbc5390aad02d16c74``.  The protocol is:

1. Autoregressively roll out DLESyM from an ERA5 initial condition
   at ``--ic`` for ``--years`` simulated years.
2. Regrid the HEALPix monthly SST output to the canonical 2-degree
   basin grid expected by ``multi_basin_quiescence.py``.
3. Run the unchanged Quiescence pipeline on the DLESyM SST.
4. Cross-predict against the observed GLORYS12 climatological EKE
   field and report ``Q_X`` with its spatial-block-permutation null
   plus the non-linear fit of ``(1 + tau * EKE) ** -0.5``.

Heavy GPU-side dependencies (``earth2studio``, ``healpy``, ``torch``)
are imported lazily inside the functions that need them, so this
module imports cleanly on a CPU-only workstation for static checks.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

# Reuse the unchanged P1-P5 pipeline.  These imports happen at module
# load time on purpose -- they document the reuse contract and a static
# check catches any rename of the entry points.
from multi_basin_quiescence import (
    BASINS,
    basin_target_grid,
    instantaneous_phase,
    local_r_mean,
    preprocess_anomaly,
    regrid_basin,
)

from dcps.spatial_stats import spatial_block_permutation

# Pre-registered constants.
TAU_OBS_NA = 144.0
Q_SUPPORT_THRESHOLD = 0.20
Q_FALSIFY_THRESHOLD = 0.10
TAU_FACTOR_SUPPORT = 3.0
TAU_FACTOR_FALSIFY = 10.0
P_PERM_SUPPORT = 0.01
P_PERM_FALSIFY = 0.05
CHI2_NU_FALSIFY = 5.0

# Cresswell-Clay 2025 drift bound: |global-mean SST trend over rollout|
# must stay below this many K/yr (Table 2 of the published paper).
DRIFT_K_PER_YEAR = 0.02

# DLESyM v1 ERA5 input variable list (matches the checkpoint's training).
DLESYM_IC_VARIABLES = (
    "z250", "z500", "z1000", "t850", "tau300-700",
    "t2m", "tcwv", "ws10m", "sst",
)


def _ic_source(ic_date: str, era5_mirror: Path | None) -> Any:
    """Construct an earth2studio data source for the IC.

    Priority order:
      1. ``LocalERA5DataSource`` if ``era5_mirror`` is given and covers
         the requested variables at ``ic_date``.
      2. ``earth2studio.data.ARCO`` (Google-Cloud ARCO-ERA5 mirror;
         requires login-node internet, single-timestamp pull).
      3. ``earth2studio.data.CDS`` (Copernicus CDS API; slowest).
    """
    from earth2studio.data import ARCO, CDS

    if era5_mirror is not None:
        try:
            from earth2studio.data import DataArrayFile
        except ImportError:
            DataArrayFile = None  # type: ignore
        if DataArrayFile is not None and Path(era5_mirror).exists():
            return DataArrayFile(str(era5_mirror))

    try:
        return ARCO()
    except Exception:
        return CDS()


def _validate_drift(sst_ds: xr.Dataset, years: int) -> tuple[float, bool]:
    """Return (|trend K/yr|, exceeded_bound)."""
    sst = sst_ds["sst"] if "sst" in sst_ds else sst_ds[list(sst_ds.data_vars)[0]]
    spatial_dims = [d for d in sst.dims if d != "time"]
    global_mean = sst.mean(dim=spatial_dims, skipna=True).values
    t = np.arange(global_mean.size, dtype=np.float64)
    slope, _ = np.polyfit(t, global_mean, 1)
    trend_k_per_year = abs(slope) * (global_mean.size / max(years, 1))
    return float(trend_k_per_year), trend_k_per_year > DRIFT_K_PER_YEAR


def rollout_dlesym(
    years: int,
    ic_date: str,
    checkpoint: Path,
    out_path: Path,
    era5_mirror: Path | None = None,
    atmos_member: int = 0,
    ocean_member: int = 0,
) -> Path:
    """Autoregressively roll out NVIDIA DLESyM v1 ERA5.

    Writes monthly-mean SST on the native HEALPix grid to ``out_path``
    as NetCDF.  Returns the path.
    """
    from earth2studio.models.px.dlesym import DLESyMLatLon
    from earth2studio.models.auto.package import Package
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "rollout_dlesym requires CUDA; submit via run_dlesym_p6.sbatch "
            "or another GPU node."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # DLESyMLatLon takes raw ERA5 lat/lon variables (z300, z700, u10m, v10m,
    # …) and derives tau300-700 and ws10m internally; base DLESyM expects
    # pre-projected HEALPix input with the derived vars already present.
    # ARCO/local NetCDF only provides raw vars, so we use the LatLon variant.
    package = Package(str(checkpoint)) if checkpoint is not None \
        else DLESyMLatLon.load_default_package()
    model = DLESyMLatLon.load_model(
        package,
        atmos_model_idx=atmos_member,
        ocean_model_idx=ocean_member,
    ).to("cuda")
    print(f"[P6] loaded DLESyMLatLon member (atmos={atmos_member}, "
          f"ocean={ocean_member})", flush=True)

    ic_dt = datetime.fromisoformat(ic_date)
    data = _ic_source(ic_date, era5_mirror)

    # Replace earth2studio.run.deterministic with a custom rollout.
    # `deterministic` was designed for a uniform-shape iterator: step 0 yields
    # the IC context (9 lead-times) while later steps yield 16 future leads,
    # so its NetCDF4Backend buffer (sized for 16) mis-broadcasts on step 0.
    # We iterate manually, skip the IC step, and accumulate monthly-mean SST
    # on the model's lat/lon grid.
    from earth2studio.data import fetch_data

    in_coords = model.input_coords()
    out_coords_step = model.output_coords(in_coords)
    sst_idx = list(out_coords_step["variable"]).index("sst")
    lats = np.asarray(out_coords_step["lat"], dtype=np.float64)
    lons = np.asarray(out_coords_step["lon"], dtype=np.float64)
    leads_per_step = len(out_coords_step["lead_time"])  # 16
    step_hours = 6  # atmos step
    total_hours = years * 365 * 24
    n_outer_steps = int(np.ceil(total_hours / (leads_per_step * step_hours)))

    x_ic, c_ic = fetch_data(
        source=data,
        time=[ic_dt],
        variable=in_coords["variable"],
        lead_time=in_coords["lead_time"],
        device="cuda",
    )

    print(f"[P6] custom rollout: years={years} outer_steps={n_outer_steps} "
          f"leads_per_step={leads_per_step}", flush=True)

    monthly_sum: dict[np.datetime64, np.ndarray] = {}
    monthly_count: dict[np.datetime64, int] = {}

    iterator = model.create_iterator(x_ic, c_ic)
    # Step 0 == IC history; skip.
    next(iterator)
    for step in range(n_outer_steps):
        try:
            x_out, c_out = next(iterator)
        except StopIteration:
            break
        leads = c_out["lead_time"]
        sst = x_out[0, :, sst_idx].detach().cpu().numpy()  # (lead, lat, lon)
        for i, lead in enumerate(leads):
            ts = np.datetime64(ic_dt) + lead
            month = ts.astype("datetime64[M]")
            if month not in monthly_sum:
                monthly_sum[month] = sst[i].astype(np.float64).copy()
                monthly_count[month] = 1
            else:
                monthly_sum[month] += sst[i].astype(np.float64)
                monthly_count[month] += 1
        if (step + 1) % 20 == 0:
            print(f"[P6] step {step+1}/{n_outer_steps} "
                  f"({(step+1)*leads_per_step*step_hours/24:.1f} days simulated)",
                  flush=True)

    months = np.array(sorted(monthly_sum.keys()))
    sst_monthly = np.stack(
        [monthly_sum[m] / monthly_count[m] for m in months], axis=0
    ).astype(np.float32)

    da = xr.DataArray(
        sst_monthly,
        dims=("time", "lat", "lon"),
        coords={"time": months.astype("datetime64[ns]"), "lat": lats, "lon": lons},
        name="sst",
    )
    da.to_netcdf(str(out_path), engine="netcdf4",
                 encoding={"sst": {"zlib": True, "complevel": 4}})
    print(f"[P6] wrote monthly-mean SST {sst_monthly.shape} -> {out_path}",
          flush=True)

    return out_path


def healpix_to_basin_grid(
    dlesym_sst_path: Path,
    basin: str = "atlantic",
    var_name: str = "sst",
) -> xr.DataArray:
    """Regrid DLESyM SST to the canonical 2-degree basin grid.

    Handles either HEALPix (flat spatial dim) or lat/lon (the format
    emitted by the DLESyMLatLon custom rollout) automatically. Reuses
    ``regrid_basin`` from ``multi_basin_quiescence.py``.
    """
    ds = xr.open_dataset(str(dlesym_sst_path))
    sst = ds[var_name] if var_name in ds else ds[list(ds.data_vars)[0]]

    # Path A: lat/lon output (DLESyMLatLon custom rollout)
    if "lat" in sst.dims and "lon" in sst.dims:
        lat = sst["lat"].values.astype(np.float64)
        lon = sst["lon"].values.astype(np.float64)
        lon2d, lat2d = np.meshgrid(lon, lat)
        lat_1d = lat2d.ravel()
        lon_1d = ((lon2d.ravel() + 180.0) % 360.0) - 180.0
        rot_lon_1d = (lon_1d - BASINS[basin]["lon_offset"]) % 360.0
        sst_flat = sst.stack(cell=("lat", "lon")).transpose("time", "cell")
        return regrid_basin(sst_flat, lat_1d, rot_lon_1d, basin)

    # Path B: HEALPix output (native DLESyM, flat spatial dim)
    import healpy as hp
    spatial_dim = next(d for d in sst.dims if d != "time")
    npix = sst.sizes[spatial_dim]
    nside = hp.npix2nside(npix)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    lat_1d = (90.0 - np.degrees(theta)).astype(np.float64)
    lon_1d = np.degrees(phi).astype(np.float64)
    lon_1d = ((lon_1d + 180.0) % 360.0) - 180.0
    rot_lon_1d = (lon_1d - BASINS[basin]["lon_offset"]) % 360.0
    sst = sst.rename({spatial_dim: "cell"}).transpose("time", "cell")
    return regrid_basin(sst, lat_1d, rot_lon_1d, basin)


def _law_form(eke: np.ndarray, tau: float) -> np.ndarray:
    return (1.0 + tau * eke) ** -0.5


def _verdict(
    Q_X: float, p_perm: float, tau_fit: float, chi2_nu: float
) -> str:
    tau_ratio = tau_fit / TAU_OBS_NA if tau_fit > 0 else float("inf")
    tau_within_x3 = (1.0 / TAU_FACTOR_SUPPORT) <= tau_ratio <= TAU_FACTOR_SUPPORT
    tau_off_x10 = (
        tau_ratio > TAU_FACTOR_FALSIFY or tau_ratio < 1.0 / TAU_FACTOR_FALSIFY
    )
    if (
        Q_X >= Q_SUPPORT_THRESHOLD
        and p_perm < P_PERM_SUPPORT
        and tau_within_x3
        and chi2_nu <= CHI2_NU_FALSIFY
    ):
        return "supported"
    if Q_X < Q_FALSIFY_THRESHOLD and p_perm >= P_PERM_FALSIFY:
        return "falsified"
    if tau_off_x10:
        return "falsified"
    if chi2_nu > CHI2_NU_FALSIFY:
        return "falsified"
    return "deferred"


def cross_prediction_test(
    dlesym_sst_basin: xr.DataArray,
    glorys12_eke_clim: xr.DataArray,
    basin: str = "atlantic",
    n_permutations: int = 1000,
    block_km: float = 500.0,
) -> dict[str, Any]:
    """Compute the P6 cross-product Quiescence Index and decision metrics.

    Returns a dict with ``Q_X``, ``p_perm``, ``tau_fit``, ``tau_obs_NA``,
    ``chi2_nu``, ``verdict``, ``n_cells``, and ``drift_K_per_year`` (if
    supplied via ``dlesym_sst_basin.attrs``).
    """
    bp = preprocess_anomaly(dlesym_sst_basin)
    phase = instantaneous_phase(bp)
    r_loc = local_r_mean(phase, radius_km=500.0)  # time-mean already

    eke = glorys12_eke_clim.values
    rloc_v = r_loc.values
    if eke.shape != rloc_v.shape:
        raise ValueError(
            f"shape mismatch: EKE {eke.shape} vs <r_loc> {rloc_v.shape}; "
            "regrid both to the same basin_target_grid first."
        )

    valid = np.isfinite(eke) & np.isfinite(rloc_v)
    x = rloc_v[valid]
    y = eke[valid]

    Q_X = float(-np.corrcoef(x, y)[0, 1])

    lat_c, rlon_c, _, _ = basin_target_grid(basin)
    lat_grid, rlon_grid = np.meshgrid(lat_c, rlon_c, indexing="ij")
    lat_flat = lat_grid[valid]
    lon_flat = rlon_grid[valid] + BASINS[basin]["lon_offset"]

    _perm_result = spatial_block_permutation(
        x, y, lat_flat, lon_flat,
        block_km=block_km, B=n_permutations,
    )
    p_perm = _perm_result["p_perm"] if isinstance(_perm_result, dict) else _perm_result

    try:
        popt, _ = curve_fit(_law_form, y, x, p0=[TAU_OBS_NA], maxfev=5000)
        tau_fit = float(popt[0])
        residuals = x - _law_form(y, tau_fit)
        dof = max(x.size - 1, 1)
        chi2_nu = float(np.sum(residuals ** 2) / (dof * np.var(x, ddof=1)))
    except Exception:
        tau_fit = float("nan")
        chi2_nu = float("inf")

    verdict = _verdict(Q_X, float(p_perm), tau_fit, chi2_nu)

    return {
        "Q_X": Q_X,
        "p_perm": float(p_perm),
        "tau_fit": tau_fit,
        "tau_obs_NA": TAU_OBS_NA,
        "chi2_nu": chi2_nu,
        "verdict": verdict,
        "n_cells": int(x.size),
        "drift_K_per_year": dlesym_sst_basin.attrs.get(
            "drift_K_per_year", None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=("smoke", "full", "postprocess-only"),
                   default="smoke")
    p.add_argument("--years", type=int, default=1)
    p.add_argument("--ic", default="2010-01-01")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Local dir for nvidia/dlesym-v1-era5 checkpoint.")
    p.add_argument("--out", type=Path, default=Path("dlesym_p6_sst.nc"))
    p.add_argument("--era5-mirror", type=Path, default=None,
                   help="Optional local ERA5 mirror path (NetCDF/Zarr).")
    p.add_argument("--basin", default="atlantic", choices=tuple(BASINS.keys()))
    p.add_argument("--eke-clim", type=Path, default=None,
                   help="Path to cached GLORYS12 EKE climatology on the "
                        "2-degree basin grid (NetCDF).")
    p.add_argument("--verdict-json", type=Path, default=Path("p6_verdict.json"))
    p.add_argument("--atmos-member", type=int, default=0, choices=(0, 1, 2, 3),
                   help="DLESyM atmosphere checkpoint index (0..3).")
    p.add_argument("--ocean-member", type=int, default=0, choices=(0, 1, 2, 3),
                   help="DLESyM ocean checkpoint index (0..3).")
    args = p.parse_args(argv)

    if args.mode == "smoke":
        args.years = 1

    sst_path = args.out
    if args.mode in ("smoke", "full"):
        if args.checkpoint is None:
            p.error("--checkpoint is required for smoke/full modes")
        print(f"[P6] rolling out DLESyM for {args.years} yr from {args.ic}",
              flush=True)
        rollout_dlesym(
            years=args.years, ic_date=args.ic,
            checkpoint=args.checkpoint, out_path=sst_path,
            era5_mirror=args.era5_mirror,
            atmos_member=args.atmos_member,
            ocean_member=args.ocean_member,
        )
        with xr.open_dataset(sst_path) as raw:
            drift_k_per_year, exceeded = _validate_drift(raw, args.years)
        print(f"[P6] global-mean SST drift: {drift_k_per_year:.4f} K/yr "
              f"(bound {DRIFT_K_PER_YEAR})", flush=True)
        if exceeded:
            verdict = {
                "verdict": "deferred",
                "reason": "DLESyM drift exceeds Cresswell-Clay 2025 bound",
                "drift_K_per_year": drift_k_per_year,
            }
            args.verdict_json.write_text(json.dumps(verdict, indent=2))
            print(f"[P6] verdict=deferred (drift); wrote {args.verdict_json}")
            return 0

    if args.eke_clim is None:
        print("[P6] no --eke-clim path; skipping postprocess.")
        return 0

    print(f"[P6] regridding HEALPix -> {args.basin} 2-deg grid", flush=True)
    sst_basin = healpix_to_basin_grid(sst_path, basin=args.basin)

    eke = xr.open_dataarray(args.eke_clim)
    print("[P6] running cross-prediction test", flush=True)
    result = cross_prediction_test(sst_basin, eke, basin=args.basin)
    args.verdict_json.write_text(json.dumps(result, indent=2))
    print(f"[P6] verdict={result['verdict']}; "
          f"Q_X={result['Q_X']:+.3f}, p_perm={result['p_perm']:.4f}, "
          f"tau_fit={result['tau_fit']:.1f}; wrote {args.verdict_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
