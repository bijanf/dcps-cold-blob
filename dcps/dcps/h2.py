"""H2 + H2' tests: spatiotemporal Chimera detection.

Primary H2 (pre-registered, Methods):
    For each year y, identify the largest connected component of cells with
    r_loc <= 0.5 within the Cold-Blob box (45-60 N, 40-15 W) -- area A_inc(y).
    Identify the largest connected component of r_loc >= 0.8 within the
    subtropical box (20-40 N, 60-10 W) -- area A_coh_ST(y).
    Supported iff:
        A_inc(y) >= 1e6 km^2  AND  A_coh_ST(y) >= 3e6 km^2
        in at least 50% of years 2004-present.

Fallback H2' (pre-registered, Methods):
    Compute regional Kuramoto magnitudes R_CB(t), R_ST(t) and
    inter-region phase-coupling concentration C = |<exp i (Phi_CB - Phi_ST)>|_t.
    Supported iff:
        <R_CB>, <R_ST> >= 0.5  AND  C <= 0.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from scipy.ndimage import label as cc_label

# Geographic boxes per Methods
COLD_BLOB_BOX = dict(lat=(45, 60), lon=(-40, -15))
SUBTROPICAL_BOX = dict(lat=(20, 40), lon=(-60, -10))

# Pre-registered thresholds (Methods)
LOC_INC_THRESH = 0.5
LOC_COH_THRESH = 0.8
A_INC_MIN_KM2 = 1e6
A_COH_MIN_KM2 = 3e6
H2_PERSISTENCE = 0.5

H2P_R_THRESH = 0.5
H2P_C_THRESH = 0.2

_EARTH_R_KM = 6371.0


def cell_area_km2(lat_centres: np.ndarray, lon_centres: np.ndarray) -> np.ndarray:
    """Approximate cell area in km^2 for a regular lat/lon grid."""
    grid_deg = float(np.diff(lat_centres).mean())
    dlat_km = grid_deg * (np.pi / 180.0) * _EARTH_R_KM
    dlon_km = grid_deg * (np.pi / 180.0) * _EARTH_R_KM * \
        np.cos(np.deg2rad(lat_centres))
    area_1d = dlat_km * dlon_km                        # (n_lat,)
    return np.broadcast_to(area_1d[:, None], (lat_centres.size, lon_centres.size))


@dataclass
class H2Result:
    a_inc_per_year: dict        # year -> km^2
    a_coh_per_year: dict
    fraction_years_passing: float
    n_years: int
    pass_h2: bool = False
    overlap_start: str = ""
    overlap_end: str = ""


@dataclass
class H2pResult:
    R_CB_mean: float
    R_ST_mean: float
    C: float
    mean_phase_diff_deg: float
    n_months: int
    pass_h2p: bool = False
    overlap_start: str = ""
    overlap_end: str = ""


def _select_box(da: xr.DataArray, lat_range, lon_range) -> xr.DataArray:
    return da.sel(lat=slice(*lat_range), lon=slice(*lon_range))


def _largest_cc_area(mask: np.ndarray, area_2d: np.ndarray) -> float:
    """km^2 of the largest connected (4-neighbour) True component in mask."""
    if not mask.any():
        return 0.0
    lab, n = cc_label(mask)
    if n == 0:
        return 0.0
    sizes = np.array([
        area_2d[lab == k].sum() for k in range(1, n + 1)
    ])
    return float(sizes.max())


def test_h2(
    r_loc: xr.DataArray,
    overlap_start: str = "2004-04-01",
    overlap_end: str = "2014-12-31",
) -> H2Result:
    """Pre-registered H2 area-and-threshold rule."""
    rl = r_loc.sel(time=slice(overlap_start, overlap_end))
    if rl.sizes["time"] < 12:
        raise ValueError("H2 needs at least one year of overlap.")

    lat = rl.lat.values
    lon = rl.lon.values
    area_full = cell_area_km2(lat, lon)

    # Pre-compute box masks on the full grid
    lat_idx_cb = np.where(
        (lat >= COLD_BLOB_BOX["lat"][0]) & (lat <= COLD_BLOB_BOX["lat"][1]))[0]
    lon_idx_cb = np.where(
        (lon >= COLD_BLOB_BOX["lon"][0]) & (lon <= COLD_BLOB_BOX["lon"][1]))[0]
    lat_idx_st = np.where(
        (lat >= SUBTROPICAL_BOX["lat"][0]) & (lat <= SUBTROPICAL_BOX["lat"][1]))[0]
    lon_idx_st = np.where(
        (lon >= SUBTROPICAL_BOX["lon"][0]) & (lon <= SUBTROPICAL_BOX["lon"][1]))[0]

    a_inc, a_coh = {}, {}
    years = sorted({int(str(t)[:4]) for t in rl["time"].values})
    for y in years:
        slab = rl.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))
        if slab.sizes["time"] == 0:
            continue
        annual_mean = slab.mean("time").values        # (lat, lon)

        # Cold Blob: incoherent component in CB box
        cb = annual_mean[np.ix_(lat_idx_cb, lon_idx_cb)]
        cb_mask = np.isfinite(cb) & (cb <= LOC_INC_THRESH)
        cb_area = area_full[np.ix_(lat_idx_cb, lon_idx_cb)]
        a_inc[y] = _largest_cc_area(cb_mask, cb_area)

        # Subtropical: coherent component in ST box
        st = annual_mean[np.ix_(lat_idx_st, lon_idx_st)]
        st_mask = np.isfinite(st) & (st >= LOC_COH_THRESH)
        st_area = area_full[np.ix_(lat_idx_st, lon_idx_st)]
        a_coh[y] = _largest_cc_area(st_mask, st_area)

    n_pass = sum(
        1 for y in years
        if a_inc[y] >= A_INC_MIN_KM2 and a_coh[y] >= A_COH_MIN_KM2
    )
    frac = n_pass / max(1, len(years))
    return H2Result(
        a_inc_per_year=a_inc,
        a_coh_per_year=a_coh,
        fraction_years_passing=float(frac),
        n_years=len(years),
        pass_h2=bool(frac >= H2_PERSISTENCE),
        overlap_start=overlap_start,
        overlap_end=overlap_end,
    )


def regional_phase(phase_field: xr.DataArray, lat_range, lon_range) -> tuple[xr.DataArray, xr.DataArray]:
    """Regional mean complex phase: returns (R_region(t), Phi_region(t))."""
    box = _select_box(phase_field, lat_range, lon_range)
    z = np.exp(1j * box.values)
    z = np.where(np.isfinite(box.values), z, 0.0 + 0.0j)
    n_valid = np.isfinite(box.values).sum(axis=(1, 2))
    z_sum = z.sum(axis=(1, 2))
    R = np.abs(z_sum) / np.maximum(n_valid, 1)
    angle = np.angle(z_sum)
    coords = {"time": box["time"].values}
    return (
        xr.DataArray(R.astype(np.float32), dims=("time",), coords=coords, name="R_region"),
        xr.DataArray(angle.astype(np.float32), dims=("time",), coords=coords, name="phase_region"),
    )


def test_h2_prime(
    phase_field: xr.DataArray,
    overlap_start: str = "2004-04-01",
    overlap_end: str = "2014-12-31",
) -> H2pResult:
    """Inter-region phase-coupling concentration test."""
    pf = phase_field.sel(time=slice(overlap_start, overlap_end))
    R_cb, phi_cb = regional_phase(pf, COLD_BLOB_BOX["lat"], COLD_BLOB_BOX["lon"])
    R_st, phi_st = regional_phase(pf, SUBTROPICAL_BOX["lat"], SUBTROPICAL_BOX["lon"])

    dphi = (phi_cb.values - phi_st.values + np.pi) % (2 * np.pi) - np.pi
    z = np.exp(1j * dphi)
    C = float(np.abs(z.mean()))
    mean_abs_dphi_deg = float(np.rad2deg(np.abs(dphi).mean()))

    pass_h2p = (
        float(R_cb.mean()) >= H2P_R_THRESH
        and float(R_st.mean()) >= H2P_R_THRESH
        and C <= H2P_C_THRESH
    )

    return H2pResult(
        R_CB_mean=float(R_cb.mean()),
        R_ST_mean=float(R_st.mean()),
        C=C,
        mean_phase_diff_deg=mean_abs_dphi_deg,
        n_months=int(R_cb.size),
        pass_h2p=bool(pass_h2p),
        overlap_start=overlap_start,
        overlap_end=overlap_end,
    )
