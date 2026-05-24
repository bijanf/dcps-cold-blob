"""Bandpass sensitivity sweep for the Quiescence correlation.

Pre-registered:
  Sweep three Butterworth band corner choices:
      (lo_yr, hi_yr) in {(0.5, 15), (1.0, 10), (2.0, 20)}
  Re-run the cell-wise Quiescence correlation rho(<r_loc>, |grad SSH|)
  in three basins (Atlantic, Pacific, Southern Ocean).
  Report a 3 x 3 (basin x bandpass) table.

  Pass condition (locked): the published bandpass (1, 10) yr result
  is reproduced to within Delta_rho = 0.05 in every alternative
  bandpass.
"""

from __future__ import annotations

import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.multiple_testing import fdr_bh
from dcps.nature_style import apply_nature_style
apply_nature_style()

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import (
    BASINS,
    load_oras5_basin,
    regrid_basin,
    instantaneous_phase,
    local_r_mean,
)


OUT_DIR = CACHE_DIR / "bandpass_sensitivity"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

BANDPASSES = [(0.5, 15.0), (1.0, 10.0), (2.0, 20.0),
              (3.0, 10.0), (5.0, 10.0)]   # MAJOR 12 additions


def preprocess_anomaly_band(da, lo_yr, hi_yr, order=4):
    """Climatology removal + detrend + custom bandpass."""
    clim = da.groupby("time.month").mean("time", skipna=True)
    out = (da.groupby("time.month") - clim).drop_vars("month", errors="ignore")
    coeffs = out.polyfit(dim="time", deg=1, skipna=True)
    fit = __import__("xarray").polyval(out["time"], coeffs.polyfit_coefficients)
    out = out - fit
    nyq = 0.5 * 12.0
    f_hi = 1.0 / lo_yr; f_lo = 1.0 / hi_yr
    b, a = butter(order, [f_lo / nyq, f_hi / nyq], btype="band")
    arr = out.transpose("time", ...).values
    flat = arr.reshape(arr.shape[0], -1)
    valid = np.isfinite(flat).all(axis=0)
    bp = np.full_like(flat, np.nan)
    if valid.any():
        bp[:, valid] = filtfilt(b, a, flat[:, valid], axis=0).astype(arr.dtype)
    return out.copy(data=bp.reshape(arr.shape).astype(arr.dtype))


def quiescence_correlation(basin, lo_yr, hi_yr):
    sst, lat2d, rlon2d = load_oras5_basin("sosstsst", basin)
    ssh, _, _ = load_oras5_basin("sossheig", basin)
    sst_rg = regrid_basin(sst, lat2d, rlon2d, basin)
    ssh_rg = regrid_basin(ssh, lat2d, rlon2d, basin)
    sst_anom = preprocess_anomaly_band(sst_rg, lo_yr, hi_yr)
    phi = instantaneous_phase(sst_anom)
    n_t = phi.sizes["time"]
    phi = phi.isel(time=slice(6, n_t - 6))
    rl_mean = local_r_mean(phi, radius_km=500.0)
    ssh_mean = ssh_rg.mean("time")
    grad_mag = np.sqrt(ssh_mean.differentiate("lat") ** 2
                       + ssh_mean.differentiate("rlon") ** 2)
    a = rl_mean.values.ravel()
    b = grad_mag.values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float("nan"), float("nan"), 0
    rho, p = pearsonr(a[m], b[m])
    return float(rho), float(p), int(m.sum())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" Bandpass sensitivity sweep")
    print("=" * 70)
    print(f"  {'basin':<10} | "
          + " | ".join(f"({lo:.1f},{hi:.0f}) yr" for lo, hi in BANDPASSES))

    rho_table = {}
    p_table = {}
    p_flat = []
    keys_flat = []
    for basin in BASINS:
        rhos = []
        ps = []
        ns = []
        for lo_yr, hi_yr in BANDPASSES:
            t0 = time.time()
            try:
                rho, p, n = quiescence_correlation(basin, lo_yr, hi_yr)
                rhos.append(rho); ps.append(p); ns.append(n)
                p_flat.append(p)
                keys_flat.append((basin, lo_yr, hi_yr))
                print(f"    {basin:<10} bp=({lo_yr},{hi_yr}): rho={rho:+.3f} p={p:.2e} "
                      f"({time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"    {basin:<10} bp=({lo_yr},{hi_yr}): FAILED -- {e}")
                rhos.append(float("nan")); ps.append(float("nan")); ns.append(0)
        rho_table[basin] = rhos
        p_table[basin] = ps

    # FDR correction across the 3x3 grid
    rejected, q_flat = fdr_bh(p_flat, alpha=0.05)
    q_table = {b: [None] * len(BANDPASSES) for b in BASINS}
    for (b, lo, hi), q in zip(keys_flat, q_flat):
        i = BANDPASSES.index((lo, hi))
        q_table[b][i] = float(q)

    print("\nBandpass x basin rho table (rows: basin, cols: bandpass):")
    for basin, rhos in rho_table.items():
        print(f"  {basin:<10}  " + "   ".join(f"{r:+.3f}" for r in rhos))

    # Stability check: max range across bandpasses per basin
    for basin, rhos in rho_table.items():
        arr = np.asarray(rhos)
        if np.isfinite(arr).any():
            print(f"  {basin:<10}  range = {np.ptp(arr):.3f}")

    with open(OUT_DIR / "bandpass_table.json", "w") as f:
        json.dump({
            "bandpasses": [list(bp) for bp in BANDPASSES],
            "rho": rho_table,
            "p": p_table,
            "q_fdr": q_table,
        }, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'bandpass_table.json'}")


if __name__ == "__main__":
    main()
