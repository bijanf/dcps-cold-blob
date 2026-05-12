"""Step 5 / B6: Temporal frequency-detuning analysis.

The pre-registered temporal sub-tests H1*-a and H1*-b (Pearson
correlation of basin order parameter R(t) with -dPsi/dt and
trend-sign tests) were falsified across three ocean reanalyses.
The reformulated temporal claim is: under AMOC weakening, the
spatial variance of natural frequencies sigma^2_omega(t) increases
while R(t) remains constant because K/D adjusts to compensate.

This script implements the test:
  1. Load monthly r_loc(t) on the 2-deg NA grid (uses existing
     ORAS5 pipeline output).
  2. Compute instantaneous frequencies omega_i(t) = dphi_i / dt via
     central differences on the unwrapped Hilbert phase.
  3. Sliding 5-yr (60-month) windows over the RAPID overlap
     (2004-2023).
  4. For each window, compute sigma^2_omega(t) = spatial variance
     of omega across cells.
  5. Compute R(t) = basin-wide |<exp(i phi)>|.
  6. Compute partial correlation rho(sigma^2_omega, Psi | R).

If |partial rho| is large and negative, the reformulated temporal
prediction (frequency detuning dominates over R(t) constancy) is
supported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.signal import hilbert as scipy_hilbert
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr

PKG_ROOT = Path("/home/bijanf/Documents/NEW_Theory/dcps")
sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import (
    BASINS,
    load_oras5_basin,
    regrid_basin,
    preprocess_anomaly,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "results"


def partial_corr(x, y, z):
    """Partial correlation rho(x, y | z) by residualisation."""
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if valid.sum() < 4:
        return float("nan"), float("nan")
    xv = x[valid]; yv = y[valid]; zv = z[valid]
    # Regress out z from each
    bx = np.polyfit(zv, xv, 1)
    by = np.polyfit(zv, yv, 1)
    rx = xv - np.polyval(bx, zv)
    ry = yv - np.polyval(by, zv)
    return pearsonr(rx, ry)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" B6: Temporal frequency-detuning analysis (RAPID overlap)")
    print("=" * 70)

    basin = "atlantic"
    sst, lat2d, rlon2d = load_oras5_basin("sosstsst", basin)
    sst_rg = regrid_basin(sst, lat2d, rlon2d, basin)
    sst_anom = preprocess_anomaly(sst_rg)

    # Hilbert phase per cell (vectorised)
    arr = sst_anom.transpose("time", "lat", "rlon").values
    T, ny, nx = arr.shape
    flat = arr.reshape(T, -1)
    valid = np.isfinite(flat).all(axis=0)
    phi = np.full(flat.shape, np.nan, dtype=np.float32)
    z = scipy_hilbert(flat[:, valid].astype(np.float64), axis=0)
    phi[:, valid] = np.angle(z).astype(np.float32)
    print(f"  computed Hilbert phase for {valid.sum()} cells x {T} months")

    # Unwrap and instantaneous frequency
    omega = np.full(flat.shape, np.nan, dtype=np.float32)
    if valid.any():
        phi_v = phi[:, valid].astype(np.float64)
        phi_unwrap = np.unwrap(phi_v, axis=0)
        omega_v = np.gradient(phi_unwrap, 1.0 / 12.0, axis=0)  # cycles/yr * 2pi
        omega[:, valid] = (omega_v / (2 * np.pi)).astype(np.float32)
    print(f"  computed instantaneous frequencies (omega_i in cycles/yr)")

    # Basin R(t) and spatial sigma^2_omega(t) on a 60-month sliding window.
    window_months = 60
    n_t = T
    times_idx = np.arange(window_months // 2, n_t - window_months // 2,
                              dtype=int)
    R_t = []
    sigma2_t = []
    for t_center in times_idx:
        i0, i1 = t_center - window_months // 2, t_center + window_months // 2
        # R(t) at this window: basin-mean exp(i phi), averaged over window
        z_arr = np.exp(1j * phi[i0:i1])
        z_mean = np.nanmean(z_arr, axis=(0, 1))  # average over time + cells
        R_t.append(float(np.abs(z_mean)) if np.isfinite(z_mean) else np.nan)
        # sigma^2 of <omega>_t across cells
        omega_mean = np.nanmean(omega[i0:i1], axis=0)
        sigma2_t.append(float(np.nanvar(omega_mean)))

    R_t = np.asarray(R_t); sigma2_t = np.asarray(sigma2_t)
    print(f"  computed {len(R_t)} sliding-window R(t) and sigma^2_omega(t)")

    # Load RAPID Psi(t) and align time-indices
    from dcps.io import load_rapid_amoc
    Psi = load_rapid_amoc()
    # Match window centres
    times_window = sst_anom["time"].values[times_idx]
    Psi_at_window = []
    for t_win in times_window:
        idx = np.argmin(np.abs(Psi.time.values - t_win))
        Psi_at_window.append(float(Psi.values[idx]) if idx >= 0 else np.nan)
    Psi_at_window = np.asarray(Psi_at_window)
    print(f"  aligned RAPID Psi to {len(Psi_at_window)} window centres")

    # Direct correlations
    rho_R_Psi, p_R_Psi = pearsonr(R_t[np.isfinite(R_t) & np.isfinite(Psi_at_window)],
                                       Psi_at_window[np.isfinite(R_t) & np.isfinite(Psi_at_window)])
    rho_s_Psi, p_s_Psi = pearsonr(sigma2_t[np.isfinite(sigma2_t) & np.isfinite(Psi_at_window)],
                                       Psi_at_window[np.isfinite(sigma2_t) & np.isfinite(Psi_at_window)])

    # Partial correlation rho(sigma^2_omega, Psi | R)
    rho_partial, p_partial = partial_corr(sigma2_t, Psi_at_window, R_t)

    print()
    print(f"  rho(R, Psi):              {rho_R_Psi:+.3f}  p = {p_R_Psi:.3f}")
    print(f"  rho(sigma^2_omega, Psi):  {rho_s_Psi:+.3f}  p = {p_s_Psi:.3f}")
    print(f"  partial rho(sigma^2, Psi|R) = {rho_partial:+.3f}  p = {p_partial:.3f}")

    if rho_partial < -0.3:
        verdict = "SUPPORTED (frequency detuning compensates for R constancy)"
    elif abs(rho_partial) < 0.2:
        verdict = "null (no partial correlation)"
    else:
        verdict = "ambiguous"
    print(f"  verdict: {verdict}")

    out = dict(
        window_months=window_months,
        n_windows=len(R_t),
        rho_R_Psi=float(rho_R_Psi), p_R_Psi=float(p_R_Psi),
        rho_sigma2_Psi=float(rho_s_Psi), p_sigma2_Psi=float(p_s_Psi),
        partial_rho_sigma2_Psi_given_R=float(rho_partial),
        partial_p=float(p_partial),
        verdict=verdict,
    )
    with open(OUT_DIR / "temporal_detuning.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_DIR / 'temporal_detuning.json'}")


if __name__ == "__main__":
    main()
