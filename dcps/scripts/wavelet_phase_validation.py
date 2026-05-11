"""Wavelet cross-method validation of the Quiescence pattern.

Recomputes the local phase-coherence diagnostic using a Morlet
cross-wavelet phase instead of the published Hilbert-derived
phase, at the dominant decadal period in the 1-10 yr Quiescence
band.  Reports the Spearman rank correlation between the two
<r_loc> maps per basin.

Pre-registered pass condition (locked):
  rank_corr(Hilbert <r_loc>, Morlet <r_loc>) >= 0.7 in each basin
  => Quiescence pattern is method-independent.

Minimal Morlet implementation (no pycwt dependency).  Morlet
wavelet:
   psi(t) = pi^(-1/4) * (exp(i*omega0*t) - exp(-omega0^2/2)) * exp(-t^2/2)
We use omega0 = 6 (Torrence & Compo 1998 default).  At scale s the
Fourier period is T = 4*pi*s / (omega0 + sqrt(2 + omega0^2)).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert as scipy_hilbert

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import (
    BASINS,
    load_oras5_basin,
    regrid_basin,
    preprocess_anomaly,
    instantaneous_phase,
    local_r_mean,
)


OUT_DIR = CACHE_DIR / "wavelet_validation"
OMEGA0 = 6.0
TARGET_PERIOD_YR = 5.0


def morlet_phase_at_period(x_t, period_yr, dt_yr=1.0/12.0, omega0=OMEGA0):
    """Single-scale Morlet wavelet phase at a target period.

    x_t : 1-D signal (assumed already detrended).  NaNs return NaN phase.
    period_yr : Fourier period in years to extract phase at.
    Returns: array of instantaneous phase, same shape as input.
    """
    x_t = np.asarray(x_t, dtype=float)
    n = x_t.size
    if not np.isfinite(x_t).any():
        return np.full(n, np.nan)
    # Map period -> scale (Torrence & Compo Table 1, eq. for Morlet)
    s_yr = period_yr * (omega0 + np.sqrt(2 + omega0 ** 2)) / (4 * np.pi)
    s_steps = s_yr / dt_yr
    # FFT-based convolution
    # Build daughter wavelet in Fourier domain
    freqs = np.fft.fftfreq(n, d=dt_yr) * 2 * np.pi  # angular freq
    # Heaviside step
    H = (freqs > 0).astype(float)
    psi_hat = (np.pi ** -0.25) * np.sqrt(s_steps) * H * np.exp(
        -0.5 * (s_steps * freqs - omega0) ** 2)
    X = np.fft.fft(np.nan_to_num(x_t))
    W = np.fft.ifft(X * np.conj(psi_hat))   # CWT at this scale
    phase = np.angle(W)
    # mark edges within s_steps as nan (cone of influence)
    coi = int(np.ceil(s_steps * np.sqrt(2)))
    if coi > 0:
        phase[:coi] = np.nan
        phase[-coi:] = np.nan
    return phase


def morlet_rloc_map(sst_anom, radius_km=500.0):
    """Compute <r_loc> using Morlet phase at TARGET_PERIOD_YR.

    Mirrors the local_r_mean structure but with Morlet phases instead
    of Hilbert.
    """
    arr = sst_anom.transpose("time", "lat", "rlon").values
    T, ny, nx = arr.shape
    flat = arr.reshape(T, -1)
    phase_arr = np.full(flat.shape, np.nan, dtype=np.float32)
    for cell in range(flat.shape[1]):
        x = flat[:, cell]
        if not np.isfinite(x).any():
            continue
        phase_arr[:, cell] = morlet_phase_at_period(
            np.where(np.isfinite(x), x, 0.0),
            TARGET_PERIOD_YR,
        ).astype(np.float32)
    phase_3d = phase_arr.reshape(T, ny, nx)

    # local r_loc using haversine 500-km window
    LAT, LON = np.meshgrid(sst_anom.lat.values, sst_anom.rlon.values, indexing="ij")
    flat_lat = LAT.ravel(); flat_lon = LON.ravel()
    z_full = np.where(np.isfinite(phase_3d), np.exp(1j * phase_3d), 0.0 + 0.0j)
    valid_mask = np.isfinite(phase_3d)
    z_flat = z_full.reshape(T, -1)
    valid_flat = valid_mask.reshape(T, -1)
    out_t = np.full((T, ny * nx), np.nan, dtype=np.float32)
    EARTH_R = 6371.0
    for cell in range(ny * nx):
        if not valid_mask.reshape(T, -1)[:, cell].any():
            continue
        lat1r = np.deg2rad(flat_lat[cell])
        lat2r = np.deg2rad(flat_lat)
        dlat = lat2r - lat1r
        dlon = np.deg2rad(flat_lon - flat_lon[cell])
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
        d = 2 * EARTH_R * np.arcsin(np.sqrt(a))
        idx = np.where(d <= radius_km)[0]
        if idx.size < 4:
            continue
        z_sum = z_flat[:, idx].sum(axis=1)
        n_valid = valid_flat[:, idx].sum(axis=1)
        ok = n_valid >= 4
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(ok, np.abs(z_sum) / np.maximum(n_valid, 1), np.nan)
        out_t[:, cell] = r.astype(np.float32)
    import xarray as xr
    return xr.DataArray(
        out_t.reshape(T, ny, nx), dims=("time", "lat", "rlon"),
        coords={"time": sst_anom.time.values, "lat": sst_anom.lat.values,
                "rlon": sst_anom.rlon.values},
    ).mean("time")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" Wavelet vs Hilbert Quiescence cross-method validation")
    print("=" * 70)

    results = {}
    for basin in BASINS:
        t0 = time.time()
        print(f"\n  basin: {basin}")
        try:
            sst, lat2d, rlon2d = load_oras5_basin("sosstsst", basin)
            sst_rg = regrid_basin(sst, lat2d, rlon2d, basin)
            sst_anom = preprocess_anomaly(sst_rg)

            # Hilbert
            phi_h = instantaneous_phase(sst_anom)
            n_t = phi_h.sizes["time"]
            phi_h = phi_h.isel(time=slice(6, n_t - 6))
            rl_hilbert = local_r_mean(phi_h, radius_km=500.0)
            print(f"    Hilbert r_loc done in {time.time()-t0:.0f}s")

            # Morlet
            t0 = time.time()
            rl_morlet = morlet_rloc_map(sst_anom, radius_km=500.0)
            print(f"    Morlet r_loc done in {time.time()-t0:.0f}s")

            # Spearman rank correlation
            a = rl_hilbert.values.ravel()
            b = rl_morlet.values.ravel()
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 50:
                results[basin] = {"r_spearman": float("nan"), "n_cells": int(m.sum())}
                continue
            r_sp, p_sp = spearmanr(a[m], b[m])
            verdict = "method-independent" if r_sp >= 0.7 else "below threshold"
            results[basin] = {
                "r_spearman": float(r_sp),
                "p_spearman": float(p_sp),
                "n_cells": int(m.sum()),
                "verdict": verdict,
            }
            print(f"    Spearman rank corr (Hilbert vs Morlet r_loc) = {r_sp:+.3f}  p = {p_sp:.2e}  ({verdict})")
        except Exception as e:
            print(f"    {basin}: FAILED -- {e}")
            results[basin] = {"error": str(e)}

    with open(OUT_DIR / "wavelet_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'wavelet_validation.json'}")


if __name__ == "__main__":
    main()
