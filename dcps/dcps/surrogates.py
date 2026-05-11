"""Phase-randomisation surrogates (Schreiber 2000) for significance testing of
the Kuramoto order parameter.

Per node, the FFT magnitude is preserved and the phases are randomised
independently. This destroys all inter-node phase relationships while
preserving each node's individual power spectrum, giving an appropriate null
for testing whether observed network synchrony exceeds finite-size noise.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .order_parameter import global_R
from .phase import analytic_signal


def phase_randomise(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """FFT-based phase randomisation along axis 0.

    Parameters
    ----------
    arr : (T, N) real array
        Each column is one node's bandpassed time series.
    rng : numpy random generator

    Returns
    -------
    (T, N) real array with the same per-column power spectrum, randomised phase.
    """
    if arr.ndim != 2:
        raise ValueError("Expected 2D array (T, N).")
    T, N = arr.shape
    F = np.fft.rfft(arr, axis=0)
    mag = np.abs(F)
    # Random phases for each non-DC, non-Nyquist bin.
    n_freq = F.shape[0]
    rand_phase = rng.uniform(0, 2 * np.pi, size=(n_freq, N))
    # DC stays real; if T even, Nyquist also stays real.
    rand_phase[0, :] = 0.0
    if T % 2 == 0:
        rand_phase[-1, :] = 0.0
    F_surr = mag * np.exp(1j * rand_phase)
    out = np.fft.irfft(F_surr, n=T, axis=0)
    return out


def surrogate_R_envelope(
    bandpassed: xr.DataArray,
    n_surrogates: int = 1000,
    seed: int = 0,
    quantiles: tuple[float, ...] = (0.025, 0.5, 0.975),
) -> xr.DataArray:
    """Distribution of R(t) under the null of phase-randomised, otherwise
    spectrum-matched, independent nodes.

    Returns a (quantile, time) DataArray of R(t) quantiles across the surrogate
    ensemble.
    """
    if "time" not in bandpassed.dims:
        raise ValueError("Expected time dim.")
    arr = bandpassed.transpose("time", ...).values.astype(np.float64)
    T = arr.shape[0]
    flat = arr.reshape(T, -1)
    valid_mask = np.isfinite(flat).all(axis=0)
    valid_flat = flat[:, valid_mask]                   # (T, N_valid) — no NaNs

    # Reshape back-template
    ny, nx = arr.shape[1], arr.shape[2]
    rng = np.random.default_rng(seed)
    R_samples = np.empty((n_surrogates, T - 0), dtype=np.float32)
    # We'll redo Hilbert + global_R per surrogate via in-array operations.

    # Pre-build a DataArray template for global_R reuse.
    coords_time = bandpassed["time"].values
    template_dims = ("time", "lat", "lon")
    template_coords = {
        "time": coords_time,
        "lat": bandpassed["lat"].values,
        "lon": bandpassed["lon"].values,
    }

    # Storage for surrogate phases (T, ny*nx)
    surr_full = np.full((T, ny * nx), np.nan, dtype=np.float32)

    for k in range(n_surrogates):
        surr = phase_randomise(valid_flat, rng)         # (T, N_valid)
        # Hilbert + phase per valid column
        from scipy.signal import hilbert as _hilbert
        z = _hilbert(surr, axis=0)
        phi = np.angle(z).astype(np.float32)
        surr_full[:] = np.nan
        # Place the valid columns back into the full grid
        flat_idx = np.where(valid_mask)[0]
        surr_full[:, flat_idx] = phi
        phase_da = xr.DataArray(
            surr_full.reshape(T, ny, nx),
            dims=template_dims, coords=template_coords, name="phase",
        )
        R_samples[k, :] = global_R(phase_da).values

    quants = np.quantile(R_samples, quantiles, axis=0)   # (n_q, T)
    return xr.DataArray(
        quants,
        dims=("quantile", "time"),
        coords={"quantile": list(quantiles), "time": coords_time},
        name="R_null",
        attrs={
            "long_name": "Schreiber phase-randomised null distribution of R(t)",
            "n_surrogates": int(n_surrogates),
            "seed": int(seed),
        },
    )


def empirical_p_value(R_obs: xr.DataArray, R_null_quantiles: xr.DataArray) -> xr.DataArray:
    """Coarse two-sided empirical p-value at each timestep.

    Compares observed R(t) against the surrogate quantile band. Returns 1.0 -
    coverage, where coverage = fraction of surrogates whose R(t) is below R_obs.
    Approximated from quantile array (so resolution is limited to 1/(2*n_quantiles));
    use the raw surrogate samples for a finer estimate.
    """
    # If R_obs is above the upper band, p < (1 - upper_quantile)
    qs = np.array(R_null_quantiles["quantile"].values)
    upper = R_null_quantiles.sel(quantile=qs[-1])
    lower = R_null_quantiles.sel(quantile=qs[0])
    p_above = (R_obs > upper).astype(np.float32) * (1.0 - qs[-1])
    p_below = (R_obs < lower).astype(np.float32) * qs[0]
    p = xr.where(p_above > 0, p_above, xr.where(p_below > 0, p_below, 1.0))
    p.attrs["note"] = "Approximate from quantile band; for fine p-values, retain raw samples."
    return p
