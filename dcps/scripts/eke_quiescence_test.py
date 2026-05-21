"""Geostrophic-EKE-from-SSH mechanistic test (Phase 3) and
frequency-dispersion test (Phase 4) for the Quiescence Hypothesis.

Pre-registered (locked before computation):

  Phase 3 -- Direct mechanism test:
    Compute surface geostrophic velocity anomalies from ORAS5 SSH:
        u_g'(x,t) = -(g/f) d(SSH')/dy
        v_g'(x,t) = (g/f)  d(SSH')/dx
    Geostrophic EKE per cell:
        EKE(x) = 0.5 * <(u_g'^2 + v_g'^2)>_t
    Test:
        rho(<r_loc>_t, EKE)
    Pre-registered claim: rho < -0.30 with p < 0.01 in >=2 of 3 basins.
    Stronger claim: |rho| >= |rho(<r_loc>, |grad SSH|)| in at least 2
    of 3 basins, demonstrating EKE is at least as good a predictor as
    |grad SSH|.

  Phase 4 -- Kuramoto-specific prediction:
    Compute per-cell instantaneous frequency:
        omega_i(t) = (1 / 2 pi) * d phi_i / dt
    using phase unwrapping. Take per-cell time-mean omega_i_bar.
    Compute spatial dispersion of omega_i_bar in 500-km windows:
        sigma_omega(x) = std(omega_j_bar : j in W(x; 500 km))
    Test:
        rho(sigma_omega, EKE)
    Pre-registered claim: rho > +0.30 with p < 0.01 in >=2 of 3 basins.
    This prediction CANNOT be made by |grad SSH| analysis alone; it
    requires the Kuramoto framework that treats SST phase oscillators
    as having distinct natural frequencies that get detuned by noise.
"""

from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.signal import hilbert as scipy_hilbert
from scipy.stats import pearsonr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()

import sys
sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import (
    BASINS,
    load_oras5_basin,
    regrid_basin,
    preprocess_anomaly,
    instantaneous_phase,
    local_r_mean,
    rotated_lon,
)


OUT_DIR = CACHE_DIR / "eke"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

GRAVITY = 9.81
OMEGA_EARTH = 7.2921e-5
DEG2M = 111e3   # roughly, 1 deg lat in m


def coriolis(lat_deg: np.ndarray) -> np.ndarray:
    return 2 * OMEGA_EARTH * np.sin(np.deg2rad(lat_deg))


def geostrophic_eke(ssh: xr.DataArray, lat_dim: str = "lat",
                     lon_dim: str = "rlon") -> xr.DataArray:
    """Time-mean surface geostrophic EKE from SSH.

    For each time step compute u_g' = -(g/f) dSSH'/dy and v_g' =
    (g/f) dSSH'/dx using central differences in degrees and convert
    distances via 111 km / deg lat and 111 cos(lat) km / deg lon.
    Then time-mean variance.
    """
    ssh_mean = ssh.mean("time")
    ssh_anom = ssh - ssh_mean
    arr = ssh_anom.transpose("time", lat_dim, lon_dim).values
    lat = ssh[lat_dim].values
    lon = ssh[lon_dim].values
    LAT, _LON = np.meshgrid(lat, lon, indexing="ij")
    f = coriolis(LAT)
    # Distances per cell index (degrees * m/deg) at each lat
    # For dy: distance per +1 cell in lat = (lat[1]-lat[0]) * DEG2M
    dy_m = float(np.abs(np.diff(lat).mean())) * DEG2M
    dx_m = float(np.abs(np.diff(lon).mean())) * DEG2M * np.cos(
        np.deg2rad(LAT))    # 2D

    # Central differences in space, per time step
    # u_g = -(g/f) dSSH/dy
    # v_g = (g/f) dSSH/dx
    u = np.zeros_like(arr)
    v = np.zeros_like(arr)
    u[:, 1:-1, :] = -(GRAVITY / f[1:-1, :]) * (arr[:, 2:, :] - arr[:, :-2, :]) / (2 * dy_m)
    v[:, :, 1:-1] = (GRAVITY / f[:, 1:-1]) * (arr[:, :, 2:] - arr[:, :, :-2]) / (2 * dx_m[:, 1:-1])
    # Mask the equatorial / geostrophic-breakdown zone where |lat| < 15.
    eq_mask = np.abs(LAT) < 15.0
    u[:, eq_mask] = np.nan
    v[:, eq_mask] = np.nan
    u[~np.isfinite(u)] = np.nan
    v[~np.isfinite(v)] = np.nan
    eke = 0.5 * (u ** 2 + v ** 2)
    eke_mean = np.nanmean(eke, axis=0)
    return xr.DataArray(
        eke_mean, dims=(lat_dim, lon_dim),
        coords={lat_dim: lat, lon_dim: lon}, name="EKE")


def instantaneous_frequency(phase: xr.DataArray,
                              dt_yr: float = 1.0 / 12.0) -> xr.DataArray:
    """Per-cell time-mean instantaneous frequency in cycles per yr.

    omega(t) = (1 / 2 pi) d phi / dt via finite differences on
    unwrapped phase. Return time-mean across the record.
    """
    arr = phase.transpose("time", "lat", "rlon").values
    T, ny, nx = arr.shape
    flat = arr.reshape(T, -1)
    valid = np.isfinite(flat).all(axis=0)
    omega_mean = np.full(flat.shape[1], np.nan)
    for i in np.where(valid)[0]:
        # Unwrap and differentiate
        phi_u = np.unwrap(flat[:, i])
        dphi = np.gradient(phi_u, dt_yr)
        omega = dphi / (2 * np.pi)    # cycles per year
        omega_mean[i] = float(np.nanmean(omega))
    return xr.DataArray(
        omega_mean.reshape(ny, nx), dims=("lat", "rlon"),
        coords={"lat": phase.lat.values, "rlon": phase.rlon.values},
        name="omega")


_EARTH_R_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1r, lat2r = np.deg2rad(lat1), np.deg2rad(lat2)
    dlat = lat2r - lat1r
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_R_KM * np.arcsin(np.sqrt(a))


def local_frequency_dispersion(omega: xr.DataArray,
                                  radius_km: float = 500.0,
                                  min_n: int = 4) -> xr.DataArray:
    """Spatial standard deviation of omega in a 500-km window per cell."""
    lat = omega.lat.values; lon = omega.rlon.values
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    flat_lat = LAT.ravel(); flat_lon = LON.ravel()
    flat_omega = omega.values.ravel()
    out = np.full(flat_omega.shape, np.nan)
    for i in range(flat_omega.size):
        if not np.isfinite(flat_omega[i]): continue
        d = _haversine_km(flat_lat[i], flat_lon[i], flat_lat, flat_lon)
        nb = (d <= radius_km) & np.isfinite(flat_omega)
        if nb.sum() < min_n: continue
        out[i] = float(np.nanstd(flat_omega[nb]))
    return xr.DataArray(
        out.reshape(omega.shape), dims=omega.dims,
        coords=omega.coords, name="sigma_omega")


def run_basin(basin: str) -> dict:
    print(f"\n{'='*60}\n{basin.upper()}: {BASINS[basin]['label']}\n{'='*60}")
    t0 = time.time()
    sst, lat2d, rlon2d = load_oras5_basin("sosstsst", basin)
    ssh, _, _ = load_oras5_basin("sossheig", basin)
    print(f"  loaded SST + SSH in {time.time()-t0:.1f}s")

    # Regrid
    sst_rg = regrid_basin(sst, lat2d, rlon2d, basin)
    ssh_rg = regrid_basin(ssh, lat2d, rlon2d, basin)

    # Quiescence Pipeline: SST anomaly + Hilbert + local r
    sst_anom = preprocess_anomaly(sst_rg)
    phi = instantaneous_phase(sst_anom)
    n_t = phi.sizes["time"]
    phi = phi.isel(time=slice(6, n_t - 6))    # Hilbert edge trim
    rl_mean = local_r_mean(phi, radius_km=500.0)

    # |grad SSH| (for comparison)
    ssh_mean = ssh_rg.mean("time")
    grad_mag = np.sqrt(ssh_mean.differentiate("lat") ** 2
                        + ssh_mean.differentiate("rlon") ** 2)

    # EKE
    print("  computing geostrophic EKE from SSH...")
    eke = geostrophic_eke(ssh_rg)

    # Cell-wise correlations
    rl_v = rl_mean.values.ravel()
    eke_v = eke.values.ravel()
    grad_v = grad_mag.values.ravel()

    def _corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 50: return float("nan"), float("nan"), int(m.sum())
        rho, p = pearsonr(a[m], b[m])
        return float(rho), float(p), int(m.sum())

    rho_eke, p_eke, n_eke = _corr(rl_v, eke_v)
    rho_grad, p_grad, n_grad = _corr(rl_v, grad_v)
    print(f"  rho(<r_loc>, EKE)        = {rho_eke:+.3f}  p = {p_eke:.2e}  n={n_eke}")
    print(f"  rho(<r_loc>, |grad SSH|) = {rho_grad:+.3f}  p = {p_grad:.2e}  n={n_grad}")

    # Frequency dispersion (Phase 4)
    print("  computing instantaneous frequency + spatial dispersion...")
    omega = instantaneous_frequency(phi)
    sigma_omega = local_frequency_dispersion(omega)
    rho_fd, p_fd, n_fd = _corr(sigma_omega.values.ravel(), eke_v)
    print(f"  rho(sigma_omega, EKE)    = {rho_fd:+.3f}  p = {p_fd:.2e}  n={n_fd}")

    return {
        "basin": basin,
        "rho_eke_rloc": rho_eke, "p_eke_rloc": p_eke,
        "rho_grad_rloc": rho_grad, "p_grad_rloc": p_grad,
        "rho_fd_eke": rho_fd, "p_fd_eke": p_fd,
        "EKE": eke, "rl_mean": rl_mean, "sigma_omega": sigma_omega,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for b in BASINS:
        try:
            results[b] = run_basin(b)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {b}: FAILED -- {e}")

    print("\n" + "=" * 70)
    print(" Multi-basin EKE + frequency-dispersion summary")
    print("=" * 70)
    print(f"{'basin':<12} {'rho_EKE':>8} {'rho_gradSSH':>12} {'rho_freq_disp':>14}")
    print("-" * 70)
    for b, r in results.items():
        print(f"{b:<12} {r['rho_eke_rloc']:+8.3f}  "
              f"{r['rho_grad_rloc']:+11.3f}  {r['rho_fd_eke']:+13.3f}")

    n_eke_supp = sum(1 for r in results.values()
                      if r["rho_eke_rloc"] <= -0.30 and r["p_eke_rloc"] < 0.01)
    n_fd_supp = sum(1 for r in results.values()
                     if r["rho_fd_eke"] >= 0.30 and r["p_fd_eke"] < 0.01)
    print(f"\nEKE-quiescence supported in {n_eke_supp}/3 basins "
          f"(need >=2)")
    print(f"Frequency-dispersion supported in {n_fd_supp}/3 basins "
          f"(need >=2)")

    # Persist (without xarray objects)
    summary = {
        b: {k: float(r[k]) for k in
            ("rho_eke_rloc", "p_eke_rloc", "rho_grad_rloc", "p_grad_rloc",
              "rho_fd_eke", "p_fd_eke")}
        for b, r in results.items()
    }
    summary["aggregate_eke"] = "SUPPORTED" if n_eke_supp >= 2 else "not supported"
    summary["aggregate_freq_dispersion"] = "SUPPORTED" if n_fd_supp >= 2 else "not supported"
    with open(OUT_DIR / "eke_test.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'eke_test.json'}")

    # ----- Figure: per-basin EKE map + scatter --------------------------
    # Restrict the main-text figure to the three basins referenced in the
    # manuscript (Atlantic, Pacific, Southern Ocean). Any additional basins
    # (e.g. Indian, from the multi-basin sweep) are reported in the table
    # above but plotted in a separate SI figure to keep panel labels a-i.
    FIG_BASINS = ("atlantic", "pacific", "southern")
    fig_results = {b: results[b] for b in FIG_BASINS if b in results}
    n_basin = len(fig_results)
    fig, axes = plt.subplots(n_basin, 3, figsize=(11.5, 2.6 * n_basin),
                              constrained_layout=True)
    panel_idx = 0
    for row, (b, r) in enumerate(fig_results.items()):
        # EKE map
        ax = axes[row, 0]
        eke_da = r["EKE"]
        im = ax.pcolormesh(eke_da.rlon, eke_da.lat, eke_da.values,
                            cmap="magma", shading="auto")
        plt.colorbar(im, ax=ax, label="EKE (m$^2$/s$^2$)")
        ax.text(-0.18, 1.03, chr(97 + panel_idx),
                transform=ax.transAxes,
                fontweight="bold", fontsize=13, va="bottom")
        panel_idx += 1
        ax.set_ylabel("Latitude")
        ax.set_xlabel("Rotated longitude")
        # Trim axes to the data extent so the equatorial mask and land
        # gaps don't leave large empty strips at the basin edges.
        finite_mask = np.isfinite(eke_da.values)
        if finite_mask.any():
            lat_v = eke_da.lat.values
            lon_v = eke_da.rlon.values
            lat_has = finite_mask.any(axis=1)
            lon_has = finite_mask.any(axis=0)
            lat_present = lat_v[lat_has]
            lon_present = lon_v[lon_has]
            dlat = float(np.abs(np.diff(lat_v).mean())) if lat_v.size > 1 else 1.0
            dlon = float(np.abs(np.diff(lon_v).mean())) if lon_v.size > 1 else 1.0
            ax.set_ylim(lat_present.min() - 0.5 * dlat,
                        lat_present.max() + 0.5 * dlat)
            ax.set_xlim(lon_present.min() - 0.5 * dlon,
                        lon_present.max() + 0.5 * dlon)
        # r_loc vs EKE
        ax = axes[row, 1]
        rl_v = r["rl_mean"].values.ravel()
        eke_v = r["EKE"].values.ravel()
        m = np.isfinite(rl_v) & np.isfinite(eke_v)
        ax.scatter(eke_v[m], rl_v[m], s=4, alpha=0.35,
                    color=BASINS[b]["color"])
        ax.set_xlabel("EKE (m$^2$/s$^2$)")
        ax.set_ylabel(r"$\langle r_{\mathrm{loc}}\rangle_t$")
        ax.text(-0.18, 1.03, chr(97 + panel_idx),
                transform=ax.transAxes,
                fontweight="bold", fontsize=13, va="bottom")
        panel_idx += 1
        # sigma_omega vs EKE
        ax = axes[row, 2]
        so_v = r["sigma_omega"].values.ravel()
        m = np.isfinite(so_v) & np.isfinite(eke_v)
        ax.scatter(eke_v[m], so_v[m], s=4, alpha=0.35,
                    color=BASINS[b]["color"])
        ax.set_xlabel("EKE (m$^2$/s$^2$)")
        ax.set_ylabel(r"$\sigma_\omega$ (cycles/yr)")
        ax.text(-0.18, 1.03, chr(97 + panel_idx),
                transform=ax.transAxes,
                fontweight="bold", fontsize=13, va="bottom")
        panel_idx += 1

    out_fig = MANUSCRIPT_FIGS / "fig_eke_mechanism.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
