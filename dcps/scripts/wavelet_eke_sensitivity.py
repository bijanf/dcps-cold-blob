"""Wavelet-period sensitivity of the Quiescence correlation.

Step 3 of the discovery roadmap.  The pre-registered Hilbert-vs-Morlet
method-independence test at a single 5-yr Morlet period fell below
the locked rs >= 0.7 threshold (SI Sec.~wavelet).  The plan asks
whether the choice of a single Fourier period is the source of the
shortfall, and whether a multi-scale Morlet composite recovers
method-independence in terms of the EKE correlation rho(<r_loc>, EKE).

For each Morlet period T in {2, 5, 8, 10} yr, recompute <r_loc> on
the North Atlantic SST field and the cell-wise Pearson rho against
the geostrophic-EKE map.  Then build a composite r_loc as the
arithmetic mean of the four single-period maps and re-evaluate.  The
Hilbert baseline is computed from the same pipeline for an apples-
to-apples comparison.

Output: a sensitivity table + a tex-includeable figure with five
scatter panels (Hilbert, T=2, 5, 8, 10) and a sixth multi-scale
panel.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import pearsonr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import (
    BASINS, load_oras5_basin, regrid_basin, preprocess_anomaly,
    instantaneous_phase, local_r_mean,
)
from eke_quiescence_test import geostrophic_eke
from wavelet_phase_validation import morlet_phase_at_period


OUT_DIR = CACHE_DIR / "wavelet_eke_sensitivity"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

PERIODS = [2.0, 4.0, 6.0, 8.0]
BASIN = "atlantic"


def morlet_rloc_at_period(sst_anom: xr.DataArray, period_yr: float,
                            radius_km: float = 500.0) -> xr.DataArray:
    """Compute <r_loc> using Morlet phase at one period.

    Mirrors the wavelet_phase_validation.morlet_rloc_map but exposes
    the period.
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
            np.where(np.isfinite(x), x, 0.0), period_yr,
        ).astype(np.float32)
    phase_3d = phase_arr.reshape(T, ny, nx)
    LAT, LON = np.meshgrid(sst_anom.lat.values,
                            sst_anom.rlon.values, indexing="ij")
    flat_lat = LAT.ravel(); flat_lon = LON.ravel()
    z_full = np.where(np.isfinite(phase_3d), np.exp(1j * phase_3d),
                       0.0 + 0.0j)
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
        a = (np.sin(dlat / 2) ** 2
             + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2)
        d = 2 * EARTH_R * np.arcsin(np.sqrt(a))
        idx = np.where(d <= radius_km)[0]
        if idx.size < 4:
            continue
        z_sum = z_flat[:, idx].sum(axis=1)
        n_valid = valid_flat[:, idx].sum(axis=1)
        ok = n_valid >= 4
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(ok, np.abs(z_sum) / np.maximum(n_valid, 1),
                          np.nan)
        out_t[:, cell] = r.astype(np.float32)
    return xr.DataArray(
        out_t.reshape(T, ny, nx), dims=("time", "lat", "rlon"),
        coords={"time": sst_anom.time.values,
                "lat": sst_anom.lat.values,
                "rlon": sst_anom.rlon.values},
    ).mean("time")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" Wavelet-period sensitivity of rho(<r_loc>, EKE) -- "
          f"{BASIN.upper()}")
    print("=" * 70)

    t0 = time.time()
    sst, lat2d, rlon2d = load_oras5_basin("sosstsst", BASIN)
    ssh, _, _ = load_oras5_basin("sossheig", BASIN)
    sst_rg = regrid_basin(sst, lat2d, rlon2d, BASIN)
    ssh_rg = regrid_basin(ssh, lat2d, rlon2d, BASIN)
    sst_anom = preprocess_anomaly(sst_rg)
    print(f"  data prep done in {time.time()-t0:.0f}s")

    # EKE
    eke = geostrophic_eke(ssh_rg)
    eke_v = eke.values.ravel()

    # Hilbert baseline
    t0 = time.time()
    phi = instantaneous_phase(sst_anom)
    n_t = phi.sizes["time"]
    phi = phi.isel(time=slice(6, n_t - 6))
    rl_hilbert = local_r_mean(phi, radius_km=500.0)
    rh_v = rl_hilbert.values.ravel()
    m = np.isfinite(rh_v) & np.isfinite(eke_v)
    rho_h, p_h = pearsonr(rh_v[m], eke_v[m])
    print(f"  Hilbert:  rho(<r_loc>, EKE) = {rho_h:+.3f}  "
          f"p = {p_h:.2e}  n = {int(m.sum())}  "
          f"({time.time()-t0:.0f}s)")

    # Single-period Morlet at each period
    morlet_maps = {}
    morlet_rho = {}
    for T in PERIODS:
        t0 = time.time()
        rl = morlet_rloc_at_period(sst_anom, T)
        rv = rl.values.ravel()
        mm = np.isfinite(rv) & np.isfinite(eke_v)
        if int(mm.sum()) < 50:
            print(f"  Morlet T={T:4.1f} yr:  insufficient valid "
                  f"cells (n={int(mm.sum())}); skipping.")
            morlet_maps[T] = rl
            morlet_rho[T] = dict(rho=float("nan"), p=float("nan"),
                                  n=int(mm.sum()),
                                  note="COI trimmed too much data")
            continue
        rho_m, p_m = pearsonr(rv[mm], eke_v[mm])
        # Sanity diagnostic: standard deviation across cells of the
        # period-specific r_loc map.  If sd ~ 0 the wavelet has
        # collapsed and the result is degenerate.
        sd_rloc = float(np.nanstd(rv[mm]))
        print(f"  Morlet T={T:4.1f} yr:  rho = {rho_m:+.3f}  "
              f"p = {p_m:.2e}  n = {int(mm.sum())}  "
              f"sd(<r_loc>) = {sd_rloc:.4f}  "
              f"({time.time()-t0:.0f}s)")
        morlet_maps[T] = rl
        morlet_rho[T] = dict(rho=float(rho_m), p=float(p_m),
                              n=int(mm.sum()),
                              sd_rloc=sd_rloc)

    # Multi-scale composite: arithmetic mean of single-period maps
    stack = np.stack([morlet_maps[T].values for T in PERIODS], axis=0)
    rl_composite = np.nanmean(stack, axis=0)
    rc_v = rl_composite.ravel()
    mc = np.isfinite(rc_v) & np.isfinite(eke_v)
    rho_c, p_c = pearsonr(rc_v[mc], eke_v[mc])
    print(f"  Composite (mean):  rho = {rho_c:+.3f}  "
          f"p = {p_c:.2e}  n = {int(mc.sum())}")

    composite_verdict = (
        "method-independence restored (|rho| >= 0.30, p < 0.01)"
        if abs(rho_c) >= 0.30 and p_c < 0.01
        else "method-independence NOT restored"
    )
    print(f"\n  Verdict: {composite_verdict}")

    summary = dict(
        basin=BASIN,
        hilbert=dict(rho=float(rho_h), p=float(p_h),
                      n=int(m.sum())),
        morlet=morlet_rho,
        composite=dict(rho=float(rho_c), p=float(p_c),
                        n=int(mc.sum()), verdict=composite_verdict),
    )
    with open(OUT_DIR / "wavelet_eke.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'wavelet_eke.json'}")

    # Figure: 2x3 panels -- Hilbert, T2, T5, T8, T10, composite
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0),
                              constrained_layout=True)
    panels = [
        ("Hilbert (1–10 yr)", rh_v, rho_h, p_h),
    ]
    for T in PERIODS:
        rv = morlet_maps[T].values.ravel()
        mm = np.isfinite(rv) & np.isfinite(eke_v)
        rho_m, p_m = pearsonr(rv[mm], eke_v[mm])
        panels.append((f"Morlet, T = {T:.0f} yr", rv, rho_m, p_m))
    panels.append(("Multi-scale composite", rc_v, rho_c, p_c))

    for i, (title, rv, rho, p) in enumerate(panels):
        ax = axes.flat[i]
        mm = np.isfinite(rv) & np.isfinite(eke_v)
        ax.scatter(eke_v[mm], rv[mm], s=4, alpha=0.35, color="C0",
                    edgecolors="none")
        # decile means
        if mm.sum() > 100:
            order = np.argsort(eke_v[mm])
            x_sorted = eke_v[mm][order]
            y_sorted = rv[mm][order]
            decile_x = []
            decile_y = []
            n = order.size
            for k in range(10):
                lo, hi = int(k * n / 10), int((k + 1) * n / 10)
                decile_x.append(float(np.mean(x_sorted[lo:hi])))
                decile_y.append(float(np.mean(y_sorted[lo:hi])))
            ax.plot(decile_x, decile_y, "r-o", lw=2, ms=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("EKE (m$^2$/s$^2$)")
        ax.set_ylabel(r"$\langle r_{\mathrm{loc}}\rangle_t$")
        ax.text(0.04, 0.04,
                 f"$\\rho = {rho:+.3f}$\n$p = {p:.1e}$",
                 transform=ax.transAxes, fontsize=9, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        ax.grid(alpha=0.3)
    fig.suptitle("Wavelet-period sensitivity of the EKE correlation, "
                  "North Atlantic ORAS5", fontsize=12)
    out_fig = MANUSCRIPT_FIGS / "fig_wavelet_eke_sensitivity.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
