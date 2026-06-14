"""H2 robustness: vary the local-r window radius and compute the Cold-Blob /
subtropics contrast at each scale.  Also compute between-region phase coherence
as a complementary diagnostic of inter-region Chimera structure.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR, FIGURES_DIR
from dcps.order_parameter import local_r


def regional_phase(phases: xr.DataArray, lat_min, lat_max, lon_min, lon_max) -> xr.DataArray:
    """Average complex unit phase over a region; returns the resultant phase
    angle and magnitude (regional Kuramoto OP)."""
    box = phases.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    z = np.exp(1j * box.values)
    z = np.where(np.isfinite(box.values), z, 0.0 + 0.0j)
    n_valid = np.isfinite(box.values).sum(axis=(1, 2))
    z_sum = z.sum(axis=(1, 2))
    R = np.abs(z_sum) / np.maximum(n_valid, 1)
    angle = np.angle(z_sum)
    out = xr.Dataset(
        {
            "R_region": ("time", R.astype(np.float32)),
            "phase_region": ("time", angle.astype(np.float32)),
        },
        coords={"time": box["time"].values},
    )
    return out


def main():
    phases = xr.open_dataset(CACHE_DIR / "phase2_phases.nc")
    sst_phase = phases.sst_phase

    # --- 1. Local r at multiple radii ----------------------------------------
    radii_km = [300, 500, 1000, 1500, 2000]
    h1_window = slice("2004-04", "2014-12")
    cb_box = dict(lat=slice(45, 60), lon=slice(-40, -15))
    st_box = dict(lat=slice(20, 40), lon=slice(-60, -10))

    rows = []
    fig, axes = plt.subplots(1, len(radii_km), figsize=(4 * len(radii_km), 4),
                              constrained_layout=True)
    for ax, ell in zip(axes, radii_km):
        rl = local_r(sst_phase, radius_km=ell)
        mean_field = rl.sel(time=h1_window).mean("time")
        cb_mean = float(rl.sel(**cb_box).sel(time=h1_window).mean())
        st_mean = float(rl.sel(**st_box).sel(time=h1_window).mean())
        rows.append((ell, cb_mean, st_mean, st_mean - cb_mean))

        im = ax.pcolormesh(rl.lon, rl.lat, mean_field, cmap="RdYlBu_r",
                           vmin=0, vmax=1, shading="auto")
        ax.set_title(f"radius = {ell} km\nCB={cb_mean:.2f}, ST={st_mean:.2f}")
        ax.set_xlabel("lon")
        if ax is axes[0]:
            ax.set_ylabel("lat")
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((-40, 45), 25, 15, fill=False, edgecolor="k", lw=1.2))
        ax.add_patch(Rectangle((-60, 20), 50, 20, fill=False, edgecolor="k", lw=1.0, ls="--"))
    plt.colorbar(im, ax=axes[-1], label="r_loc")
    fig.suptitle("Phase 2 H2 robustness: time-mean local r at varying window radius")
    out = FIGURES_DIR / "phase2_h2_radius_sensitivity.png"
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")

    print("\nradius_km  CB_mean   ST_mean   ST-CB")
    print("-" * 45)
    for ell, cb, st, d in rows:
        print(f"{ell:>9}  {cb:.3f}    {st:.3f}    {d:+.3f}")

    # --- 2. Between-region phase coherence -----------------------------------
    cb_reg = regional_phase(sst_phase, 45, 60, -40, -15)
    st_reg = regional_phase(sst_phase, 20, 40, -60, -10)
    # Phase difference and its concentration
    dphi = (cb_reg.phase_region - st_reg.phase_region).values
    dphi_wrapped = (dphi + np.pi) % (2 * np.pi) - np.pi
    z_diff_mag = np.abs(np.exp(1j * dphi_wrapped).mean())
    print()
    print("Between-region phase coherence (Cold Blob vs subtropics):")
    print(f"  Cold Blob region  R: mean={float(cb_reg.R_region.mean()):.3f} "
          f"std={float(cb_reg.R_region.std()):.3f}")
    print(f"  Subtropical region R: mean={float(st_reg.R_region.mean()):.3f} "
          f"std={float(st_reg.R_region.std()):.3f}")
    print(f"  Phase-difference concentration |<exp i d_phi>|: {z_diff_mag:.3f}")
    print("  (1 = perfectly locked, 0 = uniformly drifting)")
    print(f"  Mean |d_phi|: {np.abs(dphi_wrapped).mean():.3f} rad "
          f"(= {np.rad2deg(np.abs(dphi_wrapped).mean()):.1f} deg)")

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    ax[0].plot(cb_reg.time, cb_reg.phase_region, label="Cold Blob phase", color="C0", lw=0.6)
    ax[0].plot(st_reg.time, st_reg.phase_region, label="Subtropics phase", color="C3", lw=0.6, alpha=0.7)
    ax[0].set_ylim(-np.pi, np.pi); ax[0].set_ylabel("regional phase (rad)"); ax[0].legend()
    ax[0].set_title("Regional mean phase: Cold Blob vs subtropics")

    ax[1].plot(cb_reg.time, dphi_wrapped, lw=0.6, color="C2")
    ax[1].axhline(0, color="grey", lw=0.5)
    ax[1].set_ylabel("phi_CB - phi_ST (rad)"); ax[1].set_xlabel("year")
    ax[1].set_title(f"Phase-difference time series  (concentration = {z_diff_mag:.3f})")

    out2 = FIGURES_DIR / "phase2_h2_interregion.png"
    fig.savefig(out2, dpi=140)
    print(f"Wrote {out2}")
    phases.close()


if __name__ == "__main__":
    main()
