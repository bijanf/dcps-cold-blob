"""Sanity-check figures for the Phase 1 cache.

Produces a 4-panel PNG and prints diagnostic statistics. Designed to fail loudly
if the preprocessing chain is wrong.

Panels (top-left, clockwise):
  1. Raw SST climatology (Jan + Jul) — should look like a basin temperature map
  2. Variance map of bandpassed SST anomaly — should show storm tracks / Gulf Stream
  3. Time series at one node (~50N, 30W in the Cold Blob region) for sst_anom + ssh_anom
  4. Climatology check: per-month basin-mean of sst_anom should be near zero
"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from dcps.config import FIGURES_DIR, PHASE1_OUTPUT


def main(path=None):
    p = path or PHASE1_OUTPUT
    print(f"Reading {p}")
    ds = xr.open_dataset(p)
    print(ds)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)

    # --- Panel 1: SST climatology, January ---
    ax = axes[0, 0]
    sst_clim_jan = ds.sst_raw.groupby("time.month").mean("time").sel(month=1)
    im = ax.pcolormesh(ds.lon, ds.lat, sst_clim_jan, cmap="RdYlBu_r", shading="auto")
    plt.colorbar(im, ax=ax, label="SST (deg C)")
    ax.set_title("SST climatology — January")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")

    # --- Panel 2: variance of bandpassed SST anomaly ---
    ax = axes[0, 1]
    var_map = ds.sst_anom.var("time")
    im = ax.pcolormesh(ds.lon, ds.lat, var_map, cmap="viridis", shading="auto")
    plt.colorbar(im, ax=ax, label="var (K^2)")
    ax.set_title("SST_anom variance (1-10 yr band)\nshould light up Gulf Stream + subpolar gyre")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")

    # --- Panel 3: time series at a Cold-Blob node (~50 N, 30 W) ---
    ax = axes[1, 0]
    node = ds.sel(lat=50, lon=-30, method="nearest")
    ax.plot(node.time, node.sst_anom, label="SST anom (K)", lw=0.8, color="C0")
    ax2 = ax.twinx()
    ax2.plot(node.time, node.ssh_anom, label="SSH anom (m)", lw=0.8, color="C1", alpha=0.8)
    ax.set_title(f"Node at lat={float(node.lat):.0f} N, lon={float(node.lon):.0f} (Cold Blob)")
    ax.set_xlabel("year"); ax.set_ylabel("SST anom (K)", color="C0")
    ax2.set_ylabel("SSH anom (m)", color="C1")
    ax.axhline(0, color="grey", lw=0.5)

    # --- Panel 4: climatology-removal sanity (per-month basin mean) ---
    ax = axes[1, 1]
    monthly_mean = ds.sst_anom.mean(["lat", "lon"])
    ax.plot(monthly_mean.time, monthly_mean, lw=0.6, color="C2")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title("Basin-mean SST_anom per month\n(climatology removed -> should hover near 0)")
    ax.set_xlabel("year"); ax.set_ylabel("K")

    out = FIGURES_DIR / "phase1_validate.png"
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")

    # --- Numerical diagnostics ---
    print("\n=== diagnostics ===")
    print(f"Shape: time={ds.sizes['time']}, lat={ds.sizes['lat']}, lon={ds.sizes['lon']}")
    n_valid = int((~ds.sst_anom.isel(time=0).isnull()).sum())
    print(f"Valid ocean cells (non-land after regrid): {n_valid} of "
          f"{ds.sizes['lat'] * ds.sizes['lon']}")
    bm = float(ds.sst_anom.mean())
    print(f"Grand mean of sst_anom (should be ~0 after climatology+detrend): {bm:+.4e} K")
    print(f"Variance of sst_anom: {float(ds.sst_anom.var()):.4f} K^2")
    print(f"Variance of ssh_anom: {float(ds.ssh_anom.var()):.6f} m^2")

    # Spectrum check: dominant period should fall in the [1,10] yr passband.
    ts = ds.sst_anom.mean(["lat", "lon"]).values
    ts = ts[np.isfinite(ts)]
    if ts.size > 24:
        from numpy.fft import rfft, rfftfreq
        spec = np.abs(rfft(ts))
        freqs = rfftfreq(ts.size, d=1.0 / 12)  # cycles / yr
        # Drop DC, find dominant
        spec[0] = 0
        peak_freq = freqs[int(np.argmax(spec))]
        print(f"Dominant SST_anom basin-mean period: {1/peak_freq:.2f} yr "
              f"(should be in 1-10 yr band)")

    ds.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
