"""Sanity-check figures for Phase 2.

Six panels in one PNG:
 (a) global R(t) for SST, SSH, pooled, with surrogate 95% envelope (SST)
 (b) local r snapshot at the most recent month (a Chimera-state preview)
 (c) local r time-mean field over 2004-2014 (the H1 overlap window)
 (d) phase trajectory at one Cold-Blob node vs one subtropical node
 (e) instantaneous-amplitude map snapshot
 (f) histogram of R values vs the surrogate distribution
"""

from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR, FIGURES_DIR


def main():
    phases = xr.open_dataset(CACHE_DIR / "phase2_phases.nc")
    R = xr.open_dataset(CACHE_DIR / "phase2_R.nc")
    print(phases)
    print()
    print(R)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)

    # --- (a) Global R(t) -------------------------------------------------------
    a = ax[0, 0]
    a.plot(R.time, R.R_sst, label="SST", color="C0", lw=0.8)
    a.plot(R.time, R.R_ssh, label="SSH", color="C1", lw=0.8, alpha=0.85)
    a.plot(R.time, R.R_pooled, label="pooled", color="C2", lw=0.8, alpha=0.85)
    # Surrogate envelope on SST
    qs = R.R_null_sst["quantile"].values
    a.fill_between(R.time, R.R_null_sst.sel(quantile=qs[0]),
                   R.R_null_sst.sel(quantile=qs[-1]),
                   color="grey", alpha=0.25, label=f"SST surrogate {int((qs[-1]-qs[0])*100)}% band")
    a.set_title("(a) Global Kuramoto R(t)")
    a.set_xlabel("year"); a.set_ylabel("R")
    a.set_ylim(0, 1)
    a.legend(loc="upper left", fontsize=8)
    a.axhline(1/np.sqrt(1234), color="grey", lw=0.5, ls="--")
    a.text(R.time.values[5], 1/np.sqrt(1234) + 0.01, r"$1/\sqrt{N}$ floor", fontsize=7)

    # --- (b) Local r snapshot at the latest month -----------------------------
    a = ax[0, 1]
    snap = R.r_loc_sst.isel(time=-1)
    im = a.pcolormesh(R.lon, R.lat, snap, cmap="RdYlBu_r", vmin=0, vmax=1, shading="auto")
    plt.colorbar(im, ax=a, label="r_loc")
    a.set_title(f"(b) Local r SST, snapshot {str(R.time[-1].values)[:7]}")
    a.set_xlabel("lon"); a.set_ylabel("lat")
    # Cold Blob box overlay
    from matplotlib.patches import Rectangle
    a.add_patch(Rectangle((-40, 45), 25, 15, fill=False, edgecolor="k", lw=1.5))
    a.text(-37, 46, "Cold Blob", fontsize=8)

    # --- (c) Time-mean local r over the H1 overlap window ---------------------
    a = ax[0, 2]
    h1_mean = R.r_loc_sst.sel(time=slice("2004-04", "2014-12")).mean("time")
    im = a.pcolormesh(R.lon, R.lat, h1_mean, cmap="RdYlBu_r", vmin=0, vmax=1, shading="auto")
    plt.colorbar(im, ax=a, label="r_loc")
    a.set_title("(c) Local r SST, mean 2004-2014 (RAPID overlap)")
    a.set_xlabel("lon"); a.set_ylabel("lat")
    a.add_patch(Rectangle((-40, 45), 25, 15, fill=False, edgecolor="k", lw=1.5))
    a.add_patch(Rectangle((-60, 20), 50, 20, fill=False, edgecolor="k", lw=1.0, ls="--"))
    a.text(-37, 46, "Cold Blob", fontsize=8)
    a.text(-55, 22, "Subtropics", fontsize=8)

    # --- (d) Phase trajectory at two contrasting nodes ------------------------
    a = ax[1, 0]
    cb = phases.sst_phase.sel(lat=50, lon=-30, method="nearest")
    sb = phases.sst_phase.sel(lat=25, lon=-50, method="nearest")
    a.plot(cb.time, cb, label="Cold Blob (50N, 30W)", lw=0.6, color="C0")
    a.plot(sb.time, sb, label="Subtropics (25N, 50W)", lw=0.6, color="C3", alpha=0.7)
    a.set_title("(d) Instantaneous phase, two nodes")
    a.set_xlabel("year"); a.set_ylabel("phi (rad)")
    a.set_ylim(-np.pi, np.pi)
    a.legend(fontsize=8)

    # --- (e) Amplitude map snapshot -------------------------------------------
    a = ax[1, 1]
    amp_snap = phases.sst_amp.isel(time=-1)
    im = a.pcolormesh(phases.lon, phases.lat, amp_snap, cmap="magma", shading="auto")
    plt.colorbar(im, ax=a, label="amp (K)")
    a.set_title(f"(e) Inst. amplitude SST, {str(phases.time[-1].values)[:7]}")
    a.set_xlabel("lon"); a.set_ylabel("lat")

    # --- (f) Distribution: observed R vs surrogate ----------------------------
    a = ax[1, 2]
    a.hist(R.R_sst.values, bins=30, alpha=0.6, color="C0", label="observed R(t)")
    a.axvline(float(R.R_sst.mean()), color="C0", lw=2)
    # Surrogate envelope: collapse over time
    null = R.R_null_sst.sel(quantile=0.5).values
    a.hist(null, bins=30, alpha=0.45, color="grey", label="surrogate median R(t)")
    a.set_title("(f) R distribution: data vs surrogate")
    a.set_xlabel("R"); a.set_ylabel("count")
    a.legend(fontsize=8)
    a.set_xlim(0, 1)

    out = FIGURES_DIR / "phase2_validate.png"
    fig.savefig(out, dpi=140)
    print(f"\nWrote {out}")

    # --- numerical diagnostics ------------------------------------------------
    print("\n=== diagnostics ===")
    n_nodes = R.R_sst.attrs.get("n_nodes", 0)
    print(f"N nodes (SST): {n_nodes}, 1/sqrt(N) floor = {1/np.sqrt(n_nodes):.4f}")
    print(f"R_sst       mean={float(R.R_sst.mean()):.3f} std={float(R.R_sst.std()):.3f} "
          f"min={float(R.R_sst.min()):.3f} max={float(R.R_sst.max()):.3f}")
    print(f"R_ssh       mean={float(R.R_ssh.mean()):.3f} std={float(R.R_ssh.std()):.3f}")
    print(f"R_pooled    mean={float(R.R_pooled.mean()):.3f} std={float(R.R_pooled.std()):.3f}")
    null_top = float(R.R_null_sst.sel(quantile=0.975).mean())
    print(f"Surrogate 97.5% quantile mean: {null_top:.4f}")
    print(f"Fraction of time R_sst > surrogate 97.5%: "
          f"{float((R.R_sst > R.R_null_sst.sel(quantile=0.975)).mean()):.3f}")

    # Cold Blob vs subtropics in the H1 window
    cb_box = R.r_loc_sst.sel(lat=slice(45, 60), lon=slice(-40, -15)).sel(
        time=slice("2004-04", "2014-12"))
    st_box = R.r_loc_sst.sel(lat=slice(20, 40), lon=slice(-60, -10)).sel(
        time=slice("2004-04", "2014-12"))
    print("\nH2 preview (2004-2014 mean local r):")
    print(f"  Cold Blob box (45-60N, 40-15W): r_loc mean = {float(cb_box.mean()):.3f}")
    print(f"  Subtropical box (20-40N, 60-10W): r_loc mean = {float(st_box.mean()):.3f}")

    phases.close(); R.close()


if __name__ == "__main__":
    main()
